#!/usr/bin/env python3
"""
Shared setup utilities for RBM training.
This file contains setup functions that can be reused across different training scripts.
"""

from unsloth import FastVisionModel

import json
import os
from pathlib import Path
from typing import Tuple, Optional, Any
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    TrainingArguments,
    BitsAndBytesConfig,
)

# Try to import Qwen3 models if available
try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel

    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False
    Qwen3VLForConditionalGeneration = None
    Qwen3VLModel = None

from robometer.configs.experiment_configs import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    TrainingConfig,
)
from robometer.data.collators import BaseCollator, ReWiNDBatchCollator, RBMBatchCollator
from robometer.data.datasets import (
    RBMDataset,
    StrategyFirstDataset,
    BaseDataset,
    RepeatedDataset,
)
from robometer.data.datasets.custom_eval import CustomEvalDataset
from robometer.models import RBM, ReWiNDTransformer
from robometer.utils.logger import get_logger

logger = get_logger()
from robometer.utils.save import parse_hf_model_id_and_revision, resolve_checkpoint_path


def _get_rbm_peft_target(rbm_model: RBM) -> tuple[Optional[str], Optional[PeftModel]]:
    """Return the PEFT-wrapped RBM component, if any."""
    root = getattr(rbm_model, "model", None)
    if isinstance(root, PeftModel):
        return "model", root

    language_model = getattr(root, "language_model", None)
    if isinstance(language_model, PeftModel):
        return "language_model", language_model

    visual = getattr(root, "visual", None)
    if isinstance(visual, PeftModel):
        return "visual", visual

    return None, None


def model_has_peft(rbm_model: RBM) -> bool:
    """Whether any RBM submodule is currently PEFT-wrapped."""
    return _get_rbm_peft_target(rbm_model)[1] is not None


def _infer_peft_target_from_keys(keys: list[str]) -> Optional[str]:
    """Infer which RBM submodule owns the adapter keys."""
    adapter_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
    if not adapter_keys:
        return None

    for key in adapter_keys:
        if ".language_model." in key:
            return "language_model"
        if ".visual." in key:
            return "visual"
    return "model"


def _inspect_checkpoint_for_peft(checkpoint_path: str) -> dict[str, Any]:
    """Inspect a checkpoint directory for PEFT adapter artifacts or embedded adapter weights."""
    path = Path(checkpoint_path)
    info = {
        "has_adapter_files": False,
        "has_adapter_weights": False,
        "target_module": None,
        "adapter_load_path": str(path),
    }

    meta_path = path / "peft_target_module.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        info["target_module"] = meta.get("target_module")
        adapter_dir = meta.get("adapter_dir", ".")
        info["adapter_load_path"] = str(path / adapter_dir) if adapter_dir != "." else str(path)

    adapter_candidates = [
        (path, "model"),
        (path / "language_model_adapter", "language_model"),
        (path / "visual_adapter", "visual"),
    ]
    for adapter_dir, target in adapter_candidates:
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model_paths = [
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        ]
        if adapter_config.exists() and any(p.exists() for p in adapter_model_paths):
            info["has_adapter_files"] = True
            info["has_adapter_weights"] = True
            info["target_module"] = info["target_module"] or target
            info["adapter_load_path"] = str(adapter_dir)
            return info

    keys: list[str] = []
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        keys.extend(index_data.get("weight_map", {}).keys())
    else:
        for safetensors_path in path.glob("*.safetensors"):
            try:
                with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                    keys.extend(list(f.keys()))
            except Exception:
                continue

    info["target_module"] = info["target_module"] or _infer_peft_target_from_keys(keys)
    info["has_adapter_weights"] = info["target_module"] is not None
    return info


def _get_checkpoint_safetensors_files(checkpoint_path: Path, prefer_model_shards: bool = False) -> list[Path]:
    """Return checkpoint safetensors files, optionally preferring full model shards over sidecar adapter files."""
    if prefer_model_shards:
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
            weight_files = sorted(set(index_data.get("weight_map", {}).values()))
            indexed_files = [checkpoint_path / name for name in weight_files if (checkpoint_path / name).exists()]
            if indexed_files:
                return indexed_files

        model_shards = sorted(checkpoint_path.glob("model*.safetensors"))
        if model_shards:
            return model_shards

    return sorted(checkpoint_path.glob("*.safetensors"))


def _checkpoint_has_full_model_shards(checkpoint_path: str) -> bool:
    """Whether a checkpoint directory contains self-contained model safetensors shards."""
    path = Path(checkpoint_path)
    return (path / "model.safetensors.index.json").exists() or any(path.glob("model*.safetensors"))


def _load_custom_heads_from_safetensors(model: RBM, checkpoint_path: str) -> bool:
    """Load custom RBM heads from a dedicated safetensors file if present."""
    custom_heads_path = Path(checkpoint_path) / "custom_heads.safetensors"
    if not custom_heads_path.exists():
        return False

    custom_state = load_file(str(custom_heads_path))
    model.load_state_dict(custom_state, strict=False)
    logger.info(f"Loaded {len(custom_state)} custom head tensors from {custom_heads_path}")
    return True


def _load_checkpoint_weights_from_safetensors(
    model,
    checkpoint_path: str,
    cfg: ModelConfig,
    load_adapters: bool = True,
    prefer_model_shards: bool = False,
) -> None:
    """
    Load checkpoint weights from safetensors files in a checkpoint directory.
    Includes verification for PEFT adapters and progress_head.

    This is needed when using Unsloth, as we can't use from_pretrained on checkpoints.
    Instead, we load the base model with Unsloth first, then manually load the checkpoint weights.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint directory containing safetensors files
        cfg: Model configuration for verification
        load_adapters: If False, skip loading adapter weights (assumes already loaded via PeftModel.from_pretrained)
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")

    # Collect safetensors files. For self-contained checkpoints, prefer model shards and
    # ignore sidecar adapter/custom-head files that may duplicate the same weights.
    safetensors_files = _get_checkpoint_safetensors_files(
        checkpoint_path,
        prefer_model_shards=prefer_model_shards,
    )
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in checkpoint directory: {checkpoint_path}")

    logger.info(f"Loading checkpoint weights from {len(safetensors_files)} safetensors file(s) in {checkpoint_path}")

    # Capture before weights for verification (adapter and progress_head)
    before_weights = {}
    before_progress_head = model.progress_head[0].weight.clone()
    before_weights["progress_head"] = before_progress_head

    # Capture adapter weights before loading (if PEFT is enabled)
    adapter_keys_before = []
    sample_adapter_keys = []
    if cfg.use_peft:
        model_state_dict = model.state_dict()
        adapter_keys_before = [k for k in model_state_dict.keys() if "lora_A" in k or "lora_B" in k]
        if adapter_keys_before:
            # Sample a few adapter weights to verify they change
            sample_adapter_keys = adapter_keys_before[:3]  # Check first 3 adapters
            for key in sample_adapter_keys:
                before_weights[f"adapter_{key}"] = model_state_dict[key].clone()
            logger.info(f"Found {len(adapter_keys_before)} adapter parameters in model before loading checkpoint")
        else:
            logger.warning("No adapter parameters found in model - PEFT may not be applied correctly")

    # Load all safetensors files and merge into a single state dict
    checkpoint_state_dict = {}
    for safetensors_file in safetensors_files:
        logger.debug(f"Loading weights from {safetensors_file.name}")
        file_state_dict = load_file(str(safetensors_file))
        checkpoint_state_dict.update(file_state_dict)

    # Check what adapter keys are in checkpoint
    checkpoint_adapter_keys = [k for k in checkpoint_state_dict.keys() if "lora_A" in k or "lora_B" in k]
    logger.info(f"Found {len(checkpoint_adapter_keys)} adapter parameters in checkpoint")
    if checkpoint_adapter_keys:
        logger.debug(f"Sample checkpoint adapter keys: {checkpoint_adapter_keys[:5]}")

    # If load_adapters=False, filter out adapter keys (they were already loaded via PeftModel.from_pretrained)
    if not load_adapters:
        logger.info("Skipping adapter weights (already loaded via PeftModel.from_pretrained)")
        checkpoint_state_dict = {
            k: v for k, v in checkpoint_state_dict.items() if "lora_A" not in k and "lora_B" not in k
        }

    # Get model's expected state dict keys
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())

    # Log sample model keys to understand structure
    sample_model_keys = [k for k in list(model_keys)[:10]]
    logger.info(f"Sample model keys (first 10): {sample_model_keys}")

    # Log sample checkpoint keys to understand structure
    sample_ckpt_keys = [k for k in list(checkpoint_state_dict.keys())[:10]]
    logger.info(f"Sample checkpoint keys (first 10): {sample_ckpt_keys}")

    # Check if model has adapter keys and what structure they use
    model_adapter_keys = [k for k in model_keys if "lora_A" in k or "lora_B" in k]
    if model_adapter_keys:
        logger.info(f"Found {len(model_adapter_keys)} adapter keys in model")
        logger.debug(f"Sample model adapter keys: {model_adapter_keys[:3]}")

    # Remap checkpoint keys to match model structure
    # For PEFT models wrapped in RBM: checkpoint has "model.model." but model expects "model.base_model.model.model."
    # Try multiple strategies: direct match, map "model.model." -> "model.base_model.model.model.", etc.
    remapped_state_dict = {}
    remapped_count = 0
    direct_match_count = 0
    remap_strategies = {}

    for ckpt_key, ckpt_value in checkpoint_state_dict.items():
        if ckpt_key in model_keys:
            # Direct match - use as is
            remapped_state_dict[ckpt_key] = ckpt_value
            direct_match_count += 1
        else:
            # Try different remapping strategies
            potential_keys = []

            # Strategy 1 (PEFT): Map "model.model." -> "model.base_model.model.model." (for PEFT wrapped in RBM)
            # This handles the case where Unsloth saved the full model from model.model
            if ckpt_key.startswith("model.model."):
                # For PEFT: model.model.* -> model.base_model.model.model.*
                peft_key = ckpt_key.replace("model.model.", "model.base_model.model.model.", 1)
                potential_keys.append(peft_key)
                # Also try: model.model.* -> model.* (fallback)
                potential_keys.append(ckpt_key.replace("model.model.", "model.", 1))

            # Strategy 2: Remove "model." prefix entirely
            if ckpt_key.startswith("model."):
                potential_keys.append(ckpt_key.replace("model.", "", 1))

            # Strategy 3: Add "model." prefix if missing
            if not ckpt_key.startswith("model."):
                potential_keys.append(f"model.{ckpt_key}")

            # Strategy 4: Try adding "base_model." between "model." and the rest (for non-PEFT wrapped models)
            if ckpt_key.startswith("model.") and not ckpt_key.startswith("model.base_model."):
                parts = ckpt_key.split(".", 1)
                if len(parts) == 2:
                    potential_keys.append(f"model.base_model.{parts[1]}")

            # Try each potential key
            matched = False
            for potential_key in potential_keys:
                if potential_key in model_keys:
                    remapped_state_dict[potential_key] = ckpt_value
                    remapped_count += 1
                    strategy = f"{ckpt_key} -> {potential_key}"
                    remap_strategies[strategy] = remap_strategies.get(strategy, 0) + 1
                    if remapped_count <= 5:  # Log first 5 remappings
                        logger.debug(f"Remapped: {strategy}")
                    matched = True
                    break

            if not matched:
                # Key still doesn't match, will be in unexpected_keys
                remapped_state_dict[ckpt_key] = ckpt_value

    if remapped_count > 0:
        logger.info(
            f"Remapped {remapped_count} checkpoint keys to match model structure (direct matches: {direct_match_count})"
        )
        if remap_strategies:
            logger.debug(f"Remapping strategies used: {dict(list(remap_strategies.items())[:5])}")

    # Load remapped state dict into model with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

    # Filter missing keys - base model keys are expected for PEFT checkpoints
    base_model_missing = [
        k
        for k in missing_keys
        if any(
            pattern in k for pattern in ["visual.", "language_model.", "text_encoder.", "text_model.", "embed_tokens"]
        )
    ]
    other_missing = [k for k in missing_keys if k not in base_model_missing]

    if base_model_missing:
        logger.info(f"Missing base model keys (expected for PEFT checkpoints): {len(base_model_missing)} keys")
    if other_missing:
        logger.warning(f"Missing non-base model keys: {len(other_missing)} keys")
        logger.debug(
            f"Missing keys: {other_missing[:10]}..." if len(other_missing) > 10 else f"Missing keys: {other_missing}"
        )

    # Filter unexpected keys - adapter keys might be expected if structure differs slightly
    adapter_unexpected = [k for k in unexpected_keys if "lora_A" in k or "lora_B" in k]
    other_unexpected = [k for k in unexpected_keys if k not in adapter_unexpected]

    if adapter_unexpected:
        logger.warning(f"Unexpected adapter keys in checkpoint (not in model): {len(adapter_unexpected)} keys")
        logger.debug(
            f"Unexpected adapter keys: {adapter_unexpected[:10]}..."
            if len(adapter_unexpected) > 10
            else f"Unexpected adapter keys: {adapter_unexpected}"
        )
    if other_unexpected:
        logger.warning(f"Unexpected non-adapter keys: {len(other_unexpected)} keys")
        logger.debug(
            f"Unexpected keys: {other_unexpected[:10]}..."
            if len(other_unexpected) > 10
            else f"Unexpected keys: {other_unexpected}"
        )

    # Verify progress_head loaded correctly
    after_progress_head = model.progress_head[0].weight
    progress_head_loaded = not torch.allclose(before_progress_head, after_progress_head, atol=1e-6)

    logger.info(
        f"Progress head - Before: shape={before_progress_head.shape}, sum={before_progress_head.sum():.6f} | "
        f"After: shape={after_progress_head.shape}, sum={after_progress_head.sum():.6f} | "
        f"Loaded: {progress_head_loaded}"
    )

    if not progress_head_loaded:
        logger.error("Progress head weights did not change after loading checkpoint!")
        logger.error("This indicates the checkpoint weights were not loaded correctly.")

    # Verify adapter weights loaded correctly only when this helper loads them.
    # If load_adapters=False, PeftModel.from_pretrained has already loaded adapters
    # and this helper only restores custom RBM weights.
    adapter_loaded_correctly = True
    if cfg.use_peft and adapter_keys_before and load_adapters:
        model_state_dict_after = model.state_dict()
        adapter_keys_after = [k for k in model_state_dict_after.keys() if "lora_A" in k or "lora_B" in k]

        logger.info(f"Adapter keys - Before: {len(adapter_keys_before)} | After: {len(adapter_keys_after)}")

        # Check if sample adapter weights changed
        for key in sample_adapter_keys:
            if key in before_weights:
                before_adapter = before_weights[f"adapter_{key}"]
                if key in model_state_dict_after:
                    after_adapter = model_state_dict_after[key]
                    adapter_changed = not torch.allclose(before_adapter, after_adapter, atol=1e-6)
                    logger.info(
                        f"Adapter {key} - Before: shape={before_adapter.shape}, sum={before_adapter.sum():.6f} | "
                        f"After: shape={after_adapter.shape}, sum={after_adapter.sum():.6f} | "
                        f"Loaded: {adapter_changed}"
                    )
                    if not adapter_changed:
                        logger.warning(f"Adapter {key} weights did not change after loading checkpoint!")
                        adapter_loaded_correctly = False
                else:
                    logger.warning(f"Adapter key {key} not found in model after loading!")
                    adapter_loaded_correctly = False

        # Check how many adapter keys from checkpoint were actually loaded
        # Need to check both original and remapped keys (matching the remapping strategies above)
        loaded_adapter_keys = []
        for ckpt_key in checkpoint_adapter_keys:
            # Check if original key exists in model
            if ckpt_key in model_state_dict_after:
                loaded_adapter_keys.append(ckpt_key)
            else:
                # Try remapping strategies to find the actual key used in model
                potential_keys = []
                if ckpt_key.startswith("model.model."):
                    # Strategy 1: PEFT wrapped in RBM
                    potential_keys.append(ckpt_key.replace("model.model.", "model.base_model.model.model.", 1))
                    # Strategy 2: Fallback
                    potential_keys.append(ckpt_key.replace("model.model.", "model.", 1))

                # Check if any remapped key exists in model
                for remapped_key in potential_keys:
                    if remapped_key in model_state_dict_after:
                        loaded_adapter_keys.append(ckpt_key)  # Count original key as loaded
                        break
        logger.info(f"Loaded {len(loaded_adapter_keys)}/{len(checkpoint_adapter_keys)} adapter keys from checkpoint")

        if len(loaded_adapter_keys) == 0:
            logger.error("No adapter weights were loaded from checkpoint!")
            adapter_loaded_correctly = False
        elif len(loaded_adapter_keys) < len(checkpoint_adapter_keys) * 0.5:  # Less than 50% loaded
            logger.warning(
                f"Only {len(loaded_adapter_keys)}/{len(checkpoint_adapter_keys)} adapter keys loaded - may indicate structure mismatch"
            )
            adapter_loaded_correctly = False

    if not adapter_loaded_correctly:
        logger.error("Adapter weights did not load correctly!")
        import ipdb

        ipdb.set_trace()  # Breakpoint if adapters didn't load correctly

    logger.info(f"Successfully loaded checkpoint weights from {checkpoint_path}")


def _load_base_model_with_unsloth(
    cfg: ModelConfig,
    torch_dtype: torch.dtype,
    extra_kwargs: dict,
    peft_config: Optional[PEFTConfig] = None,
    loading_from_checkpoint: bool = False,
    apply_peft: bool = True,
) -> Tuple[Any, Any]:
    """
    Load base model using Unsloth's FastVisionModel.

    Args:
        cfg: Model configuration
        torch_dtype: Torch dtype to use
        extra_kwargs: Extra kwargs for model loading (e.g., attn_implementation)
        peft_config: Optional PEFT configuration
        loading_from_checkpoint: If True, skip PEFT application (checkpoint already has weights)
        apply_peft: If False, do not apply PEFT (e.g. when loading from checkpoint that has no adapter files; PEFT added later in train.py)

    Returns:
        Tuple of (base_model, tokenizer)
    """
    logger.info("Using Unsloth for faster training with Qwen model")

    # Load model with unsloth
    base_model, tokenizer = FastVisionModel.from_pretrained(
        cfg.base_model_id,
        load_in_4bit=cfg.quantization,  # Use 4bit if quantization is enabled
        use_gradient_checkpointing="unsloth",  # Use unsloth's optimized checkpointing
        dtype=torch_dtype,  # Set the dtype from config,
        full_finetuning=True if not cfg.use_peft else False,
        device_map=None,
        attn_implementation=extra_kwargs["attn_implementation"],
        trust_remote_code=True,
    )

    # Apply PEFT if enabled (skip when apply_peft=False, e.g. checkpoint has no adapter files; train.py will add PEFT later)
    if apply_peft and cfg.use_peft and peft_config:
        if loading_from_checkpoint:
            logger.info("Applying PEFT configuration to base model (needed to load adapter weights from checkpoint)")
        else:
            logger.info("Applying PEFT configuration to base model")
        base_model = FastVisionModel.get_peft_model(
            base_model,
            finetune_vision_layers=cfg.train_vision_encoder,
            finetune_language_layers=cfg.train_language_model,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
        )
    elif cfg.use_peft and not apply_peft:
        logger.info("Skipping PEFT here; checkpoint has no adapter files, PEFT will be added in train.py")

    # Extract inner model after PEFT is applied (if needed for RBM wrapper)
    # IMPORTANT: After FastVisionModel.get_peft_model(), base_model.model should be a PeftModel
    # We keep it as PeftModel so we can use PeftModel.from_pretrained() later
    if cfg.model_type == "default":
        # Extract the inner model, which should be a PeftModel if PEFT was applied
        inner_model = base_model.model
        # Check if it's a PeftModel - if so, keep it as PeftModel for proper adapter loading
        if cfg.use_peft and isinstance(inner_model, PeftModel):
            logger.info(
                "Base model inner model is a PeftModel - will use PeftModel.from_pretrained() for adapter loading"
            )
            base_model = inner_model
        else:
            base_model = inner_model

    return base_model, tokenizer


def _load_base_model_standard(
    cfg: ModelConfig,
    torch_dtype: torch.dtype,
    extra_kwargs: dict,
    bnb: Optional[BitsAndBytesConfig],
) -> Any:
    """
    Load base model using standard transformers loading.

    Args:
        cfg: Model configuration
        torch_dtype: Torch dtype to use
        extra_kwargs: Extra kwargs for model loading (e.g., attn_implementation)
        bnb: Optional BitsAndBytesConfig for quantization

    Returns:
        Base model
    """
    # Check if it's Molmo, Qwen3 or Qwen2/2.5
    is_molmo = "Molmo" in cfg.base_model_id
    is_qwen3 = ("Qwen3" in cfg.base_model_id or "qwen3" in cfg.base_model_id.lower()) and HAS_QWEN3

    # Select appropriate model classes based on version and model type
    if is_molmo:
        # Molmo2 uses AutoModelForImageTextToText with trust_remote_code
        base_model = AutoModelForImageTextToText.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
            **extra_kwargs,
            quantization_config=bnb,
        )
        # Extract the base model for RBM
        base_model = base_model.model
        logger.info("Using Molmo2 models")
    elif is_qwen3:
        qwen_model_cls = Qwen3VLModel
        base_model = qwen_model_cls.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            **extra_kwargs,
            quantization_config=bnb,
        )
        logger.info("Using Qwen3 models")
    else:
        qwen_model_cls = Qwen2_5_VLModel if cfg.model_type == "default" else Qwen2_5_VLForConditionalGeneration
        base_model = qwen_model_cls.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            **extra_kwargs,
            quantization_config=bnb,
        )
        logger.info("Using Qwen2/2.5 models")

    return base_model


def _setup_processor_and_tokenizer(cfg: ModelConfig) -> AutoProcessor:
    """
    Setup processor and tokenizer for the model.

    Args:
        cfg: Model configuration

    Returns:
        Processor
    """
    if "SmolVLM" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
            size={"longest_edge": 512},
            max_image_size={"longest_edge": 512},
            use_fast=True,
        )
        logger.info(f"SmolVLM Processor: {processor}")
    elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
            # padding_side="left",
            padding_side="right",
        )
        logger.info(f"Qwen Processor: {processor}")
    else:
        raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor


def _add_special_tokens_and_resize(cfg: ModelConfig, processor: AutoProcessor, base_model: Any) -> None:
    """
    Add RBM special tokens and resize token embeddings if needed.

    Args:
        cfg: Model configuration
        processor: Processor with tokenizer
        base_model: Base model to resize embeddings for
    """
    # Add RBM special tokens if they don't exist
    special_tokens = [
        "<|split_token|>",
        "<|reward_token|>",
        "<|pref_token|>",
        "<|sim_token|>",
        "<|prog_token|>",  # Per-frame progress token
    ]
    logger.info(f"Before adding special tokens: {len(processor.tokenizer.get_vocab())}")
    num_added = 0
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            added = processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            num_added += added

    logger.info(f"Added {num_added} special tokens")

    base_model.resize_token_embeddings(len(processor.tokenizer))
    logger.info(f"Resized token embeddings to {len(processor.tokenizer)}")

    # import ipdb; ipdb.set_trace()
    # # Resize token embeddings if new tokens were added
    # vocab_size = (
    #     base_model.config.text_config.vocab_size
    #     if ("Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id)
    #     else base_model.config.vocab_size
    # )

    # if len(processor.tokenizer) != vocab_size:
    #     logger.info(f"Resizing token embeddings from {vocab_size} to {len(processor.tokenizer)}")

    #     is_molmo = "Molmo" in cfg.base_model_id
    #     if is_molmo:
    #         # Custom resize for Molmo2 - its Molmo2Embedding stores embedding as a Parameter directly
    #         new_vocab_size = len(processor.tokenizer)
    #         _embed_layer = base_model.get_input_embeddings()

    #         # Check if embedding is a Parameter (tensor) directly, or an nn.Embedding
    #         if hasattr(_embed_layer, "embedding"):
    #             old_embed_attr = _embed_layer.embedding

    #             # Case 1: embedding is a Parameter (raw tensor)
    #             if isinstance(old_embed_attr, torch.nn.Parameter):
    #                 old_num_tokens, embedding_dim = old_embed_attr.shape

    #                 # Create new parameter with expanded vocab
    #                 new_embed_data = torch.zeros(
    #                     new_vocab_size, embedding_dim, device=old_embed_attr.device, dtype=old_embed_attr.dtype
    #                 )

    #                 # Copy existing weights
    #                 new_embed_data[:old_num_tokens] = old_embed_attr.data

    #                 # Initialize new token embeddings using mean of existing embeddings
    #                 mean_embedding = old_embed_attr.data.mean(dim=0)
    #                 new_embed_data[old_num_tokens:] = mean_embedding.unsqueeze(0).expand(
    #                     new_vocab_size - old_num_tokens, -1
    #                 )

    #                 # Replace the embedding Parameter
    #                 _embed_layer.embedding = torch.nn.Parameter(new_embed_data)

    #                 # Also update config to reflect new vocab size
    #                 base_model.config.text_config.vocab_size = new_vocab_size

    #                 logger.info(
    #                     f"Custom resized Molmo2 embeddings (Parameter) from {old_num_tokens} to {new_vocab_size}"
    #                 )

    #             # Case 2: embedding is an nn.Embedding with .weight
    #             elif hasattr(old_embed_attr, "weight"):
    #                 old_num_tokens, embedding_dim = old_embed_attr.weight.shape

    #                 # Create new embedding layer with expanded vocab
    #                 new_embedding = torch.nn.Embedding(
    #                     new_vocab_size,
    #                     embedding_dim,
    #                     device=old_embed_attr.weight.device,
    #                     dtype=old_embed_attr.weight.dtype,
    #                 )

    #                 # Copy existing weights
    #                 new_embedding.weight.data[:old_num_tokens] = old_embed_attr.weight.data

    #                 # Initialize new token embeddings using mean of existing embeddings
    #                 mean_embedding = old_embed_attr.weight.data.mean(dim=0)
    #                 new_embedding.weight.data[old_num_tokens:] = mean_embedding.unsqueeze(0).expand(
    #                     new_vocab_size - old_num_tokens, -1
    #                 )

    #                 # Replace the nested embedding
    #                 _embed_layer.embedding = new_embedding

    #                 # Also update config to reflect new vocab size
    #                 base_model.config.text_config.vocab_size = new_vocab_size

    #                 logger.info(
    #                     f"Custom resized Molmo2 embeddings (Embedding) from {old_num_tokens} to {new_vocab_size}"
    #                 )
    #             else:
    #                 logger.warning(f"Cannot resize Molmo2 embeddings - unknown embedding type: {type(old_embed_attr)}")
    #         else:
    #             logger.warning(f"Cannot resize Molmo2 embeddings - no embedding attribute found")
    #     else:
    #         base_model.resize_token_embeddings(len(processor.tokenizer))
    #         logger.info(f"Resized token embeddings to {len(processor.tokenizer)}")


def _verify_checkpoint_loading(cfg: ModelConfig, model: Any, before_weights: dict) -> None:
    """
    Verify that checkpoint weights were loaded correctly by comparing before/after weights.

    Args:
        cfg: Model configuration
        model: The model after loading checkpoint
        before_weights: Dictionary of weights before loading (keys: visual, progress_head, lm_embed_tokens, lm_layer)
    """

    if "Qwen2.5" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.down_proj.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.linear_fc1.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    else:
        return

    before_visual = before_weights["visual"]
    before_progress_head = before_weights["progress_head"]
    before_lm_embed_tokens = before_weights["lm_embed_tokens"]
    before_lm_layer = before_weights["lm_layer"]

    logger.info(
        f"Before visual: {before_visual.shape}, {before_visual.sum()} | After visual: {after_visual.shape}, {after_visual.sum()}"
    )
    logger.info(
        f"Before progress head: {before_progress_head.shape}, {before_progress_head.sum()} | After progress head: {after_progress_head.shape}, {after_progress_head.sum()}"
    )
    logger.info(
        f"Before LM embed tokens: {before_lm_embed_tokens.shape}, {before_lm_embed_tokens.sum()} | After LM embed tokens: {after_lm_embed_tokens.shape}, {after_lm_embed_tokens.sum()}"
    )
    logger.info(
        f"Before LM layer: {before_lm_layer.shape}, {before_lm_layer.sum()} | After LM layer: {after_lm_layer.shape}, {after_lm_layer.sum()}"
    )

    # check that before and after are different
    if torch.allclose(before_visual, after_visual):
        logger.warning("Before and after visual are the same! Check if you loaded the pretrained model correctly")
    if torch.allclose(before_progress_head, after_progress_head):
        logger.warning(
            "Before and after progress head are the same! Check if you loaded the pretrained model correctly"
        )
    if torch.allclose(before_lm_embed_tokens, after_lm_embed_tokens):
        logger.warning(
            "Before and after LM embed tokens are the same! Check if you loaded the pretrained model correctly"
        )


def setup_model_and_processor(
    cfg: ModelConfig, hf_model_id: str = "", peft_config: PEFTConfig = None
) -> tuple[AutoProcessor, RBM]:
    """
    Shared function to set up model, processor, and tokenizer for both training and evaluation.

    Args:
        cfg: Model configuration
        hf_model_id: Optional HuggingFace model ID to load from

    Note:
        When use_unsloth is enabled for Qwen models:
        - The model will be loaded using unsloth's FastVisionModel
        - Automatically uses optimized gradient checkpointing
        - If use_peft is enabled, applies unsloth's optimized PEFT configuration
        - Use unsloth/Qwen models for best performance (e.g., unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit)
    """

    # Convert string dtype to torch dtype (used across all model loading paths)
    torch_dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
    logger.info(f"Using torch dtype: {torch_dtype}")

    # Check if unsloth should be used
    use_unsloth = cfg.use_unsloth and "Qwen" in cfg.base_model_id

    if use_unsloth:
        logger.info("Unsloth mode enabled for faster training")

    # If quantization is enabled, use bitsandbytes (unless using unsloth)
    if cfg.quantization and not use_unsloth:
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb = None

    try:
        import flash_attn  # noqa: F401

        logger.info("Flash Attention 2 CUDA is available")
        has_flash_attn = True
    except:
        logger.info("Flash Attention 2 CUDA is not available")
        has_flash_attn = False

    if has_flash_attn:
        extra_kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        extra_kwargs = {"attn_implementation": "sdpa"}

    # Determine if we're loading from a checkpoint and inspect any PEFT artifacts/weights.
    loading_from_checkpoint = bool(hf_model_id)
    checkpoint_path_for_load: Optional[str] = None
    checkpoint_peft_info = {
        "has_adapter_files": False,
        "has_adapter_weights": False,
        "target_module": None,
        "adapter_load_path": None,
    }
    if hf_model_id:
        hub_token = os.environ.get("HF_TOKEN")
        checkpoint_path_for_load = resolve_checkpoint_path(hf_model_id, hub_token=hub_token)
        if checkpoint_path_for_load:
            checkpoint_peft_info = _inspect_checkpoint_for_peft(checkpoint_path_for_load)
            if cfg.use_peft:
                if checkpoint_peft_info["has_adapter_files"]:
                    logger.info(
                        f"Checkpoint contains standalone PEFT adapter files for target "
                        f"{checkpoint_peft_info['target_module']}"
                    )
                elif checkpoint_peft_info["has_adapter_weights"]:
                    logger.info(
                        f"Checkpoint stores PEFT weights inside safetensors for target "
                        f"{checkpoint_peft_info['target_module']}; will reconstruct PEFT before loading"
                    )

    # For checkpoint loading, attach PEFT after wrapping in RBM so nested adapter layouts are preserved.
    apply_peft_before_wrap = cfg.use_peft and not loading_from_checkpoint

    # Load processor and tokenizer
    if "SmolVLM" in cfg.base_model_id or "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        if "SmolVLM" in cfg.base_model_id:
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                padding_side="left",
                size={"longest_edge": 512},
                max_image_size={"longest_edge": 512},
                use_fast=True,
            )
            logger.info(f"SmolVLM Processor: {processor}")

            base_model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                **extra_kwargs,
                quantization_config=bnb,
            )
            model_cls = RBM

        elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
            # Load base model (with or without Unsloth)
            if use_unsloth:
                base_model, tokenizer = _load_base_model_with_unsloth(
                    cfg,
                    torch_dtype,
                    extra_kwargs,
                    peft_config,
                    loading_from_checkpoint=loading_from_checkpoint,
                    apply_peft=apply_peft_before_wrap,
                )
            else:
                base_model = _load_base_model_standard(cfg, torch_dtype, extra_kwargs, bnb)
                tokenizer = None  # Will be loaded with processor

            model_cls = RBM

            # Setup processor and tokenizer
            processor = _setup_processor_and_tokenizer(cfg)
            if tokenizer is None:
                tokenizer = processor.tokenizer

        else:
            raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

        # CRITICAL: Ensure PEFT is applied to base_model BEFORE wrapping in RBM (when checkpoint has adapters or we're not loading)
        if apply_peft_before_wrap and cfg.use_peft and not isinstance(base_model, PeftModel):
            logger.warning("PEFT is enabled but base_model is not a PeftModel. Applying PEFT now...")
            if peft_config is None:
                raise ValueError("PEFT is enabled but peft_config is None. Cannot apply PEFT without configuration.")

            # Apply PEFT to base_model
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=peft_config.r,
                lora_alpha=peft_config.lora_alpha,
                target_modules=peft_config.target_modules,
                lora_dropout=peft_config.lora_dropout,
                bias=peft_config.bias,
            )
            base_model = get_peft_model(base_model, lora_config)
            logger.info("Applied PEFT to base_model before wrapping in RBM")

        # Verify PEFT was applied when we expect it
        if apply_peft_before_wrap and cfg.use_peft:
            if isinstance(base_model, PeftModel):
                logger.info("Confirmed: base_model is a PeftModel - ready to load adapter weights from checkpoint")
            else:
                logger.error("CRITICAL: PEFT is enabled but base_model is not a PeftModel after applying PEFT!")
                raise ValueError(
                    "Failed to apply PEFT to base_model. Cannot load adapter weights without PeftModel structure."
                )

        # Add special tokens and resize embeddings
        _add_special_tokens_and_resize(cfg, processor, base_model)

        # Initialize RBM model wrapper with the pre-loaded base model
        logger.info("Initializing RBM model...")
        tokenizer = processor.tokenizer

        model = model_cls(
            config=base_model.config,
            processor=processor,
            tokenizer=tokenizer,
            base_model=base_model,
            base_model_id=cfg.base_model_id,
            model_config=cfg,  # Pass ModelConfig for RBM-specific settings
        )

        # Load checkpoint if provided
        if hf_model_id:
            repo_id, revision_to_load = parse_hf_model_id_and_revision(hf_model_id, model_name="model")
            checkpoint_path = checkpoint_path_for_load
            if checkpoint_path is None:
                hub_token = os.environ.get("HF_TOKEN")
                checkpoint_path = resolve_checkpoint_path(hf_model_id, hub_token=hub_token)
            if checkpoint_path is None:
                raise ValueError(f"Could not resolve checkpoint path: {hf_model_id}")

            if cfg.use_peft:
                checkpoint_target = checkpoint_peft_info["target_module"] or (
                    "visual" if peft_config and peft_config.peft_vision_encoder else "language_model"
                )
                has_adapter_files = checkpoint_peft_info["has_adapter_files"]
                has_adapter_weights = checkpoint_peft_info["has_adapter_weights"]
                adapter_load_path = checkpoint_peft_info["adapter_load_path"] or checkpoint_path

                if (has_adapter_files or has_adapter_weights) and not model_has_peft(model):
                    if peft_config is None:
                        raise ValueError("PEFT is enabled but peft_config is None. Cannot reconstruct adapters.")
                    logger.info(f"Attaching PEFT to target '{checkpoint_target}' before loading checkpoint")
                    model = setup_peft_model(model, peft_config, target_module=checkpoint_target)

                if _checkpoint_has_full_model_shards(checkpoint_path):
                    logger.info(
                        "Checkpoint includes full model safetensors shards; "
                        "preferring embedded full weights over standalone adapter files"
                    )
                    _load_checkpoint_weights_from_safetensors(
                        model,
                        checkpoint_path,
                        cfg,
                        load_adapters=True,
                        prefer_model_shards=True,
                    )
                elif has_adapter_files:
                    logger.info(
                        f"Loading PEFT adapters from {adapter_load_path} into RBM target '{checkpoint_target}'"
                    )
                    try:
                        if checkpoint_target == "model":
                            model.model = PeftModel.from_pretrained(model.model, adapter_load_path)
                        elif checkpoint_target == "visual":
                            model.model.visual = PeftModel.from_pretrained(model.model.visual, adapter_load_path)
                        else:
                            model.model.language_model = PeftModel.from_pretrained(
                                model.model.language_model, adapter_load_path
                            )
                        logger.info("Successfully loaded PEFT adapters using PeftModel.from_pretrained()")
                        if not _load_custom_heads_from_safetensors(model, checkpoint_path):
                            logger.info("No custom_heads.safetensors found; falling back to generic safetensors loader")
                            _load_checkpoint_weights_from_safetensors(
                                model,
                                checkpoint_path,
                                cfg,
                                load_adapters=False,
                                prefer_model_shards=True,
                            )
                    except Exception as e:
                        logger.warning(f"PeftModel.from_pretrained() failed: {e}")
                        logger.info("Falling back to manual loading for all weights")
                        _load_checkpoint_weights_from_safetensors(
                            model,
                            checkpoint_path,
                            cfg,
                            load_adapters=True,
                            prefer_model_shards=True,
                        )
                elif has_adapter_weights:
                    logger.info("Checkpoint has embedded PEFT weights but no standalone adapter files")
                    _load_checkpoint_weights_from_safetensors(
                        model,
                        checkpoint_path,
                        cfg,
                        load_adapters=True,
                        prefer_model_shards=True,
                    )
                else:
                    logger.info("Checkpoint has no PEFT adapter weights - loading plain model weights")
                    _load_checkpoint_weights_from_safetensors(
                        model,
                        checkpoint_path,
                        cfg,
                        load_adapters=True,
                        prefer_model_shards=True,
                    )
            else:
                # For non-PEFT models, we can use from_pretrained as before
                # Capture before weights for verification
                before_weights = {}
                if "Qwen2.5" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.down_proj.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }
                elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.linear_fc1.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }

                # Load the model from the evaluation path
                model = model_cls.from_pretrained(
                    repo_id,
                    processor=processor,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    base_model_id=cfg.base_model_id,
                    model_config=cfg,
                    revision=revision_to_load,
                )

                # Verify weights were loaded
                if before_weights:
                    _verify_checkpoint_loading(cfg, model, before_weights)

    # elif "rewind_transformer" in cfg.base_model_id or "rewind_scale_transformer" in cfg.base_model_id:
    elif "rewind" in cfg.base_model_id:
        # Initialize new model with encoders
        # Pretrained image and text encoders
        image_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
        text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        if hf_model_id:
            repo_id, revision_to_load = parse_hf_model_id_and_revision(hf_model_id, model_name="ReWiND model")

            model = ReWiNDTransformer.from_pretrained(
                repo_id,
                processor=processor,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                revision=revision_to_load,
            )
        else:
            train_img = cfg.train_vision_encoder
            train_text = cfg.train_language_model

            for p in image_encoder.parameters():
                p.requires_grad = train_img

            for p in text_encoder.parameters():
                p.requires_grad = train_text

            logger.info("Initializing ReWiND model...")
            model = ReWiNDTransformer(
                config=cfg,
                processor=processor,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
            )

    logger.info("Model architecture initialized")
    logger.info(f"Model architecture: {model}")

    # Configure which parts of the model to train based on config
    # IMPORTANT: When using PEFT (via Unsloth or standard), PEFT already handles freezing
    # base model parameters. We should NOT override requires_grad on base model params.
    peft_applied = cfg.use_peft and (cfg.use_unsloth or cfg.peft_vision_encoder)

    # Helper function to check if a parameter is part of the base model (vision/language)
    def is_base_model_param(name: str) -> bool:
        """Check if parameter belongs to base model (vision/language) that PEFT handles."""
        base_model_patterns = ["visual", "vision", "language_model", "text_encoder", "text_model", "image_encoder"]
        return any(pattern in name for pattern in base_model_patterns)

    # Helper function to check if a parameter is a prediction head
    def is_prediction_head(name: str) -> bool:
        """Check if parameter belongs to a prediction head."""
        head_patterns = ["progress_head", "success_head", "preference_head", "similarity_head"]
        return any(pattern in name for pattern in head_patterns)

    for name, param in model.named_parameters():
        # 1. Handle prediction heads - always controlled by their individual flags
        if is_prediction_head(name):
            if "progress_head" in name:
                param.requires_grad = cfg.train_progress_head
            elif "success_head" in name:
                param.requires_grad = cfg.train_success_head
            elif "preference_head" in name:
                param.requires_grad = cfg.train_preference_head
            elif "similarity_head" in name:
                param.requires_grad = False

        # 2. Handle base model parameters (vision/language) - skip if PEFT is applied
        elif is_base_model_param(name):
            if peft_applied:
                # PEFT handles freezing/unfreezing - don't override
                continue

            # Set requires_grad based on config flags
            if "visual" in name or "vision" in name or "vision_model" in name:
                param.requires_grad = cfg.train_vision_encoder
            elif "language_model" in name or "text_encoder" in name or "text_model" in name:
                param.requires_grad = cfg.train_language_model
            elif "image_encoder" in name:
                param.requires_grad = cfg.train_vision_encoder

        # 3. Handle special cases
        elif "lm_head" in name:
            # Language modeling head should not be trainable for RBM
            param.requires_grad = False

        # 4. All other parameters (custom RBM parameters like frame_pool_attn, video_proj, text_proj)
        # should always be trainable
        else:
            param.requires_grad = True

    logger.info("Training configuration:")
    logger.info(f"  - Vision encoder: {cfg.train_vision_encoder}")
    logger.info(f"  - Language model: {cfg.train_language_model}")
    logger.info(f"  - Progress head: {cfg.train_progress_head}")
    logger.info(f"  - Success head: {getattr(cfg, 'train_success_head', False)}")
    logger.info(f"  - Preference head: {cfg.train_preference_head}")

    # When use_peft, skip verbose param list here; it will be printed after PEFT in setup_peft_model
    if not cfg.use_peft:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name:60} | {param.shape} | RG: {param.requires_grad}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return tokenizer, processor, model


def _get_vl_inner_model(rbm_model: RBM):
    """Return the inner VL model (with .visual and .language_model) from RBM, unwrapping PeftModel if present."""
    m = rbm_model.model
    if isinstance(m, PeftModel):
        return m.get_base_model()
    return m


def setup_peft_model(rbm_model: RBM, cfg: PEFTConfig, target_module: Optional[str] = None) -> RBM:
    """Shared function to apply PEFT configuration to the model."""

    logger.info("Using PEFT/LoRA training...")

    # If PEFT was already attached (e.g. loaded from a checkpoint's adapter files in
    # setup_model_and_processor), don't re-wrap - just log trainable params and return.
    m = rbm_model.model
    already_peft = (
        isinstance(m, PeftModel)
        or (hasattr(m, "language_model") and isinstance(m.language_model, PeftModel))
        or (hasattr(m, "visual") and isinstance(m.visual, PeftModel))
    )
    if already_peft:
        logger.info("PEFT already attached (likely from checkpoint load); skipping re-wrap")
        trainable_params = sum(p.numel() for p in rbm_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in rbm_model.parameters())
        logger.info(
            f"AFTER PEFT (already attached): trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
        )
        return rbm_model

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
    )
    target_module = target_module or ("visual" if cfg.peft_vision_encoder else "language_model")
    inner = _get_vl_inner_model(rbm_model)
    if target_module == "model":
        logger.info("Attaching LoRA to the whole VL backbone...")
        rbm_model.model = get_peft_model(rbm_model.model, lora_config)
    elif target_module == "visual":
        logger.info("Attaching LoRA to the vision encoder...")
        inner.visual = get_peft_model(inner.visual, lora_config)
    else:
        logger.info("Attaching LoRA to the language model...")
        inner.language_model = get_peft_model(inner.language_model, lora_config)

    # Print all trainable parameters after PEFT so adapter weights (lora_A, lora_B, etc.) are visible
    logger.info("Trainable parameters (after PEFT):")
    for name, param in rbm_model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name:70} | {param.shape} | RG: {param.requires_grad}")
    trainable_params = sum(p.numel() for p in rbm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in rbm_model.parameters())
    logger.info(
        f"AFTER PEFT: trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return rbm_model


def create_training_arguments(cfg: TrainingConfig, output_dir: str, is_eval: bool = False) -> TrainingArguments:
    """Shared function to create TrainingArguments for both training and evaluation"""

    # Base arguments that are the same for both training and evaluation
    base_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
        "learning_rate": cfg.learning_rate,
        "save_strategy": cfg.save_strategy,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "remove_unused_columns": cfg.remove_unused_columns,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataloader_pin_memory": cfg.dataloader_pin_memory,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "dataloader_persistent_workers": cfg.dataloader_persistent_workers,
        "save_safetensors": True,
        "save_total_limit": 2,
        # Evaluation settings
        "eval_strategy": cfg.evaluation_strategy,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "do_eval": cfg.do_eval,
        "prediction_loss_only": cfg.prediction_loss_only,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_steps": cfg.warmup_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "max_grad_norm": cfg.max_grad_norm,
        "weight_decay": cfg.weight_decay,
        "disable_tqdm": False,
        # # Compile settings
        # "torch_compile": True,
        # "torch_compile_mode": "max-autotune",
        # "torch_compile_backend": "inductor",
    }

    # Add eval_steps if evaluation_strategy is "steps"
    if cfg.evaluation_strategy == "steps" and cfg.eval_steps is not None:
        base_args["eval_steps"] = cfg.eval_steps

    if is_eval:
        # Evaluation-specific arguments
        base_args.update({
            "per_device_eval_batch_size": 2,
            "num_train_epochs": -1,
            "max_steps": 1,
            "report_to": [],
        })
    else:
        # Training-specific arguments
        base_args.update({
            "num_train_epochs": cfg.num_train_epochs if cfg.num_train_epochs is not None else 1,
            "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
            # Disable HuggingFace's automatic logging - we use custom Logger class instead
            "report_to": [],
        })

    return TrainingArguments(**base_args)


def setup_dataset(cfg: DataConfig, is_eval: bool = False, sampler_kwargs=None, **kwargs) -> BaseDataset:
    """Shared function to create Dataset for training or evaluation

    Args:
        cfg: Data configuration
        is_eval: Whether this is for evaluation
        sampler_kwargs: Optional dictionary of keyword arguments to pass to samplers
        **kwargs: Additional keyword arguments to pass to dataset
    """
    dataset_cls = {
        "rbm": RBMDataset,
        "strategy_first": StrategyFirstDataset,
    }

    if cfg.dataset_type not in dataset_cls:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}. Must be one of: {list(dataset_cls.keys())}")

    if sampler_kwargs is None:
        sampler_kwargs = {}

    sampler_kwargs["random_seed"] = cfg.seed
    kwargs["sampler_kwargs"] = sampler_kwargs

    # Create the base dataset
    dataset = dataset_cls[cfg.dataset_type](config=cfg, is_evaluation=is_eval, **kwargs)

    if not is_eval:
        dataset = RepeatedDataset(dataset)
    return dataset


def setup_custom_eval_dataset(cfg: DataConfig, sampler_type: str, verbose=True, sampler_kwargs=None):
    """Setup a custom evaluation dataset with a specific sampler."""
    custom_eval_dataset = CustomEvalDataset(sampler_type, cfg, verbose=verbose, sampler_kwargs=sampler_kwargs)

    return custom_eval_dataset


def setup_batch_collator(
    processor: AutoProcessor, tokenizer: AutoTokenizer, cfg: ExperimentConfig, is_eval: bool = False
) -> BaseCollator:
    """Shared function to create BatchCollator"""
    collator_kwargs = {
        "processor": processor,
        "resized_height": cfg.data.resized_height,
        "resized_width": cfg.data.resized_width,
        "base_model_id": cfg.model.base_model_id,
        "use_multi_image": cfg.data.use_multi_image,
        "prog_pref": cfg.training.predict_pref_progress,
        "use_per_frame_progress_token": getattr(cfg.data, "use_per_frame_progress_token", False),
        "shuffle_progress_frames": cfg.data.shuffle_progress_frames,
        "inference": is_eval,
    }
    # Check for unsupported Molmo2 video mode
    if "Molmo" in cfg.model.base_model_id and not cfg.data.use_multi_image:
        raise ValueError(
            "Molmo2 implementation does not yet support video mode as it requires extra imports (use_multi_image=False). "
            "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
        )

    if "Qwen" in cfg.model.base_model_id or "SmolVLM" in cfg.model.base_model_id or "Molmo" in cfg.model.base_model_id:
        batch_collator = RBMBatchCollator(**collator_kwargs)
    # elif "rewind_transformer" in cfg.model.base_model_id:
    elif "rewind" in cfg.model.base_model_id:
        batch_collator = ReWiNDBatchCollator(
            **collator_kwargs, tokenizer=tokenizer, load_embeddings=cfg.data.load_embeddings
        )
    return batch_collator
