import gc
import json
import os
import re
import shutil
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate.state import AcceleratorState
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from robometer.configs.experiment_configs import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
)
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import loguru_logger as logger

from .upload_to_hub import upload_model_to_hub


def _get_rbm_peft_target(model: Any) -> tuple[str | None, PeftModel | None]:
    """Return the PEFT-wrapped RBM component, if any."""
    root = getattr(model, "model", None)
    if isinstance(root, PeftModel):
        return "model", root

    language_model = getattr(root, "language_model", None)
    if isinstance(language_model, PeftModel):
        return "language_model", language_model

    visual = getattr(root, "visual", None)
    if isinstance(visual, PeftModel):
        return "visual", visual

    return None, None


def _save_training_random_state(trainer: Trainer, ckpt_dir: str) -> None:
    """Save dataset random state when available."""
    if hasattr(trainer, "train_dataset"):
        try:
            train_dataset = trainer.train_dataset
            if hasattr(train_dataset, "dataset"):
                train_dataset = train_dataset.dataset

            if hasattr(train_dataset, "get_random_state"):
                random_state = train_dataset.get_random_state()
                random_state_file = os.path.join(ckpt_dir, "dataset_random_state.json")
                with open(random_state_file, "w") as f:
                    json.dump(random_state, f, indent=2)
                logger.info(f"Saved dataset random state to {random_state_file}")
        except Exception as e:
            logger.warning(f"Could not save random state: {e}")


def _save_trainer_checkpoint_files(
    trainer: Trainer,
    args: TrainingArguments,
    ckpt_dir: str,
    metrics: dict | None = None,
    step: int | None = None,
) -> None:
    """Save a trainer checkpoint, including PEFT adapters and RBM heads when present."""
    os.makedirs(ckpt_dir, exist_ok=True)

    model = trainer.model
    peft_target, peft_module = _get_rbm_peft_target(model)

    if peft_module is not None:
        logger.info(f"Detected PEFT target '{peft_target}' - saving full model snapshot and adapter metadata")
        trainer.save_model(ckpt_dir)
        peft_module.save_pretrained(ckpt_dir)

        peft_meta_path = os.path.join(ckpt_dir, "peft_target_module.json")
        with open(peft_meta_path, "w") as f:
            json.dump({"target_module": peft_target, "adapter_dir": "."}, f, indent=2)
        logger.info(f"Saved PEFT target metadata to {peft_meta_path}")

        rbm_state_dict = {}
        for name, param in model.named_parameters():
            if (
                "progress_head" in name
                or "preference_head" in name
                or "similarity_head" in name
                or "success_head" in name
            ):
                rbm_state_dict[name] = param.data.cpu()
            elif "frame_pool_attn" in name or "video_proj" in name or "text_proj" in name:
                rbm_state_dict[name] = param.data.cpu()

        for name, buffer in model.named_buffers():
            if (
                "progress_head" in name
                or "preference_head" in name
                or "similarity_head" in name
                or "success_head" in name
            ):
                rbm_state_dict[name] = buffer.cpu()

        if rbm_state_dict:
            from safetensors.torch import save_file

            custom_heads_path = os.path.join(ckpt_dir, "custom_heads.safetensors")
            save_file(rbm_state_dict, custom_heads_path)
            logger.info(f"Saved {len(rbm_state_dict)} custom head parameters to {custom_heads_path}")

        adapter_config_path = os.path.join(ckpt_dir, "adapter_config.json")
        adapter_model_paths = [
            os.path.join(ckpt_dir, "adapter_model.safetensors"),
            os.path.join(ckpt_dir, "adapter_model.bin"),
        ]

        if os.path.exists(adapter_config_path):
            logger.info(f"PEFT adapter config saved to {adapter_config_path}")
        else:
            logger.warning(f"PEFT adapter config not found at {adapter_config_path}")

        adapter_model_found = any(os.path.exists(p) for p in adapter_model_paths)
        if adapter_model_found:
            adapter_path = next(p for p in adapter_model_paths if os.path.exists(p))
            logger.info(f"PEFT adapter weights saved to {adapter_path}")
        else:
            logger.warning("PEFT adapter weights file not found - adapter weights may not have been saved correctly")
    else:
        logger.info("Detected non-PEFT model - using standard save_model()")
        trainer.save_model(ckpt_dir)

    if args.should_save:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_state()
        trainer_state_src = os.path.join(args.output_dir, "trainer_state.json")
        if os.path.exists(trainer_state_src):
            shutil.copy(trainer_state_src, ckpt_dir)

    if metrics is not None:
        metrics_file = os.path.join(ckpt_dir, "metrics.json")
        metrics_to_save = {
            "step": step,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in metrics.items()},
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"📊 Saved metrics to {metrics_file}")

    _save_training_random_state(trainer, ckpt_dir)


def save_final_checkpoint(
    trainer: Trainer, ckpt_dir: str, metrics: dict | None = None, step: int | None = None
) -> None:
    """Public helper for the final training save path."""
    _save_trainer_checkpoint_files(trainer, trainer.args, ckpt_dir, metrics=metrics, step=step)


def _apply_loaded_section_to_dataclass(instance: Any, loaded: dict, valid_names: set) -> None:
    """Set attributes on instance from loaded dict only for valid field names."""
    for key, value in loaded.items():
        if key in valid_names and value is not None:
            setattr(instance, key, value)


def update_cfg_with_pretrained_ckpt(
    cfg: ExperimentConfig,
    resume_from_checkpoint: str | None,
) -> None:
    """
    When resuming from a HuggingFace (or local) checkpoint, load its config.yaml
    and update: cfg.model (full replace), and only progress_loss_type and
    progress_discrete_bins on cfg.loss and cfg.data.
    """
    if not resume_from_checkpoint:
        return

    hub_token = os.environ.get("HF_TOKEN")
    is_hub = "/" in resume_from_checkpoint and not resume_from_checkpoint.startswith(("/", "./", "../"))
    config_path: str | None = None

    if is_hub:
        repo_id, revision = parse_hf_model_id_and_revision(resume_from_checkpoint, model_name="checkpoint")
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id, filename="config.yaml", revision=revision, token=hub_token
            )
            logger.info(f"Loaded checkpoint config from Hub: {repo_id}@{revision or 'latest'}")
        except Exception as e:
            logger.warning(f"Could not load config from checkpoint repo: {e}")
            return
    else:
        resolved = resolve_checkpoint_path(resume_from_checkpoint, hub_token=hub_token)
        if not resolved:
            return
        for candidate in [Path(resolved) / "config.yaml", Path(resolved).parent / "config.yaml"]:
            if candidate.is_file():
                config_path = str(candidate)
                logger.info(f"Loaded checkpoint config from local: {config_path}")
                break
        if not config_path:
            return

    with open(config_path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return

    # Replace the model config except use_peft; sync only the progress-loss and multi-image fields.
    model_names = {f.name for f in fields(ModelConfig)} - {"use_peft"}
    progress_loss_fields = {"progress_loss_type", "progress_discrete_bins"}
    loss_names = progress_loss_fields & {f.name for f in fields(LossConfig)}
    data_sync_fields = progress_loss_fields | {"use_multi_image", "use_per_frame_progress_token"}
    data_names = data_sync_fields & {f.name for f in fields(DataConfig)}

    model_loaded = loaded.get("model")
    loss_loaded = loaded.get("loss")
    data_loaded = loaded.get("data")

    if model_loaded and isinstance(model_loaded, dict):
        _apply_loaded_section_to_dataclass(cfg.model, model_loaded, model_names)
        logger.info("Updated model config from checkpoint (full replace)")
    if loss_loaded and isinstance(loss_loaded, dict) and loss_names:
        _apply_loaded_section_to_dataclass(cfg.loss, loss_loaded, loss_names)
    if data_loaded and isinstance(data_loaded, dict) and data_names:
        _apply_loaded_section_to_dataclass(cfg.data, data_loaded, data_names)
    if loss_names or data_names:
        logger.info(
            "Updated from checkpoint: progress_loss_type, progress_discrete_bins (loss); "
            "progress_loss_type, progress_discrete_bins, use_multi_image (data)"
        )

def resolve_checkpoint_path(checkpoint_path: str | None, hub_token: str | None = None) -> str | None:
    """
    Resolve checkpoint path, supporting local paths and HuggingFace Hub with @ notation.

    Args:
        checkpoint_path: Path to checkpoint. Can be:
            - None: No checkpoint to load
            - Local path: /path/to/checkpoint
            - HF repo: username/model-name (loads best tag automatically)
            - HF repo with tag: username/model-name@tag-name
        hub_token: Optional HuggingFace token for private repos

    Returns:
        Resolved local path to checkpoint, or None if no checkpoint
    """
    if not checkpoint_path:
        return None

    # If it's a local path, return as-is
    if checkpoint_path.startswith("/") or checkpoint_path.startswith("./") or checkpoint_path.startswith("../"):
        logger.info(f"Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    # Check if it looks like a HuggingFace repo (contains /)
    if "/" in checkpoint_path:
        repo_id, revision = parse_hf_model_id_and_revision(checkpoint_path, model_name="checkpoint")

        # Download from HuggingFace Hub
        logger.info(f"Downloading checkpoint from HuggingFace Hub: {repo_id}@{revision or 'latest'}")
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=hub_token,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "*.yaml"],
        )
        logger.info(f"Downloaded checkpoint to: {local_path}")
        return local_path

    # Otherwise, treat as local path
    logger.info(f"Using checkpoint: {checkpoint_path}")
    return checkpoint_path


def parse_hf_model_id_and_revision(hf_model_id: str, model_name: str = "model") -> tuple[str, str | None]:
    """
    Parse HuggingFace model ID and determine which revision (tag) to load.

    Supports explicit revisions via repo@revision format, or automatically
    finds the best tag if no explicit revision is provided.

    Args:
        hf_model_id: HuggingFace model repository ID or local path, optionally with @revision
        model_name: Name of the model type for logging (e.g., "ReWiND model", "Qwen model")

    Returns:
        Tuple of (repo_id, revision_to_load) where:
        - repo_id: The repository ID without the @revision suffix
        - revision_to_load: The revision/tag to load, or None for latest
    """
    # Allow users to specify explicit revisions via repo@revision
    if "@" in hf_model_id:
        repo_id, explicit_revision = hf_model_id.split("@", 1)
    else:
        repo_id, explicit_revision = hf_model_id, None

    revision_to_load = explicit_revision

    # Check if this is a HuggingFace repo (not a local path) and find best tag
    if "/" in repo_id and not repo_id.startswith("/"):
        if revision_to_load:
            logger.info(f"Loading {model_name} {repo_id} at explicit revision '{revision_to_load}'")
        else:
            best_tag, best_score = find_best_model_tag(repo_id)
            if best_tag:
                revision_to_load = best_tag
                logger.info(f"Loading {model_name} from best tag: {repo_id}@{revision_to_load} (score: {best_score})")
            else:
                logger.info(f"No best tag found, loading latest revision of {repo_id}")
    else:
        logger.info(f"Loading local/explicit {model_name} from {repo_id}")

    return repo_id, revision_to_load


def find_best_model_tag(hf_model_id: str, hub_token: str | None = None) -> tuple[str | None, float | None]:
    """
    Find the best model tag from HuggingFace Hub by parsing tag names and extracting scores.

    Expected tag format: "best-{metric_short}-{score:.4f}-step-{step}"
    Example: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"

    Args:
        hf_model_id: HuggingFace model ID (e.g., "aliangdw/rewind-debug")
        hub_token: Optional HuggingFace token for private repos

    Returns:
        tuple: (best_tag_name, best_score) or (None, None) if no valid tags found
    """
    try:
        api = HfApi(token=hub_token)

        # Check if repository exists
        if not api.repo_exists(repo_id=hf_model_id, repo_type="model"):
            logger.info(f"Repository {hf_model_id} does not exist")
            return None, None

        # Get all tags for the repository
        tags = api.list_repo_refs(repo_id=hf_model_id, repo_type="model").tags

        if not tags:
            logger.info(f"No tags found in repository {hf_model_id}")
            return None, None

        logger.info(f"Found {len(tags)} tags in {hf_model_id}: {[tag.name for tag in tags]}")

        best_tag = None
        best_score = float("-inf")

        # Parse each tag to extract score
        for tag in tags:
            tag_name = tag.name

            # Match our tag pattern: "best-{metric_short}-{score}-step-{step}"
            # Examples: "best-p-rank-spearman-mw-0.8500-step-123" or "best-avg-3metrics-0.7234-step-456"
            # Score can be positive or negative (e.g., 0.8500 or -1.2300)
            pattern = r"best-.*?-(-?\d+\.\d+)-step-\d+"
            match = re.search(pattern, tag_name)

            if match:
                try:
                    score = float(match.group(1))
                    logger.info(f"Parsed tag '{tag_name}': score = {score}")

                    if score > best_score:
                        best_score = score
                        best_tag = tag_name

                except ValueError:
                    logger.info(f"Could not parse score from tag '{tag_name}'")
                    continue
            else:
                logger.info(f"Tag '{tag_name}' does not match expected pattern")

        if best_tag:
            logger.info(f"Best tag found: '{best_tag}' with score {best_score}")
        else:
            logger.info("No valid tags found matching the expected pattern")

        return best_tag, best_score

    except Exception as e:
        logger.info(f"Error finding best tag for {hf_model_id}: {e}")
        return None, None


class SaveBestCallback(TrainerCallback):
    """
    Save a checkpoint whenever `metric_name` improves.
    Works in DDP/accelerate: only rank 0 writes checkpoints.
    Optionally keeps top-k best checkpoints and uploads to Hub.
    Also saves a 'latest' checkpoint at regular intervals.
    """

    def __init__(
        self,
        metric_names: list[str] | None = None,
        greater_is_better: list[bool] | None = None,
        keep_top_k: int = 1,
        save_every: int | None = None,
        upload_to_hub: bool = False,
        hub_save_every: int | None = None,
        hub_token: str | None = None,
        hub_private: bool = False,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()
        self.metric_names = metric_names or ["custom_eval/p_rank_spearman_mw"]
        self.greater_is_better = greater_is_better or [True]

        # Validate inputs
        if len(self.metric_names) != len(self.greater_is_better):
            raise ValueError(
                f"metric_names ({len(self.metric_names)}) and greater_is_better "
                f"({len(self.greater_is_better)}) must have the same length"
            )
        self.keep_top_k = keep_top_k
        self.save_every = save_every
        self.upload_to_hub = upload_to_hub
        self.hub_save_every = hub_save_every  # Frequency for Hub uploads (None = upload every checkpoint)
        self.hub_token = hub_token
        self.hub_private = hub_private
        self.base_model = base_model
        self._best_val = None
        self._saved: list[tuple[float, str]] = []  # list of (score, path), sorted from best -> worst
        self._uploaded: list[
            tuple[float, str, str]
        ] = []  # list of (score, tag_name, commit_id), sorted from best -> worst
        self._trainer = None  # Will be set when callback is registered
        self._last_save_step = -1  # Track last step where we saved 'latest'
        self._last_best_save_step = -1  # Track last step where we saved 'best'
        self._last_hub_upload_step = -1  # Track last step where we uploaded 'best' to Hub
        self._last_latest_hub_upload_step = -1  # Track last step where we uploaded 'latest' to Hub
        self._previous_latest_ckpt_dir = None  # Track previous 'latest' checkpoint directory
        self._previous_latest_hub_tag = None  # Track previous 'latest' Hub tag
        self._run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Static timestamp for this run

    def setup_trainer_reference(self, trainer: Trainer):
        """Set the trainer reference for later use in callbacks"""
        self._trainer = trainer

    def _compute_averaged_score(self, metrics: dict) -> tuple[float, list[str]]:
        """
        Compute averaged score from multiple metrics.

        Returns:
            tuple: (averaged_score, missing_metrics)
        """
        available_scores = []
        missing_metrics = []

        for metric_name, is_better in zip(self.metric_names, self.greater_is_better, strict=False):
            if metric_name in metrics:
                score = float(metrics[metric_name])
                # Normalize score: if lower is better, negate it so higher normalized score is better
                normalized_score = score if is_better else -score
                available_scores.append(normalized_score)
            else:
                missing_metrics.append(metric_name)

        if not available_scores:
            return float("-inf"), missing_metrics

        # Return average of normalized scores
        return np.mean(available_scores), missing_metrics

    def _is_main_process(self, trainer: Trainer) -> bool:
        try:
            return trainer.is_world_process_zero() and is_rank_0()
        except Exception:
            return (not AcceleratorState().distributed_type) or AcceleratorState().is_main_process

    def _build_metric_short_name(self) -> str:
        """Build a short metric name for checkpoint naming."""
        if len(self.metric_names) == 1:
            return self.metric_names[0].split("/")[-1]
        else:
            return f"avg-{len(self.metric_names)}metrics"

    def _build_metrics_detail_string(self, metrics: dict) -> str:
        """Build a detailed metrics string for logging."""
        metrics_detail = []
        for name in self.metric_names:
            if name in metrics:
                metrics_detail.append(f"{name}:{metrics[name]:.4f}")
        return " | ".join(metrics_detail) if metrics_detail else "no metrics"

    def _build_individual_scores_string(self, metrics: dict) -> str:
        """Build individual scores string for commit messages."""
        individual_scores = []
        for name in self.metric_names:
            if name in metrics:
                individual_scores.append(f"{name.split('/')[-1]}={metrics[name]:.4f}")
        return ", ".join(individual_scores) if individual_scores else "no metrics"

    def _get_hub_model_id(self, args: TrainingArguments) -> str:
        """Get the Hub model ID from output directory with timestamp."""
        base_name = args.output_dir.split("/")[-1].replace("_", "-")
        base_name = re.sub(r"-+", "-", base_name)
        base_name = base_name.strip("-")
        return f"rewardfm/{base_name}-{self._run_timestamp}"

    def _clean_tag_name(self, tag_name: str) -> str:
        """Clean tag name for HuggingFace repo naming requirements."""
        tag_name = tag_name.replace("_", "-").replace(",", "")
        tag_name = re.sub(r"-+", "-", tag_name)
        tag_name = tag_name.strip("-")
        return tag_name

    def _save_checkpoint_files(
        self,
        args: TrainingArguments,
        ckpt_dir: str,
        metrics: dict | None = None,
        step: int | None = None,
    ):
        """Save model, trainer state files, and metrics.

        For PEFT models, uses standard PeftModel.save_pretrained() to save adapter weights.
        Also saves custom heads (progress_head, etc.) which are part of the RBM wrapper.

        Note: This should only be called from rank 0 in the current implementation.
        """
        _save_trainer_checkpoint_files(self._trainer, args, ckpt_dir, metrics=metrics, step=step)

    def _cleanup_memory(self):
        """Perform memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _upload_checkpoint_to_hub(
        self, ckpt_dir: str, hub_model_id: str, tag_name: str, commit_message: str
    ) -> tuple[str, str]:
        """Upload checkpoint to Hub and return URL and commit ID."""
        hub_url, commit_id = upload_model_to_hub(
            model_dir=ckpt_dir,
            hub_model_id=hub_model_id,
            private=self.hub_private,
            token=self.hub_token,
            commit_message=commit_message,
            base_model=self.base_model,
            tag_name=tag_name,
        )
        return hub_url, commit_id

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Callback triggered after evaluation.
        Metrics are already gathered across all processes by the trainer before being passed here.
        """
        step = state.global_step

        # Only rank 0 needs to process metrics and save checkpoints
        if not self._is_main_process(self._trainer):
            logger.debug("Skipping checkpoint save (not main process)")
            return control

        logger.info(f"SaveBestCallback.on_evaluate called at step {step} with {len(metrics)} metrics")

        score, missing_metrics = self._compute_averaged_score(metrics)

        if missing_metrics:
            logger.warning(f"⚠️ Metrics {missing_metrics} not found in evaluation metrics")
            logger.warning(f"Available metrics: {metrics.keys()}")
            # If all metrics are missing, use a dummy score for filename but still save
            if score == float("-inf"):  # All metrics missing
                score_for_filename = 0.0  # Dummy value for filename
                logger.warning("⚠️ All metrics missing, using dummy score 0.0 in checkpoint filename")
            else:
                score_for_filename = score
        else:
            score_for_filename = score

        improved = (self._best_val is None) or (score > self._best_val)

        # Check if this score is worth saving (top-k logic)
        should_save = False
        if len(self._saved) < self.keep_top_k:
            # We haven't reached top-k yet, always save
            should_save = True
        else:
            # Check if this score beats the worst in our top-k
            worst_score = self._saved[-1][0]  # Last item is worst (sorted best -> worst)
            should_save = score > worst_score  # Always use > since we normalized scores

        if should_save and self._trainer:
            # Update overall best for reference (only if we have a valid score)
            if improved and score != float("-inf"):
                self._best_val = score

            # Make a descriptive dir name
            step = state.global_step
            metric_short = self._build_metric_short_name()
            tag = f"{metric_short}={score_for_filename:.4f}_step={step}"
            ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")

            metrics_str = self._build_metrics_detail_string(metrics)
            logger.info(
                f"💾 Saving ckpt: {ckpt_dir} | avg_score: {score_for_filename:.6f} | "
                f"{metrics_str} (rank {len(self._saved) + 1}/{self.keep_top_k})"
            )

            # Create checkpoint directory
            os.makedirs(ckpt_dir, exist_ok=True)

            # Save model, trainer state, and metrics
            self._save_checkpoint_files(args, ckpt_dir, metrics, step)
            self._cleanup_memory()

            # Track that we saved a best checkpoint at this step
            self._last_best_save_step = step

            # Add to saved list and sort (always best -> worst since we normalized scores)
            self._saved.append((score, ckpt_dir))
            self._saved.sort(key=lambda x: x[0], reverse=True)

            # Remove old checkpoint if we exceed keep_top_k
            if len(self._saved) > self.keep_top_k:
                _, path_to_rm = self._saved.pop(-1)
                logger.info(f"🗑️ Removing old checkpoint: {path_to_rm}")
                if os.path.isdir(path_to_rm):
                    shutil.rmtree(path_to_rm, ignore_errors=True)

            # Upload to Hub if enabled and frequency check passes
            should_upload_to_hub = False
            if self.upload_to_hub:
                if self.hub_save_every is None:
                    # Upload every checkpoint if no frequency is set
                    should_upload_to_hub = True
                else:
                    # Check if it's time to upload based on frequency
                    if self._last_hub_upload_step == -1:
                        # First upload
                        should_upload_to_hub = True
                    elif (step - self._last_hub_upload_step) >= self.hub_save_every:
                        # Enough steps have passed
                        should_upload_to_hub = True

            if should_upload_to_hub:
                hub_model_id = self._get_hub_model_id(args)
                tag_name = self._clean_tag_name(f"best-{metric_short}-{score_for_filename:.4f}-step-{step}")
                individual_scores_str = self._build_individual_scores_string(metrics)
                commit_message = (
                    f"Checkpoint: avg_score={score_for_filename:.4f} at step {step} | {individual_scores_str}"
                )

                logger.info(f"🚀 Uploading to Hub: {hub_model_id}")

                hub_url, commit_id = self._upload_checkpoint_to_hub(
                    ckpt_dir=ckpt_dir,
                    hub_model_id=hub_model_id,
                    tag_name=tag_name,
                    commit_message=commit_message,
                )
                logger.info(f"✅ Successfully uploaded to: {hub_url}")
                logger.info(f"🏷️ Tagged as: {tag_name}")

                # Track that we uploaded to Hub at this step
                self._last_hub_upload_step = step

                # Add to uploaded list and sort (always best -> worst since we normalized scores)
                self._uploaded.append((score, tag_name, commit_id))
                self._uploaded.sort(key=lambda x: x[0], reverse=True)

                # Remove old tags if we exceed keep_top_k
                api = HfApi(token=self.hub_token)
                if len(self._uploaded) > self.keep_top_k:
                    _, old_tag, _ = self._uploaded.pop(-1)
                    logger.info(f"🗑️ Removing old Hub tag: {old_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=old_tag)
                    logger.info(f"✅ Deleted tag: {old_tag}")

                # Aggressive memory cleanup after upload to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("🧹 Cleaned up memory after Hub upload")
            elif self.upload_to_hub and self.hub_save_every is not None:
                # Hub upload is enabled but not time yet
                steps_until_upload = self.hub_save_every - (step - self._last_hub_upload_step)
                logger.info(f"⏭️ Skipping Hub upload (saving locally only). Next upload in {steps_until_upload} steps")

        # Save 'latest' checkpoint if save_every is configured and it's time to save
        # Do this after processing best checkpoints so we have the gathered metrics
        # Skip if we just saved a best checkpoint at this step
        if self.save_every is not None and state.global_step > 0 and state.global_step % self.save_every == 0:
            if state.global_step != self._last_save_step and state.global_step != self._last_best_save_step:
                self._save_latest_checkpoint(args, state, metrics)
                self._last_save_step = state.global_step
            elif state.global_step == self._last_best_save_step:
                logger.info(
                    f"⏭️ Skipping 'latest' checkpoint save at step {state.global_step} (already saved as 'best')"
                )

        # Additional cleanup on all ranks after the entire on_evaluate callback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return control

    def _save_latest_checkpoint(self, args: TrainingArguments, state: TrainerState, metrics: dict):
        """Save a 'latest' checkpoint with metrics and step in the tag.
        Tracks and deletes the previous 'latest' checkpoint.

        Args:
            args: Training arguments
            state: Trainer state
            metrics: Evaluation metrics dictionary
        """
        if not self._trainer:
            return

        # Compute score and build tag similar to best checkpoints
        score, _missing_metrics = self._compute_averaged_score(metrics)
        step = state.global_step
        metric_short = self._build_metric_short_name()

        # Build tag with metrics and step
        tag = f"latest-{metric_short}={score:.4f}_step={step}"
        ckpt_dir = os.path.join(args.output_dir, f"ckpt-{tag}")

        metrics_str = self._build_metrics_detail_string(metrics)
        logger.info(f"💾 Saving 'latest' checkpoint at step {step} to {ckpt_dir} | {metrics_str}")

        # Remove old 'latest' checkpoint if it exists
        if self._previous_latest_ckpt_dir and os.path.isdir(self._previous_latest_ckpt_dir):
            logger.info(f"🗑️ Removing previous 'latest' checkpoint: {self._previous_latest_ckpt_dir}")
            shutil.rmtree(self._previous_latest_ckpt_dir, ignore_errors=True)

        # Save model, trainer state, and metrics
        self._save_checkpoint_files(args, ckpt_dir, metrics, step)
        logger.info(f"✅ Saved 'latest' checkpoint at step {step}")

        # Upload to Hub if enabled and frequency check passes
        should_upload_latest_to_hub = False
        if self.upload_to_hub:
            if self.hub_save_every is None:
                # Upload every checkpoint if no frequency is set
                should_upload_latest_to_hub = True
            else:
                # Check if it's time to upload based on frequency
                if self._last_latest_hub_upload_step == -1:
                    # First upload
                    should_upload_latest_to_hub = True
                elif (step - self._last_latest_hub_upload_step) >= self.hub_save_every:
                    # Enough steps have passed
                    should_upload_latest_to_hub = True

        if should_upload_latest_to_hub:
            hub_model_id = self._get_hub_model_id(args)
            tag_name = self._clean_tag_name(f"latest-{metric_short}-{score:.4f}-step-{step}")
            individual_scores_str = self._build_individual_scores_string(metrics)
            commit_message = f"Latest checkpoint: avg_score={score:.4f} at step {step} | {individual_scores_str}"

            # Delete previous 'latest' Hub tag if it exists
            api = HfApi(token=self.hub_token)
            if self._previous_latest_hub_tag:
                try:
                    logger.info(f"🗑️ Removing previous 'latest' Hub tag: {self._previous_latest_hub_tag}")
                    api.delete_tag(repo_id=hub_model_id, repo_type="model", tag=self._previous_latest_hub_tag)
                    logger.info(f"✅ Deleted previous tag: {self._previous_latest_hub_tag}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete previous Hub tag {self._previous_latest_hub_tag}: {e}")

            logger.info(f"🚀 Uploading 'latest' checkpoint to Hub: {hub_model_id}")

            hub_url, _commit_id = self._upload_checkpoint_to_hub(
                ckpt_dir=ckpt_dir,
                hub_model_id=hub_model_id,
                tag_name=tag_name,
                commit_message=commit_message,
            )
            logger.info(f"✅ Successfully uploaded 'latest' to: {hub_url}")
            logger.info(f"🏷️ Tagged as: {tag_name}")

            # Track this as the new previous latest
            self._previous_latest_hub_tag = tag_name
            self._last_latest_hub_upload_step = step
        elif self.upload_to_hub and self.hub_save_every is not None:
            # Hub upload is enabled but not time yet
            steps_until_upload = self.hub_save_every - (step - self._last_latest_hub_upload_step)
            logger.info(
                f"⏭️ Skipping Hub upload for 'latest' (saving locally only). Next upload in {steps_until_upload} steps"
            )

        # Track this as the new previous latest checkpoint directory
        self._previous_latest_ckpt_dir = ckpt_dir

        # Memory cleanup after saving model
        self._cleanup_memory()


def load_model_from_hf(
    model_path: str,
    device: torch.device,
    hub_token: str | None = None,
) -> tuple[ExperimentConfig | None, Any | None, Any | None, Any | None]:
    """
    Load reward model config and model from HuggingFace or local checkpoint.

    This mirrors the logic used by the training/eval scripts:
    - Resolve checkpoint path (supports HF Hub with @ notation)
    - Locate config.yaml locally (if model_path is a directory) or download from HF
    - Use custom YAML loader for ReWiND configs
    - Filter config keys to ExperimentConfig
    - Clear training/logging sections
    - Load model artifacts via setup_model_and_processor

    Args:
        model_path: HuggingFace model repository ID or local checkpoint path.
                   Supports @ notation for tags: username/model@tag-name
        device: Device to load model on
        hub_token: Optional HuggingFace token for private repos

    Returns:
        Tuple of (exp_config, tokenizer, processor, reward_model)
    """
    # Resolve checkpoint path (handles HF Hub downloads with @ notation)
    resolved_path = resolve_checkpoint_path(model_path, hub_token=hub_token)
    if resolved_path is None:
        raise ValueError(f"Could not resolve checkpoint path: {model_path}")

    config_path: str | None = None

    # Parse repo_id and revision (tag) from model_path if using @tag format
    # This is used for downloading config.yaml if needed
    if "@" in model_path:
        repo_id, revision = model_path.split("@", 1)
    else:
        repo_id, revision = model_path, None

    resolved_path = Path(resolved_path)

    if resolved_path.exists():
        # Local checkpoint: look for config.yaml
        candidate_paths = [
            resolved_path / "config.yaml",
            resolved_path.parent / "config.yaml",
        ]
        config_path = None
        for candidate in candidate_paths:
            if candidate.is_file():
                config_path = candidate
                break

        # If config.yaml not found locally, try to download it from Hub
        if config_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub not available. Install with: pip install huggingface_hub"
                ) from e

            # Check if this is a HuggingFace repo (not a local path)
            if "/" in repo_id and not repo_id.startswith("/"):
                logger.info(
                    f"config.yaml not found locally, downloading from HuggingFace Hub: {repo_id}@{revision or 'latest'}"
                )
                try:
                    config_path = hf_hub_download(
                        repo_id=repo_id, filename="config.yaml", revision=revision, token=hub_token
                    )
                    logger.info(f"Downloaded config.yaml to: {config_path}")
                except Exception as e:
                    logger.warning(f"Could not download config.yaml from Hub: {e}")
                    raise ValueError(
                        f"config.yaml not found locally and could not be downloaded from Hub: {e}"
                    ) from e
            else:
                raise ValueError(f"config.yaml not found in checkpoint directory or parent directory: {resolved_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub") from e
        # Download config with revision if specified
        logger.info(f"Downloading config.yaml from HuggingFace Hub: {repo_id}@{revision or 'latest'}")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision, token=hub_token)
        logger.info(f"Downloaded config.yaml to: {config_path}")

    with open(config_path) as f:
        yaml_text = f.read()

    class _ConfigSafeLoader(yaml.SafeLoader):
        pass

    _ConfigSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object:robometer.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    model_config_dict = yaml.load(yaml_text, Loader=_ConfigSafeLoader)

    valid_keys = {f.name for f in fields(ExperimentConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_keys}

    exp_config = ExperimentConfig(**filtered_config)
    # Use resolved_path for loading the actual model
    # Import here to avoid circular dependency with setup_utils
    from robometer.utils.setup_utils import setup_model_and_processor

    # Extract PEFT config from the loaded experiment config
    peft_config = exp_config.peft if hasattr(exp_config, "peft") and exp_config.model.use_peft else None

    tokenizer, processor, reward_model = setup_model_and_processor(
        exp_config.model, str(resolved_path), peft_config=peft_config
    )
    reward_model = reward_model.to(device)

    # Unsloth loads models with mixed precision (e.g. float32 token embeddings and the
    # vision patch_embed alongside bf16 linears) for training stability. That mismatch
    # breaks inference: float32 activations flow into bf16 weights and the forward
    # raises "expected scalar type BFloat16 but found Float" (and an analogous error in
    # the language-model attention). Since this loader returns an eval-only model, cast
    # every floating-point parameter to the configured dtype so the forward is
    # dtype-consistent. Integer parameters and buffers (e.g. rotary inv_freq, which RoPE
    # recomputes in float32) are intentionally left untouched.
    target_dtype = getattr(torch, exp_config.model.torch_dtype, torch.bfloat16)
    for param in reward_model.parameters():
        if param.is_floating_point() and param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)

    reward_model.eval()

    return exp_config, tokenizer, processor, reward_model


def load_wandb_run_info(model_path: str, hub_token: str | None = None) -> dict[str, Any] | None:
    """
    Retrieve saved wandb metadata for a checkpoint.

    Checks for a local `wandb_info.json` (written during training) and, if the
    checkpoint lives on HuggingFace, falls back to parsing the README that
    `upload_to_hub.py` generates (which embeds wandb fields).

    Args:
        model_path: HuggingFace model repository ID or local checkpoint path.
                   Supports @ notation for tags: username/model@tag-name
        hub_token: Optional HuggingFace token for private repos
    """

    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # Resolve checkpoint path first
    resolved_path = resolve_checkpoint_path(model_path, hub_token=hub_token)
    if resolved_path:
        path = Path(resolved_path)
        if path.exists():
            candidates = []
            if path.is_file():
                candidates.append(path.parent / "wandb_info.json")
            else:
                candidates.append(path / "wandb_info.json")
                candidates.append(path.parent / "wandb_info.json")
            for candidate in candidates:
                info = _load_json(candidate)
                if info:
                    return info

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    # Parse repo_id and revision (tag) from model_path if using @tag format
    if "@" in model_path:
        repo_id, revision = model_path.split("@", 1)
    else:
        repo_id, revision = model_path, None

    try:
        readme_path = hf_hub_download(
            repo_id=repo_id, filename="README.md", revision=revision, token=hub_token, local_files_only=False
        )
    except Exception:
        return None

    try:
        readme_text = Path(readme_path).read_text()
    except OSError:
        return None

    wandb_info: dict[str, Any] = {}

    run_match = re.search(r"\*\*Wandb Run\*\*:\s*\[(?P<name>.+?)\]\((?P<url>.+?)\)", readme_text)
    if run_match:
        wandb_info["wandb_name"] = run_match.group("name")
        wandb_info["wandb_url"] = run_match.group("url")

    id_match = re.search(r"\*\*Wandb ID\*\*:\s*`(?P<id>[^`]+)`", readme_text)
    if id_match:
        wandb_info["wandb_id"] = id_match.group("id")

    project_match = re.search(r"\*\*Project\*\*:\s*(?P<project>[^\n]+)", readme_text)
    if project_match:
        wandb_info["wandb_project"] = project_match.group("project").strip()

    entity_match = re.search(r"\*\*Entity\*\*:\s*(?P<entity>[^\n]+)", readme_text)
    if entity_match:
        wandb_info["wandb_entity"] = entity_match.group("entity").strip()

    return wandb_info or None
