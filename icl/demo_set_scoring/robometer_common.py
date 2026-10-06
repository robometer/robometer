"""Shared model-loading + per-episode inference for the fine-tuned Robometer checkpoint, used by
both the reference-side batch (batch_robometer_reference.py, expanding
../viewer/robometer_finetuned_progress/ coverage) and the query-side batch
(robometer_query.py/batch_robometer_query.py, scoring icl-demo-dataset episodes).

Checkpoint: ../robometer_train/logs/ -- a LoRA adapter + custom progress/success/preference heads
on top of Qwen3-VL-4B-Instruct, fine-tuned from the base robometer/Robometer-4B checkpoint on
adityx23/icl-dataset (see ../robometer_train/outputs/2026-08-17/*/. hydra/overrides.yaml for the
exact training config). This is the "fine tuned" checkpoint the user asked for, as opposed to the
zero-shot Robometer-4B annotations already sitting in ../viewer/robometer_progress/.

Causality: Robometer is built on a decoder-only causal transformer (Qwen3VLModel). Frames are fed
as one token sequence in temporal order and progress is read out per-frame from
`hidden_state[trajectory_boundaries]` (see ../robometer_train/robometer/models/rbm.py) -- standard
causal self-attention means frame i's readout can only depend on frames 0..i, never later ones, in
a SINGLE forward pass over the whole (sampled) episode. No separate goal/future image needed,
unlike Robo-Dopamine's backward mode -- verified: no bidirectional attention override anywhere in
rbm.py, so this relies on the underlying Qwen backbone's default masking, not a special design
choice made for this pipeline.

Frame budget: this checkpoint was trained/preprocessed with max_frames=32 (see
../robometer_train/dataset_upload/configs/data_gen_configs/icl_dataset.yaml) -- feeding a full
~300-800 frame cached episode in one pass OOMs (confirmed by run_finetuned_inference.py's own
comment: 342 frames -> "Tried to allocate 3.84 GiB" on a 47GB card already in use). Both reference
and query sides here uniformly subsample to MAX_FRAMES from whatever set of already-extracted
frame indices they're given, same as run_finetuned_inference.py.

Must run in robometer_train's own .venv (has the `robometer` package + torch/transformers).
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROBOMETER_TRAIN_DIR = Path(__file__).resolve().parent.parent / "robometer_train"
sys.path.insert(0, str(ROBOMETER_TRAIN_DIR))

# transformers.modeling_utils unconditionally imports every quantizer backend it knows about
# (transformers/quantizers/auto.py), including torchao's -- even though we never quantize
# anything here (plain LoRA + dense heads on Qwen3-VL, no TorchAoConfig involved). torchao==0.18.0
# targets torch>=2.11 internally (its float8/mx_formats workflow modules import names like
# `torch.nn.functional.ScalingType`/`SwizzleType` that don't exist before 2.11); this venv is
# pinned to torch==2.8.0 (required elsewhere for Qwen/xformers), so torchao itself fails to import,
# which crashes transformers.quantizers.auto's unconditional `from .quantizer_torchao import
# TorchAoHfQuantizer` and takes the whole `from robometer... import ...` chain below down with it
# (peft -> transformers.models.bloom.modeling_bloom -> modeling_utils -> quantizers.auto ->
# quantizer_torchao -> torchao). Patching torchao's individual missing torch symbols is
# open-ended (mx_formats needs a different one than float8, and likely more beyond that) -- instead
# pre-register a stub for the one transformers module that actually imports torchao
# (transformers.quantizers.quantizer_torchao) so Python's import system uses the stub instead of
# ever executing the real (broken) file. AUTO_QUANTIZER_MAPPING just needs *a* class object at
# import time; it's never looked up unless a model is loaded with quantization_config=TorchAoConfig,
# which we never do.
_stub = types.ModuleType("transformers.quantizers.quantizer_torchao")
_stub.TorchAoHfQuantizer = type("TorchAoHfQuantizer", (), {})
sys.modules.setdefault("transformers.quantizers.quantizer_torchao", _stub)

from robometer.data.dataset_types import ProgressSample, Trajectory  # noqa: E402
from robometer.evals.eval_server import compute_batch_outputs  # noqa: E402
from robometer.utils.save import load_model_from_hf  # noqa: E402
from robometer.utils.setup_utils import setup_batch_collator  # noqa: E402

CHECKPOINT_PATH = str(ROBOMETER_TRAIN_DIR / "logs")
# Zero-shot base checkpoint, same Hub id FINETUNE_ROBOMETER.md's own worked example loads via
# training.load_from_checkpoint=robometer/Robometer-4B -- not lerobot/Robometer-4B, which is
# specifically the LeRobot integration's mirror for its own (non-PEFT-aware) RobometerRewardModel
# port and isn't what load_model_from_hf below expects.
ZERO_SHOT_CHECKPOINT_PATH = "robometer/Robometer-4B"
MAX_FRAMES = 32


class RobometerModel:
    def __init__(self, device: torch.device | None = None, model_path: str = CHECKPOINT_PATH):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        print(f"[robometer_common] loading checkpoint from {model_path} ...")
        self.exp_config, self.tokenizer, self.processor, self.reward_model = load_model_from_hf(
            model_path=model_path, device=self.device,
        )
        self.reward_model.eval()
        self.batch_collator = setup_batch_collator(self.processor, self.tokenizer, self.exp_config, is_eval=True)
        loss_config = getattr(self.exp_config, "loss", None)
        self.is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete" if loss_config else False
        )
        self.num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(self.exp_config.model, "progress_discrete_bins", 10)
        )
        print("[robometer_common] model loaded.")

    def score_frames(self, frames: np.ndarray, task: str, episode_id: str) -> np.ndarray:
        """frames: (T, H, W, C) uint8, already subsampled to <= MAX_FRAMES. Returns (T,) float32
        progress in [0, 1] (matches this project's value convention -- NOT rescaled to 0-100 like
        the raw viewer/*_progress/ JSON files use)."""
        T = int(frames.shape[0])
        traj = Trajectory(
            frames=frames, frames_shape=tuple(frames.shape), task=task, id=episode_id,
            metadata={"subsequence_length": T}, video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = self.batch_collator([progress_sample])
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)

        # autocast reconciles the zero-shot checkpoint's mixed bf16/fp32 weights (Unsloth keeps
        # LayerNorm in fp32 by default); the fine-tuned/PEFT checkpoint never needed this since
        # lora.Linear.forward casts its input to match uniformly (see robometer_query_online.py's
        # own note on the same issue).
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            results = compute_batch_outputs(
                self.reward_model, self.tokenizer, progress_inputs,
                sample_type="progress", is_discrete_mode=self.is_discrete, num_bins=self.num_bins,
            )
        progress_pred = results.get("progress_pred", [])
        return np.array(progress_pred[0], dtype=np.float32) if progress_pred else np.array([], dtype=np.float32)


def subsample_frame_paths(paths: list[Path], all_indices: list[int], max_frames: int = MAX_FRAMES) -> tuple[list[Path], list[int]]:
    """Uniformly subsample (paths, indices) together to at most max_frames -- same
    np.linspace(...).round() scheme as run_finetuned_inference.py's load_episode_frames, so
    reference and query sides pick frames the same way."""
    if len(paths) <= max_frames:
        return paths, list(all_indices)
    pick = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    return [paths[i] for i in pick], [all_indices[i] for i in pick]


def load_frames_from_paths(paths: list[Path]) -> np.ndarray:
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths])
