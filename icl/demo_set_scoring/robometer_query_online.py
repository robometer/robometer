"""Causal (online-rollout-style) counterpart to robometer_query.py, for one icl-demo-dataset
query episode, using the fine-tuned Robometer checkpoint (robometer_common.py).

robometer_query.py (via robometer_common.py's subsample_frame_paths) loads the WHOLE cached
episode up front and subsamples its FULL frame count into MAX_FRAMES=32 anchors via
np.linspace(0, len(paths)-1, 32) -- picking those anchors requires knowing len(paths), i.e. how
the episode ends, before scoring frame 0. That's a real difference from an online rollout, not
just an attention-masking detail: robometer_common.py's own docstring is right that the per-frame
READOUT is causal (frame i's hidden state can't attend past frame i -- standard Qwen3VLModel
decoder masking, verified: no bidirectional attention override anywhere in
robometer_train/robometer/models/rbm.py), but the pipeline around it still hands the model
foreknowledge of the episode's total length via the anchor-placement step.

This script instead reimplements the actual mechanics of
robometer_train/scripts/example_libero_robometer_wrapper.py's LiberoRobometerRewardWrapper --
upstream's own online-rollout reward wrapper for live gym rollouts -- rather than the
K=4-frame-steps convention used elsewhere in this project for retroactive per-frame curves over
a completed trajectory (run_all_episodes.py, run_finetuned_inference_online.py). That wrapper:

  - keeps one `deque(maxlen=max_frames)` per env, appending exactly the frame observed at the
    current env.step() -- max_frames comes from the checkpoint's own data.max_frames (16 for
    this checkpoint; see ../robometer_train/logs/config.yaml), not from any project-wide K
    constant;
  - at every step, converts the deque's *current* contents (never anything not yet observed) to
    a ProgressSample via robometer.evals.eval_utils.raw_dict_to_sample, which internally calls
    linspace_subsample_frames (a no-op pass-through here, since the deque can never exceed
    max_frames) then pad_trajectory_to_max_frames_np(..., pad_from="right") -- which pads a
    not-yet-full window by repeating the MOST RECENT frame, never a future one -- to keep every
    step's model input a fixed max_frames-length tensor;
  - scores it and reads off the last position's prediction as that step's reward.

This script drives that identical per-step construction from a pre-recorded episode instead of
a live env: it iterates the episode's cached frames one at a time, in order, appending each to
its own deque(maxlen=max_frames) exactly as the wrapper's env.step() does, and never reads the
episode's total frame count for anything other than the loop itself stopping when the recording
does (the same way a live rollout stops at `terminated`/`truncated`, not because the reward
computation was told in advance how long the episode would run). No frame after step t is ever
loaded, appended, or referenced when computing step t's value.

Multiple steps' already-built (already-padded) samples are batched together only for GPU
throughput (see BATCH_SIZE) -- each sample in a batch was built from its own step's deque
contents, so batching changes nothing about which frames contributed to which step's score; it
only changes how many independent, already-fully-determined forward passes run per GPU call.

Output: same per-frame list shape as robometer_query.py ([{"frame", "value"}, ...], value in
[0, 1]), written to <episode-dir>/online_robometer_progress.json -- prefixed, alongside (not
overwriting) the existing robometer_progress.json.

Must run in robometer_train's own .venv.
Usage: uv run python robometer_query_online.py --episode-dir query_episode_cache/episode_000011
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robometer_common import CHECKPOINT_PATH, ZERO_SHOT_CHECKPOINT_PATH, RobometerModel
from robometer.data.dataset_types import ProgressSample  # noqa: E402
from robometer.evals.eval_server import compute_batch_outputs  # noqa: E402
from robometer.evals.eval_utils import raw_dict_to_sample  # noqa: E402

DEFAULT_BATCH_SIZE = 32  # independent, already-built per-step samples scored per forward call


def _resolve_max_frames(model: RobometerModel, default: int = 16) -> int:
    """Same resolution order as LiberoRobometerRewardWrapper's _RewardModelInferenceMixin:
    the checkpoint's own data.max_frames, falling back to 16."""
    data_cfg = getattr(model.exp_config, "data", None)
    return int(getattr(data_cfg, "max_frames", default)) if data_cfg is not None else default


def compute_robometer_progress_online(
    manifest: dict, model: RobometerModel, episode_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    cam_dir = episode_dir / ".cache" / "cam_high"
    all_indices = manifest["indices"]
    task = manifest["task"]
    episode_id = str(manifest["episode_index"])
    max_frames = _resolve_max_frames(model)

    # Sliding buffer of raw frames "observed so far" -- the exact structure
    # LiberoRobometerRewardWrapper keeps per env (self._frames[key]), just fed from a pre-recorded
    # PNG cache one frame per loop iteration instead of one frame per env.step().
    window: deque = deque(maxlen=max_frames)

    progress_per_frame: list[float] = [0.0] * len(all_indices)
    pending_samples: list[ProgressSample] = []
    pending_positions: list[int] = []

    def flush() -> None:
        if not pending_samples:
            return
        batch = model.batch_collator(list(pending_samples))
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(model.device)
        # The zero-shot checkpoint keeps some weights (LayerNorm) in float32 -- Unsloth's
        # standard mixed-precision setup -- while the rest of the backbone is bfloat16, and
        # plain fp32-vs-bf16 tensor ops refuse to mix (RuntimeError: expected scalar type
        # BFloat16 but found Float). autocast is what actually reconciles that internally
        # (computing in bf16, promoting to fp32 only where a given op needs it); the
        # fine-tuned/PEFT checkpoint never hit this because LoRA's own lora.Linear.forward casts
        # its input to match, uniformly, so a plain float32-vs-bf16 op never occurred there.
        with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
            results = compute_batch_outputs(
                model.reward_model, model.tokenizer, progress_inputs,
                sample_type="progress", is_discrete_mode=model.is_discrete, num_bins=model.num_bins,
            )
        preds = results.get("progress_pred", [])
        for pos, seq in zip(pending_positions, preds):
            progress_per_frame[pos] = float(seq[-1]) if seq else 0.0
        pending_samples.clear()
        pending_positions.clear()

    for t, idx in enumerate(all_indices):
        # "Receive" frame t -- exactly what a live rollout would hand the wrapper at this
        # env.step(). Frames at any index > t do not exist yet as far as this loop is concerned.
        frame_path = cam_dir / f"frame_{idx:06d}.png"
        window.append(np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8))

        raw = dict(
            frames=np.stack(list(window), axis=0),
            task=task,
            id=episode_id,
            metadata=dict(subsequence_length=len(window)),
            video_embeddings=None,
            text_embedding=None,
        )
        pending_samples.append(raw_dict_to_sample(raw_data=raw, max_frames=max_frames, sample_type="progress"))
        pending_positions.append(t)

        if len(pending_samples) >= batch_size:
            flush()

    flush()
    return [{"frame": all_indices[i], "value": progress_per_frame[i]} for i in range(len(all_indices))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                         help="Defaults to <episode-dir>/online_robometer_progress.json "
                              "(or online_robometer_zeroshot_progress.json with --zero-shot)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--zero-shot", action="store_true",
                         help=f"Score with the base checkpoint ({ZERO_SHOT_CHECKPOINT_PATH}) instead of the "
                              "fine-tuned one -- same causal deque method either way.")
    parser.add_argument("--model-path", default=None, help="Override the checkpoint path/Hub id directly.")
    args = parser.parse_args()

    model_path = args.model_path or (ZERO_SHOT_CHECKPOINT_PATH if args.zero_shot else CHECKPOINT_PATH)
    default_name = "online_robometer_zeroshot_progress.json" if args.zero_shot else "online_robometer_progress.json"

    manifest = json.loads((args.episode_dir / "manifest.json").read_text(encoding="utf-8"))
    model = RobometerModel(model_path=model_path)
    progress = compute_robometer_progress_online(manifest, model, args.episode_dir, args.batch_size)

    out_path = args.out or (args.episode_dir / default_name)
    out_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"[robometer_query_online] wrote {out_path}")


if __name__ == "__main__":
    main()
