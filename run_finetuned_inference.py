"""Runs the just-trained LoRA-adapted RoboMeter checkpoint (./logs/, see FINETUNE_ROBOMETER.md
and dataset_upload/dataset_loaders/icl_loader.py's own docstring for how it was built and
trained -- full icl-dataset, ~3044 episodes, sparse per-episode success/fail labels) over the
same 16 one-shot episodes used to evaluate Robo-Dopamine/ProcVLM's fine-tuned checkpoints (see
../viewer/robo_dopamine_finetuned_progress.py and ../ProcVLM/run_all_episodes_finetuned.py),
for viewer-toggle consistency across all three methods.

Reuses scripts/example_inference_local.py's own compute_rewards_per_frame_local() UNMODIFIED --
only the frame source (Robo-Dopamine's already-extracted PNG cache via
reward_models_common/episode_cache.py, same convention as icl_loader.py's IclFrameLoader,
instead of decoding a video file) and the *set* of episodes processed differ from that script's
single-video CLI design.

Output shape matches the other two methods' fine-tuned drivers: {"points": [{"frame_index",
"progress"}]} JSON keyed by episode_uid, under ../viewer/robometer_finetuned_progress/.

Usage: python run_finetuned_inference.py --episodes 109 271 652 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ICL_ANNOTATIONS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ICL_ANNOTATIONS_DIR))
from reward_models_common.episode_cache import load_frame_paths, load_manifest  # noqa: E402

# compute_rewards_per_frame_local's own model-loading call (load_model_from_hf) is reused
# unmodified below, just hoisted out of a per-episode loop instead of calling that whole
# function per episode -- it reloads the ~4B checkpoint from scratch every call, which is fine
# for its own single-video CLI design but would mean loading it 16 times here.
from robometer.data.dataset_types import ProgressSample, Trajectory  # noqa: E402
from robometer.evals.eval_server import compute_batch_outputs  # noqa: E402
from robometer.utils.save import load_model_from_hf  # noqa: E402
from robometer.utils.setup_utils import setup_batch_collator  # noqa: E402

CHECKPOINT_PATH = str(Path(__file__).resolve().parent / "logs")
OUTPUT_DIR = ICL_ANNOTATIONS_DIR / "viewer" / "robometer_finetuned_progress" / "adityx23" / "icl-dataset"


def _output_stem(manifest: dict) -> str:
    uid = manifest.get("episode_uid")
    return uid.replace(":", "__") if uid else f"episode_{manifest['episode_index']}"


MAX_FRAMES = 32  # matches dataset_upload/configs/data_gen_configs/icl_dataset.yaml's own
# max_frames=32 (what this checkpoint was actually trained/preprocessed on) -- feeding it a
# full ~300-800-frame cached episode in one forward pass OOMs a 47GB card outright (confirmed:
# 342 frames -> "Tried to allocate 3.84 GiB" with 46.6GB already in use). The zero-shot
# ../robometer/run_all_episodes.py pipeline gets a dense per-frame curve via LeRobot's own
# RobometerRewardModel.compute_reward() called once per target frame with a small K=4 context
# window -- a different model wrapper (LeRobot's ported class, not this training repo's
# load_model_from_hf/compute_batch_outputs) that this checkpoint doesn't use, so that approach
# doesn't carry over directly. Uniform subsampling to MAX_FRAMES trades curve density for a
# single bounded call instead of porting that windowing algorithm to a different API.


def load_episode_frames(manifest: dict, max_frames: int = MAX_FRAMES) -> tuple[np.ndarray, list[int]]:
    paths = load_frame_paths(manifest, cam_name="cam_high")
    all_indices = manifest["indices"]
    if len(paths) > max_frames:
        pick = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in pick]
        sampled_indices = [all_indices[i] for i in pick]
    else:
        sampled_indices = list(all_indices)
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
    return np.stack(frames), sampled_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading fine-tuned checkpoint from {CHECKPOINT_PATH} ...")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(model_path=CHECKPOINT_PATH, device=device)
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete" if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    print("Model loaded.")

    for episode_index in args.episodes:
        manifest = load_manifest(episode_index)
        uid = manifest.get("episode_uid")
        print(f"[{episode_index}] ({uid}) loading frames...")
        frames, sampled_indices = load_episode_frames(manifest)

        print(f"[{episode_index}] running inference on {frames.shape[0]} frames...")
        T = int(frames.shape[0])
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=manifest["task"],
            id=str(episode_index),
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = batch_collator([progress_sample])
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(device)

        results = compute_batch_outputs(
            reward_model, tokenizer, progress_inputs,
            sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins,
        )
        progress_pred = results.get("progress_pred", [])
        progress = np.array(progress_pred[0], dtype=np.float32) if progress_pred else np.array([], dtype=np.float32)

        n = min(len(sampled_indices), len(progress))
        points = [
            {"frame_index": sampled_indices[i], "progress": round(float(progress[i]) * 100, 2)}
            for i in range(n)
        ]

        out_path = OUTPUT_DIR / f"{_output_stem(manifest)}.json"
        out_path.write_text(json.dumps({
            "episode_uid": uid,
            "episode_index": episode_index,
            "task": manifest["task"],
            "success": manifest["success"],
            "points": points,
        }, indent=2), encoding="utf-8")
        print(f"[{episode_index}] -> {len(points)} points -> {out_path}")


if __name__ == "__main__":
    main()
