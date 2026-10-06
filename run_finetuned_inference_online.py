"""Online-rollout-style scoring of the finetuned RoboMeter checkpoint (./logs/) over
adityx23/icl-dataset -- a causal counterpart to run_finetuned_inference.py.

run_finetuned_inference.py subsamples each episode's FULL cached frame count into MAX_FRAMES=32
anchor frames chosen with foreknowledge of the episode's eventual length (np.linspace(0,
len(paths)-1, 32)), then scores all 32 in one non-causal forward pass. That's a legitimate
offline "how good is this checkpoint at reading a whole finished trajectory" check -- it matches
upstream Robometer's own scripts/example_inference_local.py, which this same clone of
github.com/robometer/robometer ships for exactly that purpose -- but it is NOT what the model
would have output if queried step-by-step during an actual rollout, because picking those 32
anchors requires already knowing how the episode ends.

Upstream's own answer to "how does Robometer handle an online rollout" lives elsewhere in this
same repo: scripts/example_libero_robometer_wrapper.py's LiberoRobometerRewardWrapper, a
gym.Wrapper that appends one observation per env.step() to a per-key deque(maxlen=max_frames)
and rescoring that deque -- and only that deque, never anything not yet observed -- on every
step. The causal windowing it ultimately runs through is robometer.evals.eval_server
.process_batch_helper's own hardcoded "frame-steps" convention (NUM_SUBSAMPLED_FRAMES=4): for
frame t, linspace(0, t, 4) subsamples *only frames [0, t]*.

This script reimplements that same linspace(0, t, 4) causal windowing directly (matching
lerobot.rewards.robometer.compute_rabc_weights._build_subsample_indices, the LeRobot-side port of
the identical upstream convention already used by run_all_episodes.py, the zero-shot
lerobot/Robometer-4B pipeline in ../robometer/) rather than calling process_batch_helper(...,
use_frame_steps=True) itself -- that flag re-expands and rescoring frames [0, t] from scratch for
every t inside ONE trajectory, which is quadratic (recomputes t=0..N-2 again every time you ask
for a longer prefix) and would OOM exactly the way a single non-chunked whole-episode forward
pass already does (run_finetuned_inference.py's own docstring: 342 frames in one forward pass ->
OOM on a 47GB card). Precomputing linspace(0, t, 4) once per t and batching BATCH_SIZE independent
already-4-frame windows per forward call (mirroring run_all_episodes.py's score_episode) keeps
work linear in the number of frames and each batch item's size fixed at K=4, regardless of how
far into the episode t is.

No frame after t is ever included in frame t's window -- verified directly from the indices
themselves (np.linspace(0, t, 4) can't produce anything > t) rather than relying on the model's
attention mask, so this is causal independent of whether the backbone's own masking is airtight.

Frame source is the same Robo-Dopamine PNG cache as run_finetuned_inference.py
(../reward_models_common/episode_cache.py) -- but unlike that script, every raw cached frame is
scored (no upfront subsampling to a fixed anchor count), since online rollout has no fixed
budget: at each real step you either have a new frame to score or you don't.

Output: same shape as run_finetuned_inference.py ({"points": [{"frame_index", "progress"}],
"episode_uid", "episode_index", "task", "success"}), written to the same directory
(../viewer/robometer_finetuned_progress/adityx23/icl-dataset/) with an "online_" filename prefix
so both curves are viewable side by side. Resumable via ONLINE_SUMMARY_PATH, same convention as
../robometer/run_all_episodes.py's all_episodes_summary.jsonl.

Usage: uv run python run_finetuned_inference_online.py --episodes 109 271 652 ...
       uv run python run_finetuned_inference_online.py --num-shards 4 --shard-index 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ICL_ANNOTATIONS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ICL_ANNOTATIONS_DIR))
from reward_models_common.episode_cache import iter_episode_indices, load_frame_paths, load_manifest  # noqa: E402

from robometer.data.dataset_types import ProgressSample, Trajectory  # noqa: E402
from robometer.evals.eval_server import compute_batch_outputs  # noqa: E402
from robometer.utils.save import load_model_from_hf  # noqa: E402
from robometer.utils.setup_utils import setup_batch_collator  # noqa: E402

CHECKPOINT_PATH = str(Path(__file__).resolve().parent / "logs")
OUTPUT_DIR = ICL_ANNOTATIONS_DIR / "viewer" / "robometer_finetuned_progress" / "adityx23" / "icl-dataset"
OUTPUT_PREFIX = "online_"

# Matches process_batch_helper's own hardcoded NUM_SUBSAMPLED_FRAMES (robometer/evals/eval_server.py)
# and lerobot.rewards.robometer.compute_rabc_weights.DEFAULT_NUM_SUBSAMPLED_FRAMES -- upstream's one
# fixed convention for causal per-frame scoring, reused here rather than re-derived.
NUM_SUBSAMPLED_FRAMES = 4
DEFAULT_BATCH_SIZE = 32  # independent K=4-frame windows scored per forward call

HERE = Path(__file__).resolve().parent
ONLINE_SUMMARY_PATH = HERE / "all_episodes_online_summary.jsonl"
ONLINE_LOG_PATH = HERE / "all_episodes_online_progress.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ONLINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _output_stem(manifest: dict) -> str:
    uid = manifest.get("episode_uid")
    return uid.replace(":", "__") if uid else f"episode_{manifest['episode_index']}"


def load_done_uids() -> set[str]:
    done = set()
    if ONLINE_SUMMARY_PATH.exists():
        with open(ONLINE_SUMMARY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                uid = r.get("episode_uid")
                if uid:
                    done.add(uid)
    return done


def load_episode_frames(manifest: dict) -> np.ndarray:
    """All cached cam_high frames for one episode, in chronological order, unsubsampled --
    every raw cached frame becomes one online-rollout "step" below. No foreknowledge of the
    episode's eventual length is used (contrast run_finetuned_inference.py's
    load_episode_frames, which needs len(paths) up front to place its 32 anchors)."""
    paths = load_frame_paths(manifest, cam_name="cam_high")
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
    return np.stack(frames)


def _causal_subsample_indices(num_frames: int, k: int = NUM_SUBSAMPLED_FRAMES) -> list[np.ndarray]:
    """indices[t] = k frame indices drawn from [0, t] via linspace -- upstream's frame-steps
    convention. linspace(0, t, k) cannot produce any index > t, so this is causal by
    construction: frame t's window is a function of frames [0, t] only, never of anything past
    it. Matches lerobot.rewards.robometer.compute_rabc_weights._build_subsample_indices exactly
    (round-then-int, not truncate) so results are directly comparable to run_all_episodes.py's
    zero-shot curves."""
    return [np.linspace(0, t, k).round().astype(np.int64) for t in range(num_frames)]


def score_episode_online(
    reward_model,
    tokenizer,
    batch_collator,
    frames: np.ndarray,
    task: str,
    device: str,
    is_discrete: bool,
    num_bins: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    """Causal per-frame progress: frame t's value depends only on frames[0:t+1].

    Precomputes the linspace(0, t, K) window for every t once (O(num_frames) total), then scores
    `batch_size` independent, already-K-frame-sized windows per forward call -- same shape and
    batching pattern as ../robometer/run_all_episodes.py's score_episode, just retargeted at this
    finetuned checkpoint's own PEFT-aware model/collator (robometer.utils.save.load_model_from_hf)
    instead of LeRobot's RobometerRewardModel port, which can't load a LoRA checkpoint like this
    one (sharded safetensors, PEFT-wrapped key names, heads in a separate custom_heads.safetensors
    -- see FINETUNE_ROBOMETER.md).
    """
    num_frames = frames.shape[0]
    sub_indices = _causal_subsample_indices(num_frames)
    progress_per_frame = [0.0] * num_frames

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        samples = []
        for t in range(start, end):
            window = frames[sub_indices[t]]  # (K, H, W, C) -- only frames <= t
            traj = Trajectory(
                frames=window,
                frames_shape=tuple(window.shape),
                task=task,
                id=str(t),
                metadata={"subsequence_length": window.shape[0]},
                video_embeddings=None,
            )
            samples.append(ProgressSample(trajectory=traj, sample_type="progress"))

        batch = batch_collator(samples)
        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(device)

        results = compute_batch_outputs(
            reward_model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )
        progress_pred = results.get("progress_pred", [])
        for offset, seq in enumerate(progress_pred):
            # Each window's own last frame IS frame t (linspace always includes the endpoint) --
            # same convention as run_all_episodes.py's zero-shot curves.
            progress_per_frame[start + offset] = float(seq[-1]) if seq else 0.0

    return progress_per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Restrict to these episode indices.")
    parser.add_argument("--task-filter", type=str, nargs="+", default=None, help="AND-substring match on task (lowercased).")
    parser.add_argument("--num-shards", type=int, default=1, help="Run N parallel instances over disjoint episode subsets (episode_index %% num_shards).")
    parser.add_argument("--shard-index", type=int, default=0, help="This instance's shard, in [0, num_shards).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log(f"Loading finetuned RoboMeter checkpoint from {CHECKPOINT_PATH} ...")
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
    log("Model loaded.")

    all_indices = args.episodes if args.episodes else iter_episode_indices()
    if args.num_shards > 1:
        all_indices = [i for i in all_indices if i % args.num_shards == args.shard_index]
    done = load_done_uids()
    log(f"{len(all_indices)} episode(s) available (shard {args.shard_index}/{args.num_shards}); {len(done)} already done overall.")

    for idx in all_indices:
        manifest = load_manifest(idx)
        uid = manifest["episode_uid"]
        if uid in done:
            continue
        if args.task_filter and not all(s.lower() in manifest["task"].lower() for s in args.task_filter):
            continue

        try:
            t0 = time.time()
            frames = load_episode_frames(manifest)
            progress_per_frame = score_episode_online(
                reward_model, tokenizer, batch_collator, frames, manifest["task"],
                device, is_discrete, num_bins, batch_size=args.batch_size,
            )
            points = [
                {"frame_index": manifest["indices"][i], "progress": round(float(progress_per_frame[i]) * 100, 2)}
                for i in range(len(progress_per_frame))
            ]
            final_progress = points[-1]["progress"] if points else None

            out_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}{_output_stem(manifest)}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "episode_uid": uid,
                        "episode_index": idx,
                        "task": manifest["task"],
                        "success": manifest["success"],
                        "final_progress": final_progress,
                        "points": points,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = {
                "episode_index": idx,
                "episode_uid": uid,
                "task": manifest["task"],
                "dataset_success": manifest["success"],
                "length": manifest["num_frames"],
                "reward_model": "robometer_finetuned_online",
                "num_subsampled_frames": NUM_SUBSAMPLED_FRAMES,
                "batch_size": args.batch_size,
                "viewer_json_path": str(out_path),
                "final_progress": final_progress,
            }
            with open(ONLINE_SUMMARY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")

            log(f"[{idx}] done ({uid}): {len(points)} points, final_progress={final_progress}, {time.time()-t0:.1f}s")
        except Exception:
            log(f"[{idx}] FAILED ({uid}):\n{traceback.format_exc()}")
        finally:
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    log("ALL EPISODES PROCESSED" if not args.episodes else "REQUESTED EPISODES PROCESSED")


if __name__ == "__main__":
    main()
