"""Batch-scores every episode in Robo-Dopamine's episode_cache/ (adityx23/icl-dataset,
pinned revision 9bb0c92eb3d9d06ea81096f5a7c4048485acc36b) with LeRobot's native RoboMeter
reward model (lerobot.rewards.robometer, checkpoint lerobot/Robometer-4B), and writes results
in the same two shapes the existing GRM (Robo-Dopamine) pipeline does: an all_episodes_summary.jsonl
for resumability/bookkeeping, and one JSON per episode under ../viewer/robometer_progress/
so the shared viewer can render it as another progress curve.

Single-process, single-GPU, model loaded once -- no HTTP server / subprocess-per-GPU
machinery like the GRM pipeline's, since that exists specifically to work around vLLM's
slow load time and a CUDA_VISIBLE_DEVICES clobber bug, neither of which apply to a plain
transformers `from_pretrained` model. Add multi-GPU sharding later (N copies, disjoint
--episodes slices) only if single-GPU throughput proves insufficient.

Reads frames from Robo-Dopamine's already-extracted PNG cache (see
../reward_models_common/episode_cache.py) rather than streaming via LeRobotDataset --
RoboMeter's actual scoring API only needs a frame tensor + task string, and lerobot>=0.6.1
can't open this v2.1-format dataset without a full v2.1->v3.0 conversion that the caching
approach sidesteps entirely. Per-episode progress values come from RoboMeter's own official
per-frame batching algorithm (lerobot.rewards.robometer.compute_rabc_weights, upstream's
"frame-steps" convention: for every sampled frame t, a linspace(0, t, K) context window is
scored, K=4 by default matching Robometer's own eval server) -- ported here rather than
called wholesale, since that script assumes a whole-dataset-loaded LeRobotDataset with
global frame offsets and holds all results in memory until a single end-of-run parquet
write; this driver needs per-episode incremental output for resumability instead.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_models_common.episode_cache import iter_episode_indices, load_frames, load_manifest  # noqa: E402

from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.rewards.robometer.compute_rabc_weights import _build_subsample_indices  # noqa: E402
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig  # noqa: E402
from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel  # noqa: E402
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep  # noqa: E402

REPO_ID = "adityx23/icl-dataset"
IMAGE_KEY = "observation.images.zed"  # front/scene camera ("cam_high")
REWARD_MODEL_PATH = "lerobot/Robometer-4B"
DEFAULT_NUM_SUBSAMPLED_FRAMES = 4  # matches upstream Robometer eval-server convention (K)
DEFAULT_BATCH_SIZE = 32

HERE = Path(__file__).resolve().parent
SUMMARY_PATH = HERE / "all_episodes_summary.jsonl"
LOG_PATH = HERE / "all_episodes_progress.log"
VIEWER_OUTPUT_DIR = HERE.parent / "viewer" / "robometer_progress" / REPO_ID


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_done_uids() -> set[str]:
    done = set()
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, encoding="utf-8") as f:
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


def score_episode(model, encoder, manifest: dict, batch_size: int, num_subsampled_frames: int, device: str):
    ep_frames = load_frames(manifest, IMAGE_KEY)
    num_frames = ep_frames.shape[0]
    task = manifest["task"]

    sub_indices = _build_subsample_indices(num_frames, num_subsampled_frames)
    progress_per_frame = [0.0] * num_frames

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        frames_batch = torch.stack([ep_frames[sub_indices[i]] for i in range(start, end)])

        transition = {
            TransitionKey.OBSERVATION: {IMAGE_KEY: frames_batch},
            TransitionKey.COMPLEMENTARY_DATA: {"task": task},
        }
        encoded = encoder(transition)
        obs = encoded[TransitionKey.OBSERVATION]
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in obs.items()
        }
        with torch.no_grad():
            rewards = model.compute_reward(batch)
        progress_per_frame[start:end] = rewards.cpu().tolist()

    return progress_per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-subsampled-frames", type=int, default=DEFAULT_NUM_SUBSAMPLED_FRAMES)
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Restrict to these episode indices.")
    parser.add_argument("--task-filter", type=str, nargs="+", default=None, help="AND-substring match on task (lowercased).")
    parser.add_argument("--num-shards", type=int, default=1, help="Run N parallel instances of this script over disjoint episode subsets (by episode_index % num_shards).")
    parser.add_argument("--shard-index", type=int, default=0, help="This instance's shard, in [0, num_shards).")
    args = parser.parse_args()

    VIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Loading RoboMeter ({REWARD_MODEL_PATH}) on {args.device} ...")
    config = RobometerConfig(pretrained_path=REWARD_MODEL_PATH, device=args.device)
    config.image_key = IMAGE_KEY
    model = RobometerRewardModel.from_pretrained(REWARD_MODEL_PATH, config=config)
    model.to(args.device).eval()

    encoder = RobometerEncoderProcessorStep(
        base_model_id=config.base_model_id,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=args.num_subsampled_frames,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
    )
    log("RoboMeter ready.")

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
            progress_per_frame = score_episode(
                model, encoder, manifest, args.batch_size, args.num_subsampled_frames, args.device
            )
            points = [
                {"frame_index": manifest["indices"][i], "progress": round(float(progress_per_frame[i]) * 100, 4)}
                for i in range(len(progress_per_frame))
            ]
            final_progress = points[-1]["progress"] if points else None

            viewer_json_path = VIEWER_OUTPUT_DIR / f"{uid.replace(':', '__')}.json"
            viewer_json_path.write_text(
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
                "reward_model": "robometer",
                "num_subsampled_frames": args.num_subsampled_frames,
                "batch_size": args.batch_size,
                "viewer_json_path": str(viewer_json_path),
                "final_progress": final_progress,
            }
            with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")

            log(f"[{idx}] done ({uid}): {len(points)} points, final_progress={final_progress}, {time.time()-t0:.1f}s")
        except Exception:
            log(f"[{idx}] FAILED ({uid}):\n{traceback.format_exc()}")
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    log("ALL EPISODES PROCESSED" if not args.episodes else "REQUESTED EPISODES PROCESSED")


if __name__ == "__main__":
    main()
