"""Query-side causal progress scoring for one icl-demo-dataset episode, using the fine-tuned
Robometer checkpoint (robometer_common.py) -- the Robometer analogue of causal_value.py.

Unlike causal_value.py's Robo-Dopamine pipeline (which needs an explicit before/after image pair
per step, averaged across 3 modes -- see causal_value.py's own docstring), Robometer gets a full
per-frame causal progress curve from a SINGLE forward pass over the whole (subsampled) episode --
see robometer_common.py's module docstring for why that's still causal, not a shortcut that leaks
the future. Consequence: this only produces MAX_FRAMES=32 points per episode (uniformly
subsampled from the episode's frame_interval=10 indices), sparser than causal_value.py's ~94
frame_interval=10 points -- a real granularity difference between the two methods' reports, not a
bug; the visualizer snaps to the nearest available Robometer point for whatever frame the scrubber
is on.

Must run in robometer_train's own .venv (has the `robometer` package + torch/transformers) --
separate from both lerobot_env_v21 (streaming) and robo_dopamine_env (GRM/vllm).
"""
import argparse
import json
from pathlib import Path

from robometer_common import (
    CHECKPOINT_PATH, MAX_FRAMES, ZERO_SHOT_CHECKPOINT_PATH, RobometerModel,
    load_frames_from_paths, subsample_frame_paths,
)


def compute_robometer_progress(manifest: dict, model: RobometerModel, episode_dir: Path,
                                max_frames: int = MAX_FRAMES) -> list[dict]:
    cam_dir = episode_dir / ".cache" / "cam_high"
    all_indices = manifest["indices"]
    paths = [cam_dir / f"frame_{idx:06d}.png" for idx in all_indices]
    paths, sampled_indices = subsample_frame_paths(paths, all_indices, max_frames)
    frames = load_frames_from_paths(paths)
    progress = model.score_frames(frames, manifest["task"], str(manifest["episode_index"]))
    n = min(len(sampled_indices), len(progress))
    return [{"frame": sampled_indices[i], "value": float(progress[i])} for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                         help="Defaults to <episode-dir>/robometer_progress.json "
                              "(or robometer_zeroshot_progress.json with --zero-shot)")
    parser.add_argument("--zero-shot", action="store_true",
                         help=f"Score with the base checkpoint ({ZERO_SHOT_CHECKPOINT_PATH}) instead of the "
                              "fine-tuned one -- same hindsight MAX_FRAMES method either way.")
    parser.add_argument("--model-path", default=None, help="Override the checkpoint path/Hub id directly.")
    args = parser.parse_args()

    model_path = args.model_path or (ZERO_SHOT_CHECKPOINT_PATH if args.zero_shot else CHECKPOINT_PATH)
    default_name = "robometer_zeroshot_progress.json" if args.zero_shot else "robometer_progress.json"

    manifest = json.loads((args.episode_dir / "manifest.json").read_text(encoding="utf-8"))
    model = RobometerModel(model_path=model_path)
    progress = compute_robometer_progress(manifest, model, args.episode_dir)

    out_path = args.out or (args.episode_dir / default_name)
    out_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"[robometer_query] wrote {out_path}")


if __name__ == "__main__":
    main()
