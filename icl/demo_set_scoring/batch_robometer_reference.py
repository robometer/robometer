"""Expands ../viewer/robometer_finetuned_progress/adityx23/icl-dataset/ coverage from the original
16 one-shot-eval episodes to every icl-dataset episode of the 15 tasks reward_guided_retrieval's
value-guided pipeline covers (see Robo-Dopamine/dataset/task_episode_indices/<task>.txt for the
episode lists) -- ~1613 episodes total. Needed for a real reference pool to match the fine-tuned
Robometer query-side scores against; the original 16 episodes are nowhere near enough for
per-task nearest-value retrieval.

Reuses robometer_train/run_finetuned_inference.py's own model loading and frame-sampling
approach (robometer_common.py) and reward_models_common/episode_cache.py's frame-path reader
(same source Robo-Dopamine's own pipeline uses -- no re-streaming/re-extraction needed). Loads the
model ONCE, not once per episode. Output format matches run_finetuned_inference.py's exactly, so
this is a drop-in coverage expansion, not a new/different reference source.

Must run in robometer_train's own .venv. Usage:
    python batch_robometer_reference.py [--shard-index 0 --shard-count 1] [--limit N] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

from robometer_common import (
    MAX_FRAMES, RobometerModel, load_frames_from_paths, subsample_frame_paths,
)

ICL_ANNOTATIONS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ICL_ANNOTATIONS_DIR))
from reward_models_common.episode_cache import load_frame_paths, load_manifest  # noqa: E402

ROBO_DOPAMINE_DIR = ICL_ANNOTATIONS_DIR / "Robo-Dopamine"
TASK_EPISODE_INDICES_DIR = ROBO_DOPAMINE_DIR / "dataset" / "task_episode_indices"
OUTPUT_DIR = ICL_ANNOTATIONS_DIR / "viewer" / "robometer_finetuned_progress" / "adityx23" / "icl-dataset"

COVERED_TASK_SLUGS = [
    "put_the_circle_on_the_peg", "stack_the_three_cups_into_a_tower", "stack_the_colored_octagons",
    "open_the_notebook", "open_the_gatorade_bottle", "hit_the_yellow_cube_with_the_mallet_using_the_left_arm",
    "sort_the_items_into_their_containers", "uncap_the_red_marker",
    "pass_the_salt_shaker_from_the_left_arm_to_the_right_arm", "pass_the_salt_shaker_from_the_right_arm_to_the_left_arm",
    "pick_up_the_red_chilli_and_place_it_on_the_plate_with_the_left_arm",
    "pick_up_the_red_chilli_and_place_it_on_the_plate_with_the_right_arm",
    "put_the_eggplant_into_the_box_with_the_left_arm", "put_the_eggplant_into_the_box_with_the_right_arm",
    "put_the_green_pepper_into_the_grocery_bag",
]


def _output_stem(manifest: dict) -> str:
    uid = manifest.get("episode_uid")
    return uid.replace(":", "__") if uid else f"episode_{manifest['episode_index']}"


def covered_episode_indices() -> list[int]:
    indices = []
    for slug in COVERED_TASK_SLUGS:
        f = TASK_EPISODE_INDICES_DIR / f"{slug}.txt"
        if not f.exists():
            print(f"[batch_robometer_reference] WARNING: no task_episode_indices file for {slug}")
            continue
        indices.extend(int(x) for x in f.read_text().strip().split(",") if x.strip())
    return sorted(set(indices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_episodes = covered_episode_indices()
    shard = [e for e in all_episodes if e % args.shard_count == args.shard_index]

    pending = []
    for episode_index in shard:
        manifest = load_manifest(episode_index)
        out_path = OUTPUT_DIR / f"{_output_stem(manifest)}.json"
        if out_path.exists() and not args.force:
            continue
        pending.append((episode_index, manifest, out_path))
    if args.limit:
        pending = pending[:args.limit]

    print(f"[batch_robometer_reference] shard {args.shard_index}/{args.shard_count}: "
          f"{len(shard)} episodes in shard, {len(pending)} pending")
    if not pending:
        return

    model = RobometerModel()
    failed = []
    for i, (episode_index, manifest, out_path) in enumerate(pending, 1):
        try:
            paths = load_frame_paths(manifest, cam_name="cam_high")
            paths, sampled_indices = subsample_frame_paths(paths, manifest["indices"], MAX_FRAMES)
            frames = load_frames_from_paths(paths)
            progress = model.score_frames(frames, manifest["task"], str(episode_index))
            n = min(len(sampled_indices), len(progress))
            points = [
                {"frame_index": sampled_indices[j], "progress": round(float(progress[j]) * 100, 2)}
                for j in range(n)
            ]
            out_path.write_text(json.dumps({
                "episode_uid": manifest.get("episode_uid"),
                "episode_index": episode_index,
                "task": manifest["task"],
                "success": manifest.get("success"),
                "points": points,
            }, indent=2), encoding="utf-8")
            print(f"[batch_robometer_reference] ({i}/{len(pending)}) episode {episode_index}: "
                  f"{len(points)} points -> {out_path}")
        except Exception as e:
            print(f"[batch_robometer_reference] ({i}/{len(pending)}) episode {episode_index}: FAILED -- {e!r}")
            failed.append(episode_index)

    print(f"[batch_robometer_reference] shard {args.shard_index} done: "
          f"{len(pending) - len(failed)} episodes ({len(failed)} failed: {failed})")


if __name__ == "__main__":
    main()
