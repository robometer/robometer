"""Batch version of robometer_query.py: loads the fine-tuned Robometer checkpoint ONCE, scores
every already-streamed, keep_true query episode (see batch_stream.py). By default restricts to
tasks covered by the expanded robometer_finetuned_progress reference set
(batch_robometer_reference.py) -- pass --all-tasks to lift that. That restriction is NOT a model
requirement: compute_robometer_progress (robometer_query.py) scores a query episode purely from
its own cached frames + task string, with no reference-pool lookup at all (unlike Robo-Dopamine's
backward-mode goal image) -- the task filter exists only to keep the default scope aligned with
what retrieve.py's nearest-value matching can actually use, since a query episode's progress score
is only useful for retrieval if there's a reference pool for its task. --all-tasks scores every
keep_true episode regardless, e.g. for building dense/continuous annotations (see
build_continuous_annotations.py) where no retrieval reference is needed.

Mirrors batch_infer.py's shape (shard-index/shard-count, per-episode failure isolation,
skip-if-done).

Must run in robometer_train's own .venv.
Usage: python batch_robometer_query.py [--all-tasks] [--shard-index 0 --shard-count 1] [--limit N] [--force]
"""
import argparse
import json
from pathlib import Path

from robometer_common import CHECKPOINT_PATH, ZERO_SHOT_CHECKPOINT_PATH, RobometerModel
from robometer_query import compute_robometer_progress
from batch_robometer_reference import COVERED_TASK_SLUGS

HERE = Path(__file__).resolve().parent
DEFAULT_QUERY_CACHE = HERE / "query_episode_cache"
DEFAULT_EPISODES_JSONL = HERE / ".lerobot_cache" / "adityx23" / "icl-demo-dataset" / "meta" / "episodes.jsonl"


def slugify(task: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")


def keep_true_episode_indices(episodes_jsonl: Path = DEFAULT_EPISODES_JSONL) -> set[int]:
    out = set()
    for line in episodes_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("keep"):
            out.add(d["episode_index"])
    return out


def streamed_episodes(query_cache_dir: Path) -> list[tuple[int, Path, dict]]:
    keep_true = keep_true_episode_indices()
    out = []
    for ep_dir in sorted(query_cache_dir.glob("episode_*")):
        manifest_path = ep_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["episode_index"] not in keep_true:
            continue
        out.append((manifest["episode_index"], ep_dir, manifest))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query-cache-dir", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-tasks", action="store_true",
                         help="Score every keep_true episode, not just tasks with a retrieval reference pool.")
    parser.add_argument("--zero-shot", action="store_true",
                         help=f"Score with the base checkpoint ({ZERO_SHOT_CHECKPOINT_PATH}) instead of the "
                              "fine-tuned one -- same hindsight MAX_FRAMES method either way, written to a "
                              "differently-named output file so it doesn't overwrite the fine-tuned results.")
    parser.add_argument("--model-path", default=None, help="Override the checkpoint path/Hub id directly.")
    args = parser.parse_args()

    model_path = args.model_path or (ZERO_SHOT_CHECKPOINT_PATH if args.zero_shot else CHECKPOINT_PATH)
    out_name = "robometer_zeroshot_progress.json" if args.zero_shot else "robometer_progress.json"

    covered = set(COVERED_TASK_SLUGS)
    all_episodes = streamed_episodes(args.query_cache_dir)
    shard = [e for e in all_episodes if e[0] % args.shard_count == args.shard_index]

    pending = []
    skipped_uncovered = 0
    for episode_index, episode_dir, manifest in shard:
        if not args.force and (episode_dir / out_name).exists():
            continue
        if not args.all_tasks and slugify(manifest["task"]) not in covered:
            skipped_uncovered += 1
            continue
        pending.append((episode_index, episode_dir, manifest))
    if args.limit:
        pending = pending[:args.limit]

    print(f"[batch_robometer_query] shard {args.shard_index}/{args.shard_count}: "
          f"{len(shard)} episodes in shard, {len(pending)} pending, {skipped_uncovered} skipped (uncovered)")
    if not pending:
        return

    model = RobometerModel(model_path=model_path)
    failed = []
    for i, (episode_index, episode_dir, manifest) in enumerate(pending, 1):
        try:
            progress = compute_robometer_progress(manifest, model, episode_dir)
        except Exception as e:
            print(f"[batch_robometer_query] ({i}/{len(pending)}) episode {episode_index}: FAILED -- {e!r}")
            failed.append(episode_index)
            continue
        out_path = episode_dir / out_name
        out_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"[batch_robometer_query] ({i}/{len(pending)}) episode {episode_index}: "
              f"{len(progress)} points -> {out_path}")

    print(f"[batch_robometer_query] shard {args.shard_index} done: "
          f"{len(pending) - len(failed)} episodes ({len(failed)} failed: {failed})")


if __name__ == "__main__":
    main()
