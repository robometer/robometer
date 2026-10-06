"""Batch version of robometer_query_online.py: loads the fine-tuned Robometer checkpoint ONCE,
causally re-scores every already-streamed, keep_true icl-demo-dataset query episode that
batch_robometer_query.py already scored non-causally. See robometer_query_online.py's own
docstring for why this is a genuinely different (causal, online-rollout-style) computation, not
a reformat of the same numbers.

Mirrors batch_robometer_query.py's shape exactly (shard-index/shard-count, per-episode failure
isolation, skip-if-done, same task filter/--all-tasks default) -- reuses its
streamed_episodes()/keep_true_episode_indices()/slugify() and
batch_robometer_reference.COVERED_TASK_SLUGS directly rather than re-deriving them, so both
scripts always agree on which episodes exist and which are in-scope by default. The only
difference is which per-episode scorer runs and which output filename it writes
(online_robometer_progress.json, alongside the existing robometer_progress.json rather than
overwriting it).

Must run in robometer_train's own .venv.
Usage: python batch_robometer_query_online.py [--all-tasks] [--shard-index 0 --shard-count 1] [--limit N] [--force]
"""
import argparse
import json

from robometer_common import CHECKPOINT_PATH, ZERO_SHOT_CHECKPOINT_PATH, RobometerModel
from robometer_query_online import compute_robometer_progress_online, DEFAULT_BATCH_SIZE
from batch_robometer_query import streamed_episodes, slugify
from batch_robometer_reference import COVERED_TASK_SLUGS

from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_QUERY_CACHE = HERE / "query_episode_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query-cache-dir", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-tasks", action="store_true",
                         help="Score every keep_true episode, not just tasks with a retrieval reference pool.")
    parser.add_argument("--zero-shot", action="store_true",
                         help=f"Score with the base checkpoint ({ZERO_SHOT_CHECKPOINT_PATH}) instead of the "
                              "fine-tuned one -- same causal deque method either way, written to a "
                              "differently-named output file so it doesn't overwrite the fine-tuned results.")
    parser.add_argument("--model-path", default=None, help="Override the checkpoint path/Hub id directly.")
    args = parser.parse_args()

    model_path = args.model_path or (ZERO_SHOT_CHECKPOINT_PATH if args.zero_shot else CHECKPOINT_PATH)
    out_name = "online_robometer_zeroshot_progress.json" if args.zero_shot else "online_robometer_progress.json"

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

    print(f"[batch_robometer_query_online] shard {args.shard_index}/{args.shard_count}: "
          f"{len(shard)} episodes in shard, {len(pending)} pending, {skipped_uncovered} skipped (uncovered)")
    if not pending:
        return

    model = RobometerModel(model_path=model_path)
    failed = []
    for i, (episode_index, episode_dir, manifest) in enumerate(pending, 1):
        try:
            progress = compute_robometer_progress_online(manifest, model, episode_dir, args.batch_size)
        except Exception as e:
            print(f"[batch_robometer_query_online] ({i}/{len(pending)}) episode {episode_index}: FAILED -- {e!r}")
            failed.append(episode_index)
            continue
        out_path = episode_dir / out_name
        out_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"[batch_robometer_query_online] ({i}/{len(pending)}) episode {episode_index}: "
              f"{len(progress)} points -> {out_path}")

    print(f"[batch_robometer_query_online] shard {args.shard_index} done: "
          f"{len(pending) - len(failed)} episodes ({len(failed)} failed: {failed})")


if __name__ == "__main__":
    main()
