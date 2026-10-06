#!/usr/bin/env python3
"""icl-dataset loader for Robometer's generic dataset converter -- see CustomDataset.md.

Builds task_data from two sources that already exist elsewhere in this checkout (a sibling
`icl_annotations/` project directory, not part of robometer_train itself):

- The same two Hub files ../../viewer/segmentation.py's fetch_task_catalog() reads
  (meta/{tasks,episodes}.jsonl -- small, cheap, no parquet/video download), inlined here
  (_fetch_episode_success_labels below) rather than importing that module directly: segmentation
  .py unconditionally imports lerobot at module load (for its *other* functions, e.g.
  load_episode_dataset), which isn't installed in robometer_train's own separate venv and isn't
  needed for this one function. Gives every episode's task string and per-episode `success`
  field ("success"/"fail"/"invalid") for the FULL dataset (3149 episodes), which is exactly the
  sparse (episode-level, no dense annotation) label RoboMeter's fine-tune needs -- unlike
  Robo-Dopamine/ProcVLM's one-shot LoRA data (dense, 16-episode, from manual_annotation), this
  covers every episode.
- ../../reward_models_common/episode_cache.py's already-extracted per-episode PNG cache
  (frame_interval=10, all 3 cameras -- see that module's own docstring for why: lerobot>=0.6.1
  can't stream this dataset's v2.1 format without a full v3.0 conversion, so every other reward-
  model pipeline in this project reuses this same cache instead of touching lerobot directly).

`invalid`-labeled episodes are skipped -- validate_dataset.py's quality_label check only accepts
{"successful", "failure", "suboptimal"}, and "invalid" (episode setup/recording problems, not a
task outcome) doesn't map to any of those.
"""
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from dataset_upload.helpers import generate_unique_id

_ICL_ANNOTATIONS_DIR = Path(__file__).resolve().parents[3]
if str(_ICL_ANNOTATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_ICL_ANNOTATIONS_DIR))

QUALITY_LABEL_BY_SUCCESS = {"success": "successful", "fail": "failure"}

_HF_REPO_ID = "adityx23/icl-dataset"
_HF_REVISION = "9bb0c92eb3d9d06ea81096f5a7c4048485acc36b"


def _fetch_episode_success_labels() -> dict[int, tuple[str, str | None]]:
    """episode_index -> (task, success) for every episode, straight from the Hub's own
    meta/{tasks,episodes}.jsonl -- same two files and same logic as
    ../../viewer/segmentation.py's fetch_task_catalog(), reimplemented here without importing
    that module (see module docstring) rather than shelled out to or monkeypatched around."""
    # tasks.jsonl isn't actually needed -- episodes.jsonl's own "tasks" field already holds the
    # resolved task strings directly (confirmed empirically), not task_index integers.
    episodes_path = hf_hub_download(_HF_REPO_ID, "meta/episodes.jsonl", repo_type="dataset", revision=_HF_REVISION)

    by_episode: dict[int, tuple[str, str | None]] = {}
    for line in Path(episodes_path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        # Episodes can list multiple tasks; icl-dataset's own convention (see
        # fetch_task_catalog) is one task per episode in practice -- take the first.
        tasks = row.get("tasks")
        if not tasks:
            continue
        by_episode[row["episode_index"]] = (tasks[0], row.get("success"))
    return by_episode


class IclFrameLoader:
    """Pickle-able loader that reads one episode's cam_high frames from Robo-Dopamine's episode
    cache on demand -- same lazy, multiprocess-safe pattern as libero_loader.py's
    LiberoFrameLoader (this converter calls trajectory["frames"] once per trajectory, in a
    worker pool -- see generate_hf_dataset.py's convert_dataset_to_hf_format). Stores only the
    episode index (a plain int) so it survives being pickled across worker processes."""

    def __init__(self, episode_index: int):
        self.episode_index = episode_index

    def __call__(self) -> np.ndarray:
        """np.ndarray of shape (T, H, W, 3), dtype uint8, RGB order -- create_trajectory_video_
        optimized converts RGB->BGR itself before piping to ffmpeg (see helpers.py), so this
        must stay RGB, matching PIL's own convert("RGB") here."""
        from reward_models_common.episode_cache import load_frame_paths, load_manifest

        manifest = load_manifest(self.episode_index)
        paths = load_frame_paths(manifest, cam_name="cam_high")
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
        return np.stack(frames)


def load_icl_dataset(base_path: str) -> dict[str, list[dict]]:
    """Load icl-dataset and organize by task, for every episode with a usable success/fail
    label and a local frame cache. `base_path` is accepted for interface parity with the other
    loaders (see CustomDataset.md) but unused -- this loader's two real data sources
    (fetch_task_catalog, episode_cache) are fixed locations within this checkout, not a path the
    caller points at."""
    from reward_models_common.episode_cache import iter_episode_indices

    print(f"Loading icl-dataset (base_path={base_path!r} unused, see module docstring)")
    print("=" * 100)
    print("LOADING ICL_DATASET DATASET")
    print("=" * 100)

    cached_indices = set(iter_episode_indices())
    labels_by_episode = _fetch_episode_success_labels()

    task_data: dict[str, list[dict]] = {}
    skipped_invalid = 0
    skipped_uncached = 0
    for episode_index, (task_name, success) in tqdm(labels_by_episode.items(), desc="icl_dataset"):
        quality_label = QUALITY_LABEL_BY_SUCCESS.get(success)
        if quality_label is None:
            skipped_invalid += 1
            continue
        if episode_index not in cached_indices:
            skipped_uncached += 1
            continue
        task_data.setdefault(task_name, []).append({
            "frames": IclFrameLoader(episode_index),
            "is_robot": True,
            "quality_label": quality_label,
            "task": task_name,
            "id": generate_unique_id(),
        })

    total = sum(len(v) for v in task_data.values())
    print(
        f"Loaded {total} trajectories from {len(task_data)} tasks "
        f"(skipped {skipped_invalid} invalid-labeled, {skipped_uncached} uncached)"
    )
    return task_data
