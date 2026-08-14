#!/usr/bin/env python3
"""
RoboCasa dataset loader for Robometer fine-tuning.
Loads LeRobot-format RoboCasa trajectories: per-task directories with MP4 videos and JSONL metadata.
Supports all tasks (atomic + composite, pretrain + target) by auto-discovering from directory structure
and reading task descriptions from tasks.jsonl.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from dataset_upload.helpers import generate_unique_id
from dataset_upload.video_helpers import load_video_frames


class RoboCasaFrameLoader:
    """Pickle-able loader that reads RoboCasa video files on demand."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __call__(self):
        """Load frames from video file. Returns np.ndarray (T, H, W, 3) uint8."""
        return load_video_frames(Path(self.file_path))


# Camera to use for training
DEFAULT_CAMERA = "robot0_agentview_left"


def _load_episode_metadata(meta_dir: Path) -> list[dict]:
    """Load episode metadata from episodes.jsonl."""
    episodes = []
    episodes_file = meta_dir / "episodes.jsonl"
    if not episodes_file.exists():
        return episodes
    with open(episodes_file) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def _load_task_description(meta_dir: Path, task_name: str) -> str:
    """Load task description from tasks.jsonl. Falls back to converting task name to sentence."""
    tasks_file = meta_dir / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_text = entry.get("task", "")
                # task_index 0 is usually the natural language description
                # task_index 1 is usually the class name
                if entry.get("task_index", -1) == 0 and task_text != task_name:
                    return task_text
    # Fallback: convert CamelCase task name to sentence
    import re
    words = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', task_name).lower()
    return words + "."


def _discover_task_dirs(base_path: Path, splits: list[str] | None = None) -> list[Path]:
    """Discover all task directories under the base path.

    Handles both flat structure (base_path/TaskName/...) and
    split structure (base_path/{pretrain,target}/{atomic,composite}/TaskName/...).

    Args:
        base_path: Root path to the dataset.
        splits: Optional list of splits to include (e.g. ["pretrain"] or ["target"]).
                If None, all splits are included.
    """
    task_dirs = []

    # Check if base_path contains split subdirectories
    all_splits = ["pretrain", "target"]
    categories = ["atomic", "composite"]

    has_splits = any((base_path / s).is_dir() for s in all_splits)

    if has_splits:
        active_splits = splits if splits else all_splits
        for split in active_splits:
            for cat in categories:
                cat_dir = base_path / split / cat
                if cat_dir.is_dir():
                    for d in sorted(cat_dir.iterdir()):
                        if d.is_dir():
                            task_dirs.append(d)
    else:
        # Flat structure: task dirs directly under base_path
        task_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    return task_dirs


def _find_lerobot_dir(task_dir: Path, source: str = "human") -> Optional[Path]:
    """Find the appropriate lerobot directory based on data source.

    Args:
        task_dir: Path to a task directory (e.g. .../atomic/CloseFridge/)
        source: "human" for human demos, "mimicgen" for MimicGen-generated data

    Returns:
        Path to the lerobot directory, or None if not found.
    """
    lerobot_dirs = sorted(task_dir.rglob("lerobot"), key=lambda p: len(p.parts))
    if not lerobot_dirs:
        return None

    if source == "mimicgen":
        # MimicGen lerobot dirs are under an mg/ parent
        mg_dirs = [d for d in lerobot_dirs if "mg" in d.parts]
        return mg_dirs[0] if mg_dirs else None
    else:
        # Human lerobot dirs are NOT under an mg/ parent
        human_dirs = [d for d in lerobot_dirs if "mg" not in d.parts]
        return human_dirs[0] if human_dirs else None


def _find_video_path(lerobot_dir: Path, camera: str, ep_idx: int) -> Optional[Path]:
    """Find video file across multiple chunks (MimicGen data uses chunk-000 through chunk-009).

    Args:
        lerobot_dir: Path to the lerobot directory
        camera: Camera view name
        ep_idx: Episode index

    Returns:
        Path to the video file, or None if not found.
    """
    videos_dir = lerobot_dir / "videos"
    if not videos_dir.exists():
        return None

    # Check all chunks (chunk-000, chunk-001, etc.)
    for chunk_dir in sorted(videos_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        video_path = chunk_dir / f"observation.images.{camera}" / f"episode_{ep_idx:06d}.mp4"
        if video_path.exists():
            return video_path

    return None


def load_robocasa_dataset(
    base_path: str,
    max_trajectories: Optional[int] = None,
    camera: str = DEFAULT_CAMERA,
    splits: list[str] | None = None,
    source: str = "human",
) -> Dict[str, List[Dict]]:
    """Load RoboCasa dataset organized by task.

    Supports auto-discovery of all tasks from the directory structure.
    Task descriptions are read from tasks.jsonl metadata.

    Expected directory structure (either flat or hierarchical):
        base_path/
            [pretrain|target]/
                [atomic|composite]/
                    <TaskName>/
                        <date>/
                            lerobot/                    (human demos)
                            mg/demo/<timestamp>/lerobot/ (MimicGen data)

    Args:
        base_path: Path to the RoboCasa dataset root (e.g. datasets/v1.0/)
        max_trajectories: Maximum total trajectories to load (None = all)
        camera: Camera view to use
        splits: Optional list of splits to include (e.g. ["pretrain"] or ["target"])
        source: Data source to load - "human" (default) or "mimicgen"

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"RoboCasa dataset path not found: {base_path}")

    print(f"Loading RoboCasa dataset from: {base_path} (source={source})")
    print("=" * 80)

    task_dirs = _discover_task_dirs(base_path, splits=splits)
    print(f"Discovered {len(task_dirs)} task directories")

    task_data: Dict[str, List[Dict]] = {}
    total_loaded = 0
    skipped_tasks = []

    for task_dir in task_dirs:
        if max_trajectories and total_loaded >= max_trajectories:
            break

        task_name = task_dir.name

        # Find the lerobot directory for the requested source
        lerobot_dir = _find_lerobot_dir(task_dir, source=source)
        if lerobot_dir is None:
            skipped_tasks.append((task_name, f"no {source} lerobot dir"))
            continue

        # Load episode metadata
        meta_dir = lerobot_dir / "meta"
        episodes = _load_episode_metadata(meta_dir)
        if not episodes:
            skipped_tasks.append((task_name, "no episodes"))
            continue

        # Get task description from metadata
        task_desc = _load_task_description(meta_dir, task_name)

        trajectories = []
        for ep in tqdm(episodes, desc=f"  {task_name}", leave=False):
            if max_trajectories and total_loaded >= max_trajectories:
                break

            ep_idx = ep["episode_index"]
            ep_length = ep.get("length", 0)
            # Use per-episode task description if available
            ep_task_desc = task_desc
            if "tasks" in ep and ep["tasks"]:
                ep_task_desc = ep["tasks"][0]

            video_path = _find_video_path(lerobot_dir, camera, ep_idx)
            if video_path is None:
                continue

            actions = np.zeros((ep_length, 12), dtype=np.float32)

            trajectory = {
                "frames": RoboCasaFrameLoader(str(video_path)),
                "actions": actions,
                "is_robot": True,
                "task": ep_task_desc,
                "optimal": "optimal",  # RoboCasa data is all successful demonstrations
                "id": generate_unique_id(),
                "quality_label": "successful",
                "data_source": f"robocasa_{source}",
                "partial_success": 1.0,  # simulation data, all successful
            }
            trajectories.append(trajectory)
            total_loaded += 1

        if trajectories:
            task_data[task_name] = trajectories
            print(f"  {task_name}: {len(trajectories)} trajectories ({task_desc})")

        if max_trajectories and total_loaded >= max_trajectories:
            break

    total = sum(len(v) for v in task_data.values())
    print(f"\nTotal: {total} trajectories from {len(task_data)} tasks")
    if skipped_tasks:
        print(f"Skipped {len(skipped_tasks)} tasks: {', '.join(f'{t[0]} ({t[1]})' for t in skipped_tasks[:10])}")
        if len(skipped_tasks) > 10:
            print(f"  ... and {len(skipped_tasks) - 10} more")
    return task_data
