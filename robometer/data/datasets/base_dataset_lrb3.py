"""
LeRobot v3–aware :class:`BaseDataset` **without** eager frame materialization.

When ``ROBOMETER_LEROBOT_DATASET_ROOT`` is set, trajectories are **metadata-only** rows
(``frames=None``, ``lerobot_episode_idx``) built from parquet/tabular columns; pixels load
only when a sampler calls ``_get_traj_from_data`` (see ``LeRobotLazyFramesMixin``).

Otherwise identical to :class:`BaseDataset` (preprocessed Arrow cache).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from datasets import Dataset

from robometer.data.datasets.base import BaseDataset
from robometer.data.datasets.lerobot_episode_store_lrb3 import LeRobotEpisodeStore
from robometer.utils.logger import get_logger

logger = get_logger()


class BaseDatasetLRB3(BaseDataset):
    """Metadata-first LeRobot loading; falls back to the standard preprocessed cache."""

    def _load_all_datasets(self) -> Tuple[Dataset, Dict[str, Any]]:
        root = os.environ.get("ROBOMETER_LEROBOT_DATASET_ROOT", "").strip()
        if not root:
            return super()._load_all_datasets()

        repo_id = os.environ.get("ROBOMETER_LEROBOT_REPO_ID") or None
        video_key = os.environ.get("ROBOMETER_LEROBOT_VIDEO_KEY") or None
        logger.info(
            f"BaseDatasetLRB3: LeRobot metadata-first load root={root!r} repo_id={repo_id!r} video_key={video_key!r}"
        )

        store = LeRobotEpisodeStore.from_local_root(root, repo_id=repo_id, video_key=video_key)
        self._lerobot_store = store
        return store.build_hf_dataset_and_indices()
