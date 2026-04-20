"""Mixin: resolve ``frames=None`` + ``lerobot_episode_idx`` via :class:`LeRobotEpisodeStore.load_frames`."""

from __future__ import annotations

from typing import Any

from robometer.data.dataset_types import Trajectory


class LeRobotLazyFramesMixin:
    """Expects ``self._lerobot_store`` (set by :class:`RBMDatasetLRB3` after init)."""

    def _get_traj_from_data(
        self,
        traj: dict | Trajectory,
        subsample_strategy: str | None = None,
        frame_indices: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
        pad_frames: bool = True,
    ):
        store = getattr(self, "_lerobot_store", None)
        if (
            store is not None
            and isinstance(traj, dict)
            and traj.get("frames") is None
            and traj.get("lerobot_episode_idx") is not None
        ):
            traj = dict(traj)
            arr = store.load_frames(int(traj["lerobot_episode_idx"]))
            traj["frames"] = arr
            traj["frames_shape"] = tuple(arr.shape)
            traj["num_frames"] = int(arr.shape[0])
        return super()._get_traj_from_data(
            traj,
            subsample_strategy=subsample_strategy,
            frame_indices=frame_indices,
            metadata=metadata,
            pad_frames=pad_frames,
        )
