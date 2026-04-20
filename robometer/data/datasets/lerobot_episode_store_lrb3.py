"""
LeRobot v3 episode store: **metadata without decoding video**, lazy ``load_frames(ep_idx)`` on demand.

Used by :class:`BaseDatasetLRB3` to build a small HuggingFace ``Dataset`` of metadata rows
(``frames=None``, ``lerobot_episode_idx`` set). Samplers resolve pixels only inside
``LeRobotLazyFramesMixin._get_traj_from_data``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robometer.data.datasets.lerobot_indices_lrb3 import build_combined_indices_from_metadata_rows
from robometer.utils.logger import get_logger

logger = get_logger()


def _to_int_scalar(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, torch.Tensor):
        return int(val.reshape(-1)[0].item())
    if isinstance(val, np.ndarray):
        return int(val.reshape(-1)[0])
    if isinstance(val, (list, tuple)):
        return int(val[0])
    return int(val)


def _first_float(val: Any, *, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        return float(val.detach().reshape(-1)[0].item())
    if isinstance(val, np.ndarray):
        return float(np.asarray(val).reshape(-1)[0])
    if isinstance(val, (list, tuple)):
        return float(val[0])
    return float(val)


def _vision_to_hwc_uint8(vid: Any) -> np.ndarray:
    if isinstance(vid, torch.Tensor):
        t = vid.detach().float().cpu()
        if t.dim() == 3 and t.shape[0] == 3:
            t = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0)
            return np.asarray(t.numpy(), dtype=np.uint8)
        if t.dim() == 3 and t.shape[-1] == 3:
            t = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            return np.asarray(t.numpy(), dtype=np.uint8)
    if isinstance(vid, np.ndarray):
        v = vid
        if v.ndim == 3 and v.shape[0] == 3:
            v = np.transpose(v, (1, 2, 0))
        if v.dtype != np.uint8:
            v = np.clip(v * 255.0, 0, 255).astype(np.uint8) if float(v.max()) <= 1.0 else v.astype(np.uint8)
        return np.asarray(v, dtype=np.uint8)
    raise TypeError(f"Unsupported vision type: {type(vid)}")


def _load_categorical_maps(root: Path) -> dict[str, Any]:
    p = root / "robometer_categorical_maps.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def _id_to_label(maps: dict, section: str) -> dict[int, str]:
    sec = maps.get(section) or {}
    raw = sec.get("id_to_label") or {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


class LeRobotEpisodeStore:
    """Holds a ``LeRobotDataset`` handle and builds metadata rows + lazy frame loader."""

    def __init__(
        self,
        lr: Any,
        *,
        root: Path,
        repo_id: str,
        video_key: str,
        metadata_rows: list[dict],
    ):
        self._lr = lr
        self.root = root
        self.repo_id = repo_id
        self.video_key = video_key
        self.metadata_rows = metadata_rows

    @classmethod
    def from_local_root(
        cls,
        root: str | Path,
        *,
        repo_id: str | None = None,
        video_key: str | None = None,
    ) -> LeRobotEpisodeStore:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise ImportError(
                "Install lerobot to use LeRobotEpisodeStore (see dataset_upload/requirements_lerobot_dataset_only.txt)."
            ) from e

        root_path = Path(root).expanduser().resolve()
        repo_id = repo_id or os.environ.get("ROBOMETER_LEROBOT_REPO_ID", "local/robometer_lrb3")
        video_key = video_key or os.environ.get("ROBOMETER_LEROBOT_VIDEO_KEY", "observation.images.main")

        lr = LeRobotDataset(repo_id=repo_id, root=str(root_path), download_videos=False)

        maps = _load_categorical_maps(root_path)
        q_map = _id_to_label(maps, "quality_label")
        ds_map = _id_to_label(maps, "data_source")

        exclude: set[str] = set()
        for k, ft in (lr.meta.features or {}).items():
            if (ft or {}).get("dtype") in ("video", "image"):
                exclude.add(k)

        col_names = list(lr.hf_dataset.column_names)
        meta_cols = [c for c in col_names if c not in exclude and c != video_key]
        if "episode_index" not in meta_cols:
            raise KeyError("LeRobot hf_dataset must contain episode_index column for metadata scan.")

        pdf = lr.hf_dataset.select_columns(meta_cols).to_pandas()
        ep_table = lr.meta.episodes
        n_eps = len(ep_table)
        max_eps_env = os.environ.get("ROBOMETER_LEROBOT_MAX_EPISODES")
        if max_eps_env:
            n_eps = min(n_eps, int(max_eps_env))

        h_def, w_def = 224, 224
        vk_feat = (lr.meta.features or {}).get(video_key) or {}
        shp = vk_feat.get("shape")
        if isinstance(shp, (list, tuple)) and len(shp) == 3:
            _, h_def, w_def = int(shp[1]), int(shp[2])

        rows: list[dict] = []
        for ep_idx in range(n_eps):
            ep = ep_table[ep_idx]
            i0 = _to_int_scalar(ep["dataset_from_index"])
            i1 = _to_int_scalar(ep["dataset_to_index"])
            t = max(0, i1 - i0)
            if t == 0:
                continue

            row_pdf = pdf.iloc[i0]
            task_idx = int(row_pdf["task_index"]) if "task_index" in row_pdf else 0
            task = str(lr.meta.tasks.iloc[task_idx].name)

            qid = int(row_pdf["robometer.quality_label_id"]) if "robometer.quality_label_id" in row_pdf else -1
            if qid in q_map:
                quality_label = q_map[qid]
            elif qid >= 0:
                quality_label = str(qid)
            else:
                quality_label = "successful"

            ds_name = str(repo_id)
            if "robometer.data_source_id" in row_pdf:
                did = int(row_pdf["robometer.data_source_id"])
                ds_name = ds_map.get(did, str(did))

            is_robot = True
            if "robometer.is_robot" in row_pdf:
                is_robot = _first_float(row_pdf["robometer.is_robot"], default=1.0) > 0.5

            partial_success = None
            if "robometer.partial_success" in row_pdf:
                partial_success = _first_float(row_pdf["robometer.partial_success"], default=0.0)

            rows.append({
                "id": f"lerobot:{repo_id}:ep{ep_idx}",
                "lerobot_episode_idx": ep_idx,
                "task": task,
                "quality_label": quality_label,
                "data_source": ds_name,
                "is_robot": bool(is_robot),
                "partial_success": partial_success,
                "num_frames": t,
                "frames_shape": (t, h_def, w_def, 3),
                "frames": None,
                "lang_vector": None,
                "metadata": {"lerobot_repo_id": repo_id, "lerobot_root": str(root_path)},
            })

        if not rows:
            raise RuntimeError(f"No episodes found under LeRobot root {root_path}")

        logger.info(
            f"LeRobotEpisodeStore: {len(rows)} episodes, metadata from parquet columns only (video key={video_key!r})"
        )
        return cls(lr, root=root_path, repo_id=repo_id, video_key=video_key, metadata_rows=rows)

    def load_frames(self, episode_idx: int) -> np.ndarray:
        """Decode all frames for one episode (call only after a trajectory is chosen)."""
        ep_table = self._lr.meta.episodes
        if episode_idx < 0 or episode_idx >= len(ep_table):
            raise IndexError(f"episode_idx {episode_idx} out of range")

        ep = ep_table[episode_idx]
        i0 = _to_int_scalar(ep["dataset_from_index"])
        i1 = _to_int_scalar(ep["dataset_to_index"])
        out: list[np.ndarray] = []
        for abs_idx in range(i0, i1):
            item = self._lr[abs_idx]
            if self.video_key not in item:
                raise KeyError(f"Missing {self.video_key!r} at frame index {abs_idx}")
            out.append(_vision_to_hwc_uint8(item[self.video_key]))
        return np.stack(out, axis=0)

    def build_hf_dataset_and_indices(self) -> tuple[Any, dict[str, Any]]:
        from datasets import Dataset

        ds = Dataset.from_list(self.metadata_rows)
        idx = build_combined_indices_from_metadata_rows(self.metadata_rows)
        return ds, idx
