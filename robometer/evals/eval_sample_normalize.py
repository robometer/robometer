"""
Normalize incoming eval-server trajectories before the batch collator runs.

Training / RBMBatchCollator uses ``_resize_pil`` (longest side + pixel cap) for Qwen when
``resized_height``/``resized_width`` are unset; multi-image mode historically skipped that pass.
The eval server applies the **same** geometry rules here so oversized client videos are safe.

Temporal cap: uniform index subsampling (``numpy.linspace``), same spirit as
``linspace_subsample_frames`` in the training path—no motion-aware downsampling.

Frames stored as a list of filesystem paths (``List[str]``) are left unchanged; the collator
loads those separately.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from robometer.data.collators.rbm_heads import MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, _resize_pil
from robometer.data.dataset_types import PreferenceSample, ProgressSample, Trajectory


def _subsample_indices(num_frames: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, max_frames, dtype=np.int64)


def _resize_hwc_uint8(frames: np.ndarray, *, max_side: int, max_pixels: int) -> np.ndarray:
    """Resize each (H,W,C) uint8 frame using the same policy as ``_resize_pil``."""
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        return frames
    out = []
    for i in range(frames.shape[0]):
        arr = np.asarray(frames[i], dtype=np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        pil = Image.fromarray(arr)
        pil = _resize_pil(pil, max_side=max_side, max_pixels=max_pixels)
        out.append(np.asarray(pil, dtype=np.uint8))
    return np.stack(out, axis=0)


def _subsample_aligned(values: Any, indices: np.ndarray) -> Any:
    """Subsample per-frame annotations to match ``indices``."""
    if values is None:
        return None
    idx = indices.astype(np.int64, copy=False)
    if isinstance(values, (list, tuple)):
        return [values[int(i)] for i in idx.tolist()]
    if isinstance(values, np.ndarray):
        v = np.asarray(values)
        if v.ndim == 1:
            return v[idx].tolist()
        return v[idx]
    if isinstance(values, torch.Tensor):
        idx_t = torch.as_tensor(idx.tolist(), device=values.device, dtype=torch.long)
        if values.dim() == 1:
            return values[idx_t].detach().cpu().tolist()
        return values[idx_t]
    return values


def _is_path_frame_list(frames: Any) -> bool:
    if not isinstance(frames, (list, tuple)) or not frames:
        return False
    return all(isinstance(x, str) for x in frames)


def normalize_trajectory(
    traj: Trajectory,
    *,
    max_frames: int | None,
    resize: bool,
    max_side: int = MAX_IMAGE_SIDE,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Trajectory:
    """Return a trajectory copy with subsampled / resized ``frames`` and aligned per-frame lists."""
    frames = traj.frames
    if frames is None or _is_path_frame_list(frames):
        return traj

    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            return traj
        t = int(frames.shape[0])
        idx = _subsample_indices(t, max_frames)
        new_frames = np.asarray(frames[idx], dtype=np.uint8)
        if resize:
            new_frames = _resize_hwc_uint8(new_frames, max_side=max_side, max_pixels=max_pixels)
        new_tp = _subsample_aligned(traj.target_progress, idx)
        new_sl = _subsample_aligned(traj.success_label, idx)
        new_plm = _subsample_aligned(traj.predict_last_frame_mask, idx)
        new_shape: tuple[int, ...] = tuple(new_frames.shape)
        return traj.model_copy(
            update={
                "frames": new_frames,
                "frames_shape": new_shape,
                "target_progress": new_tp if new_tp is not None else traj.target_progress,
                "success_label": new_sl if new_sl is not None else traj.success_label,
                "predict_last_frame_mask": new_plm if new_plm is not None else traj.predict_last_frame_mask,
            }
        )

    if isinstance(frames, (list, tuple)) and frames and hasattr(frames[0], "size"):
        t = len(frames)
        idx = _subsample_indices(t, max_frames)
        if resize:
            pil_seq = [_resize_pil(frames[int(i)], max_side=max_side, max_pixels=max_pixels) for i in idx]
        else:
            pil_seq = [frames[int(i)] for i in idx]
        new_frames = list(pil_seq)
        new_tp = _subsample_aligned(traj.target_progress, idx)
        new_sl = _subsample_aligned(traj.success_label, idx)
        new_plm = _subsample_aligned(traj.predict_last_frame_mask, idx)
        w, h = new_frames[0].size
        new_shape = (len(new_frames), h, w, 3)
        return traj.model_copy(
            update={
                "frames": new_frames,
                "frames_shape": new_shape,
                "target_progress": new_tp if new_tp is not None else traj.target_progress,
                "success_label": new_sl if new_sl is not None else traj.success_label,
                "predict_last_frame_mask": new_plm if new_plm is not None else traj.predict_last_frame_mask,
            }
        )

    return traj


def normalize_eval_samples(
    samples: list[Any],
    *,
    max_frames: int | None,
    resize: bool = True,
    max_side: int = MAX_IMAGE_SIDE,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> None:
    """Mutate the sample list in place so each trajectory is capped and optionally resized."""
    if max_frames is not None and max_frames <= 0:
        max_frames = None
    if max_frames is None and not resize:
        return
    for i, sample in enumerate(samples):
        if isinstance(sample, ProgressSample):
            samples[i] = sample.model_copy(
                update={
                    "trajectory": normalize_trajectory(
                        sample.trajectory,
                        max_frames=max_frames,
                        resize=resize,
                        max_side=max_side,
                        max_pixels=max_pixels,
                    )
                }
            )
        elif isinstance(sample, PreferenceSample):
            chosen = normalize_trajectory(
                sample.chosen_trajectory,
                max_frames=max_frames,
                resize=resize,
                max_side=max_side,
                max_pixels=max_pixels,
            )
            rejected = normalize_trajectory(
                sample.rejected_trajectory,
                max_frames=max_frames,
                resize=resize,
                max_side=max_side,
                max_pixels=max_pixels,
            )
            samples[i] = sample.model_copy(update={"chosen_trajectory": chosen, "rejected_trajectory": rejected})
