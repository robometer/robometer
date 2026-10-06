"""Shared reader for Robo-Dopamine's own episode extraction cache
(../Robo-Dopamine/episode_cache/episode_{N:06d}/), reused here instead of each
new reward-model pipeline re-streaming/re-extracting the dataset itself.

All 3149 episodes of adityx23/icl-dataset (pinned revision
9bb0c92eb3d9d06ea81096f5a7c4048485acc36b) are already extracted by
Robo-Dopamine/windows/extract_episode.py (frame_interval=10, all 3 cameras,
as PNGs) -- RoboMeter's and TOPReward's actual scoring APIs only need a plain
frame tensor + task string, not a LeRobotDataset object, so there's no need
to touch lerobot's own dataset-streaming code (which can't open this v2.1
dataset under lerobot>=0.6.1 without a full v2.1->v3.0 conversion anyway).

Loaded frames match LeRobotDataset's own per-frame convention exactly (CHW,
float32, [0,1]) -- see extract_episode.py::tensor_to_pil for the inverse of
this conversion, which is what originally produced these PNGs -- since that's
what Robometer/TOPReward's encoder processor steps were built to consume
(RobometerEncoderProcessorStep's docstring: "observation[image_key]: (B, T,
C, H, W) ... frames"; _video_to_numpy explicitly handles float32-in-[0,1]
input, auto-scaling by 255).
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EPISODE_CACHE_DIR = Path(__file__).resolve().parent.parent / "Robo-Dopamine" / "episode_cache"

CAM_NAME_BY_IMAGE_KEY = {
    "observation.images.zed": "cam_high",
    "observation.images.fish0": "cam_left_wrist",
    "observation.images.fish1": "cam_right_wrist",
}


def iter_episode_indices(cache_dir: Path = EPISODE_CACHE_DIR) -> list[int]:
    """Every episode index with a manifest.json in the cache, sorted.

    `cache_dir` defaults to icl-dataset's cache above; pass e.g.
    `reward_guided_retrieval/query_episode_cache` (icl-demo-dataset's own manifest cache, same
    on-disk shape, built by `reward_guided_retrieval/query_stream.py`) to read that dataset
    instead."""
    indices = []
    for d in cache_dir.glob("episode_*"):
        if (d / "manifest.json").exists():
            indices.append(int(d.name.removeprefix("episode_")))
    return sorted(indices)


def load_manifest(episode_index: int, cache_dir: Path = EPISODE_CACHE_DIR) -> dict:
    path = cache_dir / f"episode_{episode_index:06d}" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(manifest: dict, image_key: str = "observation.images.zed", cache_dir: Path = EPISODE_CACHE_DIR) -> torch.Tensor:
    """(len(manifest["indices"]), C, H, W) float32 tensor in [0,1], frames in
    the same order as manifest["indices"] (the original episode's sampled
    frame positions -- also what frame_index values downstream output should
    use, since these PNGs don't cover every original frame)."""
    cam_name = CAM_NAME_BY_IMAGE_KEY[image_key]
    ep_dir = cache_dir / f"episode_{manifest['episode_index']:06d}"
    cam_dir = ep_dir / manifest["cam_dirs"][cam_name]
    frames = []
    for idx in manifest["indices"]:
        img = Image.open(cam_dir / f"frame_{idx:06d}.png").convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # CHW
    return torch.stack(frames)


def load_frame_paths(manifest: dict, cam_name: str = "cam_high", cache_dir: Path = EPISODE_CACHE_DIR) -> list[Path]:
    """Raw PNG file paths (not decoded), same order as manifest["indices"] -- for pipelines
    like ProcVLM whose own scoring API wants image file paths directly (via qwen_vl_utils'
    process_vision_info, which accepts local paths) rather than a pre-decoded tensor batch."""
    ep_dir = cache_dir / f"episode_{manifest['episode_index']:06d}"
    cam_dir = ep_dir / manifest["cam_dirs"][cam_name]
    return [cam_dir / f"frame_{idx:06d}.png" for idx in manifest["indices"]]
