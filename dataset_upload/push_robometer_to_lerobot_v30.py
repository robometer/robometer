#!/usr/bin/env python3
"""
Convert Robometer HF datasets (``save_to_disk``) to LeRobot v3.0 and optionally push to the Hub.

**Example — MetaWorld category** (``DATASET_MAP`` key ``mw``: ``aliangdw_metaworld_metaworld_{train,eval}``)::

    uv run python dataset_upload/push_robometer_to_lerobot_v30.py \
        --dataset-category mw \
        --processed-datasets-root /scr/shared/reward_fm/processed_datasets \
        --split train \
        --video-root /scr/shared/reward_fm/processed_datasets \
        --out-root /scr/shared/reward_fm/lerobot_mw_v30 \
        --repo-id aliang80/rbm-mw \
        --max-episodes -1 \
        --max-frames-per-episode 32

Use ``--split eval`` or ``both`` to match the ``eval`` (or both) lists under ``mw`` in
``robometer/data/dataset_category.py``. ``--video-root`` should resolve the relative ``frames``
paths in each row. If paths begin with ``processed_datasets/...`` and you also set
``--video-root`` to ``.../processed_datasets``, the script strips the duplicate segment
automatically; alternatively set ``--video-root`` to the parent (e.g. ``.../reward_fm``).

**Modes**

1. **Single dataset** — pass ``--robometer-dataset`` (path to ``processed_dataset/``).
2. **Merged category** — pass ``--dataset-category`` (e.g. ``rbm-1m-ood``) and
   ``--processed-datasets-root``. The script reads ``robometer.data.dataset_category.DATASET_MAP``,
   resolves each concrete dataset key under
   ``<root>/<dataset_key>/processed_dataset/``, and concatenates all trajectories into **one**
   LeRobot dataset. Each frame gets ``robometer.data_source_id`` (categorical) encoding which
   subset (dataset key) the row came from; see ``robometer_categorical_maps.json``.

**Fixed fields** (every frame, under ``robometer.*``):

- ``is_robot``, ``quality_label_id``, ``partial_success``, ``num_frames`` (same defaults as before).
- ``data_source_id`` — **only in category merge mode**; int64, maps to the subset name.

Install: ``dataset_upload/requirements_lerobot_dataset_only.txt`` + ``pip install lerobot --no-deps``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_dataset_map() -> dict:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from robometer.data.dataset_category import DATASET_MAP
    except ImportError as e:
        print(
            "ERROR: cannot import robometer.data.dataset_category (need repo root on PYTHONPATH).\n"
            f"  {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return DATASET_MAP


def _print_header(title: str) -> None:
    bar = "=" * min(72, max(len(title) + 8, 40))
    print(f"\n{bar}\n  {title}\n{bar}")


def _flatten_dataset_map_entries(items: Any) -> list[str]:
    """Flatten DATASET_MAP group values (strings or nested lists of strings)."""
    if items is None:
        return []
    out: list[str] = []
    for x in items:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, list):
            out.extend(_flatten_dataset_map_entries(x))
    return out


def _ordered_unique(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _split_list_arg(split: str) -> list[str]:
    if split == "both":
        return ["train", "eval"]
    return [split]


def _resolve_category_subset_paths(
    category: str, split: str, processed_root: Path, dataset_map: dict
) -> list[tuple[str, Path]]:
    if category not in dataset_map:
        print(f"ERROR: unknown --dataset-category {category!r} (not in DATASET_MAP)", file=sys.stderr)
        raise SystemExit(1)
    keys: list[str] = []
    entry = dataset_map[category]
    for sp in _split_list_arg(split):
        group = entry.get(sp)
        if not group:
            print(f"  Note: category {category!r} has no {sp!r} split in DATASET_MAP; skipping.")
            continue
        keys.extend(_flatten_dataset_map_entries(group))
    keys = _ordered_unique(keys)
    if not keys:
        print(f"ERROR: no dataset keys resolved for {category!r} with split={split!r}", file=sys.stderr)
        raise SystemExit(1)

    out: list[tuple[str, Path]] = []
    for k in keys:
        p = (processed_root / k / "processed_dataset").resolve()
        if p.is_dir():
            out.append((k, p))
        else:
            print(f"  WARNING: missing processed_dataset, skipping subset {k!r}: {p}")
    if not out:
        print("ERROR: no existing processed_dataset directories found for this category.", file=sys.stderr)
        raise SystemExit(1)
    print(f"  Merging {len(out)} subset(s) for category {category!r} (split={split!r}):")
    for k, p in out:
        print(f"    - {k}  →  {p}")
    return out


def _resolve_media_path(video_root: Path, rel: str) -> Path:
    """Join ``video_root`` with relative ``frames`` path from the HF row.

    Preprocessed rows sometimes store ``processed_datasets/<dataset_key>/frames/...`` while
    callers pass ``--video-root .../processed_datasets``, which would double that prefix;
    we try fallbacks so both layouts work.
    """
    r = rel.strip()
    while r.startswith("./"):
        r = r[2:]

    vr = video_root.expanduser().resolve()
    candidates: list[Path] = []

    def _add(p: Path) -> None:
        p = p.resolve()
        if p not in candidates:
            candidates.append(p)

    _add(vr / r)
    if r.startswith("processed_datasets/"):
        stripped = r[len("processed_datasets/") :].lstrip("/")
        if stripped:
            _add(vr / stripped)
        if vr.name == "processed_datasets":
            _add(vr.parent / r)

    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _load_npz_frames(path: Path) -> np.ndarray:
    with np.load(path) as z:
        if "frames" not in z:
            raise ValueError(f"NPZ at {path} has no 'frames' array (keys: {list(z.files)})")
        return np.asarray(z["frames"])


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as e:
        print(
            "ERROR: `lerobot` not importable. Try:\n"
            "  uv pip install -r dataset_upload/requirements_lerobot_dataset_only.txt\n"
            "  uv pip install 'lerobot>=0.4' --no-deps\n"
            f"{e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _open_video_reader(video_path: Path):
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu(0))
        return vr, "decord"
    except Exception as decord_err:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path} (decord: {decord_err!s})") from decord_err

        class _Cv2Reader:
            def __init__(self, c):
                self._cap = c
                self._n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))

            def __len__(self):
                return max(self._n, 0)

            def __getitem__(self, i: int):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = self._cap.read()
                if not ok or bgr is None:
                    raise IndexError(f"Frame {i} missing in {video_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return np.asarray(rgb, dtype=np.uint8)

        return _Cv2Reader(cap), "opencv"


def _quality_label_map_union(hf_datasets: list[Any]) -> dict[str, int]:
    seen: set[str] = set()
    for hf_ds in hf_datasets:
        if "quality_label" not in hf_ds.column_names:
            continue
        for i in range(len(hf_ds)):
            v = hf_ds[i]["quality_label"]
            if v is not None:
                seen.add(str(v))
    return {s: idx for idx, s in enumerate(sorted(seen))}


def _data_source_map_from_keys(keys: list[str]) -> dict[str, int]:
    return {k: idx for idx, k in enumerate(sorted(keys))}


def _iter_frame_indices(n_total: int, max_frames: int) -> list[int]:
    if max_frames < 0 or max_frames >= n_total:
        return list(range(n_total))
    if max_frames == 0:
        return []
    if max_frames == 1:
        return [0]
    return [int(round(i * (n_total - 1) / (max_frames - 1))) for i in range(max_frames)]


def _lerobot_feature_spec(video_key: str, h: int, w: int, *, include_data_source: bool) -> dict:
    features: dict = {
        video_key: {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["height", "width", "channels"],
        },
        "robometer.is_robot": {"dtype": "float32", "shape": (1,), "names": None},
        "robometer.quality_label_id": {"dtype": "int64", "shape": (1,), "names": None},
        "robometer.partial_success": {"dtype": "float32", "shape": (1,), "names": None},
        "robometer.num_frames": {"dtype": "int64", "shape": (1,), "names": None},
    }
    if include_data_source:
        features["robometer.data_source_id"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def _append_fixed_tabular(
    frame_dict: dict,
    row: dict,
    quality_map: dict[str, int],
    *,
    data_source_map: dict[str, int] | None,
    data_source_key: str | None,
) -> None:
    v = row.get("is_robot")
    frame_dict["robometer.is_robot"] = np.array([1.0 if v is not None and bool(v) else 0.0], dtype=np.float32)

    ql = row.get("quality_label")
    if ql is None or not quality_map:
        qid = -1
    else:
        qid = int(quality_map.get(str(ql), -1))
    frame_dict["robometer.quality_label_id"] = np.array([qid], dtype=np.int64)

    ps = row.get("partial_success")
    frame_dict["robometer.partial_success"] = np.array([0.0 if ps is None else float(ps)], dtype=np.float32)

    nf = row.get("num_frames")
    frame_dict["robometer.num_frames"] = np.array([0 if nf is None else int(nf)], dtype=np.int64)

    if data_source_map is not None and data_source_key is not None:
        dsid = int(data_source_map.get(data_source_key, -1))
        frame_dict["robometer.data_source_id"] = np.array([dsid], dtype=np.int64)


def _probe_first_trajectory(hf_ds: Any, video_root: Path) -> tuple[Path, bool, int, int, str, Any]:
    """Return probe_path, use_npz, h, w, backend_name, vr_or_none."""
    if len(hf_ds) == 0:
        raise ValueError("empty dataset")
    first_rel = hf_ds[0]["frames"]
    if not isinstance(first_rel, str):
        raise TypeError(f"`frames` must be a string path, got {type(first_rel)}")
    probe_path = _resolve_media_path(video_root, first_rel)
    if not probe_path.is_file():
        raise FileNotFoundError(str(probe_path))

    vr = None
    use_npz = probe_path.suffix.lower() == ".npz"
    if use_npz:
        stack = _load_npz_frames(probe_path)
        if stack.ndim != 4 or stack.shape[-1] != 3:
            raise ValueError(f"NPZ frames shape (T,H,W,3) expected, got {stack.shape}")
        n_frames_probe = int(stack.shape[0])
        h, w = int(stack.shape[1]), int(stack.shape[2])
        backend = "npz"
    else:
        vr, backend = _open_video_reader(probe_path)
        n_frames_probe = len(vr)
        sample = vr[0]
        arr = sample.asnumpy() if hasattr(sample, "asnumpy") else np.asarray(sample)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"frame HxWx3 expected, got {arr.shape}")
        h, w = arr.shape[0], arr.shape[1]

    _ = n_frames_probe  # logged by caller
    return probe_path, use_npz, h, w, backend, vr


def _infer_fps(probe_path: Path, use_npz: bool, backend: str, vr: Any, override: float | None) -> float:
    if override is not None:
        return max(float(override), 1e-3)
    if use_npz:
        return 10.0
    if backend == "opencv":
        import cv2

        cap = cv2.VideoCapture(str(probe_path))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(vfps) if vfps and vfps > 1e-3 else 10.0
    try:
        return float(vr.get_avg_fps()) if vr is not None else 10.0
    except Exception:
        return 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Robometer HF → LeRobot v3.0 (+ optional Hub push).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--robometer-dataset",
        type=Path,
        default=None,
        help="Single ``load_from_disk`` directory (…/processed_dataset).",
    )
    src.add_argument(
        "--dataset-category",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Merge all subsets listed under this key in robometer.data.dataset_category.DATASET_MAP "
            "(e.g. rbm-1m-ood). Requires --processed-datasets-root."
        ),
    )
    parser.add_argument(
        "--processed-datasets-root",
        type=Path,
        default=None,
        help=(
            "Parent of per-dataset folders: <root>/<DATASET_MAP key>/processed_dataset/. "
            "Required with --dataset-category."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "eval", "both"),
        default="eval",
        help="Which DATASET_MAP split(s) to include when using --dataset-category (default: eval).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help=(
            "Resolves relative ``frames`` paths from each HF row. Use the directory that makes "
            "``(video-root)/(relative frames path)`` valid; if rows use ``processed_datasets/...`` "
            "prefixes, duplicate segments under ``.../processed_datasets/processed_datasets/`` are "
            "handled automatically."
        ),
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output v3 tree; absent unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--video-key", type=str, default="observation.images.main")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=2, help="Max trajectories to consider (-1 = all).")
    parser.add_argument("--max-frames-per-episode", type=int, default=32)
    parser.add_argument("--vcodec", type=str, default="h264")
    parser.add_argument("--robot-type", type=str, default=None)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload-large-folder", action="store_true")
    args = parser.parse_args()

    category_mode = args.dataset_category is not None
    if category_mode:
        if args.processed_datasets_root is None:
            print("ERROR: --processed-datasets-root is required with --dataset-category.", file=sys.stderr)
            raise SystemExit(1)
    else:
        if args.processed_datasets_root is not None:
            print("ERROR: --processed-datasets-root only applies with --dataset-category.", file=sys.stderr)
            raise SystemExit(1)
        if args.robometer_dataset is None:
            print("ERROR: pass --robometer-dataset or --dataset-category.", file=sys.stderr)
            raise SystemExit(1)

    _print_header("Robometer → LeRobot v3.0")
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from datasets import load_from_disk

    video_root = args.video_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if not video_root.is_dir():
        print(f"ERROR: not a directory: {video_root}", file=sys.stderr)
        raise SystemExit(1)

    loaded: list[tuple[str | None, Any]] = []
    data_source_map: dict[str, int] | None = None
    subset_keys: list[str] = []

    if category_mode:
        dataset_map = _import_dataset_map()
        processed_root = args.processed_datasets_root.expanduser().resolve()
        if not processed_root.is_dir():
            print(f"ERROR: not a directory: {processed_root}", file=sys.stderr)
            raise SystemExit(1)
        pairs = _resolve_category_subset_paths(
            args.dataset_category, args.split, processed_root, dataset_map
        )
        subset_keys = [k for k, _ in pairs]
        data_source_map = _data_source_map_from_keys(subset_keys)
        for subset_name, ds_path in pairs:
            loaded.append((subset_name, load_from_disk(str(ds_path))))
    else:
        ds_path = args.robometer_dataset.expanduser().resolve()
        if not ds_path.is_dir():
            print(f"ERROR: not a directory: {ds_path}", file=sys.stderr)
            raise SystemExit(1)
        loaded.append((None, load_from_disk(str(ds_path))))

    all_ds = [hf for _, hf in loaded]
    for _, hf_ds in loaded:
        print(f"  Loaded rows={len(hf_ds)}  columns={hf_ds.column_names}")
        if {"task", "frames"} - set(hf_ds.column_names):
            print("ERROR: each dataset must include `task` and `frames`.", file=sys.stderr)
            raise SystemExit(1)
        for name in ("is_robot", "quality_label", "partial_success", "num_frames"):
            if name not in hf_ds.column_names:
                print(f"  Note: column {name!r} missing in one subset — defaults used in robometer.*")

    quality_map = _quality_label_map_union(all_ds)
    if quality_map:
        print(f"  quality_label: {len(quality_map)} distinct labels (union over subsets)")

    probe_path = None
    use_npz = False
    h = w = 0
    backend = ""
    vr = None
    last_err: Exception | None = None
    for sub, hf_try in loaded:
        if len(hf_try) == 0:
            print(f"  Note: subset {sub!r} is empty; skipping for probe.")
            continue
        try:
            probe_path, use_npz, h, w, backend, vr = _probe_first_trajectory(hf_try, video_root)
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"  Note: probe failed on subset {sub!r}: {e}")
    if probe_path is None:
        print(f"ERROR: could not probe media from any subset (last error: {last_err})", file=sys.stderr)
        raise SystemExit(1)

    fps = _infer_fps(probe_path, use_npz, backend, vr, args.fps)
    fps_int = max(1, int(round(fps)))
    print(f"  Media: {backend}  probe={probe_path.name}  {h}x{w}  fps={fps_int}")

    features = _lerobot_feature_spec(args.video_key, h, w, include_data_source=category_mode)
    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            print(f"ERROR: {out_root} exists (use --overwrite).", file=sys.stderr)
            raise SystemExit(1)

    out_root.parent.mkdir(parents=True, exist_ok=True)
    lerobot_ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps_int,
        features=features,
        root=out_root,
        robot_type=args.robot_type,
        use_videos=True,
        vcodec=args.vcodec,
    )

    episodes_written = 0
    frames_written = 0
    rows_seen = 0
    max_rows = None if args.max_episodes < 0 else args.max_episodes

    for subset_name, hf_ds in loaded:
        for i in range(len(hf_ds)):
            if max_rows is not None and rows_seen >= max_rows:
                break
            rows_seen += 1

            row = hf_ds[i]
            rel = row["frames"]
            if not isinstance(rel, str):
                continue
            vpath = _resolve_media_path(video_root, rel)
            if not vpath.is_file():
                print(f"  [{subset_name or 'single'}:{i}] skip missing {vpath}")
                continue
            if (vpath.suffix.lower() == ".npz") != use_npz:
                print(f"  [{subset_name or 'single'}:{i}] skip media type mismatch")
                continue

            task = str(row["task"]) if not isinstance(row["task"], str) else row["task"]

            if use_npz:
                try:
                    stack = _load_npz_frames(vpath)
                except Exception as e:
                    print(f"  [{subset_name or 'single'}:{i}] skip npz {e}")
                    continue
                if stack.ndim != 4 or stack.shape[-1] != 3:
                    continue
                if int(stack.shape[1]) != h or int(stack.shape[2]) != w:
                    print(
                        f"  [{subset_name or 'single'}:{i}] skip shape {stack.shape[1]}x{stack.shape[2]} "
                        f"!= probe {h}x{w}"
                    )
                    continue
                n_fr = int(stack.shape[0])
                vback = "npz"
            else:
                try:
                    vreader, vback = _open_video_reader(vpath)
                except Exception as e:
                    print(f"  [{subset_name or 'single'}:{i}] skip open {e}")
                    continue
                n_fr = len(vreader)

            indices = _iter_frame_indices(n_fr, args.max_frames_per_episode)
            if not indices:
                continue
            print(f"  ep {episodes_written:04d}  [{subset_name or 'single'}] {vpath.name}  {len(indices)}/{n_fr}  ({vback})")

            frames_this = 0
            for fi in indices:
                try:
                    if use_npz:
                        img = np.asarray(stack[fi])
                    else:
                        fr = vreader[fi]
                        img = fr.asnumpy() if hasattr(fr, "asnumpy") else np.asarray(fr)
                except Exception as e:
                    print(f"    frame {fi}: {e}")
                    continue
                if img.shape[0] != h or img.shape[1] != w:
                    print(f"    skip frame {fi}: shape {img.shape[:2]} != {h}x{w}")
                    continue
                fd: dict = {"task": task, args.video_key: img}
                _append_fixed_tabular(
                    fd,
                    row,
                    quality_map,
                    data_source_map=data_source_map,
                    data_source_key=subset_name,
                )
                lerobot_ds.add_frame(fd)
                frames_written += 1
                frames_this += 1

            if frames_this:
                lerobot_ds.save_episode()
                episodes_written += 1
        if max_rows is not None and rows_seen >= max_rows:
            break

    lerobot_ds.finalize()

    maps_payload: dict[str, Any] = {}
    if quality_map:
        maps_payload["quality_label"] = {
            "lerobot_key": "robometer.quality_label_id",
            "id_to_label": {str(i): lab for lab, i in quality_map.items()},
        }
    if category_mode and data_source_map:
        maps_payload["data_source"] = {
            "lerobot_key": "robometer.data_source_id",
            "id_to_label": {str(i): lab for lab, i in data_source_map.items()},
        }
    if maps_payload:
        p = out_root / "robometer_categorical_maps.json"
        with open(p, "w") as f:
            json.dump(maps_payload, f, indent=2)
        print(f"  Wrote {p}")

    print(f"  Done: {episodes_written} episodes, {frames_written} frames → {out_root}")
    if args.push:
        if not os.environ.get("HF_TOKEN"):
            print("ERROR: HF_TOKEN required for --push", file=sys.stderr)
            raise SystemExit(1)
        lerobot_ds.push_to_hub(private=args.private, upload_large_folder=args.upload_large_folder)
        print(f"  Pushed https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
