#!/usr/bin/env python
"""Builds the dense (one-value-per-frame) "continuous" annotation format -- same schema as the
reference-side `zs_robodopamine_icl_dataset_continuous.zip` (see VFE/) -- for the two query-side
(icl-demo-dataset) reward-model signals that are themselves already continuous-valued (not sparse
discrete key-events): non-causal (future-leaking) zero-shot Robo-Dopamine GRM progress
(`noncausal_progress.json`, ~every 10th frame) and fine-tuned Robometer progress
(`robometer_progress.json`, ~32 uniformly-subsampled points). Both are sparse in *frame coverage*
only -- fills in every frame in between via sigmoid interpolation, so the output is synced
frame-for-frame with each episode (one point per frame_index, 0..num_frames-1).

Sigmoid interpolation between consecutive sparse points (frame_a, value_a) -> (frame_b, value_b):
    t = (frame - frame_a) / (frame_b - frame_a)                        # in [0, 1]
    raw(x) = 1 / (1 + exp(-k * (2x - 1)))                              # logistic, centered at x=0.5
    s = (raw(t) - raw(0)) / (raw(1) - raw(0))                          # renormalized so s(0)=0, s(1)=1
    value(frame) = value_a + (value_b - value_a) * s
Frames before the first sparse point or after the last hold constant at that endpoint's value
(neither source is expected to need this -- both start at frame 0 and end at num_frames-1 in
practice -- but it's handled for robustness). k=1 by default (--steepness).

Output: one JSON file per episode (same 10-key schema as the reference dense annotations:
episode_uid, episode_index, task, success, final_progress, fps, num_frames, interpolation,
steepness, points), written under `<output-dir>/<zip-basename>/<episode_uid with ':' -> '__'>.json`,
then zipped to `<zip-basename>.zip` (folder entry included, matching the reference zip's layout).

Usage:
    python build_continuous_annotations.py --source noncausal --steepness 1.0
    python build_continuous_annotations.py --source robometer --steepness 1.0
    python build_continuous_annotations.py --source robometer_online --steepness 1.0
    python build_continuous_annotations.py --source noncausal --source robometer  # multiple sources; omit --source for all (default)
"""
import argparse
import json
import math
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_QUERY_CACHE = HERE / "query_episode_cache"
DEFAULT_INFO_JSON = HERE / ".lerobot_cache" / "adityx23" / "icl-demo-dataset" / "meta" / "info.json"

# source name -> (progress filename in query_episode_cache/episode_*/, output zip basename)
SOURCES = {
    "noncausal": ("noncausal_progress.json", "zs_robodopamine_icl_demo_dataset_continuous"),
    "robometer": ("robometer_progress.json", "finetuned_robometer_icl_demo_dataset_continuous"),
    "robometer_online": ("online_robometer_progress.json", "finetuned_robometer_online_icl_demo_dataset_continuous"),
    "robometer_online_zeroshot": ("online_robometer_zeroshot_progress.json", "zeroshot_robometer_online_icl_demo_dataset_continuous"),
    "robometer_zeroshot": ("robometer_zeroshot_progress.json", "zeroshot_robometer_icl_demo_dataset_continuous"),
}


def safe_filename(episode_uid: str) -> str:
    return episode_uid.replace(":", "__").replace("/", "_")


def sigmoid_interpolate(sparse_points: list[dict], num_frames: int, k: float) -> list[float]:
    """sparse_points: [{"frame": int, "value": float}, ...] sorted ascending by frame."""
    frames = [p["frame"] for p in sparse_points]
    values = [p["value"] for p in sparse_points]

    def raw(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-k * (2 * x - 1)))

    denom = raw(1.0) - raw(0.0)

    out = [0.0] * num_frames
    seg = 0  # index of the sparse point at/just before the current output frame
    for f in range(num_frames):
        while seg + 1 < len(frames) and frames[seg + 1] <= f:
            seg += 1
        if f <= frames[0]:
            out[f] = values[0]
        elif f >= frames[-1]:
            out[f] = values[-1]
        else:
            frame_a, frame_b = frames[seg], frames[seg + 1]
            value_a, value_b = values[seg], values[seg + 1]
            t = (f - frame_a) / (frame_b - frame_a) if frame_b > frame_a else 0.0
            s = (raw(t) - raw(0.0)) / denom
            out[f] = value_a + (value_b - value_a) * s
    return out


def build_one(manifest: dict, sparse_points: list[dict], fps: float, k: float) -> dict:
    num_frames = manifest["num_frames"]
    dense = sigmoid_interpolate(sparse_points, num_frames, k)
    points = [
        {"frame_index": i, "timestamp": round(i / fps, 4), "progress": round(dense[i] * 100.0, 1)}
        for i in range(num_frames)
    ]
    return {
        "episode_uid": manifest["episode_uid"],
        "episode_index": manifest["episode_index"],
        "task": manifest["task"],
        "success": "success" if manifest.get("success") else "fail",
        "final_progress": round(dense[-1], 4),
        "fps": fps,
        "num_frames": num_frames,
        "interpolation": "sigmoid",
        "steepness": k,
        "points": points,
    }


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{src_dir.name}/", "")
        for f in sorted(src_dir.glob("*.json")):
            zf.write(f, arcname=f"{src_dir.name}/{f.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", choices=list(SOURCES), default=None,
                         help="Repeatable. Defaults to both noncausal and robometer.")
    parser.add_argument("--query-cache-dir", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--steepness", type=float, default=1.0, help="Sigmoid k.")
    args = parser.parse_args()
    sources = args.source or list(SOURCES)

    fps = float(json.loads(DEFAULT_INFO_JSON.read_text(encoding="utf-8"))["fps"])

    for source in sources:
        progress_fname, zip_basename = SOURCES[source]
        out_dir = args.output_dir / zip_basename
        out_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        skipped = 0
        for ep_dir in sorted(args.query_cache_dir.glob("episode_*")):
            manifest_path = ep_dir / "manifest.json"
            progress_path = ep_dir / progress_fname
            if not manifest_path.exists() or not progress_path.exists():
                skipped += 1
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sparse_points = json.loads(progress_path.read_text(encoding="utf-8"))
            payload = build_one(manifest, sparse_points, fps, args.steepness)
            out_path = out_dir / f"{safe_filename(manifest['episode_uid'])}.json"
            out_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            written += 1

        zip_path = args.output_dir / f"{zip_basename}.zip"
        zip_dir(out_dir, zip_path)
        print(f"[build_continuous_annotations] {source}: {written} episodes written "
              f"({skipped} skipped, missing {progress_fname}) -> {zip_path}")


if __name__ == "__main__":
    main()
