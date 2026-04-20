#!/usr/bin/env python3
"""
Load a confusion matrix from .npy and generate a PNG with purple colormap.
Metrics match compile_results.run_confusion_matrix_eval (trace, off_diagonal, etc.).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix_metrics(confusion_matrix: np.ndarray) -> dict:
    """Compute metrics as in compile_results.run_confusion_matrix_eval."""
    n = confusion_matrix.shape[0]
    trace = float(np.trace(confusion_matrix))
    total_sum = float(np.sum(confusion_matrix))
    off_diag_sum = total_sum - trace
    trace_minus_offdiag = trace - off_diag_sum
    avg_diagonal = trace / n if n > 0 else 0.0
    avg_off_diag = off_diag_sum / (n * n - n) if n > 1 else 0.0
    normalized_metric = avg_diagonal - avg_off_diag
    return {
        "trace": trace,
        "off_diagonal_sum": off_diag_sum,
        "trace_minus_offdiag": trace_minus_offdiag,
        "avg_diagonal": avg_diagonal,
        "avg_off_diagonal": avg_off_diag,
        "normalized_trace_minus_offdiag": normalized_metric,
    }


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    out_path: Path,
    cmap: str = "Purples",
    figsize: tuple = (8, 8),
    cbar: bool = False,
) -> None:
    """Plot confusion matrix heatmap (style aligned with compile_results.run_confusion_matrix_eval)."""
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        cmap=cmap,
        cbar=cbar,
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Load confusion matrix .npy and save purple PNG + metrics")
    parser.add_argument(
        "npy_path",
        type=Path,
        nargs="?",
        default=Path("baseline_eval_output/robodopamine_tanhuajie2001_Robo-Dopamine-GRM-2_0-8B-Preview/confusion_matrix/usc_franka_utd_so101_clean_top_usc_xarm_confusion_matrix.npy"),
        help="Path to confusion matrix .npy file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: same dir as .npy, same stem with _purple.png)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON",
    )
    parser.add_argument(
        "--cmap",
        default="Purples",
        help="Matplotlib colormap name (default: Purples)",
    )
    args = parser.parse_args()

    npy_path = args.npy_path
    if not npy_path.exists():
        raise FileNotFoundError(f"Confusion matrix not found: {npy_path}")

    confusion_matrix = np.load(npy_path)
    if confusion_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {confusion_matrix.shape}")

    # Output PNG
    if args.output is not None:
        out_png = args.output
    else:
        out_png = npy_path.with_stem(npy_path.stem + "_purple").with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(confusion_matrix, out_png, cmap=args.cmap)
    print(f"Saved: {out_png}")

    # Metrics (same as compile_results.run_confusion_matrix_eval)
    metrics = compute_confusion_matrix_metrics(confusion_matrix)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        # Use same key style as baseline eval metrics.json (task/trace, task/off_diagonal_sum, ...)
        task_key = npy_path.stem.replace("_confusion_matrix", "").replace("_", "/")
        flat = {f"{task_key}/{k}": v for k, v in metrics.items()}
        with open(args.metrics, "w") as f:
            json.dump(flat, f, indent=2)
        print(f"Metrics saved: {args.metrics}")


if __name__ == "__main__":
    main()
