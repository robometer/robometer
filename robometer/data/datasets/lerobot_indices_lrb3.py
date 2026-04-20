"""Build ``combined_indices`` from lightweight per-episode metadata rows (no frames loaded)."""

from __future__ import annotations

from typing import Any


def build_combined_indices_from_metadata_rows(rows: list[dict]) -> dict[str, Any]:
    """Same structure as ``index_mappings.json`` from preprocessing (trajectory index = row index)."""
    robot_trajectories: list[int] = []
    human_trajectories: list[int] = []
    optimal_by_task: dict[str, list[int]] = {}
    suboptimal_by_task: dict[str, list[int]] = {}
    quality_indices: dict[str, list[int]] = {}
    task_indices: dict[str, list[int]] = {}
    source_indices: dict[str, list[int]] = {}
    partial_success_indices: dict[Any, list[int]] = {}

    for new_idx, ex in enumerate(rows):
        if ex.get("is_robot", True):
            robot_trajectories.append(new_idx)
        else:
            human_trajectories.append(new_idx)

        quality = ex.get("quality_label", "successful")
        quality_indices.setdefault(str(quality), []).append(new_idx)

        task = ex.get("task", "unknown")
        task_indices.setdefault(str(task), []).append(new_idx)

        source = ex.get("data_source", "unknown")
        source_indices.setdefault(str(source), []).append(new_idx)

        partial_success = ex.get("partial_success", None)
        if partial_success is not None and quality == "failure":
            partial_success_indices.setdefault(partial_success, []).append(new_idx)

        if task not in optimal_by_task:
            optimal_by_task[task] = []
            suboptimal_by_task[task] = []
        if quality in ["successful", "optimal"]:
            optimal_by_task[task].append(new_idx)
        elif quality in ["suboptimal", "failed", "failure"]:
            suboptimal_by_task[task].append(new_idx)

    return {
        "robot_trajectories": robot_trajectories,
        "human_trajectories": human_trajectories,
        "optimal_by_task": optimal_by_task,
        "suboptimal_by_task": suboptimal_by_task,
        "quality_indices": quality_indices,
        "task_indices": task_indices,
        "source_indices": source_indices,
        "partial_success_indices": partial_success_indices,
    }
