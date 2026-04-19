#!/usr/bin/env python3
"""
Script to compile evaluation results from JSON files.
"""

import json
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from robometer.data.datasets.helpers import load_frames_from_npz
from robometer.evals.eval_metrics_utils import compute_pearson, compute_spearman, compute_kendall
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot
from robometer.models.utils import convert_bins_to_continuous, convert_discrete_target_to_continuous


def convert_continuous_to_discrete_bin_roboreward(value: float, num_bins: int) -> int:
    value = min(max(value, 0.0), 1.0)
    return round(value * (num_bins - 1))


def run_quality_preference_eval(results: List[Dict[str, Any]], data_source: Optional[str] = None) -> Dict[str, Any]:
    """Run quality_preference evaluation analysis.

    Groups results by task and quality labels (or partial_success for RoboArena),
    computes preference accuracy per group and aggregate.
    Returns metrics, task_groups, and task_details similar to policy_ranking.
    """
    # Check if data_source contains roboreward or roboarena to determine if we should use partial_success logic
    use_partial_success = "roboreward" in str(data_source).lower() or "roboarena" in str(data_source).lower()

    # First, gather all predictions and labels, convert to arrays
    # Note: preference_pred is already binary (0/1) from the trainer
    all_preds = []
    all_labels = []
    all_tasks = []
    all_quality_combos = []
    valid_indices = []

    for idx, r in enumerate(results):
        pred = r.get("preference_pred")
        label = r.get("preference_labels")
        if pred is not None and label is not None:
            pred = float(pred.item()) if pred.size == 1 else float(pred[0])
            label = float(label.item()) if label.size == 1 else float(label[0])

            # For datasets without partial_success, extract quality combo; for datasets with partial_success, just validate metadata exists
            chosen_meta = r.get("metadata", {}).get("chosen_metadata", {})
            rejected_meta = r.get("metadata", {}).get("rejected_metadata", {})

            if use_partial_success:
                # For datasets with partial_success, just check that partial_success exists (we don't use it)
                chosen_val = chosen_meta.get("partial_success")
                rejected_val = rejected_meta.get("partial_success")
                if chosen_val is None or rejected_val is None:
                    continue
            else:
                # For datasets without partial_success, extract quality combo for later use
                chosen_val = chosen_meta.get("quality_label")
                rejected_val = rejected_meta.get("quality_label")
                if chosen_val is None or rejected_val is None:
                    continue
                combo_key = tuple(sorted([chosen_val, rejected_val]))
                all_quality_combos.append(combo_key)

            all_preds.append(pred)
            all_labels.append(label)
            all_tasks.append(r["task"])
            valid_indices.append(idx)

    if not all_preds:
        return {"error": "No valid predictions found"}, {}, {}

    # Convert to numpy arrays for vectorized operations
    # preference_pred is already binary (0/1), so no sigmoid conversion needed
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = all_preds.astype(float)

    # Group results by task (using valid indices to map back)
    task_groups = defaultdict(list)
    task_indices = defaultdict(list)
    for i, (idx, task) in enumerate(zip(valid_indices, all_tasks)):
        task_groups[task].append(results[idx])
        task_indices[task].append(i)

    # Compute preference accuracy per task group using vectorized operations
    task_details = {}
    all_correct = 0
    all_total = 0

    for task, task_results in task_groups.items():
        task_idx = task_indices[task]
        task_preds = binary_preds[task_idx]
        task_labels = all_labels[task_idx]

        task_correct = np.sum(task_preds == task_labels)
        task_total = len(task_preds)
        pref_acc = task_correct / task_total if task_total > 0 else 0.0

        task_detail = {
            "preference_accuracy": pref_acc,
            "num_correct": int(task_correct),
            "num_total": task_total,
        }

        # Only compute quality accuracies for datasets without partial_success
        if not use_partial_success:
            # Compute accuracy per quality combination using vectorized operations
            task_quality_combos = [all_quality_combos[i] for i in task_idx]
            quality_accs = {}
            unique_combos = set(task_quality_combos)
            for combo_key in unique_combos:
                combo_mask = np.array([qc == combo_key for qc in task_quality_combos])
                combo_preds = task_preds[combo_mask]
                combo_labels = task_labels[combo_mask]
                if len(combo_preds) > 0:
                    combo_correct = np.sum(combo_preds == combo_labels)
                    combo_acc = combo_correct / len(combo_preds) if len(combo_preds) > 0 else 0.0
                    quality_accs[f"{combo_key[0]}_vs_{combo_key[1]}"] = combo_acc
            task_detail["quality_accuracies"] = quality_accs
        else:
            task_detail["quality_accuracies"] = None

        task_details[task] = task_detail

        all_correct += task_correct
        all_total += task_total

    # Aggregate metrics
    aggregate_acc = all_correct / all_total if all_total > 0 else 0.0

    metrics = {
        "preference_accuracy": aggregate_acc,
    }

    return metrics, task_groups, task_details


def run_reward_alignment_eval_per_trajectory(
    results: List[Dict[str, Any]],
    progress_pred_type: str,
    is_discrete_mode: bool,
    num_bins: int,
    data_source: Optional[str],
    use_frame_steps: bool,
    train_success_head: bool,
    last_frame_only: bool = False,
) -> Tuple[Dict[str, Any], List, List, List]:
    """Run reward_alignment evaluation analysis and create plots for each trajectory.

    For failure datasets, we visualize predictions but skip metric computation.

    Args:
        use_frame_steps: If True, expects multiple results per trajectory (frame_steps mode).
                         If False, expects exactly one result per trajectory (whole trajectory mode).
        train_success_head: Whether the success head is being trained (determines if success predictions exist).

    Returns:
        Tuple of (metrics, plots, video_frames_list, trajectory_progress_data)
        where trajectory_progress_data is a list of progress_pred values
        for each trajectory (one per video in video_frames_list)
    """
    # Check if data_source contains roboreward or roboarena to determine if we should use partial_success logic
    use_partial_success = "roboreward" in str(data_source).lower() or "roboarena" in str(data_source).lower()

    # Check if this is RoboReward (needs MAE metric)
    is_roboreward = data_source and "roboreward" in str(data_source).lower()

    unique_trajectory_ids = set()

    metrics = {}

    # Collect all success_probs and success_labels for AUPRC computation
    all_success_probs = []
    all_success_labels = []
    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)

        # Collect success probabilities and labels for AUPRC
        if train_success_head:
            success_probs = r["success_probs"]
            success_labels = r["success_labels"]
            all_success_probs.append(success_probs)
            all_success_labels.append(success_labels)

    # Compute success_auprc across all collected success predictions and labels
    if all_success_probs and all_success_labels:
        # Flatten all collected probabilities and labels
        success_probs_flat = np.concatenate(all_success_probs)
        success_labels_flat = np.concatenate(all_success_labels)

        # Compute AUPRC if we have valid data
        if success_probs_flat.size > 0 and len(np.unique(success_labels_flat)) > 1:
            success_auprc = float(average_precision_score(success_labels_flat, success_probs_flat))
        else:
            success_auprc = 0.0

        # Compute positive and negative accuracy
        if success_probs_flat.size > 0:
            # Convert probabilities to binary predictions (threshold at 0.5)
            success_preds_flat = (success_probs_flat > 0.5).astype(float)

            # Compute accuracy for positive samples (where label == 1)
            positive_mask = success_labels_flat == 1
            num_positives = positive_mask.sum()
            if num_positives > 0:
                positive_correct = ((success_preds_flat == success_labels_flat) & positive_mask).sum()
                positive_success_acc = float(positive_correct / num_positives)

            # Compute accuracy for negative samples (where label == 0)
            negative_mask = success_labels_flat == 0
            num_negatives = negative_mask.sum()
            if num_negatives > 0:
                negative_correct = ((success_preds_flat == success_labels_flat) & negative_mask).sum()
                negative_success_acc = float(negative_correct / num_negatives)

        metrics["success_auprc"] = success_auprc
        metrics["positive_success_acc"] = positive_success_acc
        metrics["negative_success_acc"] = negative_success_acc

    loss_per_trajectory = np.zeros(1)
    loss_trajectories = []
    pearson_trajectories = []
    plots = []
    video_frames_list = []
    trajectory_progress_data = []

    # Collect absolute deltas between final reward and partial_success for trajectories with partial_success
    partial_success_deltas = []

    # Collect bins for MAE computation (RoboReward)
    pred_bins_mae = []
    gt_bins_mae = []

    for trajectory_id in unique_trajectory_ids:
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]

        # Assert that if use_frame_steps=False, each trajectory should have exactly 1 result
        if not use_frame_steps:
            assert len(results_for_trajectory) == 1, (
                f"Expected exactly 1 result per trajectory when use_frame_steps=False, "
                f"but found {len(results_for_trajectory)} results for trajectory_id={trajectory_id}"
            )

        # Sort by frame_step if available (for frame_steps mode)
        # This orders subsequences from shortest to longest (e.g., [0], [0,1], [0,1,2], ...)
        # Only sort if there are multiple results (indicating frame_steps mode)
        if len(results_for_trajectory) > 1:
            results_for_trajectory.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))

        # Get task and quality label from first result
        task = results_for_trajectory[0]["task"]
        quality_label = results_for_trajectory[0]["quality_label"]
        video_path = results_for_trajectory[0]["video_path"]
        partial_success = results_for_trajectory[0].get("partial_success")

        if is_discrete_mode and partial_success is not None:
            if isinstance(partial_success, torch.Tensor):
                # [num_bins] -> [1, 1, num_bins]
                partial_success_tensor = partial_success[
                    None, None
                ]  # to make it 3-dim for convert_discrete_target_to_continuous
            else:
                # number -> [1, 1]
                partial_success_tensor = torch.tensor([partial_success], dtype=torch.float32).unsqueeze(0)
            partial_success = convert_discrete_target_to_continuous(partial_success_tensor, num_bins=num_bins).item()

        # Step 1: Collect all progress predictions and targets
        raw_preds = []
        raw_targets = []
        traj_pred_logits = []  # For discrete mode: collect full logits
        traj_target_bins = []  # For discrete mode: collect bin indices
        original_sampled_indices = []

        if not use_frame_steps:
            # Whole trajectory mode: one result with full progress prediction
            r = results_for_trajectory[0]
            raw_preds.append(r["progress_pred"])
            raw_targets.append(r["target_progress"])
        else:
            # Frame steps mode: multiple results, one per subsequence
            for timestep, r in enumerate(results_for_trajectory):
                raw_preds.append(r["progress_pred"])
                raw_targets.append(r["target_progress"])
                original_sampled_indices.append(r.get("original_sampled_index"))

        # Step 2: Convert all progress predictions and targets to continuous (if discrete mode)
        # Store logits/bins for loss computation before conversion
        traj_preds_continuous = []
        traj_targets_continuous = []

        if not use_frame_steps:
            # Process single prediction/target
            pred_array = raw_preds[0]
            if is_discrete_mode:
                traj_pred_logits = pred_array
                # Convert to continuous
                continuous_preds = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).numpy()
                traj_preds_continuous.append(continuous_preds)
            else:
                traj_preds_continuous.append(pred_array)

            tgt_array = raw_targets[0]
            if is_discrete_mode:
                traj_target_bins = tgt_array
                # Convert to continuous
                continuous_targets = convert_discrete_target_to_continuous(
                    torch.tensor(tgt_array[None]), num_bins=num_bins
                )[0].numpy()
                traj_targets_continuous.append(continuous_targets)
            else:
                traj_targets_continuous.append(tgt_array)
        else:
            # Frame steps mode: process each timestep
            for timestep, (pred_array, tgt_array) in enumerate(zip(raw_preds, raw_targets)):
                # Process prediction
                if is_discrete_mode:
                    # Store logits for loss computation (store all logits from this timestep)
                    traj_pred_logits.append(pred_array)
                    # Convert to continuous
                    continuous_preds = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).numpy()
                    traj_preds_continuous.append(continuous_preds)
                else:
                    traj_preds_continuous.append(pred_array)

                # Process target
                if is_discrete_mode:
                    # Store target bins for loss computation (store all bins from this timestep)
                    traj_target_bins.append(tgt_array)
                    # Convert to continuous
                    continuous_targets = convert_discrete_target_to_continuous(
                        torch.tensor(tgt_array[None]), num_bins=num_bins
                    )[0].numpy()
                    traj_targets_continuous.append(continuous_targets)
                else:
                    traj_targets_continuous.append(tgt_array)

        # Step 3: Apply last_frame_only logic to continuous predictions/targets
        traj_preds = []
        traj_targets = []

        if not use_frame_steps:
            pred_array = traj_preds_continuous[0]
            traj_preds = pred_array.flatten()
            if last_frame_only:
                traj_preds = pred_array[-1:]

            tgt_array = traj_targets_continuous[0]
            traj_targets = tgt_array.flatten()
            if last_frame_only:
                traj_targets = tgt_array[-1:]
        else:
            # Frame steps mode
            for timestep, (pred_array, tgt_array, original_sampled_index) in enumerate(zip(traj_preds_continuous, traj_targets_continuous, original_sampled_indices)):
                if last_frame_only:
                    pred_val = pred_array[-1]
                    tgt_val = tgt_array[-1]
                elif original_sampled_index is not None:
                    pred_idx = min(int(original_sampled_index), len(pred_array) - 1)
                    tgt_idx = min(int(original_sampled_index), len(tgt_array) - 1)
                    pred_val = pred_array[pred_idx]
                    tgt_val = tgt_array[tgt_idx]
                else:
                    indx = min(timestep, len(pred_array) - 1)
                    pred_val = pred_array[indx]
                    indx = min(timestep, len(tgt_array) - 1)
                    tgt_val = tgt_array[indx]
                traj_preds.append(pred_val)
                traj_targets.append(tgt_val)

        # Step 4: Collect success predictions, labels, and probabilities separately
        traj_success_preds = []
        traj_success_labels = []
        traj_success_probs = []

        if not use_frame_steps:
            # Whole trajectory mode: process single result
            r = results_for_trajectory[0]
            if train_success_head:
                traj_success_preds = r["success_pred"].flatten()
                traj_success_labels = r["success_labels"].flatten()
                traj_success_probs = r["success_probs"].flatten()
        else:
            # Frame steps mode: process each result
            for r in results_for_trajectory:
                if train_success_head:
                    traj_success_preds.append(r["success_pred"][-1])
                    traj_success_labels.append(r["success_labels"][-1])
                    traj_success_probs.append(r["success_probs"][-1])

        # Convert to numpy arrays
        traj_preds = np.array(traj_preds)
        traj_targets = np.array(traj_targets)
        traj_success = np.array(traj_success_preds)
        traj_success_labels = np.array(traj_success_labels)
        traj_success_probs = np.array(traj_success_probs)

        # Load video frames if video path exists
        frames = None
        if video_path:
            frames = load_frames_from_npz(video_path)
            frames = frames.transpose(0, 3, 1, 2)

            # Resize frames to make them smaller for wandb table display
            resized_frames = []
            for frame in frames:
                frame_resized = cv2.resize(frame.transpose(1, 2, 0), (64, 64))
                resized_frames.append(frame_resized.transpose(2, 0, 1))
            frames = np.array(resized_frames)

        video_frames_list.append(frames)

        if progress_pred_type == "relative":
            traj_preds = np.cumsum(traj_preds)
            traj_targets = np.cumsum(traj_targets)

        trajectory_progress_data.append(traj_preds.tolist())

        # For trajectories with partial_success, compute absolute delta between final reward and partial_success
        if use_partial_success and partial_success is not None:
            final_reward = float(traj_preds[-1])
            delta = abs(final_reward - partial_success)
            partial_success_deltas.append(delta)

        # For RoboReward, collect bins for MAE computation
        if is_roboreward and partial_success is not None:
            # Get last predicted reward (final reward)
            final_predicted_reward = float(traj_preds[-1])

            # Convert predicted reward to bin (1-5)
            pred_bin = convert_continuous_to_discrete_bin_roboreward(final_predicted_reward, num_bins=5)

            # Convert partial_success to bin (0->1, 1->5)
            gt_bin = convert_continuous_to_discrete_bin_roboreward(partial_success, num_bins=5)

            pred_bins_mae.append(pred_bin)
            gt_bins_mae.append(gt_bin)

        # Only compute metrics for successful trajectories
        if quality_label == "successful":
            # Compute loss based on mode
            if is_discrete_mode and traj_pred_logits is not None and traj_target_bins is not None:
                # Discrete mode: compute cross-entropy loss between logits and target bins
                pred_logits_tensor = torch.tensor(
                    np.array(traj_pred_logits), dtype=torch.float32
                )  # [seq_len, num_bins]
                target_bins_tensor = torch.tensor(traj_target_bins)  # [seq_len, num_bins] or [seq_len]
                if len(target_bins_tensor.shape) == 1:
                    target_bins_tensor = target_bins_tensor.long()
                loss_per_timestep = F.cross_entropy(pred_logits_tensor, target_bins_tensor, reduction="none")
                traj_loss = float(loss_per_timestep.mean().item())
            else:
                # Continuous mode: compute MSE loss
                traj_loss = float(np.mean((traj_targets - traj_preds) ** 2))

            # Compute Pearson correlation
            traj_pearson = compute_pearson(traj_targets.tolist(), traj_preds.tolist())
            # Handle NaN values
            traj_pearson = float(traj_pearson) if not np.isnan(traj_pearson) else 0.0
        else:
            traj_loss = 0.0
            traj_pearson = 0.0

        # Create a wandb plot for progress predictions and, if available, success predictions
        # Use the shared helper function from eval_viz_utils
        # Limit to 10 plots to avoid creating too many
        if len(plots) < 10:
            has_success_binary = (
                train_success_head and traj_success is not None and len(traj_success) == len(traj_preds)
            )

            title = f"Task: {task} - {quality_label}\nLoss: {traj_loss:.3f}, pearson: {traj_pearson:.2f}"
            if partial_success is not None:
                title += f", partial_success: {partial_success:.3f}"

            fig = create_combined_progress_success_plot(
                progress_pred=traj_preds,
                num_frames=len(traj_preds),
                success_binary=traj_success if has_success_binary else None,
                success_probs=traj_success_probs if train_success_head else None,
                success_labels=traj_success_labels if train_success_head else None,
                is_discrete_mode=is_discrete_mode,
                title=title,
                loss=traj_loss,
                pearson=traj_pearson,
            )

            plots.append(fig)

        # Accumulate metrics only for successful trajectories
        if quality_label == "successful":
            loss_trajectories.append(traj_loss)
            if not np.isnan(traj_pearson):
                pearson_trajectories.append(traj_pearson)

    if len(unique_trajectory_ids) == 0:
        loss_per_trajectory = np.nan
        pearson_per_trajectory = np.nan
    else:
        loss_per_trajectory = np.mean(loss_trajectories).item()
        pearson_per_trajectory = np.mean(pearson_trajectories).item()

    metrics["loss"] = loss_per_trajectory
    metrics["pearson"] = pearson_per_trajectory

    # Add partial_success delta metric if available
    if use_partial_success and partial_success_deltas:
        metrics["partial_success_abs_delta"] = float(np.mean(partial_success_deltas))

    # Add RoboReward MAE metric if available
    if is_roboreward and pred_bins_mae and gt_bins_mae:
        mae = _compute_mae_between_bins(pred_bins_mae, gt_bins_mae)
        metrics["mae"] = mae

    return metrics, plots, video_frames_list, trajectory_progress_data


def _compute_mae_between_bins(pred_bins: List[int], gt_bins: List[int]) -> float:
    """Compute Mean Absolute Error (MAE) between predicted bins and ground truth bins.

    MAE(4, 5) = 1, MAE(3, 5) = 2, etc.

    Args:
        pred_bins: List of predicted bin values
        gt_bins: List of ground truth bin values

    Returns:
        Mean Absolute Error (float)
    """
    if len(pred_bins) != len(gt_bins):
        raise ValueError(
            f"Length mismatch: pred_bins has {len(pred_bins)} elements, gt_bins has {len(gt_bins)} elements"
        )

    if len(pred_bins) == 0:
        return 0.0

    # Compute absolute differences
    abs_diffs = [abs(pred - gt) for pred, gt in zip(pred_bins, gt_bins)]

    # Return mean
    return float(np.mean(abs_diffs))


def _extract_trajectory_rewards(
    progress_pred: list | np.ndarray,
    progress_pred_type: str,
    is_discrete_mode: bool,
    aggregation: str = "last",
) -> float:
    """Extract trajectory reward using different aggregation methods.

    Args:
        progress_pred: Progress predictions for a single trajectory.
                       For both continuous and discrete modes: list of floats in [0, 1]
                       (discrete mode uses convert_bins_to_continuous to get continuous values)
        progress_pred_type: "relative" or "absolute"
        is_discrete_mode: Whether predictions came from discrete mode (now converted to continuous)
        aggregation: "last", "sum", or "average"

    Returns:
        Aggregated reward (float)
    """
    # Both discrete and continuous modes now use continuous values
    pred_array = np.array(progress_pred, dtype=np.float32)

    # Apply cumsum if relative (for both discrete and continuous modes)
    if progress_pred_type == "relative":
        pred_array = np.cumsum(pred_array)

    # Apply aggregation (same logic for both modes)
    if aggregation == "last":
        reward = (
            pred_array[-1]
            if pred_array.ndim > 0 and len(pred_array) > 0
            else (pred_array if pred_array.ndim == 0 else 0)
        )
    elif aggregation == "sum":
        reward = np.sum(pred_array)
    elif aggregation == "average":
        reward = np.mean(pred_array)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Always return float (discrete mode now uses continuous values)
    return float(reward)


def _compute_policy_ranking_metrics_partial_success(
    all_rewards: np.ndarray,
    all_partial_successes: np.ndarray,
    all_tasks: List[str],
    correlation_method: str = "kendall",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute policy ranking metrics for datasets with partial_success.

    Args:
        all_rewards: Array of aggregated rewards
        all_partial_successes: Array of partial_success values (already converted to discrete bins if needed)
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    all_partial_successes = np.array(all_partial_successes)

    # Group by task
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        task_indices[task].append(i)

    if not task_indices:
        return {}, {}

    task_details = {}
    all_correct_pairs = []
    all_total_pairs = []
    all_spearman_rewind = []

    # Compute ranking accuracy for pairs based on partial_success vs predicted rewards
    for task, task_idx in task_indices.items():
        if len(task_idx) < 2:
            continue

        partial_successes = all_partial_successes[task_idx]
        predicted_rewards = all_rewards[task_idx]

        n = len(partial_successes)
        if n < 2:
            continue

        correct_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Skip pairs with None values
                if partial_successes[i] is None or partial_successes[j] is None:
                    continue
                if partial_successes[i] == partial_successes[j]:
                    continue

                total_pairs += 1
                # True ranking: partial_success[i] should be ranked higher if it's greater
                true_label = partial_successes[i] > partial_successes[j]
                # Predicted ranking: predicted_rewards[i] should be higher if it's greater
                pred_label = predicted_rewards[i] > predicted_rewards[j]

                if true_label == pred_label:
                    correct_pairs += 1

        # Compute spearman_rewind (binning between 0 and 1)
        # Filter out None values for binning
        valid_mask = np.array([ps is not None for ps in partial_successes])
        if np.any(valid_mask):
            valid_partial_successes = np.array([ps for ps in partial_successes if ps is not None])
            valid_predicted_rewards = predicted_rewards[valid_mask]
            bin_edges = np.linspace(0, 1, 4)  # Creates bins [0, 1/3), [1/3, 2/3), [2/3, 1]
            bin_assignments = np.clip(np.digitize(valid_partial_successes, bin_edges[1:], right=False), 0, 2)
        else:
            bin_assignments = np.array([])
            valid_predicted_rewards = np.array([])

        avg_rewards_per_bin = {}
        bin_ranks = []
        avg_reward_values = []

        for bin_idx in range(3):
            mask = bin_assignments == bin_idx
            if np.any(mask):
                bin_rewards = valid_predicted_rewards[mask]
                avg_reward = float(np.mean(bin_rewards))
                avg_rewards_per_bin[bin_idx] = avg_reward
                bin_ranks.append(bin_idx)
                avg_reward_values.append(avg_reward)

        correlation_rewind = None
        if len(bin_ranks) >= 2:
            if correlation_method == "kendall":
                correlation_rewind = compute_kendall(bin_ranks, avg_reward_values)
            else:  # spearman
                correlation_rewind = compute_spearman(bin_ranks, avg_reward_values)
            if not np.isnan(correlation_rewind):
                all_spearman_rewind.append(correlation_rewind)

        if total_pairs > 0:
            all_correct_pairs.append(correct_pairs)
            all_total_pairs.append(total_pairs)
            task_ranking_acc = correct_pairs / total_pairs
            task_details[task] = {
                "ranking_acc": float(task_ranking_acc),
                f"{correlation_method}_rewind": float(correlation_rewind) if correlation_rewind is not None else None,
            }

    if not all_total_pairs:
        return {}, {}

    ranking_acc = None
    if all_total_pairs:
        total_correct = sum(all_correct_pairs)
        total_pairs = sum(all_total_pairs)
        ranking_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    metrics = {
        "ranking_acc_rba": ranking_acc,
        f"{correlation_method}_rewind_rba": np.mean(all_spearman_rewind).item() if all_spearman_rewind else None,
    }

    return metrics, task_details


def _compute_policy_ranking_metrics_quality_label(
    all_rewards: np.ndarray,
    all_quality_labels: list[str],
    all_tasks: list[str],
    correlation_method: str = "kendall",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute policy ranking metrics for datasets using quality_label.

    Args:
        all_rewards: Array of aggregated rewards
        all_quality_labels: List of quality labels
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    # Group by task
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        task_indices[task].append(i)

    if not task_indices:
        return {}, {}

    task_details = {}
    all_correlations = []
    all_correlations_rewind = []
    all_succ_subopt_diffs = []
    all_subopt_fail_diffs = []
    all_succ_fail_diffs = []
    all_correct_pairs = []
    all_total_pairs = []

    # Track global ranking accuracy for all quality pairs
    global_pair_correct = {}  # (quality1, quality2) -> correct_count
    global_pair_total = {}  # (quality1, quality2) -> total_count

    # Non-RoboArena: Use quality_label
    quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}
    all_labels = ["failure", "suboptimal", "successful"]

    for task, task_idx in task_indices.items():
        task_rewards = all_rewards[task_idx]
        task_quality_labels = [all_quality_labels[i] for i in task_idx]

        quality_to_rewards = {q: [] for q in all_labels}
        for quality, reward in zip(task_quality_labels, task_rewards):
            quality_to_rewards[quality].append(float(reward))

        present_labels = [q for q in all_labels if quality_to_rewards[q]]

        if len(present_labels) < 2:
            continue

        k = len(present_labels)
        correlation_scores = []

        for labels_combo in combinations(present_labels, k):
            gold_ranks = [quality_order[q] for q in labels_combo]
            for rewards_tuple in product(*(quality_to_rewards[q] for q in labels_combo)):
                if correlation_method == "kendall":
                    corr = compute_kendall(gold_ranks, list(rewards_tuple))
                else:  # spearman
                    corr = compute_spearman(gold_ranks, list(rewards_tuple))
                if not np.isnan(corr):
                    correlation_scores.append(corr)

        avg_correlation = float(np.mean(correlation_scores)) if correlation_scores else 0.0

        avg_rewards_per_quality = {}
        quality_ranks = []
        avg_reward_values = []
        for q in present_labels:
            rewards = np.array(quality_to_rewards[q])
            if len(rewards) > 0:
                avg_reward = float(np.mean(rewards))
                avg_rewards_per_quality[q] = avg_reward
                quality_ranks.append(quality_order[q])
                avg_reward_values.append(avg_reward)

        correct_pairs = 0
        total_pairs = 0

        # Compare every pair of trajectories within this task
        for i in range(len(task_quality_labels)):
            for j in range(i + 1, len(task_quality_labels)):
                quality1 = task_quality_labels[i]
                quality2 = task_quality_labels[j]
                reward1 = task_rewards[i]
                reward2 = task_rewards[j]

                # Skip if same quality label
                if quality1 == quality2:
                    continue

                expected_order = quality_order[quality1] > quality_order[quality2]
                actual_order = reward1 > reward2
                total_pairs += 1
                if expected_order == actual_order:
                    correct_pairs += 1

                # Track global pairs by quality label combination
                pair_key = tuple(sorted([quality1, quality2]))
                if pair_key not in global_pair_total:
                    global_pair_total[pair_key] = 0
                    global_pair_correct[pair_key] = 0
                global_pair_total[pair_key] += 1
                if expected_order == actual_order:
                    global_pair_correct[pair_key] += 1

        if total_pairs > 0:
            all_correct_pairs.append(correct_pairs)
            all_total_pairs.append(total_pairs)

        correlation_rewind = None
        if len(quality_ranks) >= 2:
            if correlation_method == "kendall":
                correlation_rewind = compute_kendall(quality_ranks, avg_reward_values)
            else:  # spearman
                correlation_rewind = compute_spearman(quality_ranks, avg_reward_values)
            if not np.isnan(correlation_rewind):
                all_correlations_rewind.append(correlation_rewind)

        succ_subopt_diff = None
        subopt_fail_diff = None
        succ_fail_diff = None

        if "successful" in avg_rewards_per_quality and "suboptimal" in avg_rewards_per_quality:
            succ_subopt_diff = avg_rewards_per_quality["successful"] - avg_rewards_per_quality["suboptimal"]
            all_succ_subopt_diffs.append(succ_subopt_diff)

        if "suboptimal" in avg_rewards_per_quality and "failure" in avg_rewards_per_quality:
            subopt_fail_diff = avg_rewards_per_quality["suboptimal"] - avg_rewards_per_quality["failure"]
            all_subopt_fail_diffs.append(subopt_fail_diff)

        if "successful" in avg_rewards_per_quality and "failure" in avg_rewards_per_quality:
            succ_fail_diff = avg_rewards_per_quality["successful"] - avg_rewards_per_quality["failure"]
            all_succ_fail_diffs.append(succ_fail_diff)

        task_details[task] = {
            correlation_method: avg_correlation,
            f"{correlation_method}_rewind": correlation_rewind,
            "succ_subopt_diff": succ_subopt_diff,
            "subopt_fail_diff": subopt_fail_diff,
            "succ_fail_diff": succ_fail_diff,
        }
        all_correlations.append(avg_correlation)

    if len(all_correlations) == 0:
        return {}, {}

    ranking_acc = None
    if all_total_pairs:
        total_correct = sum(all_correct_pairs)
        total_pairs = sum(all_total_pairs)
        ranking_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    # Compute ranking accuracy for all pairs by quality label combination
    ranking_acc_all_pairs = {}
    for pair_key in global_pair_total:
        if global_pair_total[pair_key] > 0:
            pair_acc = global_pair_correct[pair_key] / global_pair_total[pair_key]
            pair_name = f"ranking_acc_{pair_key[0]}_vs_{pair_key[1]}"
            ranking_acc_all_pairs[pair_name] = pair_acc

    # Compute overall ranking accuracy across all pairs
    overall_ranking_acc_all_pairs = None
    if global_pair_total:
        total_correct_all = sum(global_pair_correct.values())
        total_pairs_all = sum(global_pair_total.values())
        overall_ranking_acc_all_pairs = total_correct_all / total_pairs_all if total_pairs_all > 0 else 0.0

    metrics = {
        correlation_method: np.mean(all_correlations).item(),
        f"{correlation_method}_rewind": np.mean(all_correlations_rewind).item() if all_correlations_rewind else None,
        "avg_succ_subopt_diff": np.mean(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "min_succ_subopt_diff": np.min(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "max_succ_subopt_diff": np.max(all_succ_subopt_diffs).item() if all_succ_subopt_diffs else None,
        "avg_subopt_fail_diff": np.mean(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "min_subopt_fail_diff": np.min(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "max_subopt_fail_diff": np.max(all_subopt_fail_diffs).item() if all_subopt_fail_diffs else None,
        "avg_succ_fail_diff": np.mean(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "min_succ_fail_diff": np.min(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "max_succ_fail_diff": np.max(all_succ_fail_diffs).item() if all_succ_fail_diffs else None,
        "ranking_acc": ranking_acc,
        "ranking_acc_all_pairs": overall_ranking_acc_all_pairs,
        **ranking_acc_all_pairs,  # Add individual pair accuracies
    }

    return metrics, task_details


def _compute_policy_ranking_metrics_from_rewards(
    all_rewards: np.ndarray,
    use_partial_success: bool,
    all_partial_successes: Optional[np.ndarray],
    all_quality_labels: Optional[List[str]],
    all_tasks: List[str],
    correlation_method: str = "kendall",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute policy ranking metrics from pre-computed trajectory rewards.

    Args:
        all_rewards: Array of aggregated rewards
        use_partial_success: Whether this dataset uses partial_success
        all_partial_successes: Array of partial_success values (already converted to discrete bins if needed)
        all_quality_labels: List of quality labels (if not use_partial_success)
        all_tasks: List of task names

    Returns:
        Tuple of (metrics dictionary, task_details dictionary)
    """
    if use_partial_success and all_partial_successes is not None:
        return _compute_policy_ranking_metrics_partial_success(
            all_rewards, all_partial_successes, all_tasks, correlation_method
        )
    else:
        return _compute_policy_ranking_metrics_quality_label(
            all_rewards, all_quality_labels, all_tasks, correlation_method
        )


def run_confusion_matrix_eval(
    results: List[Dict[str, Any]], progress_pred_type: str, is_discrete_mode: bool, num_bins: int
) -> Dict[str, Any]:
    """Run confusion_matrix evaluation analysis."""
    # First, gather all progress predictions, lang_tasks, and video_tasks
    all_progress_preds = []
    all_lang_tasks = []
    all_video_tasks = []
    valid_indices = []

    for idx, r in enumerate(results):
        progress_pred = r.get("progress_pred")
        if progress_pred is not None and len(progress_pred) > 0:
            meta = r.get("metadata", {})
            lang_task = meta.get("lang_task")
            video_task = meta.get("video_task")
            if lang_task is not None and video_task is not None:
                all_progress_preds.append(progress_pred)
                all_lang_tasks.append(lang_task)
                all_video_tasks.append(video_task)
                valid_indices.append(idx)

    if not all_progress_preds:
        return {"error": "No valid confusion matrix data found"}, np.zeros((1, 1))

    # Group results by confusion matrix task
    uniq_tasks = set(all_lang_tasks) | set(all_video_tasks)
    task_to_idx = {task: idx for idx, task in enumerate(uniq_tasks)}
    num_tasks = len(uniq_tasks)

    # Extract final rewards vectorized
    all_final_rewards = []
    for progress_pred in all_progress_preds:
        pred_array = np.array(progress_pred)

        if is_discrete_mode:
            # Discrete mode: progress_pred is logits [seq_len, num_bins]
            # Convert to continuous values using weighted sum of bin centers
            last_frame_logits = pred_array[-1] if pred_array.ndim > 1 else pred_array
            continuous_pred = convert_bins_to_continuous(torch.tensor(last_frame_logits, dtype=torch.float32)).item()
            final_reward = float(continuous_pred)
        else:
            # Continuous mode: use last frame value
            if progress_pred_type == "relative":
                pred_array = np.cumsum(pred_array)
            final_reward = float(pred_array[-1] if pred_array.ndim > 0 else pred_array)

        all_final_rewards.append(final_reward)

    all_final_rewards = np.array(all_final_rewards)
    all_lang_indices = np.array([task_to_idx[task] for task in all_lang_tasks])
    all_video_indices = np.array([task_to_idx[task] for task in all_video_tasks])

    # Build confusion matrix using vectorized operations
    confusion_matrix = np.zeros((num_tasks, num_tasks))
    count_matrix = np.zeros((num_tasks, num_tasks))

    # Use advanced indexing to accumulate rewards
    np.add.at(confusion_matrix, (all_lang_indices, all_video_indices), all_final_rewards)
    np.add.at(count_matrix, (all_lang_indices, all_video_indices), 1)

    if np.sum(count_matrix) == 0:
        return {"error": "No valid confusion matrix data found"}, np.zeros((num_tasks, num_tasks))

    # Calculate average rewards (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        confusion_matrix = np.divide(
            confusion_matrix, count_matrix, out=np.zeros_like(confusion_matrix), where=count_matrix != 0
        )

    # Create the plot
    fig = plt.figure(figsize=(8, 8))

    # Create heatmap showing average final rewards
    sns.heatmap(
        confusion_matrix,
        # annot=True,
        # fmt='.3f',
        cmap="Blues",  # White to dark blue colormap
        # xticklabels=list(uniq_tasks),
        # yticklabels=list(uniq_tasks),
        # cbar_kws={"label": "Average Final Reward (5 trajs)"},
        cbar=False,  # Remove the color bar
    )
    # plt.xlabel("Language Task", fontsize=12)
    # plt.ylabel("Video Task", fontsize=12)
    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=0)
    # Remove xticks and yticks
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # Compute trace - off-diagonal metric
    n = num_tasks
    trace = np.trace(confusion_matrix)
    total_sum = np.sum(confusion_matrix)
    off_diag_sum = total_sum - trace
    trace_minus_offdiag = trace - off_diag_sum

    # Normalized version (avg diagonal - avg off-diagonal)
    avg_diagonal = trace / n if n > 0 else 0.0
    avg_off_diag = off_diag_sum / (n * n - n) if n > 1 else 0.0
    normalized_metric = avg_diagonal - avg_off_diag

    metrics = {
        "trace": float(trace),
        "off_diagonal_sum": float(off_diag_sum),
        "trace_minus_offdiag": float(trace_minus_offdiag),
        "avg_diagonal": float(avg_diagonal),
        "avg_off_diagonal": float(avg_off_diag),
        "normalized_trace_minus_offdiag": float(normalized_metric),
    }

    return fig, confusion_matrix, metrics


def run_policy_ranking_eval(
    results: List[Dict[str, Any]],
    progress_pred_type: str,
    is_discrete_mode: bool,
    num_bins: int,
    data_source: Optional[str] = None,
    correlation_method: str = "kendall",
) -> Dict[str, Any]:
    """Run policy_ranking evaluation analysis.

    Groups results by trajectory_id (like reward_alignment) and computes policy ranking metrics
    using "last", "average", and "sum" aggregation methods.

    For datasets without partial_success: Uses quality_label and quality_order for ranking.
    For datasets with partial_success: Uses partial_success for ranking (no quality_order computation).
    """
    # Check if data_source contains roboreward or roboarena to determine if we should use partial_success logic
    use_partial_success = "roboreward" in str(data_source).lower() or "roboarena" in str(data_source).lower()

    # Group results by trajectory_id
    unique_trajectory_ids = set()
    for r in results:
        trajectory_id = r.get("id")
        if trajectory_id:
            unique_trajectory_ids.add(trajectory_id)

    if not unique_trajectory_ids:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Collect progress predictions per trajectory
    trajectory_progress_preds = {}  # trajectory_id -> list of progress_pred arrays
    trajectory_metadata = {}  # trajectory_id -> {task, quality_label/partial_success, video_path}

    for trajectory_id in unique_trajectory_ids:
        results_for_trajectory = [r for r in results if r.get("id") == trajectory_id]

        # Sort by frame_step if available (for frame_steps mode)
        # This orders subsequences from shortest to longest (e.g., [0], [0,1], [0,1,2], ...)
        # Only sort if there are multiple results (indicating frame_steps mode)
        if len(results_for_trajectory) > 1:
            results_for_trajectory.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))

        # Collect all progress predictions for this trajectory
        traj_progress_preds = [
            r.get("progress_pred") for r in results_for_trajectory if r.get("progress_pred") is not None
        ]
        trajectory_progress_preds[trajectory_id] = traj_progress_preds

        metadata = {
            "task": results_for_trajectory[0].get("task"),
            "video_path": results_for_trajectory[0].get("video_path"),
            "partial_success": results_for_trajectory[0].get("partial_success"),
            "quality_label": results_for_trajectory[0].get("quality_label"),
        }
        trajectory_metadata[trajectory_id] = metadata

    if not trajectory_progress_preds:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Compute rewards for each trajectory using different aggregation methods
    all_rewards_last = []
    all_rewards_avg = []
    all_rewards_sum = []
    all_tasks = []
    all_quality_labels = []
    all_partial_successes = []
    all_video_paths = []
    all_ids = []

    for trajectory_id, progress_preds_list in trajectory_progress_preds.items():
        metadata = trajectory_metadata[trajectory_id]

        # Process progress predictions: convert logits to continuous values if needed
        processed_progress_preds = []
        for progress_pred in progress_preds_list:
            pred_array = np.array(progress_pred)

            if is_discrete_mode:
                # Discrete mode: pred_array might be logits [seq_len, num_bins]
                # Convert to continuous values using weighted sum of bin centers
                if pred_array.ndim > 1:
                    # It's logits [seq_len, num_bins], convert to continuous values
                    continuous_preds = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).numpy()
                    processed_progress_preds.append(continuous_preds.tolist())
                elif pred_array.ndim == 1:
                    # Single frame logits [num_bins], convert to continuous
                    continuous_pred = convert_bins_to_continuous(torch.tensor(pred_array, dtype=torch.float32)).item()
                    processed_progress_preds.append([float(continuous_pred)])
                else:
                    # Scalar (shouldn't happen, but handle it)
                    processed_progress_preds.append([float(pred_array)])
            else:
                # Continuous mode: pred_array is scalar values
                if pred_array.ndim > 0:
                    processed_progress_preds.append(pred_array.tolist())
                else:
                    processed_progress_preds.append([float(pred_array)])

        if not processed_progress_preds:
            continue

        # Take the last prediction from each subsequence (e.g., if max_frames=4, take the 4th prediction)
        # Then use _extract_trajectory_rewards to compute rewards with different aggregation methods
        last_predictions = []
        for pred_list in processed_progress_preds:
            last_predictions.append(pred_list[-1])

        if not last_predictions:
            continue

        # Use _extract_trajectory_rewards with the list of last predictions from each subsequence
        reward_last = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="last",
        )
        reward_avg = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="average",
        )
        reward_sum = _extract_trajectory_rewards(
            last_predictions,
            progress_pred_type,
            is_discrete_mode,
            aggregation="sum",
        )

        # Skip trajectories with None partial_success for datasets with partial_success
        if use_partial_success:
            if metadata["partial_success"] is None:
                continue
            if is_discrete_mode:
                if isinstance(metadata["partial_success"], torch.Tensor):
                    # [num_bins] -> [1, 1, num_bins]
                    partial_success_tensor = metadata["partial_success"][
                        None, None
                    ]  # to make it 3-dim for convert_discrete_target_to_continuous
                else:
                    # number -> [1, 1]
                    partial_success_tensor = torch.tensor([metadata["partial_success"]], dtype=torch.float32).unsqueeze(
                        0
                    )
                metadata["partial_success"] = convert_discrete_target_to_continuous(
                    partial_success_tensor, num_bins=num_bins
                ).item()
            all_partial_successes.append(metadata["partial_success"])
        else:
            all_quality_labels.append(metadata["quality_label"])

        all_rewards_last.append(reward_last)
        all_rewards_avg.append(reward_avg)
        all_rewards_sum.append(reward_sum)

        all_tasks.append(metadata["task"])
        all_video_paths.append(metadata.get("video_path"))
        all_ids.append(trajectory_id)

    all_rewards_last = np.array(all_rewards_last)
    all_rewards_avg = np.array(all_rewards_avg)
    all_rewards_sum = np.array(all_rewards_sum)

    # Group by task for building task_groups (for return value)
    task_groups = {}
    task_indices = defaultdict(list)
    for i, task in enumerate(all_tasks):
        if task not in task_groups:
            task_groups[task] = []

        task_entry = {
            "final_predicted_reward_last": all_rewards_last[i],
            "final_predicted_reward_avg": all_rewards_avg[i],
            "final_predicted_reward_sum": all_rewards_sum[i],
            "video_path": all_video_paths[i],
        }

        # Add id if available
        if all_ids:
            task_entry["id"] = all_ids[i]

        # Add specific key based on dataset type
        if use_partial_success:
            task_entry["partial_success"] = all_partial_successes[i]
        else:
            task_entry["quality_label"] = all_quality_labels[i]

        task_groups[task].append(task_entry)
        task_indices[task].append(i)

    if not task_groups:
        return {"error": "No valid policy ranking data found"}, {}, {}

    # Compute policy ranking metrics for each aggregation method
    all_metrics = {}
    all_task_details = {}

    for agg_method, rewards in [("last", all_rewards_last), ("avg", all_rewards_avg), ("sum", all_rewards_sum)]:
        metrics, task_details = _compute_policy_ranking_metrics_from_rewards(
            rewards,
            use_partial_success,
            np.array(all_partial_successes) if use_partial_success and all_partial_successes else None,
            all_quality_labels if not use_partial_success else None,
            all_tasks,
            correlation_method,
        )

        if metrics:
            # Prefix metrics with aggregation method
            prefixed_metrics = {f"{k}_{agg_method}" if k != "error" else k: v for k, v in metrics.items()}
            all_metrics.update(prefixed_metrics)

            # Merge task details (keep first one, or combine if needed)
            if not all_task_details:
                all_task_details = task_details
            else:
                # Merge task details by adding aggregation suffix to keys
                for task, details in task_details.items():
                    if task not in all_task_details:
                        all_task_details[task] = {}
                    for k, v in details.items():
                        all_task_details[task][f"{k}_{agg_method}"] = v

    if not all_metrics:
        return {"error": "No valid correlations computed"}, {}, {}

    return all_metrics, task_groups, all_task_details
