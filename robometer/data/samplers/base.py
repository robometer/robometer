#!/usr/bin/env python3
from typing import Optional, Dict, Any, List, Set, Tuple, Union

import numpy as np
import random
import torch
from random import Random
from datasets import Dataset

from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.helpers import (
    load_frames_from_npz,
    get_segment_indices_with_middle,
    compute_progress_from_segment,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
    compute_success_labels,
    create_trajectory_from_dict,
    load_embeddings_from_path,
    linspace_subsample_frames,
    convert_continuous_to_discrete_bins,
)
from robometer.data.dataset_types import Trajectory
from robometer.utils.logger import get_logger
from robometer.data.dataset_category import is_preference_only_ds

logger = get_logger()


class RBMBaseSampler:
    """Base sampler class that provides trajectory retrieval functions for generating samples."""

    def __init__(
        self,
        config: DataConfig,
        dataset: Dataset,
        combined_indices: Dict[str, Any],
        dataset_success_cutoff_map: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        random_seed: int = 42,
        pad_frames: bool = True,
    ):
        """Initialize sampler with dataset and indices.

        Args:
            config: Configuration object
            dataset: The loaded dataset
            combined_indices: Dictionary of combined indices from dataset loading
            dataset_success_cutoff_map: Dictionary mapping dataset names to success cutoff percentages
            verbose: Verbose flag
            random_seed: Random seed for deterministic sampling. Creates a local Random instance to avoid affecting global random state.
        """
        self.config = config
        self.dataset = dataset
        self.verbose = verbose
        self.dataset_success_cutoff_map = dataset_success_cutoff_map or {}
        self._local_random = Random(random_seed)
        self.pad_frames = pad_frames
        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        # Build indices from combined_indices
        self._build_indices(combined_indices)

    def _build_indices(self, combined_indices):
        """Build all index mappings from combined_indices.

        Args:
            combined_indices: Dictionary of combined indices from dataset loading
        """
        # Initialize index mappings from the loaded indices
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]
        self.partial_success_indices = combined_indices["partial_success_indices"]
        self.paired_human_robot_by_task = combined_indices["paired_human_robot_by_task"]
        self.tasks_with_multiple_quality_labels = combined_indices["tasks_with_multiple_quality_labels"]

        # Build mapping from data source -> available task instructions
        self._build_tasks_by_data_source()

    def _build_tasks_by_data_source(self):
        """Cache mapping from data_source to available task instructions."""
        self.tasks_by_data_source: Dict[str, List[str]] = {}

        all_tasks = self.dataset["task"]
        all_sources = self.dataset["data_source"]

        source_to_tasks: Dict[str, Set[str]] = {}
        for task, source in zip(all_tasks, all_sources):
            if task is None or source is None:
                continue
            if source not in source_to_tasks:
                source_to_tasks[source] = set()
            source_to_tasks[source].add(task)

        self.tasks_by_data_source = {source: list(tasks) for source, tasks in source_to_tasks.items()}

    def _generate_sample(self, item):
        """Generate a sample from an item.

        This method should be overridden by subclasses to implement their specific
        sample generation logic.

        Args:
            item: An item from the dataset (typically a trajectory dict)

        Returns:
            A sample object (e.g., PreferenceSample, ProgressSample)
        """
        raise NotImplementedError("Subclasses must implement _generate_sample")

    def _get_same_task_optimal(self, ref_traj: dict) -> dict | None:
        """Get optimal trajectory from same task (different from ref).

        Args:
            ref_traj: Reference trajectory

        Returns:
            Same task optimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_optimal_indices = self.optimal_by_task.get(task_name, [])
        if not same_task_optimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_optimal: No optimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_optimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_optimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_optimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_optimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_optimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_optimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_same_task_suboptimal(self, ref_traj: dict) -> dict | None:
        """Get suboptimal trajectory from same task.

        For trajectories with partial_success, uses partial_success logic instead of quality_label logic.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Suboptimal trajectory dict or None if not available
        """
        # Check if this trajectory uses partial_success
        use_partial_success = ref_traj.get("partial_success") is not None

        if use_partial_success:
            # For trajectories with partial_success, use partial_success logic
            return self._get_different_partial_success_traj(ref_traj)

        # For trajectories without partial_success, use the standard suboptimal logic
        task_name = ref_traj["task"]
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if not same_task_suboptimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_suboptimal: No suboptimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_suboptimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_suboptimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_suboptimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_suboptimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_suboptimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_suboptimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_different_video_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from different task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Different task trajectory dict or None if not available
        """
        same_source_prob = self.config.traj_same_source_prob
        data_source = ref_traj.get("data_source")
        other_tasks = []

        if data_source and data_source in self.tasks_by_data_source and random.random() < same_source_prob:
            other_tasks = [task for task in self.tasks_by_data_source[data_source] if task != ref_traj["task"]]

        if not other_tasks:
            other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]

        if not other_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_video_traj: No other tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        # Try up to 2 times to find a valid task
        max_retries = 2
        other_task_indices = None
        other_task = None

        for attempt in range(max_retries):
            other_task = random.choice(other_tasks)
            if other_task not in self.optimal_by_task:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_video_traj: Attempt {attempt + 1}/{max_retries}: Task '{other_task}' not found in optimal_by_task"
                )
                continue

            other_task_indices = self.optimal_by_task[other_task]
            if not other_task_indices:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_video_traj: Attempt {attempt + 1}/{max_retries}: Task '{other_task}' has no optimal indices"
                )
                continue

            # Found a valid task with indices
            break

        if other_task_indices is None or not other_task_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_video_traj: Failed to find valid task after {max_retries} attempts"
            )
            return None

        other_idx = random.choice(other_task_indices)
        result = self.dataset[other_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_different_video_traj: Found trajectory {result.get('id', 'unknown')} from task '{other_task}'"
        )
        return result

    def _get_different_task_instruction(self, ref_traj: dict) -> dict | None:
        """Get the same trajectory but with a different task instruction.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different task instruction or None if not available
        """
        same_source_prob = self.config.traj_same_source_prob
        data_source = ref_traj.get("data_source")
        candidate_tasks = []

        if data_source and data_source in self.tasks_by_data_source and random.random() < same_source_prob:
            candidate_tasks = [task for task in self.tasks_by_data_source[data_source] if task != ref_traj["task"]]

        if not candidate_tasks:
            candidate_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]

        if not candidate_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_task_instruction: No candidate tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        other_task = random.choice(candidate_tasks)

        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        other_task_indices = self.optimal_by_task.get(other_task, [])
        if not other_task_indices:
            logger.trace(f"[BASE SAMPLER] _get_different_task_instruction: Task '{other_task}' has no optimal indices")
            return None

        other_task_idx = random.choice(other_task_indices)
        other_task_traj = self.dataset[other_task_idx]

        # Create a copy of the trajectory with the task changed
        # Use embeddings_path and lang_vector from the other_task trajectory
        new_traj = ref_traj.copy()
        new_traj["task"] = other_task
        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        if "embeddings_path" in other_task_traj:
            new_traj["embeddings_path"] = other_task_traj["embeddings_path"]
        if "lang_vector" in other_task_traj:
            new_traj["lang_vector"] = other_task_traj["lang_vector"]
        return new_traj

    def _get_paired_human_robot_traj(self, ref_traj: dict) -> dict | None:
        """Get paired human/robot trajectory for the same task.

        Given a reference trajectory, if it's a robot trajectory, returns a human trajectory
        from the same task. If it's a human trajectory, returns a robot trajectory from the
        same task.

        Args:
            ref_traj: Reference trajectory (can be robot or human)

        Returns:
            Paired trajectory dict (opposite type) or None if not available
        """
        task = ref_traj["task"]
        is_robot = ref_traj.get("is_robot", True)

        if task not in self.paired_human_robot_by_task:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Task '{task}' not in paired_human_robot_by_task"
            )
            return None

        task_pairs = self.paired_human_robot_by_task[task]

        # Get opposite type
        opposite_key = "human" if is_robot else "robot"
        opposite_indices = task_pairs.get(opposite_key, [])

        if not opposite_indices:
            logger.trace(f"[BASE SAMPLER] _get_paired_human_robot_traj: No {opposite_key} indices for task '{task}'")
            return None

        # Sample a paired trajectory and verify it's different from reference
        chosen_id = ref_traj["id"]
        available_indices = opposite_indices.copy()
        paired_traj = None

        # Add retry limit to prevent infinite loops
        max_retries = min(len(available_indices), 10)
        retries = 0

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Looking for {opposite_key} trajectory (chosen_id: {chosen_id}, available: {len(available_indices)})"
        )

        while (paired_traj is None or paired_traj.get("id") == chosen_id) and retries < max_retries:
            retries += 1

            if not available_indices:
                logger.trace(
                    f"[BASE SAMPLER] _get_paired_human_robot_traj: No more available indices after {retries} retries"
                )
                return None

            paired_idx = random.choice(available_indices)
            paired_traj = self.dataset[paired_idx]

            # If it matches, remove this index and try again
            if paired_traj.get("id") == chosen_id:
                available_indices = [idx for idx in available_indices if idx != paired_idx]
                paired_traj = None
                continue

        # If we exhausted retries without finding a valid trajectory, return None
        if paired_traj is None or paired_traj.get("id") == chosen_id:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Failed to find valid paired trajectory after {max_retries} retries"
            )
            return None

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Found paired trajectory {paired_traj.get('id', 'unknown')} on retry {retries}"
        )
        return paired_traj

    def _get_different_partial_success_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from same task with different partial_success.

        Finds trajectories with either higher or lower partial_success than the reference,
        using absolute difference for threshold checking.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different partial_success from same task or None if not available
        """
        task_name = ref_traj["task"]
        ref_partial_success = ref_traj.get("partial_success")

        # Check if partial_success is available
        if ref_partial_success is None:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {ref_traj.get('id', 'unknown')}"
            )
            return None

        # Get minimum threshold from config
        min_threshold = getattr(self.config, "partial_success_threshold", 0.2)

        # Get all trajectories from the same task
        same_task_indices = self.task_indices.get(task_name, [])
        if not same_task_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories found for task '{task_name}'"
            )
            return None

        # Filter to trajectories with different partial_success that meet the threshold requirement
        # Uses absolute difference to allow both higher and lower partial_success
        chosen_id = ref_traj["id"]
        candidate_indices = []

        for idx in same_task_indices:
            # Skip if same trajectory
            if self._cached_ids[idx] == chosen_id:
                continue

            # Get partial_success for this trajectory
            traj_dict = self.dataset[idx]
            traj_partial_success = traj_dict.get("partial_success", None)

            if traj_partial_success is None:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {traj_dict.get('id', 'unknown')}, task '{task_name}'"
                )
                continue

            # Include if partial_success differs from reference by at least the threshold (using abs)
            partial_success_diff = abs(ref_partial_success - traj_partial_success)
            if partial_success_diff >= min_threshold:
                candidate_indices.append(idx)

        if not candidate_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories with different partial_success (threshold: {min_threshold}) for task '{task_name}' (ref: {ref_partial_success})"
            )
            return None

        # Randomly select from candidates
        selected_idx = random.choice(candidate_indices)
        result = self.dataset[selected_idx]
        result_partial_success = result.get("partial_success")
        # If ref_partial_success is 1.0, direction is always "lower" since 1.0 is the maximum
        if ref_partial_success == 1.0:
            direction = "lower"
        else:
            direction = "higher" if result_partial_success > ref_partial_success else "lower"
        logger.trace(
            f"[BASE SAMPLER] _get_different_partial_success_traj: Found trajectory {result.get('id', 'unknown')} with partial_success {result_partial_success} ({direction} than {ref_partial_success}, abs diff: {abs(ref_partial_success - result_partial_success):.3f}, threshold: {min_threshold})"
        )
        return result

    def _get_subsample_indices(
        self, data, direction: str = "bidirectional", max_frames: int = None
    ) -> Optional[Tuple[int, int, int]]:
        """Get start, middle, and end indices for subsample strategy.

        Samples three random frames from the trajectory. The relationship between indices
        follows three main scenarios:
        1. start < middle < end: forward progress - normal forward progression through trajectory
        2. start < end < middle: rewind progress - forward from start to end, then continues to middle (simulating rewind/backtrack)
        3. end < middle < start: reverse progress - backward from start through middle to end (full backward traversal)

        Args:
            data: Trajectory data (frames or embeddings) to sample from
            direction: Sampling direction - "forward" (start < middle < end),
                      "reverse" (end < middle < start),
                      "rewind" (start < end < middle),
                      or "bidirectional" (any of the 3 orderings)
            max_frames: Maximum number of frames to subsample. If 1, returns only start. If 2, returns start and end.

        Returns:
            Tuple of (start_idx, middle_idx, end_idx), or None if insufficient frames
            For max_frames == 1: returns (start_idx, None, None)
            For max_frames == 2: returns (start_idx, None, end_idx)
        """
        num_frames_total = len(data) if hasattr(data, "__len__") else data.shape[0]

        # Handle edge cases for max_frames == 1 or 2
        if max_frames == 1:
            # Randomly sample 1 frame
            random_idx = random.randint(0, num_frames_total - 1)
            logger.trace(f"[BASE SAMPLER] _get_subsample_indices: max_frames=1, randomly sampled idx={random_idx}")
            return (random_idx, None, None)

        if max_frames == 2:
            # Sample 2 frames: either forward (start < end) or reverse (end < start)
            # No rewind possible with only 2 frames
            if direction == "reverse":
                # Reverse: sample end first, then start (end < start)
                end_idx = random.randint(0, num_frames_total - 2)
                start_idx = random.randint(end_idx + 1, num_frames_total - 1)
            else:
                # Forward: sample start first, then end (start < end)
                start_idx = random.randint(0, num_frames_total - 2)
                end_idx = random.randint(start_idx + 1, num_frames_total - 1)
            logger.trace(
                f"[BASE SAMPLER] _get_subsample_indices: max_frames=2, start_idx={start_idx}, end_idx={end_idx}, direction={direction}"
            )
            return (start_idx, None, end_idx)

        if num_frames_total < 3:
            logger.trace(f"[BASE SAMPLER] _get_subsample_indices: Not enough frames ({num_frames_total})")
            return None

        # Sample three random distinct frames
        frame_indices = sorted(random.sample(range(num_frames_total), 3))
        frame1_idx, frame2_idx, frame3_idx = frame_indices

        # Determine start, middle, and end based on direction
        # We only care about 3 cases:
        # 1. start < middle < end: forward progress
        # 2. start < end < middle: rewind progress
        # 3. end < middle < start: reverse progress

        if direction == "forward":
            # Case 1: start < middle < end
            start_idx = frame1_idx
            middle_idx = frame2_idx
            end_idx = frame3_idx
        elif direction == "reverse":
            # Case 3: end < middle < start
            end_idx = frame1_idx
            middle_idx = frame2_idx
            start_idx = frame3_idx
        elif direction == "rewind":
            # Case 2: start < end < middle
            start_idx = frame1_idx
            end_idx = frame2_idx
            middle_idx = frame3_idx
        else:  # bidirectional (default)
            # Randomly choose from the 3 cases
            pattern = random.choice([1, 2, 3])
            if pattern == 1:  # start < middle < end: forward progress
                start_idx = frame1_idx
                middle_idx = frame2_idx
                end_idx = frame3_idx
            elif pattern == 2:  # start < end < middle: rewind progress
                start_idx = frame1_idx
                end_idx = frame2_idx
                middle_idx = frame3_idx
            else:  # pattern == 3: end < middle < start: reverse progress
                end_idx = frame1_idx
                middle_idx = frame2_idx
                start_idx = frame3_idx

        logger.trace(
            f"[BASE SAMPLER] _get_subsample_indices: Selected indices start={start_idx}, middle={middle_idx}, end={end_idx} "
            f"from {num_frames_total} total frames (direction: {direction})"
        )
        return start_idx, middle_idx, end_idx

    def _get_traj_from_data(
        self,
        traj: dict | Trajectory,
        subsample_strategy: str | None = None,
        frame_indices: List[int] | None = None,
        metadata: Dict[str, Any] | None = None,
        pad_frames: bool = True,
    ) -> Trajectory:
        """Load, subsample, and optionally pad trajectory data and create a Trajectory object.

        Args:
            traj: Trajectory dict or Trajectory object
            subsample_strategy: Optional strategy for subsampling ("subsample_forward", "subsample_reverse", "subsample_rewind", or None for default/bidirectional). Ignored if frame_indices is provided.
            frame_indices: Optional list of specific frame indices to use. If provided, subsample_strategy is ignored.
            metadata: Optional metadata dict to merge into trajectory metadata.
            pad_frames: Whether to pad the trajectory data to max_frames.

        Returns:
            Trajectory object with loaded and subsampled data (padded)
        """
        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None
        data = None

        if isinstance(traj, Trajectory):
            # If already a Trajectory, just return it
            return traj

        # Load from dict
        # Check if text_embedding is already provided in the dict (for samplers that need to override it)
        if "text_embedding" in traj and traj["text_embedding"] is not None:
            text_embedding = traj["text_embedding"]

        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            # Only use loaded text_embedding if not already provided in dict
            if text_embedding is None:
                text_embedding = embeddings["text_embedding"]
            data = video_embeddings
        else:
            if isinstance(traj["frames"], str):
                frames = load_frames_from_npz(traj["frames"])
            else:
                frames = traj["frames"]
            data = frames

        # Get total frames for progress computation
        if hasattr(data, "shape"):
            num_frames_total = data.shape[0]
        else:
            num_frames_total = len(data)

        ds_key = traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

        # Determine which indices to use (construct indices first, then subsample uniformly)
        if frame_indices is not None:
            # Use provided frame indices directly
            indices = frame_indices
        elif subsample_strategy is not None:
            # Use subsampling strategy
            # Get subsample indices (handles edge cases for max_frames == 1 or 2)
            if subsample_strategy == "subsample_forward":
                strategy_indices = self._get_subsample_indices(
                    data, direction="forward", max_frames=self.config.max_frames
                )
            elif subsample_strategy == "subsample_reverse":
                strategy_indices = self._get_subsample_indices(
                    data, direction="reverse", max_frames=self.config.max_frames
                )
            elif subsample_strategy == "subsample_rewind":
                strategy_indices = self._get_subsample_indices(
                    data, direction="rewind", max_frames=self.config.max_frames
                )
            else:
                strategy_indices = self._get_subsample_indices(
                    data, direction="bidirectional", max_frames=self.config.max_frames
                )

            if strategy_indices is None:
                logger.trace("[BASE SAMPLER] _get_traj_from_data: Failed to get uniform sample indices")
                return None

            start_idx, middle_idx, end_idx = strategy_indices

            logger.trace(
                f"[BASE SAMPLER] _get_traj_from_data: Subsampling trajectory with strategy: {subsample_strategy}, start_idx: {start_idx}, middle_idx: {middle_idx}, end_idx: {end_idx}"
            )

            # Use middle_idx only for rewind strategy (requires at least 3 frames)
            use_middle = subsample_strategy == "subsample_rewind" and middle_idx is not None and num_frames_total >= 3

            # Use get_segment_indices_with_middle to construct indices
            indices = get_segment_indices_with_middle(
                num_frames_total=num_frames_total,
                start_idx=start_idx,
                end_idx=end_idx,
                middle_idx=middle_idx if use_middle else None,
                max_frames=self.config.max_frames,
            )
        else:
            # No subsampling strategy or indices provided - use all frames
            indices = list(range(num_frames_total))

        # Extract data using indices
        subsampled = data[indices]

        # Get partial_success early to pass to compute_progress_from_segment
        partial_success = traj.get("partial_success")

        # Compute progress
        target_progress = compute_progress_from_segment(
            num_frames_total=num_frames_total,
            frame_indices=indices,
            progress_pred_type=self.config.progress_pred_type,
            success_cutoff=success_cutoff,
            partial_success=partial_success,
        )

        # Subsample uniformly if needed (if we have more frames than max_frames)
        current_frame_count = len(subsampled) if hasattr(subsampled, "__len__") else subsampled.shape[0]
        if current_frame_count > self.config.max_frames:
            subsampled, frame_indices_subsample = linspace_subsample_frames(subsampled, self.config.max_frames)
            # Update indices and target_progress
            if target_progress and len(target_progress) == current_frame_count:
                target_progress = [target_progress[idx] for idx in frame_indices_subsample]
            indices = [indices[idx] for idx in frame_indices_subsample] if isinstance(indices, list) else indices

            original_sampled_index = -1

        else:
            original_sampled_index = indices[-1] # record the original last sampled time index before potential padding
        # Pad if needed
        if target_progress and pad_frames:
            if self.config.load_embeddings:
                subsampled, target_progress = pad_trajectory_to_max_frames_torch(
                    subsampled, target_progress, self.config.max_frames
                )
            else:
                subsampled, target_progress = pad_trajectory_to_max_frames_np(
                    subsampled, target_progress, self.config.max_frames
                )

        # Create predict_last_frame_mask: mark the last frame if partial_success < 1.0
        # If predict_last_frame_partial_progress is True and partial_success < 1.0 and the last original frame is in the subsampled indices,
        # mark all positions where it appears with 1.0, all others 0.0. Otherwise, all 1.0s.
        final_frame_count = len(subsampled)
        predict_last_frame_mask = [1.0] * final_frame_count  # Default: all 1.0s (no masking)

        if self.config.predict_last_frame_partial_progress and partial_success is not None:
            if partial_success == 1.0 and not is_preference_only_ds(traj["data_source"]):
                pass
            else:
                last_original_frame_idx = num_frames_total - 1
                if isinstance(indices, list) and last_original_frame_idx in indices:
                    # Find all positions where the last frame index appears
                    last_frame_positions = [
                        i for i, idx in enumerate(indices) if idx == last_original_frame_idx and i < final_frame_count
                    ]
                    if last_frame_positions:
                        # Mark all positions where the last frame appears with 1.0, all others 0.0
                        predict_last_frame_mask = [0.0] * final_frame_count
                        for pos in last_frame_positions:
                            predict_last_frame_mask[pos] = 1.0
                else:
                    predict_last_frame_mask = [0.0] * final_frame_count

        # Update frames_shape
        frames_shape = subsampled.shape if hasattr(subsampled, "shape") else tuple()

        # Set frames or video_embeddings
        if self.config.load_embeddings:
            video_embeddings = subsampled
        else:
            frames = subsampled

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=target_progress,
            data_source=traj["data_source"],
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
            quality_label=traj.get("quality_label"),
        )

        # Convert partial_success and target_progress to discrete bins if in discrete mode
        if self.config.progress_loss_type.lower() == "discrete":
            if partial_success is not None:
                partial_success = convert_continuous_to_discrete_bins(
                    [partial_success], self.config.progress_discrete_bins
                )[0]
            target_progress = convert_continuous_to_discrete_bins(target_progress, self.config.progress_discrete_bins)

        trajectory = create_trajectory_from_dict(
            traj,
            overrides={
                "frames": frames,
                "frames_shape": frames_shape,
                "video_embeddings": video_embeddings,
                "text_embedding": text_embedding,
                "target_progress": target_progress,
                "success_label": success_label,
                "partial_success": partial_success,
                "predict_last_frame_mask": predict_last_frame_mask,
                "metadata": metadata,
                "original_sampled_index": original_sampled_index,
            },
        )
        return trajectory
