from typing import Dict, Any, Optional

import random
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.data.samplers.base import RBMBaseSampler
from robometer.data.datasets.helpers import (
    DataGenStrat,
    load_embeddings_from_path,
    convert_continuous_to_discrete_bins,
)
from robometer.utils.distributed import rank_0_print
from robometer.utils.logger import get_logger

logger = get_logger()


class ProgressSampler(RBMBaseSampler):
    """Data generator for progress samples."""

    def __init__(self, is_evaluation=False, **kwargs):
        super().__init__(**kwargs)

    def _generate_sample(self, item: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None):
        return self._create_progress_sample(item, preferred_strategy=preferred_strategy)

    def _execute_strategy(self, strategy: DataGenStrat, traj: Dict[str, Any]) -> tuple[Dict[str, Any], str] | None:
        """Execute a strategy to get processed trajectory.

        Args:
            strategy: The strategy to execute
            traj: The trajectory to process

        Returns:
            Tuple of (processed_traj, subsample_strategy) or None if failed
        """
        if strategy == DataGenStrat.FORWARD_PROGRESS:
            return (traj, "subsample_forward")
        elif strategy == DataGenStrat.REVERSE_PROGRESS:
            return (traj, "subsample_reverse")
        elif strategy == DataGenStrat.REWIND:
            return (traj, "subsample_rewind")
        elif strategy == DataGenStrat.DIFFERENT_TASK_INSTRUCTION:
            processed_traj = self._get_different_task_instruction(traj)
            if processed_traj is None:
                return None
            return (processed_traj, "subsample_forward")
        else:
            return None

    def _create_progress_sample(self, traj: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None):
        """Create a progress sample using normalized and rebalanced strategy selection.

        Implements four strategies:
        1. Different Task: Use trajectory from different task (progress set to 0.0)
        2. Forward Progress: Sample with forward direction (start < middle < end)
        3. Reverse Progress: Sample with reverse direction (end < middle < start)
        4. Rewind: Sample with rewind direction (start < end < middle)
        """
        # Initialize variables for strategy selection
        processed_traj = None
        strategy_used = None
        subsample_strategy = None

        # Strategy selection: use preferred_strategy if provided, otherwise select based on ratios
        if preferred_strategy is not None:
            # Use the preferred strategy directly
            logger.trace(f"[PROGRESS SAMPLER] Using preferred strategy: {preferred_strategy.value}")
            result = self._execute_strategy(preferred_strategy, traj)
            if result is None:
                logger.trace(f"[PROGRESS SAMPLER] Preferred strategy {preferred_strategy.value} failed, returning None")
                return None
            processed_traj, subsample_strategy = result
            strategy_used = preferred_strategy
            attempt = 1  # Set attempt for preferred strategy path
        else:
            # Strategy setup with rebalancing on failure
            # [different_task_instruction, forward_progress, reverse_progress, rewind]
            strategies = [
                (
                    DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
                    self.config.progress_strategy_ratio[0] if len(self.config.progress_strategy_ratio) > 0 else 0.0,
                ),
                (
                    DataGenStrat.FORWARD_PROGRESS,
                    self.config.progress_strategy_ratio[1] if len(self.config.progress_strategy_ratio) > 1 else 0.0,
                ),
                (
                    DataGenStrat.REVERSE_PROGRESS,
                    self.config.progress_strategy_ratio[2] if len(self.config.progress_strategy_ratio) > 2 else 0.0,
                ),
                (
                    DataGenStrat.REWIND,
                    self.config.progress_strategy_ratio[3] if len(self.config.progress_strategy_ratio) > 3 else 0.0,
                ),
            ]

            # Remove strategies with zero probability
            strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

            max_attempts = 10  # Limit retry attempts to prevent infinite loops
            attempt = 0

            while processed_traj is None and attempt < max_attempts:
                attempt += 1

                # Check if we have any strategies left
                if not strategies:
                    return None

                # Rebalance probabilities based on remaining strategies
                total_prob = sum(prob for _, prob in strategies)
                if total_prob == 0:
                    return None

                # Normalize probabilities
                normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

                # Select strategy based on rebalanced probabilities
                prob = random.random()
                cumulative_prob = 0.0
                selected_strategy = None

                for strat, normalized_prob in normalized_strategies:
                    cumulative_prob += normalized_prob
                    if prob <= cumulative_prob:
                        selected_strategy = strat
                        break

                # Execute selected strategy
                result = self._execute_strategy(selected_strategy, traj)
                if result is not None:
                    processed_traj, subsample_strategy = result
                    strategy_used = selected_strategy
                else:
                    # Remove failed strategy and try again
                    strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                    continue

            # If we still don't have a sample after all attempts, return None
            if processed_traj is None:
                logger.trace(
                    f"[PROGRESS SAMPLER] Failed to generate progress sample after {max_attempts} attempts - all strategies exhausted"
                )
                return None

        progress_traj = self._get_traj_from_data(processed_traj, subsample_strategy=subsample_strategy)

        if progress_traj is None:
            return None

        # Handle special cases
        if strategy_used == DataGenStrat.DIFFERENT_TASK:
            # Restore original task embeddings (video is from a different task)
            if self.config.load_embeddings and traj.get("embeddings_path"):
                progress_traj.text_embedding = load_embeddings_from_path(traj["embeddings_path"])["text_embedding"]
            progress_traj.lang_vector = traj["lang_vector"]
            progress_traj.task = traj["task"]

        if strategy_used in [DataGenStrat.DIFFERENT_TASK, DataGenStrat.DIFFERENT_TASK_INSTRUCTION]:
            progress_traj.target_progress = [0.0] * len(progress_traj.target_progress)
            if self.config.progress_loss_type.lower() == "discrete":
                progress_traj.target_progress = convert_continuous_to_discrete_bins(
                    progress_traj.target_progress, self.config.progress_discrete_bins
                )
            # Also set success labels to 0.0 (predict 0 success for different task trajectories)
            if progress_traj.success_label is not None:
                progress_traj.success_label = [0.0] * len(progress_traj.success_label)

        strategy_value = strategy_used.value if isinstance(strategy_used, DataGenStrat) else strategy_used
        sample = ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
            data_gen_strategy=strategy_value,
        )
        sample.resample_attempts = attempt
        return sample
