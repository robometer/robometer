import random

from robometer.data.datasets.base import BaseDataset
from robometer.data.samplers.pref import PrefSampler
from robometer.data.samplers.progress import ProgressSampler
from robometer.data.dataset_category import is_preference_only
from robometer.utils.logger import get_logger

logger = get_logger()


class RBMDataset(BaseDataset):
    """Dataset that combines preference and progress generation."""

    def __init__(self, config, is_evaluation=False, max_samples=None, sampler_kwargs=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        self.pref_sampler = None
        self.progress_sampler = None

        if sampler_kwargs is None:
            sampler_kwargs = {}

        base_sampler_kwargs = {
            "config": config,
            "dataset": self.dataset,
            "combined_indices": self._combined_indices,
            "dataset_success_cutoff_map": self.dataset_success_cutoff_map,
            "verbose": False,
            **sampler_kwargs,
        }

        pref_cls = getattr(type(self), "_pref_sampler_cls", PrefSampler)
        prog_cls = getattr(type(self), "_progress_sampler_cls", ProgressSampler)
        if self.config.sample_type_ratio[0] > 0:
            self.pref_sampler = pref_cls(is_evaluation=is_evaluation, **base_sampler_kwargs)
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = prog_cls(is_evaluation=is_evaluation, **base_sampler_kwargs)

        st = getattr(self, "_lerobot_store", None)
        if st is not None:
            for _s in (self.pref_sampler, self.progress_sampler):
                if _s is not None:
                    _s._lerobot_store = st

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples

        self.data_len = len(self.dataset)

    def get_random_state(self) -> dict:
        """Get random state from all samplers for checkpointing.

        Returns:
            Dictionary containing random state for all samplers
        """
        state = {
            "pref_sampler": self.pref_sampler._local_random.getstate() if self.pref_sampler else None,
            "progress_sampler": self.progress_sampler._local_random.getstate() if self.progress_sampler else None,
        }
        return state

    def set_random_state(self, state: dict):
        """Restore random state from checkpoint.

        Args:
            state: Dictionary containing random state for all samplers
        """
        if state.get("pref_sampler") and self.pref_sampler:
            self.pref_sampler._local_random.setstate(state["pref_sampler"])
        if state.get("progress_sampler") and self.progress_sampler:
            self.progress_sampler._local_random.setstate(state["progress_sampler"])

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a data sample from the dataset."""
        idx = idx % self.data_len
        logger.trace(f"[RBMDataset] __getitem__: Starting for idx={idx}")

        # Get the item from the filtered dataset
        item = self.dataset[idx]
        traj_id = item.get("id", "unknown")
        logger.trace(f"[RBMDataset] __getitem__: Got item with ID={traj_id}, calling _generate_sample_from_item")

        sample = self._generate_sample_from_item(item)
        logger.trace(f"[RBMDataset] __getitem__: Successfully generated sample for idx={idx}, ID={traj_id}")
        return sample

    def _generate_sample_from_item(self, item):
        """Shared sampler logic that can be reused by balanced datasets."""
        traj_id = item["id"]
        data_source = item["data_source"]
        quality_label = item["quality_label"]
        task_name = item["task"]

        logger.trace(
            f"[RBMDataset] _generate_sample_from_item: Starting for ID={traj_id}, data_source={data_source}, quality={quality_label}"
        )

        # Force preference-only for non-successful trajectories
        if quality_label != "successful" and self.pref_sampler is not None:
            logger.trace(
                f"[RBMDataset] _generate_sample_from_item: Non-successful quality detected for ID={traj_id}, forcing preference-only"
            )
            sample = self.pref_sampler._generate_sample(item)
            if sample is not None:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: Preference sample generated successfully for non-successful traj ID={traj_id}"
                )
                return self._set_resample_attempts(sample, 1)
            else:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: Preference sampler returned None for non-successful traj ID={traj_id}"
                )
                # If preference fails for non-successful traj, we can't use other samplers
                raise ValueError(
                    f"Preference sampler failed for non-successful trajectory ID={traj_id} and no fallback available, task={task_name}"
                )

        # Preference-only override by data_source using raw filtered dataset entry
        if is_preference_only(data_source) and self.pref_sampler is not None:
            logger.trace(f"[RBMDataset] _generate_sample_from_item: Using preference-only override for ID={traj_id}")
            sample = self.pref_sampler._generate_sample(item)
            if sample is not None:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: Preference-only sample generated successfully for ID={traj_id}"
                )
                return self._set_resample_attempts(sample, 1)
            else:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: Preference-only sampler returned None for ID={traj_id}, task={task_name}"
                )

        # Available samplers with their probabilities
        samplers = [
            ("pref", self.sample_type_ratio[0], self.pref_sampler),
            ("progress", self.sample_type_ratio[1], self.progress_sampler),
        ]

        # Remove samplers with zero probability or None samplers
        available_samplers = [
            (name, prob, sampler) for name, prob, sampler in samplers if prob > 0 and sampler is not None
        ]

        logger.trace(
            f"[RBMDataset] _generate_sample_from_item: Available samplers for ID={traj_id}: {[name for name, _, _ in available_samplers]}"
        )

        # Fallback to progress sampler if no samplers have positive probability
        if not available_samplers:
            if self.progress_sampler is not None:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: No available samplers, using progress fallback for ID={traj_id}"
                )
                sample = self.progress_sampler._generate_sample(item)
                if sample is not None:
                    return self._set_resample_attempts(sample, 1)
            raise ValueError("No samplers available")

        # Try samplers until we get a non-None result
        # Limit max_attempts to prevent hangs with num_workers=0 in distributed training
        # With num_workers=0, blocking here blocks the entire rank's main thread
        max_attempts = min(len(available_samplers) * 2, 10)  # Cap at 10 to prevent infinite loops
        attempt = 0
        tried_samplers = set()

        logger.trace(
            f"[RBMDataset] _generate_sample_from_item: Starting sampling loop for ID={traj_id}, max_attempts={max_attempts}"
        )

        while attempt < max_attempts:
            attempt += 1

            # If we've tried all samplers, reset and try again
            if len(tried_samplers) >= len(available_samplers):
                tried_samplers.clear()

            # Filter out already tried samplers if we haven't exhausted all options
            remaining_samplers = [
                (name, prob, sampler) for name, prob, sampler in available_samplers if name not in tried_samplers
            ]

            # If no remaining samplers, reset and try all again
            if not remaining_samplers:
                tried_samplers.clear()
                remaining_samplers = available_samplers

            # Normalize probabilities for remaining samplers
            total_prob = sum(prob for _, prob, _ in remaining_samplers)
            if total_prob == 0:
                # Reset and try all samplers again
                tried_samplers.clear()
                remaining_samplers = available_samplers
                total_prob = sum(prob for _, prob, _ in remaining_samplers)

            normalized_samplers = [(name, prob / total_prob, sampler) for name, prob, sampler in remaining_samplers]

            # Select sampler based on normalized probabilities
            prob = random.random()
            cumulative_prob = 0.0
            selected_sampler = None
            selected_name = None

            for name, normalized_prob, sampler in normalized_samplers:
                cumulative_prob += normalized_prob
                if prob <= cumulative_prob:
                    selected_sampler = sampler
                    selected_name = name
                    break

            # Fallback: select first sampler if selection failed
            if selected_sampler is None:
                selected_name, _, selected_sampler = remaining_samplers[0]

            logger.trace(
                f"[RBMDataset] _generate_sample_from_item: Attempt {attempt}/{max_attempts} for ID={traj_id}, trying sampler '{selected_name}'"
            )

            # Try the selected sampler
            sample = selected_sampler._generate_sample(item)

            # If sample is not None, return it
            if sample is not None:
                logger.trace(
                    f"[RBMDataset] _generate_sample_from_item: Successfully generated sample on attempt {attempt} using '{selected_name}' for ID={traj_id}"
                )
                return self._set_resample_attempts(sample, attempt)

            # Sample is None, mark this sampler as tried
            logger.trace(
                f"[RBMDataset] _generate_sample_from_item: Attempt {attempt} failed (sampler '{selected_name}' returned None) for ID={traj_id}"
            )
            tried_samplers.add(selected_name)

        logger.trace(
            f"[RBMDataset] _generate_sample_from_item: All {max_attempts} attempts exhausted for ID={traj_id}, trying progress fallback"
        )

        # All attempts failed, try progress sampler as final fallback
        if self.progress_sampler is not None:
            sample = self.progress_sampler._generate_sample(item)
            if sample is not None:
                logger.trace(f"[RBMDataset] _generate_sample_from_item: Progress fallback succeeded for ID={traj_id}")
                return self._set_resample_attempts(sample, attempt)

        # Final fallback: raise error if all samplers returned None
        logger.error(
            f"[RBMDataset] _generate_sample_from_item: ERROR - All samplers failed for ID={traj_id} after {max_attempts} attempts"
        )
        raise ValueError(f"All samplers failed to generate a sample after {max_attempts} attempts")
