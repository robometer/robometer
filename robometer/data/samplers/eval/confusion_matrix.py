#!/usr/bin/env python3
"""
Data generator for confusion matrix analysis.
"""

import random
import torch
from collections import Counter, defaultdict
from typing import Tuple

from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.data.samplers.base import RBMBaseSampler
from robometer.utils.distributed import rank_0_print


class ConfusionMatrixSampler(RBMBaseSampler):
    """
    Data generator that creates task-trajectory pairs for confusion matrix analysis.

    For each unique task, creates samples with each trajectory to analyze
    how well the model can distinguish between different tasks.

    If multiple data sources are present, samples N random trajectories from each data source
    and prioritizes different language instructions by randomizing the pairing order.
    """

    def __init__(self, n_trajectories_per_source: int = None, **kwargs):
        """Initialize confusion matrix sampler.

        Args:
            n_trajectories_per_source: Number of trajectories to sample from each data source.
                If None, uses all available trajectories.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.n_trajectories_per_source = n_trajectories_per_source

        # Imported lazily -- sentence_transformers pulls in transformers' torchao quantizer
        # support, which crashes on import in this venv's torch==2.8.0 (torchao>=0.18 needs
        # torch>=2.11's torch.nn.functional.ScalingType). Deferring the import here means only
        # code that actually instantiates this sampler pays that cost / hits that crash.
        from sentence_transformers import SentenceTransformer

        # Load sentence transformer model and precompute embeddings for all unique tasks
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        self.sentence_model.eval()

        # Precompute language embeddings for all unique tasks
        unique_tasks = list(self.task_indices.keys())
        rank_0_print(f"Precomputing language embeddings for {len(unique_tasks)} unique tasks", verbose=self.verbose)
        self.task_embeddings = {}
        for task in unique_tasks:
            embedding = self.sentence_model.encode(task)
            self.task_embeddings[task] = torch.tensor(embedding)
        rank_0_print(f"Precomputed {len(self.task_embeddings)} language embeddings", verbose=self.verbose)

        # Free up the model after precomputation (no longer needed)
        del self.sentence_model

        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} confusion matrix sample indices from {len(self.robot_trajectories)} trajectories and {len(self.task_indices)} tasks"
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible task-trajectory pair sample indices.

        If multiple data sources exist, samples N random trajectories from each data source.
        Prioritizes different video tasks first, then prioritizes different language instructions
        when creating pairs.
        """
        sample_indices = []

        # Get unique tasks (these will be the language instructions)
        unique_lang_tasks = list(self.task_indices.keys())
        rank_0_print(f"Found {len(unique_lang_tasks)} unique language tasks: {unique_lang_tasks}", verbose=self.verbose)

        # Sample trajectories per data source (prioritizing different video tasks)
        sampled_trajectories, stats = self._sample_trajectories_by_data_source()

        rank_0_print(
            f"Processing {len(sampled_trajectories)} trajectories for confusion matrix analysis",
            verbose=self.verbose,
        )

        # Print statistics about sampled trajectories
        self._print_sampling_stats(stats)

        # Shuffle language tasks once for round-robin pairing
        shuffled_lang_tasks = unique_lang_tasks.copy()
        self._local_random.shuffle(shuffled_lang_tasks)

        # Create task-trajectory pairs with prioritized language instruction pairing
        video_task_count = Counter()

        for traj_idx in sampled_trajectories:
            traj = self.dataset[traj_idx]
            video_task = traj["task"]

            # # Limit the number of video trajectories for each task to 5
            # if video_task_count[video_task] >= 5:
            #     continue

            video_task_count[video_task] += 1

            # Pair this trajectory with all language tasks (shuffled for variety)
            traj_id = traj.get("id", str(traj_idx))
            for lang_task in shuffled_lang_tasks:
                sample_indices.append({
                    "traj_idx": traj_idx,
                    "lang_task": lang_task,
                    "video_task": video_task,
                    "video_path": traj["frames"],
                    "id": traj_id,
                })

        # Shuffle final sample indices to further randomize the order
        self._local_random.shuffle(sample_indices)

        # Print statistics about pairs created
        rank_0_print(f"Generated {len(sample_indices)} task-trajectory pairs", verbose=self.verbose)
        rank_0_print(f"  Video tasks sampled: {dict(video_task_count)}", verbose=self.verbose)
        rank_0_print(f"  Trajectories per video task: {dict(sorted(video_task_count.items()))}", verbose=self.verbose)

        return sample_indices

    def _sample_trajectories_by_data_source(self) -> Tuple[list[int], dict]:
        """Sample N random trajectories from each data source, prioritizing different video tasks.

        When sampling N trajectories, first selects one trajectory from each unique video task,
        then repeats in round-robin fashion until N trajectories are sampled.

        Returns:
            Tuple of (list of sampled trajectory indices, stats dictionary)
        """
        sampled_indices = []
        stats = {
            "by_source": {},
            "by_task": Counter(),
            "traj_to_task": {},
        }

        # Group robot trajectories by data source, then by video task
        trajectories_by_source_and_task = defaultdict(lambda: defaultdict(list))
        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            data_source = traj.get("data_source", "unknown")
            video_task = traj.get("task", "unknown")
            trajectories_by_source_and_task[data_source][video_task].append(traj_idx)

        rank_0_print(
            f"Found {len(trajectories_by_source_and_task)} data sources: {list(trajectories_by_source_and_task.keys())}",
            verbose=self.verbose,
        )

        # Sample N trajectories from each data source, prioritizing different tasks
        for data_source, tasks_to_indices in trajectories_by_source_and_task.items():
            # Shuffle trajectories within each task for randomization
            for task in tasks_to_indices:
                self._local_random.shuffle(tasks_to_indices[task])

            # Get all unique tasks for this data source
            all_tasks = list(tasks_to_indices.keys())
            self._local_random.shuffle(all_tasks)  # Randomize task order too

            source_stats = {
                "total_available": sum(len(indices) for indices in tasks_to_indices.values()),
                "tasks_available": {task: len(indices) for task, indices in tasks_to_indices.items()},
                "tasks_sampled": Counter(),
            }

            if self.n_trajectories_per_source is None:
                # Use all available trajectories
                sampled_from_source = []
                for task, indices in tasks_to_indices.items():
                    sampled_from_source.extend(indices)
                    source_stats["tasks_sampled"][task] = len(indices)
                    stats["by_task"][task] += len(indices)

                rank_0_print(
                    f"  Data source '{data_source}': Using all {len(sampled_from_source)} trajectories",
                    verbose=self.verbose,
                )
            else:
                # Sample N trajectories using round-robin to prioritize different tasks
                n_to_sample = min(self.n_trajectories_per_source, source_stats["total_available"])
                sampled_from_source = []

                # Round-robin sampling: first get one from each task, then repeat
                task_iterators = {task: iter(indices) for task, indices in tasks_to_indices.items()}
                task_list = all_tasks.copy()
                round_idx = 0

                while len(sampled_from_source) < n_to_sample:
                    # If we've gone through all tasks once, reshuffle for next round
                    if round_idx >= len(task_list):
                        round_idx = 0
                        self._local_random.shuffle(task_list)

                    # Try to get one trajectory from current task
                    task = task_list[round_idx]
                    try:
                        traj_idx = next(task_iterators[task])
                        sampled_from_source.append(traj_idx)
                        source_stats["tasks_sampled"][task] += 1
                        stats["by_task"][task] += 1
                    except StopIteration:
                        # This task is exhausted, remove it from rotation
                        task_list.pop(round_idx)
                        if not task_list:
                            break  # All tasks exhausted
                        continue

                    round_idx += 1

                rank_0_print(
                    f"  Data source '{data_source}': Sampled {len(sampled_from_source)} out of {source_stats['total_available']} trajectories",
                    verbose=self.verbose,
                )
                rank_0_print(
                    f"    Tasks sampled: {dict(sorted(source_stats['tasks_sampled'].items()))}",
                    verbose=self.verbose,
                )

            # Track trajectory to task mapping for stats
            for traj_idx in sampled_from_source:
                traj = self.dataset[traj_idx]
                traj_id = traj.get("id", str(traj_idx))
                stats["traj_to_task"][traj_id] = traj.get("task", "unknown")

            sampled_indices.extend(sampled_from_source)
            stats["by_source"][data_source] = source_stats

        return sampled_indices, stats

    def _print_sampling_stats(self, stats: dict):
        """Print detailed statistics about sampled trajectories.

        Args:
            stats: Statistics dictionary from _sample_trajectories_by_data_source
        """
        if not self.verbose:
            return

        rank_0_print("\n=== Confusion Matrix Sampling Statistics ===", verbose=self.verbose)

        # Overall task statistics
        rank_0_print(f"\nOverall trajectories per video task:", verbose=self.verbose)
        for task, count in sorted(stats["by_task"].items()):
            rank_0_print(f"  {task}: {count} trajectories", verbose=self.verbose)

        # Per data source statistics
        rank_0_print(f"\nPer data source breakdown:", verbose=self.verbose)
        for data_source, source_stats in stats["by_source"].items():
            rank_0_print(f"  Data source: {data_source}", verbose=self.verbose)
            rank_0_print(f"    Total available: {source_stats['total_available']}", verbose=self.verbose)
            rank_0_print(f"    Tasks available: {len(source_stats['tasks_available'])}", verbose=self.verbose)
            for task, count in sorted(source_stats["tasks_available"].items()):
                sampled_count = source_stats["tasks_sampled"].get(task, 0)
                rank_0_print(
                    f"      {task}: {sampled_count}/{count} trajectories sampled",
                    verbose=self.verbose,
                )

        rank_0_print("=" * 50, verbose=self.verbose)

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> PreferenceSample:
        """Generate a single task-trajectory sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        lang_task = sample_idx_info["lang_task"]
        video_task = sample_idx_info["video_task"]
        video_path = sample_idx_info["video_path"]

        video_traj = self.dataset[traj_idx]

        # Look up precomputed embedding instead of encoding
        text_embedding = self.task_embeddings[lang_task]

        metadata = {
            "id": video_traj["id"],
            "lang_task": lang_task,
            "video_task": video_task,
            "video_path": video_path,
        }

        # Override task and text_embedding in the trajectory dict
        video_traj_with_task = video_traj.copy()
        video_traj_with_task["task"] = lang_task
        video_traj_with_task["text_embedding"] = text_embedding

        sample_trajectory = self._get_traj_from_data(
            traj=video_traj_with_task,
            metadata=metadata,
            pad_frames=self.pad_frames,
        )

        sample = ProgressSample(trajectory=sample_trajectory)
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
