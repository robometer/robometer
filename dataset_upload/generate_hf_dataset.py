#!/usr/bin/env python3
"""
Main dataset converter that can convert any dataset to HuggingFace format for Robometer model training.
This is a generic converter that works with any dataset-specific loader.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide INFO/WARN/ERROR; only FATAL remains
import multiprocessing as mp

import numpy as np
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Optional

from pyrallis import wrap
from tqdm import tqdm

import datasets
from datasets import Dataset

# from robometer.data.dataset_types import Trajectory  # not needed, just type hint
from dataset_upload.helpers import (
    create_hf_trajectory,
    create_output_directory,
    flatten_task_data,
    load_sentence_transformer_model,
)
from huggingface_hub import HfApi

# make sure these come after importing torch. otherwise something breaks...
try:
    import absl.logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def push_hf_dataset_and_video_files_to_hub(dataset, hub_repo_id, hub_token, dataset_name, output_dir):
    print(f"Pushing dataset to HuggingFace Hub: {hub_repo_id}")
    dataset.push_to_hub(
        hub_repo_id,
        config_name=dataset_name.lower(),
        token=hub_token,
        private=False,
        commit_message=f"Add {dataset_name} dataset for Robometer training",
    )
    print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
    api = HfApi(token=hub_token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=hub_repo_id,
        repo_type="dataset",
        num_workers=min(4, cpu_count()),
    )
    print(f"✅ Successfully pushed video files for {dataset_name} to: https://huggingface.co/datasets/{hub_repo_id}")


def get_trajectory_subdir_path(trajectory_idx: int, files_per_subdir: int = 1000) -> str:
    """
    Generate subdirectory path for a trajectory to avoid too many files per directory.

    Args:
        trajectory_idx: Index of the trajectory
        files_per_subdir: Maximum files per subdirectory (default: 1000)

    Returns:
        str: Subdirectory name like 'batch_0000'
    """
    subdir_index = trajectory_idx // files_per_subdir
    return f"batch_{subdir_index:04d}"


# Global dataset features definition
BASE_FEATURES = {
    "id": datasets.Value("string"),
    "task": datasets.Value("string"),
    "lang_vector": datasets.Sequence(datasets.Value("float32")),
    "data_source": datasets.Value("string"),
    "frames": None,  # Will be set based on use_video parameter
    "is_robot": datasets.Value("bool"),
    "quality_label": datasets.Value("string"),
    # "preference_group_id": datasets.Value("string"),
    # "preference_rank": datasets.Value("int32"),
    "partial_success": datasets.Value("float32"),  # in [0, 1]
}


@dataclass
class DatasetConfig:
    """Config for dataset settings"""

    dataset_path: str = field(default="", metadata={"help": "Path to the dataset"})
    dataset_name: str = field(default=None, metadata={"help": "Name of the dataset (defaults to dataset_type)"})
    exclude_wrist_cam: bool = field(default=False, metadata={"help": "Exclude wrist camera views (MIT Franka only)"})


@dataclass
class OutputConfig:
    """Config for output settings"""

    output_dir: str = field(default="robometer_dataset", metadata={"help": "Output directory for the dataset"})
    max_trajectories: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of trajectories to process (None for all)"}
    )
    max_frames: int = field(
        default=64, metadata={"help": "Maximum number of frames per trajectory (-1 for no downsampling)"}
    )
    use_video: bool = field(default=True, metadata={"help": "Use MP4 videos instead of individual frame images"})
    shortest_edge_size: Optional[int] = field(default=240, metadata={"help": "Shortest edge size for video resizing"})
    center_crop: bool = field(
        default=False,
        metadata={"help": "Center crop the video to the target size. Defaults to False, which means no cropping."},
    )
    fps: int = field(default=10, metadata={"help": "Frames per second for video creation"})
    num_workers: int = field(
        default=-1, metadata={"help": "Number of parallel workers for processing (-1 for auto, 0 for sequential)"}
    )


@dataclass
class HubConfig:
    """Config for HuggingFace Hub settings"""

    push_to_hub: bool = field(default=False, metadata={"help": "Push dataset to HuggingFace Hub"})
    hub_repo_id: str = field(default=None, metadata={"help": "HuggingFace Hub repository ID"})
    hub_token: str = field(
        default=None, metadata={"help": "HuggingFace Hub token (or set HF_TOKEN environment variable)"}
    )


@dataclass
class GenerateConfig:
    """Main configuration for dataset generation"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def process_single_trajectory(args):
    """
    Worker function to process a single trajectory.

    Args:
        args: Tuple containing (trajectory_idx, trajectory, lang_vector, hf_creator_fn, output_dir, dataset_name, max_frames, use_video, fps)

    Returns:
        Dict: Processed trajectory data or None if failed
    """
    trajectory_idx, trajectory, lang_vector, hf_creator_fn, output_dir, dataset_name, max_frames, use_video, fps = args

    try:
        # Create output directory for this trajectory with subdirectory structure
        subdir_name = get_trajectory_subdir_path(trajectory_idx)
        full_video_path = os.path.join(
            output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
        )
        relative_video_path = os.path.join(dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4")
        os.makedirs(os.path.dirname(full_video_path), exist_ok=True)

        # Process trajectory (lang_vector is already computed)
        processed_trajectory = hf_creator_fn(
            traj_dict=trajectory,
            video_path=full_video_path,
            lang_vector=lang_vector,  # Pre-computed language vector
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=use_video,
            fps=fps,
        )

        if processed_trajectory is None:
            return None

        # Replace the full path with relative path in the processed trajectory
        if processed_trajectory and "frames" in processed_trajectory:
            processed_trajectory["frames"] = relative_video_path

        return processed_trajectory

    except Exception as e:
        print(f"❌ Error processing trajectory {trajectory_idx}: {e}")
        return None


def convert_dataset_to_hf_format(
    trajectories: list[dict],
    hf_creator_fn: Callable[[dict, str, str, int, Any, int, str], Any],
    output_dir: str = "robometer_dataset",
    dataset_name: str = "",
    max_trajectories: int | None = None,
    max_frames: int = -1,
    use_video: bool = True,
    fps: int = 10,
    num_workers: int = -1,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hub_token: str | None = None,
) -> Dataset:
    """Convert a list of trajectories to HuggingFace format."""

    print(f"Converting {dataset_name} dataset to HuggingFace format...")

    # Create output directory
    create_output_directory(output_dir)

    # Validate input
    if not trajectories:
        raise ValueError(f"No trajectories provided for {dataset_name} dataset.")

    print(f"Processing {len(trajectories)} trajectories")

    # Limit trajectories if specified
    if max_trajectories != -1:
        trajectories = trajectories[:max_trajectories]

    # Determine number of workers
    if num_workers == -1:
        num_workers = min(cpu_count(), len(trajectories))
    elif num_workers == 0:
        num_workers = 1  # Sequential processing

    print(f"Using {num_workers} worker(s) for parallel processing")

    # Pre-compute language embeddings to avoid loading sentence transformer in each worker
    print("Pre-computing language embeddings...")
    lang_model = load_sentence_transformer_model()

    lang_vectors = []
    unique_tasks = {}  # Cache for identical task descriptions

    for trajectory in tqdm(trajectories, desc="Computing language embeddings"):
        task_description = trajectory["task"]

        # Use cache to avoid recomputing identical task descriptions
        if task_description not in unique_tasks:
            unique_tasks[task_description] = lang_model.encode(task_description)

        lang_vectors.append(unique_tasks[task_description])

    print(f"Computed embeddings for {len(unique_tasks)} unique task descriptions")

    # Process trajectories
    all_entries = []

    if num_workers == 1:
        # Sequential processing (using pre-computed embeddings)
        for trajectory_idx, (trajectory, lang_vector) in enumerate(
            tqdm(zip(trajectories, lang_vectors, strict=False), desc="Processing trajectories")
        ):
            # Create output directory for this trajectory with subdirectory structure
            subdir_name = get_trajectory_subdir_path(trajectory_idx)
            trajectory_dir = os.path.join(
                output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
            )
            os.makedirs(os.path.dirname(trajectory_dir), exist_ok=True)

            processed_trajectory = hf_creator_fn(
                traj_dict=trajectory,
                video_path=trajectory_dir,
                lang_vector=lang_vector,  # Pre-computed language vector
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=use_video,
                fps=fps,
            )
            if processed_trajectory is None:
                continue
            all_entries.append(processed_trajectory)
    else:
        # Parallel processing
        all_entries = []  # ensure defined if Pool raises before we filter results
        print(f"Preparing {len(trajectories)} trajectories for parallel processing...")

        # Prepare arguments for worker processes
        worker_args = []
        for trajectory_idx, (trajectory, lang_vector) in enumerate(zip(trajectories, lang_vectors, strict=False)):
            args = (
                trajectory_idx,
                trajectory,
                lang_vector,  # Pre-computed language vector
                hf_creator_fn,
                output_dir,
                dataset_name,
                max_frames,
                use_video,
                fps,
            )
            worker_args.append(args)

        # Use spawn to avoid CUDA context issues from forking after TF import
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Process trajectories in parallel
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_single_trajectory, worker_args),
                    total=len(worker_args),
                    desc="Processing trajectories",
                )
            )

        # Filter out failed trajectories (None results)
        all_entries = [result for result in results if result is not None]

        if len(all_entries) < len(trajectories):
            failed_count = len(trajectories) - len(all_entries)
            print(f"⚠️  {failed_count} trajectories failed to process and were skipped")

    # Create HuggingFace dataset with proper features
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")

    # Convert list of entries to dictionary format for from_dict()
    data_dict = {
        "id": [entry["id"] for entry in all_entries],
        "task": [entry["task"] for entry in all_entries],
        "lang_vector": [entry["lang_vector"] for entry in all_entries],
        "data_source": [entry["data_source"] for entry in all_entries],
        "frames": [entry["frames"] for entry in all_entries],
        "is_robot": [entry["is_robot"] for entry in all_entries],
        "quality_label": [entry.get("quality_label") for entry in all_entries],
        "partial_success": [entry.get("partial_success") for entry in all_entries],
        # "preference_group_id": [entry.get("preference_group_id") for entry in all_entries],
        # "preference_rank": [entry.get("preference_rank") for entry in all_entries],
    }

    # Set frames feature based on video mode
    features_dict = BASE_FEATURES.copy()
    if use_video:
        features_dict["frames"] = datasets.Value("string")  # Video file paths as strings
    else:
        features_dict["frames"] = datasets.Sequence(datasets.Image())

    features = datasets.Features(features_dict)
    dataset = Dataset.from_dict(data_dict, features=features)

    print(f"{dataset_name} HuggingFace dataset created successfully!")
    print(f"Total entries: {len(all_entries)}")

    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing dataset to HuggingFace Hub: {hub_repo_id}")
        try:
            # Push the dataset to the hub with dataset name as config name
            dataset.push_to_hub(
                hub_repo_id,
                config_name=dataset_name.lower(),  # Use dataset name as config name
                token=hub_token,
                private=False,
                commit_message=f"Add {dataset_name} dataset for Robometer training",
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            print(f"📁 Dataset available as config: {dataset_name.lower()}")

            # Also push the video files folder to the hub
            print("\nPushing video files to HuggingFace Hub...")
            from huggingface_hub import HfApi

            api = HfApi(token=hub_token)

            # Upload the entire output directory (which contains all the video files)
            api.upload_large_folder(
                folder_path=output_dir,
                repo_id=hub_repo_id,
                repo_type="dataset",
                # commit_message=f"Add video files for {dataset_name} dataset"
            )
            print(f"✅ Successfully pushed video files to: https://huggingface.co/datasets/{hub_repo_id}")

        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Dataset was created locally but failed to push to hub")
    elif push_to_hub and not hub_repo_id:
        print("❌ push_to_hub=True but no hub_repo_id provided")
    else:
        # Only save locally if not pushing to hub (to avoid redundant Arrow files)
        dataset_path = os.path.join(output_dir, dataset_name.lower())
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved locally to: {dataset_path}")

    return dataset


@wrap()
def main(cfg: GenerateConfig):
    """Main function to convert any dataset to HuggingFace format."""

    # Get hub token from environment if not provided
    if cfg.hub.hub_token is None:
        cfg.hub.hub_token = os.getenv("HF_TOKEN")

    # Only require HF_USERNAME if pushing to hub
    if cfg.hub.push_to_hub:
        username = os.getenv("HF_USERNAME")
        if not username:
            raise ValueError(
                "HF_USERNAME is not set. Please export it to push to the Hub, or set hub.push_to_hub=false."
            )
        if cfg.hub.hub_repo_id:
            cfg.hub.hub_repo_id = username + "/" + cfg.hub.hub_repo_id

    # Import the appropriate dataset loader and trajectory creator
    if "icl" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.icl_loader import load_icl_dataset

        task_data = load_icl_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "libero" in cfg.dataset.dataset_name:
        from dataset_upload.dataset_loaders.libero_loader import load_libero_dataset

        # Load the trajectories using the loader
        task_data = load_libero_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "agibotworld" in (cfg.dataset.dataset_name or "").lower():
        # Stream + convert directly inside the AgiBotWorld loader
        from dataset_upload.dataset_loaders.agibotworld_loader import (
            convert_agibotworld_streaming_to_hf,
        )

        dataset = convert_agibotworld_streaming_to_hf(
            dataset_name=cfg.dataset.dataset_path,
            output_dir=cfg.output.output_dir,
            dataset_label=cfg.dataset.dataset_name or "agibotworld",
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )
        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                # Push the arrow table
                dataset.push_to_hub(
                    cfg.hub.hub_repo_id,
                    config_name=(cfg.dataset.dataset_name or "agibotworld").lower(),
                    token=cfg.hub.hub_token,
                    private=False,
                    commit_message=f"Add {cfg.dataset.dataset_name} dataset for Robometer training",
                )
                print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}")

                # Push the large video folder(s)
                print("\nPushing video files to HuggingFace Hub...")
                from huggingface_hub import HfApi

                api = HfApi(token=cfg.hub.hub_token)
                api.upload_large_folder(
                    folder_path=cfg.output.output_dir,
                    repo_id=cfg.hub.hub_repo_id,
                    repo_type="dataset",
                )
                print(f"✅ Successfully pushed video files to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}")
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name or "agibotworld").lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return

    elif "egodex" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.egodex_loader import load_egodex_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading EgoDex dataset from: {cfg.dataset.dataset_path}")
        task_data = load_egodex_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif cfg.dataset.dataset_name.lower().startswith("oxe_"):
        # Treat OXE like AgiBotWorld: create videos and HF entries directly in the loader
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from dataset_upload.dataset_loaders.oxe_loader import convert_oxe_dataset_to_hf

        print(f"Converting OXE dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_oxe_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return
    elif "robofail" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.robofail_loader import load_robofail_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading RoboFail dataset from: {cfg.dataset.dataset_path}")
        task_data = load_robofail_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "metaworld" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.mw_collected_loader import load_metaworld_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading metaworld dataset from: {cfg.dataset.dataset_path}")
        task_data = load_metaworld_dataset(
            cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
        )
        trajectories = flatten_task_data(task_data)
    elif "h2r" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly inside the H2R loader (OXE-style)
        from dataset_upload.dataset_loaders.h2r_loader import convert_h2r_dataset_to_hf

        print(f"Converting H2R dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_h2r_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return
    elif "fino_net" in cfg.dataset.dataset_name.lower() or "fino-net" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly inside the FinoNet loader (H2R/OXE-style)
        from dataset_upload.dataset_loaders.fino_net_loader import convert_fino_net_dataset_to_hf

        print(f"Converting FinoNet dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_fino_net_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return
    elif "epic" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (H2R/OXE-style)
        from dataset_upload.dataset_loaders.epic_loader import convert_epic_dataset_to_hf

        print(f"Converting EPIC-KITCHENS dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_epic_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
            shortest_edge_size=cfg.output.shortest_edge_size,
            center_crop=cfg.output.center_crop,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "roboarena" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.roboarena_loader import load_roboarena_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading RoboArena dataset from: {cfg.dataset.dataset_path}")
        task_data = load_roboarena_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "ph2d" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.ph2d_loader import load_ph2d_dataset

        print(f"Loading Ph2d dataset from: {cfg.dataset.dataset_path}")
        task_data = load_ph2d_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "galaxea" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (OXE-style, multi-dataset)
        from dataset_upload.dataset_loaders.galaxea_loader import convert_galaxea_dataset_to_hf

        rlds_datasets = getattr(cfg.dataset, "rlds_datasets", []) or []
        print(f"Converting Galaxea RLDS to HF from: {cfg.dataset.dataset_path} | datasets={rlds_datasets}")
        dataset = convert_galaxea_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                # remove the galaxea_rfm prefix from the dataset name because otherwise it won't match the video folder name
                # don't need to do this for OXE or others because I processed it in their loaders but forgot for this.
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "molmoact" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (LeRobot parquet)
        from dataset_upload.dataset_loaders.molmoact_loader import convert_molmoact_dataset_to_hf

        print(f"Converting MolmoAct dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_molmoact_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
        )

        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "auto_eval" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.autoeval_loader import load_autoeval_dataset

        print(f"Loading AutoEval dataset from: {cfg.dataset.dataset_path}")
        task_data = load_autoeval_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "usc_xarm_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.usc_xarm_policy_ranking_loader import (
            load_usc_xarm_policy_ranking_dataset,
        )

        print(f"Loading USC xArm Policy Ranking dataset from: {cfg.dataset.dataset_path}")
        task_data = load_usc_xarm_policy_ranking_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "usc_franka_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.usc_franka_policy_ranking_loader import (
            load_usc_franka_policy_ranking_dataset,
        )

        print(f"Loading USC Franka Policy Ranking dataset from: {cfg.dataset.dataset_path}")
        task_data = load_usc_franka_policy_ranking_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "utd_so101_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.utd_so101_loader import (
            load_utd_so101_dataset,
        )

        print(f"Loading UTD SO101 robot dataset from: {cfg.dataset.dataset_path}")
        task_data = load_utd_so101_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
            is_robot=True,
            data_source="utd_so101",
        )
        trajectories = flatten_task_data(task_data)
    elif "utd_so101_human" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.utd_so101_loader import (
            load_utd_so101_dataset,
        )

        print(f"Loading UTD SO101 human dataset from: {cfg.dataset.dataset_path}")
        task_data = load_utd_so101_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
            is_robot=False,
            data_source="utd_so101_human",
        )
        trajectories = flatten_task_data(task_data)
    elif "soar" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.soar_loader import convert_soar_dataset_to_hf

        print(f"Converting SOAR RLDS (local) to HF from: {cfg.dataset.dataset_path} ")
        dataset = convert_soar_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "mit_franka_p-rank" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.mit_franka_prank_loader import convert_mit_franka_prank_dataset_to_hf

        print(f"Converting MIT-Franka-Prank dataset to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_mit_franka_prank_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "rfm_new_mit_franka" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.new_mit_franka_loader import convert_new_mit_franka_dataset_to_hf

        print(f"Converting New MIT Franka dataset to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_new_mit_franka_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
            exclude_wrist_cam=cfg.dataset.exclude_wrist_cam,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "utd_so101_clean_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.utd_so101_clean_policy_ranking_loader import (
            convert_utd_so101_clean_policy_ranking_to_hf,
        )

        # Determine view from dataset name
        if "wrist" in cfg.dataset.dataset_name.lower():
            view = "wrist"
        elif "top" in cfg.dataset.dataset_name.lower():
            view = "top"
        else:
            raise ValueError(f"Dataset name must specify view (wrist or top): {cfg.dataset.dataset_name}")

        print(f"Converting UTD SO101 Clean Policy Ranking ({view} view) to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_utd_so101_clean_policy_ranking_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            view=view,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "usc_koch_human_robot_paired" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.usc_koch_human_robot_paired_loader import (
            convert_usc_koch_human_robot_paired_to_hf,
        )

        # Determine trajectory type from dataset name
        if "usc_koch_human_robot_paired_human" in cfg.dataset.dataset_name.lower():
            trajectory_type = "human"
        elif "usc_koch_human_robot_paired_robot" in cfg.dataset.dataset_name.lower():
            trajectory_type = "robot"
        else:
            raise ValueError(
                f"Dataset name must specify either 'usc_koch_human_robot_paired_human' or 'usc_koch_human_robot_paired_robot': {cfg.dataset.dataset_name}. "
            )

        print(f"Converting USC Koch Human-Robot Paired ({trajectory_type}) to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_usc_koch_human_robot_paired_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            trajectory_type=trajectory_type,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            updated_repo_id = cfg.hub.hub_repo_id.replace("usc_koch_human_robot_paired_", "")
            print(f"\nPushing dataset to HuggingFace Hub: {updated_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, updated_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(output_dir_override, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "usc_koch_p_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.usc_koch_p_ranking_loader import (  # type: ignore
            convert_usc_koch_p_ranking_to_hf,
        )

        output_dir_override = os.path.join(os.path.dirname(cfg.output.output_dir), cfg.dataset.dataset_name.lower())

        print(f"Converting USC Koch P-Ranking to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_usc_koch_p_ranking_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=output_dir_override,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            updated_repo_id = cfg.hub.hub_repo_id.replace("usc_koch_p_ranking_rfm", "")
            print(f"\nPushing dataset to HuggingFace Hub: {updated_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, updated_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, output_dir_override
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push metadata to hub")
        else:
            dataset_path_local = os.path.join(output_dir_override, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path_local)
            print(f"Dataset saved locally to: {dataset_path_local}")
        print("Dataset conversion complete!")
        return
    elif "egocot" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.egocot_loader import load_egocot_dataset

        # Load the trajectories using the loader
        print(f"Loading EgoCoT dataset from: {cfg.dataset.dataset_path}")
        task_data = load_egocot_dataset(
            cfg.dataset.dataset_path,
        )
        trajectories = flatten_task_data(task_data)
    elif "humanoid_everyday" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (OXE-style)
        from dataset_upload.dataset_loaders.humanoid_everyday_loader import convert_humanoid_everyday_dataset_to_hf

        print(f"Converting Humanoid Everyday dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_humanoid_everyday_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                push_hf_dataset_and_video_files_to_hub(
                    dataset, cfg.hub.hub_repo_id, cfg.hub.hub_token, cfg.dataset.dataset_name, cfg.output.output_dir
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return
    elif "motif" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.motif_loader import load_motif_dataset

        print(f"Loading MotIF dataset from: {cfg.dataset.dataset_path}")
        task_data = load_motif_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "failsafe" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.failsafe_loader import load_failsafe_dataset

        print(f"Loading FailSafe dataset from: {cfg.dataset.dataset_path}")
        task_data = load_failsafe_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "racer" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.racer_loader import load_racer_dataset

        print(f"Loading RACER dataset from: {cfg.dataset.dataset_path}")
        task_data = load_racer_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "hand_paired" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.hand_paired_loader import load_hand_paired_dataset

        print(f"Loading HAND_paired dataset from: {cfg.dataset.dataset_path}")
        task_data = load_hand_paired_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "roboreward" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.roboreward_loader import load_roboreward_dataset

        print(f"Loading RoboReward dataset from: {cfg.dataset.dataset_path}")
        task_data = load_roboreward_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "robofac" in cfg.dataset.dataset_name.lower():
        from dataset_upload.dataset_loaders.robofac_loader import load_robofac_dataset

        print(f"Loading RoboFAC dataset from: {cfg.dataset.dataset_path}")
        task_data = load_robofac_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_name}")

    # Convert dataset (non-streaming datasets)
    convert_dataset_to_hf_format(
        trajectories=trajectories,
        hf_creator_fn=partial(
            create_hf_trajectory,
            dataset_name=cfg.dataset.dataset_name,
            use_video=cfg.output.use_video,
            fps=cfg.output.fps,
            shortest_edge_size=cfg.output.shortest_edge_size,
            center_crop=cfg.output.center_crop,
            hub_repo_id=cfg.hub.hub_repo_id,
        ),
        output_dir=cfg.output.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        use_video=cfg.output.use_video,
        fps=cfg.output.fps,
        num_workers=cfg.output.num_workers,
        push_to_hub=cfg.hub.push_to_hub,
        hub_repo_id=cfg.hub.hub_repo_id,
        hub_token=cfg.hub.hub_token,
    )

    print("Dataset conversion complete!")


if __name__ == "__main__":
    main()
