#!/usr/bin/env python3
"""
Script to run baseline evaluations (GVL, RL-VLM-F, VLAC) on datasets.

Usage:
    # Run VLAC progress evaluation (requires separate dependency set due to trl conflict)
    uv run --extra vlac --python .venv-vlac/bin/python robometer/evals/run_baseline_eval.py \
        reward_model=vlac \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[jesbu1_roboreward_rfm_roboreward_test] \
        custom_eval.use_frame_steps=false \
        max_frames=8 \
        custom_eval.reward_alignment_max_trajectories=null
    
    # Run RoboReward-8B progress evaluation
    # reward_model=roboreward automatically loads configs/reward_model/roboreward.yaml
    uv run python robometer/evals/run_baseline_eval.py \
        reward_model=roboreward \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[franka] \
        custom_eval.use_frame_steps=true \
        max_frames=8
    
    # Run RBM model progress evaluation (reward alignment)
    # reward_model=rbm loads configs/reward_model/rbm.yaml
    uv run python robometer/evals/run_baseline_eval.py \
        reward_model=rbm \
        model_path=rewardfm/rbm_qwen_pref_prog_4frames_all_strategy \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[jesbu1_roboreward_rfm_roboreward_test] \
        custom_eval.reward_alignment_max_trajectories=null \
        max_frames=8 \
        model_config.batch_size=32
"""

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from robometer.configs.eval_configs import BaselineEvalConfig
from robometer.configs.experiment_configs import DataConfig
from robometer.utils.setup_utils import setup_custom_eval_dataset
from robometer.data.datasets.base import resolve_dataset_keys
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import get_logger
from robometer.utils.config_utils import display_config, convert_hydra_to_dataclass
from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.data.collators.utils import convert_frames_to_pil_images, frames_to_numpy_array
from robometer.evals.baselines.rlvlmf import RLVLMF
from robometer.evals.baselines.gvl import GVL
from robometer.evals.baselines.vlac import VLAC

from robometer.evals.baselines.robodopamine import RoboDopamine
from robometer.evals.baselines.roboreward import RoboReward
from robometer.evals.baselines.rbm_model import RBMModel
from robometer.evals.baselines.topreward import TopReward
from robometer.evals.compile_results import (
    run_quality_preference_eval,
    run_reward_alignment_eval_per_trajectory,
    run_policy_ranking_eval,
    run_confusion_matrix_eval,
)

logger = get_logger()


def _create_plot_with_video_gif(
    fig: plt.Figure,
    video_frames: Optional[np.ndarray],
    output_path: str,
    plot_width: int = 800,
    video_height: int = 224,
    fps: int = 2,
) -> None:
    """Create a GIF combining a static plot with animated video frames side by side.

    Args:
        fig: Matplotlib figure to include as static plot
        video_frames: Video frames array of shape [T, C, H, W] or [T, H, W, C]
        output_path: Path to save the GIF
        plot_width: Width of the plot in pixels
        video_height: Height of the video frames in pixels
        fps: Frames per second for the GIF
    """
    if video_frames is None or video_frames.size == 0:
        # If no video, just save the plot as PNG
        fig.savefig(output_path.replace(".gif", ".png"), dpi=150, bbox_inches="tight")
        return

    # Convert matplotlib figure to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plot_img = Image.open(buf)
    plot_img = plot_img.convert("RGB")

    # Resize plot to desired width while maintaining aspect ratio
    plot_aspect = plot_img.height / plot_img.width
    plot_height = int(plot_width * plot_aspect)
    plot_img = plot_img.resize((plot_width, plot_height), Image.Resampling.LANCZOS)

    # Process video frames
    # video_frames is [T, C, H, W] - need to convert to [T, H, W, C] for PIL
    if video_frames.ndim == 4:
        if video_frames.shape[1] == 3 or video_frames.shape[1] == 1:  # [T, C, H, W]
            video_frames = video_frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
        # Now it's [T, H, W, C]

    # Resize video frames to match video_height
    num_frames = video_frames.shape[0]
    frame_height, frame_width = video_frames.shape[1], video_frames.shape[2]
    video_aspect = frame_height / frame_width
    video_width = int(video_height / video_aspect)

    # Create combined frames
    combined_frames = []
    for t in range(num_frames):
        frame = video_frames[t]

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        if frame.shape[2] == 1:  # Grayscale
            frame_pil = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
        else:
            frame_pil = Image.fromarray(frame, mode="RGB")

        # Resize video frame
        frame_pil = frame_pil.resize((video_width, video_height), Image.Resampling.LANCZOS)

        # Combine plot and video side by side
        # Use the maximum height and pad if needed
        max_height = max(plot_height, video_height)
        combined = Image.new("RGB", (plot_width + video_width, max_height), color="white")

        # Paste plot on the left
        plot_y = (max_height - plot_height) // 2
        combined.paste(plot_img, (0, plot_y))

        # Paste video frame on the right
        video_y = (max_height - video_height) // 2
        combined.paste(frame_pil, (plot_width, video_y))

        # Convert to numpy array for imageio
        combined_frames.append(np.array(combined))

    # Save as GIF
    imageio.mimwrite(output_path, combined_frames, fps=fps, loop=0)
    plt.close(fig)  # Close figure to free memory


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj


def _shorten_single_name(name: str, mapping: dict) -> str:
    """Shorten a single dataset name using the mapping."""
    # Check if there's a direct mapping
    if name in mapping:
        return mapping[name]

    # Also check after normalizing (in case the input has special chars)
    normalized = name.replace("-", "_").replace("/", "_")
    if normalized in mapping:
        return mapping[normalized]

    # No mapping found, clean up the name
    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " ", ","]:
        name = name.replace(char, "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _shorten_dataset_name(dataset_name: Union[str, List[str]], max_length: int = 60) -> str:
    """Shorten dataset name for use in filenames.

    Uses DS_SHORT_NAME_MAPPING for known datasets, otherwise truncates with hash suffix.
    For lists, each element is shortened individually then combined.

    Args:
        dataset_name: Dataset name (string or list of strings)
        max_length: Maximum length for the output string (default 60)

    Returns:
        Shortened, filesystem-safe string for use in filenames
    """
    from robometer.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING

    # Handle list by shortening each element individually
    if isinstance(dataset_name, list):
        shortened_parts = [_shorten_single_name(str(x), DS_SHORT_NAME_MAPPING) for x in dataset_name]
        name_str = "_".join(shortened_parts)
    else:
        name_str = _shorten_single_name(str(dataset_name), DS_SHORT_NAME_MAPPING)

    # Collapse multiple underscores
    name_str = re.sub(r"_+", "_", name_str)
    name_str = name_str.strip("_")

    # If short enough, return as-is
    if len(name_str) <= max_length:
        return name_str

    # Generate short hash of original name for uniqueness
    original_str = "_".join(str(x) for x in dataset_name) if isinstance(dataset_name, list) else str(dataset_name)
    hash_suffix = hashlib.md5(original_str.encode()).hexdigest()[:8]

    # Truncate and add hash
    # Reserve space for hash suffix: "_" + 8 chars = 9 chars
    truncate_length = max_length - 9
    truncated = name_str[:truncate_length].rstrip("_")

    return f"{truncated}_{hash_suffix}"


def process_preference_sample(sample: PreferenceSample, model: RLVLMF) -> Dict[str, Any]:
    """Process a single preference sample with baseline."""
    chosen_traj = sample.chosen_trajectory
    rejected_traj = sample.rejected_trajectory

    # Convert frames to PIL Images
    chosen_images = convert_frames_to_pil_images(chosen_traj.frames)
    rejected_images = convert_frames_to_pil_images(rejected_traj.frames)

    assert chosen_traj.task == rejected_traj.task, "Chosen and rejected trajectories must have the same task"

    # Compute preference
    result = model.compute_preference(
        chosen_images=chosen_images,
        rejected_images=rejected_images,
        task_description=chosen_traj.task,
    )

    chosen_metadata = {
        "quality_label": chosen_traj.quality_label,
        "data_source": chosen_traj.data_source,
        "task": chosen_traj.task,
        "id": chosen_traj.id,
        "video_path": chosen_traj.frames if isinstance(chosen_traj.frames, str) else None,
    }
    if chosen_traj.partial_success is not None:
        chosen_metadata["partial_success"] = chosen_traj.partial_success

    rejected_metadata = {
        "quality_label": rejected_traj.quality_label,
        "data_source": rejected_traj.data_source,
        "task": rejected_traj.task,
        "id": rejected_traj.id,
        "video_path": rejected_traj.frames if isinstance(rejected_traj.frames, str) else None,
    }
    if rejected_traj.partial_success is not None:
        rejected_metadata["partial_success"] = rejected_traj.partial_success

    prediction_prob = result.get("prediction_prob")
    is_correct = result.get("is_correct")
    preference_pred = result.get("preference_pred")

    return {
        "preference_pred": float(preference_pred)
        if preference_pred is not None
        else (float(prediction_prob) if prediction_prob is not None else None),
        "preference_labels": 1.0,  # Always 1.0 because chosen trajectory is always preferred by construction
        "is_correct": bool(is_correct) if is_correct is not None else None,
        "task": chosen_traj.task,
        "data_source": chosen_traj.data_source or rejected_traj.data_source,
        "chosen_data_gen_strategy": chosen_traj.data_gen_strategy,
        "rejected_data_gen_strategy": rejected_traj.data_gen_strategy,
        "metadata": {
            "chosen_metadata": chosen_metadata,
            "rejected_metadata": rejected_metadata,
        },
    }


def process_progress_sample(
    sample: ProgressSample,
    model: Union[GVL, VLAC, RoboReward, RoboDopamine],
) -> Dict[str, Any]:
    """Process a single progress sample with baseline."""
    traj = sample.trajectory

    # Get frames array
    frames_array = frames_to_numpy_array(traj.frames)

    logger.info(f"Processing progress sample with {len(frames_array)} frames")

    if frames_array is None or frames_array.size == 0:
        logger.warning("No frames found in trajectory")
        return None

    progress_pred = model.compute_progress(frames_array, task_description=traj.task)

    # Convert to numpy array and normalize to [0, 1] if needed
    progress_array = np.array([p if p is not None else 0.0 for p in progress_pred])
    # Note: GVL/VLAC already return normalized [0, 1] values, so no division by 100 needed

    # Build metadata dict - get video_path and frame_step from trajectory metadata
    metadata = {}
    if traj.id is not None:
        metadata["id"] = traj.id
    if traj.metadata is not None:
        metadata.update(traj.metadata)

    # Build result dict
    result = {
        "progress_pred": progress_array,
        "task": traj.task,
        "data_source": traj.data_source,
        "data_gen_strategy": traj.data_gen_strategy,
        "metadata": metadata,
        "id": traj.id,
        "video_path": metadata.get("video_path"),
        "partial_success": traj.partial_success,
        "target_progress": np.array(traj.target_progress),
        "quality_label": traj.quality_label,
    }

    return result


def process_batched_rbm_samples(
    dataset,
    model: RBMModel,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Process RBM/ReWiND samples using batched computation with minibatching.

    Args:
        dataset: Dataset object that supports indexing (e.g., CustomEvalDataset)
        model: RBMModel instance
        batch_size: Batch size for processing samples

    Returns:
        List of result dictionaries in the same format as process_progress_sample/process_preference_sample
    """
    dataset_len = len(dataset)

    # Group indices by sample type (iterate once, only storing indices)
    progress_indices = []
    preference_indices = []

    for i in range(dataset_len):
        sample = dataset[i]
        if isinstance(sample, ProgressSample):
            progress_indices.append(i)
        elif isinstance(sample, PreferenceSample):
            preference_indices.append(i)
        else:
            logger.warning(f"Unknown sample type: {type(sample)}")

    results = []

    # Process progress samples in minibatches
    if progress_indices:
        for batch_start in tqdm(range(0, len(progress_indices), batch_size), desc="Processing progress batches"):
            batch_indices = progress_indices[batch_start : batch_start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            progress_preds = model.compute_batched_progress(batch)
            for sample, progress_pred in zip(batch, progress_preds):
                traj = sample.trajectory

                # Build metadata dict - get video_path and frame_step from trajectory metadata
                metadata = {}
                if traj.id is not None:
                    metadata["id"] = traj.id
                if traj.metadata is not None:
                    metadata.update(traj.metadata)

                # Build result dict
                result = {
                    "progress_pred": np.array(progress_pred),
                    "task": traj.task,
                    "data_source": traj.data_source,
                    "data_gen_strategy": traj.data_gen_strategy,
                    "metadata": metadata,
                    "id": traj.id,
                    "video_path": metadata.get("video_path"),
                    "partial_success": traj.partial_success,
                    "target_progress": np.array(traj.target_progress),
                    "quality_label": traj.quality_label,
                }
                results.append(result)

    # Process preference samples in minibatches
    if preference_indices:
        for batch_start in tqdm(range(0, len(preference_indices), batch_size), desc="Processing preference batches"):
            batch_indices = preference_indices[batch_start : batch_start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            preference_results = model.compute_batched_preference(batch)
            for sample, result in zip(batch, preference_results):
                chosen_traj = sample.chosen_trajectory
                rejected_traj = sample.rejected_trajectory

                chosen_metadata = {
                    "quality_label": chosen_traj.quality_label,
                    "data_source": chosen_traj.data_source,
                    "task": chosen_traj.task,
                    "id": chosen_traj.id,
                    "video_path": chosen_traj.frames if isinstance(chosen_traj.frames, str) else None,
                }
                if chosen_traj.partial_success is not None:
                    chosen_metadata["partial_success"] = chosen_traj.partial_success

                rejected_metadata = {
                    "quality_label": rejected_traj.quality_label,
                    "data_source": rejected_traj.data_source,
                    "task": rejected_traj.task,
                    "id": rejected_traj.id,
                    "video_path": rejected_traj.frames if isinstance(rejected_traj.frames, str) else None,
                }
                if rejected_traj.partial_success is not None:
                    rejected_metadata["partial_success"] = rejected_traj.partial_success

                prediction_prob = result.get("prediction_prob")
                is_correct = result.get("is_correct")
                preference_pred = result.get("preference_pred")

                formatted_result = {
                    "preference_pred": float(preference_pred)
                    if preference_pred is not None
                    else (float(prediction_prob) if prediction_prob is not None else None),
                    "preference_labels": 1.0,  # Always 1.0 because chosen trajectory is always preferred by construction
                    "is_correct": bool(is_correct) if is_correct is not None else None,
                    "task": chosen_traj.task,
                    "data_source": chosen_traj.data_source or rejected_traj.data_source,
                    "chosen_data_gen_strategy": chosen_traj.data_gen_strategy,
                    "rejected_data_gen_strategy": rejected_traj.data_gen_strategy,
                    "metadata": {
                        "chosen_metadata": chosen_metadata,
                        "rejected_metadata": rejected_metadata,
                    },
                }
                results.append(formatted_result)

    return results


def run_baseline_evaluation(cfg: BaselineEvalConfig, base_data_cfg: DataConfig) -> Dict[str, Any]:
    """Run baseline evaluation on datasets."""

    # Initialize model based on reward_model type
    model_config_dict = (
        asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config.__dict__
    )

    if cfg.reward_model == "rlvlmf":
        model = RLVLMF(**model_config_dict)
    elif cfg.reward_model == "gvl":
        # API key is read from GEMINI_API_KEY environment variable if not provided
        model = GVL(max_frames=cfg.max_frames, **model_config_dict)
    elif cfg.reward_model == "vlac":
        if not cfg.model_path:
            raise ValueError("model_path is required for VLAC baseline")
        model = VLAC(model_path=cfg.model_path, **model_config_dict)
    elif cfg.reward_model == "robodopamine":
        if not cfg.model_path:
            raise ValueError("model_path is required for Robo-Dopamine baseline")
        model = RoboDopamine(model_path=cfg.model_path, **model_config_dict)
    elif cfg.reward_model == "topreward":
        model_path = cfg.model_path or "Qwen/Qwen3-VL-8B-Instruct"
        model = TopReward(model_path=model_path, **model_config_dict)
    elif cfg.reward_model == "roboreward":
        model = RoboReward(model_path=cfg.model_path or "teetone/RoboReward-4B", **model_config_dict)
    elif cfg.reward_model in ["rewind", "rbm"]:
        if not cfg.model_path:
            raise ValueError("model_path is required for RBM/ReWiND reward model")
        model = RBMModel(checkpoint_path=cfg.model_path)
    else:
        raise ValueError(
            f"Unknown reward_model: {cfg.reward_model}. Must be 'rlvlmf', 'gvl', 'vlac', 'robodopamine', 'topreward', 'roboreward', 'rbm', or 'rewind'"
        )

    all_metrics = {}

    # Process each evaluation type
    for eval_type in cfg.custom_eval.eval_types:
        logger.info(f"=" * 80)
        logger.info(f"Running {eval_type} evaluation with {cfg.reward_model} reward model")
        logger.info(f"=" * 80)

        # Get datasets for this eval type
        eval_datasets = getattr(cfg.custom_eval, eval_type, [])
        if not eval_datasets:
            logger.warning(f"No datasets specified for {eval_type}, skipping")
            continue

        # Create eval_type subdirectory: {model_type}/{eval_type}/...
        eval_type_dir = os.path.join(cfg.output_dir, eval_type)
        os.makedirs(eval_type_dir, exist_ok=True)
        logger.info(f"Saving results to: {eval_type_dir}")

        eval_type_metrics = {}

        logger.info(f"Eval datasets: {eval_datasets}")

        resolved_datasets = []

        for eval_dataset in eval_datasets:
            if isinstance(eval_dataset, list):
                resolved_datasets.append(eval_dataset)
            else:
                resolved_datasets.extend(resolve_dataset_keys([eval_dataset], split="eval"))

            logger.info(f"Resolved datasets for {eval_type}: {resolved_datasets}")

        for dataset_name in resolved_datasets:
            # Create short name for filenames (dataset names can be very long)
            short_dataset_name = _shorten_dataset_name(dataset_name)

            # Resolve dataset keys
            if isinstance(dataset_name, list):
                resolved_dataset_name = dataset_name
            else:
                resolved_dataset_name = resolve_dataset_keys([dataset_name], split="eval")

            logger.info(f"Resolved datasets for {eval_type}: {resolved_datasets}")

            # Create data config for this dataset (similar to trainer)
            eval_data_cfg = copy.deepcopy(base_data_cfg)
            eval_data_cfg.dataset_type = "rbm"
            eval_data_cfg.eval_datasets = resolved_dataset_name

            # Setup dataset
            sampler_kwargs = {
                "random_seed": cfg.custom_eval.custom_eval_random_seed,
            }

            if cfg.reward_model == "roboreward":
                sampler_kwargs["pad_frames"] = False

            if eval_type == "reward_alignment":
                sampler_kwargs["max_trajectories"] = cfg.custom_eval.reward_alignment_max_trajectories
                sampler_kwargs["use_frame_steps"] = cfg.custom_eval.use_frame_steps
                sampler_kwargs["subsample_n_frames"] = cfg.custom_eval.subsample_n_frames
                sampler_kwargs["pad_frames"] = cfg.custom_eval.pad_frames
            elif eval_type == "policy_ranking":
                sampler_kwargs["num_examples_per_quality_pr"] = cfg.custom_eval.num_examples_per_quality_pr
                sampler_kwargs["num_partial_successes"] = cfg.custom_eval.num_partial_successes
                sampler_kwargs["max_tasks"] = cfg.custom_eval.policy_ranking_max_tasks
                sampler_kwargs["use_frame_steps"] = cfg.custom_eval.use_frame_steps
                sampler_kwargs["pad_frames"] = cfg.custom_eval.pad_frames
            elif eval_type == "confusion_matrix":
                sampler_kwargs["pad_frames"] = cfg.custom_eval.pad_frames
            elif "quality_preference" in eval_type:
                sampler_kwargs["comparisons_per_task"] = cfg.custom_eval.comparisons_per_task
                sampler_kwargs["max_comparisons"] = cfg.custom_eval.max_comparisons

            dataset = setup_custom_eval_dataset(
                cfg=eval_data_cfg, sampler_type=eval_type, verbose=True, sampler_kwargs=sampler_kwargs
            )

            # Process samples
            eval_results = []

            if cfg.reward_model in ["rewind", "rbm"]:
                # For RBM/ReWiND, process dataset using indices to avoid materializing entire dataset
                logger.info(f"Processing {len(dataset)} samples in batches for RBM/ReWiND")

                model_config_dict = (
                    asdict(cfg.model_config)
                    if hasattr(cfg.model_config, "__dataclass_fields__")
                    else cfg.model_config.__dict__
                )
                batch_results = process_batched_rbm_samples(dataset, model, batch_size=model_config_dict["batch_size"])
                eval_results.extend(batch_results)
            else:
                # For other models, process samples one at a time
                for i, sample in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
                    if cfg.reward_model == "rlvlmf" and isinstance(sample, PreferenceSample):
                        result = process_preference_sample(sample, model)
                        if result:
                            eval_results.append(result)
                    elif cfg.reward_model in ["gvl", "vlac", "roboreward", "robodopamine", "topreward"] and isinstance(
                        sample, ProgressSample
                    ):
                        # Handle ProgressSamples for gvl/vlac/roboreward (including confusion_matrix)
                        result = process_progress_sample(sample, model)
                        if result:
                            eval_results.append(result)
                    else:
                        logger.warning(f"Sample type mismatch: reward_model={cfg.reward_model}, sample={type(sample)}")

            logger.info(f"Processed {len(eval_results)} samples from {dataset_name}")

            # Save results to JSON
            if eval_type_dir:
                results_file = os.path.join(eval_type_dir, f"{short_dataset_name}_results.json")
                with open(results_file, "w") as f:
                    json.dump(_make_json_serializable(eval_results), f, indent=2)
                logger.info(f"Saved results to {results_file}")

            # Compute metrics using the same functions as the trainer
            if eval_results:
                # Determine data_source from first result
                data_source = eval_results[0].get("data_source") if eval_results else None

                if eval_type == "quality_preference":
                    # Quality preference evaluation for rlvlmf, rbm, rewind
                    if cfg.reward_model not in ["rlvlmf", "rbm", "rewind"]:
                        raise ValueError(
                            f"quality_preference evaluation only supported for rlvlmf, rbm, rewind, got {cfg.reward_model}"
                        )

                    metrics_dict, task_groups, task_details = run_quality_preference_eval(
                        results=eval_results,
                        data_source=data_source,
                    )
                    # Save task_groups and task_details if available
                    if eval_type_dir:
                        task_groups_file = os.path.join(eval_type_dir, f"{short_dataset_name}_task_groups.json")
                        task_details_file = os.path.join(eval_type_dir, f"{short_dataset_name}_task_details.json")
                        with open(task_groups_file, "w") as f:
                            json.dump(_make_json_serializable(task_groups), f, indent=2)
                        with open(task_details_file, "w") as f:
                            json.dump(_make_json_serializable(task_details), f, indent=2)
                        logger.info(f"Saved task_groups to {task_groups_file}")
                        logger.info(f"Saved task_details to {task_details_file}")

                    # Extract metrics from the returned dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            eval_type_metrics[f"{short_dataset_name}/{key}"] = float(value)

                    # Write metrics incrementally after each dataset
                    if eval_type_dir:
                        _write_metrics_incremental(eval_type_dir, eval_type_metrics)

                elif eval_type == "confusion_matrix":
                    # Confusion matrix evaluation for gvl, vlac, roboreward, rbm, rewind
                    if cfg.reward_model not in ["gvl", "vlac", "roboreward", "robodopamine", "topreward", "rbm", "rewind"]:
                        raise ValueError(
                            f"confusion_matrix evaluation only supported for gvl, vlac, roboreward, robodopamine, topreward, rbm, rewind, got {cfg.reward_model}"
                        )

                    # run_confusion_matrix_eval returns (fig, confusion_matrix, metrics)
                    fig, confusion_matrix, metrics_dict = run_confusion_matrix_eval(
                        results=eval_results,
                        progress_pred_type="absolute_wrt_total_frames",  # Baselines use absolute progress
                        is_discrete_mode=False,  # Baselines output continuous values
                        num_bins=None,
                    )

                    # Save confusion matrix plot
                    if fig and eval_type_dir:
                        plot_path = os.path.join(eval_type_dir, f"{short_dataset_name}_confusion_matrix.png")
                        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        logger.info(f"Saved confusion matrix plot to {plot_path}")

                        # Save confusion matrix as numpy array
                        matrix_path = os.path.join(eval_type_dir, f"{short_dataset_name}_confusion_matrix.npy")
                        np.save(matrix_path, confusion_matrix)
                        logger.info(f"Saved confusion matrix array to {matrix_path}")

                    # Extract metrics from the returned dict (if any)
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            eval_type_metrics[f"{short_dataset_name}/{key}"] = float(value)

                    # Write metrics incrementally after each dataset
                    if eval_type_dir:
                        _write_metrics_incremental(eval_type_dir, eval_type_metrics)

                else:
                    # Progress evaluation (reward_alignment, policy_ranking) for gvl, vlac, roboreward, rbm, rewind
                    if cfg.reward_model not in ["gvl", "vlac", "roboreward", "robodopamine", "topreward", "rbm", "rewind"]:
                        raise ValueError(
                            f"Progress evaluation only supported for gvl, vlac, roboreward, robodopamine, topreward, rbm, rewind, got {cfg.reward_model}"
                        )

                    if eval_type == "reward_alignment":
                        metrics_dict, plots, video_frames_list, _ = run_reward_alignment_eval_per_trajectory(
                            results=eval_results,
                            progress_pred_type="absolute_wrt_total_frames",  # Baselines use absolute progress
                            is_discrete_mode=False,  # Baselines output continuous values
                            num_bins=None,
                            data_source=data_source,
                            use_frame_steps=cfg.custom_eval.use_frame_steps,
                            train_success_head=False,  # Baselines don't have success head
                            last_frame_only=False,
                        )
                        # Save plots with videos as GIFs if available
                        if plots and eval_type_dir:
                            plots_dir = os.path.join(eval_type_dir, f"{short_dataset_name}_plots")
                            os.makedirs(plots_dir, exist_ok=True)
                            for i, fig in enumerate(plots[:10]):
                                video_frames = video_frames_list[i] if i < len(video_frames_list) else None
                                gif_path = os.path.join(plots_dir, f"trajectory_{i:04d}.gif")
                                _create_plot_with_video_gif(fig, video_frames, gif_path)
                            logger.info(f"Saved {len(plots)} plot+video GIFs to {plots_dir}")
                    elif eval_type == "policy_ranking":
                        metrics_dict, task_groups, task_details = run_policy_ranking_eval(
                            results=eval_results,
                            progress_pred_type="absolute_wrt_total_frames",  # Baselines use absolute progress
                            is_discrete_mode=False,  # Baselines output continuous values
                            num_bins=None,
                            data_source=data_source,
                            correlation_method="kendall",
                        )
                        # Save task_groups and task_details if available
                        if eval_type_dir:
                            task_groups_file = os.path.join(eval_type_dir, f"{short_dataset_name}_task_groups.json")
                            task_details_file = os.path.join(eval_type_dir, f"{short_dataset_name}_task_details.json")
                            with open(task_groups_file, "w") as f:
                                json.dump(_make_json_serializable(task_groups), f, indent=2)
                            with open(task_details_file, "w") as f:
                                json.dump(_make_json_serializable(task_details), f, indent=2)
                            logger.info(f"Saved task_groups to {task_groups_file}")
                            logger.info(f"Saved task_details to {task_details_file}")
                    else:
                        raise ValueError(f"Unknown eval_type: {eval_type}")

                    # Extract metrics from the returned dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            eval_type_metrics[f"{short_dataset_name}/{key}"] = float(value)

                    # Write metrics incrementally after each dataset
                    if eval_type_dir:
                        _write_metrics_incremental(eval_type_dir, eval_type_metrics)

        all_metrics[eval_type] = eval_type_metrics

    return all_metrics


def _write_metrics_incremental(eval_type_dir: str, eval_type_metrics: Dict[str, float]):
    """Write metrics to file incrementally after each dataset.

    Args:
        eval_type_dir: Directory for this eval_type (e.g., ./baseline_eval_output/robometer/policy_ranking/)
        eval_type_metrics: Dictionary of metrics (keyed by dataset_name/metric_name)
    """
    metrics_file = os.path.join(eval_type_dir, "metrics.json")

    # Write metrics to file (overwrite since we're accumulating in eval_type_metrics)
    try:
        with open(metrics_file, "w") as f:
            json.dump(_make_json_serializable(eval_type_metrics), f, indent=2)
        logger.info(f"Updated metrics file: {metrics_file}")
    except IOError as e:
        logger.error(f"Could not write metrics file: {e}")


def _normalize_model_path(model_path: Optional[str]) -> str:
    """Normalize model path for use in directory names.

    Handles HuggingFace paths like 'rewardfm/rbm-base' or local paths.
    Replaces slashes, special characters with underscores.

    Args:
        model_path: Model path string (e.g., 'rewardfm/rbm-base', '/path/to/model')

    Returns:
        Normalized string safe for directory names
    """
    if not model_path:
        return ""

    # Get the basename if it's an absolute path
    if model_path.startswith("/"):
        # For absolute paths, use the last two components (parent/name)
        parts = model_path.rstrip("/").split("/")
        model_path = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    # Replace slashes with underscores
    normalized = model_path.replace("/", "_")

    # Replace other special characters that might cause issues
    for char in [":", "\\", " ", ".", ","]:
        normalized = normalized.replace(char, "_")

    # Remove leading/trailing underscores and collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")

    return normalized


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    """Main entry point for baseline evaluation."""
    # Convert Hydra config to dataclass
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)

    # Display config
    display_config(baseline_cfg)

    # Validate reward model
    if baseline_cfg.reward_model not in ["gvl", "vlac", "rlvlmf", "roboreward", "robodopamine", "topreward", "rbm", "rewind"]:
        raise ValueError(
            f"reward_model must be 'gvl', 'vlac', 'rlvlmf', 'roboreward', 'robodopamine', 'topreward', 'rbm', or 'rewind', got {baseline_cfg.reward_model}"
        )

    # Setup output directory: {model_type}_{model_path}/{eval_type}/...
    if baseline_cfg.output_dir is None:
        normalized_path = _normalize_model_path(baseline_cfg.model_path)

        if normalized_path:
            dir_name = f"{baseline_cfg.reward_model}_{normalized_path}"
        else:
            dir_name = baseline_cfg.reward_model

        baseline_cfg.output_dir = os.path.join("./baseline_eval_output", dir_name)

    os.makedirs(baseline_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {baseline_cfg.output_dir}")

    # Create data config with default settings
    # Datasets will be set per eval type during processing
    data_cfg = DataConfig(
        max_frames=baseline_cfg.max_frames,
        load_embeddings=True if "rewind" in baseline_cfg.reward_model else False,
    )

    display_config(data_cfg)

    # Run evaluation
    metrics = run_baseline_evaluation(baseline_cfg, data_cfg)

    # Save all metrics summary to model root directory
    if metrics and is_rank_0():
        metrics_file = os.path.join(baseline_cfg.output_dir, "all_metrics.json")
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                metrics_serializable[k] = {
                    k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2 for k2, v2 in v.items()
                }
            else:
                metrics_serializable[k] = v

        with open(metrics_file, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Saved all metrics summary to: {metrics_file}")

    logger.info("\nBaseline evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
