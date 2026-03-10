#!/usr/bin/env python3
"""
FastAPI server to evaluate RBM model batches with a multi-GPU service layer.

Usage example:
    uv run python robometer/evals/eval_server.py \
        model_path=robometer/Robometer-4B \
        batch_size=16 \
        num_gpus=1 \
        server_port=8001

Endpoints:
  POST /evaluate_batch        - JSON payload
  POST /evaluate_batch_npy    - multipart payload with .npy blobs

Response payload per request contains predictions grouped by head:
  {
    "outputs_preference": {...},   # Preference logits + optional progress traces
    "outputs_progress": {...},     # Progress-only trajectories
  }
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import uvicorn

import numpy as np
import torch
from omegaconf import DictConfig
from hydra import main as hydra_main
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from robometer.evals.eval_utils import (
    parse_npy_form_data,
    reconstruct_payload_from_npy,
)
from robometer.utils.save import load_model_from_hf
from robometer.configs.eval_configs import EvalServerConfig
from robometer.configs.experiment_configs import ExperimentConfig
from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.utils.setup_utils import setup_model_and_processor, setup_batch_collator
from robometer.models.utils import ModelOutput, convert_bins_to_continuous, convert_bins_to_continuous_hard
from robometer.utils.config_utils import display_config, convert_hydra_to_dataclass
from robometer.utils.logger import get_logger, setup_loguru_logging

LOG_LEVEL = "DEBUG"
setup_loguru_logging(log_level=LOG_LEVEL)
logger = get_logger()
logger.info(f"robometer.eval_server logger initialized at level {LOG_LEVEL}")


def log_logits(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        logger.debug(f"{name} shape={tuple(value.shape)} values={value.detach().cpu().tolist()}")
    elif isinstance(value, dict):
        logger.debug(f"{name} keys={list(value.keys())}")
        for key, sub_value in value.items():
            log_logits(f"{name}.{key}", sub_value)
    elif isinstance(value, list):
        logger.debug(f"{name}: {value}")


def aggregate_frame_step_predictions(
    outputs: Dict[str, Any],
    sample_frame_counts: List[int],
    outputs_success: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregate frame-step predictions back into full sequences.

    Args:
        outputs: Dict containing progress predictions from sub-samples
        sample_frame_counts: List indicating how many frames each original sample had
        outputs_success: Optional dict containing success predictions

    Returns:
        Aggregated outputs with full sequences per original sample
    """
    progress_pred = outputs.get("progress_pred", [])

    # Aggregate progress predictions
    aggregated_progress = []
    current_idx = 0

    for num_frames in sample_frame_counts:
        if num_frames == 1:
            # Non-progress sample or single-frame sample, pass through
            if current_idx < len(progress_pred):
                aggregated_progress.append(progress_pred[current_idx])
                current_idx += 1
            else:
                aggregated_progress.append([])
        else:
            # Collect predictions from sub-samples and extract last prediction from each
            sample_predictions = []
            for i in range(num_frames):
                if current_idx < len(progress_pred):
                    sub_pred = progress_pred[current_idx]
                    # Extract the last (and only meaningful) prediction from this sub-sample
                    if isinstance(sub_pred, list) and len(sub_pred) > 0:
                        sample_predictions.append(sub_pred[-1])
                    current_idx += 1
            aggregated_progress.append(sample_predictions)

    aggregated_outputs = {"progress_pred": aggregated_progress}

    # Aggregate success predictions if present
    if outputs_success is not None:
        success_probs = outputs_success.get("success_probs", [])
        aggregated_success = []
        current_idx = 0

        for num_frames in sample_frame_counts:
            if num_frames == 1:
                if current_idx < len(success_probs):
                    aggregated_success.append(success_probs[current_idx])
                    current_idx += 1
                else:
                    aggregated_success.append([])
            else:
                # Collect success predictions from sub-samples
                sample_success = []
                for i in range(num_frames):
                    if current_idx < len(success_probs):
                        sub_success = success_probs[current_idx]
                        # Extract the last prediction from this sub-sample
                        if isinstance(sub_success, list) and len(sub_success) > 0:
                            sample_success.append(sub_success[-1])
                        current_idx += 1
                aggregated_success.append(sample_success)

        aggregated_outputs["outputs_success"] = {"success_probs": aggregated_success}

    return aggregated_outputs


def forward_model(
    model: Any, batch_inputs: Dict[str, Any], sample_type: str = "progress"
) -> Tuple[ModelOutput, Dict[str, Any]]:
    """Forward pass that mirrors trainer logic (handles ReWiND vs RBM)."""
    with torch.no_grad():
        if "rewind" in model.__class__.__name__.lower():
            model_output, extra = model(
                video_embeddings=batch_inputs.get("video_embeddings"),
                text_embeddings=batch_inputs.get("text_embeddings"),
                sample_type=sample_type,
                timing_raw=None,
            )
        else:
            model_output, extra = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs.get("pixel_values", None),
                pixel_values_videos=batch_inputs.get("pixel_values_videos", None),
                image_grid_thw=batch_inputs.get("image_grid_thw", None),
                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
                sample_type=sample_type,
                timing_raw=None,
            )

    return model_output, extra


def process_batch_helper(
    model_type: str,
    model: Any,
    tokenizer: Any,
    batch_collator: Any,
    device: torch.device,
    batch_data: List[Dict[str, Any]],
    job_id: int = 0,
    is_discrete_mode: bool = False,
    num_bins: int = 10,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    """Synchronous batch processing on specific GPU."""
    if not batch_data:
        raise ValueError("No samples found in batch data")

    logger.debug(f"[job {job_id}] Processing {len(batch_data)} samples on device {device}")

    input_samples: List[Any] = []
    for sample in batch_data:
        if isinstance(sample, (PreferenceSample, ProgressSample)):
            input_samples.append(sample)
        elif isinstance(sample, dict):
            sample_type = sample.get("sample_type")
            if sample_type == "preference":
                input_samples.append(PreferenceSample(**sample))
            elif sample_type == "progress":
                input_samples.append(ProgressSample(**sample))
            else:
                raise ValueError(f"Unsupported sample_type: {sample_type}")
        else:
            raise ValueError(f"Unsupported sample object type: {type(sample)}")

    # Handle frame steps for progress samples - expand into sub-samples, each subsampled to 4 frames
    # so they can be batched together (all same size)
    NUM_SUBSAMPLED_FRAMES = 4

    if use_frame_steps:
        expanded_samples = []
        sample_frame_counts = []  # Track how many sub-samples each original sample generates

        for sample in input_samples:
            if isinstance(sample, ProgressSample):
                # Get the frames from the trajectory
                frames = sample.trajectory.frames
                num_frames = frames.shape[0] if hasattr(frames, "shape") else len(frames)

                # Create sub-samples with increasing frame counts: 0:1, 0:2, 0:3, ..., 0:T
                # Each sub-sample is subsampled to NUM_SUBSAMPLED_FRAMES frames using linspace
                for i in range(1, num_frames + 1):
                    # Use linspace to select frame indices from 0 to i-1
                    # This ensures all sub-samples have the same number of frames for batching
                    indices = np.linspace(0, i - 1, NUM_SUBSAMPLED_FRAMES, dtype=int)
                    sub_frames = frames[indices]

                    sub_trajectory = copy.deepcopy(sample.trajectory)
                    sub_trajectory.frames = sub_frames
                    sub_trajectory.frames_shape = (
                        sub_frames.shape if hasattr(sub_frames, "shape") else (len(sub_frames),)
                    )

                    # Adjust target_progress and success_label if they exist (also subsample)
                    if hasattr(sub_trajectory, "target_progress") and sub_trajectory.target_progress is not None:
                        orig_progress = sub_trajectory.target_progress[:i]
                        if len(orig_progress) > 0:
                            sub_trajectory.target_progress = (
                                np.array(orig_progress)[indices].tolist()
                                if hasattr(orig_progress, "__len__")
                                else orig_progress
                            )
                    if hasattr(sub_trajectory, "success_label") and sub_trajectory.success_label is not None:
                        orig_success = sub_trajectory.success_label[:i]
                        if len(orig_success) > 0:
                            sub_trajectory.success_label = (
                                np.array(orig_success)[indices].tolist()
                                if hasattr(orig_success, "__len__")
                                else orig_success
                            )

                    # Create sub-sample
                    sub_sample = ProgressSample(
                        trajectory=sub_trajectory,
                        data_gen_strategy=sample.data_gen_strategy,
                    )
                    expanded_samples.append(sub_sample)

                sample_frame_counts.append(num_frames)
            else:
                # Non-progress samples are passed through unchanged
                expanded_samples.append(sample)
                sample_frame_counts.append(1)

        input_samples = expanded_samples
        logger.debug(
            f"[job {job_id}] Expanded {len(sample_frame_counts)} samples into {len(input_samples)} sub-samples with frame steps (each subsampled to {NUM_SUBSAMPLED_FRAMES} frames)"
        )
    else:
        sample_frame_counts = None

    batch_inputs = batch_collator(input_samples)

    # Move inputs to the correct GPU
    for key, value in batch_inputs["preference_inputs"].items():
        if isinstance(value, torch.Tensor):
            batch_inputs["preference_inputs"][key] = value.to(device)
    for key, value in batch_inputs["progress_inputs"].items():
        if isinstance(value, torch.Tensor):
            batch_inputs["progress_inputs"][key] = value.to(device)
    outputs_preference = None
    outputs_progress = None
    outputs_success = None

    num_preferences = batch_inputs.get("num_preferences", 0)
    num_progress = batch_inputs.get("num_progress", 0)
    logger.debug(f"[job {job_id}] Batch counts — preference: {num_preferences} progress: {num_progress}")

    if num_preferences > 0:
        outputs_preference = compute_batch_outputs(
            model,
            tokenizer,
            batch_inputs["preference_inputs"],
            sample_type="preference",
            is_discrete_mode=is_discrete_mode,
            num_bins=num_bins,
        )

    if num_progress > 0:
        outputs_progress = compute_batch_outputs(
            model,
            tokenizer,
            batch_inputs["progress_inputs"],
            sample_type="progress",
            is_discrete_mode=is_discrete_mode,
            num_bins=num_bins,
        )

        if "outputs_success" in outputs_progress:
            outputs_success = outputs_progress.pop("outputs_success")

        # Aggregate frame-step predictions back into full sequences
        if use_frame_steps and sample_frame_counts is not None:
            outputs_progress = aggregate_frame_step_predictions(outputs_progress, sample_frame_counts, outputs_success)
            if outputs_success is not None:
                outputs_success = outputs_progress.pop("outputs_success", None)

    return {
        "outputs_preference": outputs_preference,
        "outputs_progress": outputs_progress,
        "outputs_success": outputs_success,
    }


class MultiGPUEvalServer:
    """Multi-GPU inference server that schedules requests across devices."""

    def __init__(
        self,
        model_path: str,
        num_gpus: int | None = None,
        max_workers: int | None = None,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.max_workers = max_workers or self.num_gpus
        self._active_jobs = 0
        self._job_counter = 0
        self._completed_jobs = 0
        self._active_jobs_lock = Lock()

        logger.info(f"Loading experiment config and base model from {self.model_path}")
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=self.model_path,
            device=torch.device("cpu"),
        )

        self.exp_config: ExperimentConfig = exp_config
        self.base_tokenizer = tokenizer
        self.base_processor = processor
        self.base_model = reward_model
        self.base_batch_collator = setup_batch_collator(processor, tokenizer, self.exp_config, is_eval=True)

        if self.num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        logger.info(
            f"Initializing multi-GPU eval server: model_path={self.model_path} "
            f"num_gpus={self.num_gpus} max_workers={self.max_workers}"
        )

        # Initialize GPU pool
        self.gpu_pool = queue.Queue(maxsize=self.num_gpus)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gpu_stats = {}

        # Initialize GPUs
        self._initialize_gpus()

        logger.info("Multi-GPU eval server initialized successfully")

    def _initialize_gpus(self):
        """Initialize models on all GPUs."""
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            logger.info(f"Loading model replica on GPU {gpu_id} ({device})")

            # Load model on specific GPU
            tokenizer = copy.deepcopy(self.base_tokenizer)
            processor = copy.deepcopy(self.base_processor)
            model = copy.deepcopy(self.base_model)
            batch_collator = copy.deepcopy(self.base_batch_collator)

            model = model.to(device)
            model.eval()

            # Initialize GPU stats
            self.gpu_stats[gpu_id] = {
                "total_requests": 0,
                "total_processing_time": 0.0,
                "last_used": time.time(),
                "status": "ready",
            }

            # Add to pool
            self.gpu_pool.put({
                "model": model,
                "processor": processor,
                "tokenizer": tokenizer,
                "batch_collator": batch_collator,
                "device": device,
                "gpu_id": gpu_id,
                "created_at": time.time(),
            })

            logger.info(f"Successfully loaded model on GPU {gpu_id}")

    async def process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch using whichever GPU is available."""
        loop = asyncio.get_event_loop()

        # Get GPU from pool (this will block until one is available).
        # Use the default executor so worker threads remain available for compute.
        gpu_info = await loop.run_in_executor(None, self.gpu_pool.get)
        queue_size_after_acquire = self.gpu_pool.qsize()
        with self._active_jobs_lock:
            self._job_counter += 1
            job_id = self._job_counter
            self._active_jobs += 1
            active_jobs = self._active_jobs
        logger.debug(
            f"[job {job_id}] Acquired GPU {gpu_info['gpu_id']} "
            f"queue_size={queue_size_after_acquire} active_jobs={active_jobs}"
        )

        start_time = time.time()

        # Update GPU stats
        self.gpu_stats[gpu_info["gpu_id"]]["status"] = "processing"
        self.gpu_stats[gpu_info["gpu_id"]]["last_used"] = start_time

        try:
            # Determine if discrete mode is enabled
            progress_loss_type = getattr(self.exp_config.loss, "progress_loss_type", "l2")
            is_discrete_mode = progress_loss_type.lower() == "discrete"
            num_bins = getattr(
                self.exp_config.loss,
                "progress_discrete_bins",
                getattr(self.exp_config.model, "progress_discrete_bins", 10),
            )

            # Extract use_frame_steps flag from batch_data (if present)
            use_frame_steps = False
            actual_batch_data = batch_data
            if isinstance(batch_data, dict) and "samples" in batch_data:
                use_frame_steps = batch_data.get("use_frame_steps", False)
                actual_batch_data = batch_data["samples"]
            elif isinstance(batch_data, dict) and "use_frame_steps" in batch_data:
                # Legacy format: extract use_frame_steps and assume rest is data
                use_frame_steps = batch_data.get("use_frame_steps", False)
                actual_batch_data = [v for k, v in batch_data.items() if k != "use_frame_steps"]
                if len(actual_batch_data) == 1 and isinstance(actual_batch_data[0], list):
                    actual_batch_data = actual_batch_data[0]

            # Process batch in thread pool
            result = await loop.run_in_executor(
                self.executor,
                process_batch_helper,
                self.exp_config.model.model_type,
                gpu_info["model"],
                gpu_info["tokenizer"],
                gpu_info["batch_collator"],
                gpu_info["device"],
                actual_batch_data,
                job_id,
                is_discrete_mode,
                num_bins,
                use_frame_steps,
            )

            # Update stats
            processing_time = time.time() - start_time
            self.gpu_stats[gpu_info["gpu_id"]]["total_requests"] += 1
            self.gpu_stats[gpu_info["gpu_id"]]["total_processing_time"] += processing_time

            return result

        finally:
            # Always return GPU to pool and update stats
            processing_time = time.time() - start_time
            self.gpu_stats[gpu_info["gpu_id"]]["total_requests"] += 1
            self.gpu_stats[gpu_info["gpu_id"]]["total_processing_time"] += processing_time
            self.gpu_stats[gpu_info["gpu_id"]]["status"] = "ready"
            self.gpu_pool.put(gpu_info)
            queue_size_after_release = self.gpu_pool.qsize()
            with self._active_jobs_lock:
                self._active_jobs -= 1
                self._completed_jobs += 1
                active_jobs = self._active_jobs
                completed_jobs = self._completed_jobs
            logger.debug(
                f"[job {job_id}] Completed on GPU {gpu_info['gpu_id']} "
                f"active_jobs={active_jobs} completed_jobs={completed_jobs} "
                f"queue_size={queue_size_after_release} "
                f"processing_time={processing_time:.3f}s"
            )

    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of the GPU pool."""
        return {
            "total_gpus": self.num_gpus,
            "available_gpus": self.gpu_pool.qsize(),
            "max_workers": self.max_workers,
            "gpu_stats": self.gpu_stats,
            "pool_size": self.gpu_pool.maxsize,
        }

    def shutdown(self):
        """Shutdown the GPU pool and executor."""
        logger.info("Shutting down GPU pool...")
        self.executor.shutdown(wait=True)
        logger.info("GPU pool shutdown complete")


def compute_batch_outputs(
    model: Any,
    tokenizer: Any,
    batch_inputs: Dict[str, torch.Tensor],
    sample_type: str,
    is_discrete_mode: bool = False,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Run a forward pass and return the raw head outputs we need for eval logging.

    Args:
        model: RBM/ReWiND model on the target device.
        tokenizer: Tokenizer (unused for head-based inference).
        batch_inputs: Collated inputs for the requested head.
        sample_type: One of {"preference","progress"}.

    Returns:
        Dict containing logits/derived predictions keyed by head.
    """
    model.eval()
    logger.debug(f"compute_batch_outputs sample_type={sample_type}")
    model_output, _ = forward_model(model, batch_inputs, sample_type=sample_type)

    results: Dict[str, Any] = {}

    # Preference logits and metadata
    if sample_type == "preference" and model_output.pref_logits is not None:
        logits = model_output.pref_logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        results.update({
            "predictions": preds.detach().cpu().tolist(),
            "prediction_probs": probs.detach().cpu().tolist(),
            "preference_labels": batch_inputs["preference_labels"].cpu().tolist(),
        })

        # logger.debug(f"predictions: {results['predictions']}")
        # logger.debug(f"prediction_probs: {results['prediction_probs']}")
        # logger.debug(f"preference_labels: {results['preference_labels']}")

    # Progress predictions (only for progress sample type)
    progress_logits = model_output.progress_logits
    if progress_logits is not None and isinstance(progress_logits, dict) and sample_type == "progress":
        progress_pred = []
        seq_A = progress_logits.get("A")

        # Convert tensor to list
        seq_A_list = [seq_A[i] for i in range(seq_A.shape[0])] if seq_A is not None else []

        # Process seq_A
        for seq_A_item in seq_A_list:
            if seq_A_item is None:
                progress_pred.append([])
            elif is_discrete_mode:
                # seq_A_item is [seq_len, num_bins] logits, convert entire sequence to continuous
                continuous_pred = convert_bins_to_continuous(seq_A_item.detach().cpu().float())
                # continuous_pred = convert_bins_to_continuous_hard(seq_A_item.detach().cpu().float())
                progress_pred.append(continuous_pred.numpy().flatten().tolist())
            else:
                progress_pred.append(seq_A_item.detach().cpu().flatten().tolist())

        if not progress_pred:
            batch_size = len(batch_inputs.get("task", []))
            progress_pred = [[] for _ in range(batch_size)]

        results["progress_pred"] = progress_pred
        # logger.debug(f"progress_pred: {progress_pred}")
        if model_output.success_logits is not None:
            success_pred = model_output.success_logits["A"]
            success_probs = torch.sigmoid(success_pred)
            results["outputs_success"] = {
                "success_probs": success_probs.detach().cpu().tolist(),
            }
            # logger.debug(f"success_probs: {success_probs}")

    return results


def create_app(cfg: EvalServerConfig, multi_gpu_server: MultiGPUEvalServer | None = None):
    app = FastAPI(title="RBM Multi-GPU Evaluation Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize multi-GPU server
    num_gpus = getattr(cfg, "num_gpus", None)
    max_workers = getattr(cfg, "max_workers", None)

    multi_gpu_server = multi_gpu_server or MultiGPUEvalServer(cfg.model_path, num_gpus, max_workers)
    logger.info(f"Multi-GPU eval server initialized with {multi_gpu_server.num_gpus} GPUs")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a batch of preference samples using the Multi-GPU server."""
        logger.debug(f"Received /evaluate_batch request with keys: {list(batch.keys())}")
        return await multi_gpu_server.process_batch(batch)

    @app.post("/evaluate_batch_npy")
    async def evaluate_batch_npy(request: Request) -> Dict[str, Any]:
        """Evaluate a batch with .npy file support for numpy arrays.

        This endpoint handles multipart form data where:
        - numpy arrays are sent as .npy files
        - other data is sent as form fields
        """
        # Parse form data
        form_data = await request.form()

        # Extract numpy arrays and other data using shared utility (await async function)
        numpy_arrays, other_data = await parse_npy_form_data(form_data)

        # Extract use_frame_steps flag from other_data (handle both bool and string)
        use_frame_steps_value = other_data.pop("use_frame_steps", False)
        if isinstance(use_frame_steps_value, bool):
            use_frame_steps = use_frame_steps_value
        else:
            use_frame_steps = str(use_frame_steps_value).lower() == "true"

        # Reconstruct the original payload structure (RBM needs torch tensor conversion for embeddings)
        batch_data = reconstruct_payload_from_npy(
            numpy_arrays,
            other_data,
            trajectory_keys=[
                "chosen_trajectory",
                "rejected_trajectory",
                "reference_trajectory",
                "traj_sim_trajectory",
                "traj_diff_trajectory",
                "trajectory",
            ],
            convert_embeddings_to_torch=True,
        )

        # Add use_frame_steps flag to batch_data
        batch_payload = {
            "samples": batch_data,
            "use_frame_steps": use_frame_steps,
        }

        # Process the batch
        logger.debug(
            f"Received /evaluate_batch_npy request with {len(numpy_arrays)} numpy arrays "
            f"and {len(other_data)} other fields, use_frame_steps={use_frame_steps}"
        )
        return await multi_gpu_server.process_batch(batch_payload)

    @app.get("/gpu_status")
    def get_gpu_status() -> Dict[str, Any]:
        """Get status of all GPUs and pool."""
        return multi_gpu_server.get_pool_status()

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        status = multi_gpu_server.get_pool_status()
        return {"status": "healthy", "available_gpus": status["available_gpus"], "total_gpus": status["total_gpus"]}

    @app.get("/model_info")
    def get_model_info() -> Dict[str, Any]:
        """Get model information and experiment configuration."""

        def serialize_config(config_obj: Any) -> Any:
            """Recursively serialize dataclass to dict, handling nested dataclasses."""
            if hasattr(config_obj, "__dataclass_fields__"):
                result = {}
                for field_name, field_value in asdict(config_obj).items():
                    result[field_name] = serialize_config(field_value)
                return result
            elif isinstance(config_obj, dict):
                return {k: serialize_config(v) for k, v in config_obj.items()}
            elif isinstance(config_obj, list):
                return [serialize_config(item) for item in config_obj]
            elif isinstance(config_obj, (str, int, float, bool, type(None))):
                return config_obj
            else:
                # Fallback: convert to string for non-serializable types
                return str(config_obj)

        def get_model_architecture_info(model: Any) -> Dict[str, Any]:
            """Extract model architecture information."""
            model_info = {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            }

            # Count parameters
            try:
                all_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_params = all_params - trainable_params

                model_info.update({
                    "total_parameters": all_params,
                    "trainable_parameters": trainable_params,
                    "frozen_parameters": frozen_params,
                    "trainable_percentage": (100 * trainable_params / all_params) if all_params > 0 else 0.0,
                })
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
                model_info["parameter_count_error"] = str(e)

            # Get model architecture summary (first few layers)
            try:
                architecture_summary = []
                for name, module in list(model.named_children())[:10]:  # First 10 top-level modules
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters())
                    architecture_summary.append({
                        "name": name,
                        "type": module_type,
                        "parameters": num_params,
                    })
                model_info["architecture_summary"] = architecture_summary
            except Exception as e:
                logger.warning(f"Could not get architecture summary: {e}")
                model_info["architecture_summary_error"] = str(e)

            return model_info

        exp_config_dict = serialize_config(multi_gpu_server.exp_config)

        # Get model architecture info from the base model
        model_arch_info = None
        try:
            model_arch_info = get_model_architecture_info(multi_gpu_server.base_model)
        except Exception as e:
            logger.warning(f"Could not get model architecture info: {e}")
            model_arch_info = {"error": str(e)}

        return {
            "model_path": multi_gpu_server.model_path,
            "num_gpus": multi_gpu_server.num_gpus,
            "experiment_config": exp_config_dict,
            "model_architecture": model_arch_info,
        }

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        multi_gpu_server.shutdown()

    return app


@hydra_main(version_base=None, config_path="../configs", config_name="eval_config_server")
def main(cfg: DictConfig):
    """Main entry point for evaluation server using Hydra configuration."""
    # Convert Hydra config to dataclass
    eval_cfg = convert_hydra_to_dataclass(cfg, EvalServerConfig)

    # Display the configuration in a nice Rich format
    display_config(eval_cfg)

    # Ensure pretrained checkpoint is specified
    if not eval_cfg.model_path:
        raise ValueError("Eval config must set model_path to a pretrained checkpoint.")

    multi_gpu_server = MultiGPUEvalServer(
        model_path=eval_cfg.model_path,
        num_gpus=eval_cfg.num_gpus,
        max_workers=eval_cfg.max_workers,
    )
    display_config(multi_gpu_server.exp_config)

    app = create_app(eval_cfg, multi_gpu_server)
    print(f"Running multi-GPU eval server on {eval_cfg.server_url}:{eval_cfg.server_port}")
    print(f"Using {eval_cfg.num_gpus or torch.cuda.device_count()} GPUs")
    uvicorn.run(app, host=eval_cfg.server_url, port=eval_cfg.server_port)


if __name__ == "__main__":
    main()
