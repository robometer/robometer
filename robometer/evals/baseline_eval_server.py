#!/usr/bin/env python3
"""
FastAPI server to evaluate baseline models (RL-VLM-F, GVL, VLAC, RoboReward) with batch processing.

Usage examples:
    # RL-VLM-F baseline server
    uv run python robometer/evals/baseline_eval_server.py \
        reward_model=rlvlmf \
        reward_model.model_config.vlm_provider=gemini \
        server_port=8001
    
    # VLAC baseline server
    uv run --extra vlac --python .venv-vlac/bin/python robometer/evals/baseline_eval_server.py \
        reward_model=vlac \
        model_path=InternRobotics/VLAC \
        server_port=8021

    # RoboReward baseline server
    uv run python robometer/evals/baseline_eval_server.py \
        reward_model=roboreward \
        model_path=teetone/RoboReward-8B \
        server_port=8022

Endpoints:
  POST /evaluate_batch        - JSON payload with samples
  POST /evaluate_batch_npy    - multipart payload with .npy blobs for numpy arrays
  GET /health                 - Health check
  GET /model_info             - Model information

Response payload per request contains predictions grouped by sample type:
  {
    "outputs_preference": [...],   # Preference predictions
    "outputs_progress": [...],     # Progress predictions
  }
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
from typing import Any, Dict, List, Optional

import uvicorn
import numpy as np
from omegaconf import DictConfig
from hydra import main as hydra_main
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from robometer.configs.eval_configs import BaselineEvalConfig
from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.evals.baselines.rlvlmf import RLVLMF
from robometer.evals.baselines.gvl import GVL
from robometer.evals.baselines.vlac import VLAC

try:
    from robometer.evals.baselines.roboreward import RoboReward
except ImportError:
    RoboReward = None

from robometer.evals.run_baseline_eval import process_preference_sample, process_progress_sample
from robometer.evals.eval_utils import parse_npy_form_data, reconstruct_payload_from_npy
from robometer.utils.config_utils import display_config, convert_hydra_to_dataclass
from robometer.utils.logger import get_logger, setup_loguru_logging

LOG_LEVEL = "TRACE"
setup_loguru_logging(log_level=LOG_LEVEL)
logger = get_logger()
logger.info(f"robometer.baseline_eval_server logger initialized at level {LOG_LEVEL}")


def aggregate_frame_step_predictions_baseline(
    progress_preds: List[List[float]],
    sample_frame_counts: List[int],
) -> List[List[float]]:
    """
    Aggregate frame-step predictions back into full sequences for baseline models.

    Args:
        progress_preds: List of progress predictions from sub-samples (each is a list of floats)
        sample_frame_counts: List indicating how many frames each original sample had

    Returns:
        Aggregated progress predictions, one list per original sample
    """
    aggregated_progress = []
    current_idx = 0

    for num_frames in sample_frame_counts:
        if num_frames == 1:
            # Single-frame sample, pass through
            if current_idx < len(progress_preds):
                aggregated_progress.append(progress_preds[current_idx])
                current_idx += 1
            else:
                aggregated_progress.append([])
        else:
            # Collect predictions from sub-samples and extract last prediction from each
            sample_predictions = []
            for i in range(num_frames):
                if current_idx < len(progress_preds):
                    sub_pred = progress_preds[current_idx]
                    # Extract the last prediction from this sub-sample
                    if isinstance(sub_pred, list) and len(sub_pred) > 0:
                        sample_predictions.append(sub_pred[-1])
                    current_idx += 1
            aggregated_progress.append(sample_predictions)

    return aggregated_progress


def process_batch_helper(
    model: Any,
    reward_model: str,
    batch_data: List[Dict[str, Any]],
    job_id: int = 0,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    """Synchronous batch processing."""
    if not batch_data:
        raise ValueError("No samples found in batch data")

    logger.debug(f"[job {job_id}] Processing {len(batch_data)} samples (use_frame_steps={use_frame_steps})")

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

    # Handle frame steps for progress samples - expand into sub-samples with increasing frame counts
    # For RoboReward, we use all frames (no subsampling) since it handles variable-length inputs
    sample_frame_counts = None
    if use_frame_steps and reward_model == "roboreward":
        expanded_samples = []
        sample_frame_counts = []

        for sample in input_samples:
            if isinstance(sample, ProgressSample):
                # Get the frames from the trajectory
                frames = sample.trajectory.frames
                num_frames = frames.shape[0] if hasattr(frames, "shape") else len(frames)

                # Create sub-samples with increasing frame counts: 0:1, 0:2, 0:3, ..., 0:T
                # Use all frames in each sub-sample (no subsampling)
                for i in range(1, num_frames + 1):
                    sub_frames = frames[:i]

                    sub_trajectory = copy.deepcopy(sample.trajectory)
                    sub_trajectory.frames = sub_frames
                    sub_trajectory.frames_shape = (
                        sub_frames.shape if hasattr(sub_frames, "shape") else (len(sub_frames),)
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
        logger.debug(f"[job {job_id}] Expanded samples into {len(input_samples)} sub-samples with frame steps")

    outputs_preference = []
    outputs_progress_list = []

    # Check if we can use batched processing for RoboReward
    if reward_model == "roboreward" and hasattr(model, "compute_progress_batched"):
        # Collect all progress samples for batched processing
        progress_samples = [s for s in input_samples if isinstance(s, ProgressSample)]
        preference_samples = [s for s in input_samples if isinstance(s, PreferenceSample)]

        if progress_samples:
            # Prepare batched inputs
            frames_list = [s.trajectory.frames for s in progress_samples]
            task_descriptions = [s.trajectory.task for s in progress_samples]

            # Process in batch
            logger.debug(f"[job {job_id}] Processing {len(progress_samples)} progress samples in batch with RoboReward")
            batched_results = model.compute_progress_batched(frames_list, task_descriptions)

            # Convert to expected format
            for result in batched_results:
                outputs_progress_list.append({"progress_pred": result})

        # Process preference samples (if any, though RoboReward doesn't support them)
        for sample in preference_samples:
            logger.warning(f"Preference samples not supported for roboreward")
    else:
        # Original sequential processing for other models
        for sample in input_samples:
            if isinstance(sample, PreferenceSample):
                if reward_model != "rlvlmf":
                    logger.warning(f"Preference samples only supported for rlvlmf, got {reward_model}")
                    continue
                result = process_preference_sample(sample, model)
                if result:
                    outputs_preference.append(result)
            elif isinstance(sample, ProgressSample):
                if reward_model not in ["gvl", "vlac", "roboreward"]:
                    logger.warning(f"Progress samples only supported for gvl, vlac, roboreward, got {reward_model}")
                    continue
                result = process_progress_sample(sample, model)
                if result:
                    outputs_progress_list.append(result)

    # Format outputs to match regular eval_server format for compatibility
    # Regular eval_server returns: {"outputs_progress": {"progress_pred": [[...], [...]], ...}}
    # Each inner list is the progress_pred for one sample
    if outputs_progress_list:
        # Extract progress_pred from each result dict - each result has "progress_pred": [list of floats]
        progress_preds = [result.get("progress_pred", []) for result in outputs_progress_list]

        # Aggregate frame-step predictions if needed
        if use_frame_steps and sample_frame_counts is not None:
            progress_preds = aggregate_frame_step_predictions_baseline(progress_preds, sample_frame_counts)

        outputs_progress = {
            "progress_pred": progress_preds,  # List of lists, one per sample
        }
        # Also include success_probs if available
        success_probs_list = [
            result.get("success_probs") for result in outputs_progress_list if result.get("success_probs") is not None
        ]
        if success_probs_list:
            outputs_progress["success_probs"] = success_probs_list
    else:
        outputs_progress = {"progress_pred": []}  # Empty dict with empty list, not None

    # Format preference outputs to match regular eval_server format
    if outputs_preference:
        predictions = [r.get("preference_pred") for r in outputs_preference if r.get("preference_pred") is not None]
        prediction_probs = [
            r.get("preference_pred") for r in outputs_preference if r.get("preference_pred") is not None
        ]
        outputs_preference_dict = {
            "predictions": predictions,
            "prediction_probs": prediction_probs,
        }
    else:
        outputs_preference_dict = None

    return {
        "outputs_preference": outputs_preference_dict,
        "outputs_progress": outputs_progress,
    }


class BaselineEvalServer:
    """Baseline evaluation server that processes batches of samples."""

    def __init__(self, cfg: BaselineEvalConfig):
        self.cfg = cfg
        self.reward_model = cfg.reward_model
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._job_counter = 0
        self._job_counter_lock = Lock()

        logger.info(f"Initializing baseline eval server: reward_model={self.reward_model}")

        # Initialize model
        self._initialize_model()

        logger.info("Baseline eval server initialized successfully")

    def _initialize_model(self):
        """Initialize the baseline model based on config."""
        model_config_dict = (
            asdict(self.cfg.model_config)
            if hasattr(self.cfg.model_config, "__dataclass_fields__")
            else self.cfg.model_config.__dict__
        )

        if self.reward_model == "rlvlmf":
            self.model = RLVLMF(**model_config_dict)
        elif self.reward_model == "gvl":
            self.model = GVL(max_frames=self.cfg.max_frames, **model_config_dict)
        elif self.reward_model == "vlac":
            if not self.cfg.model_path:
                raise ValueError("model_path is required for VLAC baseline")
            self.model = VLAC(model_path=self.cfg.model_path, **model_config_dict)
        elif self.reward_model == "roboreward":
            self.model = RoboReward(model_path=self.cfg.model_path or "teetone/RoboReward-4B", **model_config_dict)
        else:
            raise ValueError(
                f"Unknown reward_model: {self.reward_model}. Must be 'rlvlmf', 'gvl', 'vlac', or 'roboreward'"
            )

        logger.info(f"Loaded {self.reward_model} baseline model")

    async def process_batch(self, batch_data: List[Dict[str, Any]], use_frame_steps: bool = False):
        """Process a batch using the executor."""
        loop = asyncio.get_event_loop()

        with self._job_counter_lock:
            self._job_counter += 1
            job_id = self._job_counter

        logger.debug(
            f"[job {job_id}] Processing batch with {len(batch_data)} samples (use_frame_steps={use_frame_steps})"
        )

        start_time = time.time()

        try:
            # Process batch in thread pool
            result = await loop.run_in_executor(
                self.executor,
                process_batch_helper,
                self.model,
                self.reward_model,
                batch_data,
                job_id,
                use_frame_steps,
            )

            processing_time = time.time() - start_time
            logger.debug(f"[job {job_id}] Completed in {processing_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"[job {job_id}] Error processing batch: {e}", exc_info=True)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "reward_model": self.reward_model,
            "config": asdict(self.cfg),
        }

    def shutdown(self):
        """Shutdown the executor."""
        logger.info("Shutting down baseline eval server...")
        self.executor.shutdown(wait=True)
        logger.info("Baseline eval server shutdown complete")


def create_app(cfg: BaselineEvalConfig, baseline_server: BaselineEvalServer | None = None):
    app = FastAPI(title="Baseline Evaluation Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize baseline server
    baseline_server = baseline_server or BaselineEvalServer(cfg)
    logger.info(f"Baseline eval server initialized with model: {baseline_server.reward_model}")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a batch of samples using the baseline server."""
        logger.debug(f"Received /evaluate_batch request with keys: {list(batch.keys())}")

        # Handle both list and dict formats
        use_frame_steps = False
        if isinstance(batch, list):
            batch_data = batch
        elif isinstance(batch, dict) and "samples" in batch:
            batch_data = batch["samples"]
            use_frame_steps = batch.get("use_frame_steps", False)
        else:
            # Assume it's a single sample wrapped in a dict
            use_frame_steps = batch.pop("use_frame_steps", False)
            batch_data = [batch]

        return await baseline_server.process_batch(batch_data, use_frame_steps=use_frame_steps)

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

        # Reconstruct the original payload structure (baselines don't need torch tensor conversion)
        batch_data = reconstruct_payload_from_npy(
            numpy_arrays,
            other_data,
            trajectory_keys=["chosen_trajectory", "rejected_trajectory", "trajectory"],
            convert_embeddings_to_torch=False,
        )

        # Process the batch
        logger.debug(
            f"Received /evaluate_batch_npy request with {len(numpy_arrays)} numpy arrays "
            f"and {len(other_data)} other fields, use_frame_steps={use_frame_steps}"
        )
        return await baseline_server.process_batch(batch_data, use_frame_steps=use_frame_steps)

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "reward_model": baseline_server.reward_model}

    @app.get("/model_info")
    def get_model_info() -> Dict[str, Any]:
        """Get model information."""
        return baseline_server.get_model_info()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        baseline_server.shutdown()

    return app


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    """Main entry point for baseline evaluation server using Hydra configuration."""
    # Convert Hydra config to dataclass
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)

    # Display the configuration
    display_config(baseline_cfg)

    # Validate reward model
    if baseline_cfg.reward_model not in ["rlvlmf", "gvl", "vlac", "roboreward"]:
        raise ValueError(
            f"reward_model must be 'rlvlmf', 'gvl', 'vlac', or 'roboreward', got {baseline_cfg.reward_model}"
        )

    baseline_server = BaselineEvalServer(baseline_cfg)
    app = create_app(baseline_cfg, baseline_server)

    print(f"Running baseline eval server on {baseline_cfg.server_url}:{baseline_cfg.server_port}")
    print(f"Using {baseline_cfg.reward_model} baseline model")
    uvicorn.run(app, host=baseline_cfg.server_url, port=baseline_cfg.server_port)


if __name__ == "__main__":
    main()
