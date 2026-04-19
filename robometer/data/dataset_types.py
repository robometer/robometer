#!/usr/bin/env python3
"""
Dataclasses for RBM model dataset trajectory structures.
Defines the standard format for HuggingFace dataset trajectories.
"""

from typing import Any, Union, List, Dict, Tuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch


class Trajectory(BaseModel):
    """Trajectory structure containing frames, metadata, and progress information."""

    # Core trajectory fields
    frames: Union[List[str], np.ndarray, None] = None
    frames_shape: Optional[Tuple] = None

    # If embeddings are precomputed
    embeddings_path: Optional[str] = None
    video_embeddings: Union[torch.Tensor, np.ndarray, None] = None
    text_embedding: Union[torch.Tensor, np.ndarray, None] = None

    id: Optional[str] = None
    task: Optional[str] = None
    lang_vector: Union[np.ndarray, List[float], None] = None
    data_source: Optional[str] = None
    quality_label: Optional[str] = None
    is_robot: Optional[bool] = None

    # Progress and metadata
    # Can be List[float] for continuous progress, np.ndarray, or List[np.ndarray] for C51 discrete distributions
    target_progress: Optional[Union[List[float], List[torch.Tensor], torch.Tensor, None]] = None
    partial_success: Optional[Union[float, torch.Tensor]] = None  # float for continuous, Tensor for C51 discrete
    success_label: Optional[List[float]] = None  # Success labels for each frame (1.0 for success, 0.0 for failure)
    predict_last_frame_mask: Optional[List[float]] = (
        None  # Mask for partial_success: 1.0 for last frame if partial_success < 1.0, otherwise all 1.0s
    )
    metadata: Optional[Dict[str, Any]] = None
    original_sampled_index: Optional[int] = None
    data_gen_strategy: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProgressSample(BaseModel):
    """Sample structure for progress evaluation."""

    trajectory: Trajectory
    sample_type: str = "progress"
    data_gen_strategy: Optional[str] = None
    resample_attempts: int = 1


class PreferenceSample(BaseModel):
    """Sample structure for preference prediction: chosen vs rejected where chosen is preferred."""

    # Trajectories
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory

    sample_type: str = "preference"
    data_gen_strategy: Optional[str] = None
    resample_attempts: int = 1


SampleType = Union[PreferenceSample, ProgressSample]
