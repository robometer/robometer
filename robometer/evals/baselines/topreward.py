#!/usr/bin/env python3
"""TOPReward baseline: token probabilities as zero-shot rewards for progress prediction.

Uses VLM log-likelihood of task completion (e.g. "True") conditioned on trajectory
prefixes to produce a dense progress curve. No task-specific training.

Reference: https://github.com/TOPReward/TOPReward
Paper: TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics (arXiv:2602.19313)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from robometer.utils.logger import get_logger

logger = get_logger()

# Default image size used by TOPReward for video frames
_TOPREWARD_IMG_SIZE = 224


def _to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a single frame (H,W,C) or (C,H,W) to PIL RGB."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        return Image.fromarray(frame, "L").convert("RGB")
    return Image.fromarray(frame[:, :, :3], "RGB").resize((_TOPREWARD_IMG_SIZE, _TOPREWARD_IMG_SIZE))


def _frames_array_to_pil_list(frames_array: np.ndarray) -> List[Image.Image]:
    """Convert (T,H,W,C) or (T,C,H,W) to list of PIL images."""
    T = frames_array.shape[0]
    out = []
    for t in range(T):
        frame = frames_array[t]
        out.append(_to_pil(frame))
    return out


@dataclass
class _InstructionRewardResult:
    reward: float
    reduction: str
    token_count: int
    prefix_lengths: Optional[List[int]] = None
    prefix_rewards: Optional[List[float]] = None
    normalized_prefix_rewards: Optional[List[float]] = None


def _normalize_rewards(rewards: Sequence[float], method: str = "minmax") -> np.ndarray:
    """Normalize rewards to [0, 1] (minmax)."""
    rewards_arr = np.array(rewards, dtype=np.float64)
    if len(rewards_arr) == 0:
        return rewards_arr
    if len(rewards_arr) == 1:
        return np.array([1.0])
    if method == "minmax":
        r_min, r_max = rewards_arr.min(), rewards_arr.max()
        if r_max == r_min:
            return np.ones_like(rewards_arr)
        return (rewards_arr - r_min) / (r_max - r_min)
    raise ValueError(f"Unknown normalization method: {method}")


class TopReward:
    """TOPReward baseline: instruction-conditioned log-likelihood progress from a video VLM."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_frames: int = 64,
        num_prefix_samples: int = 15,
        reduction: str = "mean",
        add_chat_template: bool = True,
        fps: float = 2.0,
        **kwargs: Any,
    ):
        """
        Args:
            model_path: HuggingFace model ID (Qwen3-VL recommended).
            max_frames: Max frames per trajectory (sampled if longer).
            num_prefix_samples: Number of prefix lengths to evaluate for progress curve.
            reduction: "mean" or "sum" over instruction tokens.
            add_chat_template: Whether to use chat template for instruction prompt.
            fps: Frames per second for video input to the VLM.
        """
        self.model_path = model_path
        self.max_frames = max_frames
        self.num_prefix_samples = num_prefix_samples
        self.reduction = reduction
        self.add_chat_template = add_chat_template
        self.fps = fps
        logger.info(f"Loading TOPReward model: {model_path}")
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation=attn_impl,
                )
                logger.info(f"Loaded with attn_implementation={attn_impl}")
                break
            except Exception as e:
                if attn_impl == "eager":
                    raise
                logger.warning(f"attn_implementation={attn_impl} failed: {e}, trying next.")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _compute_instruction_reward(
        self,
        frames: List[Image.Image],
        instruction: str,
    ) -> float:
        """Compute log-likelihood reward for instruction given video (single trajectory)."""
        prompt_text = (
            "The above video shows a robot manipulation trajectory that completes the following task: "
        )
        eos_token = self.processor.tokenizer.eos_token

        if self.add_chat_template:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True or not. The answer is:"
            )
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames, "fps": self.fps},
                        {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = f"{prompt_chat}True"
            image_inputs, video_inputs = process_vision_info(templated_messages)
        else:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True or not. The answer is: True"
            )
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames, "fps": self.fps},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            )
            if eos_token:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            image_inputs, video_inputs = process_vision_info(user_messages)

        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        labels = inputs["input_ids"].clone()
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]
        reward = (
            masked_log_probs.sum().item()
            if self.reduction == "sum"
            else masked_log_probs.mean().item()
        )
        return reward

    def _compute_instruction_rewards_for_prefixes(
        self,
        frames: List[Image.Image],
        instruction: str,
    ) -> _InstructionRewardResult:
        """Compute rewards for trajectory prefixes; return normalized curve."""
        num_frames = len(frames)
        num_samples = min(self.num_prefix_samples, num_frames)
        if num_frames > 2:
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted(set(int(x) for x in prefix_lengths))
        else:
            prefix_lengths = [num_frames]

        rewards = []
        for length in prefix_lengths:
            prefix_frames = frames[:length]
            r = self._compute_instruction_reward(prefix_frames, instruction)
            rewards.append(r)
        normalized = _normalize_rewards(rewards).tolist()
        return _InstructionRewardResult(
            reward=rewards[-1],
            reduction=self.reduction,
            token_count=0,
            prefix_lengths=prefix_lengths,
            prefix_rewards=rewards,
            normalized_prefix_rewards=normalized,
        )

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute progress curve from trajectory frames using TOPReward (log-likelihood of task completion).

        Args:
            frames_array: (T, H, W, C) or (T, C, H, W), RGB.
            task_description: Instruction string.
            reference_video_path: Unused (TOPReward is zero-shot).

        Returns:
            progress: (T,) float array in [0, 1].
        """
        if frames_array is None or frames_array.size == 0:
            return np.array([], dtype=np.float64)

        num_frames = frames_array.shape[0]
        if num_frames > self.max_frames:
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            frames_array = frames_array[indices]
            num_frames = frames_array.shape[0]

        frames = _frames_array_to_pil_list(frames_array)
        result = self._compute_instruction_rewards_for_prefixes(frames, task_description or "Complete the task.")

        prefix_lengths = result.prefix_lengths or []
        normalized = result.normalized_prefix_rewards or []
        if not prefix_lengths or not normalized:
            return np.zeros(num_frames, dtype=np.float64)

        # Map prefix_lengths (1-based frame counts) -> normalized reward; then interpolate to every frame
        lengths = np.array(prefix_lengths, dtype=np.float64)
        values = np.array(normalized, dtype=np.float64)
        frame_indices = np.arange(1, num_frames + 1, dtype=np.float64)
        progress = np.interp(frame_indices, lengths, values)
        return np.clip(progress.astype(np.float64), 0.0, 1.0)


