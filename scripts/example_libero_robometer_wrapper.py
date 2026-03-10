"""
Example reward-model wrapper for LIBERO environments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "LIBERO"))

import numpy as np
import gymnasium as gym
import gymnasium.vector as gym_vector

from robometer.evals.eval_utils import raw_dict_to_sample, extract_rewards_from_output, extract_success_probs_from_output
from robometer.evals.eval_server import process_batch_helper
from robometer.utils.setup_utils import setup_batch_collator
from robometer.utils.tensor_utils import t2n
from robometer.utils.save import load_model_from_hf

class GymToGymnasiumWrapper(gym.Env):
    """
    A wrapper to convert a classic Gym environment to a Gymnasium-like interface.
    It adapts `reset()` and `step()` signatures, handles info dict changes, and supports compatibility.
    """

    def __init__(self, env, time_limit: int = None):
        super().__init__()  # make sure Env is initialized
        self.env = env
        # Action space remains the same
        if hasattr(self.env, "action_space"):
            self.action_space = self.env.action_space
        if hasattr(self.env, "observation_space"):
            self.observation_space = self.env.observation_space
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", {})
        self.time_limit = time_limit
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        # Reset step counter
        self.current_step = 0
        # Gym reset sometimes does not support 'seed' or 'options'
        if seed is not None:
            try:
                obs = self.env.reset(seed=seed)
            except TypeError:
                self.env.seed(seed)
                obs = self.env.reset()
        else:
            obs = self.env.reset()
        info = {}
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.current_step += 1
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            # Gymnasium expects terminated, truncated
            if self.time_limit is not None and self.current_step >= self.time_limit:
                truncated = True
            else:
                truncated = info.get("TimeLimit.truncated", False)
            return obs, reward, terminated, truncated, info
        elif len(result) == 5:
            # Already modern API
            return result
        else:
            raise ValueError("Unexpected number of outputs from env.step")

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Forward other attributes/methods to original env
        return getattr(self.env, name)



@dataclass
class RewardModelStepOutput:
    # Kept for potential downstream debugging/typing; not required by wrappers.
    reward: float
    success_prob: float
    per_key_rewards: Dict[str, float]
    per_key_success_probs: Dict[str, float]


class _RewardModelInferenceMixin:
    """
    Shared reward-model inference logic with local model.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        max_frames: Optional[int] = None,
    ):
        if model_path is not None:
            reward_model_config, tokenizer, processor, reward_model = load_model_from_hf(
                model_path=model_path,
                device=device,
            )
            reward_model.eval()

            self.reward_model = reward_model
            self.reward_model_config = reward_model_config

        if self.reward_model is None:
            raise ValueError("reward_model must be provided")

        # Determine max_frames
        if max_frames is not None:
            self.max_frames = int(max_frames)
        elif self.reward_model_config is not None:
            self.max_frames = int(getattr(getattr(self.reward_model_config, "data", None), "max_frames", 16))
        else:
            self.max_frames = 16

        # Local model path: set up collator once
        self.processor = None
        self.tokenizer = None
        self.batch_collator = None
        self._model_device = None
        self._model_type = None
        if self.reward_model is not None:
            self.processor = getattr(self.reward_model, "processor", None)
            self.tokenizer = getattr(self.reward_model, "tokenizer", None)
            if self.processor is None or self.tokenizer is None:
                raise ValueError(
                    "processor and tokenizer must be available on reward_model "
                    "(reward_model.processor / reward_model.tokenizer)"
                )

            # Ensure multi-image behavior is enabled (matches SPUR buffer)
            if self.reward_model_config is not None:
                data_cfg = getattr(self.reward_model_config, "data", None)
                if data_cfg is not None and hasattr(data_cfg, "use_multi_image") and not data_cfg.use_multi_image:
                    data_cfg.use_multi_image = True

            # Resolve model type/device once
            self._model_type = getattr(getattr(self.reward_model_config, "model", None), "model_type", None)
            if self._model_type is None:
                raise ValueError("reward_model_config.model.model_type is required for local reward inference")
            self._model_device = getattr(self.reward_model, "device", None)
            if self._model_device is None:
                try:
                    import torch

                    self._model_device = next(self.reward_model.parameters()).device
                    if isinstance(self._model_device, torch.device):
                        self._model_device = str(self._model_device)
                except Exception:
                    self._model_device = None

            self.batch_collator = setup_batch_collator(
                self.processor, self.tokenizer, self.reward_model_config, is_eval=True
            )

    def _compute_rewards_batch(
        self, batch_raw: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[float]]:
        """
        Returns lists: (progress_rewards, success_probs).
        """
        if len(batch_raw) == 0:
            return [], []

        samples = [
            raw_dict_to_sample(raw_data=raw, max_frames=self.max_frames, sample_type="progress")
            for raw in batch_raw
        ]

        is_discrete_mode = (
            self.reward_model_config is not None
            and getattr(getattr(self.reward_model_config, "loss", None), "progress_loss_type", None) == "discrete"
        )
        num_bins = (
            getattr(getattr(self.reward_model_config, "loss", None), "progress_discrete_bins", None)
            if self.reward_model_config is not None
            else None
        )
        outputs = process_batch_helper(
            model_type=self._model_type,
            model=self.reward_model,
            tokenizer=self.tokenizer,
            batch_collator=self.batch_collator,
            device=self._model_device,
            batch_data=[s.model_dump() for s in samples],
            job_id=0,
            is_discrete_mode=bool(is_discrete_mode),
            num_bins=num_bins,
        )
        rewards = extract_rewards_from_output(outputs)
        success_probs = extract_success_probs_from_output(outputs)
        return rewards.tolist(), success_probs.tolist()


class LiberoRobometerRewardWrapper(gym.Wrapper, _RewardModelInferenceMixin):
    """
    Non-vector LIBERO wrapper that replaces rewards with reward-model predictions.
    """

    def __init__(
        self,
        env,
        model_path: str,
        device: str,
        reward_relabeling_keys: Sequence[str],
        *,
        use_relative_rewards: bool = False,
        add_estimated_reward: bool = False,
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_threshold: float = 0.65,
        max_frames: Optional[int] = None,
    ):
        self.env = GymToGymnasiumWrapper(env, time_limit=400)
        gym.Wrapper.__init__(self, self.env)
        _RewardModelInferenceMixin.__init__(
            self,
            model_path=model_path,
            device=device,
            max_frames=max_frames,
        )

        self.reward_relabeling_keys = list(reward_relabeling_keys)
        if len(self.reward_relabeling_keys) == 0:
            raise ValueError("reward_relabeling_keys must be non-empty")
        
        # Action space remains the same
        if not hasattr(self.env, "action_space"):
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)    
        else:
            self.action_space = self.env.action_space

        self.use_relative_rewards = bool(use_relative_rewards)
        self.add_estimated_reward = bool(add_estimated_reward)
        self.use_success_detection = bool(use_success_detection)
        self.success_detection_duration = int(success_detection_duration)
        self.success_detection_threshold = float(success_detection_threshold)

        self._frames: Dict[str, Deque[np.ndarray]] = {}
        self.language_instruction = self.env.language_instruction
        self.episode_id = 0
        self._step_in_episode: int = 0
        self._prev_reward: float = 0.0
        self._success_window: Deque[float] = deque(maxlen=self.success_detection_duration)

    def _get_language_instruction(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Optional[str]:
        if isinstance(info, dict) and "language_instruction" in info:
            return info.get("language_instruction")
        if isinstance(obs, dict) and isinstance(obs.get("prompt"), str):
            return obs.get("prompt")
        return self.language_instruction

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.language_instruction = self.env.language_instruction
        self.episode_id += 1

        self._frames = {k: [] for k in self.reward_relabeling_keys}
        self._step_in_episode = 0
        self._prev_reward = 0.0
        self._success_window = deque(maxlen=self.success_detection_duration)

        if isinstance(obs, dict):
            for k in self.reward_relabeling_keys:
                if k in obs:
                    self._frames[k].append(t2n(obs[k]))
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        if not isinstance(info, dict):
            info = {} if info is None else dict(info)
        # In LIBERO, done is only True when task succeeds, so success = done
        # But don't overwrite if already present in info
        if "success" not in info:
            info["success"] = terminated
        if terminated:
            assert env_reward == 1.0, "Reward should be 1.0 when task succeeds"
        
        env_reward -= 1  # reward is -1, 0

        if isinstance(obs, dict):
            for k in self.reward_relabeling_keys:
                if k in obs:
                    self._frames[k].append(t2n(obs[k]))

        # Prepare per-key inputs for this timestep
        per_key_rewards: Dict[str, float] = {}
        per_key_success: Dict[str, float] = {}

        for key_idx, key in enumerate(self.reward_relabeling_keys):
            frames = np.stack(list(self._frames[key]), axis=0) if len(self._frames[key]) > 0 else np.array([])
            raw = dict(
                frames=frames,
                task=self.language_instruction,
                id=self.episode_id,
                metadata=dict(
                        subsequence_length=len(self._frames[key]) if self._frames[key] is not None else 0,
                ),
                video_embeddings=None,
                text_embedding=None,
            )
            rewards, success_probs = self._compute_rewards_batch([raw])
            per_key_rewards[key] = rewards[0]
            per_key_success[key] = success_probs[0]

        pred_reward = np.mean(list(per_key_rewards.values())) if per_key_rewards else 0.0
        success_prob = np.mean(list(per_key_success.values())) if per_key_success else 0.0

        # Relative reward option
        if self.use_relative_rewards:
            current = pred_reward
            pred_reward = pred_reward - self._prev_reward
            self._prev_reward = current
            if terminated or truncated:
                self._prev_reward = 0.0

        # Success detection option
        if self.use_success_detection:
            self._success_window.append(success_prob)
            if len(self._success_window) == self.success_detection_duration:
                votes = sum(1 for p in self._success_window if p >= self.success_detection_threshold)
                if votes > (self.success_detection_duration / 2):
                    terminated = True
                    info["success"] = True
                    info["success_from_reward_model"] = True

        # Decide what reward to return
        if self.add_estimated_reward:
            out_reward = env_reward + pred_reward
        else:
            out_reward = pred_reward

        info["env_reward"] = env_reward
        info["predicted_reward"] = pred_reward
        info["success_prob"] = success_prob
        info["predicted_rewards_by_key"] = per_key_rewards
        info["success_probs_by_key"] = per_key_success
        info["step_in_episode"] = int(self._step_in_episode)

        self._step_in_episode += 1

        # If the env auto-resets under the hood, start a fresh history when done/truncated.
        if terminated or truncated:
            self._frames = {k: [] for k in self.reward_relabeling_keys}
            self.language_instruction = self.env.language_instruction
            self._step_in_episode = 0
            self._success_window = deque(maxlen=self.success_detection_duration)

        return obs, out_reward, terminated, truncated, info


class VectorLiberoRobometerRewardWrapper(gym_vector.VectorWrapper, _RewardModelInferenceMixin):
    """
    Vectorized LIBERO wrapper that replaces rewards with reward-model predictions per env.
    """

    def __init__(
        self,
        env: gym_vector.VectorEnv,
        model_path: str,
        device: str,
        reward_relabeling_keys: Sequence[str],
        *,
        use_relative_rewards: bool = False,
        add_estimated_reward: bool = False,
        replace_reward: bool = True,
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_threshold: float = 0.65,
        max_frames: Optional[int] = None,
    ):
        gym_vector.VectorWrapper.__init__(self, env)
        _RewardModelInferenceMixin.__init__(
            self,
            model_path=model_path,
            device=device,
            max_frames=max_frames,
        )

        self.reward_relabeling_keys = list(reward_relabeling_keys)
        if len(self.reward_relabeling_keys) == 0:
            raise ValueError("reward_relabeling_keys must be non-empty")

        self.use_relative_rewards = bool(use_relative_rewards)
        self.add_estimated_reward = bool(add_estimated_reward)
        self.replace_reward = bool(replace_reward)
        self.use_success_detection = bool(use_success_detection)
        self.success_detection_duration = int(success_detection_duration)
        self.success_detection_threshold = float(success_detection_threshold)

        self._n = int(getattr(self.env, "num_envs", 1))
        self._frames: List[Dict[str, Deque[np.ndarray]]] = []
        self._language_instructions: List[Optional[str]] = []
        self._episode_ids: List[int] = []
        self._step_in_episode: List[int] = []
        self._prev_rewards: List[float] = []
        self._success_windows: List[Deque[float]] = []

        self._init_state()

    def _init_state(self):
        self._n = int(getattr(self.env, "num_envs", self._n))
        self._frames = [
            {k: deque(maxlen=self.max_frames) for k in self.reward_relabeling_keys} for _ in range(self._n)
        ]
        self._language_instructions = [None for _ in range(self._n)]
        self._episode_ids = [0 for _ in range(self._n)]
        self._step_in_episode = [0 for _ in range(self._n)]
        self._prev_rewards = [0.0 for _ in range(self._n)]
        self._success_windows = [deque(maxlen=self.success_detection_duration) for _ in range(self._n)]

    def _get_language_instruction_vec(self, obs: Dict[str, Any], info: Any) -> List[Optional[str]]:
        getter = getattr(self.env, "get_language_instruction", None)
        if callable(getter):
            try:
                instr = getter()
                if isinstance(instr, str):
                    return [instr] * self._n
            except Exception:
                pass

        # Try prompt in obs
        if isinstance(obs, dict) and "prompt" in obs:
            p = obs["prompt"]
            if isinstance(p, list) and len(p) == self._n:
                return [str(x) for x in p]
            if isinstance(p, np.ndarray) and p.shape[0] == self._n:
                return [str(x) for x in p.tolist()]
        # Fallback: single instruction attribute (shared across envs)
        shared = getattr(self.env, "language_instruction", None)
        return [shared] * self._n

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._init_state()

        if isinstance(obs, dict):
            instrs = self._get_language_instruction_vec(obs, info)
            for i in range(self._n):
                self._language_instructions[i] = instrs[i]
                self._episode_ids[i] += 1

            for k in self.reward_relabeling_keys:
                if k in obs:
                    arr = t2n(obs[k])
                    if arr is not None and arr.shape[0] == self._n:
                        for i in range(self._n):
                            self._frames[i][k].append(arr[i])

        return obs, info

    def step(self, actions):
        obs, env_rewards, terminateds, truncateds, info = self.env.step(actions)

        # Normalize arrays
        env_rewards_np = t2n(env_rewards)
        terminateds_np = t2n(terminateds).astype(bool)
        truncateds_np = t2n(truncateds).astype(bool)

        if env_rewards_np is None:
            env_rewards_np = np.zeros((self._n,), dtype=np.float64)

        # In LIBERO, done is only True when task succeeds, so success = terminated.
        # Mirror the non-vector wrapper's reward shift (0/1 -> -1/0).
        for i in range(self._n):
            if bool(terminateds_np[i]):
                assert float(env_rewards_np[i]) == 1.0, "Reward should be 1.0 when task succeeds"
        env_rewards_shifted = env_rewards_np.astype(np.float64) - 1.0

        # Gymnasium VectorEnv may auto-reset in the same step; if so, terminal obs is in info["final_observation"]
        final_obs = None
        if isinstance(info, dict) and "final_observation" in info:
            final_obs = info.get("final_observation")

        reset_instrs = self._get_language_instruction_vec(obs, info) if isinstance(obs, dict) else [None] * self._n
        task_for_model: List[Optional[str]] = [
            (self._language_instructions[i] if self._language_instructions[i] is not None else reset_instrs[i])
            for i in range(self._n)
        ]

        # Update frame histories using the correct observation for this transition.
        # If SAME_STEP autoreset is enabled, use terminal obs from final_observation when available.
        if isinstance(obs, dict):
            for k in self.reward_relabeling_keys:
                if k not in obs:
                    continue
                arr_reset = t2n(obs[k])
                if arr_reset is None or arr_reset.shape[0] != self._n:
                    continue
                for i in range(self._n):
                    frame_i = arr_reset[i]
                    if final_obs is not None and i < len(final_obs) and final_obs[i] is not None:
                        fo_i = final_obs[i]
                        if isinstance(fo_i, dict) and k in fo_i:
                            frame_i = t2n(fo_i[k])
                    self._frames[i][k].append(frame_i)

        # Batch reward computation per key across envs
        per_env_per_key_reward: Dict[str, List[float]] = {k: [0.0] * self._n for k in self.reward_relabeling_keys}
        per_env_per_key_success: Dict[str, List[float]] = {k: [0.0] * self._n for k in self.reward_relabeling_keys}

        for key_idx, key in enumerate(self.reward_relabeling_keys):
            batch_raw: List[Dict[str, Any]] = []
            for i in range(self._n):
                frames = np.stack(list(self._frames[i][key]), axis=0) if len(self._frames[i][key]) > 0 else np.array([])
                batch_raw.append(
                    dict(
                        frames=frames,
                        task=task_for_model[i],
                        id=int(self._episode_ids[i]),
                        metadata=dict(subsequence_length=len(self._frames[i][key])),
                        video_embeddings=None,
                        text_embedding=None,
                    )
                )

            rewards_k, success_k = self._compute_rewards_batch(batch_raw)
            for i in range(self._n):
                per_env_per_key_reward[key][i] = rewards_k[i] if i < len(rewards_k) else 0.0
                per_env_per_key_success[key][i] = success_k[i] if i < len(success_k) else 0.0

        # Aggregate across keys
        pred_rewards_abs = np.zeros((self._n,), dtype=np.float64)
        success_probs = np.zeros((self._n,), dtype=np.float64)
        for i in range(self._n):
            r_vals = [per_env_per_key_reward[k][i] for k in self.reward_relabeling_keys]
            s_vals = [per_env_per_key_success[k][i] for k in self.reward_relabeling_keys]
            pred_rewards_abs[i] = np.mean(r_vals) if len(r_vals) else 0.0
            success_probs[i] = np.mean(s_vals) if len(s_vals) else 0.0

        pred_rewards_out = pred_rewards_abs.copy()
        if self.use_relative_rewards:
            for i in range(self._n):
                cur = float(pred_rewards_abs[i])
                pred_rewards_out[i] = cur - self._prev_rewards[i]
                self._prev_rewards[i] = cur
                if terminateds_np[i] or truncateds_np[i]:
                    self._prev_rewards[i] = 0.0

        # Success detection
        if self.use_success_detection:
            for i in range(self._n):
                self._success_windows[i].append(float(success_probs[i]))
                if len(self._success_windows[i]) == self.success_detection_duration:
                    votes = sum(1 for p in self._success_windows[i] if p >= self.success_detection_threshold)
                    if votes > (self.success_detection_duration / 2):
                        terminateds_np[i] = True

        # Determine reward output
        if self.add_estimated_reward:
            out_rewards = env_rewards_shifted + pred_rewards_out
        else:
            out_rewards = env_rewards_shifted if not self.replace_reward else pred_rewards_out

        # Inject info
        # Gymnasium vector env `info` is typically a dict of arrays; keep it dict-like.
        if info is None:
            info = {}
        if isinstance(info, dict):
            info = dict(info)
            if "success" not in info:
                info["success"] = terminateds_np.copy()
            info["env_reward"] = env_rewards_shifted.astype(np.float64)
            info["predicted_reward"] = pred_rewards_out.astype(np.float64)
            info["predicted_reward_abs"] = pred_rewards_abs.astype(np.float64)
            info["success_prob"] = success_probs.astype(np.float64)
            info["step_in_episode"] = np.asarray(self._step_in_episode, dtype=np.int32)
        # Also provide per-key arrays
        if isinstance(info, dict):
            for k in self.reward_relabeling_keys:
                info[f"predicted_reward/{k}"] = np.asarray(per_env_per_key_reward[k], dtype=np.float64)
                info[f"success_prob/{k}"] = np.asarray(per_env_per_key_success[k], dtype=np.float64)

        # Advance step counters and clear per-env state on episode end (to support auto-reset vector envs)
        for i in range(self._n):
            self._step_in_episode[i] += 1
            if terminateds_np[i] or truncateds_np[i]:
                self._frames[i] = {k: deque(maxlen=self.max_frames) for k in self.reward_relabeling_keys}
                self._language_instructions[i] = reset_instrs[i]
                self._step_in_episode[i] = 0
                self._success_windows[i] = deque(maxlen=self.success_detection_duration)
                self._episode_ids[i] += 1

                # If SAME_STEP autoreset happened, seed next episode history with reset obs immediately.
                if isinstance(obs, dict) and final_obs is not None and i < len(final_obs) and final_obs[i] is not None:
                    for k in self.reward_relabeling_keys:
                        if k not in obs:
                            continue
                        arr_reset = t2n(obs[k])
                        if arr_reset is not None and arr_reset.shape[0] == self._n:
                            self._frames[i][k].append(arr_reset[i])

        return obs, out_rewards.astype(np.float64), terminateds_np, truncateds_np, info

def main():
    try:
        from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
        from libero.libero import benchmark, get_libero_path
    except ImportError:
        print("LIBERO not found. Please install LIBERO.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run RBM inference locally: load model from HuggingFace and compute per-frame progress and success.",
        epilog="Outputs: <out>.npy (rewards), <out>_success_probs.npy, <out>_progress_success.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", default="robometer/Robometer-4B", help="HuggingFace model id or local checkpoint path")
    parser.add_argument("--task-suite-name", default="libero_90", help="LIBERO task suite name")
    parser.add_argument("--task-id", default=28, type=int, help="LIBERO task id")
    parser.add_argument("--vectorized", action="store_true", help="Run in vectorized mode")
    parser.add_argument("--num-envs", default=2, type=int, help="Number of environments to run in parallel")
    args = parser.parse_args()

    if not args.vectorized:
        print("Testing Single LIBERO Robometer Reward Wrapper")
        seed = np.random.randint(0, 1000000)
        # Get task info
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        task = task_suite.get_task(args.task_id)

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
        base_env = OffScreenRenderEnv(**env_args)
        base_env.seed(seed)

        robometer_libero_env = LiberoRobometerRewardWrapper(base_env, 
                                                            model_path=args.model_path, 
                                                            device="cuda", 
                                                            reward_relabeling_keys=["agentview_image"], 
                                                            add_estimated_reward=True, 
                                                            )
        obs, info = robometer_libero_env.reset()
        for i in range(10):
            action = np.random.uniform(-1, 1, 7)
            obs, reward, terminated, truncated, info = robometer_libero_env.step(action)
            print(f"Reward at step {i}: {reward}")
        
        robometer_libero_env.close()

    else:
        print("Testing Vectorized LIBERO Robometer Reward Wrapper")
        def make_env():
            seed = np.random.randint(0, 1000000)
            # Get task info
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[args.task_suite_name]()
            task = task_suite.get_task(args.task_id)

            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

            env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
            base_env = OffScreenRenderEnv(**env_args)
            base_env.seed(seed)
            sample_obs = base_env.reset()
            env = GymToGymnasiumWrapper(base_env, time_limit=400)
            # Action space remains the same
            if not hasattr(env, "action_space"):
                env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            if not hasattr(env, "observation_space"):
                # Create observation space from sample_obs, which is a dict of arrays
                obs_space_dict = {}
                for k, v in sample_obs.items():
                    # Common LIBERO obs dict may include images (uint8) and sometimes text prompts.
                    if isinstance(v, (str, bytes, bytearray)) or (isinstance(v, np.ndarray) and v.dtype.kind in {"U", "S"}):
                        obs_space_dict[k] = gym.spaces.Text(max_length=2048)
                        continue
                    v_arr = np.asarray(v)
                    dt = v_arr.dtype
                    if np.issubdtype(dt, np.uint8):
                        # Images: bounded [0, 255]
                        obs_space_dict[k] = gym.spaces.Box(
                            low=np.zeros(v_arr.shape, dtype=np.uint8),
                            high=np.full(v_arr.shape, 255, dtype=np.uint8),
                            shape=v_arr.shape,
                            dtype=np.uint8,
                        )
                    elif np.issubdtype(dt, np.integer):
                        ii = np.iinfo(dt)
                        obs_space_dict[k] = gym.spaces.Box(
                            low=np.full(v_arr.shape, ii.min, dtype=dt),
                            high=np.full(v_arr.shape, ii.max, dtype=dt),
                            shape=v_arr.shape,
                            dtype=dt,
                        )
                    else:
                        # Floats/other numeric: unbounded
                        obs_space_dict[k] = gym.spaces.Box(
                            low=np.full(v_arr.shape, -np.inf, dtype=np.float32),
                            high=np.full(v_arr.shape, np.inf, dtype=np.float32),
                            shape=v_arr.shape,
                            dtype=np.float32,
                        )
                env.observation_space = gym.spaces.Dict(obs_space_dict)

            return env

        env_fns = [make_env for _ in range(args.num_envs)]
        env = gym.vector.SyncVectorEnv(env_fns)
        robometer_libero_env = VectorLiberoRobometerRewardWrapper(env, 
                                                                model_path=args.model_path, 
                                                                device="cuda", 
                                                                reward_relabeling_keys=["agentview_image"], 
                                                                add_estimated_reward=True, 
                                                                )
        obs, info = robometer_libero_env.reset()
        for i in range(10):
            actions = np.random.uniform(-1, 1, (args.num_envs, 7))
            obs, rewards, terminateds, truncateds, infos = robometer_libero_env.step(actions)
            print(f"Rewards at step {i}: {rewards}")
        
        robometer_libero_env.close()


if __name__ == "__main__":
    main()
