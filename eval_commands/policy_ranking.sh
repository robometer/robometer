# ReWIND
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=64

# Robo-Dopamine 3B (run with venv Python so vLLM is found; do not use uv run)
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-3B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64 \
    model_config.batch_size=1

# Robo-Dopamine 8B
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview \
    model_config.eval_mode=forward \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64 \
    model_config.batch_size=1

# VlAC
uv run --extra vlac --python .venv-vlac/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    max_frames=64 \
    custom_eval.num_examples_per_quality_pr=1000

# RoboReward-4B
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.pad_frames=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=64

# Robometer-4B
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=8 \
    model_config.batch_size=32

# Robometer-4B Libero Ablation (only trained on LIBERO datasets, so don't recommend using this model for actual reward modeling.)
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/Robometer-4B-LIBERO \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=20 \
    max_frames=4 \
    model_config.batch_size=32