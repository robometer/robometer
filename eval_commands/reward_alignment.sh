# ReWIND
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=8 \
    model_config.batch_size=64

# Robo-Dopamine 3B (run with venv Python so vLLM is found; do not use uv run)
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-3B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=1

# Robo-Dopamine 8B
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview \
    model_config.eval_mode=forward \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=1

# VLAC
uv run --extra vlac --python .venv-vlac/bin/python  robometer/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=30 \
    custom_eval.pad_frames=false \
    max_frames=64

# RoboReward-4B
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-4B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=64 \
    model_config.batch_size=32

# Robometer-4B
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[rbm-1m-id,rbm-1m-ood] \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames=5 \
    custom_eval.reward_alignment_max_trajectories=30 \
    max_frames=4 \
    model_config.batch_size=32

# Robometer-4B Libero Ablation
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=aliangdw/Robometer-4B-LIBERO \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[libero_pi0] \
    custom_eval.use_frame_steps=false \
    custom_eval.reward_alignment_max_trajectories=100 \
    max_frames=4 \
    model_config.batch_size=32