# GVL
export GEMINI_API_KEY="your-api-key-here"
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.model_name=gemini-2.5-flash-lite \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8

uv run python robometer/evals/run_baseline_eval.py \
    reward_model=gvl \
    model_config.provider=openai \
    model_config.model_name=gpt-4o-mini \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8

# ReWIND
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rewind \
    model_path=rewardfm/rewind-scale-rfm1M-32layers-8frame-20260118-180522 \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    custom_eval.use_frame_steps=false \
    max_frames=8 \
    model_config.batch_size=64

# Robo-Dopamine 3B (run with venv Python so vLLM is found; do not use uv run)
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-3B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=64 \
    model_config.batch_size=1

# Robo-Dopamine 8B
.venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py \
    reward_model=robodopamine \
    model_path=tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview \
    model_config.eval_mode=forward \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=64 \
    model_config.batch_size=1

# VLAC
uv run --extra vlac --python .venv-vlac/bin/python python robometer/evals/run_baseline_eval.py \
    reward_model=vlac \
    model_path=InternRobotics/VLAC \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    custom_eval.pad_frames=false \
    max_frames=64

# RoboReward-8B
# without koch
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=64

# on all
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=roboreward \
    model_path=teetone/RoboReward-8B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]] \
    max_frames=64

# Robometer-4B
# without koch
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=8 \
    model_config.batch_size=32

# on all
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking,jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all]] \
    max_frames=8 \
    model_config.batch_size=32