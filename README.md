# Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons

[![arXiv](https://img.shields.io/badge/arXiv-2603.02115-b31b1b.svg)](https://arxiv.org/abs/2603.02115)
[![GitHub](https://img.shields.io/badge/GitHub-robometer-181717?logo=github)](https://github.com/robometer/robometer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-FFD21E?logo=huggingface)](https://huggingface.co/robometer/Robometer-4B)
[![Dataset](https://img.shields.io/badge/Dataset-RBM--1M-FFD21E?logo=huggingface)](https://huggingface.co/datasets/)
[![RBM-1M Visualizer](https://img.shields.io/badge/Visualizer-RBM--FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/visualizer)
[![RewardEval UI](https://img.shields.io/badge/%20RewardEval%20UI-FFD21E?logo=huggingface)](https://huggingface.co/spaces/rewardfm/rewardeval_ui)

<p align="center">
  <img src="assets/robometer.jpg" alt="Robometer" width="100%"/>
</p>

---

## Abstract

General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce **Robometer**, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate **RBM-1M**, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications.

---

## 📦 Package structure

```
robometer/
├── robometer/              # Main package
│   ├── data/               # Datasets and preprocessing
│   ├── configs/            # Hydra and experiment configs
│   ├── models/             # Model definitions
│   └── evals/              # Baseline evals (GVL, VLAC, Robodopamine, etc.)
├── eval_commands/          # Shell scripts for baseline evals
├── train.py                # Training entrypoint
└── pyproject.toml          # Dependencies (uv)
```

---

## 🛠️ Setup

### Prerequisites

- Git, Python 3.10+
- NVIDIA drivers (GPU)
- [uv](https://github.com/astral-sh/uv#installation) (recommended)

### Install (main env)

```bash
git clone https://github.com/aliang8/robometer.git
cd robometer

# Create venv and install
uv sync
```

### Dataset setup

```bash
hf auth
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
./scripts/download_processed_datasets.sh
./scripts/untar_processed_datasets.sh
```

For raw download and preprocessing, see [📥 Download raw datasets](#-download-raw-datasets-optional) below.

---

## 🔍 Inference

Inference runs a **pretrained RBM model** on your own videos to get per-frame progress, per-frame success, and (for two trajectories) preference scores.

**Pretrained models (Hugging Face):**

- **[Robometer-4B](https://huggingface.co/robometer/Robometer-4B)** — general-purpose, trained on RBM-1M
- ~~**Robometer-4B-LIBERO** — LIBERO-10 / Spatial / Object / Goal~~ removed because the standard Robometer model is already trained on LIBERO 10/Spatial/Object/Goal+failures and simply performs better than the version trained exclusively on LIBERO

### Inference via HTTP server

Start the eval server on your machine, then call it with a video and task:

```bash
uv run python robometer/evals/eval_server.py \
  server_url=0.0.0.0 \
  server_port=8000
```

Then run the client (no robometer dependency):

```bash
# SOAR
uv run python scripts/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video scripts/example_videos/soar_put_green_stick_in_brown_bowl.mp4 \
  --task "Put green stick in brown bowl" \
  --fps 3

# Berkeley RPT (Wrist)
uv run python scripts/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video scripts/example_videos/berkeley_rpt_stack_cup.mp4 \
  --task "Pick up the yellow cup and stack it on the other cup" \
  --fps 3

# Your own video
uv run python scripts/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video /path/to/video.mp4 \
  --task "your task description"
```

To run the model locally (loads checkpoint from Hugging Face, no server):

```bash
uv run python scripts/example_inference_local.py \
  --model-path robometer/Robometer-4B \
  --video /path/to/video.mp4 \
  --task "your task description"
```

---

## 🏋️ Training

### Training

**Train on RBM-1M in-distribution and evaluate on RBM-1M-OOD**

```bash
uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py \
  data.train_datasets=[rbm-1m-id] \
  data.eval_datasets=[rbm-1m-ood] \
  data.max_frames=4 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.max_steps=5000 \
  custom_eval.reward_alignment=[rbm-1m-ood] \
  custom_eval.policy_ranking=[rbm-1m-ood] \
  custom_eval.confusion_matrix=[rbm-1m-ood]
```

**LIBERO: train on 10 / object / spatial / goal, test on 90.**

```bash
uv run accelerate launch --config_file robometer/configs/distributed/fsdp.yaml train.py \
  data.train_datasets=[libero_pi0] \
  data.eval_datasets=[mw] \
  data.max_frames=4 \
  model.train_progress_head=true \
  model.train_preference_head=true \
  training.max_steps=5000 \
  custom_eval.reward_alignment=[libero_pi0] \
  custom_eval.policy_ranking=[libero_pi0]
```

See `robometer/configs/experiment_configs.py` for more config options.

---

## 🔧 LoRA fine-tune Robometer for new dataset

Preprocess a new dataset, LoRA fine-tune from **Robometer-4B** on your own data, upload the model to the Hub, and run inference:

- **Preprocessing:** Add your dataset to the preprocess config and run the preprocessor; for raw videos (e.g. [MINT-SJTU/RoboFAC-dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)), convert to RBM format first via `dataset_upload`, then preprocess.
- **Fine-tuning:** Set `model.use_peft=true` and `training.resume_from_checkpoint=robometer/Robometer-4B`, then train on your dataset.
- **Upload & inference:** Use `robometer/utils/upload_to_hub.py` to push checkpoints; run `scripts/example_inference_local.py` with your Hub model.

Full step-by-step: **[FINETUNE_ROBOMETER.md](FINETUNE_ROBOMETER.md)**.

---

## 📊 Evaluation

Evaluation runs **benchmark evals** (reward alignment, policy ranking, confusion matrix) on fixed datasets to measure model quality. Use this to reproduce paper results or compare checkpoints.

### Robometer evaluation

Run RBM with `reward_model=rbm`; override `model_path` and `custom_eval.*` as needed. See `eval_commands/*.sh` for ReWIND, Robo-Dopamine, VLAC, RoboReward.

**Reward alignment**

```bash
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
```

**Policy ranking**

```bash
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[policy_ranking] \
    custom_eval.policy_ranking=[rbm-1m-ood] \
    custom_eval.use_frame_steps=false \
    custom_eval.num_examples_per_quality_pr=1000 \
    max_frames=4 \
    model_config.batch_size=32
```

**Confusion matrix**

```bash
uv run python robometer/evals/run_baseline_eval.py \
    reward_model=rbm \
    model_path=robometer/Robometer-4B \
    custom_eval.eval_types=[confusion_matrix] \
    custom_eval.confusion_matrix=[[aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking,jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top,aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking]] \
    max_frames=4 \
    model_config.batch_size=32
```

Details: [robometer/evals/README.md](robometer/evals/README.md).

### Baseline evaluation (all models)

- **RBM:** use the [reward alignment](#robometer-evaluation), [policy ranking](#robometer-evaluation), or [confusion matrix](#robometer-evaluation) commands above; set `model_path` to your checkpoint.
- **ReWIND, Robo-Dopamine, VLAC, RoboReward:** see [robometer/evals/README.md](robometer/evals/README.md) and `eval_commands/reward_alignment.sh`, `eval_commands/policy_ranking.sh`, `eval_commands/confusion_matrix.sh`. For Robo-Dopamine use `.venv-robodopamine/bin/python` (vLLM) instead of `uv run`.

---

## 📊 Dataset generation

Supported: **AgiBotWorld** (streaming), **LIBERO** (HDF5), and custom configs.

```bash
# AgiBotWorld
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/agibot_world.yaml

# LIBERO
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen.yaml \
  --dataset.dataset_path=LIBERO/libero/datasets/libero_90 --dataset.dataset_name=libero_90
```

See dataset_upload README and dataset_guides for adding datasets.

---

## 📥 Download raw datasets (optional)

If you prefer not to use the processed datasets:

```bash
export ROBOMETER_DATASET_PATH=/path/to/your/robometer_dataset
./scripts/download_data.sh

# Preprocess
uv run python -m robometer.data.scripts.preprocess_datasets --config robometer/configs/preprocess.yaml
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/save/processed_datasets
```

---

## 📑 License

This project is licensed under the [MIT License](LICENSE).
