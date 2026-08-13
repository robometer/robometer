# ArmnetBench Evaluation Dataset Guide

ArmnetBench is an SO-101 robot evaluation + teleoperation dataset in Robometer (RBM) format,
produced from LeRobot v3.0 eval rollouts. It is **already converted** (one video per episode
per camera, with the unified `BASE_FEATURES` metadata), so there is **no `dataset_upload`
loader** — it is consumed directly by `preprocess_datasets.py` / training like any other RBM
Hub dataset.

The HF repo is versioned (`_v01`); the logical `data_source` stays `armnetbench` across
versions so it remains a single source in the training mix.

- HF dataset: `https://huggingface.co/datasets/villekuosmanen/armnetbench_robometer_v01`

## Overview

- `data_source = "armnetbench"` — a single source in the training mix (stable across repo versions).
- Subsets (HF configs) per embodiment: `so101` today (`bimanual_so101` will be added later).
- One trajectory per episode **per camera** (front / top / wrist), `is_robot=true`.
- `quality_label`: teleop → `successful`; policy rollouts → `successful` / `failure` /
  `suboptimal` (from the per-episode eval labels). No numeric `partial_success` — text
  classes only, so failures/suboptimals pair against same-task teleop successes.
- Successful/suboptimal episodes are trimmed to a labelled completion time, removing trailing
  idle frames so the positional progress target ramps to actual task completion.
- Videos: H.264, shortest edge 240 — matching the other RBM datasets.

## Registration in this repo

- `robometer/configs/preprocess.yaml`: `villekuosmanen/armnetbench_robometer_v01` + subset `so101`.
- `robometer/data/dataset_category.py`: `DATASET_MAP["armnetbench_v01"]` alias; `armnetbench`
  added to `DATA_SOURCE_CATEGORY["suboptimal_fail"]` (it has failures/suboptimals with same-task
  successes). Note the alias is versioned (`armnetbench_v01`) while the `data_source` is `armnetbench`.
- `robometer/data/dataset_success_cutoff.txt`: `armnetbench,0.95`.
- `robometer/data/datasets/name_mapping.py`: `villekuosmanen_armnetbench_robometer_v01_so101 →
  armnetbench_so101`.

## Usage

```bash
export ROBOMETER_DATASET_PATH=/path/to/datasets
huggingface-cli download villekuosmanen/armnetbench_robometer_v01 --repo-type dataset \
  --local-dir $ROBOMETER_DATASET_PATH/armnetbench_robometer_v01

uv run python -m robometer.data.scripts.preprocess_datasets \
  --config robometer/configs/preprocess.yaml \
  --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
```

Then train with `data.train_datasets=[armnetbench_v01]` (alias), or reference
`villekuosmanen_armnetbench_robometer_v01_so101` directly in an aggregate. It is intentionally
**not** added to the `rbm-1m-id` default mix — opt it in (and choose weighting) once evaluated.
