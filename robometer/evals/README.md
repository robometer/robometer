# Baseline Evaluation Guide

This guide documents how to run baseline evaluations for GVL, RL-VLM-F, VLAC, and Robometer on all datasets.

## Overview

The baseline evaluation system supports:

- **GVL** (Gemini Video Language): Progress prediction via Gemini API
- **RL-VLM-F**: Preference comparison via Vision-Language Models (Gemini or OpenAI)
- **VLAC**: Progress prediction using the VLAC model
- **Robometer (RBM)**: Progress prediction and preference comparison using trained Robometer checkpoints:
  - **robometer/Robometer-4B** – general-purpose model
  - **aliangdw/Robometer-4B-LIBERO** – LIBERO-focused variant
- **Robo-Dopamine (GRM)**: Progress prediction using Robo-Dopamine GRM (separate venv; see setup below)
- **ReWiND**: Progress prediction and preference using ReWiND checkpoints (same interface as RBM)

## Evaluation Types

1. **Reward Alignment**: Evaluates progress prediction along trajectories
2. **Policy Ranking**: Evaluates ability to rank trajectories by quality/partial_success
3. **Quality Preference**: Evaluates preference comparison between trajectory pairs

## Standard Evaluation Configuration

For comprehensive evaluation across all datasets, use the following settings:

- **Reward Alignment**: 20 trajectories per dataset
- **Policy Ranking**: 20 tasks per dataset, 2 trajectories per quality label
- **Quality Preference**: 100 comparisons per dataset

## Prerequisites

### Environment Variables

For **GVL** and **RL-VLM-F** (Gemini):
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

For **RL-VLM-F** (OpenAI):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### VLAC Model Setup

For **VLAC**, you need to:

1. **Create a separate virtual environment** (required because VLAC dependencies conflict with Robometer/RBM dependencies):
   ```bash
   uv venv .venv-vlac
   uv pip install -e ".[vlac]" --python .venv-vlac/bin/python
   ```
   This creates a separate `.venv-vlac` environment with VLAC-compatible dependencies.

2. **Download the model** from Hugging Face (auto-download if `model_path` is a Hugging Face repo ID) or provide a local path to the model checkpoint

### Robo-Dopamine (GRM) setup

The **Robo-Dopamine GRM** baseline is built on [Robo-Dopamine](https://github.com/FlagOpen/Robo-Dopamine). It uses vLLM, which requires **torch 2.9.x**. Reward-fm pins **torch 2.8.0** and **xformers**, so the resolver cannot satisfy both. Use a dedicated venv and install **vLLM first**, then the rest of the deps:

1. **Create a separate virtual environment** (requires CUDA 12.x). Run from the **repo root**:
   ```bash
   cd /path/to/robometer   # or your repo root
   uv venv .venv-robodopamine
   uv pip install vllm --python .venv-robodopamine/bin/python
   uv pip install -r requirements-robodopamine.txt --python .venv-robodopamine/bin/python
   uv pip install -e . --no-deps --python .venv-robodopamine/bin/python
   ```
   Verify vLLM is installed:
   ```bash
   .venv-robodopamine/bin/python -c "import vllm; print('vLLM OK')"
   ```
   Use this venv only for the Robo-Dopamine baseline.

2. **Run evals** using the venv’s Python directly (do **not** use `uv run`, or vLLM may not be found):
   ```bash
   .venv-robodopamine/bin/python robometer/evals/run_baseline_eval.py reward_model=robodopamine ...
   ```
   Run from the repo root. The config key is still `reward_model=robodopamine`.

### Robometer (RBM) / ReWiND Model Setup

For **Robometer** or **ReWiND** (`reward_model=rbm` or `reward_model=rewind`), provide:

1. **`model_path`**: Hugging Face repo ID or local checkpoint path.
   - **Robometer**: `robometer/Robometer-4B` (general) or `aliangdw/Robometer-4B-LIBERO` (LIBERO only, not recommended even for LIBERO itself as it performs worse than the general model).
   - Config is loaded automatically from the checkpoint when available.
2. **Optional**: `model_config.batch_size` for inference batch size (default: 32).


### Output

- `output_dir`: Output directory (default: `./baseline_eval_output/<reward_model>_<timestamp>`)

## Output Structure

Each evaluation run creates an output directory with:

```
baseline_eval_output/
└── <reward_model>_<timestamp>/
    ├── baseline_metrics.json                    # Aggregated metrics
    ├── <eval_type>_<dataset>_results.json       # Detailed results per eval type and dataset
    ├── <eval_type>_<dataset>_task_groups.json  # Task-level groupings (for policy_ranking, quality_preference)
    ├── <eval_type>_<dataset>_task_details.json # Task-level details (for policy_ranking, quality_preference)
    └── <eval_type>_<dataset>_plots/             # Visualization plots (for reward_alignment)
        └── trajectory_*.gif                     # Plot+video GIFs
```

## Dataset Names

Common dataset names (use the full dataset key as it appears in the Hugging Face dataset):

- `aliangdw_metaworld_metaworld_eval`
- `jesbu1_roboarena_eval_debug_nowrist`
- `jesbu1_oxe_rfm_eval`
- `abraranwar_libero_rfm`
- `ykorkmaz_libero_failure_rfm`
- And many more...

To see all available datasets, check the dataset loading code or Hugging Face.

## Notes

1. **RoboArena Datasets**: For RoboArena datasets, policy ranking uses `partial_success` values instead of quality labels. Use `num_partial_successes` parameter for circular sampling across partial_success values.

2. **Frame Steps vs Whole Trajectories**: 
   - `use_frame_steps=true`: Generates subsequences (0:frame_step, 0:2*frame_step, etc.)
   - `use_frame_steps=false`: Uses whole trajectories

3. **VLAC Dependencies**: VLAC requires a separate environment due to dependency conflicts. See `pyproject.toml` for optional dependency groups.

4. **API Rate Limits**: Be aware of API rate limits when running RL-VLM-F or GVL evaluations on large datasets.

5. **GPU Memory**: VLAC evaluations may require significant GPU memory depending on model size and batch settings.

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `GEMINI_API_KEY` or `OPENAI_API_KEY` is set in your environment
2. **VLAC Model Not Found**: Check that `vlac_model_path` is correct or that the Hugging Face model can be downloaded.
3. **Robometer/RBM Model Not Found**: Use `model_path="robometer/Robometer-4B"`, or ensure your local path is correct.
4. **CUDA Out of Memory**: Reduce `vlac_batch_num` or `model_config.batch_size`, or use a smaller model.
5. **Dataset Not Found**: Verify the dataset name matches exactly (including underscores and case).