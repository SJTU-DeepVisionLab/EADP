# Qwen-VL Experiments

This folder contains the Qwen2.5-VL and Qwen3-VL evaluation code for EADP and the baseline pruning methods.

Supported methods:

- Baseline: fixed-resolution evaluation without visual token pruning
- CDPruner
- DivPrune
- HiPrune
- EADP

The evaluation entry point is the bundled `VLMEvalKit/run.py`. Public scripts are in `scripts/`.

## Environment

Create the environment from the provided file:

```bash
cd Qwen_vl
conda env create -f environment.yml
conda activate qwen-vl
```

Install the bundled VLMEvalKit in editable mode:

```bash
pip install -e ./VLMEvalKit
```

Alternatively, install from `requirements.txt` in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

The scripts use `python` by default. To use a different interpreter:

```bash
export PYTHON=/path/to/python
```

## Models

The supported public model families are:

- `qwen2.5-3b`: `Qwen/Qwen2.5-VL-3B-Instruct`
- `qwen2.5-7b`: `Qwen/Qwen2.5-VL-7B-Instruct`
- `qwen3-8b`: `Qwen/Qwen3-VL-8B-Instruct`

The config defaults to the Hugging Face model IDs above. To use local checkpoints:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir checkpoints/Qwen2.5-VL-7B-Instruct

export QWEN2_5_VL_7B_MODEL_PATH=$(pwd)/checkpoints/Qwen2.5-VL-7B-Instruct
```

Other optional local path variables:

```bash
export QWEN2_5_VL_3B_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export QWEN3_VL_8B_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

## Datasets

VLMEvalKit stores datasets under `LMUData`. Set the data root before running evaluations:

```bash
export LMUData=$HOME/LMUData
```

For many VLMEvalKit datasets, the TSV files are downloaded automatically on first use. If a dataset requires manual preparation, place the prepared files under `LMUData` following the VLMEvalKit dataset format.

The default public script datasets are:

```text
MMBench_DEV_EN_V11 TextVQA_VAL ChartQA_TEST AI2D_TEST OCRBench InfoVQA_VAL DocVQA_VAL
```

You can override them with:

```bash
export DATASETS="MMBench_DEV_EN_V11 TextVQA_VAL"
```

## Common Configuration

All run scripts source `scripts/env.sh`. The most useful variables are:

```bash
export MODEL_FAMILY=qwen2.5-7b
export LMUData=$HOME/LMUData
export GPU_IDS="0"
export DATASETS="MMBench_DEV_EN_V11 TextVQA_VAL"
export TOKENS="128 256 512"
export OUTPUT_ROOT=$(pwd)/outputs
```

For EADP:

```bash
export EADP_ALPHA=0.5
export EADP_BETA=2.0
```

Supported `MODEL_FAMILY` values:

```text
qwen2.5-3b
qwen2.5-7b
qwen3-8b
```

Outputs are written to `outputs/`, and logs are written to `outputs/logs/` by default.

## Quick Start

Run EADP on Qwen2.5-VL-7B with 256 visual tokens:

```bash
cd Qwen_vl

export MODEL_FAMILY=qwen2.5-7b
export LMUData=$HOME/LMUData
export GPU_IDS="0"
export DATASETS="MMBench_DEV_EN_V11"
export TOKENS="256"
export EADP_ALPHA=0.5
export EADP_BETA=2.0

bash scripts/run_eadp.sh
```

Run EADP on Qwen3-VL-8B:

```bash
export MODEL_FAMILY=qwen3-8b
export GPU_IDS="0"
export DATASETS="MMBench_DEV_EN_V11"
export TOKENS="256"
export EADP_ALPHA=0.5
export EADP_BETA=2.0

bash scripts/run_eadp.sh
```

## Running Other Methods

Baseline:

```bash
bash scripts/run_baseline.sh
```

CDPruner:

```bash
bash scripts/run_cdpruner.sh
```

DivPrune:

```bash
bash scripts/run_divprune.sh
```

HiPrune:

```bash
bash scripts/run_hiprune.sh
```

Each script uses the shared variables from `scripts/env.sh`, including `MODEL_FAMILY`, `GPU_IDS`, `DATASETS`, and `TOKENS`.

## Efficiency Evaluation

Efficiency scripts are kept as Python entry points rather than shell wrappers.

For Qwen2.5-VL:

```bash
python scripts/run_efficiency_qwen2vl.py \
  --model Qwen2.5-VL-7B-EADP-256-a0.5-b2.0 \
  --datasets MMBench_DEV_EN_V11 \
  --max-samples 20 \
  --output-dir outputs/efficiency_qwen2vl
```

For Qwen3-VL:

```bash
python scripts/run_efficiency_qwen3vl.py \
  --model Qwen3-VL-8B-EADP-256-a0.5-b2.0 \
  --datasets MMBench_DEV_EN_V11 \
  --max-samples 20 \
  --output-dir outputs/efficiency_qwen3vl
```

The scripts report generation latency, prefill latency, and FLOPs when `fvcore` is available.

## Notes

- EADP is registered in `VLMEvalKit/vlmeval/config.py` as `*-EADP-{tokens}-a{alpha}-b{beta}`.
- Qwen2.5-VL uses fixed 1008 x 1008 inputs for the pruning experiments.
- Qwen3-VL uses fixed 1024 x 1024 inputs for the pruning experiments.
- `outputs/`, `.env`, model checkpoints, and local datasets are ignored by `.gitignore`.
