# LLaVA-Video Experiments

This folder contains the LLaVA-Video evaluation code for EADP and the baseline pruning methods.

Supported methods:

- Baseline: no visual token pruning
- CDPruner
- DivPrune
- HiPrune
- EADP

The default evaluation datasets are MVBench, LongVideoBench, and Video-MME. MLVU is not used here.

## Environment

Create the environment from the provided files:

```bash
cd LLaVA_Video
conda env create -f environment.yml
conda activate llava-video
```

Alternatively, install from `requirements.txt` in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

The scripts use `python` by default. To use a different interpreter:

```bash
export PYTHON=/path/to/python
```

## Model

Download LLaVA-Video-7B:

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 \
  --local-dir checkpoints/LLaVA-Video-7B-Qwen2
```

Then set:

```bash
export MODEL_PATH=$(pwd)/checkpoints/LLaVA-Video-7B-Qwen2
```

You can also point `MODEL_PATH` to any existing local copy of `lmms-lab/LLaVA-Video-7B-Qwen2`.

## Datasets

Set a Hugging Face cache directory:

```bash
export HF_HOME=$HOME/.cache/huggingface
```

Download the datasets:

```bash
bash scripts/download_all.sh
```

This downloads:

- MVBench: `OpenGVLab/MVBench`
- LongVideoBench: `longvideobench/LongVideoBench`
- Video-MME: `lmms-lab/Video-MME`

Extract archives where needed:

```bash
bash scripts/extract_datasets.sh
```

Expected dataset directories under `HF_HOME`:

```text
$HF_HOME/mvbench_video
$HF_HOME/longvideobench
$HF_HOME/videomme
```

## Common Configuration

All run scripts source `scripts/env.sh`. The most useful variables are:

```bash
export MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2
export HF_HOME=$HOME/.cache/huggingface
export GPU_IDS="0"
export DATASETS="mvbench longvideobench_val_v videomme"
export TOKENS="16 32 64"
export MAX_FRAMES=64
export BATCH_SIZE=1
```

For EADP:

```bash
export EADP_ALPHA=0.5
export EADP_BETA=2.0
```

For HiPrune:

```bash
export HIPRUNE_ALPHA=0.2
```

Outputs are written to `outputs/`, and logs are written to `outputs/logs/` by default. Override with:

```bash
export OUTPUT_ROOT=/path/to/outputs
```

## Quick Start

Run EADP on MVBench with 32 tokens per frame:

```bash
cd LLaVA_Video

export MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2
export HF_HOME=$HOME/.cache/huggingface
export GPU_IDS="0"
export DATASETS="mvbench"
export TOKENS="32"
export EADP_ALPHA=0.5
export EADP_BETA=2.0

bash scripts/run_eadp.sh
```

Run all default EADP evaluations:

```bash
export MODEL_PATH=/path/to/LLaVA-Video-7B-Qwen2
export GPU_IDS="0 1 2"
export DATASETS="mvbench longvideobench_val_v videomme"
export TOKENS="16 32 64"
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

Each script uses the same shared variables from `scripts/env.sh`, including `MODEL_PATH`, `GPU_IDS`, `DATASETS`, and `TOKENS`.

## Notes

- `pruner_type=eadp` is the public EADP entry point in `eval/llava_vid_pruned.py`.
- `CLIP_MODEL` defaults to `openai/clip-vit-large-patch14-336` and is used by CDPruner/EADP when needed.
- For memory-constrained GPUs, reduce `MAX_FRAMES`, use fewer datasets per run, or run one token setting at a time.
