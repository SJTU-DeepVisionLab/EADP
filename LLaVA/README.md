# LLaVA Experiments

This folder contains the LLaVA-1.5 / LLaVA-1.6 evaluation code for EADP and the baseline pruning methods.

Supported methods:

- Baseline: no visual token pruning
- CDPruner
- DivPrune
- HiPrune
- EADP

The standard evaluation scripts follow the original LLaVA layout. See [EVAL.md](EVAL.md) for detailed evaluation setup, dataset preparation, and benchmark commands. The efficiency evaluation entry point is `scripts/run_efficiency_eval.py`, with metric definitions and command examples in [efficiency/README.md](efficiency/README.md).

## Environment

Create the environment from the provided file:

```bash
cd LLaVA
conda env create -f environment.yml
conda activate pruner
```

Alternatively, install from `requirements.txt` in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

If your local project layout supports editable installation, you can also install the local package:

```bash
pip install -e .
```

The efficiency script uses `python` from the active environment.

## Models

The efficiency script can derive public Hugging Face model IDs from `--model-version` and `LLAVA_MODEL_SIZE`.

| `--model-version` | `LLAVA_MODEL_SIZE` | Default model |
| --- | --- | --- |
| `v1.5` | `7b` | `liuhaotian/llava-v1.5-7b` |
| `v1.5` | `13b` | `liuhaotian/llava-v1.5-13b` |
| `v1.6` | `7b` | `liuhaotian/llava-v1.6-vicuna-7b` |
| `v1.6` | `13b` | `liuhaotian/llava-v1.6-vicuna-13b` |

To use a local checkpoint, pass `--model-path` explicitly:

```bash
python scripts/run_efficiency_eval.py \
  --model-path /path/to/llava-v1.5-7b \
  --methods eadp \
  --tokens 64 \
  --datasets vizwiz_val
```

## Datasets

For standard LLaVA benchmark evaluation, refer to [EVAL.md](EVAL.md).

The default dataset root is:

```text
playground/data/eval
```

Override it with:

```bash
export DATA_ROOT=/path/to/eval/data
```

or:

```bash
python scripts/run_efficiency_eval.py --data-root /path/to/eval/data ...
```

Predefined datasets:

```text
vizwiz_val textvqa pope sqa mme gqa vqav2
```

The expected files and directory layout are documented in `efficiency/README.md`. You can also evaluate a custom dataset with:

```bash
--question-file /path/to/questions.jsonl --image-folder /path/to/images
```

## Method Selection

The pruning implementation is selected at import time by environment variables. Run different pruning methods in separate processes.

EADP is the default architecture:

```bash
python scripts/run_efficiency_eval.py --methods eadp ...
```

Other methods:

```bash
USE_LLAVA_ARCH_CDPRUNER=1 python scripts/run_efficiency_eval.py --methods cdpruner ...
USE_LLAVA_ARCH_DIVPRUNE=1 python scripts/run_efficiency_eval.py --methods divprune ...
USE_LLAVA_ARCH_HIPRUNE=1 python scripts/run_efficiency_eval.py --methods hiprune ...
```

Baseline uses the same architecture as the current process, but should be run separately with `--methods baseline --tokens 0`.

## Quick Start

Run baseline once:

```bash
cd LLaVA

python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods baseline \
  --tokens 0 \
  --datasets vizwiz_val textvqa \
  --max-samples 200 \
  --output-dir efficiency
```

Run EADP:

```bash
python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods eadp \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa \
  --max-samples 200 \
  --alpha 0.5 \
  --beta 1.0 \
  --output-dir efficiency
```

Run CDPruner:

```bash
USE_LLAVA_ARCH_CDPRUNER=1 python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods cdpruner \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa \
  --max-samples 200 \
  --output-dir efficiency
```

Run DivPrune:

```bash
USE_LLAVA_ARCH_DIVPRUNE=1 python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods divprune \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa \
  --max-samples 200 \
  --output-dir efficiency
```

Run HiPrune:

```bash
USE_LLAVA_ARCH_HIPRUNE=1 python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods hiprune \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa \
  --max-samples 200 \
  --output-dir efficiency
```

## Efficiency Outputs

Per-run results are written as:

```text
efficiency/efficiency_{method}_t{token}_{dataset}.json
```

The merged summary is:

```text
efficiency/efficiency_all_results.json
```

The summary is incrementally merged across separate runs. New results replace previous entries with the same `(method, token, dataset)` key.

## Notes

- EADP is implemented by the default LLaVA architecture in `llava/model/llava_arch.py`.
- Architecture selection happens when `llava.model.language_model.llava_llama` is imported, so method-specific runs must use separate Python processes.
- `efficiency/efficiency_*.json`, `outputs/`, local datasets, checkpoints, and caches are ignored by `.gitignore`.
