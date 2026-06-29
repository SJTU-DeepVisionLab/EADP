# LLaVA Efficiency Evaluation

This directory documents the inference-efficiency evaluation used for EADP and the baseline pruning methods on LLaVA-1.5 / LLaVA-1.6.

The actual evaluation entry point is:

```bash
scripts/run_efficiency_eval.py
```

Run all commands from the `LLaVA` directory.

## Supported Methods

- Baseline: no visual token pruning
- CDPruner
- DivPrune
- HiPrune
- EADP

The pruning architecture is selected by environment variables. Each process should evaluate one pruning method, or baseline only.

```bash
# EADP, default architecture
python scripts/run_efficiency_eval.py --methods eadp ...

# CDPruner
USE_LLAVA_ARCH_CDPRUNER=1 python scripts/run_efficiency_eval.py --methods cdpruner ...

# DivPrune
USE_LLAVA_ARCH_DIVPRUNE=1 python scripts/run_efficiency_eval.py --methods divprune ...

# HiPrune
USE_LLAVA_ARCH_HIPRUNE=1 python scripts/run_efficiency_eval.py --methods hiprune ...
```

For backward compatibility, `--methods ours` is accepted and internally mapped to `eadp`.

## Metrics

| Metric | Description |
| --- | --- |
| `prefill_ms_mean/std` | Time for multimodal input preparation, pruning, and one language-model forward pass before decoding. |
| `latency_ms_mean/std` | End-to-end `model.generate(...)` latency, including prefill and autoregressive decoding. |
| `flops_gflops` | Prefill FLOPs measured by `fvcore`, when tracing succeeds. |

If FLOPs tracing fails for unsupported PyTorch operations, the script still reports prefill time and latency.

## Model

By default, the script derives a public Hugging Face model ID from `--model-version` and `LLAVA_MODEL_SIZE`.

| `--model-version` | `LLAVA_MODEL_SIZE` | Default model |
| --- | --- | --- |
| `v1.5` | `7b` | `liuhaotian/llava-v1.5-7b` |
| `v1.5` | `13b` | `liuhaotian/llava-v1.5-13b` |
| `v1.6` | `7b` | `liuhaotian/llava-v1.6-vicuna-7b` |
| `v1.6` | `13b` | `liuhaotian/llava-v1.6-vicuna-13b` |

You can also pass a local checkpoint explicitly:

```bash
python scripts/run_efficiency_eval.py \
  --model-path /path/to/llava-v1.5-7b \
  --methods eadp \
  --tokens 64 \
  --datasets vizwiz_val
```

## Datasets

The default data root is:

```text
playground/data/eval
```

Override it with:

```bash
export DATA_ROOT=/path/to/eval/data
```

or pass:

```bash
--data-root /path/to/eval/data
```

Predefined datasets:

| Name | Question file | Image folder |
| --- | --- | --- |
| `vizwiz_val` | `vizwiz/llava_vizwiz_val.jsonl` | `vizwiz/val` |
| `textvqa` | `textvqa/llava_textvqa_val_v051_ocr.jsonl` | `textvqa/train_images` |
| `pope` | `pope/llava_pope_test.jsonl` | `pope/val2014` |
| `sqa` | `scienceqa/llava_test_CQM-I.json` | `scienceqa/images/test` |
| `mme` | `MME/llava_mme.jsonl` | `MME/MME_Benchmark_release_version` |
| `gqa` | `gqa/llava_gqa_testdev_balanced.jsonl` | `gqa/data/images` |
| `vqav2` | `vqav2/llava_vqav2_mscoco_test-dev2015.jsonl` | `vqav2/test2015` |

To evaluate a custom dataset, provide both:

```bash
--question-file /path/to/questions.jsonl --image-folder /path/to/images
```

The question file can be JSONL or a single JSON array. Each sample should contain an `image` field and either `text`, `instruction`, or the first user turn in `conversations`.

## Common Arguments

- `--model-path`: local checkpoint path or Hugging Face model ID.
- `--model-version`: `v1.5` or `v1.6`.
- `--methods`: one or more method names. Non-current pruning architectures are skipped.
- `--tokens`: visual token counts. Use `0` only for baseline.
- `--datasets`: predefined dataset names.
- `--max-samples`: maximum samples per dataset.
- `--output-dir`: output directory, defaulting to `efficiency/`.
- `--repeat`: timing repeats per sample.
- `--warmup`: warmup samples before timing.
- `--alpha`: EADP alpha.
- `--beta`: EADP beta.

## Quick Start

Run baseline once:

```bash
cd LLaVA

python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods baseline \
  --tokens 0 \
  --datasets vizwiz_val textvqa pope mme \
  --max-samples 200 \
  --output-dir efficiency
```

Run EADP:

```bash
python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods eadp \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa pope mme \
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
  --datasets vizwiz_val textvqa pope mme \
  --max-samples 200 \
  --output-dir efficiency
```

Run DivPrune:

```bash
USE_LLAVA_ARCH_DIVPRUNE=1 python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods divprune \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa pope mme \
  --max-samples 200 \
  --output-dir efficiency
```

Run HiPrune:

```bash
USE_LLAVA_ARCH_HIPRUNE=1 python scripts/run_efficiency_eval.py \
  --model-version v1.5 \
  --methods hiprune \
  --tokens 32 64 128 \
  --datasets vizwiz_val textvqa pope mme \
  --max-samples 200 \
  --output-dir efficiency
```

## Outputs

Per-run files:

```text
efficiency_{method}_t{token}_{dataset}.json
```

Example:

```text
efficiency_eadp_t64_vizwiz_val.json
```

Summary file:

```text
efficiency_all_results.json
```

The summary is incrementally merged with existing results in the same output directory. New runs replace previous entries with the same `(method, token, dataset)` key.

Each result includes method, token count, dataset, prefill time, latency, optional FLOPs, and number of evaluated samples.

## Notes

- Baseline should be run separately with `--methods baseline --tokens 0`.
- Pruning methods should be run in separate processes because architecture selection happens at import time.
- If memory is limited, reduce `--max-samples`, evaluate fewer datasets per run, or use one token setting at a time.
