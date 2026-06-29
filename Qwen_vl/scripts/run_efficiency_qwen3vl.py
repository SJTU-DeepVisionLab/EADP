#!/usr/bin/env python3
"""
Qwen3-VL inference efficiency evaluation: latency, prefill time, FLOPs.

This mirrors scripts/run_efficiency_qwen2vl.py, but uses the Qwen3 fixed-res
wrapper path for 1024x1024 inputs and the Qwen3 pruner wrappers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(SCRIPT_DIR)
VLMEVALKIT = os.path.join(QWEN_ROOT, "VLMEvalKit")
for path in (VLMEVALKIT, QWEN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_method_and_token(model_name: str):
    if model_name == "Qwen3-VL-8B-Instruct-1024":
        return "baseline", 0
    match = re.search(r"CDPruner-(\d+)", model_name)
    if match:
        return "cdpruner", int(match.group(1))
    match = re.search(r"DivPruner-(\d+)", model_name)
    if match:
        return "divprune", int(match.group(1))
    match = re.search(r"HiPrune-(\d+)", model_name)
    if match:
        return "hiprune", int(match.group(1))
    match = re.search(r"EADP-(\d+)-a[\d.]+-b[\d.]+", model_name)
    if match:
        return "ours", int(match.group(1))
    return "unknown", 0


def set_max_new_tokens(model, max_new_tokens: int) -> None:
    if hasattr(model, "max_new_tokens"):
        model.max_new_tokens = max_new_tokens
    if hasattr(model, "generate_kwargs"):
        model.generate_kwargs["max_new_tokens"] = max_new_tokens


def build_message(model, dataset, dataset_name, row):
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(row, dataset=dataset_name)
    return dataset.build_prompt(row)


def build_prefill_inputs(model, message, dataset_name):
    """Return raw Qwen3 processor inputs or pruned inputs_embeds for one prefill."""
    messages = model._build_messages(message, dataset=dataset_name)
    inputs = model._processor_inputs(messages)
    has_images = inputs.get("pixel_values", None) is not None
    image_grid_thw = inputs.get("image_grid_thw", None)

    if (
        has_images
        and image_grid_thw is not None
        and hasattr(model, "_build_pruned_inputs")
        and hasattr(model, "_get_pruned_image_features")
    ):
        instruction_embeds_seq = model._get_instruction_sequence_embedding(
            message, dataset=dataset_name
        )
        pruned_embeds, pruned_sizes = model._get_pruned_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            instruction_embeds_seq=instruction_embeds_seq,
            message=message,
            dataset=dataset_name,
        )
        inputs_embeds, attention_mask = model._build_pruned_inputs(
            inputs["input_ids"],
            inputs["attention_mask"],
            pruned_embeds,
            pruned_sizes,
        )
        return ("embeds", inputs_embeds, attention_mask)

    return ("raw", {key: value for key, value in inputs.items() if value is not None})


def qwen3_language_model(model):
    inner = model.model
    if hasattr(inner, "model") and hasattr(inner.model, "language_model"):
        return inner.model.language_model
    if hasattr(inner, "language_model"):
        return inner.language_model
    return None


def simple_position_ids(inputs_embeds: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = inputs_embeds.shape
    pos = torch.arange(seq_len, device=inputs_embeds.device)
    pos = pos.view(1, 1, -1).expand(3, batch, -1)
    return pos


def forward_prefill(model, prefill):
    with torch.no_grad():
        if prefill[0] == "raw":
            return model.model(**prefill[1], use_cache=False)

        inputs_embeds, attention_mask = prefill[1], prefill[2]
        try:
            return model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except Exception:
            language_model = qwen3_language_model(model)
            if language_model is None:
                raise
            return language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=simple_position_ids(inputs_embeds),
                use_cache=False,
            )


def measure_prefill_time(model, message, dataset_name, repeat):
    prefill = build_prefill_inputs(model, message, dataset_name)
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward_prefill(model, prefill)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.mean(times_ms)), float(np.std(times_ms))


def capture_language_inputs_for_baseline(model, raw_kwargs):
    captured = []

    def hook(_module, args):
        if args and isinstance(args[0], torch.Tensor):
            captured.append(args[0].detach())

    language_model = qwen3_language_model(model)
    if language_model is None or not hasattr(language_model, "layers") or not language_model.layers:
        return None

    handle = language_model.layers[0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model.model(**raw_kwargs, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        return None
    return captured[0], raw_kwargs.get("attention_mask")


def measure_flops(model, message, dataset_name):
    if not HAS_FVCORE:
        return None
    prefill = build_prefill_inputs(model, message, dataset_name)

    import logging
    loggers = [logging.getLogger("fvcore"), logging.getLogger("fvcore.nn")]
    old_levels = [logger.getEffectiveLevel() for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

    try:
        if prefill[0] == "embeds":
            inputs_embeds, attention_mask = prefill[1], prefill[2]
        else:
            captured = capture_language_inputs_for_baseline(model, prefill[1])
            if captured is None:
                return None
            inputs_embeds, attention_mask = captured
        if attention_mask is None:
            return None

        language_model = qwen3_language_model(model)
        if language_model is None:
            return None

        class LanguageWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, inputs_embeds, attention_mask, position_ids):
                return self.inner(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )

        wrapper = LanguageWrapper(language_model)
        position_ids = simple_position_ids(inputs_embeds)
        flops = FlopCountAnalysis(wrapper, (inputs_embeds, attention_mask, position_ids))
        total = flops.total()
        return total / 1e9 if total is not None else None
    except Exception as err:
        warnings.warn(f"FLOPs measurement skipped: {err}", UserWarning)
        return None
    finally:
        for logger, level in zip(loggers, old_levels):
            logger.setLevel(level)


def measure_latency(model, message, dataset_name, repeat):
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model.generate(message=message, dataset=dataset_name)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.mean(times_ms)), float(np.std(times_ms))


def run_one_dataset(model, model_name, dataset, dataset_name, config):
    method_label, token = parse_method_and_token(model_name)
    max_samples = config["max_samples"]
    warmup = config["warmup"]
    repeat = config["repeat"]
    set_max_new_tokens(model, config["max_new_tokens"])

    indices = list(range(min(max_samples, len(dataset))))
    if not indices:
        return None

    model.set_dump_image(dataset.dump_image)
    samples = [
        build_message(model, dataset, dataset_name, dataset.data.iloc[i])
        for i in indices
    ]

    for sample in samples[: min(warmup, len(samples))]:
        with torch.no_grad():
            model.generate(message=sample, dataset=dataset_name)
    torch.cuda.synchronize()
    gc.collect()

    prefill_mean, prefill_std = measure_prefill_time(
        model, samples[0], dataset_name, repeat
    )
    flops_gflops = measure_flops(model, samples[0], dataset_name)

    latency_times = []
    for sample in tqdm(samples, desc=f"{method_label} t={token} {dataset_name}", leave=False):
        latency_mean, _ = measure_latency(model, sample, dataset_name, repeat)
        latency_times.append(latency_mean)

    result = {
        "method": method_label,
        "token": token,
        "dataset": dataset_name,
        "model_name": model_name,
        "latency_ms_mean": float(np.mean(latency_times)),
        "latency_ms_std": float(np.std(latency_times)),
        "num_samples": len(samples),
        "prefill_ms_mean": prefill_mean,
        "prefill_ms_std": prefill_std,
    }
    if flops_gflops is not None:
        result["flops_gflops"] = round(flops_gflops, 2)
    return result


def ensure_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and cuda_home.rstrip(os.sep).endswith("bin"):
        cuda_home = os.path.dirname(cuda_home.rstrip(os.sep))
        os.environ["CUDA_HOME"] = cuda_home
        if "CUDA_PATH" in os.environ:
            os.environ["CUDA_PATH"] = cuda_home
        return
    if cuda_home:
        return
    for candidate in ("/usr/local/cuda-12.1", "/usr/local/cuda"):
        if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = candidate
            return


def main():
    ensure_cuda_home()
    os.environ.setdefault(
        "QWEN3_VL_8B_MODEL_PATH",
        "Qwen/Qwen3-VL-8B-Instruct",
    )

    parser = argparse.ArgumentParser(
        description="Qwen3-VL inference efficiency: latency, prefill time, FLOPs."
    )
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", default=os.path.join(QWEN_ROOT, "outputs", "efficiency_qwen3"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="Skip model/dataset runs with existing JSON output.")
    args = parser.parse_args()

    ws_bak = os.environ.pop("WORLD_SIZE", None)

    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset

    dataset_map = {}
    for name in args.datasets:
        dataset = build_dataset(name)
        if dataset is None:
            print(f"Skip dataset (build failed): {name}", file=sys.stderr)
            continue
        dataset_map[name] = dataset
    if not dataset_map:
        print("No datasets available.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    config = {
        "max_samples": args.max_samples,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
    }

    for model_name in args.model:
        if model_name not in supported_VLM:
            print(f"Skip unknown model: {model_name}", file=sys.stderr)
            continue
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        pending_datasets = []
        for dataset_name in dataset_map:
            out_path = os.path.join(
                args.output_dir, f"efficiency_{safe_name}_{dataset_name}.json"
            )
            if args.resume and os.path.exists(out_path):
                print(f"Skip existing: {out_path}")
                with open(out_path, encoding="utf-8") as file:
                    all_results.append(json.load(file))
                continue
            pending_datasets.append(dataset_name)
        if not pending_datasets:
            continue

        try:
            model = supported_VLM[model_name](max_new_tokens=args.max_new_tokens)
        except Exception as err:
            print(f"Failed to load model {model_name}: {err}", file=sys.stderr)
            continue

        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "model") and hasattr(model.model, "cuda"):
            model.model.cuda()
            model.model.eval()

        for dataset_name in pending_datasets:
            dataset = dataset_map[dataset_name]
            result = run_one_dataset(model, model_name, dataset, dataset_name, config)
            if result is None:
                continue
            all_results.append(result)
            out_path = os.path.join(
                args.output_dir, f"efficiency_{safe_name}_{dataset_name}.json"
            )
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
            print(f"Wrote {out_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if ws_bak is not None:
        os.environ["WORLD_SIZE"] = ws_bak

    summary_path = os.path.join(args.output_dir, "efficiency_all_results.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)
    print(f"Wrote {summary_path} ({len(all_results)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
