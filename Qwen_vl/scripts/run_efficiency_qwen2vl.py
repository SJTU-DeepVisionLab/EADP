#!/usr/bin/env python3
"""
Qwen2.5-VL inference efficiency evaluation: Latency, Prefill time, FLOPs.
Uses VLMEvalKit datasets (TSV under LMUDataRoot ~/LMUData); same build_prompt as normal eval.
Outputs per-run JSON and efficiency_all_results.json for collect_efficiency_results.py.
Run from Qwen_vl root: python scripts/run_efficiency_qwen2vl.py --model <name> --datasets <name> ...

Warnings you may see (safe to ignore for correctness):
- fvcore "Unsupported operator" / "uncalled submodules": FLOPs trace only counts the LLM forward;
  visual encoder and some ops (e.g. FlashAttn) are not counted. The GFLOPs value is still useful
  for comparing methods. Script suppresses these logs during FLOPs measurement.
- transformers "temperature ... sample-based generation": harmless when do_sample=False; comes from
  the library default and does not affect greedy decoding.
"""

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

# Optional: fvcore for FLOPs
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

# Add Qwen2_5_vl and VLMEvalKit to path so vlmeval and model.* resolve
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_ROOT = os.path.dirname(SCRIPT_DIR)
VLMEVALKIT = os.path.join(QWEN_ROOT, "VLMEvalKit")
for p in (VLMEVALKIT, QWEN_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _expand2square(pil_img, background_color):
    """Pad image to square (same as model_fixed_res)."""
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def _build_prefill_inputs(model, message, dataset_name):
    """Build inputs for one forward (prefill). Returns (inputs_embeds, attention_mask) for pruner path,
    or dict of kwargs for model.model(**kwargs) for baseline path. Returns None on failure.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return None
    messages = []
    if getattr(model, "system_prompt", None) is not None:
        messages.append({"role": "system", "content": model.system_prompt})
    messages.append({
        "role": "user",
        "content": model._prepare_content(message, dataset=dataset_name),
    })
    text = model.processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info([messages])
    if images is not None:
        bg_color = tuple(
            int(x * 255) for x in model.processor.image_processor.image_mean
        )
        images = [
            _expand2square(img, bg_color).resize((36 * 28, 36 * 28))
            for img in images
        ]
    inputs = model.processor(
        text=text, images=images, videos=videos,
        padding=True, return_tensors="pt",
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    has_images = inputs.get("pixel_values", None) is not None
    image_grid_thw = inputs.get("image_grid_thw", None)

    if has_images and image_grid_thw is not None and hasattr(model, "_build_pruned_inputs") and hasattr(model, "_get_pruned_image_features"):
        if not hasattr(model, "_get_instruction_sequence_embedding"):
            return None
        instruction_embeds_seq = model._get_instruction_sequence_embedding(message, dataset=dataset_name)
        pruned_embeds, pruned_sizes = model._get_pruned_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"],
            instruction_embeds_seq=instruction_embeds_seq, message=message, dataset=dataset_name,
        )
        new_inputs_embeds, new_attention_mask = model._build_pruned_inputs(
            inputs["input_ids"], inputs["attention_mask"], pruned_embeds, pruned_sizes
        )
        return ("embeds", new_inputs_embeds, new_attention_mask)
    if has_images and image_grid_thw is not None:
        kwargs = {k: v for k, v in inputs.items() if v is not None}
        return ("raw", kwargs)
    return None


def measure_prefill_time(model, message, dataset_name, repeat):
    """Time one forward (prefill) with CUDA events. Returns (mean_ms, std_ms) or (None, None) on failure."""
    prefill = _build_prefill_inputs(model, message, dataset_name)
    if prefill is None:
        return None, None
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            if prefill[0] == "embeds":
                _ = model.model(inputs_embeds=prefill[1], attention_mask=prefill[2])
            else:
                _ = model.model(**prefill[1])
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.mean(times_ms)), float(np.std(times_ms))


def _capture_merged_embeds_for_baseline(model, raw_kwargs):
    """Run one forward with raw inputs and capture merged inputs_embeds via hook (for baseline FLOPs).
    Returns (inputs_embeds, attention_mask) or None on failure.
    """
    captured = []

    def hook(_module, args):
        # Pre-hook receives (module, input); input is tuple of positional args to forward.
        # First argument to decoder layer is hidden_states (merged inputs_embeds).
        if args and isinstance(args[0], torch.Tensor):
            captured.append(args[0].detach().clone())

    inner = model.model
    if not hasattr(inner, "model") or not hasattr(inner.model, "layers") or len(inner.model.layers) == 0:
        return None
    handle = inner.model.layers[0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            _ = inner(**raw_kwargs)
    finally:
        handle.remove()
    if not captured:
        return None
    attn_mask = raw_kwargs.get("attention_mask")
    if attn_mask is None:
        return None
    return captured[0], attn_mask


def measure_flops(model, message, dataset_name):
    """FLOPs for one prefill forward via fvcore. Returns GFLOPs or None.
    For both baseline and pruner we trace only the LLM forward (inputs_embeds->logits) so values
    are comparable; baseline uses a hook to capture merged embeds and avoids tracing vision/Triton.
    """
    if not HAS_FVCORE:
        return None
    prefill = _build_prefill_inputs(model, message, dataset_name)
    if prefill is None:
        return None
    import logging
    to_suppress = [logging.getLogger("fvcore"), logging.getLogger("fvcore.nn")]
    old_levels = [log.getEffectiveLevel() for log in to_suppress]
    for log in to_suppress:
        log.setLevel(logging.ERROR)
    try:
        if prefill[0] == "embeds":
            inp_emb, attn_mask = prefill[1], prefill[2]
        else:
            # Baseline: capture merged inputs_embeds via one forward + hook, then count LLM FLOPs only
            raw_kwargs = prefill[1]
            captured = _capture_merged_embeds_for_baseline(model, raw_kwargs)
            if captured is None:
                return None
            inp_emb, attn_mask = captured

        class WrapperEmbeds(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.inner = m
            def forward(self, inputs_embeds, attention_mask):
                return self.inner(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        wrapper = WrapperEmbeds(model.model)
        flops = FlopCountAnalysis(wrapper, (inp_emb, attn_mask))
        total = flops.total()
        return total / 1e9 if total is not None else None
    except Exception as e:
        warnings.warn(f"FLOPs measurement skipped: {e}", UserWarning)
        return None
    finally:
        for log, old_level in zip(to_suppress, old_levels):
            log.setLevel(old_level)


def parse_method_and_token(model_name: str):
    """Parse method_label and token from config model name. Returns (method_label, token)."""
    if "Instruct-1008" in model_name or re.match(r"Qwen2\.5-VL-\d+B-Instruct-1008", model_name):
        return "baseline", 0
    m = re.search(r"CDPruner-(\d+)", model_name)
    if m:
        return "cdpruner", int(m.group(1))
    m = re.search(r"DivPruner-(\d+)", model_name)
    if m:
        return "divprune", int(m.group(1))
    m = re.search(r"HiPrune-(\d+)", model_name)
    if m:
        return "hiprune", int(m.group(1))
    m = re.search(r"EADP-(\d+)-a[\d.]+-b[\d.]+", model_name)
    if m:
        return "ours", int(m.group(1))
    return "unknown", 0


def measure_latency(model, message, dataset_name, repeat, max_new_tokens):
    """Measure end-to-end generate latency (ms) with CUDA events."""
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
    """Run efficiency on one model over one VLMEvalKit dataset. dataset has .data, .build_prompt, .dump_image."""
    method_label, token = parse_method_and_token(model_name)
    max_samples = config.get("max_samples", 200)
    warmup = config.get("warmup", 5)
    repeat = config.get("repeat", 3)
    max_new_tokens = config.get("max_new_tokens", 128)
    if hasattr(model, "max_new_tokens"):
        model.max_new_tokens = max_new_tokens

    n_total = len(dataset)
    indices = list(range(min(max_samples, n_total)))
    if not indices:
        return None

    # Build prompts the same way as inference: model.build_prompt or dataset.build_prompt
    model.set_dump_image(dataset.dump_image)
    samples = []
    for i in indices:
        line = dataset.data.iloc[i]
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(line, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(line)
        samples.append(struct)

    # Warmup
    for i in range(min(warmup, len(samples))):
        with torch.no_grad():
            model.generate(message=samples[i], dataset=dataset_name)
    torch.cuda.synchronize()
    gc.collect()

    # Prefill time and FLOPs on first sample (same as LLaVA script)
    prefill_mean, prefill_std = measure_prefill_time(model, samples[0], dataset_name, repeat)
    flops_gflops = measure_flops(model, samples[0], dataset_name)

    latency_times = []
    for msg in tqdm(samples, desc=f"{method_label} t={token} {dataset_name}", leave=False):
        l_mean, _ = measure_latency(model, msg, dataset_name, repeat, max_new_tokens)
        latency_times.append(l_mean)

    result = {
        "method": method_label,
        "token": token,
        "dataset": dataset_name,
        "model_name": model_name,
        "latency_ms_mean": float(np.mean(latency_times)),
        "latency_ms_std": float(np.std(latency_times)),
        "num_samples": len(samples),
    }
    if prefill_mean is not None:
        result["prefill_ms_mean"] = prefill_mean
        result["prefill_ms_std"] = prefill_std
    if flops_gflops is not None:
        result["flops_gflops"] = round(flops_gflops, 2)
    return result


def _ensure_cuda_home():
    """Ensure CUDA_HOME points to CUDA root (not .../bin) so nvcc is at CUDA_HOME/bin/nvcc.
    Fixes 'No such file or directory: .../cuda-11.1/bin/bin/nvcc' when CUDA_HOME was set to .../bin.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        # If CUDA_HOME ends with /bin, use parent so nvcc is at CUDA_HOME/bin/nvcc
        if cuda_home.rstrip(os.sep).endswith("bin"):
            cuda_home = os.path.dirname(cuda_home.rstrip(os.sep))
            os.environ["CUDA_HOME"] = cuda_home
            if "CUDA_PATH" in os.environ:
                os.environ["CUDA_PATH"] = cuda_home
    else:
        # Default to cuda-12.1 (match run_single.sh) or /usr/local/cuda if present
        for candidate in ("/usr/local/cuda-12.1", "/usr/local/cuda"):
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
                os.environ["CUDA_HOME"] = candidate
                break


def main():
    _ensure_cuda_home()

    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL inference efficiency: Latency, Prefill time, FLOPs (VLMEvalKit datasets, run from Qwen2_5_vl root). Install fvcore for FLOPs."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="Model name(s) from config (e.g. Qwen2.5-VL-3B-CDPruner-128)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["MMBench_DEV_EN_V11", "TextVQA_VAL"],
        help="VLMEvalKit dataset names (e.g. MMBench_DEV_EN_V11, TextVQA_VAL, ChartQA_TEST). Data under LMUDataRoot (~/LMUData).",
    )
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(QWEN_ROOT, "outputs", "efficiency"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    ws_bak = os.environ.pop("WORLD_SIZE", None)

    try:
        from vlmeval.config import supported_VLM
        from vlmeval.dataset import build_dataset
    except Exception as e:
        print("Failed to import vlmeval. Run from Qwen2_5_vl root or set PYTHONPATH.", file=sys.stderr)
        raise e

    # Build dataset objects (VLMEvalKit: TSV under LMUDataRoot, same as normal eval)
    dataset_map = {}
    for name in args.datasets:
        ds = build_dataset(name)
        if ds is None:
            print(f"Skip dataset (build failed): {name}")
            continue
        dataset_map[name] = ds

    if not dataset_map:
        print("No datasets available. Ensure LMUData (~/LMUData) contains the required TSV files.", file=sys.stderr)
        if ws_bak is not None:
            os.environ["WORLD_SIZE"] = ws_bak
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    config_base = {
        "max_samples": args.max_samples,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
    }

    for model_name in args.model:
        if model_name not in supported_VLM:
            print(f"Skip unknown model: {model_name}", file=sys.stderr)
            continue
        try:
            model = supported_VLM[model_name]()
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}", file=sys.stderr)
            continue
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "model") and hasattr(model.model, "cuda"):
            model.model.cuda()
            if hasattr(model.model, "eval"):
                model.model.eval()
        elif hasattr(model, "cuda"):
            model.cuda()

        for dataset_name, dataset in dataset_map.items():
            config = {**config_base}
            res = run_one_dataset(model, model_name, dataset, dataset_name, config)
            if res is not None:
                all_results.append(res)
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
                out_path = os.path.join(
                    args.output_dir, f"efficiency_{safe_name}_{dataset_name}.json"
                )
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"Wrote {out_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if ws_bak is not None:
        os.environ["WORLD_SIZE"] = ws_bak

    summary_path = os.path.join(args.output_dir, "efficiency_all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {summary_path} ({len(all_results)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
