#!/usr/bin/env python3
"""
LLaVA inference efficiency evaluation: FLOPs, Latency, Prefilling Time.
Uses local datasets; supports multiple methods (via env) and token settings.
Outputs per-run JSON files and an efficiency_all_results.json summary.
"""

import argparse
import gc
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CDPRUNER_ROOT = os.path.dirname(SCRIPT_DIR)
if CDPRUNER_ROOT not in sys.path:
    sys.path.insert(0, CDPRUNER_ROOT)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

# Predefined datasets: name -> {question_file, image_folder} (paths relative to data_root)
DEFAULT_DATA_ROOT = os.path.join(CDPRUNER_ROOT, "playground", "data", "eval")
PREDEFINED_DATASETS = {
    "vizwiz_val": {
        "question_file": "vizwiz/llava_vizwiz_val.jsonl",
        "image_folder": "vizwiz/val",
    },
    "textvqa": {
        "question_file": "textvqa/llava_textvqa_val_v051_ocr.jsonl",
        "image_folder": "textvqa/train_images",
    },
    "pope": {
        "question_file": "pope/llava_pope_test.jsonl",
        "image_folder": "pope/val2014",
    },
    "sqa": {
        "question_file": "scienceqa/llava_test_CQM-I.json",
        "image_folder": "scienceqa/images/test",
    },
    "gqa": {
        "question_file": "gqa/llava_gqa_testdev_balanced.jsonl",
        "image_folder": "gqa/data/images",
    },
    "mme": {
        "question_file": "MME/llava_mme.jsonl",
        "image_folder": "MME/MME_Benchmark_release_version",
    },
    "vqav2": {
        "question_file": "vqav2/llava_vqav2_mscoco_test-dev2015.jsonl",
        "image_folder": "vqav2/test2015",
    },
}


def _current_method_from_env():
    if os.environ.get("USE_LLAVA_ARCH_CDPRUNER") == "1":
        return "cdpruner"
    if os.environ.get("USE_LLAVA_ARCH_DIVPRUNE") == "1":
        return "divprune"
    if os.environ.get("USE_LLAVA_ARCH_HIPRUNE") == "1":
        return "hiprune"
    return "eadp"


def _normalize_method_name(method_name):
    method_name = method_name.strip()
    if method_name.lower() == "ours":
        return "eadp"
    return method_name.lower()


def load_questions_and_image_folder(dataset_name, data_root, question_file_override, image_folder_override):
    if question_file_override and image_folder_override:
        q_path = os.path.expanduser(question_file_override)
        img_path = os.path.expanduser(image_folder_override)
        return q_path, img_path
    if dataset_name not in PREDEFINED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Predefined: {list(PREDEFINED_DATASETS.keys())}")
    info = PREDEFINED_DATASETS[dataset_name]
    root = os.path.expanduser(data_root)
    return (
        os.path.join(root, info["question_file"]),
        os.path.join(root, info["image_folder"]),
    )


def load_questions(question_file, max_samples):
    with open(question_file, "r") as f:
        content = f.read()
    # Support both JSONL (one JSON object per line) and single JSON array (e.g. ScienceQA llava_test_CQM-I.json)
    content_stripped = content.strip()
    if content_stripped.startswith("["):
        questions = json.loads(content)
        if not isinstance(questions, list):
            questions = [questions]
    else:
        questions = [json.loads(line) for line in content.splitlines() if line.strip()]
    if max_samples is not None and max_samples > 0:
        questions = questions[:max_samples]
    return questions


def build_inputs_for_sample(line, image_folder, tokenizer, image_processor, model_config, conv_mode):
    if "image" not in line:
        return None
    image_file = line["image"]
    conv0 = (line.get("conversations") or [{}])[0] if isinstance(line.get("conversations"), list) else {}
    qs = line.get("text") or line.get("instruction") or conv0.get("value") or ""
    if not qs:
        return None
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_path = os.path.join(image_folder, image_file)
    if not os.path.exists(image_path):
        return None
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    return {
        "input_ids": input_ids.unsqueeze(0),
        "image_tensor": image_tensor,
        "image_size": image.size,
        "question": qs,
    }


def measure_prefill_time(model, batch, device, repeat):
    input_ids = batch["input_ids"].to(device)
    image_tensor = batch["image_tensor"].to(device=device, dtype=torch.float16)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    question = batch["question"]
    image_sizes = [batch["image_size"]]
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            out = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None,
                image_tensor, image_sizes=image_sizes, texts=question
            )
            _, _, attn_mask, _, inputs_embeds, _, _ = out
            _ = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.mean(times_ms)), float(np.std(times_ms))


def measure_latency(model, batch, device, repeat, max_new_tokens, temperature):
    input_ids = batch["input_ids"].to(device)
    image_tensor = batch["image_tensor"].to(device=device, dtype=torch.float16)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    question = batch["question"]
    image_sizes = [batch["image_size"]]
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                texts=question,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.mean(times_ms)), float(np.std(times_ms))


def measure_flops(model, batch, device):
    if not HAS_FVCORE:
        return None
    try:
        input_ids = batch["input_ids"].to(device)
        image_tensor = batch["image_tensor"].to(device=device, dtype=torch.float16)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        question = batch["question"]
        image_sizes = [batch["image_size"]]

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, input_ids, images, texts, attention_mask, image_sizes):
                out = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None, images,
                    image_sizes=image_sizes, texts=texts
                )
                _, _, attn_mask, _, inputs_embeds, _, _ = out
                return self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)

        wrapper = Wrapper(model)
        flops = FlopCountAnalysis(wrapper, (input_ids, image_tensor, question, attention_mask, image_sizes))
        return flops.total() / 1e9
    except Exception as e:
        # fvcore uses torch.jit.trace; ops like scatter_(..., True) can fail under JIT
        import warnings
        warnings.warn(f"FLOPs measurement skipped (trace failed): {e}", UserWarning)
        return None


def run_one_dataset(model, tokenizer, image_processor, config, dataset_name, question_file, image_folder):
    conv_mode = "vicuna_v1"
    token = config["token"]
    method_label = config["method_label"]
    max_samples = config.get("max_samples", 200)
    warmup = config.get("warmup", 5)
    repeat = config.get("repeat", 3)
    max_new_tokens = config.get("max_new_tokens", 128)
    temperature = config.get("temperature", 0)

    questions = load_questions(question_file, max_samples)
    if not questions:
        return None

    samples = []
    for line in questions:
        inp = build_inputs_for_sample(line, image_folder, tokenizer, image_processor, model.config, conv_mode)
        if inp is not None:
            samples.append(inp)
    if not samples:
        return None

    for i in range(min(warmup, len(samples))):
        with torch.no_grad():
            b = samples[i]
            _ = model.prepare_inputs_labels_for_multimodal(
                b["input_ids"].cuda(),
                None, torch.ones_like(b["input_ids"].cuda(), dtype=torch.bool),
                None, None,
                b["image_tensor"].cuda().half().unsqueeze(0),
                image_sizes=[b["image_size"]], texts=b["question"]
            )
    torch.cuda.synchronize()
    gc.collect()

    prefill_times = []
    latency_times = []
    for b in tqdm(samples, desc=f"{method_label} t={token} {dataset_name}", leave=False):
        p_mean, _ = measure_prefill_time(model, b, model.device, repeat)
        prefill_times.append(p_mean)
        l_mean, _ = measure_latency(model, b, model.device, repeat, max_new_tokens, temperature)
        latency_times.append(l_mean)

    flops_gflops = measure_flops(model, samples[0], model.device)

    result = {
        "method": method_label,
        "token": token,
        "dataset": dataset_name,
        "prefill_ms_mean": float(np.mean(prefill_times)),
        "prefill_ms_std": float(np.std(prefill_times)),
        "latency_ms_mean": float(np.mean(latency_times)),
        "latency_ms_std": float(np.std(latency_times)),
        "num_samples": len(samples),
    }
    if flops_gflops is not None:
        result["flops_gflops"] = round(flops_gflops, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="LLaVA inference efficiency: FLOPs, Latency, Prefilling Time")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path; if unset, a public Hugging Face model ID is derived from --model-version and LLAVA_MODEL_SIZE (default 7b)")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1.5", choices=["v1.5", "v1.6"])
    parser.add_argument("--methods", nargs="+", default=["eadp", "baseline"],
                        help="Method names to run (must match current arch + baseline)")
    parser.add_argument("--tokens", nargs="+", type=int, default=[0, 32, 64, 128])
    parser.add_argument("--datasets", nargs="+", default=["vizwiz_val"],
                        help="Predefined dataset names")
    parser.add_argument("--question-file", type=str, default=None, help="Override for single custom dataset")
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CDPRUNER_ROOT, "efficiency"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    # Model path: explicit --model-path wins; else derive from model_version + LLAVA_MODEL_SIZE (default 7b)
    if args.model_path is None:
        model_size = os.environ.get("LLAVA_MODEL_SIZE", "7b").strip().lower()
        if model_size not in ("7b", "13b"):
            model_size = "7b"
        if args.model_version == "v1.5":
            args.model_path = f"liuhaotian/llava-v1.5-{model_size}"
        elif args.model_version == "v1.6":
            args.model_path = f"liuhaotian/llava-v1.6-vicuna-{model_size}"

    current_method = _current_method_from_env()
    args.methods = [_normalize_method_name(method) for method in args.methods]
    if args.question_file and args.image_folder:
        dataset_list = [("custom", args.question_file, args.image_folder)]
    else:
        dataset_list = []
        for name in args.datasets:
            qf, imf = load_questions_and_image_folder(name, args.data_root, None, None)
            dataset_list.append((name, qf, imf))

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    config_base = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "model_version": args.model_version,
        "max_samples": args.max_samples,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
    }

    for token in args.tokens:
        for method_label in args.methods:
            if method_label == "baseline":
                if token != 0:
                    continue
                run_token = 0
            else:
                if method_label != current_method:
                    continue
                run_token = token

            disable_torch_init()
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, _ = load_pretrained_model(
                args.model_path, args.model_base, model_name,
                visual_token_num=run_token,
                beta=args.beta,
                alpha=args.alpha,
            )
            if not hasattr(model, "visual_token_num"):
                model.visual_token_num = run_token
            model.eval()
            model.cuda()

            for dataset_name, question_file, image_folder in dataset_list:
                if not os.path.exists(question_file):
                    print(f"Skip {dataset_name}: {question_file} not found")
                    continue
                config = {
                    **config_base,
                    "token": run_token,
                    "method_label": method_label,
                    "dataset_name": dataset_name,
                    "question_file": question_file,
                    "image_folder": image_folder,
                }
                res = run_one_dataset(model, tokenizer, image_processor, config, dataset_name, question_file, image_folder)
                if res is not None:
                    all_results.append(res)
                    out_path = os.path.join(args.output_dir, f"efficiency_{method_label}_t{run_token}_{dataset_name}.json")
                    with open(out_path, "w") as f:
                        json.dump(res, f, indent=2)
                    print(f"Wrote {out_path}")
            del model
            gc.collect()
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "efficiency_all_results.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = []
        except Exception:
            existing_results = []
    else:
        existing_results = []

    merged = {}
    for item in existing_results + all_results:
        key = (item.get("method"), item.get("token"), item.get("dataset"))
        merged[key] = item
    all_results = list(merged.values())

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {summary_path} ({len(all_results)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
