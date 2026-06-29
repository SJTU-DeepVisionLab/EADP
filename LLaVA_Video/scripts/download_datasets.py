#!/usr/bin/env python
"""Download public video benchmark datasets used by the LLaVA-Video experiments.

Examples:
    HF_HOME=$HOME/.cache/huggingface python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets mvbench videomme
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List

from huggingface_hub import snapshot_download


DATASETS: Dict[str, Dict[str, str]] = {
    "mvbench": {
        "name": "MVBench",
        "repo_id": "OpenGVLab/MVBench",
        "repo_type": "dataset",
        "revision": "video",
        "local_subdir": "mvbench_video",
    },
    "longvideobench": {
        "name": "LongVideoBench",
        "repo_id": "longvideobench/LongVideoBench",
        "repo_type": "dataset",
        "local_subdir": "longvideobench",
    },
    "videomme": {
        "name": "Video-MME",
        "repo_id": "lmms-lab/Video-MME",
        "repo_type": "dataset",
        "local_subdir": "videomme",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="Dataset download root. Defaults to HF_HOME or ~/.cache/huggingface.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=sorted(DATASETS.keys()),
        help="Datasets to download.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume partially downloaded files when supported by huggingface_hub.",
    )
    return parser.parse_args()


def download_dataset(key: str, hf_home: str, resume_download: bool) -> bool:
    info = DATASETS[key]
    local_dir = os.path.join(hf_home, info["local_subdir"])
    os.makedirs(local_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"Downloading {info['name']}")
    print(f"  repo:      {info['repo_id']}")
    print(f"  local_dir: {local_dir}")
    if "revision" in info:
        print(f"  revision:  {info['revision']}")
    print("=" * 72)

    kwargs = {
        "repo_id": info["repo_id"],
        "repo_type": info.get("repo_type", "dataset"),
        "local_dir": local_dir,
        "local_dir_use_symlinks": False,
        "resume_download": resume_download,
    }
    if "revision" in info:
        kwargs["revision"] = info["revision"]

    try:
        snapshot_download(**kwargs)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        print(f"[ERROR] {info['name']} failed: {exc}")
        return False

    print(f"[OK] {info['name']} downloaded.")
    return True


def main() -> None:
    args = parse_args()
    hf_home = os.path.abspath(os.path.expanduser(args.hf_home))
    os.makedirs(hf_home, exist_ok=True)

    print(f"HF_HOME = {hf_home}")
    results = {
        key: download_dataset(key, hf_home, args.resume_download)
        for key in args.datasets
    }

    print("\n" + "=" * 72)
    print("Download summary")
    print("=" * 72)
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {DATASETS[key]['name']:16s} {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
