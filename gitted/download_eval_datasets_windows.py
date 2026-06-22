"""
Download CoVFT playground evaluation datasets into this project's playground folder.

Run from the project root on a Windows machine with network access:

    python gitted/download_eval_datasets_windows.py

Final portable eval files are exported under:

    playground/data/eval/<benchmark>/questions.jsonl
    playground/data/eval/<benchmark>/images/

The HuggingFace download cache is temporary by default and is removed after export.
After copying the project to an offline server, run:

    bash scripts/eval/run_eval.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


HF_DATASETS = [
    ("ade", "SaiCharithaAkula21/benchmark_ade_manual", "train"),
    ("ai2d", "lmms-lab/ai2d", "test"),
    ("coco", "SaiCharithaAkula21/benchmark_coco_filtered", "train"),
    ("omni", "nyu-visionx/CV-Bench", "test"),
    ("realworldqa", "lmms-lab/RealWorldQA", "test"),
]


MMVP_REPO = "MMVP/MMVP"
MMVP_IMAGE_SUBFOLDER = "MMVP Images"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_hf_cache(cache_dir: Path) -> None:
    hf_home = cache_dir.resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "ds")
    hf_home.mkdir(parents=True, exist_ok=True)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    (hf_home / "ds").mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and len(str(hf_home)) > 80:
        print(
            "\nWarning: HF cache path is still fairly long on Windows. "
            "If FileLock errors continue, rerun with --cache-dir C:\\hf_qwen_eval "
            "or move the project to a shorter path before downloading."
        )


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def jsonable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    return str(value)


def save_image(image, output_path: Path) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(output_path, quality=95)
        return
    if isinstance(image, str):
        shutil.copyfile(image, output_path)
        return
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            output_path.write_bytes(image["bytes"])
            return
        if image.get("path"):
            shutil.copyfile(image["path"], output_path)
            return
    raise TypeError(f"Unsupported image value: {type(image)!r}")


def export_dataset(benchmark: str, dataset, eval_root: Path) -> None:
    benchmark_dir = eval_root / benchmark
    image_dir = benchmark_dir / "images"
    questions_file = benchmark_dir / "questions.jsonl"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if benchmark == "omni":
        dataset = dataset.filter(lambda example: example["type"] == "3D")

    count = 0
    with questions_file.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            record = {}
            for key, value in sample.items():
                if key != "image":
                    record[key] = jsonable_value(value)

            image = sample.get("image")
            if image is not None:
                image_name = f"{idx:06d}.jpg"
                save_image(image, image_dir / image_name)
                record["image"] = f"images/{image_name}"

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  exported portable playground rows: {count} -> {questions_file}")


def download_hf_datasets(eval_root: Path) -> None:
    from datasets import load_dataset

    for benchmark, dataset_name, split in HF_DATASETS:
        print(f"\n[HF dataset] {dataset_name} split={split}")
        dataset = load_dataset(dataset_name, split=split)
        print(f"  cached rows: {len(dataset)}")
        export_dataset(benchmark, dataset, eval_root)


def give_options(input_string: str) -> list[str]:
    parts = input_string.split("(")
    return [part.split(")", 1)[1].strip() for part in parts[1:] if ")" in part]


def download_mmvp(eval_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    print(f"\n[HF dataset files] {MMVP_REPO}")
    question_path = hf_hub_download(repo_id=MMVP_REPO, filename="Questions.csv", repo_type="dataset")
    mmvp_dir = eval_root / "mmvp"
    image_dir = mmvp_dir / "images"
    mmvp_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(300):
        image_path = hf_hub_download(
            repo_id=MMVP_REPO,
            filename=f"{idx + 1}.jpg",
            subfolder=MMVP_IMAGE_SUBFOLDER,
            repo_type="dataset",
        )
        shutil.copyfile(image_path, image_dir / f"{idx + 1}.jpg")
        if (idx + 1) % 25 == 0:
            print(f"  cached images: {idx + 1}/300")

    questions_file = mmvp_dir / "questions.jsonl"
    count = 0
    with open(question_path, "r", encoding="utf-8") as src, questions_file.open("w", encoding="utf-8") as dst:
        reader = csv.reader(src)
        for row in reader:
            if not row or row[0] in {"lndex", "Index"}:
                continue
            image_id = int(row[0])
            record = {
                "question": str(row[1]),
                "image": f"images/{image_id}.jpg",
                "imageId": image_id - 1,
                "options": str(row[2]),
                "text_options": give_options(str(row[2])),
                "answer": str(row[3]),
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  exported portable playground rows: {count} -> {questions_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Do not download HuggingFace datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Temporary HuggingFace cache directory. Defaults to playground/data/eval/.hf_cache.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep the temporary HuggingFace cache after exporting playground files.",
    )
    parser.add_argument(
        "--skip-mmvp",
        action="store_true",
        help="Do not download MMVP files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    eval_root = root / "playground" / "data" / "eval"
    default_cache_dir = eval_root / ".hf_cache"
    cache_dir = args.cache_dir or default_cache_dir
    set_hf_cache(cache_dir)

    print(f"Project root: {root}")
    print(f"Eval root:    {eval_root}")
    print(f"Temp cache:   {os.environ['HF_HOME']}")

    if not args.skip_hf:
        download_hf_datasets(eval_root)
    if not args.skip_mmvp:
        download_mmvp(eval_root)

    print("\nDone.")
    print(f"Exported eval files under: {eval_root}")
    if args.cache_dir is None and not args.keep_cache and cache_dir.exists():
        print(f"Removing temporary cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    elif cache_dir.exists():
        print(f"HF cache size: {human_size(folder_size(cache_dir))}")
    print("\nCopy playground/data/eval to the offline server and run:")
    print("  bash scripts/eval/run_eval.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
