import argparse
import json
import math
import os
import random
import re
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from eval.model_vqa_loader import build_messages, load_model, resize_to_max_pixels, str2bool


QUESTION_EXTENSIONS = {
    "ade": "Answer with the option's letter from the given choices directly.",
    "ai2d": "Answer with the option's letter from the given choices directly.",
    "coco": "Answer with the option's letter from the given choices directly.",
    "mmvp": "Answer with the option's letter from the given choices directly.",
    "omni": "Answer with the option's letter from the given choices directly.",
    "realworldqa": "Answer with a single word or phrase.",
}


def split_range(length: int, chunks: int, chunk_idx: int):
    chunk_size = math.ceil(length / chunks)
    start = chunk_idx * chunk_size
    end = min(length, start + chunk_size)
    return start, end


def as_rgb(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def give_options(input_string: str) -> List[str]:
    parts = input_string.split("(")
    return [part.split(")", 1)[1].strip() for part in parts[1:] if ")" in part]


def local_benchmark_file(benchmark: str) -> str:
    return os.path.join("playground", "data", "eval", benchmark, "questions.jsonl")


def load_local_benchmark(benchmark: str) -> List[Dict[str, Any]]:
    questions_file = local_benchmark_file(benchmark)
    if not os.path.exists(questions_file):
        return []

    base_dir = os.path.dirname(questions_file)
    samples = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            image_path = sample.get("image")
            if image_path and not os.path.isabs(image_path):
                sample["image"] = os.path.join(base_dir, image_path)
            samples.append(sample)
    return samples


def load_benchmark(args) -> List[Dict[str, Any]]:
    local_samples = load_local_benchmark(args.benchmark)
    if local_samples:
        return local_samples

    raise FileNotFoundError(
        f"Could not find local playground benchmark file: {local_benchmark_file(args.benchmark)}. "
        "Run python gitted/download_eval_datasets_windows.py on a networked Windows machine, "
        "then copy playground/data/eval to the server."
    )


def prompt_for_sample(benchmark: str, sample: Dict[str, Any], question_extension: str) -> str:
    if benchmark in {"ade", "coco"}:
        return f"{sample['prompt']}\n{question_extension}"
    if benchmark == "ai2d":
        prompt = sample["question"]
        keys = ["A", "B", "C", "D"]
        for idx, option in enumerate(sample["options"]):
            prompt += f"\n{keys[idx]}. {option}"
        return f"{prompt}\n{question_extension}"
    if benchmark == "mmvp":
        prompt = sample["question"] + " Options:"
        parts = sample["options"].split("(b)")
        parts = [part.strip() for part in parts]
        parts = [part.replace("(a)", "A.").replace("(b)", "B.") for part in parts]
        if len(parts) > 1:
            parts[1] = "B. " + parts[1]
        for part in parts:
            prompt += f"\n{part}"
        return f"{prompt}\n{question_extension}"
    if benchmark == "omni":
        return f"{sample['prompt']}\n{question_extension}"
    if benchmark == "realworldqa":
        return f"{sample['question']}\n{question_extension}"
    raise ValueError(f"Unsupported playground benchmark: {benchmark}")


def context_for_sample(benchmark: str, prompt: str, question_extension: str) -> str:
    context = prompt.replace(f"\n{question_extension}", "")
    if benchmark == "omni":
        context = re.sub(r"\([A-Z]\).*", "", context)
        context = "\n".join(line.strip() for line in context.splitlines() if line.strip())
        context = context.replace("Estimate the real-world distances between objects in this image.", "")
    return context.strip()


def image_for_sample(sample: Dict[str, Any], max_pixels: int):
    image = as_rgb(sample.get("image"))
    if image is None:
        return None
    return resize_to_max_pixels(image, max_pixels)


def covft_owner(model):
    for module in model.modules():
        if hasattr(module, "covft_context_encoder"):
            return module
    return None


def covft_moe_modules(model):
    return [
        module
        for module in model.modules()
        if module.__class__.__name__ == "ContextualMoEMLP" and hasattr(module, "set_context")
    ]


def set_covft_context(model, context_prompt: str, device: torch.device):
    owner = covft_owner(model)
    moe_modules = covft_moe_modules(model)
    if owner is None or not moe_modules:
        return []

    dtype = owner.get_input_embeddings().weight.dtype
    context = owner.covft_context_encoder([context_prompt], device=device, dtype=dtype)
    for module in moe_modules:
        module.set_context(context)
    return moe_modules


def clear_covft_context(moe_modules):
    for module in moe_modules:
        module.set_context(None)


def output_record(benchmark: str, idx: int, sample: Dict[str, Any], prompt: str, answer: str, model_id: str):
    if benchmark in {"ade", "coco"}:
        return {
            "questionId": idx,
            "image": sample.get("img_name"),
            "prompt": prompt,
            "answer": answer,
            "gt_answer": sample["answer"],
            "category": sample.get("sub_task", ""),
            "options": sample.get("choices", []),
            "model_id": model_id,
        }
    if benchmark == "ai2d":
        gt_answer = str(sample["answer"])
        options = sample.get("options", [])
        text_answer = options[int(gt_answer)] if gt_answer.isdigit() and int(gt_answer) < len(options) else ""
        return {
            "question_id": idx,
            "prompt": prompt,
            "answer": answer,
            "gt_answer": gt_answer,
            "text_answer": text_answer,
            "model_id": model_id,
        }
    if benchmark == "mmvp":
        return {
            "question_id": idx,
            "prompt": prompt,
            "answer": answer,
            "gt_answer": sample["answer"],
            "model_id": model_id,
            "text_options": sample.get("text_options", []),
        }
    if benchmark == "omni":
        return {
            "questionId": idx,
            "prompt": prompt,
            "answer": answer,
            "gt_answer": sample["answer"],
            "category": sample.get("sub_task", sample.get("type", "")),
            "options": sample.get("choices", []),
            "model_id": model_id,
        }
    if benchmark == "realworldqa":
        return {
            "question_id": idx,
            "prompt": prompt,
            "answer": answer,
            "gt_answer": sample["answer"],
            "model_id": model_id,
        }
    raise ValueError(f"Unsupported playground benchmark: {benchmark}")


def generate_answer(model, processor, args, prompt: str, image, context_prompt: str):
    images = [image] if image is not None else []
    messages = build_messages(prompt, len(images))
    chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_prompt], images=images if images else None, return_tensors="pt")
    inputs = inputs.to(args.device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "num_beams": args.num_beams,
        "use_cache": True,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generate_kwargs["top_p"] = args.top_p

    moe_modules = []
    try:
        if args.tuning_strategy == "covft":
            moe_modules = set_covft_context(model, context_prompt, inputs["input_ids"].device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)
    finally:
        clear_covft_context(moe_modules)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, processor = load_model(args)
    samples = load_benchmark(args)
    start, end = split_range(len(samples), args.num_chunks, args.chunk_idx)
    selected = list(enumerate(samples))[start:end]

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file.")
    basename = os.path.splitext(os.path.basename(answers_file))[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_file = os.path.join(answers_dir, f"{basename}_{args.chunk_idx}.jsonl")
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    model_id = os.path.basename(os.path.normpath(args.model_path))
    question_extension = args.question_extension or QUESTION_EXTENSIONS[args.benchmark]
    with open(chunk_file, "w", encoding="utf-8") as f:
        for idx, sample in tqdm(selected, total=len(selected)):
            prompt = prompt_for_sample(args.benchmark, sample, question_extension)
            context_prompt = context_for_sample(args.benchmark, prompt, question_extension)
            answer = generate_answer(
                model,
                processor,
                args,
                prompt,
                image_for_sample(sample, args.image_max_pixels),
                context_prompt,
            )
            record = output_record(args.benchmark, idx, sample, prompt, answer, model_id)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(QUESTION_EXTENSIONS))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, default="gitted/Qwen3-VL-4B-Instruct")
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--tuning-strategy", type=str, default="auto", choices=["auto", "covft"])
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--double-quant", type=str2bool, default=True)
    parser.add_argument("--quant-type", type=str, default="nf4")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--context-embedding-model", type=str, default="gitted/bert-base-uncased")
    parser.add_argument("--context-embedding-max-length", type=int, default=256)
    parser.add_argument("--moe-num-experts", type=int, default=2)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-start-layer", type=int, default=-2)
    parser.add_argument("--covft-attn-heads", type=int, default=4)
    parser.add_argument("--covft-llm-lora", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--question-extension", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    eval_model(parse_args())
