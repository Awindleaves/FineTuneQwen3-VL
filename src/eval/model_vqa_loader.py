import argparse
import json
import math
import os
import string
import types
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from data.constants import DEFAULT_IMAGE_TOKEN
from models import apply_qwen_covft

try:
    from transformers import Qwen3VLForConditionalGeneration as VLModel
except ImportError as exc:
    raise ImportError(
        "Your transformers build does not expose Qwen3VLForConditionalGeneration."
    ) from exc


LOCAL_FILES_ONLY = True
VISION_KEYWORDS = ("visual", "vision")


def split_list(items: List[Any], n: int) -> List[List[Any]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: List[Any], n: int, k: int) -> List[Any]:
    return split_list(items, n)[k]


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Question file must be a JSON list or JSONL file.")
    return data


def normalize_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"Unsupported conversation role: {role}")


def question_id(sample: Dict[str, Any], fallback: int) -> str:
    for key in ("question_id", "id", "uid"):
        if key in sample:
            return str(sample[key])
    return str(fallback)


def question_text(sample: Dict[str, Any]) -> str:
    if "conversations" in sample:
        for turn in sample["conversations"]:
            if normalize_role(str(turn.get("from", ""))) == "user":
                return str(turn.get("value", "")).replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return str(sample.get("text") or sample.get("question") or sample.get("prompt") or "").strip()


def image_paths(image_field, image_folder: Optional[str]) -> List[str]:
    if image_field is None:
        return []
    image_names = image_field if isinstance(image_field, list) else [image_field]
    paths = []
    for image_name in image_names:
        image_name = str(image_name)
        paths.append(image_name if os.path.isabs(image_name) else os.path.join(image_folder or "", image_name))
    return paths


def resize_to_max_pixels(image: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    if not max_pixels or max_pixels <= 0:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def load_images(paths: List[str], max_pixels: Optional[int]) -> List[Image.Image]:
    images = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path).convert("RGB")
        images.append(resize_to_max_pixels(image, max_pixels))
    return images


def build_messages(prompt: str, num_images: int) -> List[Dict[str, Any]]:
    content = []
    content.extend({"type": "image"} for _ in range(num_images))
    if prompt:
        content.append({"type": "text", "text": prompt})
    if not content:
        content.append({"type": "text", "text": ""})
    return [{"role": "user", "content": content}]


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def compute_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def model_kwargs(args, dtype: torch.dtype) -> Dict[str, Any]:
    kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.bits in {4, 8}:
        from transformers import BitsAndBytesConfig

        kwargs.update(
            {
                "device_map": "auto",
                "load_in_4bit": args.bits == 4,
                "load_in_8bit": args.bits == 8,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=args.double_quant,
                    bnb_4bit_quant_type=args.quant_type,
                ),
            }
        )
    elif dtype != torch.float32:
        kwargs["torch_dtype"] = dtype
    return kwargs


def is_vision_name(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in VISION_KEYWORDS)


def find_linear_module_names(model, include_vision=True, full_names=False) -> List[str]:
    names = set()
    for module_name, module in model.named_modules():
        if not include_vision and is_vision_name(module_name):
            continue
        if isinstance(module, torch.nn.Linear):
            if module_name.split(".")[-1] == "lm_head":
                continue
            names.add(module_name if full_names else module_name.split(".")[-1])
    return sorted(names)


def apply_covft_llm_lora(model, args):
    from peft import LoraConfig, get_peft_model

    targets = [
        name
        for name in find_linear_module_names(model, include_vision=False, full_names=True)
        if "language_model" in name and "covft_" not in name
    ]
    if not targets:
        raise ValueError("No non-vision linear modules were found for CoVFT LLM LoRA.")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def make_covft_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_start_layer=args.moe_start_layer,
        covft_attn_heads=args.covft_attn_heads,
        covft_context_detach=True,
        covft_train_layernorm=True,
        context_embedding_model=args.context_embedding_model,
        context_embedding_max_length=args.context_embedding_max_length,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )


def same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    return os.path.abspath(left) == os.path.abspath(right)


def load_checkpoint_state(model, model_dir: str):
    if os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")) or os.path.exists(
        os.path.join(model_dir, "pytorch_model.bin.index.json")
    ):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            if shard_file.endswith(".safetensors"):
                from safetensors.torch import load_file

                shard_state = load_file(shard_path, device="cpu")
            else:
                shard_state = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_state)
        return model.load_state_dict(state_dict, strict=False)

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights under '{model_dir}'.")
    return model.load_state_dict(state_dict, strict=False)


def merge_peft_for_generation(model):
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def patch_qwen3vl_generation_inputs(model):
    if getattr(model, "_qwen3vl_generation_inputs_patched", False):
        return model
    if not hasattr(model, "prepare_inputs_for_generation"):
        return model

    original_prepare = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation(self, input_ids, *args, mm_token_type_ids=None, **kwargs):
        model_inputs = original_prepare(input_ids, *args, **kwargs)
        has_multimodal_grid = (
            model_inputs.get("image_grid_thw") is not None
            or model_inputs.get("video_grid_thw") is not None
        )
        if mm_token_type_ids is not None and has_multimodal_grid:
            input_length = model_inputs["input_ids"].shape[-1]
            if mm_token_type_ids.shape[-1] != input_length:
                mm_token_type_ids = mm_token_type_ids[:, -input_length:]
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model)
    model._qwen3vl_generation_inputs_patched = True
    return model


def load_model(args):
    dtype = compute_dtype(args.dtype)
    processor_path = args.processor_path or args.model_path
    if args.tuning_strategy == "covft":
        processor_path = args.processor_path or args.base_model_path

    processor = AutoProcessor.from_pretrained(
        processor_path,
        cache_dir=args.cache_dir,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.tuning_strategy == "covft":
        if not args.base_model_path:
            raise ValueError("--base-model-path is required for CoVFT evaluation.")
        if same_path(args.model_path, args.base_model_path):
            raise ValueError(
                "You are evaluating the base model path with --tuning-strategy covft. "
                "For no-training baseline evaluation, use --tuning-strategy auto."
            )
        model = VLModel.from_pretrained(args.base_model_path, **model_kwargs(args, dtype))
        apply_qwen_covft(model, make_covft_args(args))
        if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.model_path)
            model = merge_peft_for_generation(model)
        else:
            if args.covft_llm_lora:
                model = apply_covft_llm_lora(model, args)
            missing, unexpected = load_checkpoint_state(model, args.model_path)
            if missing:
                print(f"Warning: missing {len(missing)} keys while loading checkpoint.")
            if unexpected:
                print(f"Warning: unexpected {len(unexpected)} keys while loading checkpoint.")
    elif os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_path = args.base_model_path or peft_config.base_model_name_or_path
        model = VLModel.from_pretrained(base_path, **model_kwargs(args, dtype))
        model = PeftModel.from_pretrained(model, args.model_path)
        model = merge_peft_for_generation(model)
    else:
        model = VLModel.from_pretrained(args.model_path, **model_kwargs(args, dtype))

    model.eval()
    if args.bits not in {4, 8}:
        model.to(args.device)
    patch_qwen3vl_generation_inputs(model)
    return model, processor


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def eval_model(args):
    model, processor = load_model(args)
    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for idx, line in enumerate(questions):
            qid = question_id(line, idx)
            cur_prompt = question_text(line)
            context_prompt = cur_prompt
            model_prompt = cur_prompt
            if args.single_pred_prompt:
                model_prompt = model_prompt + "\n" + "Answer with the option's letter from the given choices directly."

            paths = image_paths(line.get("image"), args.image_folder)
            images = load_images(paths, args.image_max_pixels) if paths else []
            messages = build_messages(model_prompt, len(images))
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images if images else None, return_tensors="pt")
            inputs = inputs.to(args.device)

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature,
                "num_beams": args.num_beams,
                "use_cache": True,
            }
            if args.top_p is not None:
                generate_kwargs["top_p"] = args.top_p
            if args.tuning_strategy == "covft":
                generate_kwargs["conversations_context"] = [context_prompt]

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generate_kwargs)
            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            outputs = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            output_prompt = model_prompt
            if args.include_image_token_in_output_prompt and images:
                output_prompt = DEFAULT_IMAGE_TOKEN + "\n" + output_prompt

            ans_file.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "prompt": output_prompt,
                        "text": outputs,
                        "answer_id": str(uuid.uuid4()),
                        "model_id": os.path.basename(os.path.normpath(args.model_path)),
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, default="gitted/Qwen3-VL-4B-Instruct")
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--tuning-strategy", type=str, default="auto", choices=["auto", "covft"])
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--include-image-token-in-output-prompt", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    eval_model(parse_args())
