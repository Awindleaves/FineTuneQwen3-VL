import os
from typing import Iterable, Optional

import torch
import transformers
from transformers import AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from config import DataArguments, ModelArguments, TrainingArguments
from data.dataset import make_supervised_data_module
from models import apply_qwen_covft
from trainer import QwenVLTrainer

try:
    from transformers import Qwen3VLForConditionalGeneration as VLModel
except ImportError as exc:
    raise ImportError(
        "Your transformers build does not expose Qwen3VLForConditionalGeneration. "
        "Install a Qwen3-VL capable transformers version in the local environment."
    ) from exc


VISION_KEYWORDS = ("visual", "vision")
LOCAL_FILES_ONLY = True
COVFT_STRATEGY = "covft"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _local_rank(training_args) -> int:
    return _env_int("LOCAL_RANK", training_args.local_rank)


def rank0_print(training_args, *args):
    if _local_rank(training_args) in {-1, 0}:
        print(*args)


def _compute_dtype(training_args):
    if training_args.fp16:
        return torch.float16
    if training_args.bf16:
        return torch.bfloat16
    return torch.float32


def _align_model_special_tokens(model, tokenizer):
    for config in (model.config, getattr(model, "generation_config", None)):
        if config is None:
            continue
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.bos_token_id = tokenizer.bos_token_id


def _is_vision_name(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in VISION_KEYWORDS)


def _set_trainable(params: Iterable[tuple[str, torch.nn.Parameter]], predicate):
    for name, param in params:
        param.requires_grad = bool(predicate(name, param))


def _split_modules(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_parameter_device(module, fallback_device):
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return fallback_device


def _ensure_visual_dtype_anchor(model, dtype, device):
    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is None:
        return
    if any(param.is_floating_point() for param in visual.parameters()):
        return
    if "_dtype_anchor" in dict(visual.named_parameters(recurse=False)):
        return

    anchor_device = _first_parameter_device(visual, device)
    visual.register_parameter(
        "_dtype_anchor",
        torch.nn.Parameter(torch.zeros(1, device=anchor_device, dtype=dtype), requires_grad=False),
    )


def _guard_data_parallel(training_args):
    if _local_rank(training_args) != -1 or training_args.n_gpu <= 1:
        return
    raise RuntimeError(
        "Multiple GPUs are visible, but the script was launched without torchrun/accelerate, "
        "so PyTorch would use DataParallel. Launch with NPROC_PER_NODE=<num_gpus> bash "
        "scripts/train_*.sh, or expose only one GPU with CUDA_VISIBLE_DEVICES=<gpu_id>."
    )


def _kbit_device_map(training_args):
    local_rank = _local_rank(training_args)
    if local_rank != -1:
        return {"": local_rank}
    return {"": training_args.device}


def _require_local_path(path: Optional[str], name: str):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} must be a local path in this offline training project, but got '{path}'. "
            "Place the model files on the server and pass that directory path."
        )


def _is_covft_strategy(model_args) -> bool:
    return model_args.tuning_strategy.lower() == COVFT_STRATEGY


def _disable_incompatible_gradient_checkpointing(model_args, training_args):
    if not _is_covft_strategy(model_args) or not training_args.gradient_checkpointing:
        return
    rank0_print(
        training_args,
        "CoVFT uses context-conditioned visual modules and is incompatible with "
        "gradient checkpointing in this implementation. Disabling gradient checkpointing.",
    )
    training_args.gradient_checkpointing = False


def _enable_covft_unused_parameter_detection(model_args, data_args, training_args):
    if not _is_covft_strategy(model_args):
        return
    if data_args.skip_text_only:
        return
    if training_args.ddp_find_unused_parameters is True:
        return
    rank0_print(
        training_args,
        "CoVFT keeps visual-side trainable modules, but text-only samples do not use them. "
        "Setting ddp_find_unused_parameters=True for distributed training.",
    )
    training_args.ddp_find_unused_parameters = True


def find_linear_module_names(model, include_vision=True, full_names=False):
    names = set()
    for module_name, module in model.named_modules():
        if not include_vision and _is_vision_name(module_name):
            continue
        if isinstance(module, torch.nn.Linear):
            if module_name.split(".")[-1] == "lm_head":
                continue
            names.add(module_name if full_names else module_name.split(".")[-1])
    return sorted(names)


def find_all_linear_leaf_names(model, include_vision=True):
    return find_linear_module_names(model, include_vision=include_vision, full_names=False)


def apply_lora(model, training_args, include_vision=True, target_modules=None):
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = _split_modules(training_args.lora_target_modules)
    if target_modules == ["auto"]:
        target_modules = find_all_linear_leaf_names(model, include_vision=include_vision)

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def apply_tuning_strategy(model, model_args, training_args):
    strategy = model_args.tuning_strategy.lower()

    if strategy in {"qlora", "lora"} or training_args.lora_enable:
        model = apply_lora(model, training_args, include_vision=True)
        return model

    if strategy == "vision_lora":
        model = apply_lora(model, training_args, include_vision=True)
        _set_trainable(
            model.named_parameters(),
            lambda name, _: "lora_" in name and _is_vision_name(name),
        )
        return model

    if strategy == COVFT_STRATEGY:
        covft_modules = apply_qwen_covft(model, model_args)
        covft_ids = {id(param) for module in covft_modules for param in module.parameters()}
        llm_lora_ids = set()

        if model_args.covft_llm_lora:
            llm_lora_targets = [
                name
                for name in find_linear_module_names(model, include_vision=False, full_names=True)
                if "language_model" in name and "covft_" not in name
            ]
            if not llm_lora_targets:
                raise ValueError("No non-vision linear modules were found for CoVFT LLM LoRA.")
            model = apply_lora(
                model,
                training_args,
                include_vision=False,
                target_modules=llm_lora_targets,
            )
            llm_lora_ids = {
                id(param)
                for name, param in model.named_parameters()
                if "lora_" in name and not _is_vision_name(name)
            }

        def covft_predicate(name, param):
            if "covft_context_encoder" in name:
                return False
            if id(param) in covft_ids:
                return True
            if id(param) in llm_lora_ids:
                return True
            if _is_vision_name(name):
                return model_args.covft_train_layernorm and "norm" in name.lower()
            return model_args.covft_train_llm

        _set_trainable(model.named_parameters(), covft_predicate)
        return model

    if strategy == "full":
        model.requires_grad_(True)
    elif strategy == "freeze_vision":
        _set_trainable(model.named_parameters(), lambda name, _: not _is_vision_name(name))
    elif strategy == "freeze_llm":
        _set_trainable(model.named_parameters(), lambda name, _: _is_vision_name(name))
    elif strategy == "bitfit":
        _set_trainable(model.named_parameters(), lambda name, _: name.endswith(".bias"))
    else:
        raise ValueError(f"Unsupported tuning_strategy: {model_args.tuning_strategy}")

    return model


def report_trainable_parameters(model, training_args):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / max(total, 1)
    rank0_print(training_args, f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def last_checkpoint(output_dir: str) -> Optional[str]:
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = _compute_dtype(training_args)
    if model_args.tuning_strategy.lower() == "qlora" and training_args.bits == 16:
        training_args.bits = 4
        
    _guard_data_parallel(training_args)
    _require_local_path(model_args.model_name_or_path, "model_name_or_path")

    if _is_covft_strategy(model_args):
        _require_local_path(model_args.context_embedding_model, "context_embedding_model")
    _disable_incompatible_gradient_checkpointing(model_args, training_args)
    _enable_covft_unused_parameter_detection(model_args, data_args, training_args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "local_files_only": LOCAL_FILES_ONLY,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    if training_args.bits in {4, 8}:
        from transformers import BitsAndBytesConfig

        model_kwargs.update(
            {
                "device_map": _kbit_device_map(training_args),
                "load_in_4bit": training_args.bits == 4,
                "load_in_8bit": training_args.bits == 8,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            }
        )
    else:
        if compute_dtype != torch.float32:
            model_kwargs["torch_dtype"] = compute_dtype

    model = VLModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    _align_model_special_tokens(model, tokenizer)
    _ensure_visual_dtype_anchor(model, compute_dtype, training_args.device)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    if training_args.bits in {4, 8}:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = apply_tuning_strategy(model, model_args, training_args)

    if training_args.report_trainable_parameters:
        report_trainable_parameters(model, training_args)

    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        training_args=training_args,
        include_conversations_context=_is_covft_strategy(model_args),
    )

    trainer = QwenVLTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        **data_module,
    )

    checkpoint = last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
