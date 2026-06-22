from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gitted/Qwen3-VL-4B-Instruct")
    trust_remote_code: bool = field(default=True)
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Use flash_attention_2 when the local environment supports it."},
    )
    tuning_strategy: str = field(
        default="lora",
        metadata={
            "help": (
                "One of: full, freeze_vision, freeze_llm, bitfit, lora, qlora, "
                "vision_lora, covft."
            )
        },
    )
    freeze_backbone: bool = field(default=False)

    # CoVFT/Qwen adaptation knobs.
    moe_num_experts: int = field(default=4)
    moe_top_k: int = field(default=4)
    moe_start_layer: int = field(default=-8)
    covft_attn_heads: int = field(default=4)
    covft_train_layernorm: bool = field(default=True)
    covft_train_llm: bool = field(
        default=False,
        metadata={"help": "Train non-vision Qwen parameters during CoVFT. This enables text-only samples but uses more memory."},
    )
    covft_llm_lora: bool = field(
        default=True,
        metadata={"help": "Train lightweight LoRA adapters on non-vision Qwen modules during CoVFT."},
    )
    covft_context_detach: bool = field(default=True)
    context_embedding_model: Optional[str] = field(
        default="gitted/bert-base-uncased",
        metadata={
            "help": (
                "Local text encoder path used by CoVFT CVE, e.g. gitted/bert-base-uncased."
            )
        },
    )
    context_embedding_max_length: int = field(default=256)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a LLaVA-style JSON file."},
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Root folder for relative image paths in the JSON file."},
    )
    image_max_pixels: Optional[int] = field(
        default=262144,
        metadata={"help": "Resize images down to this maximum pixel area before processing."},
    )
    skip_missing_images: bool = field(
        default=True,
        metadata={"help": "Skip samples whose referenced image files are missing."},
    )
    missing_image_log_limit: int = field(
        default=20,
        metadata={"help": "Maximum number of missing image paths to print when filtering the dataset."},
    )
    skip_text_only: bool = field(
        default=False,
        metadata={"help": "Skip samples without images. Useful when only visual-side modules are trainable."},
    )
    is_multimodal: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=2048)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    bits: int = field(
        default=16,
        metadata={"help": "Use 16 for normal fine-tuning, 8/4 for k-bit loading."},
    )
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")

    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated PEFT target module names."},
    )

    group_by_modality_length: bool = field(default=True)
    report_trainable_parameters: bool = field(default=True)
