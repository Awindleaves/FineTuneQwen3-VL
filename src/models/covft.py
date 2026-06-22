import copy
import inspect
import os
import types
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


LOCAL_FILES_ONLY = True


def _load_local_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"Could not find model.safetensors or pytorch_model.bin under '{model_dir}'."
    )


def _load_encoder_only(model_dir: str, load_kwargs: dict) -> nn.Module:
    config = AutoConfig.from_pretrained(model_dir, **load_kwargs)
    model = AutoModel.from_config(config, trust_remote_code=load_kwargs.get("trust_remote_code", True))
    target_state = model.state_dict()
    source_state = _load_local_state_dict(model_dir)

    filtered = {}
    for key, value in source_state.items():
        candidates = [key]
        prefix = f"{model.base_model_prefix}."
        if key.startswith(prefix):
            candidates.append(key[len(prefix) :])
        candidates.extend(
            candidate.replace(".gamma", ".weight").replace(".beta", ".bias")
            for candidate in list(candidates)
        )
        for candidate in candidates:
            if candidate in target_state and target_state[candidate].shape == value.shape:
                filtered[candidate] = value
                break

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if not filtered:
        raise ValueError(f"No compatible encoder weights were found in '{model_dir}'.")
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} encoder weights while loading '{model_dir}'. "
            f"First missing keys: {sample}"
        )
    if unexpected:
        raise ValueError(
            f"Unexpected encoder keys after filtering '{model_dir}': {unexpected[:5]}"
        )
    return model


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc2(self.act(self.fc1(x)))
        return self.norm(x + residual)


class ContextVectorExtractor(nn.Module):
    """CVE: text-guided cross-attention over visual tokens and text context."""

    def __init__(self, token_dim: int, context_dim: int, attn_heads: int = 4):
        super().__init__()
        self.image_resblock = ResidualBlock(token_dim)
        self.text_resblock = ResidualBlock(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=context_dim,
            kdim=token_dim + context_dim,
            vdim=token_dim + context_dim,
            num_heads=attn_heads,
            batch_first=True,
        )

    def forward(self, image_tokens: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_resblock(image_tokens)
        text_feat = self.text_resblock(text_context)
        text_expand = text_feat.unsqueeze(1).expand(-1, image_feat.size(1), -1)
        fused = torch.cat([image_feat, text_expand], dim=-1)
        task_vec, _ = self.cross_attn(text_feat.unsqueeze(1), fused, fused)
        return task_vec.squeeze(1)


class TextContextEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 256,
        cache_dir: str | None = None,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        if not os.path.exists(model_name_or_path):
            raise FileNotFoundError(
                f"CoVFT context embedding model must be a local path, but got '{model_name_or_path}'. "
                "Pass a local text encoder directory such as gitted/bert-base-uncased."
            )
        load_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": LOCAL_FILES_ONLY,
            "trust_remote_code": trust_remote_code,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
            self.encoder = _load_encoder_only(model_name_or_path, load_kwargs)
        except OSError as exc:
            raise OSError(
                f"Could not load CoVFT context embedding model from '{model_name_or_path}'. "
                "This project only loads local model files. Pass "
                "--context_embedding_model /path/to/local/text-encoder."
            ) from exc
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.max_length = max_length

    def forward(self, texts: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        encoder_device = next(self.encoder.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(encoder_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.encoder(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        context = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return context.to(device=device, dtype=dtype)


class ContextualMoEMLP(nn.Module):
    """CoMoE wrapper for a Qwen3-VL visual MLP."""

    def __init__(
        self,
        original_mlp: nn.Module,
        token_dim: int,
        context_dim: int,
        num_experts: int,
        top_k: int,
        attn_heads: int = 4,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("moe_num_experts must be >= 1.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("moe_top_k must satisfy 1 <= moe_top_k <= moe_num_experts.")

        self.experts = nn.ModuleList(copy.deepcopy(original_mlp) for _ in range(num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.cve = ContextVectorExtractor(
            token_dim=token_dim,
            context_dim=context_dim,
            attn_heads=attn_heads,
        )
        self.gate = nn.Linear(context_dim, num_experts)
        self.current_context = None

    def set_context(self, context: torch.Tensor | None):
        self.current_context = context

    def _to_batched(self, hidden_states: torch.Tensor):
        if hidden_states.dim() == 2:
            return hidden_states.unsqueeze(0), True
        if hidden_states.dim() == 3:
            return hidden_states, False
        raise ValueError(f"Unsupported visual hidden state shape: {tuple(hidden_states.shape)}")

    def _match_context(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.size(0) == batch_size:
            return context
        pooled = context.mean(dim=0, keepdim=True)
        return pooled.expand(batch_size, -1)

    def _route(self, task_vec: torch.Tensor) -> torch.Tensor:
        logits = self.gate(task_vec)
        if self.top_k == self.num_experts:
            return F.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        sparse_probs = F.softmax(top_values, dim=-1)
        probs = torch.zeros_like(logits)
        return probs.scatter(dim=-1, index=top_indices, src=sparse_probs)

    def forward(self, hidden_states, *args, **kwargs):
        if self.current_context is None:
            return self.experts[0](hidden_states, *args, **kwargs)

        image_tokens, squeezed = self._to_batched(hidden_states)
        text_context = self._match_context(self.current_context, image_tokens.size(0))
        text_context = text_context.to(device=image_tokens.device, dtype=image_tokens.dtype)

        task_vec = self.cve(image_tokens, text_context)
        gate_probs = self._route(task_vec)

        expert_outputs = []
        for expert in self.experts:
            out = expert(hidden_states, *args, **kwargs)
            out, _ = self._to_batched(out)
            expert_outputs.append(out.unsqueeze(2))
        stacked = torch.cat(expert_outputs, dim=2)
        output = torch.sum(stacked * gate_probs[:, None, :, None], dim=2)

        if squeezed:
            return output.squeeze(0)
        return output


def _infer_token_dim(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return int(child.in_features)
    raise ValueError(f"Could not infer visual MLP token dimension from {module.__class__.__name__}.")


def _module_device_dtype(module: nn.Module):
    for param in module.parameters():
        return param.device, param.dtype
    return torch.device("cpu"), torch.float32


def _vision_mlp_modules(model) -> List[tuple[str, nn.Module, str, nn.Module]]:
    candidates = []
    for module_name, module in model.named_modules():
        lowered = module_name.lower()
        if not any(key in lowered for key in ("visual", "vision")):
            continue
        if hasattr(module, "mlp") and isinstance(module.mlp, nn.Module):
            candidates.append((module_name, module, "mlp", module.mlp))
    return candidates


def _select_layers(candidates, start_layer: int):
    if start_layer < 0:
        return candidates[start_layer:]
    return candidates[start_layer:]


def _install_forward_context(
    model,
    moe_modules: List[ContextualMoEMLP],
    detach: bool,
    context_encoder: TextContextEncoder,
):
    original_forward = model.forward
    original_signature = inspect.signature(original_forward)

    def forward_with_covft(self, *args, **kwargs):
        conversations_context = kwargs.pop("conversations_context", None)
        input_ids = kwargs.get("input_ids")
        context_was_set = False
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            if conversations_context is None:
                if self.training:
                    raise ValueError(
                        "CoVFT requires conversations_context in the batch. "
                        "Make sure include_conversations_context=True for CoVFT data loading."
                    )
            else:
                context = context_encoder(
                    conversations_context,
                    device=input_ids.device,
                    dtype=self.get_input_embeddings().weight.dtype,
                )
                context = context.detach() if detach else context
                for module in moe_modules:
                    module.set_context(context)
                context_was_set = True
        try:
            return original_forward(*args, **kwargs)
        finally:
            if context_was_set:
                for module in moe_modules:
                    module.set_context(None)

    forward_with_covft.__signature__ = original_signature
    model.forward = types.MethodType(forward_with_covft, model)


def apply_qwen_covft(model, model_args):
    """Patch Qwen3-VL visual MLPs with CoVFT-style CVE + CoMoE."""
    candidates = _vision_mlp_modules(model)
    selected = _select_layers(candidates, model_args.moe_start_layer)
    if not selected:
        raise ValueError(
            "Could not find Qwen3-VL vision MLP modules to patch. "
            "Inspect model.named_modules() and adjust the CoVFT adapter."
        )

    if model_args.moe_top_k < model_args.moe_num_experts:
        # The original CoVFT release recommends dense routing unless a balancing
        # loss is added. We support top-k routing here, but keep the warning
        # visible in saved config for reproducibility.
        model.config.covft_sparse_routing = True
    else:
        model.config.covft_sparse_routing = False

    context_model_name = getattr(model_args, "context_embedding_model", None)
    if not context_model_name:
        raise ValueError(
            "CoVFT requires --context_embedding_model to point to a local text encoder. "
            "Use gitted/bert-base-uncased to match the original CoVFT-style context path."
        )

    context_encoder = TextContextEncoder(
        model_name_or_path=context_model_name,
        max_length=model_args.context_embedding_max_length,
        cache_dir=getattr(model_args, "cache_dir", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
    )
    device, dtype = _module_device_dtype(model.get_input_embeddings())
    context_encoder.to(device=device, dtype=dtype)
    model.covft_context_encoder = context_encoder
    context_dim = int(context_encoder.encoder.config.hidden_size)

    moe_modules: List[ContextualMoEMLP] = []
    for _, parent, attr_name, original_mlp in selected:
        token_dim = _infer_token_dim(original_mlp)
        device, dtype = _module_device_dtype(original_mlp)
        wrapped = ContextualMoEMLP(
            original_mlp=original_mlp,
            token_dim=token_dim,
            context_dim=context_dim,
            num_experts=model_args.moe_num_experts,
            top_k=model_args.moe_top_k,
            attn_heads=model_args.covft_attn_heads,
        ).to(device=device, dtype=dtype)
        setattr(parent, attr_name, wrapped)
        moe_modules.append(wrapped)

    _install_forward_context(
        model,
        moe_modules=moe_modules,
        detach=model_args.covft_context_detach,
        context_encoder=context_encoder,
    )
    model.config.vfm_tuning_type = "covft"
    model.config.moe_num_experts = model_args.moe_num_experts
    model.config.moe_top_k = model_args.moe_top_k
    model.config.moe_start_layer = model_args.moe_start_layer
    model.config.covft_attn_heads = model_args.covft_attn_heads
    model.config.covft_context_detach = model_args.covft_context_detach
    model.config.covft_train_layernorm = model_args.covft_train_layernorm
    model.config.context_embedding_model = context_model_name
    model.config.context_embedding_max_length = model_args.context_embedding_max_length
    return moe_modules
