import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

ImageFile.LOAD_TRUNCATED_IMAGES = True
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def _normalize_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"Unsupported conversation role: {role}")


def _clean_user_text(value: str) -> str:
    return value.replace(DEFAULT_IMAGE_TOKEN, "").strip()


def _image_paths(image_field, image_folder: str | None) -> List[str]:
    if image_field is None:
        return []
    image_names = image_field if isinstance(image_field, list) else [image_field]
    paths = []
    for image_name in image_names:
        image_name = str(image_name)
        path = image_name if os.path.isabs(image_name) else os.path.join(image_folder or "", image_name)
        paths.append(path)
    return paths


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _missing_paths(paths: Sequence[str]) -> List[str]:
    return [path for path in paths if not os.path.exists(path)]


def _resize_to_max_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    if not max_pixels or max_pixels <= 0:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / pixels)
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return image.resize(new_size, RESAMPLE_BICUBIC)


def _load_images(image_paths: Sequence[str], max_pixels: int | None = None) -> List[Image.Image]:
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path).convert("RGB")
        images.append(_resize_to_max_pixels(image, max_pixels))
    return images


def llava_to_qwen_messages(conversations: Sequence[Dict], num_images: int) -> List[Dict]:
    """Convert LLaVA-style turns into Qwen3-VL chat-template messages."""
    messages: List[Dict] = []
    remaining_images = num_images

    for turn in conversations:
        role = _normalize_role(turn["from"])
        value = str(turn.get("value", ""))

        if role == "user":
            content = []
            if remaining_images > 0:
                content.extend({"type": "image"} for _ in range(remaining_images))
                remaining_images = 0
            text = _clean_user_text(value)
            if text:
                content.append({"type": "text", "text": text})
            if not content:
                content.append({"type": "text", "text": ""})
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": value.strip()})
        else:
            messages.append({"role": "system", "content": value.strip()})

    return messages


def context_from_messages(messages: Sequence[Dict]) -> str:
    chunks = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            chunks.append(content)
            continue
        for item in content:
            if item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip())


class LlavaLazySupervisedDataset(Dataset):
    """Lazy loader for LLaVA-format multimodal instruction data."""

    def __init__(self, data_path: str, data_args):
        super().__init__()
        if data_path is None:
            raise ValueError("data_path is required.")
        with open(data_path, "r", encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.data_args = data_args
        if getattr(data_args, "skip_text_only", False):
            self._filter_text_only()
        if getattr(data_args, "skip_missing_images", True):
            self._filter_missing_images()

    def _filter_text_only(self):
        original_len = len(self.list_data_dict)
        self.list_data_dict = [sample for sample in self.list_data_dict if sample.get("image")]
        skipped = original_len - len(self.list_data_dict)
        if skipped:
            if not self.list_data_dict:
                raise ValueError("All samples were filtered because they do not contain images.")
            if _is_rank0():
                print(f"Skipped {skipped} text-only samples.")

    def _filter_missing_images(self):
        kept = []
        missing_examples = []
        missing_samples = 0

        for sample in self.list_data_dict:
            paths = _image_paths(sample.get("image"), self.data_args.image_folder)
            missing = _missing_paths(paths)
            if missing:
                missing_samples += 1
                missing_examples.extend(missing)
                continue
            kept.append(sample)

        if missing_samples:
            if not kept:
                raise ValueError("All samples were filtered because their image files are missing.")
            self.list_data_dict = kept
            if _is_rank0():
                limit = max(0, int(getattr(self.data_args, "missing_image_log_limit", 20)))
                print(f"Skipped {missing_samples} samples with missing images.")
                for path in missing_examples[:limit]:
                    print(f"  missing image: {path}")
                remaining = len(missing_examples) - limit
                if remaining > 0:
                    print(f"  ... and {remaining} more missing image paths.")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [
            sum(len(turn.get("value", "").split()) for turn in sample["conversations"])
            + (128 if sample.get("image") else 0)
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        lengths = []
        for sample in self.list_data_dict:
            cur_len = sum(len(turn.get("value", "").split()) for turn in sample["conversations"])
            lengths.append(cur_len if sample.get("image") else -cur_len)
        return lengths

    def __getitem__(self, index: int) -> Dict:
        sample = copy.deepcopy(self.list_data_dict[index])
        image_paths = _image_paths(sample.get("image"), self.data_args.image_folder)
        images = _load_images(image_paths, max_pixels=self.data_args.image_max_pixels)
        messages = llava_to_qwen_messages(sample["conversations"], num_images=len(images))
        return {
            "id": sample.get("id", str(index)),
            "messages": messages,
            "images": images,
            "conversations_context": context_from_messages(messages),
        }


@dataclass
class Qwen3VLDataCollator:
    processor: object
    model_max_length: int
    include_conversations_context: bool = False
    ignore_index: int = IGNORE_INDEX

    def _token_len(self, text: str) -> int:
        tokenizer = self.processor.tokenizer
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    def _assistant_spans(self, messages: Sequence[Dict]) -> List[tuple[int, int]]:
        spans = []
        for idx, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix = self.processor.apply_chat_template(
                messages[:idx],
                tokenize=False,
                add_generation_prompt=True,
            )
            upto = self.processor.apply_chat_template(
                messages[: idx + 1],
                tokenize=False,
                add_generation_prompt=False,
            )
            spans.append((self._token_len(prefix), self._token_len(upto)))
        return spans

    def _subsequence_index(self, sequence: List[int], pattern: List[int], start: int = 0) -> int:
        if not pattern:
            return -1
        last = len(sequence) - len(pattern)
        for idx in range(start, last + 1):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return -1

    def _assistant_marker_ids(self) -> List[int]:
        tokenizer = self.processor.tokenizer
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if im_start_id is None or im_start_id < 0:
            return []
        role_ids = tokenizer("assistant\n", add_special_tokens=False).input_ids
        return [im_start_id] + role_ids

    def _im_end_id(self) -> int | None:
        tokenizer = self.processor.tokenizer
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is None or im_end_id < 0:
            return None
        return im_end_id

    def _mark_assistant_labels_from_input_ids(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> bool:
        marker_ids = self._assistant_marker_ids()
        im_end_id = self._im_end_id()
        if not marker_ids or im_end_id is None:
            return False

        ids = input_ids.tolist()
        cursor = 0
        found = False
        while cursor < len(ids):
            marker_start = self._subsequence_index(ids, marker_ids, cursor)
            if marker_start < 0:
                break

            content_start = marker_start + len(marker_ids)
            content_end = ids.index(im_end_id, content_start) if im_end_id in ids[content_start:] else len(ids)
            label_end = min(content_end + 1, len(ids))
            if label_end > content_start:
                labels[content_start:label_end] = input_ids[content_start:label_end]
                found = True
            cursor = content_end + 1
        return found

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                instance["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for instance in instances
        ]
        flat_images = [image for instance in instances for image in instance["images"]]
        batch = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )

        labels = torch.full_like(batch["input_ids"], self.ignore_index)
        for row, instance in enumerate(instances):
            marked = self._mark_assistant_labels_from_input_ids(labels[row], batch["input_ids"][row])
            if not marked:
                for start, end in self._assistant_spans(instance["messages"]):
                    start = min(start, labels.size(1))
                    end = min(end, labels.size(1))
                    if end > start:
                        labels[row, start:end] = batch["input_ids"][row, start:end]

        labels[batch["attention_mask"] == 0] = self.ignore_index
        batch["labels"] = labels
        if self.include_conversations_context:
            batch["conversations_context"] = [instance["conversations_context"] for instance in instances]
        return batch


def make_supervised_data_module(processor, data_args, training_args, include_conversations_context=False) -> Dict:
    train_dataset = LlavaLazySupervisedDataset(
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = Qwen3VLDataCollator(
        processor=processor,
        model_max_length=training_args.model_max_length,
        include_conversations_context=include_conversations_context,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
