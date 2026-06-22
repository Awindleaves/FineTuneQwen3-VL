import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODELS = [
    {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "local_dir": "gitted/Qwen3-VL-4B-Instruct",
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.py",
            "*.tiktoken",
            "*.md",
        ],
    },
    {
        "repo_id": "bert-base-uncased",
        "local_dir": "gitted/bert-base-uncased",
        "allow_patterns": [
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.txt",
            "*.md",
        ],
    },
]


def download_repo(repo_id: str, local_dir: str, allow_patterns: list[str]):
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded {repo_id} to: {local_dir}")


for model in MODELS:
    download_repo(**model)
