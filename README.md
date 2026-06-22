# Fine-tuning of Qwen3-VL+CoVFT parameters for multi-task visual question answering (FineTuneQwen3-VL)

FineTuneQwen3-VL is a local fine-tuning and evaluation project built around Qwen3-VL. It is intended for multimodal instruction tuning, CoVFT-style adaptation experiments, and benchmark comparison across different checkpoints. The project follows the LLaVA-style data format and keeps both training and evaluation scripts organized around local paths, which makes it suitable for users who already have their own models, datasets, and server environment.

In a typical workflow, you prepare local model weights, organize the training data in LLaVA format, choose one of the provided training scripts to produce a checkpoint, and then evaluate either the base model or trained checkpoints with the exported playground benchmarks.

## Project Layout

The core code lives under `src`. The training entry point is `src/train.py`; the data loader is under `src/data`; CoVFT-related modules are implemented under `src/models`; and evaluation-time model loading and generation are handled under `src/eval`. Common training and evaluation commands are placed under `scripts`. Exported evaluation data is expected under `playground/data/eval`, and training outputs are written to `checkpoints` by default.

```text
FineTuneQwen3-VL/
|-- src/
|-- scripts/
|-- gitted/
|-- playground/data/eval/
|-- checkpoints/
`-- README.md
```

## Environment and Models

The project depends on common fine-tuning components such as PyTorch, Transformers, PEFT, bitsandbytes, and datasets. A separate conda environment is recommended. The installed `transformers` package must support `Qwen3VLForConditionalGeneration`.

```bash
conda create -n FTQwen3 python=3.10
conda activate FTQwen3
pip install -r requirements.txt
```

The default local model directories are shown below. The Qwen3-VL directory is used as the main vision-language model, while the BERT directory is used by CoVFT as the context encoder.

```text
gitted/Qwen3-VL-4B-Instruct
gitted/bert-base-uncased
```

You can download the models on a machine with network access and then copy the directories to an offline server.

```bash
mkdir -p gitted

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir gitted/Qwen3-VL-4B-Instruct \
  --local-dir-use-symlinks False

huggingface-cli download google-bert/bert-base-uncased \
  --local-dir gitted/bert-base-uncased \
  --local-dir-use-symlinks False
```

## Data Organization

Training data follows the LLaVA convention. The annotation file is a JSON list, where each sample contains an image path and a conversation. Image paths are usually relative paths, and the full path is resolved by combining the sample's `image` field with the training argument `image_folder`. If your dataset can be converted to this format, it can be used directly by the existing training scripts.

```text
/path/to/llava_finetune/
|-- llava_v1_5_mix665k.json
|-- coco/
|   `-- train2017/
|-- gqa/
|-- ocr_vqa/
`-- ...
```

A single training sample looks like this:

```json
{
  "id": "sample-id",
  "image": "coco/train2017/000000000009.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe this image."
    },
    {
      "from": "gpt",
      "value": "The image shows ..."
    }
  ]
}
```

During training, `data_path` should point to the annotation file, and `image_folder` should point to the image root.

```bash
--data_path /path/to/llava_finetune/llava_v1_5_mix665k.json
--image_folder /path/to/llava_finetune
```

Evaluation data is prepared with a separate export script. The script downloads the required benchmarks from Hugging Face and exports them into a local directory that can be copied directly to the server. After export, only `playground/data/eval` is needed for evaluation; the server does not need a Hugging Face dataset cache.

```bash
python gitted/download_eval_datasets_windows.py
```

```text
playground/data/eval/
|-- mmvp/
|   |-- questions.jsonl
|   `-- images/
|-- realworldqa/
|   |-- questions.jsonl
|   `-- images/
|-- coco/
|-- ade/
|-- omni/
`-- ai2d/
```

## Training

Training scripts are located in `scripts`. Before running them, update the paths for `model_name_or_path`, `data_path`, `image_folder`, and `output_dir` according to your environment. Different tuning strategies correspond to different scripts, and outputs are saved under `checkpoints` by default.

| Training method | Command | Default output |
| --- | --- | --- |
| LoRA | `bash scripts/train_lora.sh` | `checkpoints/qwen3vl-lora` |
| QLoRA | `bash scripts/train_qlora.sh` | `checkpoints/qwen3vl-qlora` |
| Freeze vision encoder | `bash scripts/train_freeze_vision.sh` | `checkpoints/qwen3vl-freeze-vision` |
| CoVFT | `bash scripts/train_covft.sh` | `checkpoints/qwen3vl-covft` |

CoVFT uses `gitted/bert-base-uncased` as the default context encoder. You can replace it with another local text encoder through environment variables. Common CoVFT experiment settings can also be overridden before launching the script.

```bash
CONTEXT_EMBEDDING_MODEL=gitted/bert-base-uncased \
COVFT_MOE_NUM_EXPERTS=2 \
COVFT_MOE_TOP_K=2 \
COVFT_MOE_START_LAYER=-2 \
COVFT_MODEL_MAX_LENGTH=1024 \
bash scripts/train_covft.sh
```

For multi-GPU training, set `NPROC_PER_NODE` to the number of processes to launch.

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train_lora.sh
```

## Evaluation

Evaluation scripts are located in `scripts/eval`. `run_eval.sh` runs multiple benchmarks using the current configuration, while `eval_playground.sh` can evaluate a single benchmark. Before evaluation, make sure `playground/data/eval` exists and the checkpoint path matches the command you run.

| Evaluation target | Command |
| --- | --- |
| CoVFT checkpoint | `CKPT_DIR=checkpoints/qwen3vl-covft bash scripts/eval/run_eval.sh` |
| Base model | `EVAL_MODE=base bash scripts/eval/run_eval.sh` |
| LoRA checkpoint | `CKPT_DIR=checkpoints/qwen3vl-lora TUNING_STRATEGY=auto bash scripts/eval/run_eval.sh` |
| QLoRA checkpoint | `CKPT_DIR=checkpoints/qwen3vl-qlora TUNING_STRATEGY=auto bash scripts/eval/run_eval.sh` |
| Freeze-vision checkpoint | `CKPT_DIR=checkpoints/qwen3vl-freeze-vision TUNING_STRATEGY=auto bash scripts/eval/run_eval.sh` |
| Single benchmark | `CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_playground.sh mmvp checkpoints/qwen3vl-covft` |

Evaluation outputs are written back to the corresponding benchmark directory and include model answers, incorrect samples, and summary files.

```text
playground/data/eval/<benchmark>/answers/
playground/data/eval/<benchmark>/incorrect/
playground/data/eval/<benchmark>/experiments.csv
```

## Offline Server Setup

For an offline server, prepare the model directories, exported evaluation data, and the checkpoint you want to evaluate. Training additionally requires the training annotation file and image directory.

```text
gitted/Qwen3-VL-4B-Instruct
gitted/bert-base-uncased
playground/data/eval
checkpoints/<your-checkpoint>
/path/to/llava_finetune/llava_v1_5_mix665k.json
/path/to/llava_finetune/
```

In practice, the model directories, training data path, evaluation data path, and checkpoint path should match the paths configured in the scripts.

## References

- [LLaVA](https://github.com/haotian-liu/LLaVA)
