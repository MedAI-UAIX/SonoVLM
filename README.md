# SonoCopilot: A Generalist Ultrasound Vision-Language Model  


A multimodal AI system for ultrasound analysis with capabilities in cross-organ understanding, abnormality detection, diagnostic reasoning, structured reporting, and patient-centric dialogue.

---

## 🔥 Latest News  
- **2025/05/24**: 🎉 Official repository launched!  
- **2025/06/10**: Code is now available
> 📌 **Code is now publicly available** 

---

## 🚀 Getting Started
## 📦 Dependencies

### System Requirements

| Component | Minimum Specification | Recommended Specification |
|-----------|----------------------|---------------------------|
| **OS** | Ubuntu 20.04 LTS | Ubuntu 20.04/22.04 LTS |
| **GPU** | NVIDIA GPU with 24GB VRAM | NVIDIA A100 (40GB/80GB) or RTX 4090 (24GB) |
| **CPU** | 8 cores | 16+ cores (Intel Xeon / AMD EPYC) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 50GB free space (SSD) | 200GB+ NVMe SSD |
| **CUDA** | 11.8 | 12.1+ |
### Installation

```bash
# Clone the repository
git clone https://github.com/MedAI-UAIX/SonoCopilot.git
cd SonoCopilot

# Install dependencies
conda create -n SonoCopilot python=3.12
conda activate SonoCopilot
pip install -r requirements.txt
```

## Model Download and vLLM Deployment

### Download Model Weights

The SonoCopilot model weights are open-sourced on Hugging Face. You can download them as follows:

```bash
# Install huggingface-hub (if not already installed)
pip install huggingface-hub

# Download the full model weights
huggingface-cli download MedAIusai/SonoCopilot --local-dir /path/to/SonoCopilot_checkpoint
```
## Deploy with vLLM

We recommend using vLLM for efficient inference. Below is a complete deployment command example:

```bash
CUDA_VISIBLE_DEVICES=2 setsid vllm serve /path/to/SonoCopilot_checkpoint \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 20000 \
    --port 7000 \
    --limit-mm-per-prompt '{"image": 50}' \
    > /path/to/deploy.log 2>&1 &
```
## Verify Deployment
After the service starts, you can verify it is running correctly with:
Check service health
```bash
curl http://localhost:7000/health
tail -f /path/to/deploy.log
```
## 🔥 Demo Run
After deployment, you can run the evaluation demo with:
```bash
cd demo
python swift_eval_v2.py \
    --model_path /path/to/SonoCopilot_checkpoint \
    --benchmark_file caption_generation_benchmark.json \
    --output_dir ./results
```


# Perform finetuning
We use [SWIFT](https://github.com/modelscope/ms-swift) (Scalable lightWeight Infrastructure for Fine-Tuning) for efficient multimodal model training. For complete training arguments and advanced configurations, please refer to the [SWIFT Pre-training and Fine-tuning Documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/Pre-training-and-Fine-tuning.html).

### Prepare your finetuning data
We use the SWIFT framework for training. For complete data format specifications, please refer to the [SWIFT Custom Dataset Documentation](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html).
Like LLaVA, we anticipate that the data will reside within a JSON file, composed of a collection of dictionaries. In this structure, each individual dictionary corresponds to a distinct sample.
```json
   [
    {
        "id": "215168",
        "system_prompt": "You are a helpful assistant.",//Optional
        "image": [
            "215168_1.jpeg"
        ],
        "description": "The bladder is slightly full, the bladder wall is continuous and intact, and no obvious abnormal echoes are seen inside.",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},//Optional
            {
                "role": "user",
                "content": "<image>Based on the ultrasound image, could you briefly describe what's in the image?"
            },
            {
                "role": "assistant",
                "value": "The bladder is slightly full, the bladder wall is continuous and intact, and no obvious abnormal echoes are seen inside."
            },
        ]
    }
]
```
| Field           | Type        | Required | Description                                                              |
| --------------- | ----------- | -------- | ------------------------------------------------------------------------ |
| `id`            | string      | Yes      | Unique sample identifier                                                 |
| `image`         | string/list | Yes      | Path(s) to image file(s). Use list for multi-image samples               |
| `messages`      | list        | Yes      | Conversation history following OpenAI format                             |
| `system_prompt` | string      | Optional | System-level instruction (alternative to `messages` with `role: system`) |
| `description`   | string      | Optional | Additional text description (optional metadata)                          |


Stage 1: Aligner Fine-tuning

This stage freezes the ViT and LLM components, only fine-tuning the alignment module to establish basic cross-modal mapping.

```bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model lingshu-medical-mllm/Lingshu-7B \
    --model_type qwen2_5_vl \
    --tuner_type full \
    --dataset xxx  \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 128 \
    --deepspeed zero2
    --padding_free True \
    --packing True
```

Stage 2: LoRA Fine-tuning (Full Model Adaptation)

This stage uses LoRA to fine-tune the entire model  to optimize cross-modal understanding capabilities.

```bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/stage1_checkpoint \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --tuner_type lora\
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset xxx \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 32768\
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 128 \
    --deepspeed zero3 \
    --use_dora True \
    --padding_free True \
    --packing True

```


Inference (CLI)

Run inference with the trained model using CLI, supporting batch inference on validation datasets:

```bash

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 4096 \
    --val_dataset <dataset-path> \
    --max_batch_size 1
```
Deployment (vLLM Acceleration)

Deploy the model as a service with vLLM for high-throughput inference, supporting multi-GPU tensor parallelism:

```bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FPS_MAX_FRAMES=768\
swift deploy \
    --model /path/to/stage2_checkpoint \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 32768 \
    --vllm_limit_mm_per_prompt '{"image": 50, "video": 1}' \
    --tensor-parallel-size 8 \
    --port 8008

```
