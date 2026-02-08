# SonoVLM: A Generalist Ultrasound Vision-Language Model  


A multimodal AI system for ultrasound analysis with capabilities in cross-organ understanding, abnormality detection, diagnostic reasoning, structured reporting, and patient-centric dialogue.

---

## 🔥 Latest News  
- **2025/05/24**: 🎉 Official repository launched!  
- **2025/06/10**: Code is now available
> 📌 **Code is now publicly available** 

---
## 🔥 Demo  
<video controls>
  <source src="VLM演示.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
## 📈 Technical Architecture  
pass

## 🚀 Getting Started  
**Note**: Code will be released publicly after publication of our paper.  
<summary><b>1. Installation (Coming Soon) </b></summary>
pass

<summary><b>2. Prepare your finetuning data</b></summary>

Like LLaVA, we anticipate that the data will reside within a JSON file, composed of a collection of dictionaries. In this structure, each individual dictionary corresponds to a distinct sample.
```json
   [
    {
        "id": "215168",
        "system_prompt": "You are a helpful assistant.",//Optional
        "image": [
            "215168_1.jpeg",
            "215168_2.jpeg"
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

<summary><b>3. Perform finetuning</b></summary>

Stage 1: Aligner Fine-tuning

This stage freezes the ViT and LLM components, only fine-tuning the alignment module to establish basic cross-modal mapping.

```bash
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MAX_PIXELS=153664 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/Lingshu-7B \
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
MAX_PIXELS=153664 \
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

Stage 3: Modeling of human-in-the-loop clinical reasoning

```bash

NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MAX_PIXELS=153664 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \

swift sft \
 --model /path/to/stage2_checkpoint \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --tuner_type lora\
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset 'teacher_generated_data.jsonl'\
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

4. Inference (CLI)

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

5. Deployment (vLLM Acceleration)

Deploy the model as a service with vLLM for high-throughput inference, supporting multi-GPU tensor parallelism:

```bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=153664 \
VIDEO_MAX_PIXELS=153664 \
FPS_MAX_FRAMES=768\
swift deploy \
    --model /path/to/stage3_checkpoint \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 32768 \
    --vllm_limit_mm_per_prompt '{"image": 50, "video": 1}' \
    --tensor-parallel-size 8 \
    --port 8008

```
