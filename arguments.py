from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field

import transformers
from transformers import SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy


# from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    model_id: str = field(default="Ultrasound-MOE")
    #["Qwen2-VL-MOE","Qwen2-VL",'LLaVA-OneVision'，'LLaVA-Med','LLaVA','Ultrasound-MOE']
    model_local_path: Optional[str] = field(default='/home/user02/SCY/Model/Qwen2-VL-7B-Instruct')
    auto_processor_local_path: Optional[str] = field(default='/home/user02/SCY/Model/Qwen2-VL-7B-Instruct')
    #choices = ['stage1', 'stage2', 'stage3','Comparative_experiment','ablation']
    training_stage: Optional[str] = field(default='stage1')


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(
        default_factory=lambda: [
            '/home/user02/SCY/public_data/PubMedVision/PubMedVision_Alignment_VQA.json',
            '/home/user02/SCY/public_data/PubMedVision/PubMedVision_Chinese.json',
            '/home/user02/SCY/public_data/PubMedVision/PubMedVision_InstructionTuning_VQA.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/breast_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/gynaecology_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/heart_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/kidney_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/liver_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/thyroid_alignment.json',
            '/home/user02/SCY/VLM/my_vlm/数据集/alignment/vessel_alignment.json'
            #                     '/home/user02/SCY/VLM/my_vlm/json_2025/breast_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/gynaecology_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/heart_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/kidney_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/liver_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/thyroid_train_clear_2025.json',
            #                      '/home/user02/SCY/VLM/my_vlm/json_2025/vessel_train_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/breast_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/gynaecology_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/heart_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/kidney_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/liver_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/thyroid_val_clear_2025.json',
            # '/home/user02/SCY/VLM/my_vlm/json_2025/vessel_val_clear_2025.json',
                                 # '/home/user02/SCY/VLM/my_vlm/json_2025/llava-med-zh-instruct-60k.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_breast_train.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_breast_val.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_breast_test.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_liver_train.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_liver_val.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_liver_test.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_thynet_train.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_thynet_val.json',
                                 # '/home/user02/LRF/VLM/json/train/multicenter_thynet_test.json'
                                 ],
        metadata={"help": "Path to the training json."})
    #/data/scy/SCY/my_vlm/dataset/public_breast_val.json
    eval_data_path: Optional[List[str]] = field(
        default_factory=lambda: ['/home/user02/SCY/VLM/my_vlm/json_2025/breast_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/gynaecology_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/heart_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/kidney_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/liver_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/thyroid_val_clear_2025.json',
                                 '/home/user02/SCY/VLM/my_vlm/json_2025/vessel_val_clear_2025.json',
                                 ],
        metadata={"help": "Path to the evaluation data json file."})

    image_folder: Optional[str] = field(default='/home/user02/LRF/VLM/Data')
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataloader_num_workers: int = field(
        default=8,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    output_dir: str = field(default='/home/user02/SCY/VLM/my_vlm/checkpoints/Ultrasound-MOE',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: Union[None, str, List[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    tf32: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )

    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    learning_rate: float = field(default=1e-3, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})


    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    save_steps: float = field(
        default=5000,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=5000,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})

    deepspeed: Optional[Union[dict, str]] = field(
        default='/home/user02/SCY/VLM/my_vlm/ds_configs/zero2.json',
        # default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    cache_dir: Optional[str] = field(default='./cache_dir')
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")

    # use_flash_attn: bool = field(default=False)
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=False)
    train_llm: bool = field(default=False)
    train_other: bool = field(default=False)
    train_all: bool = field(default=False)
    train_moe: bool = field(default=True)
    mask_question_tokens: bool = field(default=True)
    vision_encoder_keys: Optional[List[str]] = field(default_factory=
    lambda: ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"])
    vision_projector_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual.merger"])
    llm_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["model"])
    other_param_keys: Optional[List[str]] = field(default_factory=
    lambda:  [None])
    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = field(default=False)
    train_vision_encoder_lora: bool = field(default=True)
    train_vision_projector_lora: bool = field(default=True)
    train_llm_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_all_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual", "model"])
    lora_vision_encoder_keys: Optional[List[str]] = field(default_factory=
    lambda:  ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks","vision_tower"])
    lora_vision_projector_keys: Optional[List[str]] = field(default_factory=
    lambda: ["visual.merger",'multi_modal_projector'])
    lora_llm_keys: Optional[List[str]] = field(default_factory=
    lambda:   ["language_model",'multi_modal_projector'])