import re
from typing import Dict, List, Sequence, Union, Optional
import PIL
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional
import numpy as np
import torch
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig, DataCollatorForSeq2Seq
from . import register_collator
import torch.distributed as dist
# from .base import BaseDataCollator
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)

SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
IGNORE_INDEX = -100


# @register_collator("qwen2-vl")
class Qwen2VLDataCollator(ABC):
    def __init__(
            self,
            config: Optional[AutoConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            processor: Optional[AutoProcessor] = None,
            mask_question_tokens: bool = True
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "images" in instances[0]:
            is_video = False
        elif "videos" in instances[0]:
            is_video = True
            
        if not is_video:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            videos = None
            images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        else:
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            images = None
            videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        max_len = self.tokenizer.model_max_length

        total_image_tokens = 0
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_vision_grid_thw = []
        
        for b_idx, (system_prompt, cur_convs) in enumerate(zip(system_prompts, conversations)):
            cur_input_ids = []
            cur_labels = []
            cur_pixel_values = []
            cur_vision_grid_thw = []            
            cur_text = []

            if system_prompt is None:
                system_prompt = SYSTEM_MESSAGE
            #处理btch中的每一条
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                    total_image_tokens += num_image_tokens

                    cur_text.append({
                        "role": "user",
                        "content": replace_image_tokens(text, is_video=is_video)
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": text
                    })
            
            # heavily borrowed from https://github.com/2U1/Qwen2-VL-Finetune
            #这里处理system_prompt
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            cur_input_ids.append(system_message_input_ids.squeeze(0))
            cur_labels.append(system_labels.squeeze(0))
            #关于添加图像的ID号：https://moon-ci-docs.huggingface.co/docs/transformers/pr_35264/en/model_doc/qwen2_vl
            # apply_chat_template_text = self.processor.apply_chat_template(cur_text, tokenize=False,add_generation_prompt=True, )
            for idx, j in enumerate(range(0, len(cur_text), 2)):    #开始处理一条数据，就是一条数据的所有对话和图片
                user_input = cur_text[j]
                gpt_response = cur_text[j + 1]
                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                gpt_response = f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                
                if idx == 0:
                    if not is_video:
                        inputs = self.processor(text=[user_input], images=images[b_idx], videos=None, padding=False, return_tensors='pt')
                    else:
                        inputs = self.processor(text=[user_input], images=None, videos=videos[b_idx], padding=False, return_tensors='pt')
                    prompt_input_ids = inputs['input_ids']
                    pixel_values = inputs[pixel_key]
                    vision_grid_thw = inputs[grid_key]
                else:
                    prompt_input_ids = self.processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                response_input_ids = self.processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                if self.mask_question_tokens:
                    labels = torch.cat([torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),response_input_ids.squeeze(0),],dim=0,)
                else:
                    labels = cur_input_ids.clone()
                cur_input_ids.append(input_ids)
                cur_labels.append(labels)
                cur_pixel_values.append(pixel_values)
                cur_vision_grid_thw.append(vision_grid_thw)
            
            cur_input_ids = torch.cat(cur_input_ids, dim=0).to(torch.long)
            cur_labels = torch.cat(cur_labels, dim=0).to(torch.long)
            cur_pixel_values = torch.cat(cur_pixel_values, dim=0)
            cur_vision_grid_thw = torch.cat(cur_vision_grid_thw, dim=0)
            # manual truncation
            if cur_input_ids.shape[0] > max_len:
                rank0_print(f"超过model_max_length：{max_len},现在长度为：{cur_input_ids.shape[0]}")
                cur_input_ids = cur_input_ids[:max_len]
                cur_labels = cur_labels[:max_len]

            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"
            
            cur_input_ids = cur_input_ids.unsqueeze(0)
            cur_labels = cur_labels.unsqueeze(0)
            # DataCollatorForSeq2Seq
            # padding
            if self.tokenizer.padding_side=='left' and cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    torch.full((cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        fill_value=self.tokenizer.pad_token_id,dtype=cur_input_ids.dtype,device=cur_input_ids.device),
                    cur_input_ids], dim=1)
                cur_labels = torch.cat([torch.full(
                    (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        fill_value=IGNORE_INDEX,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    ),cur_labels
                ], dim=1)
            elif self.tokenizer.padding_side == 'right' and cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([cur_input_ids,
                                           torch.full((cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                                           fill_value=self.tokenizer.pad_token_id,dtype=cur_input_ids.dtype,
                                           device=cur_input_ids.device)], dim=1)
                cur_labels = torch.cat([cur_labels, torch.full(
                    (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device
                )], dim=1)
            batch_input_ids.append(cur_input_ids)
            batch_labels.append(cur_labels)
            batch_pixel_values.append(cur_pixel_values)
            batch_vision_grid_thw.append(cur_vision_grid_thw)
            
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_labels = torch.cat(batch_labels, dim=0)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_vision_grid_thw = torch.cat(batch_vision_grid_thw, dim=0)

        # sanity check
        assert total_image_tokens == count_innermost_elements(images), "Number of image tokens does not match the number of images"

        data_dict = dict(
            input_ids=batch_input_ids,
            labels=batch_labels,
            attention_mask=batch_input_ids.ne(self.tokenizer.pad_token_id),
        )
        data_dict[pixel_key] = batch_pixel_values
        data_dict[grid_key] = batch_vision_grid_thw
        
        return data_dict
    

def count_innermost_elements(nested_list):
    if not isinstance(nested_list, list):
        return 1
    return sum(count_innermost_elements(item) for item in nested_list)


def _findall(token_list: torch.Tensor, token: int) -> torch.Tensor:
    if not isinstance(token_list, torch.Tensor):
        raise ValueError("token_list must be a PyTorch Tensor")
    mask = token_list == token
    indices = torch.where(mask)[0]

    return indices


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        input_string = input_string.replace("<video>"+'\n', "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<video>", "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")

    else:
        input_string = input_string.replace("<image>"+'\n', "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<image>", "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")

    return input_string