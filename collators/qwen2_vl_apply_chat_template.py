import json
import os
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



class Qwen2VLDataCollator_apply_chat_template(ABC):
    def __init__(
            self,
            config: Optional[AutoConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            processor: Optional[AutoProcessor] = None,
            mask_question_tokens: bool = True,
            checkpoints_path: str = None,

    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.start_token_sequence = [151644, 77091, 198]
        self.end_token_sequence = [151645, 198]
        self.checkpoints_path = checkpoints_path
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.checkpoints_path = os.path.join(self.checkpoints_path,'traing_dataset_checkpoints.json')

    def find_assistant_content_sublist_indexes(self,input_list):
        '''
        A message from train_data/data.json may look like below:
            {
                "messages": [
                    {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
                ]
            }
        After apply_chat_template, the text will look like below:
            ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

        This function tries to find the indexes of the assistant content in the input_ids list to build labels.
        '''
        # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
        # [151644, 77091, 198]
        # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
        # [151645, 198]

        start_indexes = []
        end_indexes = []
        # Iterate through the list to find starting points
        for i in range(13,len(input_list) - 2):
            # Check if the current and next elements form the start sequence
            if input_list[i:i+3] == self.start_token_sequence :
                start_indexes.append(i+3)#
                # Now look for the first 151645 and 198 after the start
                for j in range(i + 3, len(input_list) - 1):
                    if input_list[j:j+2] == self.end_token_sequence:
                        end_indexes.append(j + 2)  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                        break  # Move to the next start after finding the end
        assert len(start_indexes)-1 == len(end_indexes), "Number of start_indexes does not match the number of end_indexes"
        return list(zip(start_indexes, end_indexes))

    def replace_image_tokens_to_start_pad_end(self,input_string, is_video=False):
        if is_video:
            input_string = input_string.replace("<video>" + '\n',
                                                "<|vision_start|>" + "<|video_pad|>" + "<|vision_end|>")
            input_string = input_string.replace("<video>", "<|vision_start|>" + "<|video_pad|>" + "<|vision_end|>")
        else:
            input_string = input_string.replace("<image>" + '\n',
                                                "<|vision_start|>" + "<|image_pad|>" + "<|vision_end|>")
            input_string = input_string.replace("<image>", "<|vision_start|>" + "<|image_pad|>" + "<|vision_end|>")

    def replace_image_tokens_to_none(self, input_string, is_video=False):
        if is_video:
            input_string = input_string.replace("<video>" + '\n',
                                                "<|vision_start|>" + "<|video_pad|>" + "<|vision_end|>")
            input_string = input_string.replace("<video>",
                                                "<|vision_start|>" + "<|video_pad|>" + "<|vision_end|>")
        else:
            input_string = input_string.replace("<image>\n", "")
            input_string = input_string.replace("<image>", "")
            return input_string

        return input_string
    def save_trained_data(self, id):
        with open(self.checkpoints_path, 'a') as f:
            json.dump(id, f)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ids = [instance["id"] for instance in instances]
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        # videos: List[Union[np.ndarray, None]] = [x for instance in instances for x in instance["videos"]]
        videos = [instance["videos"] for instance in instances]
        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        batch_labels = []
        batch_apply_chat_template_text =[]
        for id,system_prompt, cur_images, cur_videos, cur_convs in zip(ids,system_prompts, images, videos, conversations):
            self.save_trained_data(id)
            cur_text = []
            cur_num_images =cur_num_videos = 0
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            #处理btch中的每一条
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images  = text.count("<image>")
                    cur_num_images += num_images
                    num_videos = text.count("<video>")
                    cur_num_videos += num_videos
                    text = text.replace("<image>\n", "").replace("<video>\n", "").strip()
                    text = text.replace("<image>", "").replace("<video>", "").strip()
                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                                   [{"type": "image"}] * num_images + \
                                   [{"type": "video"}] * num_videos
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}]
                    })
            assert len(cur_images) == cur_num_images, "Not all images were used"
            assert len(cur_videos) == cur_num_videos, "Not all videos were used"
            # heavily borrowed from https://github.com/2U1/Qwen2-VL-Finetune
            #关于添加图像的ID号：https://moon-ci-docs.huggingface.co/docs/transformers/pr_35264/en/model_doc/qwen2_vl
            apply_chat_template_text = self.processor.apply_chat_template(cur_text, tokenize=False,add_generation_prompt=True,add_vision_id=False)
            batch_apply_chat_template_text.append(apply_chat_template_text)
        # 无论怎么样都是返回的一个列表的图像，没有列表套列表，可以传[[image]]也可以传[image]
        batch_input_ids = self.processor(text=batch_apply_chat_template_text, images=images, padding=True,return_tensors="pt")
        batch_input_ids_lists = batch_input_ids['input_ids'].tolist()
        for ids_list in batch_input_ids_lists:
            label_ids = [IGNORE_INDEX] * len(ids_list)  # -100 is the ignore index in loss function
            for begin_end_indexs in self.find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            batch_labels.append(label_ids)
        batch_labels=torch.tensor(batch_labels,dtype=torch.int64)
        data_dict = dict(
            input_ids=batch_input_ids['input_ids'],
            labels=batch_labels,
            attention_mask=batch_input_ids['attention_mask'],
            pixel_values = batch_input_ids['pixel_values'],
            image_grid_thw=batch_input_ids['image_grid_thw']
        )
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


