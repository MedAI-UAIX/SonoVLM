import math
import re
from typing import Dict, List, Sequence, Union, Optional
import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessorKwargs
from transformers.utils import logging
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig, DataCollatorForSeq2Seq
from . import register_collator
from .base import BaseDataCollator
import time
logger = logging.get_logger(__name__)
# slightly different from https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/chat_template.json
# to include <|im_end|> of assistant's response as labels
template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + ' '}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all video then #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<video>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{{'<|im_end|>'}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% if message['role'] != 'assistant' %}"
    "{{'<|im_end|>'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


class LLaVAOnevisionDataCollator(BaseDataCollator):
    def __init__(self,
            config: Optional[AutoConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            processor: Optional[AutoProcessor] = None,
            mask_question_tokens: bool = True
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
        self.start_token_sequence = [151644, 77091]
        self.end_token_sequence = [151645]
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
        # user:872

        start_indexes = []
        end_indexes = []
        # Iterate through the list to find starting points
        for i in range(13,len(input_list) - 1):
            # Check if the current and next elements form the start sequence
            if input_list[i:i+2] == self.start_token_sequence :
                start_indexes.append(i+2)#
                # Now look for the first 151645 and 198 after the start
                for j in range(i + 2, len(input_list)-1):
                    if input_list[j:j+1] == self.end_token_sequence:
                        end_indexes.append(j + 1)  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                        break  # Move to the next start after finding the end
        return list(zip(start_indexes, end_indexes))
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        images = [instance["images"] for instance in instances]
        videos = [instance["videos"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        batch_cur_text = []
        batch_labels=[]
        for system_prompt, cur_images, cur_videos, cur_convs in zip(system_prompts, images, videos, conversations):
            cur_num_images = 0
            cur_num_videos = 0
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })

            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

                    num_videos = len([m.start() for m in re.finditer("<video>", text)])
                    cur_num_videos += num_videos

                    # .strip(): whitespaces and newlines are handled by chat_template
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
            temp = self.processor.apply_chat_template(
                cur_text,
                # chat_template=template,
                add_generation_prompt=False,
                tokenize = False,  # True
                return_assistant_tokens_mask=False,  # True
                return_dict=False,  # True
                return_tensors="pt",
                truncation=False  # the assistant tokens mask seems wrong when truncation is enabled
            )
            batch_cur_text.append(temp)
        batch_input_ids = self.processor(text = batch_cur_text,images=images,return_tensors="pt",padding=True)
        batch_input_ids_lists = batch_input_ids['input_ids'].tolist()
        for ids_list in batch_input_ids_lists:
            label_ids = [-100] * len(ids_list)  # -100 is the ignore index in loss function
            for begin_end_indexs in self.find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            batch_labels.append(label_ids)
        batch_labels=torch.tensor(batch_labels,dtype=torch.int64)
        return dict(
            input_ids=batch_input_ids['input_ids'],
            labels=batch_labels,
            attention_mask=batch_input_ids['attention_mask'],
            pixel_values=batch_input_ids['pixel_values'],
            image_sizes=batch_input_ids['image_sizes']
        )