import math
import re
from typing import Dict, List, Sequence, Union, Optional
import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
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



class LLaVAMedDataCollator(BaseDataCollator):
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
        # self.start_token_sequence = [733,28748,16289,28793,28705]
        self.start_token_sequence = [319,1799,9047,13566,29901,29871]
        self.start_token_len =len(self.start_token_sequence)
        self.end_token_sequence = [3148,1001]
        self.end_token_len = len(self.end_token_sequence)
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
          [INST] <image>
<image>
请解释图片中所呈现的是什么。 [/INST] 双侧乳腺腺体结构排列紊乱，乳导管不扩张，左侧乳腺未见明确占位性病变。于右侧乳腺_Loc_区距乳头_SCM_处可见一低回声结节，大小约_3DS_，边界清晰，形态规整，CDFI示边缘处可探及搏动性血流信号。双侧腋下未见明显肿大淋巴结。</s> [INST] 如果结节是乳腺癌，如何治疗？ [/INST] 如果结节是乳腺癌，治疗可以包括手术切除、化疗、放疗等，具体治疗方案需要根据个体情况确定。</s> [INST] CDFI示边缘处可探及搏动性血流信号的临床意义是什么？ [/INST] CDFI示边缘处可探及搏动性血流信号可能意味着结节有血液供应，需要进一步检查以确认结节的性质。</s> [INST] 左侧乳腺未见明确占位性病变的临床意义是什么？ [/INST] 左侧乳腺未见明确占位性病变可能意味着左侧乳腺没有明显的肿瘤或占位性病变，但仍需要进一步检查以排除早期癌症。</s> [INST] 乳导管不扩张的临床意义是什么？ [/INST] 乳导管不扩张可能是乳腺癌的早期征象，需要进一步检查以排除癌症。</s> [INST] 双侧腋下未见明显肿大淋巴结的临床意义是什么？ [/INST] 双侧腋下未见明显肿大淋巴结可能意味着没有明显的淋巴结转移，但仍需要进一步检查以排除早期癌症。</s> [INST] 如何预防乳腺癌？ [/INST] 预防乳腺癌可以包括定期乳腺检查、保持健康的生活方式（如均衡饮食、定期锻炼）、避免高风险因素（如吸烟、饮酒）等。</s> [INST] 右侧乳腺_Loc_区距乳头_SCM_处可见一低回声结节的临床意义是什么？ [/INST] 右侧乳腺_Loc_区距乳头_SCM_处可见一低回声结节可能是乳腺癌的早期征象，需要进一步检查以确认结节的性质。</s> [INST] 如何进一步检查以确认结节的性质？ [/INST] 进一步检查可以包括乳腺活检、乳腺成像（如乳腺X线摄影、乳腺超声波检查）等，以确认结节的性质。</s> [INST] 边界清晰，形态规整的临床意义是什么？ [/INST] 边界清晰，形态规整可能意味着结节是良性的，但仍需要进一步检查以确认结节的性质。</s>
        This function tries to find the indexes of the assistant content in the input_ids list to build labels.
        '''
        # (Pdb++) processor.tokenizer.encode(" [/INST] ")###<s>[INST] 用户输入 [/INST]注意有空格，
        #<s>▁[INST]▁用户输入文字内容和<image>▁[/INST]
        # <s>:1,    </s>:2    ▁[:733, INST:16289, ]:28793,    ▁:28705这个是空格,   <image>:32000,   /:28748
        #所以要查找▁[/INST]▁:[733,28748,16289,28793,28705]和</s>▁:[2,28705]之间的内容


        #<s> ▁US ER : ▁ 用户输入内容  ▁A SS IST ANT : ▁
        #<s>:1, ▁US: 3148, ER:1001,  ":": 29901, ▁:29871, "<0x0A>": 13,
        # "▁A": 319,"SS": 1799,"IST": 9047,"ANT": 13566,":": 29901,▁:29871
        #找[319,1799,9047,13566,29901,29871]->[3148,1001,29901,29871]

        start_indexes = []
        end_indexes = []
        # Iterate through the list to find starting points
        for i in range(0,len(input_list)-self.start_token_len + 1):
            # Check if the current and next elements form the start sequence
            if input_list[i:i+self.start_token_len] == self.start_token_sequence :
                start_indexes.append(i+self.start_token_len)#
                # Now look for the first 151645 and 198 after the start
                for j in range(i + self.start_token_len, len(input_list)-1):
                    if input_list[j:j+2] == self.end_token_sequence:
                        end_indexes.append(j+2)  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
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
            temp = self.processor.apply_chat_template(
                cur_text,
                # chat_template=template,
                add_generation_prompt=False,
                # tokenize = False,  # True
                # return_assistant_tokens_mask=False,  # True
                # return_dict=False,  # True
                # return_tensors="pt",
                # truncation=False  # the assistant tokens mask seems wrong when truncation is enabled
            )
            batch_cur_text.append(temp)
        images_flat_list = []
        # # 使用 extend 方法
        for sublist in images:
            images_flat_list.extend(sublist)
        # output_kwargs = self.processor._merge_kwargs(
        #     LlavaProcessorKwargs,
        #     tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        # )
        batch_input_ids = self.processor(text = batch_cur_text,images=images_flat_list,return_tensors="pt",padding=True)
        batch_input_ids_lists = batch_input_ids['input_ids'].tolist()
        print(batch_cur_text)
        print(batch_input_ids_lists)
        return
        for ids_list in batch_input_ids_lists:
            label_ids = [-100] * len(ids_list)  # -100 is the ignore index in loss function
            for begin_end_indexs in self.find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            batch_labels.append(label_ids)
        batch_labels=torch.tensor(batch_labels,dtype=torch.int64)
        return dict(
            # **dict(pixel_values=batch_input_ids['pixel_values']),
            input_ids=batch_input_ids['input_ids'],
            labels=batch_labels,
            attention_mask=batch_input_ids['attention_mask'],
            pixel_values=batch_input_ids['pixel_values'].clone(),
            # image_sizes=batch_input_ids['image_sizes']
        )