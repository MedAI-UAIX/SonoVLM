import random
import PIL.PngImagePlugin

import av
import os
import json

import pandas as pd
from PIL import Image
from typing import Dict, List, Optional
import math
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as V2, InterpolationMode
# 将限制提高到 1000MB（或按需调整）
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 1000 * 1024 * 1024  # 10MB
TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "llava-onevision": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
}


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        data_path: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
        training_stage=None,
        checkpoint=None,
        test_excel_file=None
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []
        #breast:13000,gynaecology:66492,heart:47061,kidney:16388,liver:18043,thyriod:21969,vessel:4320
        random.seed(42)
        self.training_stage = training_stage
        try:
            for data_file in data_path:
                with open(data_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if training_stage=='stage2' and 'llava' in data_file:
                            data = random.sample(data, 10000)
                    self.list_data_dict.extend(data)
        except Exception as e:
            print(f'数据初始化失败：{data}文件错误:Error with {e}')
            raise
        if checkpoint and test_excel_file is not None:
            test_df = pd.read_excel(test_excel_file,dtype=str)
            excel_ids = set(test_df['id'])
            self.list_data_dict = [item for item in self.list_data_dict if item['id'] not in excel_ids]
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = True
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

        self.image_transformer = V2.Compose([
            V2.ToPILImage(),
            # V2.RandomCrop(384,)
            V2.Resize((392,392)),
            # V2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
            # V2.RandomHorizontalFlip(p=0.5)
        ])



    def __len__(self) -> int:
        return len(self.list_data_dict)



    def round_by_factor(self,number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self,number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self,number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(self,
            height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS,
            max_pixels: int = MAX_PIXELS
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def __getitem__(self, i) -> Dict[str, List]:      
        source = self.list_data_dict[i]
        images = []
        if "image" in source:
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")
            if len(image_sources) > 8:
                image_sources = random.sample(image_sources, 8)
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = self.image_transformer(img)
                except Exception as e:
                    raise ValueError(f"图片加载错误:{image_path}")
                # width, height = img.size
                # resized_height, resized_width = self.smart_resize(
                #     height,
                #     width,
                #     factor=IMAGE_FACTOR,
                #     min_pixels=MIN_PIXELS,
                #     max_pixels=MAX_PIXELS,
                # )
                # img = img.resize((resized_width, resized_height))
                images.append( img if self.load_image else image_path)

        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)
        conversations_list = None
        # 提前获取所有可能需要的字段值
        system_prompt = source.get("system_prompt")
        conversations = source.get("conversations")
        alignment_conversations = source.get("Alignment_VQA_conversations")
        instruction_conversations = source.get("Instruction-Tuning_VQA_conversations")
        # 直接使用缓存的值进行条件判断
        if conversations is not None:
            conversations_list = conversations
        elif self.training_stage == 'stage1' and alignment_conversations is not None:
            conversations_list = alignment_conversations
        elif self.training_stage == 'stage2' and instruction_conversations is not None:
            conversations_list = instruction_conversations

        convs = []
        assert len(conversations_list) > 0, f"No conversations found {source}"
        for i, conv in enumerate(conversations_list):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), f"Invalid conversation{source['id']}"
            if i ==0:
                if len(source["image"])>8:
                    conv["value"] = '<image>'*8+conv["value"].replace('<image>','')
                else:
                    conv["value"] = '<image>' * len(source["image"]) + conv["value"]
            convs.append(conv["value"])

        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        return dict(
            id = source['id'],
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )