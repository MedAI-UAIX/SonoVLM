import os
import sys
from typing import List, Dict, Optional, Union, Literal
import argparse
import json
import re
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import AutoProcessor, DINOv3ViTImageProcessorFast, LlavaForConditionalGeneration, \
    Qwen2VLForConditionalGeneration, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, \
    LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SonoVLM_V2.models.internvl_3_5_moe import InternVL_MOE_ForConditionalGeneration
from SonoVLM_V2.models.internvl_3_5_moe_usfm_512 import InternVL_MOE_USFM_512_ForConditionalGeneration
from SonoVLM_V2.models.lingshu_multi_vision import Multi_Vision_ForConditionalGeneration
from SonoVLM_V2.models.qwen2_vl_continued_moe import Qwen2VLMOEForConditionalGeneration
from SonoVLM_V2.collators.datacollator_apply_chat_template import InternVLProcessor_Custom
from SonoVLM_V2.collators.datasetV2 import LazySupervisedDatasetV2
from custom_dataset import EvalDataset

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download,
    get_model_tokenizer, get_template, InferRequest, VllmEngine, BaseArguments
)
from swift.tuners import Swift


class MultiTurnModelValidator:
    """
    支持多轮对话的验证器，核心逻辑：
    1. 初始只发送user消息，不发送数据集中的assistant gt
    2. 每轮将模型预测结果作为assistant消息添加到下一轮对话
    3. 一条完整的多轮对话全部结束后，统一存储所有历史
    """

    def __init__(
            self,
            model: str,
            lora_checkpoint: str,
            data_path: str,
            data_folder: Optional[str] = None,
            video_folder: Optional[str] = None,
            output_file: str = "validation_results.json",
            batch_size: int = 8,
            max_tokens: int = 512,
            temperature: float = 0.0,
            num_workers: int = 8,
            template_type: Optional[str] = None,
            default_system: Optional[str] = None,
            infer_backend: Literal['vllm', 'pt'] = 'vllm',
            num_turns: Optional[int] = None,
    ):
        self.output_file = output_file
        self.batch_size = batch_size
        self.num_turns = num_turns
        self.data_folder = data_folder

        # 1. 加载模型
        if infer_backend == 'pt':
            model, processor = get_model_tokenizer(model_id_or_path=model, attn_impl='flash_attention_2',
                                                   model_type=args.model_type, device_map=args.gpu)
            # model = Swift.from_pretrained(model=model, model_id=lora_checkpoint)
            # model = model.merge_and_unload()
            template_type = template_type or model.model_meta.template
            self.template = get_template(template_type, processor, default_system=default_system, max_length=None)
            self.engine = PtEngine.from_model_template(model, self.template, max_batch_size=batch_size)
        elif infer_backend == 'vllm':
            self.engine = VllmEngine(
                model_id_or_path=model,
                tensor_parallel_size=args.tensor_parallel_size,
                limit_mm_per_prompt={'image': 100, 'video': 1},
                gpu_memory_utilization=args.gpu_memory_utilization,
                model_type=args.model_type,

            )
            self.template = get_template(args.model_type, self.engine.processor, default_system=default_system,
                                         max_length=None)

        self.request_config = RequestConfig(max_tokens=max_tokens,n=5, temperature=temperature,top_k=20,top_p=0.7,repetition_penalty=1.05)

        # 2. 加载数据集
        self.dataset = EvalDataset(
            data_path=data_path,
            image_folder=data_folder,
            video_folder=video_folder,
            test_file=self.output_file if args.checkpoint else None,
            num_frames=8,
            num_workers=num_workers,
        )

    def _load_media_files(self, media_paths: Union[str, List[str]], folder: Optional[str]) -> List[str]:
        """加载媒体文件路径"""
        if isinstance(media_paths, str):
            media_paths = [media_paths]

        files = []
        for path in media_paths:
            abs_path = os.path.join(folder, path) if folder else path
            if os.path.isfile(abs_path):
                files.append(abs_path)
            else:
                raise FileNotFoundError(f"文件不存在: {abs_path}")
        return files

    def _get_num_turns(self, sample: Dict) -> int:
        """从样本中推断对话轮次 = user消息数量"""
        messages = sample.get('messages', [])
        return (len(messages) + 1) // 2

    def _apply_image_video_limit(self, images: List[str], message: Dict,videos=None) -> tuple:
        """应用图像数量限制并更新消息内容"""
        limitation = 50
        if len(images) > limitation:
            images = images[:limitation]
            image_tags = '<image>' * limitation
            original_content = message.get('content', '')
            cleaned_content = original_content.replace('<image>', '')
            message['content'] = image_tags + cleaned_content
        # if videos is not None and len(videos)>1:
        #     videos = videos[:limitation]
        #     # image_tags = '<video>' * limitation
        #     original_content = message.get('content', '')
        #     cleaned_content = original_content.replace('<video>', '')
        #     message['content'] = '<video>' + cleaned_content
        #     return images, videos
        return images, message

    def _get_current_user_message(self, sample: Dict, turn: int) -> Dict:
        """获取当前轮次的user消息"""
        messages = sample.get('messages', [])
        if turn * 2 >= len(messages):
            raise ValueError(f"轮次 {turn} 超出数据集范围，样本 {sample.get('id')}")

        return {
            "role": "user",
            "content": messages[turn * 2]["content"]
        }

    def _get_ground_truth(self, sample: Dict, turn: int) -> str:
        """获取当前轮次的ground truth"""
        messages = sample.get('messages', [])
        if turn * 2 + 1 < len(messages):
            return messages[turn * 2 + 1]["content"]
        return ""

    def run(self):
        """执行多轮对话验证，完整对话结束后统一存储"""
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        n = len(self.dataset)
        for batch_idx, start in enumerate(tqdm(range(0, n, self.batch_size), desc="Validating")):
            batch_indices = list(range(start, min(start + self.batch_size, n)))
            batch_samples = [self.dataset[i] for i in batch_indices]

            # 确定每个样本的对话轮次
            if self.num_turns is not None:
                batch_turns = [self.num_turns] * len(batch_samples)
            else:
                batch_turns = [self._get_num_turns(sample) for sample in batch_samples]

            # 为每个样本创建对话状态存储
            # 结构: {
            #   "sample_id": str,
            #   "image": Optional[List[str]],
            #   "turns": List[Dict],  # 每轮的结果
            #   "conversation_history": List[Dict]  # 完整对话历史（包含模型预测）
            # }
            conversation_results = []

            # 初始化对话历史：每个样本维护自己的对话状态
            conversation_states = [{"messages": []} for _ in batch_samples]

            max_turns_in_batch = max(batch_turns)
            for turn in range(max_turns_in_batch):
                active_indices = []
                active_requests = []

                # 为每个样本构建当前轮次的请求
                for idx_in_batch, (sample, total_turns, state) in enumerate(
                        zip(batch_samples, batch_turns, conversation_states)):
                    if turn >= total_turns:
                        continue  # 该样本已结束所有轮次

                    try:
                        # 获取当前轮次的user消息
                        current_user_msg = self._get_current_user_message(sample, turn)

                        # 构建本轮推理用的消息列表
                        if turn == 0:
                            # 第一轮：只发送当前user消息 + 处理图像
                            # messages_for_inference = [current_user_msg]
                            images = self._load_media_files(sample.get('images', []), self.data_folder)
                            videos = self._load_media_files(sample.get('videos', sample.get('video',[])), self.data_folder)
                            # 应用图像限制
                            images, current_user_msg = self._apply_image_video_limit(images, current_user_msg)
                            # 更新消息列表
                            messages_for_inference = [current_user_msg]
                        else:
                            # 后续轮次：发送完整对话历史 + 当前user消息
                            messages_for_inference = state["messages"] + [current_user_msg]
                            images = None
                            videos = None

                        # 创建推理请求
                        request = InferRequest(
                            messages=messages_for_inference,
                            images=images,
                            videos=videos
                        )
                        active_indices.append(idx_in_batch)
                        active_requests.append(request)
                    except Exception as e:
                        print(f"构建请求时出错 - 样本 {sample.get('id')}, 轮次 {turn + 1}: {e}")
                        continue

                # 如果本轮没有活跃请求，跳过
                if not active_requests:
                    continue

                # 批量推理
                try:
                    responses = self.engine.infer(active_requests, self.request_config, template=self.template)

                except Exception as e:
                    print(f"推理失败 - 批次 {batch_idx}, 轮次 {turn + 1}: {e}")
                    responses = [None] * len(active_requests)

                # 处理每个活跃样本的响应
                for idx_in_active, (sample_idx, sample, req, resp) in enumerate(
                        zip(active_indices, [batch_samples[i] for i in active_indices],
                            active_requests, responses)):
                    if resp is None:
                        # 推理失败，记录错误
                        turn_result = {
                            "turn": turn + 1,
                            "status": "error",
                            "error": "推理失败"
                        }
                    else:
                        assistant_pred = [choice.message.content for choice in resp.choices]
                        print(f"responses:{assistant_pred}")
                        # assistant_pred = resp.choices[0].message.content
                        gt_content = self._get_ground_truth(sample, turn)

                        turn_result = {
                            "turn": turn + 1,
                            "prompt": req.messages[-1]["content"],  # 当前user消息
                            "pre": assistant_pred,
                            "gt": gt_content,
                        }

                    # 初始化该样本的结果存储（仅在第一轮）
                    if turn == 0:
                        sample_result = {
                            "id": sample.get("id"),
                            "images": req.images if req.images else None,
                            "videos": req.videos if req.videos else None,
                            "total_turns": batch_turns[sample_idx],
                            "turns": []
                        }
                        conversation_results.append(sample_result)
                    else:
                        # 找到对应的样本结果
                        sample_result = conversation_results[active_indices.index(sample_idx)]

                    # 将本轮结果添加到样本的turns列表
                    sample_result["turns"].append(turn_result)

                    # 更新对话状态：将本轮的user消息和assistant预测添加到历史
                    if resp is not None:
                        state = conversation_states[sample_idx]
                        # 添加user消息
                        state["messages"].append({
                            "role": "user",
                            "content": sample['messages'][turn * 2]["content"]
                        })
                        # 添加assistant预测
                        state["messages"].append({
                            "role": "assistant",
                            "content": assistant_pred
                        })

            # 批次所有轮次完成后，统一存储每个样本的完整对话历史
            for sample_result in conversation_results:
                # 添加完整对话历史（所有轮次）
                sample_id = sample_result["id"]
                # 找到对应的对话状态
                for idx, state in enumerate(conversation_states):
                    if batch_samples[idx].get("id") == sample_id and state["messages"]:
                        sample_result["conversation_history"] = state["messages"]
                        # sample_result["messages"] = state["messages"]
                        break

                # 写入文件
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')

        # 转换为标准JSON格式
        self._convert_to_standard_json()

    def _convert_to_standard_json(self):
        """将JSON Lines转换为标准JSON数组格式"""
        all_results = []
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    all_results.append(data)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] 第 {line_num} 行JSON解析失败: {e}")
                    continue

        standard_file = self.output_file.replace(".json", "_standard.json")
        with open(standard_file, 'w', encoding='utf-8') as f_json:
            json.dump(all_results, f_json, ensure_ascii=False, indent=2)
        print(f"标准格式已保存至: {standard_file}")


class Inference:
    """保留原有Inference类作为备选方案"""
    # ... (保留原有Inference类的实现) ...




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # os.environ["IMAGE_MAX_TOKEN_NUM"] = "262144"
    # os.environ['MAX_PIXELS'] = '262144'
    # os.environ['MAX_PIXELS'] = '50176'
    # os.environ['VIDEO_MAX_PIXELS'] = '262144'
    # os.environ['FPS_MAX_FRAMES'] = '1100'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/user02/SCY/SonoVLM_V2/checkpoints/ablation/lora/lora64/checkpoint-685-merged", help="本地模型目录或 ModelScope id")
    parser.add_argument("--lora_checkpoint", default="/path/to/lora", help="本地 LoRA 目录")
    #/home/user02/SCY/VLM/my_vlm/202511/挑選圖像/reader_study_abnormal_300_vqa.json
    #/home/user02/SCY/VLM/my_vlm/202511/report_val_json/report_val_2543.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/reader_study_normal_35.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark_final_report.json#3000
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark_final_caption.json#5000
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/reader_study_report_2578.json
    #/home/user02/SCY/SonoVLM_V2/dataset/video_test/thyroid_video_test_single.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/reader_study_normal_35_vqa.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/caption_hard.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/caption_simple.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/HasLesion.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/HasMeasurement.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/Laterality.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/LesionEcho.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/Modality.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/benchmark5q_json_output_repeat_option/Organ.json
    #/home/user02/SCY/SonoVLM_V2/dataset/video_test/dicom/vlm_video_data.json
    #/home/user02/SCY/SonoVLM_V2/dataset/SCY/benchmark_swift/report_val_2543.json
    #/home/user02/SCY/SonoVLM_V2/dataset/benchmark_swift/reader_study_all_335_vqa1.json
    parser.add_argument("--data_path", default='/home/user02/SCY/SonoVLM_V2/dataset/video_test/MP4/CardiacNet_images.json', help="验证集路径")
    parser.add_argument("--image_folder", default='', help="图片文件夹路径")
    #/home/user02/SCY/SonoVLM_V2/dataset/video_test/thyroid
    #/home/user02/LRF/VLM/Data/diagnosis_acc
    #/home/user02/LRF/VLM
    #/home/user02/LRF/VLM/Data
    #/home/user02/LRF/VLM/Data/report_val
    parser.add_argument("--video_folder", help="视频文件夹路径")#
    parser.add_argument("--output_file", default="/home/user02/SCY/SonoVLM_V2/test_results/video/sonovlm_zeroshot_top5_CardiacNet_images.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--model_type", type=str, default='qwen2_5_vl', choices=['qwen2_5_vl', 'llava1_5_hf','gemma3_vision'])
    parser.add_argument('--gpu', type=str, default="cuda:0")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument("--num_turns", type=int, help="对话轮次，不指定则自动从数据推断")
    parser.add_argument("--infer_backend", type=str, default='vllm', choices=['vllm', 'pt'])

    args = parser.parse_args()

    validator = MultiTurnModelValidator(
        model=args.model,
        lora_checkpoint=args.lora_checkpoint,
        data_path=args.data_path,
        data_folder=args.image_folder,
        video_folder=args.video_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.num_workers,
        infer_backend=args.infer_backend,
        num_turns=args.num_turns,
    )
    validator.run()