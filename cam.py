import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration, \
    LlavaNextForConditionalGeneration
import torch.nn.functional as F
from datasets import LazySupervisedDataset
from models.qwen2_vl_continued_moe import Qwen2VLMOEForConditionalGeneration


class ModelEvaluator:
    def __init__(self, model_path, original_model_path, json_file_path, image_folder,question_json):
        self.model_path = model_path
        self.original_model_path = original_model_path
        self.json_file_path = json_file_path
        self.image_folder = image_folder
        # self.mode = mode
        self.processor = None
        self.model = None
        self.test_dataset_loader = None
        self.feature_maps = {}
        self.gradients = {}
        self.current_layer_name = None
        with open(question_json, "r", encoding="utf-8") as file:
            self.question_json = json.load(file)

    def replace_image_tokens(self, input_string):
        input_string = input_string.replace("<image>\n", "")
        input_string = input_string.replace("<image>", "")
        return input_string

    def find_vision_token_indexes(self, input_list):
        start_token = 151652
        end_token = 151653
        start_indices = []
        end_indices = []

        try:
            for i in range(len(input_list)):
                if input_list[i] == start_token:
                    start_indices.append(i + 1)  # 记录开始 token 的索引
                elif input_list[i] == end_token:
                    end_indices.append(i)  # 记录结束 token 的索引
            if len(start_indices) != len(end_indices):
                raise ValueError("开始和结束 token 的数量不一致")
            return start_indices, end_indices
        except Exception as e:
            print(f"An error occurred: {e}")
            return [-1], [-1]

    def setup_seeds(self, config):
        seed = config.run_cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    def collate_fn(self, batch):
        ids, images, videos, conversations, system_prompt = zip(*batch)
        ids = [item['id'] for item in batch]
        images = [item['images'] for item in batch]
        videos = [item['videos'] for item in batch]
        conversations = [item['conversations'] for item in batch]
        system_prompts = [item['system_prompt'] for item in batch]
        return {
            'ids': ids,
            'images': images,
            'videos': videos,
            'conversations': conversations,
            'system_prompts': system_prompts
        }



    # 修改特征图保存函数
    def save_feature_maps(self, layer_name, module, input, output):
        # 更新当前处理的层名称
        self.current_layer_name = layer_name
        # 如果层名称不存在于字典中，初始化一个空列表
        if layer_name not in self.feature_maps:
            self.feature_maps[layer_name] = []
        # 将特征图数据添加到对应层的列表中
        feature_map_data = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
        self.feature_maps[layer_name].append(feature_map_data)
        # 如果需要保留梯度，可以在这里实现
        # output[0].retain_grad() if isinstance(output, tuple) else output.retain_grad()

    # 修改梯度保存函数
    def save_gradients(self, layer_name, module, grad_input, grad_output):
        # 如果层名称不存在于字典中，初始化一个空列表
        if layer_name not in self.gradients:
            self.gradients[layer_name] = []
        # 将梯度数据添加到对应层的列表中
        gradient_data = grad_output[0].detach().cpu() if isinstance(grad_output, tuple) else grad_output.detach().cpu()
        self.gradients[layer_name].append(gradient_data)

    def register_hooks(self, model):
        layers = []
        if 'llava-onevision' in self.model_path.lower():
            for i in range(len(model.vision_tower.vision_model.encoder.layers)):
                layers.append(model.vision_tower.vision_model.encoder.layers[i])
        elif 'llava' in self.model_path.lower():
            for i in range(len(model.vision_tower.vision_model.encoder.layers)):
                layers.append(model.vision_tower.vision_model.encoder.layers[i])
        elif 'moe'in self.model_path.lower():
            if args.cam_mode=="vision":#visual.blocks.0.norm2.weight
                for i in range(len(model.visual.blocks)):
                    layers.append(model.visual.blocks[i].norm2)
            else:
                for i in range(len(model.model.layers)):
                    layers.append(model.model.layers[i].post_attention_layernorm)
        elif 'qwen2-vl' in self.model_path.lower():
            for i in range(len(model.visual.blocks)):
                # layers.append(model.model.layers[i].post_attention_layernorm)
                layers.append(model.visual.blocks[i])

        # 为每一层注册钩子，并传递层名称
        for layer_idx, layer in enumerate(layers):
            layer_name = f"layer_{layer_idx}"
            # 为前向传播注册钩子
            layer.register_forward_hook(
                lambda module, input, output, name=layer_name: self.save_feature_maps(name, module, input, output))
            # 为反向传播注册钩子
            layer.register_full_backward_hook(
                lambda module, grad_input, grad_output, name=layer_name: self.save_gradients(name, module, grad_input,
                                                                                             grad_output))
    def initialize_model(self):
        if 'llava-onevision' in self.model_path.lower():
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(self.model_path,
                                                                                device_map="cuda:1",
                                                                                attn_implementation="flash_attention_2",
                                                                                torch_dtype=torch.bfloat16).eval()
        elif 'llava' in self.model_path.lower():
            self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path,
                                                                        device_map="cuda:1",
                                                                        attn_implementation="flash_attention_2",
                                                                        torch_dtype=torch.bfloat16).eval()
        elif 'moe'in self.model_path.lower():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Current device: {device}")
            self.model = Qwen2VLMOEForConditionalGeneration.from_pretrained(self.model_path,
                                                                        device_map="cuda:1",
                                                                        attn_implementation="flash_attention_2",
                                                                        torch_dtype=torch.bfloat16).eval()
        elif 'qwen2-vl' in self.model_path.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path,
                                                                        device_map="cuda:1",
                                                                        attn_implementation="flash_attention_2",
                                                                        torch_dtype=torch.bfloat16).eval()

        self.processor = AutoProcessor.from_pretrained(self.original_model_path)
        self.test_dataset_loader = DataLoader(LazySupervisedDataset(data_path=self.json_file_path, image_folder=self.image_folder),
                                              num_workers=4, batch_size=1, collate_fn=self.collate_fn)

    def process_batch(self, batch_data):
        image_list = batch_data['images'][0]
        content = [{"type": "text", "text": batch_data['conversations']}]
        if len(image_list) >8:
            image_list = random.sample(image_list, 8)
        for _ in image_list:
            content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        # generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=4096,
        #                                     # pad_token_id=processor.tokenizer.eos_token_id,
        #                                     return_dict_in_generate=True,
        #                                     # output_attentions=True,
        #                                     # output_scores=True
        #                                     )
        # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
        #                                           clean_up_tokenization_spaces=False)
        # self.plot_cosine_similarity()
        vision_token_start, vision_token_end = self.find_vision_token_indexes(inputs['input_ids'].squeeze(0).tolist())
        return inputs, vision_token_start, vision_token_end, image_list
    def plot_cosine_similarity(self, dim: int = 1, eps: float = 1e-8):
        num_features=len(self.feature_maps)
        similarity_matrix = torch.zeros((num_features, num_features))
        # 计算两两之间的余弦相似度
        for i,layer_key_i in enumerate(self.feature_maps.keys()):
            for j,layer_key_j in enumerate(self.feature_maps.keys()):
                similarity_matrix[i, j] = F.cosine_similarity(self.feature_maps[layer_key_i][0], self.feature_maps[layer_key_j][0]).mean().abs()
                #a=cosine_similarity(features_list[i].detach().float().cpu().numpy(), features_list[j].detach().float().cpu().numpy())
                b=0
        # self.similarity_matrix+=similarity_matrix
        # self.similarity_matrix_num+=1
        # # 绘制相似度矩阵
        # # if self.similarity_matrix_num>=100:
        # self.similarity_matrix /=self.similarity_matrix_num

        min_sim = similarity_matrix.min()
        max_sim = similarity_matrix.max()
        similarity_matrix_normalized = (similarity_matrix - min_sim) / (max_sim - min_sim)
        # alpha = 0.5  # 调整这个参数可以控制差异拉大的程度
        # similarity_matrix_stretched = similarity_matrix_normalized * alpha
        #
        # # 将拉伸后的相似度矩阵限制在 [0, 1] 范围
        # similarity_matrix_stretched = torch.clamp(similarity_matrix_stretched, 0, 1)

        plt.figure(figsize=(20, 16))
        plt.imshow(similarity_matrix_normalized, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.title('32 Feature Maps Cosine Similarity Matrix')
        plt.xlabel('Feature Map Index')
        plt.ylabel('Feature Map Index')
        plt.xticks(range(num_features))
        plt.yticks(range(num_features))
        plt.tight_layout()
        plt.savefig('image.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        # plt.savefig('similarity_matrix.png')
        # rgb_image = plt.cm.viridis(similarity_matrix_normalized)[:, :, :3]  # 去除 alpha 通道
        # image_bgr = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('similarity_matrix.png', image_bgr)

        a=0
    def generate_and_save_cams(self, inputs,all_token, vision_token_start, vision_token_end, batch_data,image_list):
        for layer_idx in self.feature_maps.keys():
            feature_map = self.feature_maps[layer_idx]
            gradient = self.gradients[layer_idx]
            if args.cam_mode == "vision":#28*28
                feature_map = feature_map[0]
                gradient = gradient[0]
                one_img_patch = feature_map.shape[0]//len(image_list)
                for img_idx in range(len(image_list)):
                    h = w = int(np.sqrt(one_img_patch))
                    img_feature_map = rearrange(feature_map[img_idx*one_img_patch:(img_idx+1)*one_img_patch], '(h w) c ->c h w',h=h, w=w)
                    img_gradient = rearrange(gradient[img_idx*one_img_patch:(img_idx+1)*one_img_patch],'(h w) c -> h w c', h=h, w=w)
                    img_gradient = nn.ReLU()(img_gradient)
                    pooled_gradients = torch.mean(img_gradient, dim=[0,1])
                    activation = img_feature_map
                    for i in range(activation.size(0)):
                        activation[i, :, :] *= pooled_gradients[i]
                    heatmap = torch.mean(activation, dim=0).cpu().float().numpy()
                    heatmap = np.maximum(heatmap, 0)
                    heatmap /= np.max(heatmap)
                    # 特征筛选
                    threshold = 0.5  # 可以调整这个值
                    heatmap[heatmap < threshold] = 0  # 将低于阈值的值设为0
                    patch_size = 14
                    heatmap = cv2.resize(heatmap, (336, 336))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    original_image = np.array(image_list[img_idx])
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    superimposed_img = heatmap * 0.4 + original_image
                    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                    save_path = f'/home/user02/SCY/VLM/my_vlm/heatmap_vision/{batch_data['ids'][0]}'
                    path_cam_img = os.path.join(save_path, f"vision_layer_{layer_idx}_img_num{img_idx}.png")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(path_cam_img, superimposed_img)

            elif args.cam_mode=="per_token":
                feature_map = torch.cat(feature_map, dim=1)
                for grad_idx, grad in enumerate(gradient):
                    if grad.shape[1]==1:
                        continue
                    for img_idx in range(len(image_list)):
                        h=w=int(np.sqrt(feature_map[:, vision_token_start[img_idx]:vision_token_end[img_idx], :].shape[1]))
                        img_feature_map = rearrange(feature_map[:, vision_token_start[img_idx]:vision_token_end[img_idx], :], 'b (h w) c ->b c h w', h=h, w=w)
                        img_gradient = rearrange(grad[:, vision_token_start[img_idx]:vision_token_end[img_idx], :], 'b (h w) c -> b h w c', h=h, w=w)

                        img_gradient = nn.ReLU()(img_gradient)
                        pooled_gradients = torch.mean(img_gradient, dim=[0, 1, 2])
                        activation = img_feature_map.squeeze(0)
                        for i in range(activation.size(0)):
                            activation[i, :, :] *= pooled_gradients[i]
                        heatmap = torch.mean(activation, dim=0).cpu().float().numpy()
                        heatmap = np.maximum(heatmap, 0)
                        heatmap /= np.max(heatmap)
                        # 特征筛选
                        threshold = 0.3  # 可以调整这个值
                        heatmap[heatmap < threshold] = 0  # 将低于阈值的值设为0
                        patch_size = 14
                        heatmap = cv2.resize(heatmap, (384, 384))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        original_image = np.array(image_list[img_idx])
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                        superimposed_img = heatmap * 0.4 + original_image
                        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                        save_path = f'/home/user02/SCY/VLM/my_vlm/heatmap_vision/{batch_data['ids'][0]}'
                        path_cam_img = os.path.join(save_path, f"per_token_layer_{layer_idx}_img_num{img_idx}_grad_idx{grad_idx}.png")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(path_cam_img, superimposed_img)
            elif args.cam_mode == "sentence":
                    feature_map = torch.cat(feature_map, dim=1)
                    gradient = torch.cat(gradient, dim=1)
                    for img_idx in range(len(image_list)):
                        h = w = int(
                            np.sqrt(feature_map[:, vision_token_start[img_idx]:vision_token_end[img_idx], :].shape[1]))
                        img_feature_map = rearrange(
                            feature_map[:, vision_token_start[img_idx]:vision_token_end[img_idx], :],
                            'b (h w) c ->b c h w', h=h, w=w)
                        img_gradient = rearrange(gradient[:, vision_token_start[img_idx]:vision_token_end[img_idx], :],
                                                 'b (h w) c -> b h w c', h=h, w=w)

                        img_gradient = nn.ReLU()(img_gradient)
                        pooled_gradients = torch.mean(img_gradient, dim=[0, 1, 2])
                        activation = img_feature_map.squeeze(0)
                        for i in range(activation.size(0)):
                            activation[i, :, :] *= pooled_gradients[i]
                        heatmap = torch.mean(activation, dim=0).cpu().float().numpy()
                        heatmap = np.maximum(heatmap, 0)
                        heatmap /= np.max(heatmap)

                        patch_size = 14
                        heatmap = cv2.resize(heatmap, (392, 392))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                        original_image = np.array(image_list[img_idx])
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                        superimposed_img = heatmap * 0.4 + original_image
                        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                        save_path = f'/home/user02/SCY/VLM/my_vlm/heatmap/{batch_data['ids'][0]}'
                        path_cam_img = os.path.join(save_path, f"sentence_layer_{layer_idx}_img_num{img_idx}.png")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(path_cam_img, superimposed_img)
    def train_and_generate(self):
        self.register_hooks(self.model)
        for idx, batch_data in tqdm(enumerate(self.test_dataset_loader), total=len(self.test_dataset_loader)):
            if idx<6:
                continue
            conversations = batch_data['conversations'][0]
            flag = False
            for conv_idx in range(0,len(conversations),2):
                for question in self.question_json:
                    if question in conversations[conv_idx] or conversations[conv_idx] in question:
                        batch_data['conversations'] = question
                        flag = True
                        break
                if flag:
                    break

            # batch_data['conversations'] = conversations[0].replace("<image>", "")

            inputs, vision_token_start, vision_token_end, image_list = self.process_batch(batch_data)
            generated_tokens = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            if 'llava-onevision' in self.model_path.lower() or 'llava' in self.model_path.lower():
                image_sizes = inputs["image_sizes"]
                position_ids = torch.arange(0, inputs.input_ids.shape[1]).to(self.model.device)
            elif 'moe'in self.model_path.lower() or 'qwen2-vl' in self.model_path.lower():
                image_grid_thw = inputs["image_grid_thw"]
            past_key_values = None
            cache_position = torch.arange(0, inputs.input_ids.shape[1]).to(self.model.device)
            i = 0

            # generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=4096,
            #                                     # pad_token_id=processor.tokenizer.eos_token_id,
            #                                     # return_dict_in_generate=True,
            #                                     # output_attentions=True,
            #                                     # output_scores=True
            #                                     )
            # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            # output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
            #                                           clean_up_tokenization_spaces=False)

            sequence_logits = []
            # all_logits = []
            while True:
                self.model.zero_grad()
                with torch.set_grad_enabled(True):
                    if 'llava-onevision' in self.model_path.lower():
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:].to(self.model.device) if past_key_values else generated_tokens,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position,
                            logits_to_keep=1
                        )
                    elif 'llava' in self.model_path.lower():
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:].to(self.model.device) if past_key_values else generated_tokens,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position,
                            logits_to_keep=1
                        )
                    elif 'moe'in self.model_path.lower():
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:] if past_key_values else generated_tokens,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position
                        )
                    elif 'qwen2-vl' in self.model_path.lower():
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:] if past_key_values else generated_tokens,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position
                        )

                    logits = outputs.logits
                    # all_logits.append(logits)
                    sequence_logits.append(logits[:, -1, :])
                    past_key_values = outputs.past_key_values
                    pixel_values = None
                    cache_position = cache_position[-1].unsqueeze(-1) + 1
                    if args.cam_mode=="per_token":
                        logits_max = logits[:, -1, :].max(-1)[0]
                        logits_max.backward(retain_graph=True)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long).to(attention_mask.device)], dim=-1)
                    i += 1
                    # if i>25:
                    #     break
                    if next_token.item() == self.processor.tokenizer.eos_token_id:
                        break
                    if generated_tokens.shape[1] >= 2048:
                        break


            input_token_len = inputs.input_ids.shape[1]
            if args.cam_mode == "sentence" or args.cam_mode == "vision":
                sequence_logits = torch.cat(sequence_logits, dim=0)[1:,:]
                # all_logits = torch.cat(all_logits, dim=1)[1:]
                target_logits = torch.sum(sequence_logits[
                                          torch.arange(sequence_logits.shape[0]), torch.argmax(sequence_logits, dim=-1)])
                target_logits.backward(retain_graph=True)
            self.generate_and_save_cams(inputs,generated_tokens, vision_token_start, vision_token_end, batch_data,image_list)
            self.feature_maps={}
            self.gradients = {}
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default='/home/user02/SCY/VLM/my_vlm/checkpoints/Qwen2VL_MOE_stage2/merged_full_parameters',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B'])
    parser.add_argument("--original_model_path", type=str, default='/home/user02/SCY/Model/Qwen2-VL-7B-Instruct',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B',
                                 '/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--json_file_path", type=str, default=['/home/user02/LRF/VLM/json/2024test-code/test300/test300.json'],
                        choices=['/home/user02/LRF/VLM/json/2024test-code/test300/test300.json',
                                 '/home/user02/SCY/VLM/my_vlm/heatmap_json_q.json',
                                '/home/user02/SCY/VLM/my_vlm/heatmap_img0608.json']
                        )
    parser.add_argument("--image_folder", type=str, default='/home/user02/LRF/VLM/Data/2024testnew')
    parser.add_argument("--mode", type=str, default='moe',choices=['qwen2vl','llava','moe'])
    parser.add_argument("--cam_mode", type=str, default='sentence', choices=['per_token', 'sentence','vision'])
    parser.add_argument("--question_json", type=str, default='/home/user02/SCY/VLM/my_vlm/dataset/classification_question.json',)
    args = parser.parse_known_args()[0]

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        original_model_path=args.original_model_path,
        json_file_path=args.json_file_path,
        image_folder=args.image_folder,
        question_json = args.question_json
        # mode=args.mode
    )
    evaluator.initialize_model()
    evaluator.train_and_generate()