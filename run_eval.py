import argparse
import json
import collections
import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
# import nltk
# nltk.download('wordnet')
from rouge_chinese  import Rouge
from peft import PeftModel
from textstat import textstat
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForSeq2Seq, \
    LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration, \
    LlavaNextForConditionalGeneration
from qwen_vl_utils import process_vision_info
import warnings

from collators.qwen2_vl import replace_image_tokens
from datasets import LazySupervisedDataset
from evaluation.eval_metrics.glossary import normalize_word
from models.sonovlm import SonoVLM_ForConditionalGeneration


warnings.simplefilter('ignore')


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 



class VLM_Evaluation:
    def __init__(self, excel_path):
        body_list=['breast', 'gynaecology', 'heart', 'kidney', 'liver', 'vessel', 'thyroid']
        self.excel_path = excel_path
        # 读取Excel文件，假设预测值在第二列，真实值在第三列
        # self.target_part = args.eval_out_file
        if '300' in excel_path:
            df = pd.read_excel(self.excel_path, dtype=str)#sheet_name = 'qwen_trained'
            if args.part=='all':
                pass
            elif args.part in body_list:
                df=df[df['set'] == args.part]#        breast,gynaecology,heart,kidney,liver,vessel,thyroid
            else:
                df = df[df['test'] == args.part]  # emergency,exam,in

            self.pred = df.loc[:, 'answer'].tolist()
            self.gt = df.loc[:, 'ground_truth'].tolist()
            # df = pd.read_excel(self.excel_path, dtype=str)
            # df=df[df['set'] == 'breast']#        breast,gynaecology,heart,kidney,liver,vessel,thyroid
            # self.pred = []
            # self.gt = []
            # answer_cols = [col for col in df.columns if re.match(r'answer\d+', col)]
            # ground_truth_cols = [col for col in df.columns if re.match(r'ground_truth\d+', col)]
            #
            # # 确保 answer 和 ground_truth 列是成对的
            # if len(answer_cols) != len(ground_truth_cols):
            #     print("警告：answer 和 ground_truth 列的数量不匹配")
            # answer_cols=['answer1']
            # ground_truth_cols=['ground_truth1']
            # # 遍历每一行
            # for index, row in df.iterrows():
            #     for answer_col, ground_truth_col in zip(answer_cols, ground_truth_cols):
            #         # 提取值
            #         answer_value = row[answer_col]
            #         ground_truth_value = row[ground_truth_col]
            #         # 将值添加到列表中
            #         self.pred.append(answer_value)
            #         self.gt.append(ground_truth_value)
        else:
            df = pd.read_excel(self.excel_path, dtype=str)
            self.pred = df.loc[:, 'answer'].tolist()
            self.gt = df.loc[:, 'ground_truth'].tolist()
        self.wrong_answers = []
        self.closed_questions_count = 0
        self.closed_questions_correct = 0
        self.bertscore=BERTScorer( model_type='/home/user02/SCY/Model/bert-base-chinese',
                   lang="zh", num_layers=12, rescale_with_baseline=True, device='cuda:0',
                   baseline_path='/home/user02/miniconda3/envs/pytorch_2.5.1/lib/python3.12/site-packages/bert_score/rescale_baseline/zh/bert-base-chinese.tsv',
                   use_fast_tokenizer=True)

    def evaluate(self,):
        scores = collections.defaultdict(list)

        for pred_item,gt_item in zip(self.pred,self.gt ):
            pred_value = pred_item.lower()#.replace(' ','')
            gt_value = gt_item.lower()#.replace(' ','')
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)
            pred_value_token = normalize_word(pred_value,exact_match=False)
            gt_value_token = normalize_word(gt_value,exact_match=False)
            weights = [(1.,),(1./2., 1./2.),(1./3., 1./3., 1./3.),(1./4., 1./4., 1./4., 1./4.)]
            bleu_scores = []
            for w in weights:
                bleu_score = sentence_bleu([gt_value_token], pred_value_token, weights=w,
                                           smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
            scores['bleu_scores'].append(bleu_scores)
            # 计算 ROUGE 分数
            rouge = Rouge(metrics=["rouge-1","rouge-2","rouge-3","rouge-4","rouge-5","rouge-l"])
            scores['rouge'].append(rouge.get_scores(' '.join(pred_value_token), ' '.join(gt_value_token)))########这里要不要用空格连接######
            scores['meteor'].append(meteor_score([gt_value_token], pred_value_token))
            bert_precision, bert_recall, bert_f1 = self.bertscore.score([pred_value],[gt_value])
            scores['bertscore'].append([bert_precision.numpy()[0], bert_recall.numpy()[0], bert_f1.numpy()[0]])

        exact_match_avg = sum(scores['exact_match']) / len(scores['exact_match'])
        f1_score_avg = sum(scores['f1']) / len(scores['f1'])
        precision_avg = sum(scores['precision']) / len(scores['precision'])
        recall_avg = sum(scores['recall']) / len(scores['recall'])
        meteor = sum(scores['meteor']) / len(scores['meteor'])
        bert_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bertscore'])]
        bleu_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bleu_scores'])]
        rouge_sums = {}
        rouge_counts = {}

        # 遍历 scores['rouge'] 中的每个字典
        for score_list in scores['rouge']:
            for rouge_type, metrics in score_list[0].items():
                if rouge_type not in rouge_sums:
                    # 初始化总和和计数
                    rouge_sums[rouge_type] = {'f': 0, 'p': 0, 'r': 0}
                    rouge_counts[rouge_type] = 0
                # 累加每个子指标的值
                rouge_sums[rouge_type]['f'] += metrics['f']
                rouge_sums[rouge_type]['p'] += metrics['p']
                rouge_sums[rouge_type]['r'] += metrics['r']
                # 增加计数
                rouge_counts[rouge_type] += 1

        # 计算每个 ROUGE 类型的子指标平均值
        rouge_averages = {}
        for rouge_type, total in rouge_sums.items():
            rouge_averages[rouge_type] = {
                'f': total['f'] / rouge_counts[rouge_type],
                'p': total['p'] / rouge_counts[rouge_type],
                'r': total['r'] / rouge_counts[rouge_type],
            }

        results = [
                ['Exact Match Score', exact_match_avg * 100],
                ['Precision', precision_avg * 100],
                ['Recall', recall_avg * 100],
                ['F1 Score', f1_score_avg * 100],
                ['Meteor', meteor * 100],
                ['BLEU Score (Weight 1)', bleu_scores_avg[0] * 100],
                ['BLEU Score (Weight 2)', bleu_scores_avg[1] * 100],
                ['BLEU Score (Weight 3)', bleu_scores_avg[2] * 100],
                ['BLEU Score (Weight 4)', bleu_scores_avg[3] * 100],
                ['Rouge-1', rouge_averages['rouge-1']['f']*100],
                ['Rouge-2', rouge_averages['rouge-2']['f'] * 100],
                ['Rouge-3', rouge_averages['rouge-3']['f'] * 100],
                ['Rouge-4', rouge_averages['rouge-4']['f'] * 100],
                ['Rouge-5', rouge_averages['rouge-5']['f'] * 100],
                ['Rouge-L', rouge_averages['rouge-l']['f'] * 100],
                ['BERTScorer_Precision', bert_scores_avg[0] * 100],
                ['BERTScorer_Recall', bert_scores_avg[1] * 100],
                ['BERTScorer_F1', bert_scores_avg[2] * 100],
            ]
        results_table = tabulate(
            results,
            headers=['Metric', 'Performance (%)']
        )
        print(results_table)
        output_path = self.excel_path.replace('.xlsx', f'_{args.part}.xlsx')
        df_results = pd.DataFrame(results, columns=['Metric', 'Performance (%)'])
        df_results.to_excel(output_path, index=False)


class Inference:
    def __init__(self, fintune_model_dir,original_model_dir,data_path,image_folder,
                 pred_gt_excel_path,use_cache,checkpoint=None):
        self.test_dataset_loader = DataLoader(LazySupervisedDataset(data_path=data_path,image_folder=image_folder,
                                                                    checkpoint=checkpoint,test_excel_file=pred_gt_excel_path),
                                              num_workers=4,batch_size=1,collate_fn=self.collate_fn)
        self.processor = AutoProcessor.from_pretrained(original_model_dir)
        if args.model_name == 'Qwen2VLMOE' :
            self.model = Qwen2VLMOEForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                            attn_implementation="flash_attention_2",
                                                                            device_map=args.gpu, torch_dtype=torch.bfloat16,
                                                                            return_dict=False).eval()
            self.model.config.output_router_logits = False
            # self.model = PeftModel.from_pretrained(self.model, fintune_model_dir)
            self.model_name = 'Qwen2VLMOE'
        elif args.model_name == 'UltrasoundMOE':
            self.model = UltrasoundMOEForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                            attn_implementation="flash_attention_2",
                                                                            device_map=args.gpu, torch_dtype=torch.bfloat16,
                                                                            return_dict=False).eval()
            self.model.config.output_router_logits = False
            # self.model = PeftModel.from_pretrained(self.model, fintune_model_dir)
            self.model_name = 'UltrasoundMOE'
        elif args.model_name == 'Qwen2VL':
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                         attn_implementation="flash_attention_2",
                                                                         device_map=args.gpu,
                                                                         torch_dtype=torch.bfloat16).eval()
            self.model_name = 'Qwen2VL'
        elif args.model_name =='LLaVAOnevision':
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                                device_map=args.gpu,
                                                                                attn_implementation="flash_attention_2",
                                                                                torch_dtype=torch.bfloat16).eval()
            self.model_name = 'LLaVA-Onevision-7B'
        elif args.model_name =='Llama3.2Vision':
            self.model = MllamaForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                                device_map=args.gpu,
                                                                                # attn_implementation="flash_attention_2",
                                                                                torch_dtype=torch.bfloat16).eval()
            self.model_name = 'Llama-3.2-11B-Vision-Instruct'
        elif args.model_name =='LLaVAMed':
            self.model = LlavaForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                  torch_dtype=torch.bfloat16,
                                                                  attn_implementation="flash_attention_2",
                                                                  device_map=args.gpu,
                                                                  # ignore_mismatched_sizes=True
                                                                  ).eval()
            self.model_name = 'LLaVA-Med'
        elif args.model_name =='LLaVA':
            self.model = LlavaNextForConditionalGeneration.from_pretrained(fintune_model_dir,
                                                                  torch_dtype=torch.bfloat16,
                                                                  attn_implementation="flash_attention_2",
                                                                  device_map=args.gpu,
                                                                  # ignore_mismatched_sizes=True
                                                                  ).eval()
            self.model_name = 'LLaVA'


        # if 'Llama-3.2-11B-Vision-Instruct' in fintune_model_dir:
        #     self.processor.tokenizer.padding_side='left'
        self.list_data_dict=[]
        self.pred_gt_excel_path=pred_gt_excel_path
        self.use_cache = use_cache
        print(f'Model Name:{self.model_name}')
        self.activations = []
        self.gradients = []
        self.cam = args.cam
        if  self.cam:
            self.output_attentions=True
        # try:
        #     for data in data_path:
        #         data = json.load(open(data, "r"))
        #         self.list_data_dict.extend(data)
        # except Exception as e:
        #     print(f'数据初始化json文件错误:Error with {e}')
        #     raise
        # self.image_folder = image_folder

    def collate_fn(self,batch):
            # 假设每个样本都是一个列表，我们将其堆叠成一个更大的列表
        ids, images, videos,conversations,system_prompt = zip(*batch)
            # 将ids转换为列表
        # ids = list(ids)
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

    def replace_image_tokens(self, input_string):
        input_string = input_string.replace("<image>\n", "")
        input_string = input_string.replace("<image>", "")
        return input_string

    def generate_batch(self,):

        # 初始化每个样本的独立上下文存储
        batch_contexts = [[] for _ in range(self.batch_size)]  # 每个样本维护自己的对话历史
        all_results = []  # 用于暂存所有结果再统一写入

        for batch_idx, batch_data in tqdm(enumerate(self.test_dataset_loader), total=len(self.test_dataset_loader)):
            current_batch_size = len(batch_data['ids'])

            # 处理每个样本的对话轮次
            for conv_idx in range(0, len(batch_data['conversations'][0]), 2):  # 假设所有样本对话轮次相同
                batch_inputs = []
                batch_images = []

                # 为当前对话轮次构建每个样本的输入
                for sample_idx in range(current_batch_size):
                    # 获取当前样本的对话历史和图像
                    messages = batch_contexts[sample_idx].copy()
                    one_data = batch_data['conversations'][sample_idx]

                    # 处理用户输入
                    user_content = []
                    user_text = self.replace_image_tokens(one_data[conv_idx])
                    user_content.append({"type": "text", "text": user_text})

                    # 处理图像token
                    num_images = len(re.findall("<image>", one_data[conv_idx]))
                    for _ in range(num_images):
                        user_content.append({"type": "image"})

                    messages.append({"role": "user", "content": user_content})

                    # 应用聊天模板（保持原始消息结构）
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    batch_inputs.append(text)
                    batch_images.append(batch_data['images'][sample_idx][:num_images])  # 假设图像已对齐

                # 批量处理输入
                inputs = self.processor(
                    text=batch_inputs,
                    images=[img for img_list in batch_images for img in img_list],  # 展开图像列表
                    padding=True,
                    return_tensors="pt"
                ).to(self.model.device)

                # 批量生成
                with torch.no_grad():
                    if self.model_name == 'LLaVA-Onevision-7B':
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=4096,
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                    else:
                        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)

                # 解码并更新上下文
                generated_texts = self.processor.batch_decode(
                    [gen_ids[len(in_ids):] for gen_ids, in_ids in zip(generated_ids, inputs.input_ids)],
                    skip_special_tokens=True
                )

                # 收集结果并更新上下文
                for sample_idx in range(current_batch_size):
                    # 获取当前样本的真实答案
                    gt = batch_data['conversations'][sample_idx][conv_idx + 1] if conv_idx + 1 < len(
                        batch_data['conversations'][sample_idx]) else ""

                    # 记录结果
                    all_results.append({
                        'id': batch_data['ids'][sample_idx],
                        'question': batch_data['conversations'][sample_idx][conv_idx],
                        'answer': generated_texts[sample_idx],
                        'ground_truth': gt
                    })

                    # 更新该样本的对话上下文
                    batch_contexts[sample_idx].append({
                        "role": "user",
                        "content": [{"type": "text", "text": batch_data['conversations'][sample_idx][conv_idx]}]
                    })
                    batch_contexts[sample_idx].append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_texts[sample_idx]}]
                    })

            # 每处理完一个batch立即保存结果
            df = pd.DataFrame(all_results)
            if not os.path.exists(self.pred_gt_excel_path):
                df.to_excel(self.pred_gt_excel_path, index=False)
            else:
                existing_df = pd.read_excel(self.pred_gt_excel_path)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_excel(self.pred_gt_excel_path, index=False)
            all_results = []  # 清空临时存储

        # else:
        #     ground_truth_list.append(conversation_data)
        #             # ground_truth.append({
        #             #     "role": "ground_truth",
        #             #     "content": conversation_data
        #             # })

    def generate(self,):
        # 应用聊天模板并预处理输入
        id_list,question_list,answer_list,ground_truth_list =[], [],[],[]
        for idx , batch_data in tqdm(enumerate(self.test_dataset_loader),total=len(self.test_dataset_loader)):
            for i ,one_data in enumerate(batch_data['conversations']):
                messages = []
                for conversations_id, conversation_data in enumerate(one_data):
                    if conversations_id % 2 == 0:
                        num_images = conversation_data.count("<image>")
                        num_videos = conversation_data.count("<video>")
                        conversation_data = conversation_data.replace("<image>\n", "").replace("<video>\n", "").strip()
                        conversation_data = conversation_data.replace("<image>", "").replace("<video>", "").strip()
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": conversation_data}] + \
                                       [{"type": "image"}] * num_images + \
                                       [{"type": "video"}] * num_videos
                        })
                        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        # image_inputs, _ = process_vision_info(data['images'])
                        inputs = self.processor(text=[text], images=batch_data['images'][i], padding=True, return_tensors="pt")
                        inputs = inputs.to(self.model.device)
                        if self.model_name == 'LLaVA-Onevision-7B' or self.model_name == 'LLaVA':
                            generated_ids = self.model.generate(**inputs, max_new_tokens=4096,use_cache=self.use_cache,
                                                                pad_token_id=self.processor.tokenizer.eos_token_id)
                        else:
                            #do_sample=True,
                            generated_ids = self.model.generate(**inputs,repetition_penalty=1.1, max_new_tokens=4096,use_cache=self.use_cache,
                                                                # output_attentions=self.output_attentions,return_dict=False
                                                                )
                        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)
                        messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": output_text[0]}]
                        })
                        print(conversation_data)
                        print(output_text)
                        new_df = pd.DataFrame({
                        'id':batch_data['ids'],
                        'question': [conversation_data],
                        'answer': [output_text[0]],
                        'ground_truth': [one_data[conversations_id+1]]
                        })
                        if not os.path.exists(self.pred_gt_excel_path):
                            new_df.to_excel(self.pred_gt_excel_path, index=False)
                        else:
                            with pd.ExcelWriter(self.pred_gt_excel_path, mode='a', engine='openpyxl',
                                                if_sheet_exists='overlay') as writer:
                                start_row = writer.book.active.max_row
                                new_df.to_excel(writer,  startrow=start_row, header=False,
                                                index=False)
                # 清除 KV 缓存
            if hasattr(self.model, "reset_cache"):
                self.model.reset_cache()
            elif hasattr(self.model, "past_key_values"):
                self.model.past_key_values = None

            # 释放 GPU 缓存
            torch.cuda.empty_cache()


    def count_total_parameters(self):
        """计算模型的总参数量（包括所有专家和共享参数）"""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e9:.4f}B")


    def count_activated_expert_parameters(self):
        activated_params=0
        exclude_expert_module_name=['experts.0','experts.1']
        for name, param in self.model.named_parameters():
            if 'experts.0' in name or 'experts.1' in name:
                continue
            activated_params += param.numel()
            print(name)
        print(f"Activated parameters: {activated_params / 1e9:.4f}B")
        # return expert_params
        #
        # with torch.no_grad():
        #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_gates=True)
        #
        # # 假设门控输出存储在 outputs.gate_logits 中
        # gate_logits = outputs.gate_logits  # shape: [batch_size, seq_length, num_experts]
        #
        # # 获取激活的专家索引（假设门控网络输出是稀疏的，即只有某些专家被激活）
        # activated_experts = torch.nonzero(gate_logits > 0.5)  # 假设阈值为 0.5
        #
        # # 获取激活的专家 ID
        # activated_expert_ids = activated_experts[:, -1].unique()
        #
        # # 统计激活的专家参数
        # activated_params = 0
        # for name, param in model.named_parameters():
        #     if expert_module_name in name:
        #         # 提取专家 ID
        #         expert_id = int(name.split(expert_module_name)[-1].split('.')[0])
        #         if expert_id in activated_expert_ids:
        #             activated_params += param.numel()
        #
        # return activated_params

    # 针对Qwen2-VL的ViT结构注册钩子[[6]]
    def register_hook(self,):
        target_layer = self.model.visual.merger.mlp
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def gradcam(self,outputs):
        # 前向计算
        # outputs = self.model(**inputs, output_hidden_states=True)

        # 获取跨模态对齐特征[[7]]
        text_features = outputs.text_hidden_states[-1][:, -1, :]
        image_features = outputs.image_hidden_states[-1].mean(dim=1)

        # 计算相似度得分
        scores = torch.matmul(text_features, image_features.T)

        # 反向传播
        self.model.zero_grad()
        scores.backward(torch.ones_like(scores), retain_graph=True)

        # 生成CAM（适配ViT特征维度）
        weights = torch.mean(self.gradients[-1], dim=[1, 2], keepdim=True)
        cam = torch.sum(weights * self.activations[-1], dim=1)

        return cam

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to ground truth file')
    parser.add_argument('--model_name', type=str, default="LLaVA",choices=[
        'UltrasoundMOE','Qwen2VLMOE','Qwen2VL','LLaVAOnevision','Llama3.2Vision','LLaVAMed','LLaVA']

                        )
    parser.add_argument("--model_path", type=str, default='/home/user02/SCY/VLM/my_vlm/checkpoints/LLaVA_fine_tuning/merged_full_parameters',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B','/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--original_model_path", type=str, default='/home/user02/SCY/Model/llava-v1.6-mistral-7b-hf',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B',
                                 '/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--json_file_path", type=str, default=[
        # '/home/user02/LRF/VLM/json/2024test-code/合并版test/emergency_gynaecology_test.json',
        '/home/user02/LRF/VLM/json/2024test-code/合并版test/emergency_kidney_test.json',
        # '/home/user02/LRF/VLM/json/2024test-code/合并版test/emergency_liver_test.json'
     # '/home/user02/LRF/VLM/json/2024test-code/合并版test/exam_gynaecology_test.json',
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/exam_heart_test.json',
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/exam_liver_test.json',
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/exam_thyroid_test.json'
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/exam_vessel_test.json'
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_breast_test.json',
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_gynaecology_test.json'
     #    '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_heart_test.json'
# '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_kidney_test.json',
#         '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_liver_test.json'
#         '/home/user02/LRF/VLM/json/2024test-code/合并版test/in_thyroid_test.json'
#         '/home/user02/SCY/VLM/my_vlm/Missing/qwen-trainED-qa3.json'
#         '/home/user02/SCY/VLM/my_vlm/Missing/ours-qa-5.json'
                                                               ],
                        choices=['breast', 'gynaecology', 'heart', 'kidney', 'liver', 'vessel', 'thyroid'])
    parser.add_argument("--image_folder", type=str, default='/home/user02/LRF/VLM/Data/2024testnew')
    parser.add_argument('--pred_gt_excel_path', type=str,
                        default="/home/user02/SCY/VLM/my_vlm/test_llava_train/llava_emergency_kidney_test.xlsx", help='path to prediction file',
                        choices=['llava_onevision_7B_test.xlsx','Llama-3_2-Vision_11B_test.xlsx'])
    parser.add_argument('--output', type=str, default="wrong_answers.json", help='path to output file for wrong answers')
    parser.add_argument('--use_cache', type=str, default="./cache_dir",
                        help='path to output file for wrong answers')
    parser.add_argument('--gpu', type=str, default="cuda:2",
                        help='path to output file for wrong answers')
    #下面是计算指标的代码
    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='path to output file for wrong answers')
    parser.add_argument('--eval_excel_file_path', type=str,
                        default='/home/user02/SCY/VLM/my_vlm/ablation/Qwen2VL_MOE_2Expert_Top2/kidney_only_QA.xlsx',
    help='path to prediction file',)
    parser.add_argument('--part', type=str, default='kidney',
                        choices=['急诊','住院','体检','emergency','in','exam',
                                 'breast','gynaecology', 'heart', 'kidney', 'liver', 'vessel', 'thyroid','all'])
    parser.add_argument('--eval_out_path', type=str, default='/home/user02/SCY/VLM/my_vlm/test_2024_llava_onevision_full_finetuning/300',
                        help='cam')
    parser.add_argument('--cam', type=bool, default=False,
                        help='cam')



    args, unparsed = parser.parse_known_args()
    # args.eval_out_file = os.path.join(args.eval_out_path,f'{args.part}.xlsx')
    # args = parse_option()
    inference = Inference(args.model_path,args.original_model_path,data_path=args.json_file_path,image_folder=args.image_folder,
                          pred_gt_excel_path=args.pred_gt_excel_path,use_cache=args.use_cache,
                          checkpoint=args.checkpoint)
    # # # inference.register_hook()
    inference.generate()
    # from transformers import Qwen2VLForConditionalGeneration
    # inference.count_total_parameters()
    # inference.count_activated_expert_parameters()
    # #
    # vlm_evaluation = VLM_Evaluation(args.eval_excel_file_path)
    # vlm_evaluation.evaluate()

