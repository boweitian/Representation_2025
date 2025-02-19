from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from repe.rep_reading_pipeline import transform
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
from repe.pipelines import repe_pipeline_registry
from PIL import Image

repe_pipeline_registry()
# **检查 PyTorch 看到的 GPU**
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6,7"

print(f"PyTorch 可见的 GPU 数量: {torch.cuda.device_count()}")  
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} 设备名称: {torch.cuda.get_device_name(i)}")
    
direction_method = 'pca'
rep_token = -1
debug = False

# **2. 加载模型与Processor**
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b", load_in_8bit=True, device_map="auto"  # 启用 8-bit 量化
)
model_name_or_path = "ehartford/Wizard-Vicuna-30B-Uncensored"
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0

hidden_layers = list(range(-1, -33, -1)) # 33: num_layers
n_difference = 1

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer = tokenizer, image_processor=processor.image_processor)

import json

with open("file/coco_honesty_dataset.json", "r") as f:
    dataset_raw = json.load(f)
if debug:
    dataset_raw['data'] = dataset_raw['data'][0:4]
# 解析数据
data = []
labels = []
images = []
for item in dataset_raw["data"]:
    honest = item["honest"]
    untruthful = item["untruthful"]
    route = item["image_id"]

    data.append(honest)
    data.append(untruthful)
    labels.append([1, 0])
    images.append(route)

# 形成最终的数据集格式
dataset = {
    "train": {
        "data": data,
        "labels": labels,
        "images": images
    }
}
dataset['train']['images'] = [img for img in dataset['train']['images'] for _ in range(2)]
dataset_list = [
    {"data": dataset["train"]["data"][i], "images": dataset["train"]["images"][i]}
    for i in range(len(dataset["train"]["data"]))
]

# 输出数据规模
print(f"Total sentences: {len(dataset['train']['data'])}")  # 应该是 `labels` 长度的两倍
print(f"Total label pairs: {len(dataset['train']['labels'])}")  # 对应 `honest/untruthful` 对

# 可选：保存处理后的数据
with open("file/processed_coco_honesty_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

honesty_rep_reader = rep_reading_pipeline.get_directions(
    dataset_list,
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    n_difference=n_difference, 
    train_labels=dataset['train']['labels'], 
    direction_method=direction_method,
    batch_size=4, processor = processor
)

def format_output(output_text, question, image_path):
    """
    重新格式化 output_text，使其符合 {'data': ..., 'images': ...} 结构
    
    Args:
        output_text (str): 模型生成的原始文本
        question (str): 用户的提问
        image_path (str): 图片路径

    Returns:
        dict: 格式化后的数据
    """
    # **构造新的 USER 部分**
    formatted_user = f"USER: <image> {question}"

    # **解析 ASSISTANT 部分**
    if "ASSISTANT:" in output_text:
        assistant_part = output_text.split("ASSISTANT:")[1].strip()
    else:
        assistant_part = output_text.strip()

    # **构造最终格式**
    formatted_text = f"{formatted_user} ASSISTANT: {assistant_part}"

    # **返回符合格式的字典**
    return {"data": formatted_text, "images": image_path}


def evaluate_honesty(image_path, question):
    """
    计算给定图片+文本生成的回答的 honesty 分数

    Args:
        image_path (str): 图片路径
        question (str): 对图片的提问

    Returns:
        str: 模型生成的回答
        float: honesty 分数
    """
    # **加载图片**
    image = Image.open(image_path).convert("RGB")

    # **构造符合聊天格式的输入**
    prompt = f"USER: <image> {question}\nASSISTANT:"

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    # **生成回答**
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # **计算 honesty 评分**
    results = []
    tokenized_output = processor.tokenizer(output_text, return_tensors="pt")
    input_ids = tokenized_output["input_ids"][0]  # 获取 token 序列
    for ice_pos in range(len(input_ids)):
        ice_pos = -len(hidden_layers) + ice_pos
        # {'data': "USER: <image> Pretend you're an honest person making statements about the world. ASSISTANT: this is,", 'images': '000000000139.jpg'}
        H_tests = rep_reading_pipeline(
            [format_output(output_text, question, image_path)],
            processor = processor, 
            rep_reader=honesty_rep_reader,
            rep_token=ice_pos,
            hidden_layers=hidden_layers,
            image_base_path = './'
        )
        breakpoint()
        results.append(H_tests)

    # **计算 honesty score**
    honesty_scores = [round(r[0][-1].item(), 1) for r in results]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    print("\n📊 Token-wise Honesty Scores:")
    for token, score in zip(tokens, honesty_scores):
        print(f"Token: {token.ljust(15)} | Score: {score}")

    return output_text, honesty_scores

image_path = "image.jpg"  # 你的图片路径
question = "Pretend you're an honest person making statements about the world."

answer, score = evaluate_honesty(image_path, question)

print(f"🖼️ 图片: {image_path}")
print(f"❓ 问题: {question}")
print(f"💬 模型回答: {answer}")

question = "Pretend you're an untruthful person making statements about the world."

answer, score = evaluate_honesty(image_path, question)

print(f"🖼️ 图片: {image_path}")
print(f"❓ 问题: {question}")
print(f"💬 模型回答: {answer}")
breakpoint()