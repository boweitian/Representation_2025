import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForVision2Seq
from dataset import COCOVal2017Dataset, transform  # 自定义数据集
from tqdm import tqdm  # 用于进度条
from PIL import Image

# **1. 设备设定**
cuda_num = 1
DEVICE = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"

# **2. 加载模型与Processor**
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b", load_in_8bit=True, device_map={"": cuda_num}  # 启用 8-bit 量化
)

# **3. 加载数据集**
root_dir = "data/COCO/val2017"
dataset = COCOVal2017Dataset(root_dir=root_dir, transform=transform)

train_size = 64
test_size = len(dataset) - train_size

train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))  # 前 1024 张图片作为训练集
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))  # 剩下的 512 张图片作为测试集

# **5. 创建DataLoader**
def collate_fn(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)  # **保持 PIL Image 格式**

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

# **6. 处理数据并进行推理**
def generate_captions(dataloader, split_name="train"):
    model.eval()
    results = []
    latent_features = []
    
    with torch.no_grad():
        for batch_idx, (images, image_ids) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            images = list(images)  # 确保 images 是 List[PIL Image]
            
            # **1. 逐个图片创建 prompt**
            prompts = [
                processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Pretend you're an honest person making statements about the world."}
                            ]
                        }
                    ],
                    add_generation_prompt=True
                ) for _ in images  # 确保每张图片有自己的 message
            ]

            # **2. 处理输入**
            inputs = processor(text=prompts, images=images, return_tensors="pt")  # **batch_size 张图片一起输入**
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # **3. 通过 forward() 获取 latent 表示**
            outputs = model(**inputs, output_hidden_states=True)  # 让模型返回 hidden states
            latent = outputs.hidden_states[-1]  # 获取最后一层 hidden state
            latent_features.append(latent.cpu().numpy())  # 存储到列表
            
            
            # **3. 生成文本**
            generated_ids = model.generate(**inputs, max_new_tokens=50, temperature=2.0, do_sample=True)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # **4. 保存结果**
            for img_id, caption in zip(image_ids, generated_texts):
                results.append({"image_id": img_id, "caption": caption})

    return results

# **7. 生成训练集和测试集的描述**
train_results = generate_captions(train_loader, "train")

import json
with open("file/coco.json", "w") as f:
    json.dump(train_results, f, indent=4)
    
# test_results = generate_captions(test_loader, "test")
# with open("test_captions.json", "w") as f:
#     json.dump(test_results, f, indent=4)

print("Caption generation complete. Results saved.")
