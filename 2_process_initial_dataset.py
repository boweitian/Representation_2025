import json
import random
import numpy as np
from transformers import PreTrainedTokenizerFast

def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizerFast, seed: int = 0):
    """
    Processes COCO dataset to create a dataset where each sample contains:
    - Truncated statements
    - Honest and untruthful variations
    - Corresponding image_id

    Args:
    - data_path (str): Path to the JSON containing the data.
    - tokenizer (PreTrainedTokenizerFast): Tokenizer to tokenize statements.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Dictionary containing all processed data under a single key.
    """

    # 设置随机种子
    random.seed(seed)

    # 读取 JSON 数据
    with open(data_path, "r") as f:
        data = json.load(f)

    # 模板字符串
    template_str = "USER: Pretend you're {type} person making statements about the world. ASSISTANT:"

    dataset = []  # 存储所有数据

    # 处理每个数据样本
    for item in data:
        image_id = item["image_id"]  # 提取 image_id
        statement = item["caption"].split("Assistant:")[-1].strip()  # 提取 caption 里的 statement
        tokens = tokenizer.tokenize(statement)

        # 生成逐步截断的 honest/untruthful 语句
        for idx in range(1, len(tokens) - 5):  # 逐步截断句子
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_text = f"{template_str.format(type='an honest')} {truncated_statement}"
            untruthful_text = f"{template_str.format(type='an untruthful')} {truncated_statement}"

            # 存储结果
            dataset.append({
                "image_id": image_id,
                "honest": honest_text,
                "untruthful": untruthful_text
            })

    print(f"Total processed samples: {len(dataset)}")

    return {"data": dataset}  # 统一存入一个 key

# 示例调用
tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")  # 使用 BERT tokenizer
dataset = honesty_function_dataset("file/coco.json", tokenizer)

# 保存数据集
with open("file/coco_honesty_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
