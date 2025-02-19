import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class COCOVal2017Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=1536):
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有图片路径并排序
        
        self.image_filenames = sorted(os.listdir(root_dir))[:max_images]
        self.image_paths = [os.path.join(root_dir, img) for img in self.image_filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = self.image_filenames[idx]  # 文件名作为 ID
        image = Image.open(img_path).convert("RGB")  # **返回 PIL Image**
        
        return image, image_id  # 返回 (PIL Image, image_id)



# 定义数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])