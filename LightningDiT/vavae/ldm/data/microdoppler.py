import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MicroDopplerDataset(Dataset):
    """微多普勒时频图像数据集"""

    def __init__(self, data_root, size=256, interpolation="bicubic", flip_p=0.5):
        self.data_root = data_root
        self.size = size
        self.interpolation = {"linear": Image.BILINEAR,  # 修复：LINEAR已被移除，使用BILINEAR
                            "bilinear": Image.BILINEAR,
                            "bicubic": Image.BICUBIC,
                            "lanczos": Image.LANCZOS,}[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # 收集所有图像文件
        self.image_paths = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"Found {len(self.image_paths)} images in {data_root}")

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=self.interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = self.flip(image)
        return {"image": image}
