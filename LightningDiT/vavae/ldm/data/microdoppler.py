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

        # 数据预处理 - 针对256×256图像优化
        if size == 256:
            # 如果目标尺寸是256×256，跳过resize和crop（假设输入已经是256×256）
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
            ])
            print(f"⚡ 优化：跳过resize和crop，假设输入图像已是{size}×{size}")
        else:
            # 如果需要其他尺寸，保留完整预处理
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=self.interpolation),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
            ])
            print(f"📐 标准预处理：resize到{size}×{size}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = self.flip(image)
        return {"image": image}
