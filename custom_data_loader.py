#!/usr/bin/env python3
"""
自定义数据加载器，适用于分类目录结构的数据集
支持 /kaggle/input/dataset/ID_1/, ID_2/, ... ID_31/ 结构
"""

import os
import glob
from PIL import Image
import numpy as np
import albumentations
from torch.utils.data import Dataset
from omegaconf import OmegaConf


class CustomImageDataset(Dataset):
    """
    自定义图像数据集加载器
    适用于按类别组织的目录结构: data_root/class1/, data_root/class2/, ...
    """
    
    def __init__(self, data_root, size=256, config=None):
        self.data_root = data_root
        self.size = size
        self.config = config or {}
        
        # 支持的图像格式
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # 图像预处理
        self.preprocessor = albumentations.Compose([
            albumentations.Resize(size, size),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载所有图像路径
        self._load_image_paths()
        
        print(f"✅ 加载了 {len(self.image_paths)} 张图像，来自 {len(self.class_names)} 个类别")
        
    def _load_image_paths(self):
        """加载所有图像路径和标签"""
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # 扫描所有子目录
        class_dirs = sorted([d for d in os.listdir(self.data_root) 
                           if os.path.isdir(os.path.join(self.data_root, d))])
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(self.data_root, class_name)
            self.class_names.append(class_name)
            
            # 扫描该类别下的所有图像
            class_images = []
            for ext in self.image_extensions:
                class_images.extend(glob.glob(os.path.join(class_path, ext)))
            
            # 添加到总列表
            for img_path in class_images:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                
        print(f"📁 发现类别: {self.class_names}")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # 确保是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 转换为numpy数组
        image = np.array(image)
        
        # 应用预处理
        processed = self.preprocessor(image=image)
        image = processed['image']
        
        # 返回数据字典（兼容LDM格式）
        return {
            'image': image.astype(np.float32),
            'class_label': self.labels[idx],
            'class_name': self.class_names[self.labels[idx]],
            'file_path_': img_path
        }


class CustomImageTrain(CustomImageDataset):
    """训练数据集"""
    def __init__(self, data_root, config=None, **kwargs):
        super().__init__(data_root, config=config, **kwargs)


class CustomImageValidation(CustomImageDataset):
    """验证数据集"""
    def __init__(self, data_root, config=None, **kwargs):
        super().__init__(data_root, config=config, **kwargs)
        # 可以在这里添加验证集特定的逻辑，比如只使用部分数据
