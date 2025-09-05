#!/usr/bin/env python3
"""
微多普勒数据集 - 完全匹配VA-VAE的预处理格式
用于扩散模型训练
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset


class MicrodopplerDataset(Dataset):
    """
    微多普勒数据集 - 完全匹配step4_train_vavae.py的预处理方式
    """
    
    def __init__(self, root_dir, split_file, split='train', transform=None, 
                 return_user_id=False, image_size=256):
        self.data_root = Path(root_dir)
        self.image_size = image_size
        self.split = split
        self.return_user_id = return_user_id
        self.transform = transform  # 保留兼容性，但不使用
        
        # 加载数据划分
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # 获取对应split的样本
        if split not in split_data:
            raise ValueError(f"Split '{split}' not found in {split_file}")
            
        self.samples = []
        
        # 检查split_data的格式
        if isinstance(split_data[split], dict):
            # 按用户划分的格式：{'ID_1': ['path1', 'path2'], 'ID_2': [...]}
            for user_id, image_paths in split_data[split].items():
                for image_path in image_paths:
                    # image_path可能是相对路径或绝对路径
                    if Path(image_path).is_absolute():
                        sample_path = Path(image_path)
                    else:
                        sample_path = self.data_root / image_path
                    
                    if sample_path.exists() and sample_path.is_file():
                        self.samples.append({
                            'path': sample_path,
                            'user_id': user_id
                        })
                    else:
                        print(f"⚠️ 图像文件不存在: {sample_path}")
                        
        elif isinstance(split_data[split], list):
            # 列表格式：['ID_1', 'ID_2', ...]（旧格式兼容）
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
            for sample in split_data[split]:
                if isinstance(sample, str):
                    sample_path = self.data_root / sample
                    
                    if sample_path.is_dir():
                        # 是目录，扫描其中的图像文件
                        user_id = sample
                        
                        for img_file in sample_path.iterdir():
                            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                                self.samples.append({
                                    'path': img_file,
                                    'user_id': user_id
                                })
                    elif sample_path.is_file():
                        # 是文件，直接使用
                        path_parts = Path(sample).parts
                        user_id = path_parts[0] if path_parts else 'unknown'
                        self.samples.append({
                            'path': sample_path,
                            'user_id': user_id
                        })
        else:
            raise ValueError(f"不支持的split数据格式: {type(split_data[split])}")
        
        print(f"✅ 加载{split}集: {len(self.samples)}个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像 - 完全匹配step4_train_vavae.py的方式
        img = Image.open(sample['path']).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # **关键**：完全匹配VA-VAE的预处理方式
        # 从step4_train_vavae.py第118-122行：
        img_array = np.array(img).astype(np.float32)  # HWC格式 [256,256,3]
        img_array = img_array / 127.5 - 1.0  # 归一化到[-1,1]
        
        # 转为tensor，保持HWC格式（这很重要！）
        img_tensor = torch.from_numpy(img_array)  # [256,256,3]
        
        if self.return_user_id:
            return img_tensor, sample['user_id']
        else:
            return img_tensor, 0  # 返回dummy label，保持DataLoader兼容性
