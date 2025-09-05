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
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for sample in split_data[split]:
            # 处理不同的JSON格式
            if isinstance(sample, dict):
                # 标准格式：{'path': '...', 'user_id': '...'}
                sample_path = self.data_root / sample['path']
                user_id = sample.get('user_id', 'unknown')
                
                if sample_path.is_file():
                    self.samples.append({
                        'path': sample_path,
                        'user_id': user_id
                    })
                    
            elif isinstance(sample, str):
                # 字符串格式：可能是目录名（如ID_1）或文件路径
                sample_path = self.data_root / sample
                
                if sample_path.is_dir():
                    # 是目录，扫描其中的图像文件
                    user_id = sample  # 使用目录名作为用户ID
                    
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
                    print(f"⚠️ 路径不存在: {sample_path}")
            else:
                print(f"❌ 未知sample格式: {type(sample)} - {sample}")
                continue
        
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
