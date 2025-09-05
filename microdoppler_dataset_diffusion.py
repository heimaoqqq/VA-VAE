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
        for sample in split_data[split]:
            # 调试：检查sample的实际格式
            print(f"🔍 Debug sample type: {type(sample)}, content: {sample}")
            
            # 处理不同的JSON格式
            if isinstance(sample, dict):
                # 标准格式：{'path': '...', 'user_id': '...'}
                sample_path = self.data_root / sample['path']
                user_id = sample.get('user_id', 'unknown')
            elif isinstance(sample, str):
                # 字符串格式：直接是路径
                sample_path = self.data_root / sample
                # 从路径推断用户ID（假设格式为 ID_X/xxx.jpg）
                path_parts = Path(sample).parts
                user_id = path_parts[0] if path_parts else 'unknown'
            else:
                print(f"❌ 未知sample格式: {type(sample)} - {sample}")
                continue
                
            if sample_path.exists():
                self.samples.append({
                    'path': sample_path,
                    'user_id': user_id
                })
            else:
                print(f"⚠️ 文件不存在: {sample_path}")
        
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
