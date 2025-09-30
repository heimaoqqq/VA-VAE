#!/usr/bin/env python3
"""
基于策略选择的数据集
严格保证 Support 和 Test 无重叠
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from smart_sample_selector import SmartSampleSelector


class StrategicDataset(Dataset):
    """
    使用智能策略选择的数据集
    严格分离 Support 和 Test
    """
    
    def __init__(self, data_dir, support_size, strategy, model, 
                 mode='support', seed=42, device='cuda', transform=None):
        """
        Args:
            data_dir: 数据目录
            support_size: 支持集大小
            strategy: 选择策略 ('random', 'confidence', 'diversity', 'uncertainty', 'hybrid')
            model: 分类器模型（用于智能选择）
            mode: 'support' 或 'test'
            seed: 随机种子
            device: 设备
            transform: 数据变换
        """
        self.data_dir = Path(data_dir)
        self.support_size = support_size
        self.strategy = strategy
        self.model = model
        self.mode = mode
        self.seed = seed
        self.device = device
        self.transform = transform
        self.samples = []
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据并根据策略选择"""
        selector = SmartSampleSelector(self.model, self.device)
        
        user_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for user_dir in user_dirs:
            user_name = user_dir.name
            if not user_name.startswith('ID_'):
                continue
            
            user_id = int(user_name.split('_')[1]) - 1  # ID_1 -> 0
            
            # 获取所有图像
            image_files = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
            if len(image_files) == 0:
                print(f"⚠️ No images found for {user_name}")
                continue
            
            # 排序确保可重复性
            image_files = sorted(image_files)
            
            # 使用策略选择支持集索引
            support_indices = selector.select_samples(
                image_files, user_id, self.support_size, self.strategy, self.seed
            )
            support_indices_set = set(support_indices)
            
            # 根据mode选择数据
            if self.mode == 'support':
                selected_files = [image_files[i] for i in support_indices]
                print(f"✓ {user_name}: Support set {len(selected_files)} samples ({self.strategy})")
            else:  # test
                selected_files = [img for i, img in enumerate(image_files) 
                                if i not in support_indices_set]
                print(f"✓ {user_name}: Test set {len(selected_files)} samples")
            
            # 添加到数据集
            for img_path in selected_files:
                self.samples.append({
                    'path': img_path,
                    'label': user_id,
                    'user_name': user_name
                })
        
        print(f"\n📊 {self.mode.capitalize()} dataset ({self.strategy}): {len(self.samples)} samples total")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['user_name'] 
