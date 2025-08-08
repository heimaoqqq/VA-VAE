#!/usr/bin/env python3
"""
步骤4: micro-Doppler数据适配器
将micro-Doppler时频图像适配为LightningDiT的输入格式
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from typing import Dict, List, Optional, Tuple
import pandas as pd

class MicroDopplerDataset(Dataset):
    """micro-Doppler数据集类"""
    
    def __init__(
        self,
        data_dir: str,
        user_labels: Dict[str, int],
        image_size: int = 256,
        split: str = "train",
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.user_labels = user_labels
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        # 加载图像路径和标签
        self.samples = self._load_samples()
        print(f"✅ 加载{split}数据: {len(self.samples)}张图像，{len(set(self.user_labels.values()))}个用户")
    
    def _load_samples(self) -> List[Dict]:
        """加载样本数据"""
        samples = []
        
        for image_path in self.data_dir.rglob("*.png"):  # 假设图像是PNG格式
            # 从文件名或目录结构提取用户ID
            # 这里需要根据您的数据组织方式调整
            user_id = self._extract_user_id(image_path)
            
            if user_id in self.user_labels:
                samples.append({
                    'image_path': image_path,
                    'user_id': user_id,
                    'user_class': self.user_labels[user_id]
                })
        
        return samples
    
    def _extract_user_id(self, image_path: Path) -> str:
        """从文件路径提取用户ID"""
        # 示例实现，需要根据您的数据结构调整
        # 假设文件名格式为: user001_sample123.png
        filename = image_path.stem
        if '_' in filename:
            return filename.split('_')[0]
        else:
            # 假设按目录组织: data_dir/user001/sample123.png
            return image_path.parent.name
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path'])
        
        # 转换为RGB（如果是灰度图）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 转换为张量
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # 归一化到[-1, 1]
        image_tensor = (image_tensor / 255.0) * 2.0 - 1.0
        
        # 应用变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            'image': image_tensor,
            'user_id': sample['user_id'],
            'user_class': sample['user_class'],
            'image_path': str(sample['image_path'])
        }

class UserConditionEncoder(nn.Module):
    """用户条件编码器"""
    
    def __init__(
        self,
        num_users: int,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 特征处理网络
        self.feature_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.02)
    
    def forward(self, user_classes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_classes: (batch_size,) 用户类别ID
        Returns:
            user_features: (batch_size, embed_dim) 用户特征向量
        """
        # 获取用户嵌入
        user_embed = self.user_embedding(user_classes)
        
        # 特征处理
        user_features = self.feature_net(user_embed)
        
        return user_features

class MicroDopplerDataModule:
    """数据模块"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 2,
        image_size: int = 256,
        val_split: float = 0.2
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        
        # 自动发现用户标签
        self.user_labels = self._discover_users()
        self.num_users = len(self.user_labels)
        
        print(f"📊 发现用户: {self.num_users}个")
        for user_id, class_idx in self.user_labels.items():
            print(f"   - {user_id}: 类别{class_idx}")
    
    def _discover_users(self) -> Dict[str, int]:
        """自动发现数据中的用户"""
        user_ids = set()
        data_path = Path(self.data_dir)
        
        # 从文件名或目录结构提取用户ID
        for image_path in data_path.rglob("*.png"):
            # 这里需要根据您的数据结构调整
            if '_' in image_path.stem:
                user_id = image_path.stem.split('_')[0]
            else:
                user_id = image_path.parent.name
            user_ids.add(user_id)
        
        # 分配类别索引
        user_labels = {user_id: idx for idx, user_id in enumerate(sorted(user_ids))}
        return user_labels
    
    def setup(self):
        """设置数据集"""
        # 创建完整数据集
        full_dataset = MicroDopplerDataset(
            data_dir=self.data_dir,
            user_labels=self.user_labels,
            image_size=self.image_size,
            split="full"
        )
        
        # 分割训练/验证集
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 数据分割: 训练{train_size}张，验证{val_size}张")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

def test_data_loading(data_dir: str):
    """测试数据加载"""
    print("🔍 测试micro-Doppler数据加载...")
    
    try:
        # 创建数据模块
        data_module = MicroDopplerDataModule(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0,  # 测试时使用0避免多进程问题
        )
        
        # 设置数据
        data_module.setup()
        
        # 测试训练数据加载器
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✅ 训练batch加载成功:")
        print(f"   - 图像形状: {batch['image'].shape}")
        print(f"   - 用户类别: {batch['user_class']}")
        print(f"   - 用户ID: {batch['user_id']}")
        
        # 测试条件编码器
        condition_encoder = UserConditionEncoder(
            num_users=data_module.num_users,
            embed_dim=768
        )
        
        user_features = condition_encoder(batch['user_class'])
        print(f"✅ 用户条件编码成功: {user_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_config_template():
    """创建配置文件模板"""
    config = {
        'model': {
            'target': 'ConditionalDiT',
            'params': {
                'model': 'DiT-XL/1',
                'num_classes': 10,  # 用户数量，需要根据实际调整
                'condition_dim': 768,
                'frozen_backbone': True,
                'dropout': 0.15
            }
        },
        'data': {
            'target': 'MicroDopplerDataModule',
            'params': {
                'data_dir': '/path/to/microdoppler/data',  # 需要修改
                'batch_size': 2,
                'num_workers': 2,
                'image_size': 256,
                'val_split': 0.2
            }
        },
        'trainer': {
            'precision': 'bf16-mixed',
            'max_epochs': 50,
            'check_val_every_n_epoch': 2,
            'gradient_clip_val': 0.5,
            'accumulate_grad_batches': 4,
            'log_every_n_steps': 50
        },
        'optimizer': {
            'target': 'torch.optim.AdamW',
            'params': {
                'lr': 5e-6,
                'weight_decay': 1e-3,
                'betas': [0.9, 0.95]
            }
        },
        'scheduler': {
            'target': 'torch.optim.lr_scheduler.CosineAnnealingLR',
            'params': {
                'T_max': 50,
                'eta_min': 1e-7
            }
        }
    }
    
    return config

def main():
    """主函数"""
    print("🚀 步骤4: micro-Doppler数据适配")
    print("="*60)
    
    # 创建配置模板
    config = create_config_template()
    config_path = Path("configs/microdoppler_finetune.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"✅ 配置模板已创建: {config_path}")
    
    # 测试数据加载（需要提供实际数据目录）
    data_dir = "/path/to/microdoppler/data"  # 需要修改为实际路径
    if Path(data_dir).exists():
        if test_data_loading(data_dir):
            print("✅ 数据加载测试通过")
        else:
            print("❌ 数据加载测试失败")
    else:
        print(f"ℹ️ 数据目录不存在: {data_dir}")
        print("ℹ️ 请修改配置文件中的data_dir路径")
    
    print("\n" + "="*60)
    print("✅ micro-Doppler适配器创建完成")
    print("📝 下一步: 修改配置文件中的数据路径，然后运行条件微调")

if __name__ == "__main__":
    main()
