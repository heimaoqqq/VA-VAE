"""
微多普勒Latent数据集 - 直接适配现有数据格式
完全兼容LightningDiT官方训练流程
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

class MicroDopplerLatentDataset(Dataset):
    """
    微多普勒Latent数据集
    直接读取我们的latent数据格式：[{'latent': tensor, 'user_id': int, 'original_idx': int}, ...]
    """
    
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = Path(data_dir)
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 查找latent文件
        latent_files = list(self.data_dir.glob("*_latents.pt"))
        if not latent_files:
            raise FileNotFoundError(f"未找到latent文件在 {data_dir}")
        
        # 加载所有latent数据
        print(f"📂 加载latent数据从 {data_dir}")
        self.latents = []
        self.labels = []
        
        for latent_file in latent_files:
            print(f"   加载 {latent_file.name}...")
            data = torch.load(latent_file, map_location='cpu', weights_only=False)
            
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # 我们的格式：[{'latent': tensor, 'user_id': int, ...}, ...]
                for item in data:
                    latent = item['latent']
                    if isinstance(latent, np.ndarray):
                        latent = torch.from_numpy(latent)
                    self.latents.append(latent)
                    
                    # 无条件生成，所有标签为0
                    self.labels.append(0)
            else:
                raise ValueError(f"不支持的数据格式: {type(data)}")
        
        print(f"✅ 加载完成：{len(self.latents)} 个latents")
        print(f"   Shape: {self.latents[0].shape}")
        
        # 计算归一化统计
        if self.latent_norm:
            self._compute_latent_stats()
    
    def _compute_latent_stats(self):
        """计算channel-wise归一化统计"""
        print("📊 计算channel-wise统计...")
        
        # 随机采样计算统计（节省内存）
        num_samples = min(1000, len(self.latents))
        indices = np.random.choice(len(self.latents), num_samples, replace=False)
        
        sample_latents = torch.stack([self.latents[i] for i in indices])
        
        # Channel-wise统计
        self._latent_mean = sample_latents.mean(dim=[0, 2, 3], keepdim=True)  # [1, 32, 1, 1]
        self._latent_std = sample_latents.std(dim=[0, 2, 3], keepdim=True)
        
        print(f"   Channel-wise mean范围: {self._latent_mean.min():.3f} ~ {self._latent_mean.max():.3f}")
        print(f"   Channel-wise std范围: {self._latent_std.min():.3f} ~ {self._latent_std.max():.3f}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx].clone()
        label = self.labels[idx]
        
        # Channel-wise归一化
        if self.latent_norm:
            latent = (latent - self._latent_mean) / self._latent_std
        
        # 应用multiplier
        latent = latent * self.latent_multiplier
        
        return latent, label
