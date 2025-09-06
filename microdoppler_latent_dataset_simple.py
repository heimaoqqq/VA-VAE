"""
微多普勒Latent数据集 - 完全匹配官方ImgLatentDataset格式
基于LightningDiT/datasets/img_latent_dataset.py
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from safetensors import safe_open


class MicroDopplerLatentDataset(Dataset):
    """
    完全匹配官方ImgLatentDataset的数据加载方式
    """
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        # 查找所有safetensors文件
        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        if not self.files:
            raise FileNotFoundError(f"No safetensors files found in {data_dir}")
        
        # 建立图像索引到文件的映射
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        # 加载或计算归一化统计
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        """建立全局索引到文件内索引的映射"""
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        """获取latent统计信息"""
        # 尝试加载缓存的统计文件
        latent_stats_cache_file = os.path.join(self.data_dir, "latent_stats.pt")
        if os.path.exists(latent_stats_cache_file):
            print(f"📊 加载缓存的latent统计: {latent_stats_cache_file}")
            latent_stats = torch.load(latent_stats_cache_file)
            return latent_stats['mean'], latent_stats['std']
        
        # 否则计算统计
        print(f"📊 计算latent统计...")
        latent_stats = self.compute_latent_stats()
        # 保存统计
        torch.save(latent_stats, latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        """计算channel-wise统计"""
        num_samples = min(1000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        
        for idx in tqdm(random_indices, desc="计算统计"):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        
        print(f"   Mean范围: {mean.min().item():.3f} ~ {mean.max().item():.3f}")
        print(f"   Std范围: {std.min().item():.3f} ~ {std.max().item():.3f}")
        
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            # 官方使用随机选择原始或翻转的latent，我们暂时只用原始的
            # tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            tensor_key = "latents"  # 我们没有做数据增强，所以latents和latents_flip相同
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        # Channel-wise归一化
        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        
        # 应用multiplier
        feature = feature * self.latent_multiplier
        
        # 移除batch维度
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        
        return feature, label
