"""
微多普勒潜在编码数据集类
用于加载预编码的VA-VAE latent和对应的用户标签
"""

import os
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file
import numpy as np
from pathlib import Path


class MicroDopplerLatentDataset(Dataset):
    """
    微多普勒潜在编码数据集
    加载预编码的latent和对应的用户类别标签
    """
    
    def __init__(self, data_path, latent_norm=True, latent_multiplier=1.0):
        """
        Args:
            data_path: 包含latent文件的目录路径
            latent_norm: 是否对latent进行归一化
            latent_multiplier: latent缩放因子
        """
        self.data_path = Path(data_path)
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 获取所有latent文件
        self.latent_files = sorted(list(self.data_path.glob("*.safetensors")))
        
        if len(self.latent_files) == 0:
            raise ValueError(f"No safetensors files found in {data_path}")
        
        print(f"Found {len(self.latent_files)} latent files in {data_path}")
        
        # 预加载所有数据到内存（数据量不大）
        self.latents = []
        self.labels = []
        
        for file_path in self.latent_files:
            # 加载safetensors文件
            data = load_file(str(file_path))
            
            # 获取latent和label
            if 'latent' in data:
                latent = data['latent']
            elif 'z' in data:
                latent = data['z']
            else:
                raise KeyError(f"No 'latent' or 'z' key found in {file_path}")
            
            if 'label' in data:
                label = data['label']
            elif 'user_id' in data:
                label = data['user_id']
            elif 'class' in data:
                label = data['class']
            else:
                # 尝试从文件名解析用户ID
                # 假设文件名格式为: user{id}_{index}.safetensors
                filename = file_path.stem
                if 'user' in filename:
                    try:
                        user_id = int(filename.split('user')[1].split('_')[0])
                        label = torch.tensor(user_id, dtype=torch.long)
                    except:
                        raise ValueError(f"Cannot parse user_id from filename: {filename}")
                else:
                    raise KeyError(f"No label key found in {file_path} and cannot parse from filename")
            
            # 确保label是标量
            if isinstance(label, torch.Tensor):
                if label.numel() > 1:
                    label = label[0]
                label = label.long()
            else:
                label = torch.tensor(label, dtype=torch.long)
            
            # 应用归一化和缩放
            if self.latent_norm:
                # 标准化到[-1, 1]
                latent = (latent - latent.mean()) / (latent.std() + 1e-8)
            
            latent = latent * self.latent_multiplier
            
            self.latents.append(latent)
            self.labels.append(label)
        
        # 统计类别分布
        unique_labels = torch.stack(self.labels).unique()
        print(f"Dataset contains {len(unique_labels)} unique classes: {unique_labels.tolist()}")
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        """
        返回(latent, label)对
        latent: [C, H, W] 潜在编码
        label: 用户类别标签（0-30）
        """
        return self.latents[idx], self.labels[idx]


class MicroDopplerLatentDatasetFromMemory(Dataset):
    """
    从内存中的预编码数据创建数据集
    用于已经在内存中有latent数据的情况
    """
    
    def __init__(self, latents, labels, latent_norm=True, latent_multiplier=1.0):
        """
        Args:
            latents: 预编码的latent张量列表或numpy数组
            labels: 对应的标签列表或numpy数组
            latent_norm: 是否对latent进行归一化
            latent_multiplier: latent缩放因子
        """
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 转换为tensor列表
        if isinstance(latents, (list, tuple)):
            self.latents = [torch.tensor(l, dtype=torch.float32) if not isinstance(l, torch.Tensor) else l 
                           for l in latents]
        elif isinstance(latents, np.ndarray):
            self.latents = [torch.tensor(latents[i], dtype=torch.float32) for i in range(len(latents))]
        elif isinstance(latents, torch.Tensor):
            self.latents = [latents[i] for i in range(len(latents))]
        else:
            raise TypeError(f"Unsupported latents type: {type(latents)}")
        
        # 转换标签
        if isinstance(labels, (list, tuple)):
            self.labels = [torch.tensor(l, dtype=torch.long) if not isinstance(l, torch.Tensor) else l.long() 
                          for l in labels]
        elif isinstance(labels, (np.ndarray, torch.Tensor)):
            labels_tensor = torch.tensor(labels, dtype=torch.long) if isinstance(labels, np.ndarray) else labels.long()
            self.labels = [labels_tensor[i] for i in range(len(labels_tensor))]
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")
        
        assert len(self.latents) == len(self.labels), \
            f"Latents and labels must have same length, got {len(self.latents)} vs {len(self.labels)}"
        
        # 应用归一化和缩放
        if self.latent_norm:
            for i in range(len(self.latents)):
                latent = self.latents[i]
                self.latents[i] = (latent - latent.mean()) / (latent.std() + 1e-8)
        
        if self.latent_multiplier != 1.0:
            for i in range(len(self.latents)):
                self.latents[i] = self.latents[i] * self.latent_multiplier
        
        # 统计信息
        unique_labels = torch.stack(self.labels).unique()
        print(f"Memory dataset contains {len(self.latents)} samples with {len(unique_labels)} classes")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


def create_latent_dataloader(data_path, batch_size, num_workers=4, shuffle=True, 
                            latent_norm=True, latent_multiplier=1.0):
    """
    创建潜在编码数据加载器的便捷函数
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 加载线程数
        shuffle: 是否打乱
        latent_norm: 是否归一化
        latent_multiplier: 缩放因子
    
    Returns:
        DataLoader实例
    """
    from torch.utils.data import DataLoader
    
    dataset = MicroDopplerLatentDataset(
        data_path=data_path,
        latent_norm=latent_norm,
        latent_multiplier=latent_multiplier
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if shuffle else False
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据集加载
    print("Testing MicroDopplerLatentDataset...")
    
    # 测试路径（在Kaggle环境中使用）
    test_path = "/kaggle/input/microdoppler-latents/train"
    
    if os.path.exists(test_path):
        dataset = MicroDopplerLatentDataset(test_path)
        print(f"Dataset size: {len(dataset)}")
        
        # 测试获取一个样本
        latent, label = dataset[0]
        print(f"Latent shape: {latent.shape}")
        print(f"Label: {label}")
        print(f"Latent dtype: {latent.dtype}")
        print(f"Label dtype: {label.dtype}")
        
        # 测试DataLoader
        dataloader = create_latent_dataloader(
            test_path, 
            batch_size=4, 
            num_workers=2
        )
        
        for batch_idx, (latents, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: latents shape={latents.shape}, labels shape={labels.shape}")
            if batch_idx >= 2:
                break
    else:
        print(f"Test path {test_path} not found. This is expected in local environment.")
        print("Creating mock dataset for testing...")
        
        # 创建模拟数据测试
        mock_latents = [torch.randn(32, 16, 16) for _ in range(100)]
        mock_labels = [torch.tensor(i % 31) for i in range(100)]
        
        dataset = MicroDopplerLatentDatasetFromMemory(mock_latents, mock_labels)
        print(f"Mock dataset size: {len(dataset)}")
        
        latent, label = dataset[0]
        print(f"Mock latent shape: {latent.shape}")
        print(f"Mock label: {label}")
