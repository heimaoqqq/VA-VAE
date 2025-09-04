#!/usr/bin/env python3
"""
微多普勒数据加载器：支持平衡批次采样和原型学习
确保每个批次包含多个用户，用于有效的对比学习
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from pathlib import Path
import json
from PIL import Image
import random
from collections import defaultdict


class MicroDopplerLatentDataset(Dataset):
    """微多普勒Latent数据集"""
    
    def __init__(self, latent_dir, split='train'):
        """
        Args:
            latent_dir: latent数据目录，包含.pt文件
            split: 'train' 或 'val'
        """
        self.latent_dir = Path(latent_dir)
        self.split = split
        
        # 加载数据索引
        index_file = self.latent_dir / f'{split}_index.json'
        if not index_file.exists():
            raise FileNotFoundError(f"未找到索引文件: {index_file}")
        
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        
        # 构建样本列表
        self.samples = []
        self.user_to_indices = defaultdict(list)
        
        for idx, item in enumerate(self.index):
            self.samples.append({
                'path': self.latent_dir / item['path'],
                'user_id': item['user_id'],
                'user_idx': item['user_idx']  # 数字索引
            })
            self.user_to_indices[item['user_idx']].append(idx)
        
        self.num_users = len(self.user_to_indices)
        self.users = list(self.user_to_indices.keys())
        
        print(f"📊 {split}集统计:")
        print(f"   总样本数: {len(self.samples)}")
        print(f"   用户数: {self.num_users}")
        print(f"   平均样本/用户: {len(self.samples) / self.num_users:.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 加载latent
        latent = torch.load(item['path'], map_location='cpu')
        
        # 确保维度正确 [32, 16, 16]
        if latent.dim() == 4 and latent.size(0) == 1:
            latent = latent.squeeze(0)
        
        return latent, item['user_idx']


class BalancedBatchSampler(Sampler):
    """平衡批次采样器：确保每批包含多个用户"""
    
    def __init__(self, dataset, batch_size, num_users_per_batch=4, drop_last=True):
        """
        Args:
            dataset: MicroDopplerLatentDataset实例
            batch_size: 批次大小
            num_users_per_batch: 每批的用户数
            drop_last: 是否丢弃最后不完整批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_users_per_batch = min(num_users_per_batch, dataset.num_users)
        self.drop_last = drop_last
        
        # 每个用户的样本数
        self.samples_per_user = batch_size // self.num_users_per_batch
        
        # 计算批次数
        min_samples = min(len(indices) for indices in dataset.user_to_indices.values())
        self.num_batches = min_samples // self.samples_per_user
        
        if not drop_last:
            # 如果不丢弃最后批次，增加批次数
            self.num_batches = len(dataset) // batch_size
    
    def __iter__(self):
        """生成批次索引"""
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 随机选择用户
            selected_users = random.sample(self.dataset.users, self.num_users_per_batch)
            
            # 从每个用户采样
            for user_idx in selected_users:
                user_indices = self.dataset.user_to_indices[user_idx]
                sampled = random.sample(user_indices, min(self.samples_per_user, len(user_indices)))
                batch_indices.extend(sampled)
            
            # 打乱批次内顺序
            random.shuffle(batch_indices)
            
            # 确保批次大小正确
            if len(batch_indices) >= self.batch_size:
                yield batch_indices[:self.batch_size]
            elif not self.drop_last and len(batch_indices) > 0:
                yield batch_indices
    
    def __len__(self):
        return self.num_batches


def create_balanced_dataloader(latent_dir, batch_size=32, num_users_per_batch=4, 
                              split='train', shuffle=True, num_workers=2):
    """
    创建平衡数据加载器
    
    Args:
        latent_dir: latent数据目录
        batch_size: 批次大小
        num_users_per_batch: 每批用户数（用于对比学习）
        split: 数据集划分
        shuffle: 是否打乱（仅对val有效）
        num_workers: 工作进程数
    
    Returns:
        DataLoader实例
    """
    dataset = MicroDopplerLatentDataset(latent_dir, split)
    
    if split == 'train':
        # 训练集使用平衡采样
        sampler = BalancedBatchSampler(
            dataset, 
            batch_size=batch_size,
            num_users_per_batch=num_users_per_batch,
            drop_last=True
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # 验证集使用普通加载
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )


def prepare_latent_dataset(image_dir, vae_model, output_dir, split_file, device='cuda'):
    """
    准备latent数据集
    
    Args:
        image_dir: 原始图像目录
        vae_model: SimplifiedVAVAE模型
        output_dir: 输出目录
        split_file: 数据划分文件
        device: 设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据划分
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    # 用户ID到数字索引的映射
    all_users = sorted(set(
        user_id for split in split_data.values() 
        for user_id in split.keys()
    ))
    user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    
    # 处理每个split
    for split_name in ['train', 'val']:
        print(f"\n🔄 处理{split_name}集...")
        
        split_output = output_dir / split_name
        split_output.mkdir(exist_ok=True)
        
        index = []
        total = 0
        
        for user_id, img_paths in split_data[split_name].items():
            user_idx = user_to_idx[user_id]
            
            for img_path in img_paths:
                img_path = Path(img_path)
                if not img_path.exists():
                    continue
                
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256))
                
                # 转换为tensor
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 编码到latent
                with torch.no_grad():
                    latent = vae_model.encode(img_tensor)
                
                # 保存latent
                save_name = f"{user_id}_{total:05d}.pt"
                save_path = split_output / save_name
                torch.save(latent.cpu(), save_path)
                
                # 添加到索引
                index.append({
                    'path': f"{split_name}/{save_name}",
                    'user_id': user_id,
                    'user_idx': user_idx
                })
                
                total += 1
                
                if total % 100 == 0:
                    print(f"   已处理 {total} 样本...")
        
        # 保存索引
        index_file = output_dir / f'{split_name}_index.json'
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"✅ {split_name}集完成: {total}个样本")
    
    print(f"\n✅ Latent数据集准备完成: {output_dir}")
    return output_dir


if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试平衡批次采样器...")
    
    # 模拟数据集
    class DummyDataset:
        def __init__(self):
            self.num_users = 10
            self.users = list(range(10))
            self.user_to_indices = {
                i: list(range(i*15, (i+1)*15)) 
                for i in range(10)
            }
    
    dataset = DummyDataset()
    sampler = BalancedBatchSampler(dataset, batch_size=32, num_users_per_batch=4)
    
    print(f"数据集: {10}个用户，每个15个样本")
    print(f"批次设置: batch_size=32, num_users_per_batch=4")
    print(f"期望: 每批8个样本/用户")
    
    # 测试几个批次
    for i, batch in enumerate(sampler):
        if i >= 3:
            break
        print(f"\n批次{i}: {len(batch)}个样本")
        
        # 统计用户分布
        user_counts = defaultdict(int)
        for idx in batch:
            user_id = idx // 15
            user_counts[user_id] += 1
        
        print(f"  用户分布: {dict(user_counts)}")
    
    print("\n✅ 测试完成！")
