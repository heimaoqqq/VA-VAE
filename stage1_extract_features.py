#!/usr/bin/env python3
"""
阶段1: 特征提取
基于LightningDiT原项目的extract_features.py
使用预训练VA-VAE提取微多普勒图像的潜在特征
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 导入LightningDiT组件
import sys
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

class MicroDopplerDataset(torch.utils.data.Dataset):
    """微多普勒数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 收集所有图像文件和用户ID
        self.samples = []
        for user_dir in sorted(self.data_dir.iterdir()):
            if user_dir.is_dir() and user_dir.name.startswith('user'):
                user_id = int(user_dir.name.replace('user', ''))
                for img_file in user_dir.glob('*.png'):
                    self.samples.append((str(img_file), user_id))
        
        print(f"加载了 {len(self.samples)} 个微多普勒样本")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, user_id = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, user_id

def main(args):
    """
    基于原项目extract_features.py的特征提取主函数
    """
    assert torch.cuda.is_available(), "特征提取需要至少一个GPU"

    # 设置分布式训练 (参考原项目)
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"启动 rank={rank}, seed={seed}, world_size={world_size}")
    except:
        print("分布式初始化失败，使用本地模式")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # 设置输出目录
    output_dir = Path(args.output_path)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 创建VA-VAE模型 (VA-VAE在初始化时已经设置为eval模式并移到GPU)
    tokenizer = VA_VAE(args.vavae_config)

    # 数据预处理 (与原项目一致)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 处理训练集和验证集
    for split in ['train', 'val']:
        split_dir = Path(args.data_dir) / split
        if not split_dir.exists():
            print(f"跳过不存在的分割: {split}")
            continue
            
        print(f"\n📊 处理 {split} 数据...")
        
        # 创建数据集
        dataset = MicroDopplerDataset(split_dir, transform=transform)
        
        # 创建分布式采样器
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            sampler = None
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        # 提取特征
        all_latents = []
        all_user_ids = []
        
        with torch.no_grad():
            for batch_idx, (images, user_ids) in enumerate(tqdm(
                dataloader, 
                desc=f"Rank {rank}",
                disable=(rank != 0)
            )):
                images = images.cuda()
                
                # 使用VA-VAE编码 (使用正确的方法)
                latents = tokenizer.encode_images(images)  # (B, 32, 16, 16)
                
                all_latents.append(latents.cpu())
                all_user_ids.extend(user_ids.tolist())
        
        # 合并所有特征
        if all_latents:
            latents_tensor = torch.cat(all_latents, dim=0)
            user_ids_tensor = torch.tensor(all_user_ids, dtype=torch.long)
            
            print(f"Rank {rank}: 提取了 {len(latents_tensor)} 个特征")
            print(f"特征形状: {latents_tensor.shape}")
            print(f"用户ID范围: [{min(all_user_ids)}, {max(all_user_ids)}]")
            
            # 保存特征 (参考原项目的safetensors格式)
            save_dict = {
                'latents': latents_tensor,
                'user_ids': user_ids_tensor,
                'num_samples': torch.tensor(len(latents_tensor)),
                'num_users': torch.tensor(len(set(all_user_ids))),
            }
            
            # 保存到rank特定的文件
            rank_file = output_dir / f"{split}_rank{rank:02d}.safetensors"
            save_file(save_dict, rank_file)
            print(f"✅ 特征已保存到: {rank_file}")

    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    # 主进程合并所有rank的结果
    if rank == 0:
        print("\n🔄 合并所有rank的特征...")
        merge_features(output_dir, world_size)
        
        # 计算统计信息 (参考原项目)
        print("\n📊 计算潜在特征统计信息...")
        compute_latent_stats(output_dir)
    
    print("✅ 特征提取完成!")

def merge_features(output_dir, world_size):
    """合并所有rank的特征文件"""
    for split in ['train', 'val']:
        all_latents = []
        all_user_ids = []
        
        # 收集所有rank的数据
        for rank in range(world_size):
            rank_file = output_dir / f"{split}_rank{rank:02d}.safetensors"
            if rank_file.exists():
                from safetensors import safe_open
                with safe_open(rank_file, framework="pt", device="cpu") as f:
                    latents = f.get_tensor('latents')
                    user_ids = f.get_tensor('user_ids')
                    all_latents.append(latents)
                    all_user_ids.append(user_ids)
                
                # 删除临时文件
                rank_file.unlink()
        
        if all_latents:
            # 合并数据
            merged_latents = torch.cat(all_latents, dim=0)
            merged_user_ids = torch.cat(all_user_ids, dim=0)
            
            print(f"合并后的 {split} 特征:")
            print(f"  样本数量: {len(merged_latents)}")
            print(f"  特征形状: {merged_latents.shape}")
            print(f"  用户数量: {len(set(merged_user_ids.tolist()))}")
            
            # 保存最终文件
            save_dict = {
                'latents': merged_latents,
                'user_ids': merged_user_ids,
                'num_samples': torch.tensor(len(merged_latents)),
                'num_users': torch.tensor(len(set(merged_user_ids.tolist()))),
            }
            
            final_file = output_dir / f"{split}.safetensors"
            save_file(save_dict, final_file)
            print(f"✅ 最终特征保存到: {final_file}")

def compute_latent_stats(output_dir):
    """计算潜在特征统计信息 (参考原项目)"""
    from safetensors import safe_open
    
    train_file = output_dir / "train.safetensors"
    if not train_file.exists():
        print("❌ 训练集特征文件不存在")
        return
    
    # 加载训练集特征
    with safe_open(train_file, framework="pt", device="cpu") as f:
        latents = f.get_tensor('latents')  # (N, 32, 16, 16)
    
    print(f"计算 {len(latents)} 个样本的统计信息...")
    
    # 计算统计信息 (参考原项目方式)
    mean = latents.mean(dim=[0, 2, 3], keepdim=True)  # (1, 32, 1, 1)
    std = latents.std(dim=[0, 2, 3], keepdim=True)    # (1, 32, 1, 1)
    
    print(f"潜在特征统计信息:")
    print(f"  均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  标准差范围: [{std.min():.4f}, {std.max():.4f}]")
    
    # 保存统计信息 (与原项目格式一致)
    stats = {
        'mean': mean,
        'std': std
    }
    
    stats_file = output_dir / "latents_stats.pt"
    torch.save(stats, stats_file)
    print(f"✅ 统计信息保存到: {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='微多普勒特征提取')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--vavae_config', type=str, required=True, help='VA-VAE配置文件')
    parser.add_argument('--output_path', type=str, required=True, help='输出路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    main(args)
