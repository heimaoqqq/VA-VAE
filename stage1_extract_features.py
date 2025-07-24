#!/usr/bin/env python3
"""
阶段1: 特征提取
使用预训练VA-VAE提取微多普勒图像的潜在特征
遵循LightningDiT原项目的extract_features.py实现
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
import sys
from pathlib import Path
from safetensors.torch import save_file
from tqdm import tqdm
import numpy as np

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.autoencoder import AutoencoderKL

# 导入我们的数据集
from minimal_micro_doppler_dataset import MicroDopplerDataset

def setup_distributed():
    """设置分布式训练"""
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        print(f"Rank {rank}/{world_size} initialized")
        return rank, device, world_size
    except:
        print("Failed to initialize DDP. Running in local mode.")
        return 0, 0, 1

def extract_latent_features(args):
    """
    提取潜在特征的主函数
    """
    print("🔄 开始提取微多普勒图像的潜在特征...")
    
    # 设置分布式
    rank, device, world_size = setup_distributed()
    torch.cuda.set_device(device)
    
    # 创建输出目录
    output_dir = Path(args.output_path)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_dir}")
    
    # 加载预训练VA-VAE
    print(f"📥 加载预训练VA-VAE: {args.vavae_path}")
    vavae = AutoencoderKL(
        embed_dim=32,  # f16d32配置
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vavae_path,
        model_type='vavae'
    )
    vavae.eval()
    vavae.cuda(device)
    
    # 处理每个数据分割
    for split in ['train', 'val']:
        print(f"\n📊 处理 {split} 数据...")
        
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_dir=os.path.join(args.data_dir, split),
            split=split
        )
        
        # 分布式采样器
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank,
                shuffle=False
            )
        else:
            sampler = None
        
        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 存储特征
        all_latents = []
        all_user_ids = []
        all_indices = []
        
        print(f"开始提取 {len(dataset)} 个样本的特征...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank}")):
                images = batch['image'].cuda(device)  # (B, 3, 256, 256)
                user_ids = batch['user_id']  # (B,)
                
                # 提取潜在特征
                posterior = vavae.encode(images)
                latents = posterior.sample()  # (B, 32, 16, 16)
                
                # 收集数据
                all_latents.append(latents.cpu())
                all_user_ids.append(user_ids)
                all_indices.extend([
                    batch_idx * args.batch_size + i 
                    for i in range(len(user_ids))
                ])
                
                # 定期清理GPU内存
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
        
        # 合并所有特征
        if all_latents:
            latents_tensor = torch.cat(all_latents, dim=0)  # (N, 32, 16, 16)
            user_ids_tensor = torch.cat(all_user_ids, dim=0)  # (N,)
            indices_tensor = torch.tensor(all_indices)  # (N,)
            
            print(f"Rank {rank}: 提取了 {len(latents_tensor)} 个特征")
            print(f"特征形状: {latents_tensor.shape}")
            print(f"用户ID范围: [{user_ids_tensor.min()}, {user_ids_tensor.max()}]")
            
            # 保存特征 (使用safetensors格式，遵循原项目)
            output_file = output_dir / f"{split}_rank{rank}.safetensors"
            
            save_data = {
                'latents': latents_tensor,
                'user_ids': user_ids_tensor,
                'indices': indices_tensor,
                'metadata': {
                    'num_samples': len(latents_tensor),
                    'latent_shape': list(latents_tensor.shape[1:]),
                    'num_users': len(torch.unique(user_ids_tensor)),
                    'rank': rank,
                    'world_size': world_size
                }
            }
            
            # 转换为safetensors格式
            save_dict = {}
            for key, value in save_data.items():
                if key != 'metadata':
                    save_dict[key] = value
                else:
                    # 将metadata转换为tensor
                    for meta_key, meta_value in value.items():
                        if isinstance(meta_value, (int, float)):
                            save_dict[f'meta_{meta_key}'] = torch.tensor([meta_value])
                        elif isinstance(meta_value, list):
                            save_dict[f'meta_{meta_key}'] = torch.tensor(meta_value)
            
            save_file(save_dict, output_file)
            print(f"✅ 特征已保存到: {output_file}")
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()

    # 主进程合并所有rank的结果
    if rank == 0:
        print("\n🔄 合并所有rank的特征...")
        merge_features(output_dir, world_size)

        # 计算微多普勒数据的潜在特征统计信息
        print("\n📊 计算微多普勒潜在特征统计信息...")
        compute_micro_doppler_stats(output_dir)

    print("✅ 特征提取完成!")

def merge_features(output_dir, world_size):
    """合并所有rank的特征文件"""
    from safetensors import safe_open
    
    for split in ['train', 'val']:
        print(f"合并 {split} 特征...")
        
        all_latents = []
        all_user_ids = []
        all_indices = []
        
        # 读取所有rank的文件
        for rank in range(world_size):
            rank_file = output_dir / f"{split}_rank{rank}.safetensors"
            if rank_file.exists():
                with safe_open(rank_file, framework="pt", device="cpu") as f:
                    latents = f.get_tensor('latents')
                    user_ids = f.get_tensor('user_ids')
                    indices = f.get_tensor('indices')
                    
                    all_latents.append(latents)
                    all_user_ids.append(user_ids)
                    all_indices.append(indices)
                
                # 删除临时文件
                rank_file.unlink()
        
        if all_latents:
            # 合并数据
            merged_latents = torch.cat(all_latents, dim=0)
            merged_user_ids = torch.cat(all_user_ids, dim=0)
            merged_indices = torch.cat(all_indices, dim=0)
            
            # 按索引排序，保持原始顺序
            sort_idx = torch.argsort(merged_indices)
            merged_latents = merged_latents[sort_idx]
            merged_user_ids = merged_user_ids[sort_idx]
            
            print(f"合并后的 {split} 特征:")
            print(f"  样本数量: {len(merged_latents)}")
            print(f"  特征形状: {merged_latents.shape}")
            print(f"  用户数量: {len(torch.unique(merged_user_ids))}")
            
            # 保存最终文件
            final_file = output_dir / f"{split}.safetensors"
            save_dict = {
                'latents': merged_latents,
                'user_ids': merged_user_ids,
                'num_samples': torch.tensor([len(merged_latents)]),
                'num_users': torch.tensor([len(torch.unique(merged_user_ids))]),
                'latent_shape_h': torch.tensor([merged_latents.shape[2]]),
                'latent_shape_w': torch.tensor([merged_latents.shape[3]]),
                'latent_channels': torch.tensor([merged_latents.shape[1]])
            }
            
            save_file(save_dict, final_file)
            print(f"✅ 最终特征保存到: {final_file}")

def compute_micro_doppler_stats(output_dir):
    """
    计算微多普勒数据的潜在特征统计信息
    这对于正确的数据归一化很重要
    """
    from safetensors import safe_open

    print("计算微多普勒潜在特征的统计信息...")

    # 使用训练集计算统计信息
    train_file = output_dir / "train.safetensors"
    if not train_file.exists():
        print("❌ 训练集特征文件不存在")
        return

    # 加载训练集特征
    with safe_open(train_file, framework="pt", device="cpu") as f:
        latents = f.get_tensor('latents')  # (N, 32, 16, 16)

    print(f"计算 {len(latents)} 个样本的统计信息...")

    # 计算均值和标准差 (在空间维度上)
    # 保持通道维度，对batch和空间维度求统计
    mean = latents.mean(dim=[0, 2, 3], keepdim=True)  # (1, 32, 1, 1)
    std = latents.std(dim=[0, 2, 3], keepdim=True)    # (1, 32, 1, 1)

    print(f"潜在特征统计信息:")
    print(f"  均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  标准差范围: [{std.min():.4f}, {std.max():.4f}]")
    print(f"  全局均值: {mean.mean():.4f}")
    print(f"  全局标准差: {std.mean():.4f}")

    # 保存统计信息
    stats = {
        'mean': mean,
        'std': std,
        'num_samples': len(latents),
        'data_type': 'micro_doppler'
    }

    stats_file = output_dir / "latents_stats.pt"
    torch.save(stats, stats_file)
    print(f"✅ 统计信息保存到: {stats_file}")

    # 与ImageNet统计信息对比
    imagenet_stats_file = "/kaggle/working/pretrained/latents_stats.pt"
    if os.path.exists(imagenet_stats_file):
        print("\n📊 与ImageNet统计信息对比:")
        imagenet_stats = torch.load(imagenet_stats_file)
        imagenet_mean = imagenet_stats['mean']  # 保持原始形状
        imagenet_std = imagenet_stats['std']    # 保持原始形状

        print(f"  ImageNet统计形状 - 均值: {imagenet_mean.shape}, 标准差: {imagenet_std.shape}")
        print(f"  微多普勒统计形状 - 均值: {mean.shape}, 标准差: {std.shape}")
        print(f"  ImageNet - 全局均值: {imagenet_mean.mean():.4f}, 全局标准差: {imagenet_std.mean():.4f}")
        print(f"  微多普勒 - 全局均值: {mean.mean():.4f}, 全局标准差: {std.mean():.4f}")

        # 计算差异
        mean_diff = abs(mean.mean() - imagenet_mean.mean())
        std_diff = abs(std.mean() - imagenet_std.mean())

        if mean_diff > 0.5 or std_diff > 0.5:
            print("⚠️  统计信息差异较大，使用微多普勒自己的统计信息")
            recommendation = "micro_doppler"
        else:
            print("✅ 统计信息相近，可以选择使用ImageNet统计信息")
            recommendation = "imagenet"

            # 如果相近，提供ImageNet统计信息作为选项
            imagenet_stats_copy = output_dir / "latents_stats_imagenet.pt"
            torch.save(imagenet_stats, imagenet_stats_copy)
            print(f"📋 ImageNet统计信息副本: {imagenet_stats_copy}")

        # 保存推荐信息
        recommendation_file = output_dir / "stats_recommendation.txt"
        with open(recommendation_file, 'w') as f:
            f.write(f"Recommendation: {recommendation}\n")
            f.write(f"Mean difference: {mean_diff:.4f}\n")
            f.write(f"Std difference: {std_diff:.4f}\n")
        print(f"📝 推荐信息保存到: {recommendation_file}")

def main():
    parser = argparse.ArgumentParser(description='提取微多普勒图像的潜在特征')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='微多普勒数据目录 (包含train和val子目录)')
    parser.add_argument('--vavae_path', type=str, required=True,
                       help='预训练VA-VAE模型路径')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出特征的目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print("🎯 微多普勒特征提取 - 阶段1")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"VA-VAE模型: {args.vavae_path}")
    print(f"输出目录: {args.output_path}")
    print(f"批次大小: {args.batch_size}")
    
    extract_latent_features(args)

if __name__ == "__main__":
    main()
