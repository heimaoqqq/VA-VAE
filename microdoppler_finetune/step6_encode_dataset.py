#!/usr/bin/env python3
"""
步骤6: 将微多普勒图像通过VA-VAE编码成latent并保存为safetensors格式
这样训练时不需要加载VA-VAE，节省显存
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from safetensors.torch import save_file
import sys
import gc

# 添加LightningDiT路径
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')
sys.path.append('/kaggle/working/LightningDiT')

# 导入LightningDiT的VA_VAE
try:
    from tokenizer.vavae import VA_VAE
except ImportError as e:
    print(f"导入VA_VAE失败: {e}")
    print("请确认LightningDiT路径正确")
    sys.exit(1)

def load_vavae_model(config_path, device='cuda'):
    """使用官方VA_VAE类加载模型"""
    print(f"📦 加载VA-VAE模型配置: {config_path}")
    
    # 使用官方VA_VAE类
    vae = VA_VAE(config_path)
    vae.model = vae.model.to(device)
    
    print("✅ VA-VAE模型加载成功")
    return vae

def encode_images_to_latents(vae_model, image_paths, batch_size=4, device='cuda'):
    """批量编码图像为latents，使用官方VA_VAE"""
    
    # 使用官方的图像预处理（无水平翻转）
    transform = vae_model.img_transform(p_hflip=0.0, img_size=256)
    
    all_latents = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        # 堆叠成batch
        batch_tensor = torch.stack(batch_images)
        
        # 使用官方编码方法
        latents = vae_model.encode_images(batch_tensor)
        
        # 应用缩放因子
        latents = latents * 0.18215
        
        all_latents.append(latents.cpu())
        
        # 清理显存
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    # 合并所有latents
    all_latents = torch.cat(all_latents, dim=0)
    return all_latents

def create_latent_dataset():
    """创建latent数据集"""
    
    # 路径设置
    data_root = Path('/kaggle/working/microdoppler_dataset') 
    output_dir = Path('/kaggle/working/latent_dataset')
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据划分信息（从正确的位置）
    split_file = Path('/kaggle/working/data_split/dataset_split.json')
    with open(split_file, 'r') as f:
        dataset_split = json.load(f)
    
    labels_file = Path('/kaggle/working/data_split/user_labels.json')
    with open(labels_file, 'r') as f:
        user_labels = json.load(f)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 加载VA-VAE模型
    vae_config = '/kaggle/working/VA-VAE/LightningDiT/tokenizer/configs/vavae_f16d32.yaml'
    vae_model = load_vavae_model(vae_config, device)
    
    # 分别处理训练集和验证集
    for split in ['train', 'val']:
        print(f"\n📊 处理{split}集...")
        
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # 收集所有图像路径和标签
        all_image_paths = []
        all_labels = []
        
        for user_key, image_paths in dataset_split[split].items():
            user_label = user_labels[user_key]
            all_image_paths.extend(image_paths)
            all_labels.extend([user_label] * len(image_paths))
        
        print(f"   总图像数: {len(all_image_paths)}")
        
        # 分批处理并保存
        images_per_file = 500  # 每个safetensors文件保存500个样本
        num_files = (len(all_image_paths) + images_per_file - 1) // images_per_file
        
        for file_idx in range(num_files):
            start_idx = file_idx * images_per_file
            end_idx = min((file_idx + 1) * images_per_file, len(all_image_paths))
            
            batch_paths = all_image_paths[start_idx:end_idx]
            batch_labels = all_labels[start_idx:end_idx]
            
            print(f"\n   处理文件 {file_idx+1}/{num_files} (样本 {start_idx}-{end_idx-1})")
            
            # 编码图像
            latents = encode_images_to_latents(vae_model, batch_paths, batch_size=8, device=device)
            
            # 准备保存数据
            save_dict = {
                'latents': latents,
                'labels': torch.tensor(batch_labels, dtype=torch.long)
            }
            
            # 保存为safetensors
            output_file = split_dir / f'{split}_{file_idx:04d}.safetensors'
            save_file(save_dict, str(output_file))
            print(f"   ✅ 保存到: {output_file}")
            
            # 清理内存
            del latents
            gc.collect()
            torch.cuda.empty_cache()
    
    # 计算并保存latent统计信息
    print("\n📈 计算latent统计信息...")
    
    train_dir = output_dir / 'train'
    all_latents = []
    
    for safe_file in sorted(train_dir.glob('*.safetensors')):
        from safetensors.torch import load_file
        data = load_file(str(safe_file))
        all_latents.append(data['latents'])
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # 计算均值和标准差
    latent_mean = all_latents.mean(dim=[0, 2, 3], keepdim=True)
    latent_std = all_latents.std(dim=[0, 2, 3], keepdim=True)
    
    stats = {
        'mean': latent_mean.squeeze(),
        'std': latent_std.squeeze()
    }
    
    # 保存统计信息
    stats_file = output_dir / 'train' / 'latents_stats.pt'
    torch.save(stats, str(stats_file))
    print(f"✅ 统计信息保存到: {stats_file}")
    
    # 创建数据集配置
    dataset_config = {
        'train_dir': str(output_dir / 'train'),
        'val_dir': str(output_dir / 'val'),
        'num_classes': 31,  # 31个用户
        'latent_channels': 32,
        'latent_size': 32,  # 256 / 8 = 32
        'stats_file': str(stats_file),
        'user_labels': user_labels
    }
    
    config_file = output_dir / 'dataset_config.json'
    with open(config_file, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    print(f"✅ 配置保存到: {config_file}")
    
    print("\n" + "="*60)
    print("🎉 Latent数据集创建完成!")
    print(f"   输出目录: {output_dir}")
    print(f"   训练样本: {dataset_split['statistics']['train_images']}")
    print(f"   验证样本: {dataset_split['statistics']['val_images']}")
    
    # 清理模型
    del vae_model
    torch.cuda.empty_cache()
    
    return output_dir

if __name__ == "__main__":
    create_latent_dataset()
