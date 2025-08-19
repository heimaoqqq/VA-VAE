#!/usr/bin/env python3
"""
步骤6: 将微多普勒图像通过VA-VAE编码成latent并保存为safetensors格式
这样训练时不需要加载VA-VAE，节省显存

包含集成的MicroDopplerLatentDataset类，用于后续DiT训练
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, DataLoader
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

def load_vavae_model(checkpoint_path, device='cuda'):
    """使用官方VA_VAE类加载模型，但使用自定义检查点路径"""
    print(f"📦 加载VA-VAE模型: {checkpoint_path}")
    
    # 创建自定义配置（不修改官方文件）
    import tempfile
    import yaml
    from omegaconf import OmegaConf
    
    # 基于官方配置创建自定义配置
    custom_config = {
        'ckpt_path': checkpoint_path,
        'model': {
            'base_learning_rate': 1.0e-04,
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'monitor': 'val/rec_loss',
                'embed_dim': 32,
                'use_vf': 'dinov2',
                'reverse_proj': True,
                'lossconfig': {
                    'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
                    'params': {
                        'disc_start': 1,
                        'kl_weight': 1.0e-06,
                        'disc_weight': 0.5,
                        'vf_weight': 0.1,
                        'adaptive_vf': True,
                        'vf_loss_type': 'combined_v3',
                        'distmat_margin': 0.25,
                        'cos_margin': 0.5
                    }
                },
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [16],
                    'dropout': 0.0
                }
            }
        }
    }
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_config, f, default_flow_style=False)
        temp_config_path = f.name
    
    try:
        # 使用官方VA_VAE类加载
        vae = VA_VAE(temp_config_path)
        vae.model = vae.model.to(device)
        print("✅ VA-VAE模型加载成功")
        return vae
    finally:
        # 清理临时文件
        import os
        os.unlink(temp_config_path)

def encode_images_to_latents(vae_model, image_paths, batch_size=4, device='cuda'):
    """批量编码图像为latents，使用官方VA_VAE"""
    
    # 使用官方的图像预处理（无水平翻转）
    transform = vae_model.img_transform(p_hflip=0.0, img_size=256)
    
    all_latents = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path)
            
            # 微多普勒时频图正确处理：保持单通道或转换为灰度
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ['L', 'RGB']:
                img = img.convert('L')  # 转为灰度
            
            # 如果是RGB但实际是灰度图（三通道相同），转换为单通道
            if img.mode == 'RGB':
                img_array = np.array(img)
                # 检查是否三通道相同（实际是灰度图）
                if np.allclose(img_array[:,:,0], img_array[:,:,1]) and np.allclose(img_array[:,:,1], img_array[:,:,2]):
                    img = img.convert('L')
                    
            # 确保最终为RGB格式以兼容VA-VAE（但保持原始强度信息）
            if img.mode == 'L':
                # 将单通道扩展为三通道，保持强度信息
                img_array = np.array(img)
                img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
                img = Image.fromarray(img_rgb)
            
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        # 堆叠成batch
        batch_tensor = torch.stack(batch_images)
        
        # 使用官方编码方法
        latents = vae_model.encode_images(batch_tensor)
        
        # VA-VAE不需要额外缩放因子（已在模型内部处理）
        # latents = latents * 0.18215  # 移除错误的SD缩放因子
        
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
    
    # 加载VA-VAE模型（使用微调后的检查点）
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    vae_model = load_vavae_model(vae_checkpoint, device)
    
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
            
            # 准备保存数据（使用与数据集类兼容的格式）
            save_dict = {
                'latents': latents,  # [B, C, H, W] batch格式
                'labels': torch.tensor(batch_labels, dtype=torch.long)  # [B] 标签
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

# ============================================================================
# 集成的MicroDopplerLatentDataset类（原microdoppler_latent_dataset.py内容）
# ============================================================================

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
            
            # 获取latent（支持多种键名）
            if 'latents' in data:  # 我们生成的格式
                latent = data['latents']
            elif 'latent' in data:
                latent = data['latent']
            elif 'z' in data:
                latent = data['z']
            else:
                raise KeyError(f"No 'latents', 'latent' or 'z' key found in {file_path}")
            
            # 获取标签（支持多种键名）
            if 'labels' in data:  # 我们生成的格式
                labels_batch = data['labels']
            elif 'label' in data:
                labels_batch = data['label']
            elif 'user_id' in data:
                labels_batch = data['user_id']
            elif 'class' in data:
                labels_batch = data['class']
            else:
                raise KeyError(f"No label key found in {file_path}")
            
            # 确保是batch格式，添加每个样本
            if latent.dim() == 4:  # [B, C, H, W]
                for i in range(latent.shape[0]):
                    sample_latent = latent[i]  # [C, H, W]
                    sample_label = labels_batch[i] if labels_batch.dim() > 0 else labels_batch
                    
                    # 应用归一化和缩放
                    if self.latent_norm:
                        sample_latent = (sample_latent - sample_latent.mean()) / (sample_latent.std() + 1e-8)
                    sample_latent = sample_latent * self.latent_multiplier
                    
                    self.latents.append(sample_latent)
                    self.labels.append(sample_label.long())
            else:
                raise ValueError(f"Expected 4D latent tensor, got {latent.dim()}D")
        
        # 统计类别分布
        unique_labels = torch.stack(self.labels).unique()
        print(f"Dataset contains {len(self.latents)} samples with {len(unique_labels)} unique classes: {unique_labels.tolist()}")
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        """
        返回(latent, label)对
        latent: [C, H, W] 潜在编码
        label: 用户类别标签（0-30）
        """
        return self.latents[idx], self.labels[idx]


def create_latent_dataloader(data_path, batch_size, num_workers=4, shuffle=True, 
                            latent_norm=True, latent_multiplier=1.0):
    """
    创建潜在编码数据加载器的便捷函数
    """
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
    create_latent_dataset()
