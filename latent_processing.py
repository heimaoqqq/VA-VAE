#!/usr/bin/env python3
"""
Latent预编码和数据集处理
集成预编码生成和数据集加载功能
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from simplified_vavae import SimplifiedVAVAE
from microdoppler_dataset_diffusion import MicrodopplerDataset


class PreEncodedLatentDataset(Dataset):
    """
    预编码latent数据集 - 直接加载latent而非图像
    显著加速训练过程
    """
    
    def __init__(self, latent_file, return_user_id=False):
        """
        Args:
            latent_file: 预编码latent文件路径 (.pt格式)
            return_user_id: 是否返回用户ID
        """
        self.latent_file = Path(latent_file)
        self.return_user_id = return_user_id
        
        # 加载预编码数据
        print(f"📊 加载预编码latent: {latent_file}")
        self.data = torch.load(latent_file, map_location='cpu', weights_only=False)
        
        print(f"✅ 加载完成: {len(self.data)}个latent样本")
        if len(self.data) > 0:
            sample_shape = self.data[0]['latent'].shape
            print(f"   Latent形状: {sample_shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 转换为tensor
        latent = torch.from_numpy(item['latent']).float()
        
        if self.return_user_id:
            return latent, item['user_id']
        else:
            return (latent,)  # 保持元组格式兼容
            

class MixedLatentDataset(Dataset):
    """
    混合数据集：优先使用预编码latent，fallback到图像编码
    """
    
    def __init__(self, latent_file=None, image_dataset=None, return_user_id=False):
        self.return_user_id = return_user_id
        
        # 尝试加载预编码数据
        if latent_file and Path(latent_file).exists():
            print(f"🚀 使用预编码latent数据集")
            self.use_preencoded = True
            self.latent_dataset = PreEncodedLatentDataset(
                latent_file, return_user_id=return_user_id
            )
        else:
            print(f"📊 使用图像数据集（实时编码）")
            self.use_preencoded = False
            self.image_dataset = image_dataset
            
    def __len__(self):
        if self.use_preencoded:
            return len(self.latent_dataset)
        else:
            return len(self.image_dataset)
    
    def __getitem__(self, idx):
        if self.use_preencoded:
            return self.latent_dataset[idx]
        else:
            return self.image_dataset[idx]


def precompute_latents(args):
    """预编码所有图像到latent空间"""
    
    # 初始化VAE
    print("🔧 加载VA-VAE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = SimplifiedVAVAE(args.vae_checkpoint)
    vae.to(device)
    vae.eval()
    
    # 输出目录
    latent_dir = Path(args.output_dir)
    latent_dir.mkdir(exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n📊 处理{split}集...")
        
        # 加载数据集
        try:
            dataset = MicrodopplerDataset(
                root_dir=args.image_dir,
                split_file=args.split_file,
                split=split,
                return_user_id=True,  # 需要用户ID
                image_size=256
            )
        except ValueError as e:
            print(f"⚠️ {split}集不存在，跳过: {e}")
            continue
        
        if len(dataset) == 0:
            print(f"⚠️ {split}集为空，跳过")
            continue
            
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,  # 保持顺序
            num_workers=4
        )
        
        latent_data = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"编码{split}")):
                images, user_ids = batch
                images = images.to(device)
                
                # 格式转换：BHWC -> BCHW
                if images.dim() == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)
                
                # VAE编码
                latents = vae.encode(images)
                
                # 保存每个样本的latent和元数据
                for i in range(latents.shape[0]):
                    latent_item = {
                        'latent': latents[i].cpu().numpy(),  # [32, 16, 16]
                        'user_id': user_ids[i],
                        'original_idx': batch_idx * args.batch_size + i
                    }
                    latent_data.append(latent_item)
        
        # 保存latent数据
        split_file = latent_dir / f"{split}_latents.pt"
        torch.save(latent_data, split_file)
        
        print(f"✅ {split}集已保存: {len(latent_data)}个latent -> {split_file}")
        print(f"   Latent形状: {latent_data[0]['latent'].shape}")
        
        # 计算统计信息
        all_latents = np.stack([item['latent'] for item in latent_data])
        mean_val = np.mean(all_latents)
        std_val = np.std(all_latents)
        
        print(f"   统计信息: Mean={mean_val:.6f}, Std={std_val:.6f}")
    
    print(f"\n🎉 所有latent已保存到: {latent_dir}")


def main():
    """命令行入口 - 预编码latent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="预编码图像到latent空间")
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像数据目录')
    parser.add_argument('--split_file', type=str, required=True,
                       help='数据划分JSON文件')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                       help='VA-VAE checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='./latents',
                       help='latent输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    
    args = parser.parse_args()
    precompute_latents(args)


if __name__ == '__main__':
    main()
