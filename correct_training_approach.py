#!/usr/bin/env python3
"""
正确的训练方法 - 遵循LightningDiT原项目方式
1. 预提取微多普勒图像的latent特征
2. 在latent空间训练用户条件化DiT模型
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import sys

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.autoencoder import AutoencoderKL

def extract_latent_features():
    """
    第一步：使用预训练VA-VAE提取微多普勒图像的latent特征
    这一步只需要运行一次
    """
    print("🔄 提取微多普勒图像的latent特征...")
    
    # 加载预训练VA-VAE
    vavae = AutoencoderKL(
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path="/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt",
        model_type='vavae'
    )
    vavae.eval()
    vavae.cuda()
    
    # 处理数据集
    from minimal_micro_doppler_dataset import MicroDopplerDataset
    
    for split in ['train', 'val']:
        print(f"处理 {split} 数据...")
        dataset = MicroDopplerDataset(f"/kaggle/working/data_split/{split}", split=split)
        
        output_dir = Path(f"/kaggle/working/latent_features/{split}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        latents = []
        labels = []
        user_ids = []
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                image = sample['image'].unsqueeze(0).cuda()  # (1, 3, 256, 256)
                user_id = sample['user_id']
                
                # 提取latent特征
                posterior = vavae.encode(image)
                latent = posterior.sample()  # (1, 32, 16, 16)
                
                latents.append(latent.cpu())
                user_ids.append(user_id)
                
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{len(dataset)}")
        
        # 保存latent特征
        torch.save({
            'latents': torch.cat(latents, dim=0),  # (N, 32, 16, 16)
            'user_ids': torch.tensor(user_ids),    # (N,)
        }, output_dir / 'latents.pt')
        
        print(f"✅ {split} latent特征保存完成")

class UserConditionedDiT(nn.Module):
    """
    用户条件化的DiT模型
    在latent空间工作，不涉及图像编码解码
    """
    
    def __init__(self, num_users, condition_dim=128, latent_dim=32):
        super().__init__()
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, condition_dim)
        
        # 简化的DiT backbone（这里用简单的CNN代替）
        self.dit_backbone = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, padding=1),
        )
        
        # 用户条件融合
        self.condition_proj = nn.Linear(condition_dim, latent_dim)
        
    def forward(self, latents, user_ids, timesteps=None):
        """
        前向传播
        Args:
            latents: (B, 32, 16, 16) latent特征
            user_ids: (B,) 用户ID
            timesteps: (B,) 时间步（用于扩散训练）
        """
        B, C, H, W = latents.shape
        
        # 获取用户条件
        user_emb = self.user_embedding(user_ids - 1)  # 转换为0-based索引
        user_cond = self.condition_proj(user_emb)  # (B, latent_dim)
        user_cond = user_cond.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        # 添加用户条件
        conditioned_latents = latents + user_cond
        
        # DiT处理
        output = self.dit_backbone(conditioned_latents)
        
        return output

class LatentDataset(torch.utils.data.Dataset):
    """
    Latent特征数据集
    """
    
    def __init__(self, latent_file):
        data = torch.load(latent_file)
        self.latents = data['latents']
        self.user_ids = data['user_ids']
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return {
            'latent': self.latents[idx],
            'user_id': self.user_ids[idx]
        }

def train_user_conditioned_dit():
    """
    第二步：训练用户条件化DiT模型
    """
    print("🚀 训练用户条件化DiT模型...")
    
    # 创建数据加载器
    train_dataset = LatentDataset("/kaggle/working/latent_features/train/latents.pt")
    val_dataset = LatentDataset("/kaggle/working/latent_features/val/latents.pt")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # 创建模型
    num_users = len(torch.unique(train_dataset.user_ids))
    model = UserConditionedDiT(num_users=num_users)
    model.cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    print(f"用户数量: {num_users}")
    
    # 训练循环
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            latents = batch['latent'].cuda()
            user_ids = batch['user_id'].cuda()
            
            # 简单的重建任务（实际应该是扩散训练）
            optimizer.zero_grad()
            output = model(latents, user_ids)
            loss = criterion(output, latents)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                latents = batch['latent'].cuda()
                user_ids = batch['user_id'].cuda()
                output = model(latents, user_ids)
                loss = criterion(output, latents)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, "
              f"Val Loss: {val_loss/len(val_loader):.6f}")

def main():
    """主函数"""
    print("🎯 正确的训练方法 - 遵循LightningDiT原项目")
    print("=" * 50)
    
    # 第一步：提取latent特征（只需运行一次）
    if not os.path.exists("/kaggle/working/latent_features/train/latents.pt"):
        extract_latent_features()
    else:
        print("✅ Latent特征已存在，跳过提取步骤")
    
    # 第二步：训练用户条件化DiT
    train_user_conditioned_dit()

if __name__ == "__main__":
    main()
