#!/usr/bin/env python3
"""
VA-VAE微调脚本 - 适配微多普勒数据
基于原项目的训练框架，针对小数据集优化
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

class MicroDopplerDataset(Dataset):
    """微多普勒数据集 - 用于VA-VAE微调"""
    
    def __init__(self, data_dir, transform=None, max_images_per_user=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = []
        self.user_labels = []
        
        # 收集所有用户目录下的图像
        user_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
        user_dirs.sort()
        
        for user_dir in user_dirs:
            user_id = user_dir.name
            images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
            
            # 限制每个用户的图像数量
            if max_images_per_user and len(images) > max_images_per_user:
                images = images[:max_images_per_user]
            
            self.image_files.extend(images)
            self.user_labels.extend([user_id] * len(images))
        
        print(f"📁 微调数据集: {len(self.image_files)} 张图像，来自 {len(set(self.user_labels))} 个用户")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        user_id = self.user_labels[idx]
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, user_id
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            # 返回黑色图像作为fallback
            black_image = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, user_id

class VAEFineTuner:
    """VA-VAE微调器"""
    
    def __init__(self, vae_model_path, device='cuda'):
        self.device = device
        self.vae_model_path = vae_model_path
        
        # 加载预训练VA-VAE
        print("🔧 加载预训练VA-VAE模型...")
        self.vae = self.load_vae_model()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_vae_model(self):
        """加载VA-VAE模型"""
        try:
            config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['ckpt_path'] = self.vae_model_path
            
            temp_config = "temp_finetune_vavae_config.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            vae = VA_VAE(config=temp_config)
            print("✅ VA-VAE模型加载成功")
            return vae
        except Exception as e:
            print(f"❌ VA-VAE模型加载失败: {e}")
            return None
    
    def freeze_encoder(self, freeze=True):
        """冻结/解冻编码器"""
        for param in self.vae.model.encoder.parameters():
            param.requires_grad = not freeze
        
        status = "冻结" if freeze else "解冻"
        print(f"🔒 编码器已{status}")
    
    def create_optimizer(self, learning_rate, freeze_encoder=False):
        """创建优化器"""
        if freeze_encoder:
            # 只优化解码器
            params = list(self.vae.model.decoder.parameters()) + list(self.vae.model.quant_conv.parameters()) + list(self.vae.model.post_quant_conv.parameters())
        else:
            # 优化全模型
            params = self.vae.model.parameters()
        
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
        return optimizer
    
    def compute_loss(self, images):
        """计算重建损失"""
        with torch.cuda.amp.autocast():
            # 编码
            latents = self.vae.model.encode(images).latent_dist.sample()
            
            # 解码
            reconstructed = self.vae.model.decode(latents).sample
            
            # 重建损失 (L1 + L2)
            l1_loss = F.l1_loss(reconstructed, images)
            l2_loss = F.mse_loss(reconstructed, images)
            recon_loss = l1_loss + 0.1 * l2_loss
            
            # KL散度损失
            kl_loss = torch.mean(torch.sum(latents ** 2, dim=[1, 2, 3]))
            
            # 总损失
            total_loss = recon_loss + 1e-6 * kl_loss
            
            return total_loss, recon_loss, kl_loss, reconstructed
    
    def train_epoch(self, dataloader, optimizer, epoch, freeze_encoder=False):
        """训练一个epoch"""
        self.vae.model.train()
        self.freeze_encoder(freeze_encoder)
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            loss, recon_loss, kl_loss, _ = self.compute_loss(images)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Recon': f'{recon_loss.item():.6f}',
                'KL': f'{kl_loss.item():.8f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, dataloader):
        """验证"""
        self.vae.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                loss, recon_loss, _, _ = self.compute_loss(images)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        
        return avg_loss, avg_recon_loss
    
    def save_checkpoint(self, epoch, optimizer, loss, save_path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.vae.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        print(f"💾 检查点已保存: {save_path}")
    
    def finetune(self, data_dir, output_dir, config):
        """执行微调"""
        print("🚀 开始VA-VAE微调")
        print("="*60)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建数据集
        dataset = MicroDopplerDataset(data_dir, transform=self.transform)
        
        # 划分训练和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
        
        print(f"📊 训练集: {len(train_dataset)} 张图像")
        print(f"📊 验证集: {len(val_dataset)} 张图像")
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        # 阶段1: 解码器微调
        if config.get('stage1_epochs', 0) > 0:
            print(f"\n🔥 阶段1: 解码器微调 ({config['stage1_epochs']} epochs)")
            optimizer = self.create_optimizer(config['stage1_lr'], freeze_encoder=True)
            
            for epoch in range(1, config['stage1_epochs'] + 1):
                train_loss, train_recon, train_kl = self.train_epoch(train_loader, optimizer, epoch, freeze_encoder=True)
                val_loss, val_recon = self.validate(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
                # 保存检查点
                if epoch % 2 == 0:
                    self.save_checkpoint(epoch, optimizer, train_loss, output_path / f"stage1_epoch_{epoch}.pt")
        
        # 阶段2: 全模型微调
        if config.get('stage2_epochs', 0) > 0:
            print(f"\n🔥 阶段2: 全模型微调 ({config['stage2_epochs']} epochs)")
            optimizer = self.create_optimizer(config['stage2_lr'], freeze_encoder=False)
            
            for epoch in range(1, config['stage2_epochs'] + 1):
                train_loss, train_recon, train_kl = self.train_epoch(train_loader, optimizer, epoch, freeze_encoder=False)
                val_loss, val_recon = self.validate(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
                # 保存检查点
                self.save_checkpoint(epoch, optimizer, train_loss, output_path / f"stage2_epoch_{epoch}.pt")
        
        # 保存最终模型
        final_model_path = output_path / "finetuned_vavae.pt"
        torch.save(self.vae.model.state_dict(), final_model_path)
        print(f"✅ 最终模型已保存: {final_model_path}")
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, output_path / "training_curves.png")
        
        return final_model_path

    def plot_training_curves(self, train_losses, val_losses, save_path):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VA-VAE Fine-tuning Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 训练曲线已保存: {save_path}")

def main():
    """主函数"""
    print("🎯 VA-VAE微调工具")
    print("="*50)
    
    # 配置
    config = {
        'batch_size': 4,
        'stage1_epochs': 2,  # 解码器微调
        'stage1_lr': 5e-5,
        'stage2_epochs': 3,  # 全模型微调
        'stage2_lr': 1e-5,
    }
    
    # 路径
    data_dir = "/kaggle/input/dataset"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    output_dir = "vavae_finetuned"
    
    print("📋 使用说明:")
    print("1. 确保数据在 /kaggle/input/dataset/ 目录")
    print("2. 确保预训练模型在 models/ 目录")
    print("3. 调用微调函数:")
    print("   from finetune_vavae import VAEFineTuner")
    print("   tuner = VAEFineTuner('models/vavae-imagenet256-f16d32-dinov2.pt')")
    print("   tuner.finetune('/kaggle/input/dataset', 'vavae_finetuned', config)")

if __name__ == "__main__":
    main()
