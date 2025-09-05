"""
使用HuggingFace diffusers训练无条件扩散模型
适配32通道VA-VAE latent空间
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from simplified_vavae import SimplifiedVAVAE
from microdoppler_data_loader import MicrodopplerDataset


class DiffusersTrainer:
    """基于diffusers的稳定训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化VAE
        print("🔧 加载VA-VAE...")
        self.vae = SimplifiedVAVAE(args.vae_checkpoint)
        self.vae.eval()
        
        # 初始化UNet - 使用diffusers的标准UNet
        print("🔧 初始化UNet2D模型...")
        self.unet = UNet2DModel(
            sample_size=16,  # latent空间大小 16x16
            in_channels=32,  # VA-VAE latent通道数
            out_channels=32,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D", 
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D", 
                "UpBlock2D",
            ),
        ).to(self.device)
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"  # 预测噪声
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 数据加载器
        print("📊 准备数据...")
        self.train_loader = self.prepare_dataloader()
        
        # 学习率调度
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=len(self.train_loader) * args.num_epochs
        )
        
        # 分布对齐参数
        self.use_distribution_alignment = False
        self.latent_mean = None
        self.latent_std = None
        
    def prepare_dataloader(self):
        """准备数据加载器"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        dataset = MicrodopplerDataset(
            root_dir=self.args.image_dir,
            split_file=self.args.split_file,
            split='train',
            transform=transform,
            return_user_id=False  # 无条件生成不需要用户ID
        )
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def detect_distribution_alignment(self, latents):
        """检测并配置分布对齐"""
        with torch.no_grad():
            mean = latents.mean().item()
            std = latents.std().item()
            
            print(f"\n📊 Latent分布分析:")
            print(f"   Mean: {mean:.6f}")
            print(f"   Std: {std:.6f}")
            print(f"   Range: [{latents.min().item():.2f}, {latents.max().item():.2f}]")
            
            # 如果std偏离1.0较多，启用分布对齐
            if abs(std - 1.0) > 0.3:
                print(f"✅ 启用分布对齐 (std={std:.3f} 偏离1.0)")
                self.use_distribution_alignment = True
                self.latent_mean = mean
                self.latent_std = std
                
                # 验证对齐效果
                aligned = (latents - mean) / std
                print(f"   对齐后: Mean={aligned.mean().item():.6f}, Std={aligned.std().item():.6f}")
            else:
                print(f"📊 分布正常，无需对齐 (std={std:.3f})")
                
    def normalize_latents(self, latents):
        """归一化latents"""
        if not self.use_distribution_alignment:
            return latents
        return (latents - self.latent_mean) / self.latent_std
    
    def denormalize_latents(self, latents):
        """反归一化latents"""
        if not self.use_distribution_alignment:
            return latents
        return latents * self.latent_std + self.latent_mean
    
    def train_step(self, batch):
        """单个训练步骤"""
        images = batch[0].to(self.device)  # 只取图像，忽略用户ID
        
        # VAE编码
        with torch.no_grad():
            latents = self.vae.encode(images)
            
            # 归一化（如果需要）
            if self.use_distribution_alignment:
                latents = self.normalize_latents(latents)
        
        # 采样噪声和时间步
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        )
        
        # 添加噪声
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.unet(noisy_latents, timesteps).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate_samples(self, num_samples=4, num_inference_steps=50):
        """生成样本"""
        self.unet.eval()
        
        # 初始噪声
        latents = torch.randn(
            num_samples, 32, 16, 16,  # 32通道，16x16
            device=self.device
        )
        
        # DDIM推理
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        ddim_scheduler.set_timesteps(num_inference_steps)
        
        for timestep in tqdm(ddim_scheduler.timesteps, desc="生成中"):
            timestep_batch = timestep.repeat(num_samples).to(self.device)
            
            # 预测噪声
            noise_pred = self.unet(latents, timestep_batch).sample
            
            # 去噪
            latents = ddim_scheduler.step(noise_pred, timestep, latents).prev_sample
        
        # 反归一化
        if self.use_distribution_alignment:
            latents = self.denormalize_latents(latents)
        
        # VAE解码
        images = self.vae.decode(latents)
        
        return images
    
    def save_samples(self, images, epoch, save_dir):
        """保存生成的样本 - 参考VA-VAE训练脚本的可视化方式"""
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建网格可视化 - 参考step4_train_vavae.py的风格
        num_samples = min(8, len(images))
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        fig.suptitle(f'Epoch {epoch} - 扩散生成样本')
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # 反归一化到[0,1]
            img_tensor = (images[i] + 1) / 2
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转换为numpy显示格式
            img_array = img_tensor.cpu().numpy()
            if img_array.shape[0] == 3:  # RGB
                img_array = np.transpose(img_array, (1, 2, 0))
            elif img_array.shape[0] == 1:  # 灰度
                img_array = img_array.squeeze(0)
                
            axes[i].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
            axes[i].set_title(f'样本 {i+1}')
            axes[i].axis('off')
        
        # 保存可视化图像
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch:03d}_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 同时保存单个图像文件
        for i, img_tensor in enumerate(images):
            # 反归一化到[0,1]
            img_tensor = (img_tensor + 1) / 2
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转换为PIL
            img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(2)
                img = Image.fromarray((img_array * 255).astype(np.uint8), 'L')
            else:
                img = Image.fromarray((img_array * 255).astype(np.uint8))
            
            img.save(f"{save_dir}/epoch_{epoch:03d}_sample_{i:02d}.png")
    
    def train(self):
        """主训练循环"""
        print(f"🚀 开始训练，设备: {self.device}")
        
        # 首次检测分布对齐
        first_batch = next(iter(self.train_loader))
        with torch.no_grad():
            sample_latents = self.vae.encode(first_batch[0][:4].to(self.device))
            self.detect_distribution_alignment(sample_latents)
        
        # 训练循环
        global_step = 0
        
        for epoch in range(self.args.num_epochs):
            self.unet.train()
            epoch_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for batch in pbar:
                # 训练步骤
                loss = self.train_step(batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # 记录
                epoch_losses.append(loss.item())
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}"
                })
                
                global_step += 1
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.args.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.unet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'use_distribution_alignment': self.use_distribution_alignment,
                    'latent_mean': self.latent_mean,
                    'latent_std': self.latent_std,
                }
                
                save_path = f"checkpoints/diffusion_epoch_{epoch+1:03d}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"💾 保存检查点: {save_path}")
            
            # 每轮生成样本 - 参考step4_train_vavae.py的做法
            print("🎨 生成样本...")
            sample_images = self.generate_samples(num_samples=8)
            self.save_samples(sample_images, epoch+1, "samples")
            print(f"✅ Epoch {epoch+1} 样本已保存到 samples/ 目录")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像数据目录')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                       help='VAE检查点路径')
    parser.add_argument('--split_file', type=str, required=True,
                       help='数据集划分文件')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='预热步数')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='保存频率')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = DiffusersTrainer(args)
    trainer.train()
    
    print("🎉 训练完成！")


if __name__ == "__main__":
    main()
