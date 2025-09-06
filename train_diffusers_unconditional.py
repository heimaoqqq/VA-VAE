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
from pathlib import Path

from simplified_vavae import SimplifiedVAVAE
from microdoppler_dataset_diffusion import MicrodopplerDataset
from latent_processing import MixedLatentDataset, PreEncodedLatentDataset


class DiffusersTrainer:
    """基于diffusers的稳定训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化VAE
        print("🔧 加载VA-VAE...")
        self.vae = SimplifiedVAVAE(args.vae_checkpoint)
        self.vae.to(self.device)  # 移动到CUDA设备
        self.vae.eval()
        
        # 初始化UNet - 增强架构提升特征学习能力
        print("🔧 初始化UNet2D模型...")
        self.unet = UNet2DModel(
            sample_size=16,  # latent空间大小 16x16
            in_channels=32,  # VA-VAE latent通道数
            out_channels=32,
            layers_per_block=3,  # 增加层数
            block_out_channels=(128, 256, 512, 768),  # 增强容量
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
        
        # 优化器 - 降低学习率提升细节学习
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.learning_rate * 0.5,  # 降低学习率
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 数据加载器
        print("📊 准备数据...")
        self.train_loader = self.prepare_dataloader('train')
        self.val_loader = self.prepare_dataloader('val')
        
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
        
    def prepare_dataloader(self, split='train'):
        """准备数据加载器 - 支持预编码latent和图像"""
        # 检查是否有预编码latent文件
        latent_file = None
        if hasattr(self.args, 'latent_dir') and self.args.latent_dir:
            latent_file = Path(self.args.latent_dir) / f"{split}_latents.pt"
            
        # 创建fallback图像数据集
        image_dataset = MicrodopplerDataset(
            root_dir=self.args.image_dir,
            split_file=self.args.split_file,
            split=split,
            transform=None,
            return_user_id=False,
            image_size=256
        )
        
        # 使用混合数据集（优先latent，fallback图像）
        dataset = MixedLatentDataset(
            latent_file=latent_file,
            image_dataset=image_dataset,
            return_user_id=False
        )
        
        # 检测数据集类型（只在训练集时设置）
        if split == 'train':
            self.use_preencoded_latents = hasattr(dataset, 'use_preencoded') and dataset.use_preencoded
            if self.use_preencoded_latents:
                print("🚀 使用预编码latent训练 - 显著加速！")
            else:
                print("📊 使用图像训练 - 实时编码")
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == 'train'),  # 只有训练集打乱
            num_workers=4 if not getattr(self, 'use_preencoded_latents', False) else 0,
            pin_memory=True
        )
    
    def detect_distribution_alignment(self, latents):
        """检测并配置分布对齐 - 使用channel-wise归一化"""
        with torch.no_grad():
            # Channel-wise统计（LightningDiT官方方法）
            # latents shape: [B, C=32, H=16, W=16]
            # 计算每个通道的mean和std
            self.latent_mean = latents.mean(dim=[0, 2, 3], keepdim=True)  # [1, 32, 1, 1]
            self.latent_std = latents.std(dim=[0, 2, 3], keepdim=True)    # [1, 32, 1, 1]
            
            # 全局统计用于显示
            global_mean = latents.mean().item()
            global_std = latents.std().item()
            
            print(f"\n📊 Latent分布分析:")
            print(f"   全局 Mean: {global_mean:.6f}")
            print(f"   全局 Std: {global_std:.6f}")
            print(f"   Range: [{latents.min().item():.2f}, {latents.max().item():.2f}]")
            print(f"🔧 使用Channel-wise归一化（LightningDiT官方方法）")
            print(f"   策略: 每个通道独立归一化，保持通道间相对关系")
            
            # 验证归一化效果
            normalized = (latents - self.latent_mean) / self.latent_std
            print(f"   归一化后: mean={normalized.mean().item():.6f}, std={normalized.std().item():.6f}")
                
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
        """单个训练步骤 - 支持预编码latent和图像"""
        data = batch[0].to(self.device)  # 可能是图像或latent
        
        if self.use_preencoded_latents:
            # 直接使用预编码latent
            latents = data  # 已经是latent格式 [B, 32, 16, 16]
        else:
            # VAE编码图像
            with torch.no_grad():
                # 格式转换：BHWC -> BCHW
                if data.dim() == 4 and data.shape[-1] == 3:  # BHWC格式
                    data = data.permute(0, 3, 1, 2)  # 转为BCHW
                
                latents = self.vae.encode(data)
        
        # Channel-wise归一化（保持语义空间）
        # 使用.to(self.device)确保在正确设备上
        latents = (latents - self.latent_mean.to(self.device)) / self.latent_std.to(self.device)
        
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
    
    def validate(self):
        """验证集评估"""
        self.unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.train_step(batch)
                val_losses.append(loss.item())  # 转换为Python标量
        
        return np.mean(val_losses)
    
    def generate_samples(self, num_samples=2, num_inference_steps=100):
        """生成样本"""
        self.unet.eval()
        
        # 初始噪声 - 始终从标准正态开始，训练时归一化确保一致性
        print(f"📊 从标准正态分布开始去噪: N(0, 1)")
        latents = torch.randn(
            num_samples, 32, 16, 16, device=self.device
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
        
        # Channel-wise反归一化到原始latent空间
        latents = latents * self.latent_std.to(self.device) + self.latent_mean.to(self.device)
        
        # VAE解码 - 匹配VA-VAE的调用方式
        images = self.vae.decode(latents)
        
        # 确保输出格式正确
        if images.dim() == 4 and images.shape[1] == 3:  # BCHW格式
            # 保持BCHW格式用于后续处理
            pass
        
        return images
    
    def save_samples(self, images, epoch, save_dir):
        """保存生成的样本 - 只生成一张8样本网格图"""
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建网格可视化 - 减少到2个样本节省内存
        num_samples = min(2, len(images))
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
        fig.suptitle(f'Epoch {epoch} - 扩散生成样本')
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # VA-VAE decode已经返回[0,1]范围，无需再归一化
            img_tensor = torch.clamp(images[i], 0, 1)
            
            # 转换为numpy显示格式
            img_array = img_tensor.cpu().numpy()
            if img_array.shape[0] == 3:  # RGB - BCHW格式
                img_array = np.transpose(img_array, (1, 2, 0))  # CHW -> HWC
            elif img_array.shape[0] == 1:  # 灰度
                img_array = img_array.squeeze(0)
                
            axes[i].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
            axes[i].set_title(f'样本 {i+1}')
            axes[i].axis('off')
        
        # 只保存网格可视化图像
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch:03d}_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """主训练循环"""
        print(f"🚀 开始训练，设备: {self.device}")
        
        # 首次检测分布对齐
        first_batch = next(iter(self.train_loader))
        with torch.no_grad():
            sample_data = first_batch[0][:4].to(self.device)
            
            if self.use_preencoded_latents:
                # 直接使用预编码latent
                sample_latents = sample_data
            else:
                # VAE编码图像
                # 格式转换：BHWC -> BCHW
                if sample_data.dim() == 4 and sample_data.shape[-1] == 3:  # BHWC格式
                    sample_data = sample_data.permute(0, 3, 1, 2)  # 转为BCHW
                
                sample_latents = self.vae.encode(sample_data)
            
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
            
            # 验证集评估
            val_loss = self.validate()
            print(f"📊 验证损失: {val_loss:.4f}")
            
            # 每轮生成样本 - 进一步减少内存使用
            print("🎨 生成样本...")
            sample_images = self.generate_samples(num_samples=2)
            self.save_samples(sample_images, epoch+1, "samples")
            print(f"✅ Epoch {epoch+1} 完成 - 训练损失: {avg_loss:.4f}, 验证损失: {val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像数据目录')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                       help='VAE检查点路径')
    parser.add_argument('--split_file', type=str, required=True,
                       help='数据集划分文件')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小 - 匹配VA-VAE默认值')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率 - 匹配VA-VAE Stage1')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='预热步数')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='保存频率')
    parser.add_argument('--latent_dir', type=str, default=None,
                       help='预编码latent目录（可选，使用后显著加速训练）')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = DiffusersTrainer(args)
    trainer.train()
    
    print("🎉 训练完成！")


if __name__ == "__main__":
    main()
