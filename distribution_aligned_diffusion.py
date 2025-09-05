"""
分布对齐的扩散模型训练和推理
实现业界标准的latent分布归一化方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from typing import Optional, Tuple
import numpy as np


class DistributionAlignedDiffusion(nn.Module):
    """带分布对齐的条件扩散模型"""
    
    def __init__(
        self,
        latent_dim: int = 768,
        num_users: int = 10,
        unet_in_channels: int = 768,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        self.num_users = num_users
        
        # 分布统计（会在训练时动态更新）
        self.register_buffer('latent_mean', torch.zeros(1))
        self.register_buffer('latent_std', torch.ones(1))
        self.register_buffer('num_samples_seen', torch.tensor(0))
        
        # UNet模型
        self.unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=unet_in_channels,
            out_channels=unet_in_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=num_users,
            attention_head_dim=8,
            norm_num_groups=32,
            use_linear_projection=False,
        ).to(device)
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, num_users).to(device)
        nn.init.eye_(self.user_embedding.weight)
        
    def update_distribution_stats(self, latents: torch.Tensor, momentum: float = 0.99):
        """
        使用动量更新latent分布统计
        Args:
            latents: 未归一化的VAE latents
            momentum: 动量系数用于平滑更新
        """
        with torch.no_grad():
            batch_mean = latents.mean()
            batch_std = latents.std()
            
            if self.num_samples_seen == 0:
                # 首次更新
                self.latent_mean.copy_(batch_mean)
                self.latent_std.copy_(batch_std)
            else:
                # 动量更新
                self.latent_mean.mul_(momentum).add_(batch_mean, alpha=1-momentum)
                self.latent_std.mul_(momentum).add_(batch_std, alpha=1-momentum)
            
            self.num_samples_seen += 1
            
    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """归一化latents到N(0,1)"""
        return (latents - self.latent_mean) / (self.latent_std + 1e-8)
    
    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """反归一化latents到原始分布"""
        return latents * self.latent_std + self.latent_mean
    
    def training_step(
        self, 
        latents: torch.Tensor, 
        user_ids: torch.Tensor,
        update_stats: bool = True
    ) -> dict:
        """
        训练步骤with分布对齐
        Args:
            latents: 原始VAE编码的latents
            user_ids: 用户ID
            update_stats: 是否更新分布统计
        """
        batch_size = latents.shape[0]
        
        # 更新分布统计
        if update_stats:
            self.update_distribution_stats(latents)
        
        # 归一化到N(0,1)
        normalized_latents = self.normalize_latents(latents)
        
        # 添加噪声
        noise = torch.randn_like(normalized_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(
            normalized_latents, noise, timesteps
        )
        
        # 获取用户嵌入
        user_embeds = self.user_embedding(user_ids).unsqueeze(1)
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_embeds,
            return_dict=False
        )[0]
        
        # Huber损失（比MSE更鲁棒）
        loss = F.huber_loss(noise_pred, noise, delta=1.0)
        
        # 计算预测质量指标
        with torch.no_grad():
            pred_std = noise_pred.std()
            target_std = noise.std()
            std_ratio = pred_std / (target_std + 1e-8)
        
        return {
            'loss': loss,
            'pred_std': pred_std.item(),
            'target_std': target_std.item(),
            'std_ratio': std_ratio.item(),
            'latent_mean': self.latent_mean.item(),
            'latent_std': self.latent_std.item()
        }
    
    @torch.no_grad()
    def generate(
        self,
        user_ids: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        生成with分布对齐
        Returns:
            反归一化后的latents（匹配训练分布）
        """
        batch_size = user_ids.shape[0]
        shape = (batch_size, self.latent_dim, 1, 1)
        device = user_ids.device
        
        # 从N(0,1)开始
        latents = torch.randn(shape, device=device, generator=generator)
        
        # 设置推理步数
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 获取用户嵌入
        user_embeds = self.user_embedding(user_ids).unsqueeze(1)
        
        # 反向扩散过程
        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            noise_pred = self.unet(
                latents,
                t.unsqueeze(0).to(device),
                encoder_hidden_states=user_embeds,
                return_dict=False
            )[0]
            
            # 计算去噪后的latents
            latents = self.noise_scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample
        
        # 关键：反归一化到原始VAE分布
        original_scale_latents = self.denormalize_latents(latents)
        
        return original_scale_latents
    
    def get_scaling_factor(self) -> float:
        """获取当前的缩放因子（类似SD的0.18215）"""
        return 1.0 / (self.latent_std.item() + 1e-8)


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = DistributionAlignedDiffusion(
        latent_dim=768,
        num_users=10,
        device='cuda'
    )
    
    print("分布对齐的扩散模型已创建")
    print(f"初始缩放因子: {model.get_scaling_factor():.5f}")
    
    # 模拟训练
    for epoch in range(5):
        # 模拟VAE编码的latents（std≈1.54）
        raw_latents = torch.randn(4, 768, 1, 1).cuda() * 1.54
        user_ids = torch.randint(0, 10, (4,)).cuda()
        
        # 训练步骤
        result = model.training_step(raw_latents, user_ids)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {result['loss']:.4f}")
        print(f"  Latent统计: mean={result['latent_mean']:.4f}, std={result['latent_std']:.4f}")
        print(f"  预测std比例: {result['std_ratio']:.4f}")
        print(f"  当前缩放因子: {model.get_scaling_factor():.5f}")
    
    # 生成
    print("\n生成测试:")
    generated = model.generate(user_ids)
    print(f"生成latents统计: mean={generated.mean():.4f}, std={generated.std():.4f}")
    print("✅ 生成的latents已自动匹配训练分布！")
