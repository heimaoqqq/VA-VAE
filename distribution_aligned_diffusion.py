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
        vae,
        num_users: int = 31,
        prototype_dim: int = 768,
        enable_alignment: bool = True,
        track_statistics: bool = True
    ):
        super().__init__()
        
        # VAE (冻结)
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.num_users = num_users
        self.prototype_dim = prototype_dim
        self.enable_alignment = enable_alignment
        self.track_statistics = track_statistics
        
        # 分布统计（会在训练时动态更新）
        self.register_buffer('latent_mean', torch.zeros(1))
        self.register_buffer('latent_std', torch.ones(1))
        self.register_buffer('n_samples', torch.tensor(0))
        
        # UNet模型 - 匹配VA-VAE的latent维度
        self.unet = UNet2DConditionModel(
            sample_size=16,  # VAE latent size
            in_channels=32,  # VAE latent channels
            out_channels=32,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=prototype_dim,
            attention_head_dim=8,
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        # 用户原型系统
        self.user_prototypes = nn.ParameterDict()
        for i in range(num_users):
            self.user_prototypes[str(i)] = nn.Parameter(torch.randn(prototype_dim))
        
        # 对比学习
        self.contrastive_loss = nn.MSELoss()
        
    def update_statistics(self, latents: torch.Tensor, momentum: float = 0.99):
        """
        使用动量更新latent分布统计
        Args:
            latents: 未归一化的VAE latents
            momentum: 动量系数用于平滑更新
        """
        with torch.no_grad():
            batch_mean = latents.mean()
            batch_std = latents.std()
            
            if self.n_samples == 0:
                # 首次更新
                self.latent_mean.copy_(batch_mean)
                self.latent_std.copy_(batch_std)
            else:
                # 动量更新
                self.latent_mean.mul_(momentum).add_(batch_mean, alpha=1-momentum)
                self.latent_std.mul_(momentum).add_(batch_std, alpha=1-momentum)
            
            self.n_samples += 1
            
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
        if update_stats and self.track_statistics:
            self.update_statistics(latents)
        
        # 归一化到N(0,1) (如果启用对齐)
        if self.enable_alignment:
            normalized_latents = self.normalize_latents(latents)
        else:
            normalized_latents = latents
        
        # 添加噪声
        noise = torch.randn_like(normalized_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,), device=normalized_latents.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(
            normalized_latents, noise, timesteps
        )
        
        # 获取用户条件
        user_conditions = self.get_user_condition(user_ids)
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_conditions,
            return_dict=False
        )[0]
        
        # Huber损失（比MSE更鲁棒）
        diffusion_loss = F.huber_loss(noise_pred, noise, delta=1.0)
        
        # 简单的对比损失
        contrastive_loss = self.contrastive_loss(noise_pred.mean(dim=[1,2,3]), noise.mean(dim=[1,2,3]))
        
        total_loss = diffusion_loss + 0.1 * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def get_user_condition(self, user_ids):
        """获取用户条件"""
        if torch.is_tensor(user_ids):
            # 如果是tensor，转换为list
            if user_ids.dim() == 0:
                # 标量tensor
                user_id_list = [user_ids.item()]
            else:
                # 向量tensor
                user_id_list = user_ids.cpu().tolist()
        else:
            # 如果已经是list
            user_id_list = user_ids
        
        conditions = []
        for user_id in user_id_list:
            # 确保user_id是整数
            if isinstance(user_id, torch.Tensor):
                user_id = user_id.item()
            prototype = self.user_prototypes[str(user_id)]
            conditions.append(prototype)
        
        # Stack and add sequence dimension
        conditions = torch.stack(conditions).unsqueeze(1)  # [batch, 1, prototype_dim]
        return conditions
    
    def update_user_prototypes(self, user_latents_dict):
        """更新用户原型"""
        with torch.no_grad():
            for user_id, latents in user_latents_dict.items():
                # 计算该用户latents的均值作为原型
                prototype = latents.mean(dim=[0, 2, 3])  # 对batch和空间维度求均值
                self.user_prototypes[str(user_id)].data.copy_(prototype)
    
    def generate(self, user_ids, num_samples=1, num_inference_steps=50, guidance_scale=7.5):
        """生成latents"""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 准备用户条件
            if isinstance(user_ids, list):
                total_samples = len(user_ids) * num_samples // len(user_ids)
                user_ids_expanded = []
                for user_id in user_ids:
                    user_ids_expanded.extend([user_id] * (num_samples // len(user_ids)))
                user_ids_tensor = torch.tensor(user_ids_expanded, device=device)
            else:
                user_ids_tensor = torch.tensor([user_ids] * num_samples, device=device)
            
            # 获取用户条件
            user_conditions = self.get_user_condition(user_ids_tensor)
            
            # 初始化随机噪声
            shape = (len(user_ids_tensor), 32, 16, 16)
            latents = torch.randn(shape, device=device)
            
            # 设置调度器
            self.noise_scheduler.set_timesteps(num_inference_steps)
            
            # 去噪过程
            for t in self.noise_scheduler.timesteps:
                # 预测噪声
                timesteps = t.expand(latents.shape[0]).to(device)
                noise_pred = self.unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=user_conditions,
                    return_dict=False
                )[0]
                
                # 去噪步骤
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            
            # 如果启用了分布对齐，反归一化
            if self.enable_alignment and self.latent_std > 0:
                latents = self.denormalize_latents(latents)
            
            return latents
