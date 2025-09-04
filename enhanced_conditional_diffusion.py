#!/usr/bin/env python3
"""
增强条件扩散生成：专门处理小数据集微差异条件生成
结合Prototypical Networks、Contrastive Learning、Few-Shot Meta-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import numpy as np
from sklearn.cluster import KMeans
import random
from tqdm import tqdm

class PrototypicalEncoder(nn.Module):
    """原型编码器：从少量样本学习用户原型"""
    
    def __init__(self, input_dim=8192, prototype_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, prototype_dim)
        )
        
    def forward(self, latents):
        """
        Args:
            latents: [N, 32, 16, 16] - 用户的latent样本
        Returns:
            prototypes: [N, prototype_dim] - 原型特征
        """
        # 展平latent
        flat_latents = latents.view(latents.size(0), -1)
        prototypes = self.encoder(flat_latents)
        return F.normalize(prototypes, dim=1)  # L2归一化

class ContrastiveLearning(nn.Module):
    """对比学习模块：增强用户间判别能力"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, user_ids):
        """
        Args:
            features: [B, D] 用户原型特征
            user_ids: [B] 用户ID
        Returns:
            contrastive_loss: 对比损失
        """
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签矩阵
        user_ids = user_ids.view(-1, 1)
        labels = (user_ids == user_ids.T).float()
        
        # 创建mask（排除对角线）
        batch_size = features.size(0)
        mask = torch.eye(batch_size, device=features.device).bool()
        labels = labels.masked_fill(mask, 0)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = (exp_sim * labels).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)  # 排除自己
        
        # 避免除零
        pos_sim = torch.clamp(pos_sim, min=1e-8)
        all_sim = torch.clamp(all_sim, min=1e-8)
        
        loss = -torch.log(pos_sim / all_sim).mean()
        
        return loss


class EnhancedConditionalDiffusion(nn.Module):
    """
    增强条件扩散模型 - 正确实现版本
    
    核心改进：
    1. 使用缩放因子标准化latent到N(0,1)空间（类似Stable Diffusion）
    2. 训练和生成都在标准化空间进行
    3. 无硬限制或clamp，让模型自然学习分布
    4. 生成后反标准化回原始VAE空间
    """
    
    def __init__(
        self,
        vae,
        num_timesteps=1000,
        noise_schedule="linear", 
        num_users=31,
        prototype_dim=768,
        latent_mean=0.059,
        latent_std=1.54,
    ):
        super().__init__()
        
        # VAE组件
        self.vae = vae
        
        # 核心：缩放因子（将VAE latent标准化到接近N(0,1)）
        # 类似Stable Diffusion的0.18215，我们使用1/std
        self.scale_factor = 1.0 / latent_std  # ≈ 0.649
        print(f"使用缩放因子: {self.scale_factor:.4f}")
        
        # 记录原始分布参数
        self.register_buffer('latent_mean', torch.tensor(latent_mean))
        self.register_buffer('latent_std', torch.tensor(latent_std))
        
        # 原型学习和对比学习组件
        self.prototype_encoder = PrototypicalEncoder(
            input_dim=32*16*16,  # VA-VAE latent维度
            prototype_dim=prototype_dim
        )
        self.contrastive_learning = ContrastiveLearning(temperature=0.07)
        
        # UNet扩散模型 - 在标准化空间工作
        self.unet = UNet2DConditionModel(
            in_channels=32,  # VA-VAE有32个通道
            out_channels=32,
            down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 384, 512),
            layers_per_block=2,
            attention_head_dim=8,
            cross_attention_dim=prototype_dim,
            sample_size=16
        )
        
        self.num_timesteps = num_timesteps
        
        # 噪声调度器 - 标准配置，无剪裁
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=noise_schedule,
            clip_sample=False,  # 关键：不剪裁
            prediction_type="epsilon",
        )
        
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=noise_schedule,
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        # 用户原型缓存
        self.user_prototypes = {}
    
    def encode_to_standard_space(self, latents):
        """编码到标准化空间（类似Stable Diffusion）"""
        return latents * self.scale_factor
    
    def decode_to_original_space(self, latents):
        """解码回原始VAE空间"""
        return latents / self.scale_factor
    
    def update_user_prototypes(self, user_samples_dict):
        """更新用户原型缓存"""
        self.user_prototypes = {}
        
        for user_id, samples in user_samples_dict.items():
            # samples: [N, 32, 16, 16] - 在原始空间
            with torch.no_grad():
                prototypes = self.prototype_encoder(samples)
                user_prototype = torch.mean(prototypes, dim=0, keepdim=True)
                self.user_prototypes[user_id] = user_prototype
    
    def set_training_stats(self, mean, std):
        """更新训练分布统计量"""
        if mean is not None:
            self.latent_mean.copy_(torch.tensor(mean))
        if std is not None:
            self.latent_std.copy_(torch.tensor(std))
            self.scale_factor = 1.0 / std
        print(f"✅ 已更新分布: mean={self.latent_mean:.4f}, std={self.latent_std:.4f}")
        print(f"   缩放因子: {self.scale_factor:.4f}")
    
    def get_user_condition(self, user_ids):
        """获取用户条件编码"""
        batch_size = len(user_ids)
        device = next(self.parameters()).device
        
        # 确保返回正确的3D张量形状 [batch_size, sequence_length=1, hidden_dim=768]
        conditions = torch.zeros(batch_size, 1, 768, device=device)
        
        for i, user_id in enumerate(user_ids):
            user_id_int = user_id.item() if torch.is_tensor(user_id) else user_id
            if user_id_int in self.user_prototypes:
                # 确保原型是正确的形状 [1, 768]
                prototype = self.user_prototypes[user_id_int]
                if prototype.dim() == 1:
                    prototype = prototype.unsqueeze(0)  # [768] -> [1, 768]
                conditions[i] = prototype
            else:
                # 随机初始化为 [1, 768]
                conditions[i] = torch.randn(1, 768, device=device) * 0.1
        
        # 确保返回的张量形状正确
        if conditions.dim() != 3 or conditions.shape[-1] != 768:
            print(f"⚠️ 警告：user_conditions形状异常: {conditions.shape}")
            conditions = conditions.view(batch_size, 1, 768)
            
        return conditions
    
    def training_step(self, clean_latents, user_conditions):
        """
        训练步骤 - 正确实现版本
        在标准化空间进行训练
        """
        batch_size = clean_latents.shape[0]
        device = clean_latents.device
        
        # ✅ 步骤1：编码到标准化空间N(0,1)
        latents_standard = self.encode_to_standard_space(clean_latents)
        
        # 随机时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # ✅ 步骤2：在标准化空间添加噪声N(0,1)
        noise = torch.randn_like(latents_standard)
        noisy_latents = self.scheduler.add_noise(latents_standard, noise, timesteps)
        
        # ✅ 步骤3：预测噪声（在标准化空间）
        # 确保所有张量都在同一设备上
        noisy_latents = noisy_latents.to(device)
        timesteps = timesteps.to(device)
        user_conditions = user_conditions.to(device)
        
        # 调试：打印张量形状
        print(f"🔍 Debug shapes - noisy_latents: {noisy_latents.shape}, user_conditions: {user_conditions.shape}")
        
        # 确保user_conditions是3D张量 [batch_size, seq_len, hidden_dim]
        if user_conditions.dim() == 2:
            user_conditions = user_conditions.unsqueeze(1)  # [B, H] -> [B, 1, H]
        elif user_conditions.dim() == 1:
            user_conditions = user_conditions.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
        
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_conditions,
        ).sample
        
        # 扩散损失
        noise = noise.to(device)  # 确保噪声在正确设备上
        diff_loss = F.mse_loss(noise_pred, noise)
        
        # 对比损失（需要在原始空间计算原型）
        predicted_clean = self.scheduler.step(
            noise_pred, timesteps[0], noisy_latents
        ).pred_original_sample
        
        # 反标准化以计算原型
        predicted_clean_orig = self.decode_to_original_space(predicted_clean)
        pred_features = self.prototype_encoder(predicted_clean_orig)
        
        # 为对比学习创建用户ID张量
        batch_size = user_conditions.shape[0]
        user_ids_tensor = torch.arange(batch_size, device=device)  # 简化的用户ID
        contrastive_loss = self.contrastive_learning(pred_features, user_ids_tensor)
        
        # 总损失
        total_loss = diff_loss + 0.1 * contrastive_loss
        
        return total_loss, diff_loss, contrastive_loss
    
    @torch.no_grad()
    def generate(
        self,
        user_ids,
        num_samples_per_user=4,
        num_inference_steps=50,
        guidance_scale=7.5,
        use_ddim=True,
    ):
        """
        生成样本 - 正确实现版本
        在标准化空间生成，最后反标准化
        """
        self.eval()
        
        if isinstance(user_ids, int):
            user_ids = [user_ids]
        
        batch_size = len(user_ids) * num_samples_per_user
        device = next(self.parameters()).device
        
        # 选择调度器
        scheduler = self.ddim_scheduler if use_ddim else self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # ✅ 步骤1：从标准高斯N(0,1)开始（在标准化空间）
        shape = (batch_size, 32, 16, 16)
        latents = torch.randn(shape, device=device)
        
        print(f"📊 生成配置:")
        print(f"   标准化空间N(0,1)初始化")
        print(f"   初始std: {latents.std():.4f}")
        
        # 准备条件
        user_ids_expanded = []
        for uid in user_ids:
            user_ids_expanded.extend([uid] * num_samples_per_user)
        user_conditions = self.get_user_condition(user_ids_expanded)
        
        # CFG准备
        if guidance_scale > 1.0:
            uncond_conditions = torch.zeros_like(user_conditions)
        
        # ✅ 步骤2：去噪循环（在标准化空间，无任何限制）
        for t in tqdm(scheduler.timesteps, desc="生成中"):
            # 条件预测
            noise_pred_cond = self.unet(
                latents, t, encoder_hidden_states=user_conditions
            ).sample
            
            if guidance_scale > 1.0:
                # 无条件预测
                noise_pred_uncond = self.unet(
                    latents, t, encoder_hidden_states=uncond_conditions
                ).sample
                
                # CFG
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond
            
            # 去噪步骤（无clamp或限制）
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # ✅ 步骤3：反标准化到原始VAE空间
        latents_original = self.decode_to_original_space(latents)
        
        # 验证分布
        print(f"📊 生成latent分布:")
        print(f"   标准化空间: mean={latents.mean():.4f}, std={latents.std():.4f}")
        print(f"   原始空间: mean={latents_original.mean():.4f}, std={latents_original.std():.4f}")
        print(f"   ✅ 期望: mean≈{self.latent_mean:.3f}, std≈{self.latent_std:.3f}")
        
        return latents_original


# 使用示例
def create_enhanced_diffusion_system(vae_checkpoint, num_users=31):
    """创建增强条件扩散系统"""
    
    # 加载简化VA-VAE
    from simplified_vavae import SimplifiedVAVAE
    vae = SimplifiedVAVAE(vae_checkpoint)
    
    # 创建增强扩散模型
    diffusion_model = EnhancedConditionalDiffusion(
        vae=vae,
        num_users=num_users,
        prototype_dim=768
    )
    
    return diffusion_model, vae


if __name__ == "__main__":
    print("✅ 增强条件扩散模型 - 正确标准化版本")
    print("🔑 核心改进:")
    print("   - 使用缩放因子标准化latent到N(0,1)空间")
    print("   - 训练和生成都在标准化空间进行")
    print("   - 无硬限制，让模型自然学习分布")
    print("   - 生成后反标准化回原始VAE空间")
    print("📝 使用 train_enhanced_conditional.py 进行训练")
