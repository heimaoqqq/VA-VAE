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

class PrototypicalEncoder(nn.Module):
    """原型编码器：从少量样本学习用户原型"""
    
    def __init__(self, input_dim=8192, prototype_dim=256):
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
        
    def triplet_loss(self, anchor, positive, negative, margin=0.2):
        """三元组损失"""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()
    
    def infonce_loss(self, query, keys, positive_mask):
        """InfoNCE损失"""
        # 计算相似度
        logits = torch.matmul(query, keys.T) / self.temperature
        
        # 创建标签
        labels = positive_mask.float()
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, prototypes, user_ids):
        """
        计算对比损失
        Args:
            prototypes: [B, D] - 原型特征
            user_ids: [B] - 用户ID
        """
        batch_size = prototypes.size(0)
        loss = 0
        count = 0
        
        for i in range(batch_size):
            # 找同用户样本（positive）
            same_user = (user_ids == user_ids[i])
            same_user[i] = False  # 排除自己
            
            if same_user.any():
                positive_idx = torch.where(same_user)[0]
                positive = prototypes[positive_idx[0]]
                
                # 找不同用户样本（negative）
                diff_user = (user_ids != user_ids[i])
                if diff_user.any():
                    negative_idx = torch.where(diff_user)[0]
                    negative = prototypes[negative_idx[0]]
                    
                    # 计算三元组损失
                    anchor = prototypes[i]
                    loss += self.triplet_loss(anchor.unsqueeze(0), 
                                            positive.unsqueeze(0), 
                                            negative.unsqueeze(0))
                    count += 1
        
        return loss / max(count, 1)


class EnhancedConditionalDiffusion(nn.Module):
    """增强条件扩散模型"""
    
    def __init__(self, num_users=31, prototype_dim=256):
        super().__init__()
        
        # 核心组件
        self.prototype_encoder = PrototypicalEncoder(
            input_dim=8192, prototype_dim=prototype_dim
        )
        self.contrastive_learning = ContrastiveLearning(temperature=0.07)
        
        # U-Net扩散模型
        self.unet = UNet2DConditionModel(
            sample_size=(16, 16),
            in_channels=32,
            out_channels=32,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D", 
                "AttnDownBlock2D",
                "DownBlock2D"
            ],
            up_block_types=[
                "UpBlock2D",
                "AttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D"
            ],
            attention_head_dim=8,
            cross_attention_dim=prototype_dim,
            encoder_hid_dim=prototype_dim
        )
        
        # 调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # DDIM调度器用于生成
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        
        # 用户原型缓存
        self.user_prototypes = {}
        
    def update_user_prototypes(self, user_samples_dict):
        """更新用户原型缓存"""
        self.user_prototypes = {}
        
        for user_id, samples in user_samples_dict.items():
            # samples: [N, 32, 16, 16]
            with torch.no_grad():
                prototypes = self.prototype_encoder(samples)
                # 使用均值作为用户原型
                user_prototype = torch.mean(prototypes, dim=0, keepdim=True)
                self.user_prototypes[user_id] = user_prototype
    
    def get_user_condition(self, user_ids):
        """获取用户条件编码"""
        batch_size = len(user_ids)
        device = next(self.parameters()).device
        
        conditions = torch.zeros(batch_size, 1, 256, device=device)
        
        for i, user_id in enumerate(user_ids):
            if user_id in self.user_prototypes:
                conditions[i] = self.user_prototypes[user_id]
            else:
                # 如果没有缓存原型，使用零向量
                conditions[i] = torch.zeros(1, 256, device=device)
        
        return conditions
    
    def training_step(self, clean_latents, user_ids, support_ratio=0.3):
        """
        训练步骤：结合原型学习和对比学习
        Args:
            clean_latents: [B, 32, 16, 16]
            user_ids: [B]
            support_ratio: 用作support set的比例
        """
        batch_size = clean_latents.size(0)
        device = clean_latents.device
        
        # 分割support和query set
        n_support = int(batch_size * support_ratio)
        indices = torch.randperm(batch_size)
        support_indices = indices[:n_support]
        query_indices = indices[n_support:]
        
        # 原型学习
        if len(support_indices) > 0:
            support_latents = clean_latents[support_indices]
            support_user_ids = user_ids[support_indices]
            
            # 编码原型
            prototypes = self.prototype_encoder(support_latents)
            
            # 对比学习损失
            contrastive_loss = self.contrastive_learning(prototypes, support_user_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        
        # 扩散损失（在query set上）
        if len(query_indices) > 0:
            query_latents = clean_latents[query_indices]
            query_user_ids = user_ids[query_indices]
        else:
            query_latents = clean_latents
            query_user_ids = user_ids
        
        # 扩散训练
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (query_latents.size(0),), device=device
        ).long()
        
        noise = torch.randn_like(query_latents)
        noisy_latents = self.noise_scheduler.add_noise(query_latents, noise, timesteps)
        
        # 获取用户条件
        user_conditions = self.get_user_condition(query_user_ids.tolist())
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_conditions
        ).sample
        
        # 扩散损失
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # 总损失
        total_loss = diffusion_loss + 0.1 * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def generate(self, user_ids, num_samples_per_user=4, num_inference_steps=100, 
                 guidance_scale=2.0, use_ddim=True):
        """生成样本（支持DDIM和DDPM）"""
        self.eval()
        
        if isinstance(user_ids, int):
            user_ids = [user_ids]
        
        total_samples = len(user_ids) * num_samples_per_user
        device = next(self.parameters()).device
        
        # 扩展用户ID
        expanded_user_ids = []
        for user_id in user_ids:
            expanded_user_ids.extend([user_id] * num_samples_per_user)
        
        # 初始噪声
        latents = torch.randn(total_samples, 32, 16, 16, device=device)
        
        # 选择调度器
        scheduler = self.ddim_scheduler if use_ddim else self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 获取用户条件
        user_conditions = self.get_user_condition(expanded_user_ids)
        
        # 去噪过程
        for t in scheduler.timesteps:
            with torch.no_grad():
                # 条件预测
                noise_pred_cond = self.unet(
                    latents, t, encoder_hidden_states=user_conditions
                ).sample
                
                # 无条件预测（用于CFG）
                if guidance_scale > 1.0:
                    uncond_conditions = torch.zeros_like(user_conditions)
                    noise_pred_uncond = self.unet(
                        latents, t, encoder_hidden_states=uncond_conditions
                    ).sample
                    
                    # Classifier-free guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond
                
                # 去噪步骤
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents

# 使用示例
def create_enhanced_diffusion_system(vae_checkpoint, num_users=31):
    """创建增强条件扩散系统"""
    
    # 加载简化VA-VAE
    from simplified_vavae import SimplifiedVAVAE
    vae = SimplifiedVAVAE(vae_checkpoint)
    
    # 创建增强扩散模型
    enhanced_diffusion = EnhancedConditionalDiffusion(num_users=num_users)
    
    return vae, enhanced_diffusion

if __name__ == "__main__":
    # 使用 train_enhanced_conditional.py 进行完整训练
    print("✅ 增强条件扩散模型定义完成")
    print("🎯 专门处理小数据集微差异条件生成挑战")
    print("📊 集成: Prototypical Networks + Contrastive Learning")
    print("📝 使用 train_enhanced_conditional.py 进行训练")
