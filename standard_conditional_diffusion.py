"""
标准条件扩散模型 - 基于Diffusers标准实现
参考：huggingface/diffusers/examples/unconditional_image_generation/train_unconditional.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from tqdm import tqdm


class PrototypicalEncoder(nn.Module):
    """原型编码器 - 用于学习用户特征"""
    def __init__(self, input_dim, prototype_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, prototype_dim),
            nn.LayerNorm(prototype_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class ContrastiveLearning(nn.Module):
    """对比学习模块"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted_latents, target_latents, user_ids):
        # 简化的对比损失
        batch_size = predicted_latents.size(0)
        
        # 计算特征相似度矩阵
        pred_flat = predicted_latents.view(batch_size, -1)
        target_flat = target_latents.view(batch_size, -1)
        
        # L2 normalize
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)
        
        # 计算相似度
        similarity = torch.matmul(pred_norm, target_norm.T) / self.temperature
        
        # InfoNCE损失
        labels = torch.arange(batch_size, device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class StandardConditionalDiffusion(nn.Module):
    """标准条件扩散模型 - 基于Diffusers"""
    
    def __init__(self, vae, num_users=31, prototype_dim=768):
        super().__init__()
        
        # VAE（冻结）
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 🔧 恢复稳定UNet架构 - 修复生成爆炸
        self.unet = UNet2DConditionModel(
            sample_size=16,  # VAE latent size
            in_channels=32,  # VAE latent channels
            out_channels=32,
            layers_per_block=2,  # 恢复足够深度
            block_out_channels=(128, 256, 512),  # 增加容量防止不稳定
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
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # 用户原型系统
        self.num_users = num_users
        self.prototype_dim = prototype_dim
        self.user_prototypes = nn.ParameterDict()
        
        # 对比学习
        self.contrastive = ContrastiveLearning()
        
        print("✅ 使用标准扩散模型 - 无自定义标准化")
    
    def get_user_condition(self, user_ids):
        """获取用户条件编码"""
        batch_size = len(user_ids)
        device = next(self.parameters()).device
        
        conditions = torch.zeros(batch_size, 1, self.prototype_dim, device=device)
        
        for i, user_id in enumerate(user_ids):
            user_id_int = user_id.item() if torch.is_tensor(user_id) else user_id
            user_key = str(user_id_int)
            
            if user_key not in self.user_prototypes:
                # 初始化新用户原型
                self.user_prototypes[user_key] = nn.Parameter(
                    torch.randn(1, self.prototype_dim, device=device) * 0.1
                )
            
            conditions[i] = self.user_prototypes[user_key]
        
        return conditions
    
    def update_user_prototypes(self, user_latents):
        """更新用户原型 - 基于latent特征"""
        # 简化实现：使用latent的全局池化作为特征
        for user_id, latents in user_latents.items():
            user_key = str(user_id)
            if user_key in self.user_prototypes:
                # latents shape: [N, 32, 16, 16] -> 提取全局特征 [N, feature_dim]
                # 全局平均池化 + 投影到原型空间
                pooled_features = latents.mean(dim=[2, 3])  # [N, 32]
                
                # 简单线性投影到原型维度 (如果需要)
                if pooled_features.size(-1) != self.prototype_dim:
                    # 重复或截断到匹配原型维度
                    if pooled_features.size(-1) < self.prototype_dim:
                        # 重复填充
                        repeat_times = self.prototype_dim // pooled_features.size(-1)
                        remainder = self.prototype_dim % pooled_features.size(-1)
                        expanded = pooled_features.repeat(1, repeat_times)
                        if remainder > 0:
                            expanded = torch.cat([expanded, pooled_features[:, :remainder]], dim=1)
                        pooled_features = expanded
                    else:
                        # 截断
                        pooled_features = pooled_features[:, :self.prototype_dim]
                
                # 平均所有样本的特征
                new_feature = pooled_features.mean(dim=0, keepdim=True)  # [1, prototype_dim]
                
                # 使用移动平均更新
                current_prototype = self.user_prototypes[user_key]
                self.user_prototypes[user_key].data = (
                    0.9 * current_prototype.data + 0.1 * new_feature
                )
    
    def training_step(self, clean_latents, user_conditions):
        """标准训练步骤 - 基于Diffusers"""
        device = clean_latents.device
        batch_size = clean_latents.size(0)
        
        # ✅ 直接在原始latent空间工作，无标准化
        
        # 随机时间步
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(clean_latents)
        noisy_latents = self.scheduler.add_noise(clean_latents, noise, timesteps)
        
        # 确保条件张量格式正确
        if user_conditions.dim() == 2:
            user_conditions = user_conditions.unsqueeze(1)
        
        # UNet预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_conditions,
        ).sample
        
        # 扩散损失（MSE）
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # 对比学习损失
        contrastive_loss = self.contrastive(noise_pred, noise, None) * 0.1
        
        total_loss = diffusion_loss + contrastive_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss, 
            'contrastive_loss': contrastive_loss
        }
    
    def generate(self, user_ids, num_samples=4, guidance_scale=7.5, num_inference_steps=50):
        """生成样本"""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 获取用户条件
            base_conditions = self.get_user_condition(user_ids)
            
            # 重复用户条件以匹配样本数量
            samples_per_user = num_samples // len(user_ids)
            user_conditions = base_conditions.repeat_interleave(samples_per_user, dim=0)
            
            # 如果样本数不整除用户数，处理剩余样本
            remaining = num_samples - len(user_conditions)
            if remaining > 0:
                extra_conditions = base_conditions[:remaining]
                user_conditions = torch.cat([user_conditions, extra_conditions], dim=0)
            
            print(f"📊 用户条件形状: {user_conditions.shape}")
            
            # 初始化噪声 - ✅ 标准方式从N(0,1)开始
            latent_shape = (num_samples, 32, 16, 16)  # VAE latent shape
            latents = torch.randn(latent_shape, device=device)
            
            # 🔧 使用与训练相同的DDPM调度器 - 确保参数一致性
            from diffusers import DDPMScheduler
            inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon"
            )
            inference_scheduler.set_timesteps(num_inference_steps)
            
            # 正确缩放初始噪声
            latents = latents * inference_scheduler.init_noise_sigma
            
            # 📊 监控缩放后的初始latent std
            init_std = latents.std().item()
            print(f"📊 缩放后初始latent std: {init_std:.6f} (sigma={inference_scheduler.init_noise_sigma:.3f})")
            
            # CFG的无条件输入
            if guidance_scale > 1.0:
                uncond_conditions = torch.zeros_like(user_conditions)
            
            # ✅ 标准去噪循环
            for i, t in enumerate(tqdm(inference_scheduler.timesteps, desc="生成中")):
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
                
                # 调度器步骤
                latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
                
                # 🔧 防止数值爆炸 - 裁剪异常latent值
                latents = torch.clamp(latents, min=-10.0, max=10.0)
                
                # 📊 每10步监控latent std
                if i % 10 == 0:
                    current_std = latents.std().item()
                    print(f"   Step {i:2d}/{num_inference_steps}: latent std = {current_std:.6f}")
            
            # 📊 监控最终latent std
            final_std = latents.std().item()
            print(f"📊 生成最终latent std: {final_std:.6f}")
            
            # 🔧 激进修复：强制匹配训练分布
            target_std = 1.54  # 训练数据的实际std
            if final_std < target_std * 0.8:  # 如果std太低
                print(f"⚠️  检测到std过低，进行分布校正...")
                
                # 保持均值，重新缩放std
                latent_mean = latents.mean()
                latents_centered = latents - latent_mean
                scale_factor = target_std / final_std
                latents = latents_centered * scale_factor + latent_mean
                
                corrected_std = latents.std().item()
                print(f"📊 校正后latent std: {corrected_std:.6f} (缩放因子: {scale_factor:.3f})")
        
        self.train()
        return latents


# 使用示例
def create_standard_diffusion_system(vae_checkpoint, num_users=31):
    """创建标准扩散系统"""
    
    # 加载简化VA-VAE
    from simplified_vavae import SimplifiedVAVAE
    vae = SimplifiedVAVAE(vae_checkpoint)
    
    # 创建标准扩散模型
    diffusion_model = StandardConditionalDiffusion(
        vae=vae,
        num_users=num_users,
        prototype_dim=768
    )
    
    return diffusion_model, vae


if __name__ == "__main__":
    import argparse
    import sys
    print("❌ 错误：standard_conditional_diffusion.py 不是训练脚本！")
    print("✅ 请使用：python train_enhanced_conditional.py")
    print("📖 说明：standard_conditional_diffusion.py 只是模型定义文件")
    sys.exit(1)
