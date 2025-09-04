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
        return F.mse_loss(predicted_latents, target_latents)


class StandardConditionalDiffusion(nn.Module):
    """标准条件扩散模型 - 采用Diffusers标准做法"""
    
    def __init__(
        self,
        vae,
        num_timesteps=1000,
        noise_schedule="linear",
        num_users=31,
        prototype_dim=768,
    ):
        super().__init__()
        
        # VAE组件
        self.vae = vae
        
        # 原型学习和对比学习组件
        self.prototype_encoder = PrototypicalEncoder(
            input_dim=32*16*16,  # VA-VAE latent维度
            prototype_dim=prototype_dim
        )
        self.contrastive_learning = ContrastiveLearning(temperature=0.07)
        
        # 标准UNet扩散模型
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
        
        # 标准噪声调度器 - 直接使用默认配置
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=noise_schedule,
            clip_sample=False,
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
        
        print("✅ 使用标准扩散模型 - 无自定义标准化")
    
    def get_user_condition(self, user_ids):
        """获取用户条件编码"""
        batch_size = len(user_ids)
        device = next(self.parameters()).device
        
        # 确保返回正确的3D张量形状 [batch_size, sequence_length=1, hidden_dim=768]
        conditions = torch.zeros(batch_size, 1, self.prototype_encoder.encoder[-1].normalized_shape[0], device=device)
        
        for i, user_id in enumerate(user_ids):
            user_id_int = user_id.item() if torch.is_tensor(user_id) else user_id
            if user_id_int in self.user_prototypes:
                prototype = self.user_prototypes[user_id_int]
                if prototype.dim() == 1:
                    prototype = prototype.unsqueeze(0)
                conditions[i] = prototype
            else:
                # 随机初始化未见过的用户
                conditions[i] = torch.randn(1, self.prototype_encoder.encoder[-1].normalized_shape[0], device=device) * 0.1
        
        # 确保维度正确
        if conditions.dim() != 3 or conditions.shape[-1] != self.prototype_encoder.encoder[-1].normalized_shape[0]:
            conditions = conditions.view(batch_size, 1, self.prototype_encoder.encoder[-1].normalized_shape[0])
        
        return conditions
    
    def training_step(self, clean_latents, user_conditions):
        """标准训练步骤 - 参考Diffusers实现"""
        batch_size = clean_latents.shape[0]
        device = clean_latents.device
        
        # ✅ 关键：直接在VAE latent上工作，无额外标准化
        # 这就是Stable Diffusion等标准实现的做法
        
        # 随机时间步
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # ✅ 标准做法：直接添加噪声到clean_latents
        noise = torch.randn_like(clean_latents)
        noisy_latents = self.scheduler.add_noise(clean_latents, noise, timesteps)
        
        # ✅ 预测噪声
        # 确保所有张量都在同一设备上
        noisy_latents = noisy_latents.to(device)
        timesteps = timesteps.to(device)
        user_conditions = user_conditions.to(device)
        
        # 确保user_conditions是3D张量
        if user_conditions.dim() == 2:
            user_conditions = user_conditions.unsqueeze(1)
        elif user_conditions.dim() == 1:
            user_conditions = user_conditions.unsqueeze(0).unsqueeze(0)
        
        # UNet预测噪声
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=user_conditions,
        ).sample
        
        # 扩散损失
        diff_loss = F.mse_loss(noise_pred, noise)
        
        # 对比损失（在原始clean latent上计算）
        # 获取预测的clean latent（用于原型学习）
        predicted_clean = self.scheduler.step(
            noise_pred, timesteps[0], noisy_latents
        ).pred_original_sample
        
        # 在原始空间计算对比损失
        clean_latents_flat = clean_latents.view(batch_size, -1)
        predicted_clean_flat = predicted_clean.view(batch_size, -1)
        
        # 计算原型
        clean_prototypes = self.prototype_encoder(clean_latents_flat)
        predicted_prototypes = self.prototype_encoder(predicted_clean_flat)
        
        # 对比损失
        contrastive_loss = self.contrastive_learning(
            predicted_prototypes, clean_prototypes, None
        )
        
        # 总损失
        total_loss = diff_loss + 0.1 * contrastive_loss
        
        return total_loss, diff_loss, contrastive_loss
    
    def update_user_prototypes(self, user_samples_dict):
        """更新用户原型缓存"""
        self.user_prototypes = {}
        
        for user_id, samples in user_samples_dict.items():
            # samples: [N, 32, 16, 16] - VAE latent空间
            with torch.no_grad():
                samples_flat = samples.view(samples.shape[0], -1)
                prototypes = self.prototype_encoder(samples_flat)
                user_prototype = torch.mean(prototypes, dim=0, keepdim=True)
                self.user_prototypes[user_id] = user_prototype
    
    @torch.no_grad()
    def generate_samples(
        self,
        user_ids,
        num_samples_per_user=2,
        num_inference_steps=100,
        guidance_scale=1.0,
        use_ddim=True
    ):
        """标准生成过程"""
        self.eval()
        
        if isinstance(user_ids, int):
            user_ids = [user_ids]
        
        batch_size = len(user_ids) * num_samples_per_user
        device = next(self.parameters()).device
        
        # 选择调度器
        scheduler = self.ddim_scheduler if use_ddim else self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # ✅ 标准初始化：从纯噪声开始
        shape = (batch_size, 32, 16, 16)
        latents = torch.randn(shape, device=device)
        
        print(f"📊 生成配置:")
        print(f"   标准高斯N(0,1)初始化")
        print(f"   初始std: {latents.std():.4f}")
        
        # 准备条件
        user_ids_expanded = []
        for uid in user_ids:
            user_ids_expanded.extend([uid] * num_samples_per_user)
        user_conditions = self.get_user_condition(user_ids_expanded)
        
        # CFG准备
        if guidance_scale > 1.0:
            uncond_conditions = torch.zeros_like(user_conditions)
        
        # ✅ 标准去噪循环
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
            
            # ✅ 标准去噪步骤
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 验证分布
        print(f"📊 生成latent分布:")
        print(f"   最终latent: mean={latents.mean():.4f}, std={latents.std():.4f}")
        print(f"   应该接近训练数据的latent分布")
        
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
