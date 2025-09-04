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
        
        # 扩散组件
        self.unet = UNet2DConditionModel(
            sample_size=16,  # VAE latent size
            in_channels=32,  # VAE latent channels
            out_channels=32,
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
            user_conditions = self.get_user_condition(user_ids)
            
            # 初始化噪声 - ✅ 标准方式从N(0,1)开始
            latent_shape = (num_samples, 32, 16, 16)  # VAE latent shape
            latents = torch.randn(latent_shape, device=device)
            
            # 📊 监控生成初始latent std
            init_std = latents.std().item()
            print(f"📊 生成初始latent std: {init_std:.6f}")
            
            # 设置推理调度器
            inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon"
            )
            inference_scheduler.set_timesteps(num_inference_steps)
            
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
                
                # 📊 每10步监控latent std
                if i % 10 == 0:
                    current_std = latents.std().item()
                    print(f"   Step {i:2d}/{num_inference_steps}: latent std = {current_std:.6f}")
            
            # 📊 监控最终latent std
            final_std = latents.std().item()
            print(f"📊 生成最终latent std: {final_std:.6f}")
        
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
