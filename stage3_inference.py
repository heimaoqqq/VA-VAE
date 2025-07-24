#!/usr/bin/env python3
"""
阶段3: 推理和生成
使用训练好的用户条件化DiT生成微多普勒图像
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os
from PIL import Image

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.autoencoder import AutoencoderKL
from transport import create_transport

# 导入我们的模型
from stage2_train_dit import UserConditionedDiT

class MicroDopplerGenerator:
    """
    微多普勒图像生成器
    """
    
    def __init__(self, dit_checkpoint, vavae_path, device='cuda'):
        self.device = device
        
        print("🔄 加载模型...")
        
        # 加载VA-VAE
        print("📥 加载VA-VAE...")
        self.vavae = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=vavae_path,
            model_type='vavae'
        )
        self.vavae.eval()
        self.vavae.to(device)
        
        # 加载DiT模型
        print("📥 加载DiT模型...")
        self.dit_model = UserConditionedDiT.load_from_checkpoint(dit_checkpoint)
        self.dit_model.eval()
        self.dit_model.to(device)
        
        # 创建扩散传输
        self.transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=1e-5,
            sample_eps=1e-4,
        )
        
        print(f"✅ 模型加载完成")
        print(f"  用户数量: {self.dit_model.num_users}")
    
    @torch.no_grad()
    def generate(
        self,
        user_ids,
        num_samples_per_user=4,
        guidance_scale=4.0,
        num_steps=250,
        seed=None
    ):
        """
        生成微多普勒图像
        
        Args:
            user_ids: 用户ID列表 (1-based)
            num_samples_per_user: 每个用户生成的样本数
            guidance_scale: classifier-free guidance强度
            num_steps: 扩散步数
            seed: 随机种子
        
        Returns:
            generated_images: 生成的图像 (B, 3, 256, 256)
            user_labels: 对应的用户ID
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"🎨 开始生成图像...")
        print(f"  用户ID: {user_ids}")
        print(f"  每用户样本数: {num_samples_per_user}")
        print(f"  引导强度: {guidance_scale}")
        print(f"  扩散步数: {num_steps}")
        
        # 准备批次
        batch_size = len(user_ids) * num_samples_per_user
        
        # 创建用户条件 (转换为0-based)
        user_conditions = []
        user_labels = []
        for user_id in user_ids:
            for _ in range(num_samples_per_user):
                user_conditions.append(user_id - 1)  # 转换为0-based
                user_labels.append(user_id)
        
        user_conditions = torch.tensor(user_conditions, device=self.device)
        
        # 初始噪声
        latent_shape = (batch_size, 32, 16, 16)  # VA-VAE的潜在空间形状
        noise = torch.randn(latent_shape, device=self.device)
        
        print(f"  初始噪声形状: {noise.shape}")
        
        # 扩散采样
        print("🔄 执行扩散采样...")
        
        def model_fn(x, t):
            """模型函数，支持classifier-free guidance"""
            # 无条件预测
            uncond_pred = self.dit_model.dit(x, t, y=None)
            
            # 有条件预测
            cond_pred = self.dit_model.dit(x, t, y=user_conditions)
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred
            
            return pred
        
        # 使用transport进行采样
        samples = self.transport.sample(
            model_fn,
            noise,
            num_steps=num_steps,
            clip_denoised=True
        )
        
        print(f"  采样完成，潜在特征形状: {samples.shape}")
        
        # 使用VA-VAE解码为图像
        print("🎨 解码为图像...")
        generated_images = self.vavae.decode(samples)
        
        # 后处理：裁剪到[0,1]范围
        generated_images = torch.clamp(generated_images, 0, 1)
        
        print(f"✅ 生成完成，图像形状: {generated_images.shape}")
        
        return generated_images, user_labels
    
    def save_images(self, images, user_labels, output_dir, prefix="generated"):
        """保存生成的图像"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存图像到: {output_dir}")
        
        # 转换为numpy格式
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        # 保存单独的图像
        for i, (image, user_id) in enumerate(zip(images_np, user_labels)):
            # 转换为HWC格式
            image = np.transpose(image, (1, 2, 0))
            
            # 保存为PNG
            filename = f"{prefix}_user{user_id:02d}_{i:03d}.png"
            filepath = output_dir / filename
            
            Image.fromarray(image).save(filepath)
        
        # 创建网格图像
        self.create_grid_image(images, user_labels, output_dir / f"{prefix}_grid.png")
        
        print(f"✅ 保存了 {len(images)} 张图像")
    
    def create_grid_image(self, images, user_labels, output_path):
        """创建网格图像"""
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten() if grid_size > 1 else [axes]
        
        for i in range(grid_size * grid_size):
            ax = axes[i]
            
            if i < num_images:
                # 显示图像
                image = images[i].cpu().numpy()
                image = np.transpose(image, (1, 2, 0))
                
                ax.imshow(image)
                ax.set_title(f'User {user_labels[i]}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 网格图像保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成用户条件化的微多普勒图像')
    parser.add_argument('--dit_checkpoint', type=str, required=True,
                       help='训练好的DiT模型检查点')
    parser.add_argument('--vavae_path', type=str, required=True,
                       help='预训练VA-VAE模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--user_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                       help='要生成的用户ID列表')
    parser.add_argument('--num_samples_per_user', type=int, default=4,
                       help='每个用户生成的样本数')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                       help='Classifier-free guidance强度')
    parser.add_argument('--num_steps', type=int, default=250,
                       help='扩散采样步数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    print("🎯 微多普勒图像生成 - 阶段3")
    print("=" * 50)
    
    # 创建生成器
    generator = MicroDopplerGenerator(
        dit_checkpoint=args.dit_checkpoint,
        vavae_path=args.vavae_path,
        device=args.device
    )
    
    # 生成图像
    generated_images, user_labels = generator.generate(
        user_ids=args.user_ids,
        num_samples_per_user=args.num_samples_per_user,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed
    )
    
    # 保存图像
    generator.save_images(
        images=generated_images,
        user_labels=user_labels,
        output_dir=args.output_dir,
        prefix="micro_doppler"
    )
    
    print("✅ 生成完成!")

if __name__ == "__main__":
    main()
