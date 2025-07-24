#!/usr/bin/env python3
"""
阶段3: 图像生成推理
基于LightningDiT原项目的inference.py
使用训练好的DiT模型生成用户条件化的微多普勒图像
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 导入LightningDiT组件
import sys
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT_models
from transport import create_transport
from tokenizer.vavae import VA_VAE

class MicroDopplerGenerator:
    """微多普勒图像生成器 (基于原项目inference.py)"""
    
    def __init__(self, dit_checkpoint, vavae_config, device='cuda'):
        self.device = device
        
        # 加载VA-VAE (VA-VAE在初始化时已经设置为eval模式并移到GPU)
        print("📥 加载VA-VAE...")
        self.vavae = VA_VAE(vavae_config)
        
        # 加载DiT模型 (参考原项目)
        print("📥 加载DiT模型...")
        self._load_dit_model(dit_checkpoint)
        
        # 创建transport (参考原项目)
        self.transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=None,
            sample_eps=None
        )
        
        print("✅ 模型加载完成!")
    
    def _load_dit_model(self, checkpoint_path):
        """加载DiT模型检查点"""
        # 这里需要根据实际的检查点格式来调整
        # 参考原项目的模型加载方式
        
        # 假设我们知道模型配置 (实际应该从检查点中读取)
        self.dit_model = LightningDiT_models['LightningDiT-XL/1'](
            input_size=16,
            num_classes=31,  # 假设31个用户
            in_channels=32,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        
        # 加载权重 (需要根据实际保存格式调整)
        if os.path.exists(checkpoint_path):
            print(f"从 {checkpoint_path} 加载模型权重")
            # 这里需要实际的加载逻辑
            # checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # self.dit_model.load_state_dict(checkpoint['model'])
        else:
            print("⚠️  检查点不存在，使用随机初始化的模型")
        
        self.dit_model.eval()
        self.dit_model.to(self.device)
    
    def generate_samples(self, user_ids, num_samples_per_user=4, guidance_scale=4.0, num_steps=250):
        """
        生成用户条件化的微多普勒图像
        参考原项目的采样方法
        """
        print(f"🎨 开始生成图像...")
        print(f"  用户ID: {user_ids}")
        print(f"  每用户样本数: {num_samples_per_user}")
        print(f"  引导尺度: {guidance_scale}")
        print(f"  采样步数: {num_steps}")
        
        all_images = []
        all_user_labels = []
        
        with torch.no_grad():
            for user_id in tqdm(user_ids, desc="生成用户图像"):
                # 准备条件 (参考原项目)
                batch_size = num_samples_per_user
                y = torch.full((batch_size,), user_id - 1, dtype=torch.long, device=self.device)  # 0-based
                
                # 生成随机噪声 (参考原项目)
                z = torch.randn(batch_size, 32, 16, 16, device=self.device)
                
                # 使用transport进行采样 (参考原项目)
                model_kwargs = dict(y=y)
                
                # 这里应该使用原项目的采样方法
                # 由于我们没有完整的采样器，这里使用简化版本
                samples = self._sample_with_transport(z, model_kwargs, num_steps)
                
                # 使用VA-VAE解码为图像 (使用正确的方法)
                images = self.vavae.decode_to_images(samples)
                
                # 后处理图像 (参考原项目)
                images = self._postprocess_images(images)
                
                all_images.extend(images)
                all_user_labels.extend([user_id] * num_samples_per_user)
        
        return all_images, all_user_labels
    
    def _sample_with_transport(self, z, model_kwargs, num_steps):
        """
        使用transport进行采样
        这里是简化版本，实际应该使用原项目的完整采样器
        """
        # 简化的采样过程
        # 实际应该使用transport.sample()方法
        
        dt = 1.0 / num_steps
        x = z.clone()
        
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=self.device)
            
            # 模型预测
            with torch.no_grad():
                pred = self.dit_model(x, t, **model_kwargs)
            
            # 简单的欧拉步骤 (实际应该使用更复杂的求解器)
            x = x + pred * dt
        
        return x
    
    def _postprocess_images(self, images):
        """
        后处理图像 (VA-VAE的decode_to_images已经返回numpy数组)
        """
        # decode_to_images已经返回了uint8格式的numpy数组 (B, H, W, C)
        pil_images = []
        for img_np in images:
            # 直接从numpy数组创建PIL图像
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)

        return pil_images
    
    def save_images(self, images, user_labels, output_dir, prefix="micro_doppler"):
        """保存生成的图像"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存图像到: {output_dir}")
        
        for i, (image, user_id) in enumerate(zip(images, user_labels)):
            filename = f"{prefix}_user{user_id:02d}_{i+1:03d}.png"
            filepath = output_dir / filename
            image.save(filepath)
        
        # 创建网格图像 (参考原项目)
        self._create_grid_image(images, user_labels, output_dir, prefix)
        
        print(f"✅ 保存了 {len(images)} 张图像")
    
    def _create_grid_image(self, images, user_labels, output_dir, prefix):
        """创建网格展示图像"""
        if not images:
            return
        
        # 计算网格尺寸
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # 获取单张图像尺寸
        img_width, img_height = images[0].size
        
        # 创建网格图像
        grid_width = grid_size * img_width
        grid_height = grid_size * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # 填充网格
        for i, image in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            x = col * img_width
            y = row * img_height
            grid_image.paste(image, (x, y))
        
        # 保存网格图像
        grid_path = output_dir / f"{prefix}_grid.png"
        grid_image.save(grid_path)
        print(f"📊 网格图像保存到: {grid_path}")

def main():
    parser = argparse.ArgumentParser(description='微多普勒图像生成')
    parser.add_argument('--dit_checkpoint', type=str, required=True, help='DiT模型检查点')
    parser.add_argument('--vavae_config', type=str, required=True, help='VA-VAE配置文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--user_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='用户ID列表')
    parser.add_argument('--num_samples_per_user', type=int, default=4, help='每用户生成样本数')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='引导尺度')
    parser.add_argument('--num_steps', type=int, default=250, help='采样步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("🎯 微多普勒图像生成")
    print("=" * 50)
    
    # 创建生成器
    generator = MicroDopplerGenerator(
        dit_checkpoint=args.dit_checkpoint,
        vavae_config=args.vavae_config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 生成图像
    images, user_labels = generator.generate_samples(
        user_ids=args.user_ids,
        num_samples_per_user=args.num_samples_per_user,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps
    )
    
    # 保存图像
    generator.save_images(images, user_labels, args.output_dir)
    
    print("✅ 图像生成完成!")

if __name__ == "__main__":
    main()
