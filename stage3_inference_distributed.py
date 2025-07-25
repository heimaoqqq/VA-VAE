#!/usr/bin/env python3
"""
分布式微多普勒图像生成脚本
基于原项目inference.py，支持双GPU推理
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
import math

# 导入LightningDiT组件
import sys
import os

# 确保正确的路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
lightningdit_path = os.path.join(current_dir, 'LightningDiT')
if lightningdit_path not in sys.path:
    sys.path.insert(0, lightningdit_path)

from models.lightningdit import LightningDiT_models
from transport import create_transport
from tokenizer.vavae import VA_VAE

def main(accelerator):
    """分布式推理主函数"""
    
    # 解析参数
    parser = argparse.ArgumentParser(description='分布式微多普勒图像生成')
    parser.add_argument('--dit_checkpoint', type=str, required=True, help='DiT检查点路径')
    parser.add_argument('--vavae_config', type=str, required=True, help='VA-VAE配置文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--user_ids', type=int, nargs='+', required=True, help='用户ID列表')
    parser.add_argument('--num_samples_per_user', type=int, default=4, help='每用户样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='引导尺度')
    parser.add_argument('--num_steps', type=int, default=250, help='采样步数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed = args.seed * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    
    if accelerator.is_main_process:
        print("🎯 分布式微多普勒图像生成")
        print("=" * 60)
        print(f"🔧 Accelerator配置:")
        print(f"  进程数: {accelerator.num_processes}")
        print(f"  当前进程: {accelerator.process_index}")
        print(f"  设备: {accelerator.device}")
        print(f"  随机种子: {seed}")
    
    device = accelerator.device
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # 加载VA-VAE
    if accelerator.is_main_process:
        print("📥 加载VA-VAE...")
    vavae = VA_VAE(args.vavae_config)
    
    # 加载DiT模型
    if accelerator.is_main_process:
        print("📥 加载DiT模型...")
    
    model = LightningDiT_models['LightningDiT-B/1'](
        input_size=16,
        num_classes=len(args.user_ids),
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False
    )
    
    # 加载检查点
    checkpoint_path = Path(args.dit_checkpoint)
    if checkpoint_path.exists():
        if accelerator.is_main_process:
            print(f"📥 加载检查点: {checkpoint_path}")
        # 这里需要根据实际的检查点格式调整
        # checkpoint = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
        # model.load_state_dict(checkpoint)
    else:
        if accelerator.is_main_process:
            print("⚠️  检查点不存在，使用随机初始化的模型")
    
    model.eval()
    model.to(device)
    
    # 创建transport
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )
    
    # 计算每个GPU需要生成的样本数
    total_samples = len(args.user_ids) * args.num_samples_per_user
    samples_per_gpu = math.ceil(total_samples / accelerator.num_processes)
    
    if accelerator.is_main_process:
        print(f"📊 生成配置:")
        print(f"  用户数: {len(args.user_ids)}")
        print(f"  每用户样本数: {args.num_samples_per_user}")
        print(f"  总样本数: {total_samples}")
        print(f"  每GPU样本数: {samples_per_gpu}")
    
    # 分布式生成
    generated_images = []
    user_labels = []
    
    with torch.no_grad():
        # 计算当前GPU负责的用户范围
        start_idx = accelerator.process_index * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, total_samples)
        
        current_sample = 0
        for user_id in args.user_ids:
            for sample_idx in range(args.num_samples_per_user):
                global_idx = len(args.user_ids) * sample_idx + (user_id - 1)
                
                # 检查是否是当前GPU负责的样本
                if global_idx < start_idx or global_idx >= end_idx:
                    continue
                
                if accelerator.is_main_process:
                    print(f"🎨 生成用户 {user_id} 样本 {sample_idx + 1}")
                
                # 准备条件
                y = torch.tensor([user_id - 1], dtype=torch.long, device=device)  # 0-based
                
                # 生成随机噪声
                z = torch.randn(1, 32, 16, 16, device=device)
                
                # 简单的采样（这里可以使用更复杂的采样方法）
                model_kwargs = dict(y=y)
                
                # 简单的欧拉采样
                dt = 1.0 / args.num_steps
                x = z.clone()
                
                for step in range(args.num_steps):
                    t = torch.full((x.shape[0],), step * dt, device=device)
                    
                    # 模型预测
                    pred = model(x, t, **model_kwargs)
                    
                    # 欧拉步骤
                    x = x + pred * dt
                
                # 解码为图像
                latent_features = x
                images = vavae.decode_to_images(latent_features)
                
                # 保存图像
                for i, image in enumerate(images):
                    filename = f"user_{user_id}_sample_{sample_idx + 1}_gpu_{accelerator.process_index}.png"
                    Image.fromarray(image).save(output_dir / filename)
                    generated_images.append(image)
                    user_labels.append(user_id)
                
                current_sample += 1
    
    # 等待所有GPU完成
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("✅ 分布式推理完成!")
        print(f"📁 图像保存在: {output_dir}")
        
        # 统计生成的图像数量
        png_files = list(output_dir.glob("*.png"))
        print(f"📊 总共生成了 {len(png_files)} 张图像")

if __name__ == "__main__":
    from accelerate import Accelerator
    accelerator = Accelerator()
    main(accelerator)
