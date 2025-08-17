#!/usr/bin/env python3
"""
步骤6: 基于官方inference.py的样本生成脚本
用于从微调后的LightningDiT模型生成微多普勒样本
"""

import os
import math
import json
import torch
import numpy as np
from time import time
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# 添加LightningDiT路径
sys.path.append('/kaggle/working/LightningDiT')
from tokenizer.vavae import VA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
import torchvision

def generate_samples(args):
    """
    生成样本 - 基于官方inference.py的do_sample函数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"使用设备: {device}")
    
    # 模型配置
    model_config = {
        'model_type': 'LightningDiT-XL/1',
        'input_size': 16,  # 256/16 = 16
        'num_classes': 32,  # 31个用户 + 1个无条件
        'in_channels': 32,  # VA-VAE f16d32
        'use_qknorm': False,
        'use_swiglu': True,
        'use_rope': True,
        'use_rmsnorm': True,
        'wo_shift': False,
    }
    
    # 创建模型
    model = LightningDiT_models[model_config['model_type']](
        input_size=model_config['input_size'],
        num_classes=model_config['num_classes'],
        in_channels=model_config['in_channels'],
        use_qknorm=model_config['use_qknorm'],
        use_swiglu=model_config['use_swiglu'],
        use_rope=model_config['use_rope'],
        use_rmsnorm=model_config['use_rmsnorm'],
        wo_shift=model_config['wo_shift'],
    )
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    
    if "ema" in checkpoint:  # 支持train.py的检查点格式
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 移除module.前缀（如果有）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建Transport和Sampler（使用官方配置）
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        use_cosine_loss=True,
        use_lognorm=True,
    )
    sampler = Sampler(transport)
    
    # 配置采样函数
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=args.timestep_shift,
    )
    
    # 加载VAE解码器
    print("加载VA-VAE解码器...")
    vae = VA_VAE('tokenizer/configs/vavae_f16d32.yaml')
    vae.to(device)
    
    # 准备生成
    n_samples = args.num_samples
    batch_size = args.batch_size
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    # 准备类别条件
    if args.user_id is not None:
        # 特定用户生成
        y = torch.full((batch_size,), args.user_id, dtype=torch.long, device=device)
        print(f"生成用户{args.user_id}的样本")
    else:
        # 随机用户生成
        y = None
        print("生成随机用户的样本")
    
    # 是否使用CFG
    using_cfg = args.cfg_scale > 1.0
    if using_cfg:
        print(f"使用CFG, scale={args.cfg_scale}, interval_start={args.cfg_interval_start}")
    
    all_samples = []
    
    # 生成循环
    for batch_idx in tqdm(range(num_batches), desc="生成批次"):
        # 计算当前批次大小
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # 生成噪声
        z = torch.randn(current_batch_size, model_config['in_channels'], 
                       model_config['input_size'], model_config['input_size'], 
                       device=device)
        
        # 准备条件
        if y is None:
            # 随机用户
            y_batch = torch.randint(0, 31, (current_batch_size,), device=device)
        else:
            y_batch = y[:current_batch_size]
        
        # 准备模型参数
        model_kwargs = dict(y=y_batch)
        
        # CFG采样
        if using_cfg:
            # 准备CFG的条件和无条件输入
            z_combined = torch.cat([z, z], 0)
            y_null = torch.full_like(y_batch, 31)  # 31是无条件类别
            y_combined = torch.cat([y_batch, y_null], 0)
            model_kwargs = dict(y=y_combined)
            
            # 定义CFG模型函数
            def cfg_model_fn(x, t):
                # 获取条件和无条件预测
                pred = model(x, t, **model_kwargs)
                pred_cond, pred_uncond = pred.chunk(2)
                
                # 应用CFG（带interval）
                if t >= args.cfg_interval_start:
                    pred = pred_uncond + args.cfg_scale * (pred_cond - pred_uncond)
                else:
                    pred = pred_cond
                
                return pred
            
            # 采样
            with torch.no_grad():
                samples = sample_fn(z, cfg_model_fn)
        else:
            # 无CFG采样
            def model_fn(x, t):
                return model(x, t, **model_kwargs)
            
            with torch.no_grad():
                samples = sample_fn(z, model_fn)
        
        # VAE解码
        with torch.no_grad():
            samples = vae.decode(samples / 0.18215)  # 反归一化
            samples = torch.clamp(samples, -1, 1)
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        
        all_samples.append(samples.cpu())
    
    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)[:n_samples]
    
    # 保存样本
    print(f"保存{len(all_samples)}个样本...")
    for i, sample in enumerate(all_samples):
        # 转换为PIL图像
        sample_np = sample.permute(1, 2, 0).numpy()
        sample_np = (sample_np * 255).astype(np.uint8)
        img = Image.fromarray(sample_np)
        
        # 保存
        if args.user_id is not None:
            filename = f"user{args.user_id:02d}_sample_{i:04d}.png"
        else:
            filename = f"sample_{i:04d}.png"
        
        img.save(output_dir / filename)
    
    # 生成网格图像（如果样本数足够）
    if len(all_samples) >= 16:
        grid = torchvision.utils.make_grid(all_samples[:16], nrow=4, padding=2)
        grid_np = grid.permute(1, 2, 0).numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
        grid_img = Image.fromarray(grid_np)
        
        if args.user_id is not None:
            grid_filename = f"user{args.user_id:02d}_grid.png"
        else:
            grid_filename = "samples_grid.png"
        
        grid_img.save(output_dir / grid_filename)
        print(f"保存网格图像: {grid_filename}")
    
    print(f"生成完成! 样本保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='LightningDiT样本生成')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=16,
                       help='生成样本数量')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--user_id', type=int, default=None,
                       help='指定用户ID (0-30)，不指定则随机')
    
    # 采样参数
    parser.add_argument('--sampling_method', type=str, default='dopri5',
                       choices=['euler', 'heun', 'dopri5', 'dopri8'],
                       help='ODE求解器')
    parser.add_argument('--num_steps', type=int, default=150,
                       help='采样步数')
    parser.add_argument('--timestep_shift', type=float, default=0.1,
                       help='时间步偏移')
    
    # CFG参数
    parser.add_argument('--cfg_scale', type=float, default=7.0,
                       help='CFG scale')
    parser.add_argument('--cfg_interval_start', type=float, default=0.11,
                       help='CFG interval start')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/generated_samples',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.user_id is not None:
        assert 0 <= args.user_id <= 30, "user_id必须在0-30之间"
    
    generate_samples(args)

if __name__ == "__main__":
    main()
