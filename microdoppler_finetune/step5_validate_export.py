#!/usr/bin/env python3
"""
Step 5: 验证训练结果并导出最终模型
用于Kaggle输出和下游DiT训练
"""

import os
import sys
import argparse
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加LightningDiT路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root / 'LightningDiT'))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model(checkpoint_path, config_path=None):
    """加载训练好的模型"""
    
    print(f"📦 加载模型: {checkpoint_path}")
    
    # 加载配置
    if config_path and Path(config_path).exists():
        config = OmegaConf.load(config_path)
    else:
        # 使用默认配置
        config = OmegaConf.create({
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 32,
                'use_vf': 'dinov2',
                'reverse_proj': True,
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [16],
                    'dropout': 0.0
                },
                'lossconfig': {
                    'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                    'params': {
                        'disc_start': 1,
                        'vf_weight': 0.1,
                        'distmat_weight': 1.0,
                        'cos_weight': 1.0
                    }
                }
            }
        })
    
    # 实例化模型
    model = instantiate_from_config(config if isinstance(config, dict) else config.model)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model


def validate_reconstruction(model, data_root, split_file, num_samples=16, device='cuda'):
    """验证重建质量"""
    
    print("\n🔍 验证重建质量...")
    
    # 加载数据
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    val_data = split_data['val'][:num_samples]
    
    # 准备图像
    images = []
    reconstructions = []
    
    model = model.to(device)
    
    with torch.no_grad():
        for item in tqdm(val_data, desc="处理图像"):
            # 加载图像
            img_path = Path(data_root) / item['path']
            if not img_path.exists():
                continue
                
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.LANCZOS)
            
            # 转换为tensor
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 重建
            reconstructed, _, _, _ = model(img_tensor)
            
            images.append(img_tensor.cpu())
            reconstructions.append(reconstructed.cpu())
    
    # 计算指标
    images_cat = torch.cat(images, dim=0)
    recons_cat = torch.cat(reconstructions, dim=0)
    
    # MSE
    mse = torch.mean((images_cat - recons_cat) ** 2).item()
    
    # PSNR
    psnr = 20 * np.log10(2.0) - 10 * np.log10(mse)
    
    print(f"✅ 重建指标:")
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    # 保存可视化
    save_reconstruction_grid(images_cat, recons_cat, 'reconstruction_results.png')
    
    return mse, psnr


def save_reconstruction_grid(images, reconstructions, save_path, num_show=8):
    """保存重建对比图"""
    
    num_show = min(num_show, len(images))
    
    fig, axes = plt.subplots(2, num_show, figsize=(num_show * 2, 4))
    
    for i in range(num_show):
        # 原图
        img = images[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # 重建
        rec = reconstructions[i].permute(1, 2, 0).numpy()
        rec = (rec + 1) / 2
        axes[1, i].imshow(np.clip(rec, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 保存重建对比图: {save_path}")


def extract_latent_statistics(model, data_root, split_file, device='cuda'):
    """提取潜在空间统计信息"""
    
    print("\n📈 提取潜在空间统计...")
    
    # 加载数据
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    train_data = split_data['train']
    
    model = model.to(device)
    
    # 收集潜在向量
    all_latents = []
    user_latents = {}
    
    with torch.no_grad():
        for item in tqdm(train_data, desc="编码图像"):
            img_path = Path(data_root) / item['path']
            if not img_path.exists():
                continue
            
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 编码
            posterior = model.encode(img_tensor)
            z = posterior.sample()
            
            all_latents.append(z.cpu())
            
            # 按用户分组
            user_id = item['user_id']
            if user_id not in user_latents:
                user_latents[user_id] = []
            user_latents[user_id].append(z.cpu())
    
    # 计算全局统计
    all_latents = torch.cat(all_latents, dim=0)
    mean = all_latents.mean(dim=[0, 2, 3])  # [C]
    std = all_latents.std(dim=[0, 2, 3])    # [C]
    
    # 计算用户间差异
    user_means = {}
    for user_id, latents in user_latents.items():
        user_latents_cat = torch.cat(latents, dim=0)
        user_means[user_id] = user_latents_cat.mean(dim=[0, 2, 3])
    
    # 保存统计信息
    stats = {
        'global_mean': mean.numpy().tolist(),
        'global_std': std.numpy().tolist(),
        'num_samples': len(all_latents),
        'latent_dim': all_latents.shape[1],
        'spatial_size': [all_latents.shape[2], all_latents.shape[3]]
    }
    
    stats_path = 'latent_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 潜在空间统计:")
    print(f"   维度: {stats['latent_dim']} x {stats['spatial_size'][0]} x {stats['spatial_size'][1]}")
    print(f"   样本数: {stats['num_samples']}")
    print(f"   均值范围: [{min(mean):.3f}, {max(mean):.3f}]")
    print(f"   标准差范围: [{min(std):.3f}, {max(std):.3f}]")
    print(f"📊 统计信息保存至: {stats_path}")
    
    return stats


def export_for_dit(checkpoint_path, output_path):
    """导出模型用于DiT训练"""
    
    print(f"\n📦 导出模型用于DiT训练...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取必要的组件
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # 只保留编码器和量化层（DiT只需要编码器）
    encoder_keys = [k for k in state_dict.keys() if 
                   k.startswith('encoder.') or 
                   k.startswith('quant_conv.') or
                   k.startswith('linear_proj.')]
    
    encoder_state = {k: state_dict[k] for k in encoder_keys}
    
    # 保存DiT版本
    dit_checkpoint = {
        'encoder_state_dict': encoder_state,
        'full_state_dict': state_dict,  # 也保留完整版本
        'config': {
            'embed_dim': 32,
            'z_channels': 32,
            'use_vf': 'dinov2',
            'reverse_proj': True,
            'resolution': 256
        },
        'type': 'vavae_encoder_for_dit'
    }
    
    torch.save(dit_checkpoint, output_path)
    print(f"✅ DiT编码器导出至: {output_path}")
    
    # 检查文件大小
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   文件大小: {file_size:.2f} MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/stage3/last.ckpt',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str,
                       default='checkpoints/stage3/config.yaml',
                       help='模型配置文件')
    parser.add_argument('--data_root', type=str,
                       default='/kaggle/input/micro-doppler-data',
                       help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                       default='dataset_split.json',
                       help='数据划分文件')
    
    # 功能选择
    parser.add_argument('--validate', action='store_true',
                       help='验证重建质量')
    parser.add_argument('--extract_stats', action='store_true',
                       help='提取潜在空间统计')
    parser.add_argument('--export_dit', action='store_true',
                       help='导出DiT编码器')
    parser.add_argument('--all', action='store_true',
                       help='执行所有功能')
    
    # Kaggle标志
    parser.add_argument('--kaggle', action='store_true',
                       help='Kaggle环境标志')
    
    args = parser.parse_args()
    
    # Kaggle环境检测
    if args.kaggle:
        kaggle_input = Path('/kaggle/input')
        kaggle_working = Path('/kaggle/working')
        if kaggle_input.exists():
            print("✅ 检测到Kaggle环境")
            # 查找checkpoint
            if (kaggle_working / 'checkpoints').exists():
                ckpt_dir = kaggle_working / 'checkpoints'
                # 查找最新阶段
                for stage in [3, 2, 1]:
                    stage_dir = ckpt_dir / f'stage{stage}'
                    if stage_dir.exists() and (stage_dir / 'last.ckpt').exists():
                        args.checkpoint = str(stage_dir / 'last.ckpt')
                        args.config = str(stage_dir / 'config.yaml')
                        print(f"使用第{stage}阶段checkpoint")
                        break
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, args.config)
    
    # 执行功能
    if args.all:
        args.validate = args.extract_stats = args.export_dit = True
    
    if args.validate:
        validate_reconstruction(model, args.data_root, args.split_file, device=device)
    
    if args.extract_stats:
        extract_latent_statistics(model, args.data_root, args.split_file, device=device)
    
    if args.export_dit:
        output_path = 'vavae_encoder_for_dit.pt'
        export_for_dit(args.checkpoint, output_path)
    
    print("\n✅ 所有任务完成!")


if __name__ == '__main__':
    main()
