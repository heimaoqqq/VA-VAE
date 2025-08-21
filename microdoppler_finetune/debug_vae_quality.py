#!/usr/bin/env python3
"""
调试VA-VAE重建质量，检查是否为生成质量问题的根源
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from PIL import Image

# 添加路径
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')
from tokenizer.vavae import VA_VAE

def test_vae_reconstruction():
    """测试VA-VAE的重建质量"""
    
    # 初始化VA-VAE
    vae_config_path = '/kaggle/working/VA-VAE/microdoppler_finetune/vavae_config_for_dit.yaml'
    vae = VA_VAE(vae_config_path)
    
    # 加载微调的权重
    vae_checkpoint_path = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    if os.path.exists(vae_checkpoint_path):
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤VAE权重
        vae_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('loss.') and not key.startswith('foundation_model.') and key != 'linear_proj.weight':
                vae_state_dict[key] = value
        
        vae.model.load_state_dict(vae_state_dict, strict=True)
        print("✅ VA-VAE权重加载成功")
    
    # 测试数据目录
    test_data_dir = '/kaggle/working/data_resized'
    if not os.path.exists(test_data_dir):
        print(f"❌ 测试数据目录不存在: {test_data_dir}")
        return
    
    # 随机选择几张图像进行重建测试
    image_files = list(Path(test_data_dir).rglob("*.png"))
    if len(image_files) == 0:
        print("❌ 未找到测试图像")
        return
    
    # 测试5张图像
    test_images = np.random.choice(image_files, min(5, len(image_files)), replace=False)
    
    reconstruction_errors = []
    
    fig, axes = plt.subplots(2, len(test_images), figsize=(15, 6))
    
    for i, img_path in enumerate(test_images):
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        with torch.no_grad():
            # 编码
            latent = vae.encode(img_tensor)
            print(f"图像 {i+1} - Latent shape: {latent.shape}, mean: {latent.mean():.4f}, std: {latent.std():.4f}")
            
            # 解码
            reconstructed = vae.decode(latent)
            
            # 计算重建误差
            mse_error = torch.mean((img_tensor - reconstructed) ** 2).item()
            reconstruction_errors.append(mse_error)
            
            print(f"图像 {i+1} - 重建MSE: {mse_error:.4f}")
        
        # 可视化
        axes[0, i].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f'原图 {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(torch.clamp(reconstructed[0], 0, 1).permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f'重建 MSE:{mse_error:.4f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/vae_reconstruction_test.png', dpi=150, bbox_inches='tight')
    print(f"✅ 重建对比图保存至: /kaggle/working/vae_reconstruction_test.png")
    
    # 统计结果
    avg_error = np.mean(reconstruction_errors)
    print(f"\n📊 VA-VAE重建质量统计:")
    print(f"   平均MSE误差: {avg_error:.4f}")
    print(f"   误差范围: {min(reconstruction_errors):.4f} - {max(reconstruction_errors):.4f}")
    
    if avg_error > 0.05:
        print("⚠️  VA-VAE重建误差偏高，可能影响DiT训练质量")
        print("💡 建议：延长VA-VAE微调或调整重建损失权重")
    else:
        print("✅ VA-VAE重建质量良好，问题可能在DiT配置")
    
    return avg_error

def check_latent_statistics():
    """检查latent的统计分布"""
    from safetensors.torch import load_file
    
    # 检查编码后的latent文件
    latent_dir = Path('/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256')
    latent_files = list(latent_dir.glob("*.safetensors"))
    
    if not latent_files:
        print("❌ 未找到latent文件")
        return
    
    all_latents = []
    for file_path in latent_files:
        data = load_file(str(file_path))
        latents = data['latents']  # [B, C, H, W]
        all_latents.append(latents)
    
    all_latents = torch.cat(all_latents, dim=0)
    print(f"📊 Latent统计信息:")
    print(f"   形状: {all_latents.shape}")
    print(f"   均值: {all_latents.mean():.4f}")
    print(f"   标准差: {all_latents.std():.4f}")
    print(f"   最小值: {all_latents.min():.4f}")
    print(f"   最大值: {all_latents.max():.4f}")
    
    # 检查是否有异常值
    outlier_ratio = (torch.abs(all_latents) > 5).float().mean()
    print(f"   异常值比例 (|x|>5): {outlier_ratio:.4f}")
    
    if outlier_ratio > 0.01:
        print("⚠️  发现较多异常值，可能需要调整latent_norm或latent_multiplier")
    
    return all_latents.mean().item(), all_latents.std().item()

if __name__ == "__main__":
    print("🔍 开始VA-VAE重建质量测试...")
    
    # 测试重建质量
    vae_error = test_vae_reconstruction()
    
    print("\n" + "="*60)
    
    # 检查latent统计
    print("📊 检查latent统计分布...")
    check_latent_statistics()
    
    print("\n🎯 调试建议:")
    if vae_error and vae_error > 0.05:
        print("1. VA-VAE重建质量需要改进")
        print("2. 考虑延长VA-VAE Stage3训练")
        print("3. 调整DiT训练时的latent_multiplier")
    else:
        print("1. VA-VAE质量良好，问题在DiT配置")
        print("2. 尝试使用更大的模型 (LightningDiT-L)")
        print("3. 调整loss_weight和训练超参数")
