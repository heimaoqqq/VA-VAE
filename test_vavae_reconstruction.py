#!/usr/bin/env python3
"""
测试VA-VAE的重建质量
验证编码-解码过程是否正确
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, 'LightningDiT')

def test_vavae_reconstruction():
    """测试VA-VAE重建"""
    print("🧪 测试VA-VAE重建质量...")
    
    try:
        from tokenizer.vavae import VA_VAE
        
        # 加载VA-VAE
        vavae = VA_VAE('vavae_config.yaml')
        print("✅ VA-VAE加载成功")
        
        # 创建测试图像 (模拟微多普勒时频图)
        batch_size = 2
        height, width = 256, 256
        
        # 创建具有时频图特征的测试图像
        test_images = []
        for i in range(batch_size):
            # 创建频率-时间网格
            freq = np.linspace(-1, 1, height)
            time = np.linspace(-1, 1, width)
            F, T = np.meshgrid(freq, time, indexing='ij')
            
            # 模拟多普勒频移模式
            doppler_pattern = np.sin(2 * np.pi * F * 3) * np.exp(-T**2 / 0.5)
            doppler_pattern += 0.5 * np.sin(2 * np.pi * F * 5 + T * np.pi)
            
            # 归一化到[0, 1]
            doppler_pattern = (doppler_pattern - doppler_pattern.min()) / (doppler_pattern.max() - doppler_pattern.min())
            
            # 转换为RGB (重复3个通道)
            rgb_image = np.stack([doppler_pattern] * 3, axis=-1)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            test_images.append(rgb_image)
        
        # 转换为tensor
        test_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
            for img in test_images
        ])
        
        print(f"测试图像形状: {test_tensor.shape}")
        print(f"测试图像范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
        
        # 编码
        with torch.no_grad():
            latents = vavae.encode_images(test_tensor)
            print(f"潜在特征形状: {latents.shape}")
            print(f"潜在特征范围: [{latents.min():.3f}, {latents.max():.3f}]")
            
            # 解码
            reconstructed = vavae.decode_to_images(latents)
            print(f"重建图像形状: {reconstructed.shape}")
            print(f"重建图像范围: [{reconstructed.min()}, {reconstructed.max()}]")
        
        # 保存结果
        for i in range(batch_size):
            # 原始图像
            orig_img = Image.fromarray(test_images[i])
            orig_img.save(f"test_original_{i}.png")
            
            # 重建图像
            recon_img = Image.fromarray(reconstructed[i])
            recon_img.save(f"test_reconstructed_{i}.png")
            
            print(f"✅ 保存图像对 {i}: test_original_{i}.png, test_reconstructed_{i}.png")
        
        # 计算重建误差
        mse_errors = []
        for i in range(batch_size):
            orig = test_images[i].astype(np.float32)
            recon = reconstructed[i].astype(np.float32)
            mse = np.mean((orig - recon) ** 2)
            mse_errors.append(mse)
            print(f"图像 {i} MSE: {mse:.2f}")
        
        avg_mse = np.mean(mse_errors)
        print(f"平均MSE: {avg_mse:.2f}")
        
        if avg_mse < 1000:  # 合理的阈值
            print("✅ VA-VAE重建质量良好")
            return True
        else:
            print("⚠️  VA-VAE重建质量较差")
            return False
        
    except Exception as e:
        print(f"❌ VA-VAE重建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_latent_distribution():
    """测试潜在特征分布"""
    print("\n🧪 测试潜在特征分布...")
    
    try:
        from tokenizer.vavae import VA_VAE
        
        vavae = VA_VAE('vavae_config.yaml')
        
        # 测试不同范围的潜在特征
        test_ranges = [
            ("标准正态", torch.randn(2, 32, 16, 16)),
            ("小范围", torch.randn(2, 32, 16, 16) * 0.5),
            ("大范围", torch.randn(2, 32, 16, 16) * 2.0),
            ("归一化", torch.randn(2, 32, 16, 16) / 0.18215),
        ]
        
        for name, latents in test_ranges:
            print(f"\n测试 {name}:")
            print(f"  潜在特征范围: [{latents.min():.3f}, {latents.max():.3f}]")
            
            with torch.no_grad():
                images = vavae.decode_to_images(latents)
                
            print(f"  解码图像范围: [{images.min()}, {images.max()}]")
            
            # 保存第一张图像
            if len(images) > 0:
                img = Image.fromarray(images[0])
                filename = f"test_latent_{name.replace(' ', '_')}.png"
                img.save(filename)
                print(f"  保存: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ 潜在特征分布测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔬 VA-VAE重建测试")
    print("=" * 40)
    
    success = True
    
    # 测试1: 重建质量
    if not test_vavae_reconstruction():
        success = False
    
    # 测试2: 潜在特征分布
    if not test_latent_distribution():
        success = False
    
    if success:
        print("\n✅ VA-VAE测试完成!")
        print("检查生成的图像文件，对比原始和重建质量")
    else:
        print("\n❌ VA-VAE测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
