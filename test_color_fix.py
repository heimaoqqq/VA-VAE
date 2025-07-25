#!/usr/bin/env python3
"""
测试颜色修复
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, 'LightningDiT')

def test_vavae_decode():
    """测试VA-VAE解码过程"""
    print("🧪 测试VA-VAE解码过程...")
    
    try:
        from tokenizer.vavae import VA_VAE
        
        # 加载VA-VAE
        vavae = VA_VAE('vavae_config.yaml')
        print("✅ VA-VAE加载成功")
        
        # 创建测试潜在特征
        batch_size = 2
        latent_dim = 32  # 根据配置调整
        height, width = 16, 16  # 根据配置调整
        
        # 测试不同范围的潜在特征
        test_ranges = [
            ("标准正态分布", torch.randn(batch_size, latent_dim, height, width)),
            ("[-1, 1]范围", torch.rand(batch_size, latent_dim, height, width) * 2 - 1),
            ("[0, 1]范围", torch.rand(batch_size, latent_dim, height, width)),
            ("较小范围[-0.5, 0.5]", torch.rand(batch_size, latent_dim, height, width) - 0.5),
        ]
        
        for name, z in test_ranges:
            print(f"\n🔍 测试 {name}:")
            print(f"  潜在特征范围: [{z.min():.3f}, {z.max():.3f}]")
            
            # 解码
            with torch.no_grad():
                images = vavae.decode_to_images(z)
            
            print(f"  解码图像形状: {images.shape}")
            print(f"  解码图像范围: [{images.min()}, {images.max()}]")
            print(f"  解码图像类型: {images.dtype}")
            
            # 保存第一张图像
            if len(images) > 0:
                img = images[0]
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                pil_img = Image.fromarray(img)
                filename = f"test_decode_{name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_').replace('.', 'p')}.png"
                pil_img.save(filename)
                print(f"  保存图像: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ VA-VAE测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_range_conversion():
    """测试数据范围转换"""
    print("\n🧪 测试数据范围转换...")
    
    # 模拟不同范围的数据
    test_data = torch.randn(1, 3, 64, 64)
    
    print(f"原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # 测试不同的归一化方法
    methods = [
        ("tanh", torch.tanh(test_data)),
        ("clamp+tanh", torch.tanh(torch.clamp(test_data, -3, 3))),
        ("sigmoid*2-1", torch.sigmoid(test_data) * 2 - 1),
        ("直接clamp", torch.clamp(test_data, -1, 1)),
    ]
    
    for name, normalized in methods:
        print(f"{name}: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # 模拟VA-VAE解码公式
        decoded = torch.clamp(127.5 * normalized + 128.0, 0, 255)
        print(f"  解码后: [{decoded.min():.1f}, {decoded.max():.1f}]")

def test_color_mapping():
    """测试颜色映射"""
    print("\n🧪 测试颜色映射...")
    
    # 创建测试图像
    height, width = 64, 64
    
    # 创建渐变图像
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    test_image = np.tile(gradient, (height, 1))
    test_image = np.stack([test_image, test_image, test_image], axis=-1)
    
    # 保存原始图像
    Image.fromarray(test_image).save("test_gradient_original.png")
    print("✅ 保存原始渐变图像: test_gradient_original.png")
    
    # 测试不同的颜色空间转换
    import cv2
    
    # RGB to HSV
    hsv = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
    Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).save("test_gradient_hsv.png")
    
    # RGB to LAB
    lab = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
    Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)).save("test_gradient_lab.png")
    
    print("✅ 保存颜色空间测试图像")

def main():
    """主函数"""
    print("🎨 颜色修复测试")
    print("=" * 40)
    
    success = True
    
    # 测试1: VA-VAE解码
    if not test_vavae_decode():
        success = False
    
    # 测试2: 数据范围转换
    test_data_range_conversion()
    
    # 测试3: 颜色映射
    test_color_mapping()
    
    if success:
        print("\n✅ 颜色测试完成!")
        print("现在可以运行修复后的推理:")
        print("python stage3_inference.py \\")
        print("    --dit_checkpoint /kaggle/working/trained_models/best_model \\")
        print("    --vavae_config vavae_config.yaml \\")
        print("    --output_dir /kaggle/working/generated_images \\")
        print("    --user_ids 1 2 3 4 5 \\")
        print("    --num_samples_per_user 2 \\")
        print("    --seed 42")
    else:
        print("\n❌ 颜色测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
