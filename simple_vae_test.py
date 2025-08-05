#!/usr/bin/env python3
"""
简化版VA-VAE重建测试
直接使用官方方式，避免复杂的封装
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import yaml

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

def test_vae_reconstruction():
    """测试VA-VAE重建效果"""
    print("🚀 简化版VA-VAE重建测试")
    print("="*50)
    
    # 数据路径配置
    data_dir = "/kaggle/input/dataset"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
    
    # 检查文件
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    if not Path(vae_model_path).exists():
        print(f"❌ 模型文件不存在: {vae_model_path}")
        return False
    
    # 更新配置文件
    print("🔧 更新配置文件...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['ckpt_path'] = vae_model_path
    
    temp_config = "temp_vavae_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # 加载VA-VAE模型
    print("🔧 加载VA-VAE模型...")
    try:
        vae = VA_VAE(config=temp_config)
        print("✅ VA-VAE模型加载成功")
        print(f"📊 模型参数量: {sum(p.numel() for p in vae.model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 收集测试图像
    print("📁 收集测试图像...")
    user_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('ID_')]
    user_dirs.sort()
    
    test_images = []
    for user_dir in user_dirs[:5]:  # 只测试前5个用户
        images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
        if images:
            test_images.append((images[0], user_dir.name))  # 每个用户取一张
    
    print(f"🔍 选择了 {len(test_images)} 张测试图像")
    
    # 创建输出目录
    output_dir = Path("simple_vae_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 测试重建
    results = []
    
    for i, (image_path, user_id) in enumerate(test_images):
        print(f"🔍 测试 {i+1}/{len(test_images)}: {user_id}")
        
        try:
            # 加载图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 使用VA-VAE的官方预处理
            transform = vae.img_transform(p_hflip=0)
            image_tensor = transform(image).unsqueeze(0)
            
            # 编码和解码
            with torch.no_grad():
                latent = vae.encode_images(image_tensor)
                reconstructed_images = vae.decode_to_images(latent)
            
            # 转换回PIL图像
            reconstructed_pil = Image.fromarray(reconstructed_images[0])
            
            # 计算简单的像素差异
            original_array = np.array(image.resize((256, 256)))
            reconstructed_array = np.array(reconstructed_pil)
            
            mse = np.mean((original_array.astype(float) - reconstructed_array.astype(float)) ** 2) / (255.0 ** 2)
            
            results.append({
                'user_id': user_id,
                'mse': mse,
                'original': image.resize((256, 256)),
                'reconstructed': reconstructed_pil
            })
            
            print(f"   MSE: {mse:.6f}")
            
            # 保存对比图
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image.resize((256, 256)))
            axes[0].set_title(f'{user_id} - Original')
            axes[0].axis('off')
            
            axes[1].imshow(reconstructed_pil)
            axes[1].set_title(f'Reconstructed\nMSE: {mse:.6f}')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{user_id}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 处理失败 {user_id}: {e}")
            continue
    
    # 统计结果
    if results:
        mse_values = [r['mse'] for r in results]
        avg_mse = np.mean(mse_values)
        
        print(f"\n📊 测试结果:")
        print(f"   测试用户数: {len(results)}")
        print(f"   平均MSE: {avg_mse:.6f}")
        print(f"   MSE范围: {np.min(mse_values):.6f} - {np.max(mse_values):.6f}")
        
        print(f"\n👥 各用户结果:")
        for result in results:
            print(f"   {result['user_id']}: MSE={result['mse']:.6f}")
        
        print(f"\n💡 建议:")
        if avg_mse < 0.01:
            print("   ✅ 重建质量很好！可以直接使用预训练VA-VAE")
        elif avg_mse < 0.05:
            print("   ⚠️ 重建质量一般，建议考虑微调VA-VAE")
        else:
            print("   ❌ 重建质量较差，可能需要重新训练")
        
        print(f"\n📁 对比图像已保存到: {output_dir}/")
        
        return True
    else:
        print("❌ 没有成功处理任何图像")
        return False

if __name__ == "__main__":
    success = test_vae_reconstruction()
    if success:
        print("\n🎉 VA-VAE测试完成！")
    else:
        print("\n❌ VA-VAE测试失败！")
    
    sys.exit(0 if success else 1)
