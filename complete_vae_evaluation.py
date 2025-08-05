#!/usr/bin/env python3
"""
完整的VA-VAE评估脚本
包含MSE、FID等多种评估指标
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import tempfile
import shutil

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

def calculate_fid_score(real_images_dir, fake_images_dir):
    """计算FID分数"""
    try:
        from pytorch_fid import fid_score
        
        print("📊 计算FID分数...")
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_images_dir), str(fake_images_dir)],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048
        )
        
        return fid_value
        
    except ImportError:
        print("⚠️ pytorch_fid未安装，跳过FID计算")
        print("💡 可以运行: pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        return None

def complete_vae_evaluation():
    """完整的VA-VAE评估"""
    print("🚀 完整VA-VAE评估 (MSE + FID)")
    print("="*60)
    
    # 配置
    data_dir = "/kaggle/input/dataset"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
    output_dir = Path("complete_vae_evaluation")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    temp_original_dir = output_dir / "temp_original"
    temp_reconstructed_dir = output_dir / "temp_reconstructed"
    temp_original_dir.mkdir(exist_ok=True)
    temp_reconstructed_dir.mkdir(exist_ok=True)
    
    # 检查文件
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    if not Path(vae_model_path).exists():
        print(f"❌ 模型文件不存在: {vae_model_path}")
        return False
    
    # 加载VA-VAE模型
    print("🔧 加载VA-VAE模型...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['ckpt_path'] = vae_model_path
    
    temp_config = "temp_complete_vavae_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    try:
        vae = VA_VAE(config=temp_config)
        print("✅ VA-VAE模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 收集测试图像（每个用户5张）
    print("📁 收集测试图像...")
    user_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('ID_')]
    user_dirs.sort()
    
    test_images = []
    for user_dir in user_dirs:
        images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
        # 每个用户取5张图像
        selected_images = images[:5] if len(images) >= 5 else images
        for img_path in selected_images:
            test_images.append((img_path, user_dir.name))
    
    print(f"🔍 选择了 {len(test_images)} 张测试图像")
    
    # 处理图像并计算MSE
    results = []
    mse_values = []
    
    print("🔄 处理图像...")
    for i, (image_path, user_id) in enumerate(test_images):
        if i % 20 == 0:
            print(f"   进度: {i}/{len(test_images)}")
        
        try:
            # 加载图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整尺寸并保存原图（用于FID计算）
            original_resized = image.resize((256, 256), Image.LANCZOS)
            original_path = temp_original_dir / f"{user_id}_{i:04d}_original.png"
            original_resized.save(original_path)
            
            # VA-VAE重建
            transform = vae.img_transform(p_hflip=0)
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                latent = vae.encode_images(image_tensor)
                reconstructed_images = vae.decode_to_images(latent)
            
            # 保存重建图像（用于FID计算）
            reconstructed_pil = Image.fromarray(reconstructed_images[0])
            reconstructed_path = temp_reconstructed_dir / f"{user_id}_{i:04d}_reconstructed.png"
            reconstructed_pil.save(reconstructed_path)
            
            # 计算MSE
            original_array = np.array(original_resized)
            reconstructed_array = np.array(reconstructed_pil)
            mse = np.mean((original_array.astype(float) - reconstructed_array.astype(float)) ** 2) / (255.0 ** 2)
            
            mse_values.append(mse)
            results.append({
                'user_id': user_id,
                'image_path': str(image_path),
                'mse': mse
            })
            
        except Exception as e:
            print(f"❌ 处理失败 {image_path}: {e}")
            continue
    
    # MSE统计
    if not results:
        print("❌ 没有成功处理任何图像")
        return False
    
    avg_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    
    print(f"\n📊 MSE评估结果:")
    print(f"   处理图像数量: {len(results)}")
    print(f"   平均MSE: {avg_mse:.6f}")
    print(f"   MSE标准差: {std_mse:.6f}")
    print(f"   MSE范围: {min_mse:.6f} - {max_mse:.6f}")
    
    # 按用户统计MSE
    user_mse = {}
    for result in results:
        user_id = result['user_id']
        if user_id not in user_mse:
            user_mse[user_id] = []
        user_mse[user_id].append(result['mse'])
    
    print(f"\n👥 各用户MSE统计:")
    for user_id in sorted(user_mse.keys()):
        user_avg = np.mean(user_mse[user_id])
        print(f"   {user_id}: {len(user_mse[user_id])}张, 平均MSE={user_avg:.6f}")
    
    # 计算FID
    print(f"\n📊 FID评估...")
    fid_score = calculate_fid_score(temp_original_dir, temp_reconstructed_dir)
    
    if fid_score is not None:
        print(f"✅ FID分数: {fid_score:.4f}")
    else:
        print("❌ FID计算失败")
    
    # 综合评估和建议
    print(f"\n💡 综合评估:")
    print(f"   MSE: {avg_mse:.6f}")
    if fid_score is not None:
        print(f"   FID: {fid_score:.4f}")
    
    print(f"\n🎯 建议:")
    
    # 基于MSE的建议
    if avg_mse < 0.02:
        mse_advice = "MSE表现很好"
        mse_action = "可以直接使用"
    elif avg_mse < 0.03:
        mse_advice = "MSE表现一般"
        mse_action = "建议微调"
    else:
        mse_advice = "MSE表现较差"
        mse_action = "需要重新训练"
    
    # 基于FID的建议
    if fid_score is not None:
        if fid_score < 50:
            fid_advice = "FID表现很好"
            fid_action = "可以直接使用"
        elif fid_score < 100:
            fid_advice = "FID表现一般"
            fid_action = "建议微调"
        else:
            fid_advice = "FID表现较差"
            fid_action = "需要重新训练"
        
        print(f"   {mse_advice} ({mse_action})")
        print(f"   {fid_advice} ({fid_action})")
        
        # 综合建议
        if "直接使用" in mse_action and "直接使用" in fid_action:
            final_action = "✅ 可以直接使用预训练VA-VAE进入阶段2"
        elif "微调" in mse_action or "微调" in fid_action:
            final_action = "⚠️ 建议微调VA-VAE后再进入阶段2"
        else:
            final_action = "❌ 建议重新训练VA-VAE或考虑其他方案"
    else:
        print(f"   {mse_advice} ({mse_action})")
        final_action = f"基于MSE: {mse_action}"
    
    print(f"\n🔄 下一步行动:")
    print(f"   {final_action}")
    
    # 保存详细结果
    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("VA-VAE完整评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理图像数量: {len(results)}\n")
        f.write(f"平均MSE: {avg_mse:.6f}\n")
        f.write(f"MSE标准差: {std_mse:.6f}\n")
        f.write(f"MSE范围: {min_mse:.6f} - {max_mse:.6f}\n")
        if fid_score is not None:
            f.write(f"FID分数: {fid_score:.4f}\n")
        f.write(f"\n下一步建议: {final_action}\n")
    
    print(f"\n📁 详细结果已保存到: {results_file}")
    
    # 清理临时文件
    try:
        shutil.rmtree(temp_original_dir)
        shutil.rmtree(temp_reconstructed_dir)
        os.remove(temp_config)
    except:
        pass
    
    return True

if __name__ == "__main__":
    success = complete_vae_evaluation()
    if success:
        print("\n🎉 完整VA-VAE评估完成！")
    else:
        print("\n❌ VA-VAE评估失败！")
    
    sys.exit(0 if success else 1)
