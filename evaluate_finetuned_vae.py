#!/usr/bin/env python3
"""
评估微调后的VA-VAE效果
对比微调前后的重建质量
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

def load_finetuned_vae(original_model_path, finetuned_weights_path, device='cuda'):
    """加载微调后的VA-VAE模型"""
    try:
        # 加载原始配置
        config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['ckpt_path'] = original_model_path
        
        temp_config = "temp_eval_vavae_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        # 创建模型
        vae = VA_VAE(config=temp_config)
        
        # 加载微调后的权重
        finetuned_weights = torch.load(finetuned_weights_path, map_location=device)
        vae.model.load_state_dict(finetuned_weights)
        vae.model.eval()
        
        print("✅ 微调后的VA-VAE模型加载成功")
        return vae
        
    except Exception as e:
        print(f"❌ 微调模型加载失败: {e}")
        return None

def load_original_vae(model_path, device='cuda'):
    """加载原始VA-VAE模型"""
    try:
        config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['ckpt_path'] = model_path
        
        temp_config = "temp_original_vavae_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        vae = VA_VAE(config=temp_config)
        vae.model.eval()
        
        print("✅ 原始VA-VAE模型加载成功")
        return vae
        
    except Exception as e:
        print(f"❌ 原始模型加载失败: {e}")
        return None

def calculate_fid_score(real_images_dir, fake_images_dir):
    """计算FID分数"""
    try:
        from pytorch_fid import fid_score
        
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_images_dir), str(fake_images_dir)],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048
        )
        
        return fid_value
        
    except ImportError:
        print("⚠️ pytorch_fid未安装，跳过FID计算")
        return None
    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        return None

def compare_models(original_vae, finetuned_vae, data_dir, output_dir, num_samples=50):
    """对比原始和微调后的模型"""
    print("🔍 对比原始模型 vs 微调模型")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    original_dir = output_path / "original_reconstructions"
    finetuned_dir = output_path / "finetuned_reconstructions"
    real_dir = output_path / "real_images"
    
    original_dir.mkdir(exist_ok=True)
    finetuned_dir.mkdir(exist_ok=True)
    real_dir.mkdir(exist_ok=True)
    
    # 收集测试图像
    user_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('ID_')]
    test_images = []
    
    for user_dir in user_dirs:
        images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
        if images:
            test_images.append((images[0], user_dir.name))  # 每个用户取一张
    
    test_images = test_images[:num_samples]
    print(f"📊 测试 {len(test_images)} 张图像")
    
    # 对比重建
    original_mse = []
    finetuned_mse = []
    
    for i, (image_path, user_id) in enumerate(test_images):
        try:
            # 加载图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 保存原图
            original_resized = image.resize((256, 256), Image.LANCZOS)
            original_resized.save(real_dir / f"{user_id}_{i:03d}_real.png")
            
            # 预处理
            transform = original_vae.img_transform(p_hflip=0)
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                # 原始模型重建
                latent_orig = original_vae.encode_images(image_tensor)
                recon_orig = original_vae.decode_to_images(latent_orig)
                recon_orig_pil = Image.fromarray(recon_orig[0])
                recon_orig_pil.save(original_dir / f"{user_id}_{i:03d}_original.png")
                
                # 微调模型重建
                latent_fine = finetuned_vae.encode_images(image_tensor)
                recon_fine = finetuned_vae.decode_to_images(latent_fine)
                recon_fine_pil = Image.fromarray(recon_fine[0])
                recon_fine_pil.save(finetuned_dir / f"{user_id}_{i:03d}_finetuned.png")
                
                # 计算MSE
                original_array = np.array(original_resized)
                recon_orig_array = np.array(recon_orig_pil)
                recon_fine_array = np.array(recon_fine_pil)
                
                mse_orig = np.mean((original_array.astype(float) - recon_orig_array.astype(float)) ** 2) / (255.0 ** 2)
                mse_fine = np.mean((original_array.astype(float) - recon_fine_array.astype(float)) ** 2) / (255.0 ** 2)
                
                original_mse.append(mse_orig)
                finetuned_mse.append(mse_fine)
                
                # 保存对比图
                if i < 10:  # 只保存前10张对比图
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(original_resized)
                    axes[0].set_title(f'{user_id} - Original')
                    axes[0].axis('off')
                    
                    axes[1].imshow(recon_orig_pil)
                    axes[1].set_title(f'Pretrained\nMSE: {mse_orig:.6f}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(recon_fine_pil)
                    axes[2].set_title(f'Fine-tuned\nMSE: {mse_fine:.6f}')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(output_path / f"comparison_{user_id}_{i:03d}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                
        except Exception as e:
            print(f"❌ 处理失败 {image_path}: {e}")
            continue
    
    # 计算统计结果
    if original_mse and finetuned_mse:
        orig_avg = np.mean(original_mse)
        fine_avg = np.mean(finetuned_mse)
        improvement = (orig_avg - fine_avg) / orig_avg * 100
        
        print(f"\n📊 MSE对比结果:")
        print(f"   原始模型平均MSE: {orig_avg:.6f}")
        print(f"   微调模型平均MSE: {fine_avg:.6f}")
        print(f"   改善幅度: {improvement:.1f}%")
        
        # 计算FID
        print(f"\n📊 计算FID分数...")
        original_fid = calculate_fid_score(real_dir, original_dir)
        finetuned_fid = calculate_fid_score(real_dir, finetuned_dir)
        
        if original_fid is not None and finetuned_fid is not None:
            fid_improvement = (original_fid - finetuned_fid) / original_fid * 100
            print(f"   原始模型FID: {original_fid:.4f}")
            print(f"   微调模型FID: {finetuned_fid:.4f}")
            print(f"   FID改善幅度: {fid_improvement:.1f}%")
        
        # 保存结果
        results = {
            'original_mse': orig_avg,
            'finetuned_mse': fine_avg,
            'mse_improvement': improvement,
            'original_fid': original_fid,
            'finetuned_fid': finetuned_fid,
            'fid_improvement': fid_improvement if original_fid and finetuned_fid else None
        }
        
        with open(output_path / "comparison_results.txt", 'w') as f:
            f.write("VA-VAE微调效果对比\n")
            f.write("=" * 30 + "\n")
            f.write(f"测试图像数量: {len(original_mse)}\n")
            f.write(f"原始模型平均MSE: {orig_avg:.6f}\n")
            f.write(f"微调模型平均MSE: {fine_avg:.6f}\n")
            f.write(f"MSE改善幅度: {improvement:.1f}%\n")
            if original_fid and finetuned_fid:
                f.write(f"原始模型FID: {original_fid:.4f}\n")
                f.write(f"微调模型FID: {finetuned_fid:.4f}\n")
                f.write(f"FID改善幅度: {fid_improvement:.1f}%\n")
        
        return results
    
    return None

def main():
    """主函数"""
    print("📊 评估微调后的VA-VAE效果")
    print("="*50)
    
    # 路径配置
    data_dir = "/kaggle/input/dataset"
    original_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    finetuned_weights_path = "vavae_finetuned/finetuned_vavae.pt"
    output_dir = "vae_comparison_results"
    
    # 检查文件
    if not Path(finetuned_weights_path).exists():
        print(f"❌ 微调模型不存在: {finetuned_weights_path}")
        print("💡 请先运行 run_vavae_finetune.py")
        return False
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    original_vae = load_original_vae(original_model_path, device)
    finetuned_vae = load_finetuned_vae(original_model_path, finetuned_weights_path, device)
    
    if original_vae is None or finetuned_vae is None:
        print("❌ 模型加载失败")
        return False
    
    # 对比评估
    results = compare_models(original_vae, finetuned_vae, data_dir, output_dir)
    
    if results:
        print(f"\n🎉 评估完成！")
        print(f"📁 详细结果: {output_dir}/")
        print(f"📊 对比图像: {output_dir}/comparison_*.png")
        
        # 给出建议
        mse_improvement = results['mse_improvement']
        if mse_improvement > 20:
            print(f"✅ 微调效果显著！建议使用微调后的模型")
        elif mse_improvement > 10:
            print(f"⚠️ 微调有一定改善，可以考虑使用")
        else:
            print(f"❌ 微调改善有限，可能不值得使用")
        
        return True
    else:
        print("❌ 评估失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
