"""
诊断latent缩放问题
"""

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from PIL import Image
import sys
sys.path.append('/kaggle/working/LightningDiT')
sys.path.append('/kaggle/working')

def diagnose_latent_statistics():
    """诊断latent数据集的统计特性"""
    
    # 加载一个latent文件样本
    latent_dir = Path('/kaggle/working/latent_dataset/train')  # 修正路径
    latent_files = list(latent_dir.glob('*.safetensors'))
    
    if not latent_files:
        print("❌ 未找到latent文件")
        return
    
    # 加载第一个文件
    data = load_file(str(latent_files[0]))
    latents = data['latents']
    
    print("=" * 60)
    print("📊 Latent统计分析")
    print("=" * 60)
    print(f"Latent形状: {latents.shape}")
    print(f"数据类型: {latents.dtype}")
    print(f"\n数值范围:")
    print(f"  最小值: {latents.min().item():.6f}")
    print(f"  最大值: {latents.max().item():.6f}")
    print(f"  均值: {latents.mean().item():.6f}")
    print(f"  标准差: {latents.std().item():.6f}")
    
    # 检查是否可能已经应用了0.18215缩放
    print(f"\n🔍 缩放因子分析:")
    expected_std_with_scaling = 0.18215  # 如果应用了SD缩放
    actual_std = latents.std().item()
    
    if abs(actual_std - expected_std_with_scaling) < 0.05:
        print(f"  ⚠️ 检测到可能使用了0.18215缩放因子")
        print(f"  实际std: {actual_std:.6f} ≈ 0.18215")
        print(f"  建议: 需要在解码时除以0.18215")
    else:
        print(f"  ✅ 未检测到0.18215缩放")
        print(f"  实际std: {actual_std:.6f}")
    
    # 测试解码
    print(f"\n🖼️ 测试直接解码（不进行反归一化）:")
    
    # 加载VA-VAE
    sys.path.append('/kaggle/working/VA-VAE')
    from microdoppler_finetune.step6_encode_dataset import load_vavae_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    vae = load_vavae_model(vae_checkpoint, device)
    vae.eval()
    
    # 取第一个样本
    sample_latent = latents[0:1].to(device)  # [1, C, H, W]
    
    print(f"\n测试不同的解码方式:")
    
    # 方式1: 直接解码（不做任何处理）
    with torch.no_grad():
        decoded1 = vae.decode(sample_latent)
        decoded1 = torch.clamp(decoded1, -1, 1)
        decoded1 = (decoded1 + 1) / 2  # [-1,1] -> [0,1]
        img1 = (decoded1[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img1).save('/kaggle/working/test_decode_direct.png')
        print(f"  1. 直接解码: 已保存到 test_decode_direct.png")
        print(f"     值范围: [{decoded1.min():.3f}, {decoded1.max():.3f}]")
    
    # 方式2: 除以0.18215后解码（如果数据带有SD缩放）
    with torch.no_grad():
        decoded2 = vae.decode(sample_latent / 0.18215)
        decoded2 = torch.clamp(decoded2, -1, 1)
        decoded2 = (decoded2 + 1) / 2
        img2 = (decoded2[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img2).save('/kaggle/working/test_decode_descaled.png')
        print(f"  2. 除以0.18215后解码: 已保存到 test_decode_descaled.png")
        print(f"     值范围: [{decoded2.min():.3f}, {decoded2.max():.3f}]")
    
    # 方式3: 使用VA_VAE的decode_to_images（如果存在）
    if hasattr(vae, 'decode_to_images'):
        from tokenizer.vavae import VA_VAE
        vae_wrapper = VA_VAE('/kaggle/working/VA-VAE/microdoppler_finetune/vavae_config_for_dit.yaml')
        with torch.no_grad():
            img3 = vae_wrapper.decode_to_images(sample_latent)[0]
            Image.fromarray(img3).save('/kaggle/working/test_decode_vavae.png')
            print(f"  3. VA_VAE.decode_to_images: 已保存到 test_decode_vavae.png")
    
    print("\n📌 建议:")
    print("  查看生成的测试图像，判断哪种解码方式产生正确的结果")
    print("  如果图像颜色异常，可能需要:")
    print("  1. 重新编码数据集（使用修复后的step6_encode_dataset.py）")
    print("  2. 或在训练代码中添加相应的缩放/反缩放逻辑")

if __name__ == "__main__":
    diagnose_latent_statistics()
