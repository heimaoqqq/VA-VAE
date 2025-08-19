"""
诊断latent缩放问题 - 增强版
提供更准确的缩放检测和多种测试
"""

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from PIL import Image
import sys
import os
sys.path.append('/kaggle/working/LightningDiT')
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/working/VA-VAE')

def diagnose_latent_statistics():
    """诊断latent数据集的统计特性 - 增强版"""
    
    # 加载latent文件样本
    latent_dir = Path('/kaggle/working/latent_dataset/train')  # 修正路径
    latent_files = list(latent_dir.glob('*.safetensors'))
    
    if not latent_files:
        print("❌ 未找到latent文件")
        return
    
    print("=" * 70)
    print("🔬 VA-VAE Latent缩放诊断工具 v2.0")
    print("=" * 70)
    
    # 分析多个文件以获得更准确的统计
    num_files_to_analyze = min(5, len(latent_files))
    all_latents = []
    
    print(f"\n📊 分析{num_files_to_analyze}个latent文件的统计特性...")
    for i in range(num_files_to_analyze):
        data = load_file(str(latent_files[i]))
        latents = data['latents']
        all_latents.append(latents)
        print(f"  文件{i+1}: shape={latents.shape}, std={latents.std().item():.6f}")
    
    # 合并所有latents进行统计
    combined_latents = torch.cat(all_latents, dim=0)
    
    print(f"\n📈 综合统计结果:")
    print(f"  总样本数: {combined_latents.shape[0]}")
    print(f"  Latent形状: {combined_latents.shape}")
    print(f"  数据类型: {combined_latents.dtype}")
    print(f"\n  数值范围:")
    print(f"    最小值: {combined_latents.min().item():.6f}")
    print(f"    最大值: {combined_latents.max().item():.6f}")
    print(f"    均值: {combined_latents.mean().item():.6f}")
    print(f"    标准差: {combined_latents.std().item():.6f}")
    print(f"    中位数: {combined_latents.median().item():.6f}")
    
    # 更精确的缩放检测
    print(f"\n🔍 缩放因子检测:")
    actual_std = combined_latents.std().item()
    actual_mean = combined_latents.mean().item()
    
    # 检测标准差是否接近常见缩放因子
    scaling_factors = {
        0.18215: "Stable Diffusion VAE标准缩放",
        1.0: "无缩放（原始latent）",
        0.5: "半缩放"
    }
    
    detected_scaling = None
    for factor, description in scaling_factors.items():
        if abs(actual_std - factor) < 0.02:  # 更严格的阈值
            detected_scaling = factor
            print(f"  ✅ 检测到缩放因子: {factor} ({description})")
            print(f"     实际std={actual_std:.6f} ≈ {factor}")
            break
    
    if detected_scaling is None:
        print(f"  ⚠️ 未检测到标准缩放因子")
        print(f"     实际std={actual_std:.6f}")
        print(f"     可能是自定义缩放或未缩放的原始latent")
    
    # 基于统计推断是否需要缩放
    print(f"\n💡 缩放推断:")
    if actual_std < 0.3 and abs(actual_std - 0.18215) < 0.05:
        print(f"  📌 很可能已应用0.18215缩放")
        print(f"     理由: std≈0.18215，这是SD-VAE的标准")
        needs_descaling = True
    elif actual_std > 0.8:
        print(f"  📌 很可能是未缩放的原始latent")
        print(f"     理由: std={actual_std:.3f}，接近原始VAE输出")
        needs_descaling = False
    else:
        print(f"  📌 缩放状态不确定，需要解码测试")
        needs_descaling = None
    
    return combined_latents[0:3], needs_descaling  # 返回前3个样本用于解码测试
    
def test_decoding(sample_latents, needs_descaling_hint):
    """测试不同的解码方式以确定正确的缩放"""
    
    print(f"\n\n🖼️ 解码测试（使用3个样本）")
    print("=" * 70)
    
    # 加载VA-VAE
    from microdoppler_finetune.step6_encode_dataset import load_vavae_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    
    print("\n正在加载VA-VAE模型...")
    vae = load_vavae_model(vae_checkpoint, device)
    vae.model.eval()
    
    # 准备输出目录
    output_dir = Path('/kaggle/working/diagnose_output')
    output_dir.mkdir(exist_ok=True)
    
    # 测试每个样本的不同解码方式
    decode_results = []
    
    for idx in range(min(3, len(sample_latents))):
        sample_latent = sample_latents[idx:idx+1].to(device)
        print(f"\n📍 测试样本 {idx+1}:")
        print(f"  Latent stats: mean={sample_latent.mean():.4f}, std={sample_latent.std():.4f}")
        
        results = {}
        
        # 方式1: 直接解码（假设latent未缩放）
        with torch.no_grad():
            decoded_direct = vae.model.decode(sample_latent)
            decoded_direct = torch.clamp(decoded_direct, -1, 1)
            decoded_direct = (decoded_direct + 1) / 2
            img_direct = (decoded_direct[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 计算图像质量指标
            img_mean = img_direct.mean()
            img_std = img_direct.std()
            is_valid_direct = 30 < img_mean < 225 and img_std > 10  # 合理的图像应该有适当的对比度
            
            path_direct = output_dir / f'sample{idx+1}_direct.png'
            Image.fromarray(img_direct).save(path_direct)
            
            results['direct'] = {
                'mean': img_mean,
                'std': img_std,
                'valid': is_valid_direct
            }
            
            print(f"  ✓ 直接解码: mean={img_mean:.1f}, std={img_std:.1f}, valid={is_valid_direct}")
        
        # 方式2: 除以0.18215后解码（假设latent已缩放）
        with torch.no_grad():
            decoded_descaled = vae.model.decode(sample_latent / 0.18215)
            decoded_descaled = torch.clamp(decoded_descaled, -1, 1)
            decoded_descaled = (decoded_descaled + 1) / 2
            img_descaled = (decoded_descaled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            img_mean = img_descaled.mean()
            img_std = img_descaled.std()
            is_valid_descaled = 30 < img_mean < 225 and img_std > 10
            
            path_descaled = output_dir / f'sample{idx+1}_descaled.png'
            Image.fromarray(img_descaled).save(path_descaled)
            
            results['descaled'] = {
                'mean': img_mean,
                'std': img_std,
                'valid': is_valid_descaled
            }
            
            print(f"  ✓ ÷0.18215解码: mean={img_mean:.1f}, std={img_std:.1f}, valid={is_valid_descaled}")
        
        # 方式3: 乘以0.18215后解码（测试相反情况）
        with torch.no_grad():
            decoded_scaled = vae.model.decode(sample_latent * 0.18215)
            decoded_scaled = torch.clamp(decoded_scaled, -1, 1)
            decoded_scaled = (decoded_scaled + 1) / 2
            img_scaled = (decoded_scaled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            img_mean = img_scaled.mean()
            img_std = img_scaled.std()
            is_valid_scaled = 30 < img_mean < 225 and img_std > 10
            
            path_scaled = output_dir / f'sample{idx+1}_scaled.png'
            Image.fromarray(img_scaled).save(path_scaled)
            
            results['scaled'] = {
                'mean': img_mean,
                'std': img_std,
                'valid': is_valid_scaled
            }
            
            print(f"  ✓ ×0.18215解码: mean={img_mean:.1f}, std={img_std:.1f}, valid={is_valid_scaled}")
        
        decode_results.append(results)
    
    # 分析结果
    print("\n" + "=" * 70)
    print("📊 解码测试结果分析")
    print("=" * 70)
    
    # 统计哪种方式产生了有效图像
    valid_counts = {'direct': 0, 'descaled': 0, 'scaled': 0}
    for result in decode_results:
        for method, data in result.items():
            if data['valid']:
                valid_counts[method] += 1
    
    print(f"\n有效图像计数（共{len(decode_results)}个样本）:")
    for method, count in valid_counts.items():
        print(f"  {method:10s}: {count}/{len(decode_results)} 有效")
    
    # 给出最终判断
    print("\n" + "=" * 70)
    print("🎯 最终诊断结果")
    print("=" * 70)
    
    if valid_counts['descaled'] > valid_counts['direct'] and valid_counts['descaled'] > valid_counts['scaled']:
        print("\n✅ 结论: Latents已被0.18215缩放")
        print("   - 解码时需要除以0.18215")
        print("   - 训练配置: latent_multiplier=1.0（不再缩放）")
        print("   - 生成代码: samples / 0.18215 后再解码")
        final_verdict = "scaled"
    elif valid_counts['direct'] > valid_counts['descaled'] and valid_counts['direct'] > valid_counts['scaled']:
        print("\n✅ 结论: Latents未缩放（原始值）")
        print("   - 解码时直接使用")
        print("   - 训练配置: latent_multiplier=0.18215（需要缩放）")
        print("   - 生成代码: 直接解码samples")
        final_verdict = "unscaled"
    elif valid_counts['scaled'] > valid_counts['direct'] and valid_counts['scaled'] > valid_counts['descaled']:
        print("\n⚠️ 结论: Latents可能过度缩放")
        print("   - 需要乘以0.18215才能正确解码")
        print("   - 这种情况比较异常，建议重新编码数据集")
        final_verdict = "over_scaled"
    else:
        print("\n❓ 结论: 无法确定缩放状态")
        print("   - 请检查生成的图像并手动判断")
        print(f"   - 图像保存在: {output_dir}")
        final_verdict = "unknown"
    
    print(f"\n📁 所有测试图像已保存到: {output_dir}")
    print("   请查看图像质量来验证诊断结果")
    
    return final_verdict

def main():
    """主诊断流程"""
    # 步骤1: 统计分析
    result = diagnose_latent_statistics()
    
    if result is None:
        print("\n❌ 无法完成诊断（未找到数据）")
        return
    
    sample_latents, needs_descaling_hint = result
    
    # 步骤2: 解码测试
    final_verdict = test_decoding(sample_latents, needs_descaling_hint)
    
    # 步骤3: 输出配置建议
    print("\n" + "=" * 70)
    print("📝 配置文件建议")
    print("=" * 70)
    
    if final_verdict == "scaled":
        print("""
config_dit_base.yaml:
  latent_norm: false
  latent_multiplier: 1.0  # latents已缩放，不需要再缩放

step8_train_dit_from_scratch.py生成代码:
  samples_for_decode = samples / 0.18215  # 还原到VAE尺度
  images = vae.decode_to_images(samples_for_decode)
        """)
    elif final_verdict == "unscaled":
        print("""
config_dit_base.yaml:
  latent_norm: false
  latent_multiplier: 0.18215  # 需要缩放到标准空间

step8_train_dit_from_scratch.py生成代码:
  images = vae.decode_to_images(samples)  # 直接解码
        """)
    else:
        print("\n⚠️ 需要手动检查生成的图像来确定正确配置")
    
    print("\n" + "=" * 70)
    print("✅ 诊断完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
