"""
检查实际latent数据的统计特征，确定正确的缩放配置
"""
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

def analyze_raw_latents():
    """分析原始保存的latent数据"""
    
    print("=" * 70)
    print("🔍 分析实际保存的Latent数据")
    print("=" * 70)
    
    # 检查latent目录
    latent_dir = Path('G:/VA-VAE/latent_dataset/train')
    if not latent_dir.exists():
        print(f"❌ 目录不存在: {latent_dir}")
        return
    
    latent_files = list(latent_dir.glob('*.safetensors'))
    if not latent_files:
        print(f"❌ 未找到latent文件")
        return
    
    print(f"📊 找到 {len(latent_files)} 个latent文件")
    
    # 分析前几个文件
    all_stats = []
    for i, file_path in enumerate(latent_files[:5]):
        data = load_file(str(file_path))
        latents = data['latents']
        
        stats = {
            'file': file_path.name,
            'shape': latents.shape,
            'mean': latents.mean().item(),
            'std': latents.std().item(),
            'min': latents.min().item(),
            'max': latents.max().item()
        }
        all_stats.append(stats)
        
        print(f"\n📁 文件 {i+1}: {file_path.name}")
        print(f"   形状: {stats['shape']}")
        print(f"   统计: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"   范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # 综合分析
    overall_std = np.mean([s['std'] for s in all_stats])
    overall_mean = np.mean([s['mean'] for s in all_stats])
    
    print(f"\n📈 综合统计:")
    print(f"   平均std: {overall_std:.6f}")
    print(f"   平均mean: {overall_mean:.6f}")
    
    # 判断缩放状态
    print(f"\n💡 缩放分析:")
    if abs(overall_std - 0.18215) < 0.05:
        print(f"   ✅ std≈0.18215，很可能已应用SD缩放因子")
        print(f"   推荐配置: latent_multiplier=0.18215")
        return "scaled"
    elif overall_std > 0.8:
        print(f"   ✅ std>{0.8}，很可能是原始未缩放latent")
        print(f"   推荐配置: latent_multiplier=1.0")
        return "unscaled"
    else:
        print(f"   ⚠️ std={overall_std:.4f}，缩放状态不明确")
        print(f"   需要进一步测试确定")
        return "uncertain"

def check_step6_encoding():
    """检查step6编码时是否应用了缩放"""
    
    print(f"\n🔍 检查step6编码过程...")
    
    try:
        from step6_encode_dataset import load_vavae_model
        print("   ✅ 可以加载step6模块")
        
        # 检查编码时是否有缩放逻辑
        import inspect
        source = inspect.getsource(load_vavae_model)
        if "0.18215" in source:
            print("   ⚠️ step6编码时可能应用了0.18215缩放")
        else:
            print("   ✅ step6编码时没有明显的缩放操作")
            
    except Exception as e:
        print(f"   ❌ 无法检查step6: {e}")

if __name__ == "__main__":
    scaling_status = analyze_raw_latents()
    check_step6_encoding()
    
    print(f"\n" + "=" * 70)
    print("📋 结论和建议:")
    print("=" * 70)
    
    if scaling_status == "scaled":
        print("✅ latent数据已缩放，使用:")
        print("   latent_multiplier: 0.18215")
        print("   latent_norm: true")
    elif scaling_status == "unscaled":
        print("✅ latent数据未缩放，使用:")
        print("   latent_multiplier: 1.0") 
        print("   latent_norm: true")
    else:
        print("⚠️ 需要进一步测试确定配置")
