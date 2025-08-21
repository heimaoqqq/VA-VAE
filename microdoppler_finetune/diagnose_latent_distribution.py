"""
诊断VA-VAE latent分布，找到最优训练配置
"""
import torch
import numpy as np
from safetensors import safe_open
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT'))
from tokenizer.vavae import VA_VAE
import matplotlib.pyplot as plt

def analyze_latent_distribution(latent_path):
    """分析保存的latent分布"""
    print("=" * 60)
    print("📊 分析Latent分布")
    print("=" * 60)
    
    # 加载latent数据
    latent_files = [f for f in os.listdir(latent_path) if f.endswith('.safetensors')]
    all_latents = []
    
    for file in latent_files[:1]:  # 只加载第一个文件测试
        file_path = os.path.join(latent_path, file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            latents = f.get_tensor("latents")
            all_latents.append(latents)
            print(f"加载 {file}: shape={latents.shape}")
    
    latents = torch.cat(all_latents, dim=0) if all_latents else None
    
    if latents is None:
        print("❌ 未找到latent文件")
        return None
        
    # 计算统计
    mean = latents.mean().item()
    std = latents.std().item()
    channel_mean = latents.mean(dim=[0, 2, 3])  # [C]
    channel_std = latents.std(dim=[0, 2, 3])    # [C]
    
    print(f"\n📈 原始Latent统计:")
    print(f"   整体: mean={mean:.4f}, std={std:.4f}")
    print(f"   通道均值范围: [{channel_mean.min():.4f}, {channel_mean.max():.4f}]")
    print(f"   通道标准差范围: [{channel_std.min():.4f}, {channel_std.max():.4f}]")
    
    # 测试不同的缩放因子
    print(f"\n🔧 测试不同缩放因子:")
    scaling_factors = [1.0, 0.18215, 0.5, 2.0, 5.0]
    
    for factor in scaling_factors:
        scaled = latents * factor
        print(f"   factor={factor:6.4f}: mean={scaled.mean():.4f}, std={scaled.std():.4f}")
    
    # 测试归一化效果
    print(f"\n🔄 归一化测试:")
    # Channel-wise normalization (官方方式)
    norm_mean = latents.mean(dim=[0, 2, 3], keepdim=True)
    norm_std = latents.std(dim=[0, 2, 3], keepdim=True)
    normalized = (latents - norm_mean) / (norm_std + 1e-8)
    print(f"   归一化后: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # 不同multiplier测试
    for mult in [0.18215, 1.0]:
        scaled_norm = normalized * mult
        print(f"   归一化+mult={mult}: mean={scaled_norm.mean():.4f}, std={scaled_norm.std():.4f}")
    
    return latents, mean, std

def test_vae_reconstruction(vae_path, latents):
    """测试VAE重建质量"""
    print("\n" + "=" * 60)
    print("🔬 测试VAE重建")
    print("=" * 60)
    
    # 加载VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VA_VAE(vae_path)
    
    # 测试不同的输入分布
    test_configs = [
        ("原始", latents[:1]),
        ("缩放0.18215", latents[:1] * 0.18215),
        ("归一化", (latents[:1] - latents.mean()) / latents.std()),
    ]
    
    for name, test_latent in test_configs:
        test_latent = test_latent.to(device)
        with torch.no_grad():
            # 解码
            decoded = vae.decode_to_images(test_latent)
            # 检查输出范围
            print(f"\n{name}:")
            print(f"   输入分布: mean={test_latent.mean():.4f}, std={test_latent.std():.4f}")
            print(f"   输出范围: [{decoded.min():.0f}, {decoded.max():.0f}]")
            
            # 再编码检查一致性
            # 需要将decoded转为tensor并处理
            if isinstance(decoded, np.ndarray):
                # decoded是[B,H,W,C]的numpy array
                img_tensor = torch.from_numpy(decoded).float()
                img_tensor = img_tensor.permute(0, 3, 1, 2)  # [B,C,H,W]
                img_tensor = img_tensor.to(device)
                # 归一化到[-1,1]
                img_tensor = img_tensor / 127.5 - 1.0
                
                # 重新编码
                re_encoded = vae.encode_images(img_tensor)
                
                # 比较
                mse = ((test_latent - re_encoded) ** 2).mean().item()
                print(f"   重建MSE: {mse:.6f}")

def main():
    """主诊断流程"""
    # 路径配置
    train_latent_path = "/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256"
    vae_config_path = "../LightningDiT/vavae/config.yaml"
    
    # Windows本地测试路径
    if not os.path.exists(train_latent_path):
        train_latent_path = "g:/VA-VAE/latents_official/vavae_config_for_dit/microdoppler_train_256"
        vae_config_path = "g:/VA-VAE/LightningDiT/vavae/config.yaml"
    
    print("🚀 VA-VAE Latent分布诊断工具")
    print(f"   数据路径: {train_latent_path}")
    
    # 分析latent分布
    latents, mean, std = analyze_latent_distribution(train_latent_path)
    
    if latents is not None and os.path.exists(vae_config_path):
        # 测试VAE重建
        test_vae_reconstruction(vae_config_path, latents)
    
    # 推荐配置
    print("\n" + "=" * 60)
    print("💡 推荐配置")
    print("=" * 60)
    
    if mean is not None and std is not None:
        # 基于分析结果推荐
        if abs(mean) < 0.1 and abs(std - 1.0) < 0.2:
            print("✅ Latent已接近标准分布N(0,1)")
            print("   推荐: latent_norm=false, latent_multiplier=1.0")
        elif abs(std - 0.18215) < 0.05:
            print("✅ Latent似乎已被0.18215缩放")
            print("   推荐: latent_norm=false, latent_multiplier=1.0")
        else:
            print("⚠️ Latent分布非标准")
            print(f"   当前: mean={mean:.4f}, std={std:.4f}")
            print("   选项1: latent_norm=true, latent_multiplier=1.0 (归一化到N(0,1))")
            print("   选项2: latent_norm=false, latent_multiplier=0.18215 (缩放到SD-VAE空间)")
            print("   选项3: latent_norm=false, latent_multiplier=1.0 (保持原始)")
            
            # 根据std大小判断
            if std > 1.0:
                print(f"\n   📌 建议: 由于std={std:.2f}>1.0，推荐选项1或2来降低方差")
            else:
                print(f"\n   📌 建议: 由于std={std:.2f}接近1.0，推荐选项3保持原始")

if __name__ == "__main__":
    main()
