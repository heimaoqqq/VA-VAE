"""
调试VA-VAE归一化流程，验证各个阶段的数据范围
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from simplified_vavae import SimplifiedVAVAE

def test_vae_normalization():
    """测试VA-VAE的归一化和反归一化流程"""
    
    print("="*60)
    print("VA-VAE 归一化流程测试")
    print("="*60)
    
    # 1. 初始化VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_path = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    vae = SimplifiedVAVAE(vae_path)
    vae.eval()
    vae.freeze()
    vae = vae.to(device)  # 确保VAE在正确设备上
    
    print(f"\n📌 VAE缩放因子: {vae.scale_factor}")
    
    # 2. 创建测试图像
    print("\n" + "="*40)
    print("测试1: 不同输入范围的编码")
    print("="*40)
    
    # 测试不同范围的输入
    test_ranges = [
        ("[0, 1]", torch.rand(1, 3, 256, 256).to(device)),
        ("[-1, 1]", (torch.rand(1, 3, 256, 256) * 2 - 1).to(device)),
        ("[0, 255]", (torch.rand(1, 3, 256, 256) * 255).to(device))
    ]
    
    for range_name, test_img in test_ranges:
        print(f"\n输入范围 {range_name}:")
        print(f"  Min: {test_img.min():.3f}, Max: {test_img.max():.3f}")
        
        # 编码
        with torch.no_grad():
            latent = vae.encode(test_img)
        
        print(f"  编码后latent:")
        print(f"    形状: {latent.shape}")
        print(f"    Min: {latent.min():.3f}, Max: {latent.max():.3f}")
        print(f"    Mean: {latent.mean():.3f}, Std: {latent.std():.3f}")
    
    # 3. 测试解码流程
    print("\n" + "="*40)
    print("测试2: 解码流程")
    print("="*40)
    
    # 创建测试图像
    test_image = torch.rand(1, 3, 256, 256).to(device)  # [0,1]范围
    
    print(f"原始图像 [0,1]:")
    print(f"  Min: {test_image.min():.3f}, Max: {test_image.max():.3f}")
    
    # 编码
    with torch.no_grad():
        latent = vae.encode(test_image)
    
    print(f"\n编码后latent:")
    print(f"  形状: {latent.shape}")
    print(f"  Min: {latent.min():.3f}, Max: {latent.max():.3f}")
    print(f"  Mean: {latent.mean():.3f}, Std: {latent.std():.3f}")
    
    # 解码
    with torch.no_grad():
        reconstructed = vae.decode(latent)
    
    print(f"\n解码后图像:")
    print(f"  形状: {reconstructed.shape}")
    print(f"  Min: {reconstructed.min():.3f}, Max: {reconstructed.max():.3f}")
    print(f"  Mean: {reconstructed.mean():.3f}, Std: {reconstructed.std():.3f}")
    
    # 4. 测试扩散模型生成的latent解码
    print("\n" + "="*40)
    print("测试3: 模拟扩散模型输出")
    print("="*40)
    
    # 模拟扩散模型的输出（不同尺度的噪声）
    noise_scales = [0.1, 0.5, 1.0, 2.0]
    
    for scale in noise_scales:
        noise_latent = torch.randn(1, 32, 16, 16).to(device) * scale
        print(f"\n噪声尺度 {scale}:")
        print(f"  Latent Min: {noise_latent.min():.3f}, Max: {noise_latent.max():.3f}")
        print(f"  Latent Std: {noise_latent.std():.3f}")
        
        # 解码
        with torch.no_grad():
            decoded = vae.decode(noise_latent)
        
        print(f"  解码后 Min: {decoded.min():.3f}, Max: {decoded.max():.3f}")
        
        # 检查是否需要额外处理
        if decoded.min() < 0 or decoded.max() > 1:
            print(f"  ⚠️ 解码结果超出[0,1]范围!")
    
    # 5. 测试VAE内部处理
    print("\n" + "="*40)
    print("测试4: VAE内部归一化检查")
    print("="*40)
    
    # 直接调用VAE的encode和decode（不通过wrapper）
    test_img_normalized = test_image * 2 - 1  # 转换到[-1,1]
    
    print(f"输入VAE.encode的图像[-1,1]:")
    print(f"  Min: {test_img_normalized.min():.3f}, Max: {test_img_normalized.max():.3f}")
    
    # 直接编码
    with torch.no_grad():
        posterior = vae.vae.encode(test_img_normalized)
        z = posterior.sample()
    
    print(f"\n原始latent（未缩放）:")
    print(f"  Min: {z.min():.3f}, Max: {z.max():.3f}")
    print(f"  Mean: {z.mean():.3f}, Std: {z.std():.3f}")
    
    # 缩放后
    z_scaled = z * vae.scale_factor
    print(f"\n缩放后latent（×{vae.scale_factor}）:")
    print(f"  Min: {z_scaled.min():.3f}, Max: {z_scaled.max():.3f}")
    print(f"  Mean: {z_scaled.mean():.3f}, Std: {z_scaled.std():.3f}")
    
    # 还原缩放并解码
    z_unscaled = z_scaled / vae.scale_factor
    with torch.no_grad():
        x_decoded = vae.vae.decode(z_unscaled)
    
    print(f"\n原始解码输出:")
    print(f"  Min: {x_decoded.min():.3f}, Max: {x_decoded.max():.3f}")
    
    # 转换到[0,1]
    x_01 = (x_decoded + 1.0) / 2.0
    x_01 = torch.clamp(x_01, 0, 1)
    
    print(f"\n转换到[0,1]后:")
    print(f"  Min: {x_01.min():.3f}, Max: {x_01.max():.3f}")
    
    print("\n" + "="*60)
    print("总结:")
    print("="*60)
    print("1. VA-VAE期望输入: [-1,1]范围的图像")
    print("2. 编码输出: latent * scale_factor")
    print("3. 解码输入: latent / scale_factor")
    print("4. 解码输出: [-1,1]范围，需要转换到[0,1]")
    print(f"5. 当前scale_factor: {vae.scale_factor}")
    
    return vae

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行测试
    vae = test_vae_normalization()
    
    # 6. 实际图像测试
    print("\n" + "="*60)
    print("测试5: 真实微多普勒图像")
    print("="*60)
    
    # 尝试加载一张真实图像
    # 使用正确的数据集路径
    possible_paths = [
        "/kaggle/input/dataset/ID_1/ID1_case1_1_Doppler1.jpg",
        "/kaggle/input/dataset/ID_1/ID1_case1_1_Doppler10.jpg",
        "/kaggle/input/dataset/ID_1/ID1_case1_1_Doppler100.jpg"
    ]
    
    image_path = None
    for path in possible_paths:
        if Path(path).exists():
            image_path = path
            break
    
    if image_path:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0  # [0,1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)  # 直接移动到GPU
        
        print(f"真实图像:")
        print(f"  原始范围: [0, 255]")
        print(f"  归一化后: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # 编码-解码
        with torch.no_grad():
            latent = vae.encode(img_tensor)
            reconstructed = vae.decode(latent)
        
        print(f"\n编码latent:")
        print(f"  Min: {latent.min():.3f}, Max: {latent.max():.3f}")
        print(f"  Std: {latent.std():.3f}")
        
        print(f"\n重建图像:")
        print(f"  Min: {reconstructed.min():.3f}, Max: {reconstructed.max():.3f}")
        
        # 保存对比图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img_tensor[0].permute(1, 2, 0).cpu())
        axes[0].set_title(f"原始 [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed[0].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1].set_title(f"重建 [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/vae_normalization_test.png', dpi=100, bbox_inches='tight')
        print("\n✅ 对比图已保存到 /kaggle/working/vae_normalization_test.png")
    else:
        print("⚠️ 未找到真实图像文件，跳过此测试")
        print("   尝试过的路径:")
        for path in possible_paths:
            print(f"   - {path}")
    
    print("\n" + "="*60)
    print("测试完成！")
