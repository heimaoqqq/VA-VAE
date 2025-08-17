#!/usr/bin/env python3
"""
Step 8: VAE编解码管道测试
验证VA-VAE的编码-解码链路是否正常工作
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 环境设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))

# 内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 设置taming路径
def setup_taming_path():
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),
        Path('/kaggle/working/.taming_path'),
        project_root / 'taming-transformers',
    ]
    
    for location in taming_locations:
        if location.name == '.taming_path' and location.exists():
            with open(location, 'r') as f:
                taming_path = f.read().strip()
            if Path(taming_path).exists():
                sys.path.insert(0, taming_path)
                return True
        elif location.exists():
            sys.path.insert(0, str(location))
            return True
    return False

setup_taming_path()

# 导入VA-VAE
from tokenizer.autoencoder import AutoencoderKL

class VAEPipelineTest:
    """VAE管道测试类"""
    
    def __init__(self, vae_checkpoint, device='cuda'):
        self.device = device
        self.vae_checkpoint = vae_checkpoint
        self.vae = None
        self.results = {}
        
    def load_vae(self):
        """加载VAE模型"""
        print(f"📦 加载VAE模型: {self.vae_checkpoint}")
        
        self.vae = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.vae_checkpoint
        ).to(self.device).eval()
        
        print("✅ VAE加载成功")
        return self.vae
    
    def test_reconstruction(self, image_path):
        """测试单张图像的重建质量"""
        print(f"\n🔬 测试图像重建: {image_path}")
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        # 归一化到[-1, 1]
        img_normalized = img_tensor * 2.0 - 1.0
        img_normalized = img_normalized.to(self.device)
        
        with torch.no_grad():
            # 编码
            print("  📥 编码中...")
            posterior = self.vae.encode(img_normalized)
            if hasattr(posterior, 'sample'):
                latent = posterior.sample()
            else:
                latent = posterior
            
            # 缩放潜在表示
            latent_scaled = latent * 0.13025
            
            # 记录统计信息
            self.results['latent_shape'] = latent.shape
            self.results['latent_mean'] = latent.mean().item()
            self.results['latent_std'] = latent.std().item()
            self.results['latent_min'] = latent.min().item()
            self.results['latent_max'] = latent.max().item()
            
            print(f"  📊 潜在表示统计:")
            print(f"     形状: {latent.shape}")
            print(f"     均值: {latent.mean().item():.4f}")
            print(f"     标准差: {latent.std().item():.4f}")
            print(f"     最小值: {latent.min().item():.4f}")
            print(f"     最大值: {latent.max().item():.4f}")
            
            # 解码
            print("  📤 解码中...")
            latent_unscaled = latent_scaled / 0.13025
            reconstructed = self.vae.decode(latent_unscaled)
            
            # 反归一化到[0, 1]
            reconstructed = (reconstructed + 1) / 2
            reconstructed = reconstructed.clamp(0, 1)
            
            # 计算重建误差
            mse = torch.mean((img_tensor.to(self.device) - reconstructed) ** 2).item()
            self.results['reconstruction_mse'] = mse
            
            print(f"  📏 重建MSE误差: {mse:.6f}")
            
        return img_tensor[0], reconstructed[0], latent

    def test_random_generation(self, num_samples=4):
        """测试从随机噪声生成"""
        print(f"\n🎲 测试随机生成 ({num_samples}个样本)")
        
        generated_samples = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # 生成随机潜在向量
                z = torch.randn(1, 32, 16, 16, device=self.device)
                
                # 记录噪声统计
                print(f"  样本 {i+1}:")
                print(f"    噪声均值: {z.mean().item():.4f}")
                print(f"    噪声标准差: {z.std().item():.4f}")
                
                # 解码
                z_scaled = z * 0.13025
                z_unscaled = z_scaled / 0.13025
                generated = self.vae.decode(z_unscaled)
                
                # 反归一化
                generated = (generated + 1) / 2
                generated = generated.clamp(0, 1)
                
                generated_samples.append(generated[0])
                
                # 检查生成图像统计
                print(f"    生成图像均值: {generated.mean().item():.4f}")
                print(f"    生成图像标准差: {generated.std().item():.4f}")
        
        return generated_samples

    def test_interpolation(self, image1_path, image2_path, num_steps=5):
        """测试潜在空间插值"""
        print(f"\n🔄 测试潜在空间插值")
        
        # 加载两张图像
        images = []
        latents = []
        
        for img_path in [image1_path, image2_path]:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_normalized = (img_tensor * 2.0 - 1.0).to(self.device)
            
            images.append(img_tensor[0])
            
            # 编码
            with torch.no_grad():
                posterior = self.vae.encode(img_normalized)
                if hasattr(posterior, 'sample'):
                    latent = posterior.sample()
                else:
                    latent = posterior
                latents.append(latent)
        
        # 插值
        interpolated_samples = []
        
        with torch.no_grad():
            for alpha in np.linspace(0, 1, num_steps):
                # 潜在空间线性插值
                z_interp = latents[0] * (1 - alpha) + latents[1] * alpha
                
                # 解码
                z_scaled = z_interp * 0.13025
                z_unscaled = z_scaled / 0.13025
                decoded = self.vae.decode(z_unscaled)
                
                # 反归一化
                decoded = (decoded + 1) / 2
                decoded = decoded.clamp(0, 1)
                
                interpolated_samples.append(decoded[0])
                
                print(f"  步骤 {len(interpolated_samples)}/{num_steps} (α={alpha:.2f})")
        
        return images, interpolated_samples

    def visualize_results(self, output_dir):
        """可视化测试结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建综合报告图
        fig = plt.figure(figsize=(20, 12))
        
        # 打印测试报告
        print("\n" + "="*60)
        print("📊 VAE管道测试报告")
        print("="*60)
        
        if 'latent_shape' in self.results:
            print(f"潜在表示维度: {self.results['latent_shape']}")
            print(f"潜在表示统计:")
            print(f"  均值: {self.results['latent_mean']:.4f}")
            print(f"  标准差: {self.results['latent_std']:.4f}")
            print(f"  范围: [{self.results['latent_min']:.4f}, {self.results['latent_max']:.4f}]")
        
        if 'reconstruction_mse' in self.results:
            print(f"重建误差(MSE): {self.results['reconstruction_mse']:.6f}")
        
        # 保存报告
        report_path = output_dir / 'vae_test_report.txt'
        with open(report_path, 'w') as f:
            f.write("VAE管道测试报告\n")
            f.write("="*60 + "\n")
            for key, value in self.results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ 报告已保存: {report_path}")
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vae_test_visualization.png', dpi=150)
        plt.close()

def run_comprehensive_test():
    """运行全面的VAE测试"""
    
    print("="*60)
    print("🚀 VA-VAE管道全面测试")
    print("="*60)
    
    # 配置
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    test_image_dir = '/kaggle/input/dataset/ID_5'  # 选择一个用户的数据
    output_dir = '/kaggle/working/vae_test_results'
    
    # 如果checkpoint不存在，尝试其他路径
    if not Path(vae_checkpoint).exists():
        alt_paths = [
            '/kaggle/working/vavae_checkpoints/best_model.ckpt',
            '/kaggle/working/stage3_outputs/checkpoints/best_model.ckpt'
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                vae_checkpoint = alt_path
                break
    
    # 初始化测试器
    tester = VAEPipelineTest(vae_checkpoint)
    tester.load_vae()
    
    # 获取测试图像
    test_images = list(Path(test_image_dir).glob('*.png'))[:5]
    
    if len(test_images) >= 2:
        # 1. 测试重建
        print("\n" + "="*40)
        print("测试1: 图像重建")
        print("="*40)
        original, reconstructed, latent = tester.test_reconstruction(test_images[0])
        
        # 保存重建对比
        save_dir = Path(output_dir) / 'reconstruction'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原图
        orig_np = original.permute(1, 2, 0).cpu().numpy()
        orig_np = (orig_np * 255).astype(np.uint8)
        Image.fromarray(orig_np).save(save_dir / 'original.png')
        
        # 保存重建图
        recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).astype(np.uint8)
        Image.fromarray(recon_np).save(save_dir / 'reconstructed.png')
        
        # 2. 测试随机生成
        print("\n" + "="*40)
        print("测试2: 随机生成")
        print("="*40)
        random_samples = tester.test_random_generation(num_samples=4)
        
        # 保存随机生成样本
        random_dir = Path(output_dir) / 'random_generation'
        random_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(random_samples):
            sample_np = sample.permute(1, 2, 0).cpu().numpy()
            sample_np = (sample_np * 255).astype(np.uint8)
            Image.fromarray(sample_np).save(random_dir / f'random_{i}.png')
        
        # 3. 测试插值
        if len(test_images) >= 2:
            print("\n" + "="*40)
            print("测试3: 潜在空间插值")
            print("="*40)
            originals, interpolated = tester.test_interpolation(
                test_images[0], test_images[1], num_steps=5
            )
            
            # 保存插值结果
            interp_dir = Path(output_dir) / 'interpolation'
            interp_dir.mkdir(parents=True, exist_ok=True)
            
            for i, sample in enumerate(interpolated):
                sample_np = sample.permute(1, 2, 0).cpu().numpy()
                sample_np = (sample_np * 255).astype(np.uint8)
                Image.fromarray(sample_np).save(interp_dir / f'interp_{i}.png')
    
    # 生成测试报告
    tester.visualize_results(output_dir)
    
    print("\n✨ VAE管道测试完成！")
    print(f"结果保存在: {output_dir}")

def test_normalization_chain():
    """专门测试归一化链路"""
    
    print("\n" + "="*60)
    print("🔍 归一化链路测试")
    print("="*60)
    
    # 创建测试张量
    test_tensor = torch.randn(1, 3, 256, 256)
    
    print("原始张量统计:")
    print(f"  形状: {test_tensor.shape}")
    print(f"  范围: [{test_tensor.min():.4f}, {test_tensor.max():.4f}]")
    print(f"  均值: {test_tensor.mean():.4f}")
    print(f"  标准差: {test_tensor.std():.4f}")
    
    # 步骤1: 归一化到[0,1]
    tensor_01 = (test_tensor - test_tensor.min()) / (test_tensor.max() - test_tensor.min())
    print("\n[0,1]归一化后:")
    print(f"  范围: [{tensor_01.min():.4f}, {tensor_01.max():.4f}]")
    
    # 步骤2: 转换到[-1,1]
    tensor_11 = tensor_01 * 2.0 - 1.0
    print("\n[-1,1]归一化后:")
    print(f"  范围: [{tensor_11.min():.4f}, {tensor_11.max():.4f}]")
    
    # 模拟编码后的潜在表示
    latent = torch.randn(1, 32, 16, 16)
    print("\n模拟潜在表示:")
    print(f"  形状: {latent.shape}")
    print(f"  范围: [{latent.min():.4f}, {latent.max():.4f}]")
    
    # 步骤3: 缩放潜在表示
    latent_scaled = latent * 0.13025
    print("\n缩放后潜在表示(×0.13025):")
    print(f"  范围: [{latent_scaled.min():.4f}, {latent_scaled.max():.4f}]")
    
    # 步骤4: 反缩放
    latent_unscaled = latent_scaled / 0.13025
    print("\n反缩放后潜在表示(÷0.13025):")
    print(f"  范围: [{latent_unscaled.min():.4f}, {latent_unscaled.max():.4f}]")
    
    # 验证是否恢复
    diff = torch.abs(latent - latent_unscaled).max()
    print(f"\n缩放-反缩放误差: {diff:.10f}")
    
    # 步骤5: 模拟解码后
    decoded = torch.randn(1, 3, 256, 256) * 2  # 模拟[-2,2]范围
    print("\n模拟解码输出:")
    print(f"  范围: [{decoded.min():.4f}, {decoded.max():.4f}]")
    
    # 步骤6: 反归一化到[0,1]
    final = (decoded + 1) / 2
    final = final.clamp(0, 1)
    print("\n最终输出[0,1]:")
    print(f"  范围: [{final.min():.4f}, {final.max():.4f}]")
    
    print("\n✅ 归一化链路测试完成")

def main():
    """主函数"""
    
    # 选择测试模式
    test_mode = 'comprehensive'  # 'comprehensive', 'normalization', 或 'quick'
    
    if test_mode == 'comprehensive':
        # 运行全面测试
        run_comprehensive_test()
    elif test_mode == 'normalization':
        # 测试归一化链路
        test_normalization_chain()
    elif test_mode == 'quick':
        # 快速测试
        print("🚀 快速VAE测试")
        tester = VAEPipelineTest(
            '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
        )
        tester.load_vae()
        
        # 测试随机生成
        samples = tester.test_random_generation(num_samples=2)
        print(f"✅ 生成了{len(samples)}个样本")

if __name__ == "__main__":
    main()
