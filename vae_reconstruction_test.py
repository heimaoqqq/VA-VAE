#!/usr/bin/env python3
"""
阶段1: VA-VAE重建效果测试
测试预训练VA-VAE在微多普勒数据上的直接使用效果
专注于重建质量评估，不涉及用户标签
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

class MicroDopplerDataset(Dataset):
    """微多普勒数据集加载器"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 支持的图像格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        
        # 收集所有图像文件
        for ext in self.image_extensions:
            self.image_files.extend(list(self.data_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        print(f"📁 找到 {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 确保是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, str(image_path)
            
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            # 返回一个黑色图像作为fallback
            black_image = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, str(image_path)

class VAEReconstructionTester:
    """VA-VAE重建测试器"""
    
    def __init__(self, vae_model_path, device='cuda'):
        self.device = device
        self.vae_model_path = vae_model_path
        
        # 加载VA-VAE模型
        print("🔧 加载预训练VA-VAE模型...")
        self.vae = self.load_vae_model()
        
        # 图像预处理（与ImageNet训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        # 反归一化用于显示
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # [0, 1]
        ])
    
    def load_vae_model(self):
        """加载预训练的VA-VAE模型"""
        try:
            vae = VA_VAE(
                model_name='vavae_f16d32',
                ckpt_path=self.vae_model_path
            ).to(self.device)
            vae.eval()
            
            print(f"✅ VA-VAE模型加载成功")
            print(f"📊 模型参数量: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")
            return vae
            
        except Exception as e:
            print(f"❌ VA-VAE模型加载失败: {e}")
            return None
    
    def test_single_image(self, image_tensor, image_path):
        """测试单张图像的重建"""
        with torch.no_grad():
            # 编码到潜在空间
            latent = self.vae.encode(image_tensor.unsqueeze(0).to(self.device))
            
            # 从潜在空间解码
            reconstructed = self.vae.decode(latent)
            
            # 计算重建误差
            mse_loss = F.mse_loss(image_tensor.to(self.device), reconstructed.squeeze(0)).item()
            
            return {
                'original': image_tensor,
                'reconstructed': reconstructed.squeeze(0).cpu(),
                'latent': latent.cpu(),
                'mse_loss': mse_loss,
                'image_path': image_path
            }
    
    def test_batch_reconstruction(self, data_dir, output_dir, batch_size=8, max_images=50):
        """批量测试重建效果"""
        print(f"🚀 开始批量重建测试")
        print(f"📁 数据目录: {data_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建数据集和数据加载器
        dataset = MicroDopplerDataset(data_dir, transform=self.transform)
        
        if len(dataset) == 0:
            print("❌ 未找到图像文件")
            return None
        
        # 限制测试图像数量
        if len(dataset) > max_images:
            indices = np.random.choice(len(dataset), max_images, replace=False)
            dataset.image_files = [dataset.image_files[i] for i in indices]
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_results = []
        total_mse = 0
        processed_count = 0
        
        print(f"🔍 开始处理 {len(dataset)} 张图像...")
        
        for batch_idx, (images, image_paths) in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}/{len(dataloader)}")
            
            batch_results = []
            
            for i in range(images.size(0)):
                result = self.test_single_image(images[i], image_paths[i])
                batch_results.append(result)
                total_mse += result['mse_loss']
                processed_count += 1
            
            # 保存批次对比图
            self.save_batch_comparison(batch_results, output_path / f"batch_{batch_idx + 1:03d}.png")
            all_results.extend(batch_results)
        
        # 计算总体统计
        avg_mse = total_mse / processed_count
        mse_values = [r['mse_loss'] for r in all_results]
        
        print(f"\n📊 重建测试结果:")
        print(f"   处理图像数量: {processed_count}")
        print(f"   平均MSE: {avg_mse:.6f}")
        print(f"   MSE标准差: {np.std(mse_values):.6f}")
        print(f"   MSE范围: {np.min(mse_values):.6f} - {np.max(mse_values):.6f}")
        
        # 保存统计结果
        self.save_statistics(all_results, output_path / "reconstruction_stats.txt")
        
        return all_results

    def save_batch_comparison(self, batch_results, save_path):
        """保存批次对比图"""
        batch_size = len(batch_results)
        fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))

        if batch_size == 1:
            axes = axes.reshape(2, 1)

        for i, result in enumerate(batch_results):
            # 原图
            original = self.denormalize(result['original'])
            original = torch.clamp(original, 0, 1)
            axes[0, i].imshow(original.permute(1, 2, 0))
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # 重建图
            reconstructed = self.denormalize(result['reconstructed'])
            reconstructed = torch.clamp(reconstructed, 0, 1)
            axes[1, i].imshow(reconstructed.permute(1, 2, 0))
            axes[1, i].set_title(f'Recon {i+1}\nMSE: {result["mse_loss"]:.4f}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_statistics(self, results, save_path):
        """保存统计结果到文件"""
        mse_values = [r['mse_loss'] for r in results]

        with open(save_path, 'w') as f:
            f.write("VA-VAE重建测试统计结果\n")
            f.write("=" * 40 + "\n")
            f.write(f"测试图像数量: {len(results)}\n")
            f.write(f"平均MSE: {np.mean(mse_values):.6f}\n")
            f.write(f"MSE标准差: {np.std(mse_values):.6f}\n")
            f.write(f"MSE最小值: {np.min(mse_values):.6f}\n")
            f.write(f"MSE最大值: {np.max(mse_values):.6f}\n")
            f.write(f"MSE中位数: {np.median(mse_values):.6f}\n")
            f.write("\n详细结果:\n")

            for i, result in enumerate(results):
                filename = Path(result['image_path']).name
                f.write(f"{i+1:3d}. {filename}: MSE={result['mse_loss']:.6f}\n")

        print(f"📊 统计结果已保存: {save_path}")

def main():
    """主函数"""
    print("🚀 阶段1: VA-VAE重建效果测试")
    print("="*60)

    # 检查模型文件
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    if not Path(vae_model_path).exists():
        print(f"❌ VA-VAE模型文件不存在: {vae_model_path}")
        print("💡 请先运行 step2_download_models.py 下载模型")
        return False

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 使用设备: {device}")

    # 创建测试器
    tester = VAEReconstructionTester(vae_model_path, device)
    if tester.vae is None:
        return False

    print("\n📋 使用说明:")
    print("1. 将您的微多普勒图像放在一个目录中（如 'micro_doppler_data/'）")
    print("2. 调用测试函数:")
    print("   results = tester.test_batch_reconstruction('micro_doppler_data/', 'vae_test_output/')")
    print("\n💡 示例用法:")
    print("   # 在Python中运行:")
    print("   from vae_reconstruction_test import VAEReconstructionTester")
    print("   tester = VAEReconstructionTester('models/vavae-imagenet256-f16d32-dinov2.pt')")
    print("   results = tester.test_batch_reconstruction('your_data_dir/', 'output_dir/')")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
