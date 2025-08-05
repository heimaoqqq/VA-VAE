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
    """微多普勒数据集加载器 - 支持嵌套用户目录结构"""

    def __init__(self, data_dir, transform=None, max_images_per_user=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_images_per_user = max_images_per_user

        # 支持的图像格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        self.user_labels = []  # 存储用户ID信息

        # 收集所有用户目录下的图像文件
        self._collect_images()

        print(f"📁 找到 {len(self.image_files)} 张图像，来自 {len(set(self.user_labels))} 个用户")

    def _collect_images(self):
        """收集所有用户目录下的图像"""
        user_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
        user_dirs.sort()  # 按用户ID排序

        print(f"🔍 发现 {len(user_dirs)} 个用户目录")

        for user_dir in user_dirs:
            user_id = user_dir.name  # ID_1, ID_2, etc.
            user_images = []

            # 收集该用户的所有图像
            for ext in self.image_extensions:
                user_images.extend(list(user_dir.glob(f'*{ext}')))
                user_images.extend(list(user_dir.glob(f'*{ext.upper()}')))

            # 限制每个用户的图像数量（如果指定）
            if self.max_images_per_user and len(user_images) > self.max_images_per_user:
                user_images = user_images[:self.max_images_per_user]

            # 添加到总列表
            self.image_files.extend(user_images)
            self.user_labels.extend([user_id] * len(user_images))

            print(f"   {user_id}: {len(user_images)} 张图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        user_id = self.user_labels[idx]

        try:
            # 加载图像
            image = Image.open(image_path)

            # 确保是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 应用变换
            if self.transform:
                image = self.transform(image)

            return image, str(image_path), user_id

        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            # 返回一个黑色图像作为fallback
            black_image = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, str(image_path), user_id

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
            # 首先更新配置文件中的模型路径
            config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"

            # 读取配置
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # 更新模型路径
            config['ckpt_path'] = self.vae_model_path

            # 保存临时配置
            temp_config_path = "temp_vavae_config.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)

            # 使用正确的初始化方式
            vae = VA_VAE(config=temp_config_path)

            print(f"✅ VA-VAE模型加载成功")
            print(f"📊 模型参数量: {sum(p.numel() for p in vae.model.parameters()) / 1e6:.1f}M")
            return vae

        except Exception as e:
            print(f"❌ VA-VAE模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_single_image(self, image_tensor, image_path, user_id=None):
        """测试单张图像的重建"""
        with torch.no_grad():
            # 编码到潜在空间
            latent = self.vae.encode_images(image_tensor.unsqueeze(0))

            # 从潜在空间解码
            reconstructed = self.vae.model.decode(latent)

            # 计算重建误差
            mse_loss = F.mse_loss(image_tensor.to(self.device), reconstructed.squeeze(0)).item()

            return {
                'original': image_tensor,
                'reconstructed': reconstructed.squeeze(0).cpu(),
                'latent': latent.cpu(),
                'mse_loss': mse_loss,
                'image_path': image_path,
                'user_id': user_id
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
        dataset = MicroDopplerDataset(data_dir, transform=self.transform, max_images_per_user=max_images//31 if max_images else None)

        if len(dataset) == 0:
            print("❌ 未找到图像文件")
            return None

        # 限制测试图像数量
        if len(dataset) > max_images:
            indices = np.random.choice(len(dataset), max_images, replace=False)
            dataset.image_files = [dataset.image_files[i] for i in indices]
            dataset.user_labels = [dataset.user_labels[i] for i in indices]

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_results = []
        total_mse = 0
        processed_count = 0
        user_stats = {}  # 统计每个用户的结果

        print(f"🔍 开始处理 {len(dataset)} 张图像...")

        for batch_idx, (images, image_paths, user_ids) in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}/{len(dataloader)}")

            batch_results = []

            for i in range(images.size(0)):
                result = self.test_single_image(images[i], image_paths[i], user_ids[i])
                batch_results.append(result)
                total_mse += result['mse_loss']
                processed_count += 1

                # 统计用户结果
                user_id = user_ids[i]
                if user_id not in user_stats:
                    user_stats[user_id] = []
                user_stats[user_id].append(result['mse_loss'])

            # 保存批次对比图
            self.save_batch_comparison(batch_results, output_path / f"batch_{batch_idx + 1:03d}.png")
            all_results.extend(batch_results)
        
        # 计算总体统计
        avg_mse = total_mse / processed_count
        mse_values = [r['mse_loss'] for r in all_results]

        print(f"\n📊 重建测试结果:")
        print(f"   处理图像数量: {processed_count}")
        print(f"   用户数量: {len(user_stats)}")
        print(f"   平均MSE: {avg_mse:.6f}")
        print(f"   MSE标准差: {np.std(mse_values):.6f}")
        print(f"   MSE范围: {np.min(mse_values):.6f} - {np.max(mse_values):.6f}")

        # 打印每个用户的统计
        print(f"\n👥 各用户重建质量:")
        for user_id in sorted(user_stats.keys()):
            user_mse = user_stats[user_id]
            print(f"   {user_id}: {len(user_mse)}张图像, 平均MSE={np.mean(user_mse):.6f}")

        # 保存统计结果
        self.save_statistics(all_results, user_stats, output_path / "reconstruction_stats.txt")

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
            user_id = result.get('user_id', 'Unknown')
            axes[0, i].set_title(f'{user_id}\nOriginal')
            axes[0, i].axis('off')

            # 重建图
            reconstructed = self.denormalize(result['reconstructed'])
            reconstructed = torch.clamp(reconstructed, 0, 1)
            axes[1, i].imshow(reconstructed.permute(1, 2, 0))
            axes[1, i].set_title(f'Reconstructed\nMSE: {result["mse_loss"]:.4f}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_statistics(self, results, user_stats, save_path):
        """保存统计结果到文件"""
        mse_values = [r['mse_loss'] for r in results]

        with open(save_path, 'w') as f:
            f.write("VA-VAE重建测试统计结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试图像数量: {len(results)}\n")
            f.write(f"用户数量: {len(user_stats)}\n")
            f.write(f"平均MSE: {np.mean(mse_values):.6f}\n")
            f.write(f"MSE标准差: {np.std(mse_values):.6f}\n")
            f.write(f"MSE最小值: {np.min(mse_values):.6f}\n")
            f.write(f"MSE最大值: {np.max(mse_values):.6f}\n")
            f.write(f"MSE中位数: {np.median(mse_values):.6f}\n")

            f.write("\n各用户统计:\n")
            f.write("-" * 30 + "\n")
            for user_id in sorted(user_stats.keys()):
                user_mse = user_stats[user_id]
                f.write(f"{user_id}: {len(user_mse)}张图像, 平均MSE={np.mean(user_mse):.6f}, 标准差={np.std(user_mse):.6f}\n")

            f.write("\n详细结果:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(results):
                filename = Path(result['image_path']).name
                user_id = result.get('user_id', 'Unknown')
                f.write(f"{i+1:3d}. {user_id}/{filename}: MSE={result['mse_loss']:.6f}\n")

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
