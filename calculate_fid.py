#!/usr/bin/env python3
"""
FID计算脚本
计算原始图像和重建图像之间的FID分数
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_fid import fid_score
import tempfile
import shutil

class ImageDataset(Dataset):
    """简单的图像数据集"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # 支持的图像格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        
        # 收集所有图像文件
        for ext in self.image_extensions:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        print(f"📁 找到 {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, str(image_path)
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            return None, str(image_path)

def save_reconstructed_images(vae_tester, data_dir, output_dir):
    """使用VA-VAE重建图像并保存"""
    print("🔧 生成重建图像...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 反归一化
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    ])
    
    saved_count = 0
    
    for idx, (image, image_path) in enumerate(dataloader):
        if image is None:
            continue
            
        with torch.no_grad():
            # 编码和解码
            latent = vae_tester.vae.encode(image.to(vae_tester.device))
            reconstructed = vae_tester.vae.decode(latent)
            
            # 反归一化并保存
            reconstructed = denormalize(reconstructed.squeeze(0).cpu())
            reconstructed = torch.clamp(reconstructed, 0, 1)
            
            # 转换为PIL图像
            reconstructed_pil = transforms.ToPILImage()(reconstructed)
            
            # 保存
            filename = Path(image_path[0]).stem + '_reconstructed.png'
            save_path = output_path / filename
            reconstructed_pil.save(save_path)
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"已保存 {saved_count} 张重建图像")
    
    print(f"✅ 总共保存了 {saved_count} 张重建图像到 {output_dir}")
    return saved_count

def calculate_fid_score(original_dir, reconstructed_dir, device='cuda'):
    """计算FID分数"""
    print("📊 计算FID分数...")
    
    try:
        # 使用pytorch_fid计算FID
        fid_value = fid_score.calculate_fid_given_paths(
            [str(original_dir), str(reconstructed_dir)],
            batch_size=50,
            device=device,
            dims=2048
        )
        
        print(f"✅ FID分数: {fid_value:.4f}")
        return fid_value
        
    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        return None

def prepare_original_images(data_dir, output_dir):
    """准备原始图像（统一格式和尺寸）"""
    print("🔧 准备原始图像...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(data_dir)
    saved_count = 0
    
    for idx, (_, image_path) in enumerate(dataset):
        try:
            # 加载和处理图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整尺寸
            image = image.resize((256, 256), Image.LANCZOS)
            
            # 保存
            filename = Path(image_path).stem + '_original.png'
            save_path = output_path / filename
            image.save(save_path)
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"已处理 {saved_count} 张原始图像")
                
        except Exception as e:
            print(f"❌ 处理图像失败 {image_path}: {e}")
    
    print(f"✅ 总共处理了 {saved_count} 张原始图像到 {output_dir}")
    return saved_count

def main():
    """主函数"""
    print("📊 FID分数计算工具")
    print("="*50)
    
    print("使用说明:")
    print("1. 准备您的微多普勒图像目录")
    print("2. 运行以下代码:")
    print()
    print("# 示例用法:")
    print("from vae_reconstruction_test import VAEReconstructionTester")
    print("from calculate_fid import *")
    print()
    print("# 创建VA-VAE测试器")
    print("tester = VAEReconstructionTester('models/vavae-imagenet256-f16d32-dinov2.pt')")
    print()
    print("# 准备原始图像")
    print("prepare_original_images('your_data_dir/', 'temp_original/')")
    print()
    print("# 生成重建图像")
    print("save_reconstructed_images(tester, 'your_data_dir/', 'temp_reconstructed/')")
    print()
    print("# 计算FID")
    print("fid = calculate_fid_score('temp_original/', 'temp_reconstructed/')")
    print("print(f'FID分数: {fid:.4f}')")

if __name__ == "__main__":
    main()
