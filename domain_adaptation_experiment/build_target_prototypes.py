#!/usr/bin/env python3
"""
目标域原型构建脚本
用于构建背包步态（目标域）的类原型，支持推理期原型校准（PNC）
适配Kaggle环境
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import json
import argparse

# 添加父目录到路径以导入分类器
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train_calibrated_classifier import DomainAdaptiveClassifier


class TargetDomainDataset(Dataset):
    """目标域（背包步态）数据集"""
    
    def __init__(self, data_dir, transform=None, support_size=15, seed=42):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据变换
            support_size: 每个用户的支持集大小
            seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.support_size = support_size
        self.samples = []
        
        # 设置随机种子确保可重复
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载支持集
        self._load_support_set()
        
    def _load_support_set(self):
        """加载每个用户的支持集样本"""
        # 扫描所有用户目录（ID_1 到 ID_31）
        user_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for user_dir in user_dirs:
            # 提取用户ID（ID_x -> x-1）
            user_name = user_dir.name
            if user_name.startswith('ID_'):
                user_id = int(user_name.split('_')[1]) - 1  # ID_1 -> 0
            else:
                continue
            
            # 获取该用户的所有图像
            image_files = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
            
            if len(image_files) == 0:
                print(f"⚠️ No images found for {user_name}")
                continue
            
            # 随机采样support_size个样本
            n_samples = min(self.support_size, len(image_files))
            selected_files = random.sample(image_files, n_samples)
            
            # 添加到数据集
            for img_path in selected_files:
                self.samples.append({
                    'path': img_path,
                    'label': user_id,
                    'user_name': user_name
                })
            
            print(f"✓ {user_name}: Selected {n_samples}/{len(image_files)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['user_name']


class PrototypeBuilder:
    """原型构建器"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: 分类器模型路径
            device: 计算设备
        """
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载预训练分类器"""
        print(f"📦 Loading classifier from: {model_path}")
        
        # 加载模型
        model = DomainAdaptiveClassifier(num_classes=31).to(self.device)
        # PyTorch 2.6+ 需要设置 weights_only=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✓ Model loaded successfully")
        return model
    
    def extract_features(self, dataloader):
        """提取特征向量"""
        features = []
        labels = []
        user_names = []
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_users in tqdm(dataloader, desc="Extracting features"):
                batch_images = batch_images.to(self.device)
                
                # 提取特征：backbone + feature_projector
                # 这是域适应模型的特征表示层
                backbone_features = self.model.backbone(batch_images)
                feat = self.model.feature_projector(backbone_features)
                
                features.append(feat.cpu())
                labels.extend(batch_labels.tolist())
                user_names.extend(batch_users)
        
        features = torch.cat(features, dim=0)
        return features, labels, user_names
    
    def compute_prototypes(self, features, labels, normalize=True):
        """计算每个类的原型"""
        num_classes = max(labels) + 1
        prototypes = []
        
        for class_id in range(num_classes):
            # 获取该类的所有特征
            class_mask = [i for i, l in enumerate(labels) if l == class_id]
            if len(class_mask) == 0:
                print(f"⚠️ No samples for class {class_id}")
                prototypes.append(torch.zeros(features.shape[1]))
                continue
            
            class_features = features[class_mask]
            
            # 计算均值原型
            prototype = class_features.mean(dim=0)
            
            # L2归一化
            if normalize:
                prototype = prototype / prototype.norm(2)
            
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        return prototypes
    
    def build_and_save(self, data_dir, output_path, support_size=15):
        """构建原型并保存"""
        print("\n" + "="*60)
        print("🔧 BUILDING TARGET DOMAIN PROTOTYPES")
        print("="*60)
        
        # 数据变换（与训练时一致）
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建支持集数据集
        dataset = TargetDomainDataset(
            data_dir=data_dir,
            transform=transform,
            support_size=support_size
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
        
        print(f"\n📊 Support set statistics:")
        print(f"   • Total samples: {len(dataset)}")
        print(f"   • Samples per user: {support_size}")
        print(f"   • Number of users: {len(set([s['label'] for s in dataset.samples]))}")
        
        # 提取特征
        print("\n🎯 Extracting features...")
        features, labels, user_names = self.extract_features(dataloader)
        print(f"✓ Extracted features: {features.shape}")
        
        # 计算原型
        print("\n📐 Computing prototypes...")
        prototypes = self.compute_prototypes(features, labels, normalize=True)
        print(f"✓ Computed prototypes: {prototypes.shape}")
        
        # 构建元数据
        samples_per_user = {}
        for label, user in zip(labels, user_names):
            if user not in samples_per_user:
                samples_per_user[user] = 0
            samples_per_user[user] += 1
        
        # 保存原型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'prototypes': prototypes,
            'user_ids': list(range(prototypes.shape[0])),
            'feature_dim': prototypes.shape[1],
            'metadata': {
                'samples_per_user': samples_per_user,
                'model_path': str(self.model._get_name()),
                'data_path': str(data_dir),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'support_size': support_size,
                    'normalize': True,
                    'aggregation': 'mean'
                }
            }
        }
        
        torch.save(save_dict, output_path)
        print(f"\n💾 Prototypes saved to: {output_path}")
        
        # 打印统计信息
        print("\n📈 Prototype statistics:")
        for i, (user_id, count) in enumerate(samples_per_user.items()):
            proto_norm = prototypes[i].norm(2).item()
            print(f"   • {user_id}: {count} samples, prototype norm: {proto_norm:.3f}")
        
        print("\n✅ Prototype building completed successfully!")
        return save_dict


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Build target domain prototypes for PNC')
    
    # Kaggle环境默认路径    
    parser.add_argument('--data-dir', type=str, 
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data (背包步态)')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/input/best-calibrated-model-pth/best_calibrated_model.pth',
                       help='Path to pretrained classifier')
    parser.add_argument('--output-path', type=str,
                       default='/kaggle/working/target_prototypes.pt',
                       help='Path to save prototypes')
    parser.add_argument('--support-size', type=int, default=15,
                       help='Number of support samples per user')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for computation')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 构建原型
    builder = PrototypeBuilder(
        model_path=args.model_path,
        device=args.device
    )
    
    builder.build_and_save(
        data_dir=args.data_dir,
        output_path=args.output_path,
        support_size=args.support_size
    )


if __name__ == '__main__':
    main()
