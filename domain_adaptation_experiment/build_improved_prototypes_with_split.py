#!/usr/bin/env python3
"""
为ImprovedClassifier构建目标域原型 - 带数据集划分功能
严格划分支持集和测试集，避免数据泄漏
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
from improved_classifier_training import ImprovedClassifier


class SplitTargetDomainDataset(Dataset):
    """目标域数据集 - 严格划分支持集和测试集"""
    
    def __init__(self, data_dir, transform=None, support_size=10, mode='support', seed=42):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据变换
            support_size: 每个用户的支持集大小
            mode: 'support' or 'test'
            seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.support_size = support_size
        self.mode = mode
        self.samples = []
        
        # 设置随机种子确保可重复
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载数据集
        self._split_and_load()
        
    def _split_and_load(self):
        """严格划分并加载支持集或测试集"""
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
            
            # 随机打乱并划分
            random.shuffle(image_files)
            
            # 划分支持集和测试集
            support_files = image_files[:self.support_size]
            test_files = image_files[self.support_size:]
            
            # 根据mode选择数据
            if self.mode == 'support':
                selected_files = support_files
                print(f"✓ {user_name}: Support set {len(support_files)} samples")
            else:  # test mode
                selected_files = test_files
                print(f"✓ {user_name}: Test set {len(test_files)} samples")
            
            # 添加到数据集
            for img_path in selected_files:
                self.samples.append({
                    'path': img_path,
                    'label': user_id,
                    'user_name': user_name
                })
        
        print(f"\n📊 {self.mode.capitalize()} dataset: {len(self.samples)} samples total")
        
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


class ImprovedPrototypeBuilderWithSplit:
    """ImprovedClassifier的原型构建器 - 带数据划分"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载ImprovedClassifier"""
        print(f"📦 Loading ImprovedClassifier from: {model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取模型配置
        num_classes = checkpoint.get('num_classes', 31)
        backbone = checkpoint.get('backbone', 'resnet18')
        
        # 创建模型
        model = ImprovedClassifier(
            num_classes=num_classes,
            backbone=backbone
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model loaded successfully: {backbone} with {num_classes} classes")
        
        if 'best_val_acc' in checkpoint:
            print(f"   Original validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            
        return model
    
    def extract_features(self, dataloader):
        """提取特征向量"""
        features = []
        labels = []
        user_names = []
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_users in tqdm(dataloader, desc="Extracting features"):
                batch_images = batch_images.to(self.device)
                
                # 提取特征：直接使用backbone输出
                feat = self.model.backbone(batch_images)
                
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
    
    def build_and_save(self, data_dir, output_path, support_size=10, batch_size=32, seed=42):
        """构建并保存原型 - 同时保存数据集划分信息"""
        print("\n" + "="*60)
        print("🔧 BUILDING PROTOTYPES WITH DATA SPLIT")
        print("="*60)
        
        # 数据变换（与训练时一致）
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建支持集数据集
        support_dataset = SplitTargetDomainDataset(
            data_dir=data_dir,
            transform=transform,
            support_size=support_size,
            mode='support',
            seed=seed
        )
        
        support_loader = DataLoader(
            support_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 提取特征
        print("\n🎯 Extracting features from support set...")
        features, labels, user_names = self.extract_features(support_loader)
        print(f"✓ Extracted features: {features.shape}")
        
        # 计算原型
        print("\n📐 Computing prototypes...")
        prototypes = self.compute_prototypes(features, labels)
        print(f"✓ Computed prototypes: {prototypes.shape}")
        
        # 保存支持集文件路径（用于排除）
        support_paths = [str(sample['path']) for sample in support_dataset.samples]
        
        # 保存
        save_dict = {
            'prototypes': prototypes,
            'user_ids': list(range(prototypes.shape[0])),
            'feature_dim': prototypes.shape[1],
            'metadata': {
                'model_type': 'ImprovedClassifier',
                'support_size': support_size,
                'num_support_samples': len(support_dataset),
                'feature_extraction': 'backbone_direct',
                'timestamp': datetime.now().isoformat(),
                'seed': seed,
                'data_split': 'strict_split'
            },
            'support_paths': support_paths,  # 关键：保存支持集路径
            'user_stats': {}
        }
        
        # 统计每个用户的信息
        print("\n📈 Prototype statistics:")
        for i in range(prototypes.shape[0]):
            user_samples = sum(1 for l in labels if l == i)
            norm = prototypes[i].norm(2).item()
            save_dict['user_stats'][f'ID_{i+1}'] = {
                'support_samples': user_samples,
                'prototype_norm': norm
            }
            print(f"   • ID_{i+1}: {user_samples} support samples, prototype norm: {norm:.3f}")
        
        # 保存文件
        torch.save(save_dict, output_path)
        print(f"\n💾 Prototypes saved to: {output_path}")
        print(f"   Support set paths saved for exclusion during testing")
        
        # 创建测试集信息
        test_dataset = SplitTargetDomainDataset(
            data_dir=data_dir,
            transform=transform,
            support_size=support_size,
            mode='test',
            seed=seed
        )
        
        print(f"\n📊 Data split summary:")
        print(f"   Support set: {len(support_dataset)} samples (for prototypes)")
        print(f"   Test set: {len(test_dataset)} samples (for evaluation)")
        print(f"   No overlap between sets!")
        
        print("\n✅ Prototype building with strict data split completed!")
        return save_dict


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Build prototypes with strict data split')
    
    # Kaggle环境默认路径
    parser.add_argument('--data-dir', type=str, 
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth',
                       help='Path to ImprovedClassifier model')
    parser.add_argument('--output-path', type=str,
                       default='/kaggle/working/improved_prototypes_split.pt',
                       help='Path to save prototypes')
    parser.add_argument('--support-size', type=int, default=10,
                       help='Number of support samples per user')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data split')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建原型构建器
    builder = ImprovedPrototypeBuilderWithSplit(
        model_path=args.model_path,
        device=args.device
    )
    
    # 构建并保存原型
    builder.build_and_save(
        data_dir=args.data_dir,
        output_path=args.output_path,
        support_size=args.support_size,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
