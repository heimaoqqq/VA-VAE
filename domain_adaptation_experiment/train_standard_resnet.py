#!/usr/bin/env python3
"""
标准ResNet18分类器训练脚本 - 用于LCCS对比实验
包含模型定义和训练代码，纯分类训练，不使用对比学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedMicroDopplerDataset


class StandardResNet18Classifier(nn.Module):
    """标准ResNet18分类器 - 与论文一致"""
    
    def __init__(self, num_classes=31, pretrained=True):
        super().__init__()
        
        # 使用标准ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 保存原始特征维度
        self.feature_dim = self.backbone.fc.in_features  # 512
        
        # 替换分类头
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        # 标准前向传播
        return self.backbone(x)
    
    def extract_features(self, x):
        """提取backbone特征（用于LCCS/PNC）"""
        # 去掉最后的分类层
        features = nn.Sequential(*list(self.backbone.children())[:-1])
        x = features(x)
        x = torch.flatten(x, 1)  # [B, 512]
        return x


def create_standard_classifier(num_classes=31, model_path=None, device='cuda'):
    """创建标准分类器"""
    model = StandardResNet18Classifier(num_classes=num_classes)
    
    if model_path and model_path.exists():
        print(f"📦 Loading standard classifier from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Standard classifier loaded")
    else:
        print("⚠️ No pretrained weights, using ImageNet pretrained backbone")
    
    return model.to(device)

def train_standard_classifier(args):
    """训练标准ResNet18分类器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.3),  # 轻微增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 数据集（8:2划分）
    train_dataset = ImprovedMicroDopplerDataset(
        data_dir=args.data_dir,
        split='train',
        transform=train_transform,
        contrastive_pairs=False,  # 不使用对比学习
        generated_data_dirs=None,
        use_generated=False
    )
    
    val_dataset = ImprovedMicroDopplerDataset(
        data_dir=args.data_dir,
        split='val',
        transform=val_transform,
        contrastive_pairs=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"📊 Training set: {len(train_dataset)} samples")
    print(f"📊 Validation set: {len(val_dataset)} samples")
    
    # 模型
    model = StandardResNet18Classifier(num_classes=31, pretrained=True).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': 31
            }, args.output_path)
            print(f"✅ New best model saved: {val_acc:.2f}%")
        
        scheduler.step()
    
    print(f"\n🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {args.output_path}")


def test_model():
    """测试模型定义"""
    print("🧪 Testing StandardResNet18Classifier...")
    
    # 创建模型
    model = StandardResNet18Classifier(num_classes=31)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(4, 3, 256, 256)
    outputs = model(x)
    features = model.extract_features(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {outputs.shape}")
    print(f"✅ Feature shape: {features.shape}")
    print("🎯 Model definition test passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Standard ResNet18 Classifier')
    parser.add_argument('--data-dir', type=str, 
                       default='/kaggle/working/organized_gait_dataset/Normal_free',
                       help='Training data directory')
    parser.add_argument('--output-path', type=str,
                       default='/kaggle/working/standard_resnet18_classifier.pth',
                       help='Output model path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--test-only', action='store_true',
                       help='Only test model definition')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model()
    else:
        train_standard_classifier(args)
