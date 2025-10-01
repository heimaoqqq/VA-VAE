#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 ResNet18 分类器训练脚本（不使用对比学习）
作为真正的 Baseline 对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime


class StandardResNet18(nn.Module):
    """
    标准 ResNet18 分类器
    没有对比学习，没有特殊技巧，纯粹的分类器
    """
    def __init__(self, num_classes=31, pretrained=True):
        super(StandardResNet18, self).__init__()
        # 使用预训练的 ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.resnet(x)


class GaitDataset(Dataset):
    """简单的步态数据集加载器"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 加载所有图像
        user_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for user_dir in user_dirs:
            if not user_dir.name.startswith('ID_'):
                continue
            
            user_id = int(user_dir.name.split('_')[1]) - 1  # ID_1 -> 0
            
            image_files = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
            
            for img_path in image_files:
                self.samples.append({
                    'path': img_path,
                    'label': user_id
                })
        
        print(f"✅ 加载 {len(self.samples)} 个样本，{len(user_dirs)} 个用户")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """创建训练和验证数据加载器"""
    
    # 数据增强（训练集）
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集（不增强）
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建完整数据集
    full_dataset = GaitDataset(data_dir, transform=train_transform)
    
    # 分割训练集和验证集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 创建训练集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # 创建验证集（使用不同的transform）
    val_dataset_full = GaitDataset(data_dir, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, save_dir, patience=10):
    """完整训练流程"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "="*80)
    print("🚀 开始训练标准 ResNet18 分类器")
    print("="*80)
    
    for epoch in range(1, epochs + 1):
        print(f"\n📍 Epoch {epoch}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            torch.save(checkpoint, save_dir / 'best_standard_resnet18.pth')
            print(f"   ✅ 保存最佳模型 (验证准确率: {best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️  验证准确率未提升 ({epochs_no_improve}/{patience})")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            break
    
    # 保存训练历史
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"📊 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"💾 模型保存在: {save_dir / 'best_standard_resnet18.pth'}")
    print(f"📈 训练历史保存在: {history_path}")
    
    return model, best_model_state, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='标准 ResNet18 分类器训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./standard_classifier',
                       help='输出目录')
    parser.add_argument('--num_classes', type=int, default=31,
                       help='类别数量')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no_pretrain', action='store_true',
                       help='不使用ImageNet预训练')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建数据加载器
    print(f"\n📂 加载数据从: {args.data_dir}")
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )
    
    # 创建模型
    print(f"\n🏗️  创建标准 ResNet18 模型")
    print(f"   类别数: {args.num_classes}")
    print(f"   预训练: {'否' if args.no_pretrain else '是 (ImageNet)'}")
    
    model = StandardResNet18(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrain
    ).to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # 打印配置
    print(f"\n⚙️  训练配置:")
    print(f"   批次大小: {args.batch_size}")
    print(f"   学习率: {args.lr}")
    print(f"   权重衰减: {args.weight_decay}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   Early stopping patience: {args.patience}")
    
    # 训练模型
    model, best_model_state, best_val_acc, history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.output_dir, args.patience
    )
    
    # 保存配置
    config = {
        'num_classes': args.num_classes,
        'pretrained': not args.no_pretrain,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(history['train_loss']),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = Path(args.output_dir) / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  配置保存在: {config_path}")


if __name__ == '__main__':
    main() 
