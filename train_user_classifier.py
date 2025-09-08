"""
用户分类器训练脚本
使用真实微多普勒图像训练ResNet18分类器，用于评估生成样本质量
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class MicroDopplerDataset(Dataset):
    """微多普勒数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 扫描所有用户目录
        for user_dir in sorted(self.data_dir.glob("user_*")):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1])
                
                # 扫描该用户的所有图像
                for img_path in user_dir.glob("*.png"):
                    self.samples.append((str(img_path), user_id))
        
        print(f"Found {len(self.samples)} samples from {len(set([s[1] for s in self.samples]))} users")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_transforms():
    """创建数据变换（无数据增强）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet标准输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    return transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def split_dataset(dataset, train_ratio=0.8):
    """划分训练和验证集"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # 确保每个用户在训练和验证集中都有样本
    user_samples = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in user_samples:
            user_samples[label] = []
        user_samples[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for user_id, indices in user_samples.items():
        np.random.shuffle(indices)
        split_point = max(1, int(len(indices) * train_ratio))  # 至少保证每个用户有1个验证样本
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser(description='Train user classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to original image dataset')
    parser.add_argument('--output_dir', type=str, default='./classifier_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=31, help='Number of users')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio') 
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据变换
    transform = create_transforms()
    
    # 加载数据集
    print("Loading dataset...")
    full_dataset = MicroDopplerDataset(args.data_dir, transform=transform)
    
    # 划分训练和验证集
    np.random.seed(42)  # 固定随机种子
    train_dataset, val_dataset = split_dataset(full_dataset, args.train_ratio)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    print("Creating ResNet18 model...")
    model = timm.create_model('resnet18', pretrained=True, num_classes=args.num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
    
    # 保存最佳模型
    model_path = output_dir / 'best_classifier.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'best_val_acc': best_val_acc,
        'num_classes': args.num_classes,
        'model_name': 'resnet18'
    }, model_path)
    print(f"Best model saved to {model_path}")
    
    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()
    
    # 最终验证报告
    model.load_state_dict(best_model_state)
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # 生成分类报告
    report = classification_report(val_labels, val_preds, target_names=[f'User_{i:02d}' for i in range(args.num_classes)])
    
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Classification report saved to: {report_path}")

if __name__ == "__main__":
    main()
