"""
改进的分类器训练方案
基于小数据集分类的最佳实践，包含对比学习和正则化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import random


class InterUserContrastiveLoss(nn.Module):
    """用户间对比损失函数 - 专门处理用户间差异小的问题"""
    
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin  # 用于hard negative mining
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size] 用户ID
        """
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        
        # 计算所有样本间的相似度
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正负样本mask
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        negative_mask = 1 - positive_mask
        
        # 移除对角线（自己和自己）
        eye_mask = torch.eye(batch_size).to(features.device)
        positive_mask = positive_mask - eye_mask
        negative_mask = negative_mask - eye_mask
        
        # 计算正样本损失（同用户样本应该相似）
        pos_sim = similarity_matrix * positive_mask
        pos_loss = -pos_sim.sum() / positive_mask.sum().clamp(min=1)
        
        # 计算负样本损失（不同用户样本应该不相似）
        # 使用hard negative mining - 只关注难区分的负样本
        neg_sim = similarity_matrix * negative_mask
        hard_negatives = (neg_sim > self.margin) * negative_mask
        
        if hard_negatives.sum() > 0:
            neg_loss = neg_sim[hard_negatives > 0].mean()
        else:
            neg_loss = neg_sim[negative_mask > 0].mean()
        
        # 总损失 = 减少正样本距离 + 增加负样本距离
        total_loss = pos_loss + neg_loss
        
        return total_loss
    

class SupConLoss(nn.Module):
    """监督对比损失 - 更适合多类分类的对比学习"""
    
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算anchor和所有样本的相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # 为数值稳定性减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 移除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算正样本的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ImprovedMicroDopplerDataset(Dataset):
    """改进的微多普勒数据集，包含强数据增强"""
    
    def __init__(self, data_dir, split='train', transform=None, contrastive_pairs=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.contrastive_pairs = contrastive_pairs
        self.samples = []
        
        # 收集所有样本
        user_samples = defaultdict(list)
        for user_dir in sorted(self.data_dir.glob("ID_*")):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1]) - 1
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in user_dir.glob(ext):
                        user_samples[user_id].append(str(img_path))
        
        # 划分训练/验证集
        for user_id, paths in user_samples.items():
            random.shuffle(paths)
            split_idx = int(len(paths) * 0.8)
            
            if split == 'train':
                selected_paths = paths[:split_idx]
            else:  # validation
                selected_paths = paths[split_idx:]
            
            for path in selected_paths:
                self.samples.append((path, user_id))
        
        print(f"{split.capitalize()} set: {len(self.samples)} samples")
        
        # 微多普勒图像专用变换（最小增强，保持频谱结构）
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # 只使用极轻微的噪声增强，不破坏频谱结构
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 可选：极小的高斯噪声（模拟测量噪声）
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01 if torch.rand(1) < 0.3 else x)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if transform:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.contrastive_pairs and self.split == 'train':
                # 生成对比样本对
                image1 = self.transform(image)
                image2 = self.transform(image)  # 同一图像的不同增强
                return (image1, image2), label
            else:
                image = self.transform(image)
                return image, label
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回零张量
            if self.contrastive_pairs:
                return (torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)), label
            else:
                return torch.zeros(3, 224, 224), label


class ImprovedClassifier(nn.Module):
    """改进的分类器，专为微多普勒信号优化"""
    
    def __init__(self, num_classes, backbone='resnet18', dropout_rate=0.3, freeze_layers=True):
        super().__init__()
        
        # ResNet18是微多普勒分类的经典选择
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='avg')
        feature_dim = 512
        
        # 灵活的层冻结策略
        if freeze_layers == 'minimal':
            # 最小冻结：只冻结最早的卷积层
            for name, param in self.backbone.named_parameters():
                if any(x in name for x in ['conv1', 'bn1']):
                    param.requires_grad = False
        elif freeze_layers == 'moderate':
            # 中等冻结：冻结早期层，保留适应性
            for name, param in self.backbone.named_parameters():
                if any(x in name for x in ['conv1', 'bn1', 'layer1']):
                    param.requires_grad = False
        elif freeze_layers == 'aggressive':
            # 激进冻结：冻结更多层（小数据集）
            for name, param in self.backbone.named_parameters():
                if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
        elif freeze_layers == 'none':
            # 不冻结任何层（风险更高但可能效果更好）
            pass
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        
        if return_features:
            projected = self.projection_head(features)
            return features, projected
        
        logits = self.classifier(features)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.full_like(pred, self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        
        return F.kl_div(F.log_softmax(pred, dim=1), smooth_target, reduction='batchmean')


def train_with_contrastive_learning(model, train_loader, val_loader, device, args):
    """使用对比学习训练分类器"""
    
    # 损失函数
    if args.use_focal_loss:
        classification_criterion = FocalLoss()
    elif args.use_label_smoothing:
        classification_criterion = LabelSmoothingLoss(args.num_classes)
    else:
        classification_criterion = nn.CrossEntropyLoss()
    
    # 选择对比损失类型
    if args.contrastive_type == 'interuser':
        contrastive_criterion = InterUserContrastiveLoss(
            temperature=args.contrastive_temperature,
            margin=args.contrastive_margin
        )
    elif args.contrastive_type == 'supcon':
        contrastive_criterion = SupConLoss(temperature=args.contrastive_temperature)
    else:
        contrastive_criterion = SupConLoss(temperature=args.contrastive_temperature)  # 默认使用SupCon
    
    # 优化器 - 使用更小的学习率和更强的weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_acc = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'contrastive_loss': [], 'classification_loss': []
    }
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        contrastive_losses = []
        classification_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch_data in enumerate(pbar):
            data, target = batch_data
            target = target.to(device)
            
            # 调试信息
            # print(f"Data type: {type(data)}, Target type: {type(target)}")
            
            # 检查是否是对比学习的tuple对
            if isinstance(data, (tuple, list)) and len(data) == 2:
                data1, data2 = data
                # 确保data1和data2是tensor
                if isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
                    data1, data2 = data1.to(device), data2.to(device)
                    
                    # 前向传播
                    features1, proj1 = model(data1, return_features=True)
                    features2, proj2 = model(data2, return_features=True)
                    
                    # 分类损失
                    logits1 = model.classifier(features1)
                    logits2 = model.classifier(features2)
                    
                    cls_loss1 = classification_criterion(logits1, target)
                    cls_loss2 = classification_criterion(logits2, target)
                    classification_loss = (cls_loss1 + cls_loss2) / 2
                    
                    # 对比损失
                    combined_proj = torch.cat([proj1, proj2], dim=0)
                    combined_labels = torch.cat([target, target], dim=0)
                    contrastive_loss = contrastive_criterion(combined_proj, combined_labels)
                    
                    # 总损失
                    total_loss = classification_loss + args.contrastive_weight * contrastive_loss
                    
                    # 统计
                    pred = logits1.argmax(dim=1)
                else:
                    raise ValueError(f"Expected tensors in contrastive pair, got {type(data1)}, {type(data2)}")
                
            elif isinstance(data, torch.Tensor):  # 常规张量数据
                data = data.to(device)
                
                if args.use_contrastive:
                    # 即使是常规数据，也可以用对比学习
                    features, proj = model(data, return_features=True)
                    logits = model.classifier(features)
                    
                    classification_loss = classification_criterion(logits, target)
                    contrastive_loss = contrastive_criterion(proj, target)
                    total_loss = classification_loss + args.contrastive_weight * contrastive_loss
                else:
                    logits = model(data)
                    classification_loss = classification_criterion(logits, target)
                    contrastive_loss = torch.tensor(0.0)
                    total_loss = classification_loss
                
                pred = logits.argmax(dim=1)
            
            else:
                raise ValueError(f"Unexpected data type: {type(data)}, content: {data}")
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            contrastive_losses.append(contrastive_loss.item())
            classification_losses.append(classification_loss.item())
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                logits = model(data)
                loss = classification_criterion(logits, target)
                
                pred = logits.argmax(dim=1)
                val_loss += loss.item()
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # 更新学习率
        scheduler.step()
        
        # 统计
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['contrastive_loss'].append(np.mean(contrastive_losses))
        history['classification_loss'].append(np.mean(classification_losses))
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停和最佳模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    return model, best_model_state, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='Improved classifier training')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./improved_classifier', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (smaller for small dataset)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (smaller)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (stronger)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # 对比学习参数
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='Contrastive temperature')
    parser.add_argument('--contrastive_type', type=str, default='supcon', 
                       choices=['interuser', 'supcon'],
                       help='Contrastive loss type: interuser(hard negative mining), supcon(supervised contrastive)')
    parser.add_argument('--contrastive_margin', type=float, default=0.5, 
                       help='Margin for hard negative mining in interuser contrastive loss')
    
    # 损失函数选择
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing')
    
    # 模型选择 - ResNet18专为微多普勒优化
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--freeze_layers', type=str, default='moderate', 
                       choices=['none', 'minimal', 'moderate', 'aggressive'],
                       help='Layer freezing strategy: none(risk overfitting), minimal(conv1+bn1), moderate(+layer1), aggressive(+layer2)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 数据集
    train_dataset = ImprovedMicroDopplerDataset(
        args.data_dir, 
        split='train', 
        contrastive_pairs=args.use_contrastive
    )
    val_dataset = ImprovedMicroDopplerDataset(
        args.data_dir, 
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    model = ImprovedClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        dropout_rate=args.dropout_rate,
        freeze_layers=args.freeze_layers
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练
    model, best_state, best_acc, history = train_with_contrastive_learning(
        model, train_loader, val_loader, device, args
    )
    
    # 保存最佳模型
    model_path = output_dir / 'best_improved_classifier.pth'
    torch.save({
        'model_state_dict': best_state,
        'best_val_acc': best_acc,
        'num_classes': args.num_classes,
        'model_name': args.backbone,
        'args': vars(args)
    }, model_path)
    
    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
