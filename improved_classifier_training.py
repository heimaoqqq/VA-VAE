"""
改进的分类器训练方案
基于小数据集分类的最佳实践，包含对比学习和正则化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
import os


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化进程组
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


class GlobalNegativeContrastiveLoss(nn.Module):
    """全局负样本对比损失 - 每个用户与所有其他用户对比"""
    
    def __init__(self, num_classes, temperature=0.07, margin=0.5, memory_size=1000):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.margin = margin
        self.memory_size = memory_size
        
        # 为每个类别维护特征memory bank
        self.register_buffer('memory_bank', torch.randn(num_classes, memory_size, 512))
        self.register_buffer('memory_ptr', torch.zeros(num_classes, dtype=torch.long))
        self.memory_bank = F.normalize(self.memory_bank, dim=2)
    
    def update_memory_bank(self, features, labels):
        """更新memory bank"""
        features = F.normalize(features, dim=1)
        
        for i, label in enumerate(labels):
            label = label.item()
            ptr = self.memory_ptr[label].item()
            
            # 循环覆盖更新
            self.memory_bank[label, ptr] = features[i].detach()
            self.memory_ptr[label] = (ptr + 1) % self.memory_size
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size] 用户ID
        """
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        
        # 更新memory bank
        self.update_memory_bank(features, labels)
        
        total_loss = 0
        num_pairs = 0
        
        # 对batch中每个样本计算与全局负样本的对比损失
        for i, anchor_label in enumerate(labels):
            anchor_feature = features[i].unsqueeze(0)  # [1, feature_dim]
            
            # 正样本：同类别的其他样本（batch内 + memory bank）
            positive_features = []
            
            # batch内正样本
            batch_positives = features[labels == anchor_label]
            if len(batch_positives) > 1:  # 除了自己还有其他同类样本
                mask = torch.arange(len(batch_positives)) != (labels == anchor_label).nonzero()[0]
                if mask.any():
                    positive_features.append(batch_positives[mask])
            
            # memory bank中的正样本
            memory_positives = self.memory_bank[anchor_label]  # [memory_size, feature_dim]
            positive_features.append(memory_positives[:50])  # 取前50个避免过多
            
            if positive_features:
                positive_features = torch.cat(positive_features, dim=0)
                pos_similarity = torch.matmul(anchor_feature, positive_features.T) / self.temperature
                pos_loss = -pos_similarity.mean()
            else:
                pos_loss = torch.tensor(0.0, device=features.device)
            
            # 负样本：所有其他类别的样本（全局）
            negative_features = []
            for neg_label in range(self.num_classes):
                if neg_label != anchor_label:
                    # 从memory bank中采样负样本
                    neg_samples = self.memory_bank[neg_label][:20]  # 每个类别20个样本
                    negative_features.append(neg_samples)
            
            if negative_features:
                negative_features = torch.cat(negative_features, dim=0)  # [num_negatives, feature_dim]
                neg_similarity = torch.matmul(anchor_feature, negative_features.T) / self.temperature
                
                # Hard negative mining
                hard_mask = neg_similarity.squeeze() > self.margin
                if hard_mask.any():
                    neg_loss = neg_similarity.squeeze()[hard_mask].mean()
                else:
                    neg_loss = neg_similarity.mean()
            else:
                neg_loss = torch.tensor(0.0, device=features.device)
            
            # 累积损失
            total_loss += pos_loss + neg_loss
            num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=features.device)


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
        print(f"🔍 扫描数据目录: {self.data_dir}")
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 查找ID_*目录
        id_dirs = list(self.data_dir.glob("ID_*"))
        print(f"找到 {len(id_dirs)} 个ID目录")
        
        if len(id_dirs) == 0:
            print("❌ 未找到ID_*目录，检查以下可能的目录:")
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
            raise ValueError(f"在 {self.data_dir} 中未找到ID_*格式的用户目录")
        
        for user_dir in sorted(id_dirs):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1])  # 保持原始ID编号
                total_files = 0
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    files = list(user_dir.glob(ext))
                    total_files += len(files)
                    for img_path in files:
                        user_samples[user_id].append(str(img_path))
                print(f"  ID_{user_id}: {total_files} 个文件")
        
        if not user_samples:
            raise ValueError("未找到任何图像文件")
        
        total_samples = sum(len(paths) for paths in user_samples.values())
        print(f"总共收集到 {total_samples} 个样本，来自 {len(user_samples)} 个用户")
        
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


def train_with_contrastive_learning(model, train_loader, val_loader, device, args, rank=0):
    """改进的训练函数，集成对比学习"""
    
    # 分布式训练设置
    def is_main_process():
        return rank == 0
    
    # 验证数据集是否为空
    if len(train_loader.dataset) == 0:
        if is_main_process():
            print("❌ 训练数据集为空，无法开始训练")
        return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
    
    if len(val_loader.dataset) == 0:
        if is_main_process():
            print("❌ 验证数据集为空，无法开始训练")
        return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
    
    if is_main_process():
        print(f"✅ 数据集验证通过: 训练集 {len(train_loader.dataset)} 样本, 验证集 {len(val_loader.dataset)} 样本")
    
    # 损失函数
    if args.use_focal_loss:
        classification_criterion = FocalLoss()
    elif args.use_label_smoothing:
        classification_criterion = LabelSmoothingLoss(args.num_classes)
    else:
        classification_criterion = nn.CrossEntropyLoss()
    
    # 选择对比损失类型
    if args.contrastive_type == 'global':
        contrastive_criterion = GlobalNegativeContrastiveLoss(
            num_classes=args.num_classes,
            temperature=args.contrastive_temperature,
            margin=args.contrastive_margin,
            memory_size=200  # 每类存储200个特征向量
        )
    elif args.contrastive_type == 'interuser':
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
        
        # 只在主进程显示进度条
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            pbar = train_loader
        
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
            
            # 只在主进程更新进度条
            if is_main_process() and isinstance(pbar, tqdm):
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
        
        # 防止除零错误
        if len(train_loader) == 0:
            print("❌ 训练数据集为空，请检查数据路径和格式")
            return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
        
        if len(val_loader) == 0:
            print("❌ 验证数据集为空，请检查数据路径和格式")
            return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
        
        # 统计
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['contrastive_loss'].append(np.mean(contrastive_losses))
        history['classification_loss'].append(np.mean(classification_losses))
        
        # 只在主进程打印
        if is_main_process():
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停和最佳模型保存（只在主进程）
        if is_main_process():
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                patience_counter = 0
                print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    return model, best_model_state, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='Improved classifier training with distributed support')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./improved_classifier', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (smaller)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (stronger)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # 对比学习参数
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='Contrastive temperature')
    parser.add_argument('--contrastive_type', type=str, default='supcon', 
                       choices=['global', 'interuser', 'supcon'],
                       help='Contrastive loss type: global(memory bank all users), interuser(hard negative mining), supcon(supervised contrastive)')
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
    
    # 初始化分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cpu')
    
    if is_main_process():
        print(f"Using distributed training with {world_size} GPUs")
        print(f"Current device: {device}")
    
    # 创建输出目录（只在主进程）
    if is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42 + rank)  # 每个进程不同的随机种子
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    
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
    
    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        sampler=val_sampler,
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
    
    # 包装为分布式模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
    
    # 训练
    model, best_state, best_acc, history = train_with_contrastive_learning(
        model, train_loader, val_loader, device, args, rank
    )
    
    # 只在主进程保存模型
    if is_main_process():
        output_dir = Path(args.output_dir)
        
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
    
    # 清理分布式训练
    cleanup_distributed()


if __name__ == "__main__":
    main()
