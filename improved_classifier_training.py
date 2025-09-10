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
    """全局负样本对比损失函数"""
    
    def __init__(self, num_classes, temperature=0.07, margin=0.5, memory_size=200):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.margin = margin
        self.memory_size = memory_size
        
        # 为每个类别维护特征memory bank
        self.register_buffer('memory_bank', torch.randn(num_classes, memory_size, 512))
        self.register_buffer('memory_ptr', torch.zeros(num_classes, dtype=torch.long))
        self.memory_bank = F.normalize(self.memory_bank, dim=2)
    
    @torch.no_grad()
    def update_memory_bank(self, features, labels):
        """更新memory bank"""
        features_normalized = F.normalize(features, dim=1)
        
        for i, label in enumerate(labels):
            label = label.item()
            ptr = self.memory_ptr[label].item()
            
            # 直接更新，因为已经在no_grad上下文中
            self.memory_bank[label, ptr] = features_normalized[i].detach()
            self.memory_ptr[label] = (ptr + 1) % self.memory_size
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size] 用户ID
        """
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        
        # 更新memory bank - 使用detached features避免梯度问题
        with torch.no_grad():
            self.update_memory_bank(features.detach(), labels)
        
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
    """用户间对比损失函数 - 简化版本避免梯度问题"""
    
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim] 
        labels: [batch_size] 用户ID
        """
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 归一化特征
        features_norm = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.div(torch.matmul(features_norm, features_norm.T), self.temperature)
        
        # 创建标签mask
        labels_expanded = labels.view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T)
        
        # 正样本：同标签但不是自己
        eye_mask = torch.eye(batch_size, device=features.device)
        pos_mask = torch.mul(mask.float(), (1.0 - eye_mask))
        
        # 负样本：不同标签
        neg_mask = (~mask).float()
        
        # 计算损失
        if pos_mask.sum() > 0:
            pos_sim = sim_matrix * pos_mask
            pos_loss = -torch.sum(pos_sim) / torch.sum(pos_mask)
        else:
            pos_loss = torch.tensor(0.0, device=features.device)
        
        if neg_mask.sum() > 0:
            neg_sim = sim_matrix * neg_mask
            # 简化的负样本损失，不使用hard negative mining
            neg_loss = torch.sum(neg_sim) / torch.sum(neg_mask)
        else:
            neg_loss = torch.tensor(0.0, device=features.device)
        
        return pos_loss + neg_loss
    

class SupConLoss(nn.Module):
    """监督对比损失 - 避免所有原地操作"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 归一化特征
        features_norm = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # 创建正样本mask
        labels_expanded = labels.view(-1, 1)
        pos_mask = torch.eq(labels_expanded, labels_expanded.T).float()
        
        # 移除对角线 - 避免原地操作
        eye_mask = torch.eye(batch_size, device=device)
        pos_mask = torch.sub(pos_mask, eye_mask)
        
        # 负样本mask
        neg_mask = torch.ne(labels_expanded, labels_expanded.T).float()
        
        # 数值稳定性
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = torch.sub(sim_matrix, sim_max.detach())
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(sim_matrix)
        
        # 分母：所有负样本 + 正样本
        denominator = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True) + \
                     torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # 分子：正样本
        numerator = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # 避免除零
        loss = torch.neg(torch.log(torch.div(numerator, denominator + 1e-8)))
        
        # 只计算有正样本的行
        valid_mask = (pos_mask.sum(dim=1) > 0)
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ImprovedMicroDopplerDataset(Dataset):
    """改进的微多普勒数据集，包含强数据增强"""
    
    def __init__(self, data_dir, split='train', transform=None, contrastive_pairs=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.contrastive_pairs = contrastive_pairs
        self.samples = []
        
        # 收集所有样本
        user_samples = defaultdict(list)
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 查找ID_*目录
        id_dirs = list(self.data_dir.glob("ID_*"))
        
        if len(id_dirs) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到ID_*格式的用户目录")
        
        for user_dir in sorted(id_dirs):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1]) - 1  # 转换为0-based索引
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in user_dir.glob(ext):
                        user_samples[user_id].append(str(img_path))
        
        if not user_samples:
            raise ValueError("未找到任何图像文件")
        
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
        
        # 微多普勒图像专用变换（最小增强，保持频谱结构）
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # 只使用极轻微的噪声增强，不破坏频谱结构
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 可选：极小的高斯噪声（模拟测量噪声） - 避免原地操作
                # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01 if torch.rand(1).item() < 0.3 else x)
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
    """改进的分类器，专为微多普勒信号优化 - 完全避免inplace操作"""
    
    def __init__(self, num_classes, backbone='resnet18', dropout_rate=0.3, freeze_layers=True):
        super().__init__()
        
        # 使用标准ResNet18避免TIMM的潜在inplace问题
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=True)
        # 移除最后的分类层
        self.backbone.fc = nn.Identity()
        feature_dim = 512
        
        # 递归禁用所有ReLU的inplace操作
        self._disable_inplace_operations(self.backbone)
        
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
        
        # 分类头 - 确保所有激活函数都不是inplace
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),  # 明确禁用inplace
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=False),  # 明确禁用inplace
            nn.Linear(128, 64)
        )
    
    def _disable_inplace_operations(self, module):
        """递归禁用模块中所有的inplace操作"""
        for child_name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                # 替换inplace=True的ReLU
                setattr(module, child_name, nn.ReLU(inplace=False))
            elif isinstance(child, nn.ReLU6):
                setattr(module, child_name, nn.ReLU6(inplace=False))
            elif isinstance(child, nn.LeakyReLU):
                setattr(module, child_name, nn.LeakyReLU(child.negative_slope, inplace=False))
            else:
                # 递归处理子模块
                self._disable_inplace_operations(child)
    
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
        print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")
    
    # 损失函数
    if args.use_focal_loss:
        classification_criterion = FocalLoss()
    elif args.use_label_smoothing:
        classification_criterion = LabelSmoothingLoss(args.num_classes)
    else:
        classification_criterion = nn.CrossEntropyLoss()
    
    # 暂时完全禁用对比学习，先确保基础训练正常
    if is_main_process():
        print("🔧 暂时禁用对比学习，调试基础训练流程")
    contrastive_criterion = None
    
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
    
    # 禁用异常检测，避免额外开销
    torch.autograd.set_detect_anomaly(False)
    
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
            
            # 最简化的训练循环 - 纯分类训练
            if isinstance(data, (tuple, list)):
                data = data[0]
            
            data = data.to(device)
            
            # 纯分类训练
            logits = model(data)
            total_loss = classification_criterion(logits, target)
            classification_loss = total_loss
            contrastive_loss = torch.tensor(0.0, device=device)
            
            pred = logits.argmax(dim=1)
            
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
            
            contrastive_losses.append(contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0)
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
