"""
训练具有良好校准度的分类器
关键改进：标签平滑、混合增强、焦点损失、温度退火
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os
from PIL import Image


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失 - 防止过度自信"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        n = pred.size(0)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


class FocalLoss(nn.Module):
    """焦点损失 - 减少易分类样本权重，专注困难样本"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class MixupAugmentation:
    """Mixup数据增强 - 改善泛化和校准"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class DomainAdaptiveClassifier(nn.Module):
    """针对小数据集和域适应设计的分类器"""
    def __init__(self, num_classes=31, dropout_rate=0.3, feature_dim=512):
        super().__init__()
        
        # 使用ResNet18作为backbone（预训练权重很重要）
        self.backbone = models.resnet18(pretrained=True)
        
        # 冻结前面的层，只微调后面的层（防止小数据集过拟合）
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # 特征提取维度
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 特征投影层（用于对比学习）
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头（简单但有效，避免过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 身份特征记忆库（用于筛选时的特征匹配）
        self.register_buffer('feature_bank', torch.zeros(num_classes, feature_dim))
        self.register_buffer('feature_count', torch.zeros(num_classes))
        
        # 温度参数
        self.temperature = 1.0
    
    def forward(self, x, labels=None, update_bank=False):
        # 特征提取
        backbone_features = self.backbone(x)
        features = self.feature_projector(backbone_features)
        
        # 更新特征库（训练时）
        if update_bank and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    self.feature_bank[label] = self.feature_bank[label] * 0.95 + features[i] * 0.05
                    self.feature_count[label] += 1
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features
    
    def compute_feature_similarity(self, features):
        """计算特征与各类别原型的相似度"""
        # 归一化特征
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.feature_bank, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(features_norm, prototypes_norm.T)
        return similarity


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_contrastive=True):
    """训练一个epoch，包含对比学习"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_contrast_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播（更新特征库）
        logits, features = model(images, labels, update_bank=True)
        
        # 分类损失
        ce_loss = criterion(logits, labels)
        
        # 对比损失（增强身份特征学习）
        contrast_loss = torch.tensor(0.0).to(device)
        if use_contrastive and epoch > 5:  # 前5个epoch只用分类损失
            # 获取实际模型（处理DDP wrapper）
            actual_model = model.module if hasattr(model, 'module') else model
            
            # 计算特征相似度
            feature_sim = actual_model.compute_feature_similarity(features)
            
            # 动态获取类别数量
            num_classes = feature_sim.size(1)
            
            # 对比损失：同类特征相似，异类特征分离
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            positive_sim = (feature_sim * labels_one_hot).sum(dim=1)
            negative_sim = (feature_sim * (1 - labels_one_hot)).max(dim=1)[0]
            
            # Margin loss
            margin = 0.3
            contrast_loss = F.relu(negative_sim - positive_sim + margin).mean()
        
        # 总损失
        loss = ce_loss + 0.1 * contrast_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_contrast_loss += contrast_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_contrast_loss = total_contrast_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, avg_ce_loss, avg_contrast_loss


def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            logits, features = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            _, predicted = probs.max(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    # 计算ECE
    ece = compute_ece(np.array(all_probs), np.array(all_labels))
    
    # 计算特征多样性（用于评估身份保留）
    all_features = torch.cat(all_features, dim=0)
    feature_std = all_features.std(dim=0).mean().item()
    
    return avg_loss, accuracy, ece, feature_std


def compute_ece(probs, labels, n_bins=10):
    """计算期望校准误差"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


class SplitDataset(Dataset):
    """基于dataset_split.json的数据集类"""
    def __init__(self, split_data, transform=None):
        self.data = split_data
        self.transform = transform
        
        # 创建类别到索引的映射
        unique_classes = sorted(set(item['class'] for item in self.data))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.classes = unique_classes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['path']
        class_name = item['class']
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和类别索引
        label = self.class_to_idx[class_name]
        return image, label


def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_dataset_split(split_file):
    """加载数据集划分文件"""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    return split_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', type=str, default='/kaggle/working/dataset_split.json',
                       help='数据集划分文件路径')
    parser.add_argument('--output_dir', type=str, default='./calibrated_classifier')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)  # 小数据集用小batch
    parser.add_argument('--lr', type=float, default=5e-4)  # 更小的学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.05)  # 适度平滑
    parser.add_argument('--dropout', type=float, default=0.3)  # 适度dropout
    parser.add_argument('--mixup_alpha', type=float, default=0.0)  # 小数据集不用mixup
    parser.add_argument('--use_focal_loss', action='store_true')
    
    args = parser.parse_args()
    
    # 初始化DDP
    rank, world_size, local_rank = setup_ddp()
    is_distributed = world_size > 1
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 只在主进程创建输出目录
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_distributed:
        dist.barrier()  # 等待主进程创建完目录
    
    # 加载数据集划分
    if rank == 0:
        print(f"Loading dataset split from: {args.split_file}")
    
    split_data = load_dataset_split(args.split_file)
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    # 创建数据集
    train_dataset = SplitDataset(split_data['train'], transform=train_transform)
    val_dataset = SplitDataset(split_data['val'], transform=val_transform)
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Classes: {len(train_dataset.classes)}")
    
    # 分布式采样器
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, 
                                         num_replicas=world_size, 
                                         rank=rank, 
                                         shuffle=True)
        val_sampler = DistributedSampler(val_dataset, 
                                       num_replicas=world_size, 
                                       rank=rank, 
                                       shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size, 
                             sampler=train_sampler,
                             shuffle=(train_sampler is None),
                             num_workers=2,  # 分布式训练用较少worker
                             pin_memory=True)
    
    val_loader = DataLoader(val_dataset, 
                           batch_size=args.batch_size, 
                           sampler=val_sampler,
                           shuffle=False,
                           num_workers=2,
                           pin_memory=True)
    
    # 模型初始化
    num_classes = len(train_dataset.classes)
    model = DomainAdaptiveClassifier(num_classes=num_classes, dropout_rate=args.dropout)
    model.to(device)
    
    # 包装为DDP模型
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # 计算参数量（只在主进程打印）
    if rank == 0:
        model_for_counting = model.module if is_distributed else model
        total_params = sum(p.numel() for p in model_for_counting.parameters())
        trainable_params = sum(p.numel() for p in model_for_counting.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数选择
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=args.label_smoothing)
    
    val_criterion = nn.CrossEntropyLoss()  # 验证时用标准CE
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                  weight_decay=args.weight_decay)
    
    # 学习率调度 - 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 学习率预热（小数据集重要）
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    
    # 训练循环
    best_ece = float('inf')
    history = []
    
    for epoch in range(args.epochs):
        # 设置采样器的epoch（用于shuffle）
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 训练
        train_loss, train_acc, ce_loss, contrast_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_ece, feature_std = evaluate(
            model, val_loader, val_criterion, device
        )
        
        # 更新学习率
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_ece': val_ece
        })
        
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, Contrast: {contrast_loss:.4f})")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val ECE: {val_ece:.4f}, Feature Std: {feature_std:.4f}")
        
        # 保存最佳模型（只在主进程）
        if rank == 0:
            score = val_acc - val_ece
            if val_ece < best_ece and val_acc > 0.85:
                best_ece = val_ece
                
                # 保存模型时要处理DDP wrapper
                model_to_save = model.module if is_distributed else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': val_acc,
                    'val_ece': val_ece,
                    'num_classes': num_classes,
                    'class_names': train_dataset.classes,
                    'args': vars(args)
                }, Path(args.output_dir) / 'best_calibrated_model.pth')
                print(f"Best model saved with ECE: {val_ece:.4f}")
    
    # 保存训练历史（只在主进程）
    if rank == 0:
        with open(Path(args.output_dir) / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n训练完成！最佳ECE: {best_ece:.4f}")
        print(f"模型保存在: {args.output_dir}")
    
    # 清理DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
