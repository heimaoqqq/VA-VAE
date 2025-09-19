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
        try:
            # 添加超时保护
            import time
            start_time = time.time()
            dist.destroy_process_group()
            print(f"✅ 分布式清理完成，耗时 {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"⚠️ 分布式清理失败: {e}")


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
        sim_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # 创建标签mask
        labels_expanded = labels.view(-1, 1)
        pos_mask = torch.eq(labels_expanded, labels_expanded.T).float()
        
        # 移除对角线（自己和自己的相似度）
        eye_mask = torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask * (1.0 - eye_mask)
        neg_mask = (1.0 - torch.eq(labels_expanded, labels_expanded.T).float()) * (1.0 - eye_mask)
        
        # 数值稳定性：减去最大值
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max.detach()
        
        # 计算InfoNCE风格的对比损失
        exp_sim = torch.exp(sim_matrix_stable)
        
        # 计算每个anchor的损失
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        neg_sum = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True)
        
        # 避免除零
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        total_sum = pos_sum + neg_sum + 1e-8
        
        # InfoNCE损失：-log(pos_sum / total_sum)
        loss_per_sample = -torch.log(pos_sum / total_sum)
        
        # 只对有正样本的anchor计算损失
        has_pos = (torch.sum(pos_mask, dim=1) > 0).float()
        valid_loss = loss_per_sample.squeeze() * has_pos
        
        if has_pos.sum() > 0:
            return valid_loss.sum() / has_pos.sum()
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
    

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
    
    def __init__(self, data_dir, split='train', transform=None, contrastive_pairs=False, 
                 generated_data_dirs=None, use_generated=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.contrastive_pairs = contrastive_pairs
        self.generated_data_dirs = generated_data_dirs or []
        self.use_generated = use_generated
        self.samples = []
        
        # 加载数据
        self._load_data()
        
        # 如果是训练集且使用生成数据，则加载合成数据
        if self.split == 'train' and self.use_generated:
            self._load_generated_data()
        
        # 数据划分处理
        self._split_data()
        
        # 微多普勒图像专用变换（最小增强，保持频谱结构）
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                # 只使用极轻微的噪声增强，不破坏频谱结构
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 可选：极小的高斯噪声（模拟测量噪声） - 避免原地操作
                # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01 if torch.rand(1).item() < 0.3 else x)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if transform:
            self.transform = transform
    
    def _load_data(self):
        """加载真实数据集"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        user_samples = defaultdict(list)
        self._load_data_from_dir(self.data_dir, user_samples, 'real')
        
        # 统计信息
        total_samples = sum(len(samples) for samples in user_samples.values())
        print(f"真实数据: {len(user_samples)} 个用户的 {total_samples} 张图像")
    
    def _load_data_from_dir(self, data_dir, user_samples, data_type='real'):
        """从指定目录加载数据"""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"数据目录不存在: {data_dir}")
            return
            
        # 查找用户目录：支持ID_*, User_*, user_*格式
        id_dirs = []
        for pattern in ["ID_*", "User_*", "user_*"]:
            id_dirs.extend(data_dir.glob(pattern))
        
        if len(id_dirs) == 0:
            print(f"Warning: 在 {data_dir} 中未找到ID_*、User_*或user_*格式的用户目录")
            # 尝试数字格式目录
            id_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            id_dirs = sorted(id_dirs, key=lambda x: int(x.name))
        
        for user_dir in sorted(id_dirs):
            user_folder_name = user_dir.name
            
            # 解析用户ID
            if user_folder_name.startswith("ID_"):
                user_id = int(user_folder_name.split('_')[1]) - 1  # ID_1->0, ID_2->1, ..., ID_31->30
            elif user_folder_name.startswith(("User_", "user_")):
                user_id = int(user_folder_name.split('_')[1])  # User_00->0, User_01->1, ..., User_30->30
            else:
                # 尝试直接解析数字
                try:
                    user_id = int(user_folder_name)
                except ValueError:
                    print(f"无法解析用户ID: {user_folder_name}")
                    continue
            
            # 验证映射正确性
            if data_type == 'generated' and user_folder_name.startswith("User_"):
                expected_real_id = f"ID_{user_id + 1}"  # User_00 -> ID_1
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"   映射: {user_folder_name} -> {expected_real_id} (user_id={user_id})")
            
            # 获取所有图像文件
            image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
            
            current_data_samples = []
            for img_path in image_files:
                path_info = (str(img_path), data_type)
                current_data_samples.append(path_info)
                user_samples[user_id].append(path_info)
            
            # 添加到主样本列表
            for path_info in current_data_samples:
                path, dtype = path_info
                self.samples.append((path, user_id, dtype))
    
    def _load_generated_data(self):
        """加载合成数据"""
        if not self.generated_data_dirs:
            return
        
        total_generated = 0
        user_samples = defaultdict(list)
        
        for gen_dir in self.generated_data_dirs:
            gen_dir = Path(gen_dir)
            if gen_dir.exists():
                print(f"加载合成数据: {gen_dir}")
                self._load_data_from_dir(gen_dir, user_samples, 'generated')
                
        # 统计合成数据
        for user_id, samples in user_samples.items():
            generated_count = sum(1 for _, dtype in samples if dtype == 'generated')
            total_generated += generated_count
            
        print(f" 合成数据: {len(user_samples)} 个用户的 {total_generated} 张图像")
    
    def _split_data(self):
        """根据split参数划分训练集和验证集"""
        if not hasattr(self, '_all_samples_collected'):
            # 收集所有已加载的样本，按用户分组
            user_samples = defaultdict(list)
            for sample in self.samples:
                if len(sample) == 3:
                    path, user_id, data_type = sample
                else:
                    path, user_id = sample
                    data_type = 'real'
                user_samples[user_id].append((path, data_type))
            
            # 清空当前样本列表，准备重新分配
            self.samples = []
            
            # 按用户划分训练集和验证集（7:3比例）
            random.seed(42)  # 确保可重复
            for user_id, samples in user_samples.items():
                # 分离真实数据和合成数据
                real_samples = [(p, dt) for p, dt in samples if dt == 'real']
                generated_samples = [(p, dt) for p, dt in samples if dt == 'generated']
                
                # 只对真实数据进行训练/验证划分
                if real_samples:
                    random.shuffle(real_samples)
                    split_idx = int(len(real_samples) * 0.7)
                    
                    if self.split == 'train':
                        # 训练集：真实训练数据 + 全部合成数据
                        selected_samples = real_samples[:split_idx] + generated_samples
                    else:  # validation
                        # 验证集：只用真实验证数据
                        selected_samples = real_samples[split_idx:]
                    
                    for path, data_type in selected_samples:
                        self.samples.append((path, user_id, data_type))
            
            self._all_samples_collected = True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if len(self.samples[idx]) == 3:
            img_path, label, data_type = self.samples[idx]
        else:
            img_path, label = self.samples[idx]
            data_type = 'real'
        
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
            # 返回零张量 - 尺寸与实际图像一致
            if self.contrastive_pairs:
                return (torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)), label
            else:
                return torch.zeros(3, 256, 256), label


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


def train_with_contrastive_learning(model, train_loader, val_loader, device, args, rank=0):
    """改进的训练函数，集成对比学习"""
    
    # 分布式训练设置
    def is_main_process():
        return rank == 0
    
    # 验证数据集是否为空
    if len(train_loader.dataset) == 0:
        if is_main_process():
            print("训练数据集为空，无法开始训练")
        return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
    
    if len(val_loader.dataset) == 0:
        if is_main_process():
            print("验证数据集为空，无法开始训练")
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
    
    # 重新启用对比学习 - 根据类型选择损失函数
    if args.use_contrastive:
        if args.contrastive_type == 'interuser':
            if is_main_process():
                print("启用InterUserContrastiveLoss对比学习 - 优化用户间差异")
            contrastive_criterion = InterUserContrastiveLoss(
                temperature=args.contrastive_temperature,
                margin=args.contrastive_margin
            )
        elif args.contrastive_type == 'supcon':
            if is_main_process():
                print("启用SupConLoss对比学习 - 监督对比学习")
            contrastive_criterion = SupConLoss(temperature=args.contrastive_temperature)
        elif args.contrastive_type == 'global':
            if is_main_process():
                print("启用GlobalNegativeContrastiveLoss - 全局负样本对比")
            contrastive_criterion = GlobalNegativeContrastiveLoss(
                memory_size=64,
                temperature=args.contrastive_temperature
            )
        else:
            if is_main_process():
                print(f"未知对比学习类型: {args.contrastive_type}，使用SupConLoss")
            contrastive_criterion = SupConLoss(temperature=args.contrastive_temperature)
    else:
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
            
            # 对比学习训练循环
            if isinstance(data, (tuple, list)) and len(data) == 2:
                # 对比学习数据对
                data1, data2 = data[0].to(device), data[1].to(device)
                
                if args.use_contrastive and contrastive_criterion is not None:
                    # 完全合并输入，单次模型前向传播避免任何参数重复使用
                    combined_data = torch.cat([data1, data2], dim=0)
                    batch_size = data1.size(0)
                    combined_target = torch.cat([target, target], dim=0)
                    
                    # 单次完整前向传播获取特征和投影
                    combined_features, combined_proj = model(combined_data, return_features=True)
                    
                    # 单次分类器调用
                    if hasattr(model, 'module'):
                        combined_logits = model.module.classifier(combined_features)
                    else:
                        combined_logits = model.classifier(combined_features)
                    
                    # 分割结果
                    logits1 = combined_logits[:batch_size]
                    logits2 = combined_logits[batch_size:]
                    proj1 = combined_proj[:batch_size] 
                    proj2 = combined_proj[batch_size:]
                    
                    # 分类损失
                    cls_loss1 = classification_criterion(logits1, target)
                    cls_loss2 = classification_criterion(logits2, target)
                    classification_loss = (cls_loss1 + cls_loss2) / 2
                    
                    # 对比损失：使用投影特征
                    contrastive_loss = contrastive_criterion(combined_proj, combined_target)
                    
                    # 总损失
                    total_loss = classification_loss + args.contrastive_weight * contrastive_loss
                    pred = logits1.argmax(dim=1)
                else:
                    # 只使用第一张图进行分类
                    logits = model(data1)
                    total_loss = classification_criterion(logits, target)
                    classification_loss = total_loss
                    contrastive_loss = torch.tensor(0.0, device=device)
                    pred = logits.argmax(dim=1)
            else:
                # 单张图像训练
                if isinstance(data, (tuple, list)):
                    data = data[0]
                
                data = data.to(device)
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
            print("训练数据集为空，请检查数据路径和格式")
            return model, None, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'contrastive_loss': [], 'classification_loss': []}
        
        if len(val_loader) == 0:
            print("验证数据集为空，请检查数据路径和格式")
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
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
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
    
    parser.add_argument('--generated_data_dir', type=str, action='append', 
                       help='Generated dataset directory (can be specified multiple times)')
    parser.add_argument('--use_generated', action='store_true', 
                       help='Use generated data to augment training set')
    
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
    torch.manual_seed(42)  # 固定种子为42
    np.random.seed(42)
    random.seed(42)
    
    # 数据集
    train_dataset = ImprovedMicroDopplerDataset(
        data_dir=args.data_dir, 
        split='train', 
        contrastive_pairs=args.use_contrastive,
        generated_data_dirs=args.generated_data_dir,
        use_generated=args.use_generated
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
        num_workers=0,  # 避免多进程卡住
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,  # 避免多进程卡住
        pin_memory=True
    )
    
    # 模型
    model = ImprovedClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        dropout_rate=args.dropout_rate,
        freeze_layers=args.freeze_layers
    ).to(device)
    
    # 分布式训练设置 - 使用static_graph解决参数重复使用问题
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        # 设置静态图模式，允许参数在同一次反向传播中多次使用
        model._set_static_graph()
    
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
    
    # 确保程序正常退出
    if is_main_process():
        print("🎉 训练流程完全结束，程序即将退出")
    
    # 显式退出程序避免卡住
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
