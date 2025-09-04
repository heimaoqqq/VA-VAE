"""
Regularization utilities for DiT training on small datasets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothing(nn.Module):
    """
    Label smoothing for conditional generation
    Helps prevent overfitting on small datasets with subtle class differences
    """
    def __init__(self, smoothing=0.1, num_classes=31):
        """
        Args:
            smoothing: Label smoothing factor (0.1 means 10% smoothing)
            num_classes: Number of classes (31 users for micro-Doppler)
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, labels):
        """
        Apply label smoothing to one-hot or integer labels
        
        Args:
            labels: Integer labels [B] or one-hot labels [B, num_classes]
            
        Returns:
            Smoothed labels [B, num_classes]
        """
        if labels.dim() == 1:
            # Convert integer labels to one-hot
            batch_size = labels.size(0)
            one_hot = torch.zeros(batch_size, self.num_classes, device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            labels = one_hot
        
        # Apply smoothing
        smooth_labels = labels * self.confidence + self.smoothing / self.num_classes
        return smooth_labels


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Mixup数据增强
    对latent空间进行mixup，生成插值样本
    
    Args:
        x: Input latents [B, C, H, W]
        y: Labels [B] or [B, num_classes]
        alpha: Beta分布参数（控制混合程度）
        device: Device
        
    Returns:
        mixed_x: Mixed latents
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix数据增强（适用于latent space）
    随机裁剪并混合两个样本的latent特征
    
    Args:
        x: Input latents [B, C, H, W]
        y: Labels [B]
        alpha: Beta分布参数
        device: Device
        
    Returns:
        mixed_x: Mixed latents
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    W = x.size(3)
    H = x.size(2)
    
    # 生成随机box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机选择中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 应用CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 调整lambda以反映实际的混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class DropoutScheduler:
    """
    动态调整Dropout率的调度器
    训练初期使用较低dropout，后期逐渐增加
    """
    def __init__(self, model, initial_dropout=0.0, final_dropout=0.1, warmup_epochs=10, total_epochs=200):
        self.model = model
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
    def step(self, epoch):
        """更新dropout率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段不使用dropout
            dropout_rate = self.initial_dropout
        else:
            # 线性增加dropout
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            dropout_rate = self.initial_dropout + (self.final_dropout - self.initial_dropout) * progress
            
        # 更新模型中所有Dropout层
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
                
        return dropout_rate


def add_noise_to_labels(labels, noise_prob=0.1, num_classes=31):
    """
    向标签添加噪声（Label noise injection）
    随机翻转一小部分标签以增强鲁棒性
    
    Args:
        labels: Original labels [B]
        noise_prob: Probability of flipping each label
        num_classes: Number of classes
        
    Returns:
        Noisy labels
    """
    if noise_prob <= 0:
        return labels
        
    batch_size = labels.size(0)
    mask = torch.rand(batch_size, device=labels.device) < noise_prob
    
    # 生成随机标签
    random_labels = torch.randint(0, num_classes, (batch_size,), device=labels.device)
    
    # 应用噪声
    noisy_labels = labels.clone()
    noisy_labels[mask] = random_labels[mask]
    
    return noisy_labels


class ContrastiveRegularizer(nn.Module):
    """
    对比学习正则化器
    确保相同类别的latent相似，不同类别的latent不同
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        计算对比损失
        
        Args:
            features: Latent features [B, C, H, W]
            labels: Class labels [B]
            
        Returns:
            Contrastive loss
        """
        # 展平特征
        features = features.flatten(start_dim=1)
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        
        # 排除对角线（自身相似度）
        mask.fill_diagonal_(0)
        
        # 计算损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只对正样本对计算损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        return loss


def orthogonal_regularization(model, reg_weight=1e-4):
    """
    正交正则化：鼓励权重矩阵的正交性
    有助于防止梯度消失和特征冗余
    
    Args:
        model: DiT model
        reg_weight: Regularization weight
        
    Returns:
        Orthogonal regularization loss
    """
    orth_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # 只对权重矩阵应用
            rows = param.size(0)
            cols = param.size(1)
            
            if rows < cols:
                # 计算W @ W^T应该接近单位矩阵
                gram = torch.matmul(param, param.T)
                target = torch.eye(rows, device=param.device)
            else:
                # 计算W^T @ W应该接近单位矩阵
                gram = torch.matmul(param.T, param)
                target = torch.eye(cols, device=param.device)
                
            orth_loss += ((gram - target) ** 2).sum()
            
    return reg_weight * orth_loss


class EarlyStopping:
    """
    早停机制，防止过拟合
    """
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
