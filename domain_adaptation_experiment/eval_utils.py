#!/usr/bin/env python3
"""
评估工具函数
严格避免数据泄露，复用已验证的SplitTargetDomainDataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset


def load_classifier(model_path, device='cuda'):
    """
    加载ImprovedClassifier
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    print(f"📦 Loading classifier from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取模型配置
    num_classes = checkpoint.get('num_classes', 31)
    backbone = checkpoint.get('backbone', 'resnet18')
    
    print(f"   Model config: {num_classes} classes, backbone={backbone}")
    
    # 创建模型
    model = ImprovedClassifier(
        num_classes=num_classes,
        backbone=backbone
    ).to(device)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Classifier loaded successfully")
    
    return model


def create_data_loaders(data_dir, support_size, seed, batch_size=64, num_workers=4):
    """
    创建数据加载器（严格分离support和test）
    
    Args:
        data_dir: 数据目录（背包步态）
        support_size: 每个用户的support样本数
        seed: 随机种子
        batch_size: 批大小
        num_workers: 工作线程数
    
    Returns:
        support_loader: 支持集加载器
        test_loader: 测试集加载器
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n📂 Loading data from: {data_dir}")
    print(f"   Support size: {support_size}/user, Seed: {seed}")
    
    # 创建支持集（前support_size张）
    support_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='support',
        seed=seed
    )
    
    # 创建测试集（后面的所有样本）
    test_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='test',
        seed=seed
    )
    
    support_loader = DataLoader(
        support_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Support: {len(support_dataset)} samples, Test: {len(test_dataset)} samples")
    
    return support_loader, test_loader


def extract_features(model, data_loader, device='cuda'):
    """
    提取特征
    
    Args:
        model: 分类器
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        features: 特征 [N, D]
        labels: 标签 [N]
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            features = model.backbone(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


def build_prototypes_simple_mean(features, labels, num_classes=31):
    """
    构建原型：简单均值
    
    Args:
        features: 特征 [N, D]
        labels: 标签 [N]
        num_classes: 类别数
    
    Returns:
        prototypes: 原型 [num_classes, D]
    """
    prototypes = []
    
    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_features = features[class_mask]
        
        if len(class_features) > 0:
            prototype = class_features.mean(dim=0)
            prototype = F.normalize(prototype, dim=0)
        else:
            # 如果没有样本，使用零向量
            prototype = torch.zeros(features.shape[1])
        
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    return prototypes


def build_prototypes_weighted(model, data_loader, device='cuda', num_classes=31):
    """
    构建原型：加权均值（基于分类器置信度）
    
    Args:
        model: 分类器
        data_loader: 数据加载器
        device: 设备
        num_classes: 类别数
    
    Returns:
        prototypes: 原型 [num_classes, D]
    """
    model.eval()
    class_features = {i: [] for i in range(num_classes)}
    class_weights = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取特征
            features = model.backbone(images)
            
            # 计算置信度
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1)[0]
            
            # 按类别收集
            for feat, label, conf in zip(features, labels, confidences):
                label_id = label.item()
                class_features[label_id].append(feat.cpu())
                class_weights[label_id].append(conf.cpu().item())
    
    # 计算加权原型
    prototypes = []
    for class_id in range(num_classes):
        if len(class_features[class_id]) > 0:
            feats = torch.stack(class_features[class_id])
            weights = torch.tensor(class_weights[class_id]).unsqueeze(1)
            
            # 归一化权重
            weights = weights / weights.sum()
            
            # 加权平均
            prototype = (feats * weights).sum(dim=0)
            prototype = F.normalize(prototype, dim=0)
        else:
            prototype = torch.zeros(512)  # ResNet18特征维度
        
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    return prototypes


def build_prototypes_diversity(features, labels, num_classes=31, num_select=None):
    """
    构建原型：多样性选择（K-means）
    
    Args:
        features: 特征 [N, D]
        labels: 标签 [N]
        num_classes: 类别数
        num_select: 每个类选择的样本数（None表示全部）
    
    Returns:
        prototypes: 原型 [num_classes, D]
    """
    prototypes = []
    
    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_features = features[class_mask].numpy()
        
        if len(class_features) == 0:
            prototypes.append(torch.zeros(features.shape[1]))
            continue
        
        if num_select is None or len(class_features) <= num_select:
            # 使用全部样本
            selected_features = class_features
        else:
            # K-means选择多样化样本
            kmeans = KMeans(n_clusters=num_select, random_state=42, n_init=10)
            kmeans.fit(class_features)
            
            # 选择离聚类中心最近的样本
            selected_indices = []
            for i in range(num_select):
                center = kmeans.cluster_centers_[i]
                distances = pairwise_distances([center], class_features)[0]
                closest_idx = np.argmin(distances)
                selected_indices.append(closest_idx)
            
            selected_features = class_features[selected_indices]
        
        # 计算均值原型
        prototype = torch.tensor(selected_features).mean(dim=0)
        prototype = F.normalize(prototype, dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    return prototypes


def build_prototypes_uncertainty(model, data_loader, device='cuda', num_classes=31):
    """
    构建原型：基于不确定性选择
    
    选择分类器不确定性最高的样本构建原型
    理由：高不确定性样本通常在决策边界附近，更能代表类别边界
    
    Args:
        model: 分类器
        data_loader: 数据加载器
        device: 设备
        num_classes: 类别数
    
    Returns:
        prototypes: 原型 [num_classes, D]
    """
    model.eval()
    class_samples = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取特征
            features = model.backbone(images)
            
            # 计算不确定性（熵）
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            
            # 按类别收集
            for feat, label, unc in zip(features, labels, entropy):
                label_id = label.item()
                class_samples[label_id].append({
                    'feature': feat.cpu(),
                    'uncertainty': unc.cpu().item()
                })
    
    # 选择高不确定性样本构建原型
    prototypes = []
    for class_id in range(num_classes):
        samples = class_samples[class_id]
        
        if len(samples) == 0:
            prototypes.append(torch.zeros(512))
            continue
        
        # 按不确定性排序
        samples.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        # 选择top-K（或全部）
        top_k = min(len(samples), max(1, len(samples) // 2))  # 选择一半
        selected_features = [s['feature'] for s in samples[:top_k]]
        
        # 计算均值原型
        prototype = torch.stack(selected_features).mean(dim=0)
        prototype = F.normalize(prototype, dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    return prototypes


def evaluate_baseline(model, test_loader, device='cuda'):
    """
    评估基线（直接分类）
    
    Args:
        model: 分类器
        test_loader: 测试集加载器
        device: 设备
    
    Returns:
        accuracy: 准确率
        confidence: 平均置信度
    """
    model.eval()
    correct = 0
    total = 0
    all_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = correct / total
    mean_confidence = np.mean(all_confidences)
    
    return accuracy, mean_confidence 
