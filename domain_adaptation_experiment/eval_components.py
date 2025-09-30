#!/usr/bin/env python3
"""
域适应评估组件
包含PNC、LCCS、NCC等方法的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PNCEvaluator:
    """
    PNC (Prototype Network Calibration) 评估器
    融合分类器预测和原型匹配
    """
    
    def __init__(self, model, prototypes, device='cuda'):
        """
        Args:
            model: 分类器
            prototypes: 原型 [num_classes, D]
            device: 设备
        """
        self.model = model
        self.prototypes = prototypes.to(device)
        self.device = device
    
    def predict(self, test_loader, fusion_alpha=0.6, similarity_tau=0.01, 
               use_adaptive=False):
        """
        PNC预测
        
        Args:
            test_loader: 测试集加载器
            fusion_alpha: 融合权重（原型权重）
            similarity_tau: 温度参数
            use_adaptive: 是否使用自适应融合
        
        Returns:
            accuracy: 准确率
            confidence: 平均置信度
        """
        self.model.eval()
        correct = 0
        total = 0
        all_confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.model.backbone(images)
                features_norm = F.normalize(features, dim=1)
                
                # 分类器预测
                outputs = self.model(images)
                class_probs = F.softmax(outputs, dim=1)
                
                # 原型预测
                # 计算与每个原型的相似度
                prototypes_norm = F.normalize(self.prototypes, dim=1)
                similarities = torch.matmul(features_norm, prototypes_norm.T)
                proto_logits = similarities / similarity_tau
                proto_probs = F.softmax(proto_logits, dim=1)
                
                # 融合
                if use_adaptive:
                    # 自适应融合（基于置信度）
                    class_conf = class_probs.max(dim=1)[0]
                    proto_conf = proto_probs.max(dim=1)[0]
                    
                    # 归一化置信度作为权重
                    total_conf = class_conf + proto_conf + 1e-8
                    alpha_adaptive = proto_conf / total_conf
                    alpha_adaptive = alpha_adaptive.unsqueeze(1)
                    
                    final_probs = (alpha_adaptive * proto_probs + 
                                  (1 - alpha_adaptive) * class_probs)
                else:
                    # 固定权重融合
                    final_probs = (fusion_alpha * proto_probs + 
                                  (1 - fusion_alpha) * class_probs)
                
                confidences, predictions = final_probs.max(dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_confidences.extend(confidences.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(all_confidences)
        
        return accuracy, mean_confidence


class LCCSAdapter:
    """
    LCCS (Label-Consistent Calibration Strategy) 适配器
    适应BatchNorm统计量到目标域
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 分类器
            device: 设备
        """
        self.model = model
        self.device = device
        self.original_bn_stats = self._save_bn_stats()
    
    def _save_bn_stats(self):
        """保存原始BN统计量"""
        bn_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'momentum': module.momentum,
                    'num_batches_tracked': module.num_batches_tracked.clone() 
                        if module.num_batches_tracked is not None else None
                }
        return bn_stats
    
    def restore_bn_stats(self):
        """恢复原始BN统计量"""
        for name, module in self.model.named_modules():
            if name in self.original_bn_stats:
                module.running_mean.data = self.original_bn_stats[name]['running_mean']
                module.running_var.data = self.original_bn_stats[name]['running_var']
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.data = self.original_bn_stats[name]['num_batches_tracked']
    
    def adapt_progressive(self, support_loader, momentum=0.01, iterations=5):
        """
        渐进式LCCS适应
        
        Args:
            support_loader: 支持集加载器
            momentum: 动量参数
            iterations: 迭代次数
        """
        self.model.train()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 设置小momentum
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.momentum = momentum
        
        # 多次迭代更新BN统计量
        with torch.no_grad():
            for i in range(iterations):
                for batch in support_loader:
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    
                    images = images.to(self.device)
                    _ = self.model(images)
        
        self.model.eval()
    
    def adapt_weighted(self, support_loader, alpha=0.3):
        """
        加权融合LCCS适应
        
        Args:
            support_loader: 支持集加载器
            alpha: 目标域权重
        """
        # 保存源域统计量
        source_stats = self._save_bn_stats()
        
        # 临时重置BN统计量以收集目标域统计
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
                module.momentum = 1.0  # 快速收集
        
        # 收集目标域统计量
        with torch.no_grad():
            for _ in range(10):  # 多次迭代稳定
                for batch in support_loader:
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    
                    images = images.to(self.device)
                    _ = self.model(images)
        
        # 保存目标域统计量
        target_stats = self._save_bn_stats()
        
        # 加权融合
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in source_stats and name in target_stats:
                    module.running_mean = ((1 - alpha) * source_stats[name]['running_mean'] + 
                                          alpha * target_stats[name]['running_mean'])
                    module.running_var = ((1 - alpha) * source_stats[name]['running_var'] + 
                                         alpha * target_stats[name]['running_var'])
        
        self.model.eval()


class NCCEvaluator:
    """
    NCC (Nearest Centroid Classifier) 评估器
    使用最近质心分类
    """
    
    def __init__(self, prototypes, device='cuda'):
        """
        Args:
            prototypes: 原型 [num_classes, D]
            device: 设备
        """
        self.prototypes = prototypes.to(device)
        self.device = device
    
    def predict(self, model, test_loader, temperature=0.01, 
                distance_metric='cosine'):
        """
        NCC预测
        
        Args:
            model: 分类器（用于提取特征）
            test_loader: 测试集加载器
            temperature: 温度参数
            distance_metric: 距离度量（'cosine' or 'euclidean'）
        
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
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = model.backbone(images)
                
                if distance_metric == 'cosine':
                    # 余弦相似度
                    features_norm = F.normalize(features, dim=1)
                    prototypes_norm = F.normalize(self.prototypes, dim=1)
                    similarities = torch.matmul(features_norm, prototypes_norm.T)
                    logits = similarities / temperature
                elif distance_metric == 'euclidean':
                    # 欧氏距离
                    distances = torch.cdist(features, self.prototypes)
                    logits = -distances / temperature
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")
                
                probs = F.softmax(logits, dim=1)
                confidences, predictions = probs.max(dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_confidences.extend(confidences.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(all_confidences)
        
        return accuracy, mean_confidence


def evaluate_pnc_only(model, support_loader, test_loader, 
                     prototype_strategy, fusion_alpha, similarity_tau,
                     use_adaptive, device='cuda'):
    """
    评估PNC单独效果
    
    Returns:
        dict: 包含accuracy和confidence的字典
    """
    # 构建原型
    from eval_utils import (extract_features, build_prototypes_simple_mean,
                           build_prototypes_weighted, build_prototypes_diversity,
                           build_prototypes_uncertainty)
    
    if prototype_strategy == 'simple_mean':
        features, labels = extract_features(model, support_loader, device)
        prototypes = build_prototypes_simple_mean(features, labels)
    elif prototype_strategy == 'weighted_mean':
        prototypes = build_prototypes_weighted(model, support_loader, device)
    elif prototype_strategy == 'diversity':
        features, labels = extract_features(model, support_loader, device)
        prototypes = build_prototypes_diversity(features, labels)
    elif prototype_strategy == 'uncertainty':
        prototypes = build_prototypes_uncertainty(model, support_loader, device)
    else:
        raise ValueError(f"Unknown prototype strategy: {prototype_strategy}")
    
    # PNC评估
    pnc = PNCEvaluator(model, prototypes, device)
    acc, conf = pnc.predict(test_loader, fusion_alpha, similarity_tau, use_adaptive)
    
    return {'accuracy': acc, 'confidence': conf}


def evaluate_lccs_only(model, support_loader, test_loader,
                      lccs_method, lccs_params, device='cuda'):
    """
    评估LCCS单独效果
    
    Returns:
        dict: 包含accuracy和confidence的字典
    """
    from eval_utils import evaluate_baseline
    
    # LCCS适应
    lccs = LCCSAdapter(model, device)
    
    if lccs_method == 'progressive':
        lccs.adapt_progressive(
            support_loader,
            momentum=lccs_params.get('momentum', 0.01),
            iterations=lccs_params.get('iterations', 5)
        )
    elif lccs_method == 'weighted':
        lccs.adapt_weighted(
            support_loader,
            alpha=lccs_params.get('alpha', 0.3)
        )
    else:
        raise ValueError(f"Unknown LCCS method: {lccs_method}")
    
    # 评估
    acc, conf = evaluate_baseline(model, test_loader, device)
    
    # 恢复原始BN统计量
    lccs.restore_bn_stats()
    
    return {'accuracy': acc, 'confidence': conf}


def evaluate_pnc_lccs_combined(model, support_loader, test_loader,
                               prototype_strategy, fusion_alpha, similarity_tau,
                               use_adaptive, lccs_method, lccs_params,
                               device='cuda'):
    """
    评估PNC+LCCS组合效果
    
    Returns:
        dict: 包含accuracy和confidence的字典
    """
    from eval_utils import (extract_features, build_prototypes_simple_mean,
                           build_prototypes_weighted, build_prototypes_diversity,
                           build_prototypes_uncertainty)
    
    # 步骤1：LCCS适应
    lccs = LCCSAdapter(model, device)
    
    if lccs_method == 'progressive':
        lccs.adapt_progressive(
            support_loader,
            momentum=lccs_params.get('momentum', 0.01),
            iterations=lccs_params.get('iterations', 5)
        )
    elif lccs_method == 'weighted':
        lccs.adapt_weighted(
            support_loader,
            alpha=lccs_params.get('alpha', 0.3)
        )
    
    # 步骤2：在LCCS适应后的模型上构建原型
    if prototype_strategy == 'simple_mean':
        features, labels = extract_features(model, support_loader, device)
        prototypes = build_prototypes_simple_mean(features, labels)
    elif prototype_strategy == 'weighted_mean':
        prototypes = build_prototypes_weighted(model, support_loader, device)
    elif prototype_strategy == 'diversity':
        features, labels = extract_features(model, support_loader, device)
        prototypes = build_prototypes_diversity(features, labels)
    elif prototype_strategy == 'uncertainty':
        prototypes = build_prototypes_uncertainty(model, support_loader, device)
    
    # 步骤3：PNC评估
    pnc = PNCEvaluator(model, prototypes, device)
    acc, conf = pnc.predict(test_loader, fusion_alpha, similarity_tau, use_adaptive)
    
    # 恢复原始BN统计量
    lccs.restore_bn_stats()
    
    return {'accuracy': acc, 'confidence': conf} 
