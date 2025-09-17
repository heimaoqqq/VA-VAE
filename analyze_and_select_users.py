"""
基于多种科学指标筛选最适合训练的用户子集
适用于微多普勒数据集的特点：数据少、差异小
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import json

class UserSelectionAnalyzer:
    """
    用户筛选分析器，整合多种科学指标
    """
    
    def __init__(self, features, labels, predictions=None):
        """
        features: (N, D) 特征矩阵
        labels: (N,) 真实标签
        predictions: (N,) 预测标签（可选）
        """
        self.features = features
        self.labels = labels
        self.predictions = predictions
        self.n_users = len(np.unique(labels))
        
    def compute_fisher_ratio(self):
        """
        计算Fisher判别率 [J. Chem. Inf. Model. 2004]
        高值表示类别可分性好
        """
        fisher_scores = []
        unique_labels = np.unique(self.labels)
        
        for i, label_i in enumerate(unique_labels):
            # 该类别的样本
            mask_i = self.labels == label_i
            features_i = self.features[mask_i]
            mean_i = np.mean(features_i, axis=0)
            
            # 计算与其他所有类别的Fisher比
            fisher_sum = 0
            for j, label_j in enumerate(unique_labels):
                if i == j:
                    continue
                    
                mask_j = self.labels == label_j
                features_j = self.features[mask_j]
                mean_j = np.mean(features_j, axis=0)
                
                # 类间距离
                S_b = np.linalg.norm(mean_i - mean_j) ** 2
                
                # 类内散度
                var_i = np.var(features_i, axis=0).mean()
                var_j = np.var(features_j, axis=0).mean()
                S_w = (var_i + var_j) / 2
                
                # Fisher比
                fisher = S_b / (S_w + 1e-10)
                fisher_sum += fisher
            
            # 平均Fisher比
            avg_fisher = fisher_sum / (self.n_users - 1)
            fisher_scores.append(avg_fisher)
            
        return np.array(fisher_scores)
    
    def compute_confusion_metrics(self):
        """
        基于混淆矩阵的指标 [CVPR 2019]
        低值表示不容易混淆
        """
        if self.predictions is None:
            print("需要predictions来计算混淆矩阵")
            return None
            
        cm = confusion_matrix(self.labels, self.predictions)
        n = cm.shape[0]
        
        confusion_scores = []
        for i in range(n):
            # 计算用户i的混淆度
            # 被误分类到其他用户 + 其他用户被误分到i
            out_confusion = np.sum(cm[i, :]) - cm[i, i]  # i被误分出去
            in_confusion = np.sum(cm[:, i]) - cm[i, i]   # 别人被误分进来
            
            # 归一化混淆度
            total_samples = np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]
            confusion_score = (out_confusion + in_confusion) / (total_samples + 1e-10)
            confusion_scores.append(confusion_score)
            
        return np.array(confusion_scores)
    
    def compute_prototype_score(self):
        """
        原型性得分 [IEEE TPAMI 2023]
        高值表示是好的原型
        """
        prototype_scores = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            mask = self.labels == label
            class_features = self.features[mask]
            
            # 计算类中心
            center = np.mean(class_features, axis=0)
            
            # 类内紧凑度（到中心的平均距离）
            compactness = np.mean([
                np.linalg.norm(f - center) 
                for f in class_features
            ])
            
            # 与其他类中心的最小距离
            min_inter_dist = float('inf')
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_mask = self.labels == other_label
                other_center = np.mean(self.features[other_mask], axis=0)
                dist = np.linalg.norm(center - other_center)
                min_inter_dist = min(min_inter_dist, dist)
            
            # 原型性得分：类间距离/类内紧凑度
            prototype_score = min_inter_dist / (compactness + 1e-10)
            prototype_scores.append(prototype_score)
            
        return np.array(prototype_scores)
    
    def compute_generation_difficulty(self):
        """
        生成难度评估（基于数据分布）
        低值表示容易生成
        """
        difficulty_scores = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            mask = self.labels == label
            class_features = self.features[mask]
            
            # 计算方差（高方差=难生成）
            variance = np.var(class_features, axis=0).mean()
            
            # 计算样本密度（低密度=难生成）
            if len(class_features) > 1:
                distances = pdist(class_features)
                density = 1.0 / (np.mean(distances) + 1e-10)
            else:
                density = 0
            
            # 综合难度分数
            difficulty = variance / (density + 1e-10)
            difficulty_scores.append(difficulty)
            
        return np.array(difficulty_scores)
    
    def select_best_users(self, n_select=8, weights=None):
        """
        综合多个指标选择最佳用户子集
        
        weights: 各指标权重字典
        """
        if weights is None:
            weights = {
                'fisher': 0.3,
                'confusion': 0.3,
                'prototype': 0.2,
                'difficulty': 0.2
            }
        
        # 计算各项指标
        fisher_scores = self.compute_fisher_ratio()
        confusion_scores = self.compute_confusion_metrics()
        prototype_scores = self.compute_prototype_score()
        difficulty_scores = self.compute_generation_difficulty()
        
        # 标准化到0-1
        def normalize(scores):
            if scores is None:
                return np.zeros(self.n_users)
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        fisher_norm = normalize(fisher_scores)
        confusion_norm = 1 - normalize(confusion_scores)  # 反转，低混淆度好
        prototype_norm = normalize(prototype_scores)
        difficulty_norm = 1 - normalize(difficulty_scores)  # 反转，低难度好
        
        # 综合得分
        total_scores = (
            weights['fisher'] * fisher_norm +
            weights['confusion'] * confusion_norm +
            weights['prototype'] * prototype_norm +
            weights['difficulty'] * difficulty_norm
        )
        
        # 选择top-k用户
        selected_indices = np.argsort(total_scores)[-n_select:]
        
        # 输出详细报告
        report = {
            'selected_users': selected_indices.tolist(),
            'scores': {
                'fisher': fisher_scores[selected_indices].tolist(),
                'confusion': confusion_scores[selected_indices].tolist() if confusion_scores is not None else None,
                'prototype': prototype_scores[selected_indices].tolist(),
                'difficulty': difficulty_scores[selected_indices].tolist(),
                'total': total_scores[selected_indices].tolist()
            },
            'average_scores': {
                'selected_fisher': np.mean(fisher_scores[selected_indices]),
                'all_fisher': np.mean(fisher_scores),
                'improvement': np.mean(fisher_scores[selected_indices]) / np.mean(fisher_scores)
            }
        }
        
        return selected_indices, report

def main():
    """
    示例：分析微多普勒数据并选择用户
    """
    # 加载特征和标签
    # 这里需要替换为实际的特征提取
    print("加载数据...")
    
    # 假设已有特征矩阵
    # features = np.load('user_features.npy')  # (N_samples, D_features)
    # labels = np.load('user_labels.npy')      # (N_samples,)
    # predictions = np.load('predictions.npy') # (N_samples,)
    
    # 模拟数据用于演示
    np.random.seed(42)
    n_samples = 4650  # 31用户×150张
    n_features = 512  # VA-VAE特征维度
    n_users = 31
    
    # 生成模拟特征（实际使用时替换）
    features = []
    labels = []
    for user_id in range(n_users):
        # 每个用户有不同的特征中心和方差
        center = np.random.randn(n_features) * 5
        variance = 0.5 + np.random.rand() * 2  # 不同用户难度不同
        user_features = center + np.random.randn(150, n_features) * variance
        features.append(user_features)
        labels.extend([user_id] * 150)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # 创建分析器
    analyzer = UserSelectionAnalyzer(features, labels)
    
    # 尝试不同数量的用户选择
    for n_select in [8, 10, 12]:
        print(f"\n{'='*50}")
        print(f"选择 {n_select} 个用户")
        print('='*50)
        
        selected, report = analyzer.select_best_users(n_select)
        
        print(f"选中的用户ID: {report['selected_users']}")
        print(f"平均Fisher比提升: {report['average_scores']['improvement']:.2f}x")
        print(f"最高得分用户: User_{report['selected_users'][-1]}")
        print(f"最低得分用户: User_{report['selected_users'][0]}")
        
        # 保存结果
        with open(f'user_selection_{n_select}.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    print("\n分析完成！结果已保存。")

if __name__ == "__main__":
    main()
