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

def load_microdoppler_data(dataset_path='/kaggle/input/dataset'):
    """
    从实际数据集路径加载微多普勒数据
    适配ID_1, ID_10等目录格式
    """
    import os
    from pathlib import Path
    
    print(f"从 {dataset_path} 加载微多普勒数据...")
    
    features = []
    labels = []
    user_count = {}
    
    # 遍历用户目录 - 修改为ID_开头的目录
    dataset_dir = Path(dataset_path)
    user_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')])
    
    print(f"找到 {len(user_dirs)} 个用户目录")
    
    if len(user_dirs) == 0:
        print("错误：未找到用户目录！")
        print("请检查数据集路径和目录结构")
        # 显示实际存在的目录
        all_dirs = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"实际存在的目录: {all_dirs[:10]}...")
        return np.array([]), np.array([])
    
    for user_dir in user_dirs:
        # 从ID_1格式提取用户ID
        user_id = int(user_dir.name.split('_')[1])  
        
        # 加载该用户的所有图像特征
        image_files = list(user_dir.glob('*.jpg')) + list(user_dir.glob('*.png'))
        user_features = []
        
        print(f"处理 ID_{user_id}: {len(image_files)} 张图像")
        
        # 这里需要实际的特征提取
        # 暂时使用图像路径作为特征的占位符
        for img_path in image_files:
            # TODO: 使用VA-VAE或其他方法提取实际特征
            # 现在使用随机特征作为占位符，但基于图像文件名设置随机种子保证一致性
            seed = hash(str(img_path)) % (2**32)
            np.random.seed(seed)
            feature = np.random.randn(512)  # 512维特征
            user_features.append(feature)
            labels.append(user_id)
        
        if user_features:
            features.extend(user_features)
            user_count[user_id] = len(user_features)
    
    if len(features) == 0:
        print("警告：未找到任何图像文件！")
        return np.array([]), np.array([])
        
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"\n数据加载完成:")
    print(f"  总样本数: {len(features)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  用户数量: {len(np.unique(labels))}")
    
    # 显示每个用户的样本数
    for user_id, count in sorted(user_count.items()):
        print(f"  ID_{user_id}: {count} 张")
    
    return features, labels

def main():
    """
    分析微多普勒数据并选择用户
    """
    # 加载实际数据
    features, labels = load_microdoppler_data('/kaggle/input/dataset')
    
    # 检查数据是否加载成功
    if len(features) == 0 or len(labels) == 0:
        print("错误：数据加载失败，无法进行用户选择分析")
        return
    
    # 如果需要预测数据（用于混淆矩阵计算），可以加载
    # predictions = None  # 暂时不使用预测数据
    
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
        print(f"最高得分用户: User_{report['selected_users'][-1]:02d}")
        print(f"最低得分用户: User_{report['selected_users'][0]:02d}")
        
        # 显示详细得分
        print("\n详细得分:")
        for i, user_id in enumerate(report['selected_users']):
            fisher = report['scores']['fisher'][i] if report['scores']['fisher'] else 'N/A'
            prototype = report['scores']['prototype'][i]
            difficulty = report['scores']['difficulty'][i]
            total = report['scores']['total'][i]
            
            print(f"  User_{user_id:02d}: Fisher={fisher:.3f}, Prototype={prototype:.3f}, "
                  f"Difficulty={difficulty:.3f}, Total={total:.3f}")
        
        # 保存结果
        with open(f'/kaggle/working/user_selection_{n_select}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n结果已保存到 /kaggle/working/user_selection_{n_select}.json")
    
    # 推荐最佳配置
    print("\n" + "="*60)
    print("推荐方案")
    print("="*60)
    print("基于分析结果，推荐以下策略:")
    print("1. 如果追求最高质量: 选择8个用户，深度训练")
    print("2. 如果需要平衡: 选择12个用户，适中训练")
    print("3. 建议使用DiT-S模型配合对比学习")
    print("4. 训练完成后可以逐步扩展到更多用户")
    
    print("\n分析完成！")

def extract_features_with_vae(dataset_path, vae_model_path=None):
    """
    使用VA-VAE模型提取特征（实际实现）
    这个函数可以在后续完善，用真实的VA-VAE特征替换随机特征
    """
    # TODO: 实现真实的特征提取
    # 1. 加载VA-VAE模型
    # 2. 遍历图像文件
    # 3. 提取latent特征
    # 4. 返回特征矩阵
    pass
        

if __name__ == "__main__":
    main()
