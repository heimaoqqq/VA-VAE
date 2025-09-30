#!/usr/bin/env python3
"""
智能样本选择器
实现多种支持集选择策略，严格避免数据泄露
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import torchvision.transforms as transforms


class SmartSampleSelector:
    """
    智能样本选择器
    
    支持策略：
    - random: 随机选择
    - confidence: 基于分类器置信度（选择高置信度样本）
    - diversity: 基于K-means多样性选择
    - uncertainty: 基于不确定性选择（选择高熵样本）
    - hybrid: 混合策略（一半confidence + 一半diversity）
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 分类器模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def select_samples(self, user_images, user_id, support_size, 
                      strategy='random', seed=42):
        """
        为单个用户选择支持集样本
        
        Args:
            user_images: 用户的所有图像路径列表
            user_id: 用户ID
            support_size: 支持集大小
            strategy: 选择策略
            seed: 随机种子
        
        Returns:
            support_indices: 被选中的样本索引
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if len(user_images) <= support_size:
            # 如果样本数不足，全部使用
            return list(range(len(user_images)))
        
        # 根据策略选择
        if strategy == 'random':
            return self._select_random(user_images, support_size)
        elif strategy == 'confidence':
            return self._select_by_confidence(user_images, user_id, support_size)
        elif strategy == 'diversity':
            return self._select_by_diversity(user_images, user_id, support_size)
        elif strategy == 'uncertainty':
            return self._select_by_uncertainty(user_images, user_id, support_size)
        elif strategy == 'hybrid':
            return self._select_hybrid(user_images, user_id, support_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_random(self, user_images, support_size):
        """随机选择"""
        indices = list(range(len(user_images)))
        random.shuffle(indices)
        return sorted(indices[:support_size])
    
    def _select_by_confidence(self, user_images, user_id, support_size):
        """基于分类器置信度选择（选择高置信度样本）"""
        self.model.eval()
        
        confidences = []
        with torch.no_grad():
            for img_path in user_images:
                # 加载图像
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                
                # 获取预测置信度
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                conf = probs[0, user_id].item()  # 对正确类别的置信度
                confidences.append(conf)
        
        # 选择top-K高置信度样本
        indices = np.argsort(confidences)[::-1][:support_size]
        return sorted(indices.tolist())
    
    def _select_by_diversity(self, user_images, user_id, support_size):
        """基于K-means多样性选择"""
        self.model.eval()
        
        # 提取所有样本的特征
        features = []
        with torch.no_grad():
            for img_path in user_images:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.model.backbone(image)
                features.append(feature.cpu().numpy().flatten())
        
        features = np.array(features)
        
        # K-means聚类
        n_clusters = min(support_size, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # 每个簇选择最接近中心的样本
        selected_indices = []
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            distances = pairwise_distances([cluster_center], features)[0]
            
            # 找到最接近中心的样本
            cluster_mask = (kmeans.labels_ == i)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # 在该簇中找最接近中心的
                cluster_distances = distances[cluster_indices]
                closest_idx = cluster_indices[np.argmin(cluster_distances)]
                selected_indices.append(closest_idx)
        
        return sorted(selected_indices[:support_size])
    
    def _select_by_uncertainty(self, user_images, user_id, support_size):
        """基于不确定性选择（选择高熵样本）"""
        self.model.eval()
        
        uncertainties = []
        with torch.no_grad():
            for img_path in user_images:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                
                # 计算预测熵（不确定性）
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()
                uncertainties.append(entropy)
        
        # 选择top-K高不确定性样本
        indices = np.argsort(uncertainties)[::-1][:support_size]
        return sorted(indices.tolist())
    
    def _select_hybrid(self, user_images, user_id, support_size):
        """
        混合策略：一半高置信度 + 一半多样性
        
        理由：高置信度样本代表类别核心，多样性样本覆盖类内变化
        """
        k1 = support_size // 2  # 高置信度
        k2 = support_size - k1  # 多样性
        
        # 选择高置信度样本
        conf_indices = set(self._select_by_confidence(user_images, user_id, k1))
        
        # 从剩余样本中选择多样性样本
        remaining_images = [img for i, img in enumerate(user_images) 
                          if i not in conf_indices]
        
        if len(remaining_images) >= k2:
            # 重新索引
            remaining_indices = [i for i in range(len(user_images)) 
                               if i not in conf_indices]
            
            # 在剩余样本上做多样性选择
            div_indices_in_remaining = self._select_by_diversity(
                remaining_images, user_id, k2
            )
            
            # 映射回原始索引
            div_indices = [remaining_indices[i] for i in div_indices_in_remaining]
        else:
            # 剩余样本不足，全部使用
            div_indices = [i for i in range(len(user_images)) 
                          if i not in conf_indices]
        
        # 合并
        selected = list(conf_indices) + div_indices
        return sorted(selected[:support_size])


def create_strategic_dataset(data_dir, support_size, strategy, model, 
                            seed=42, device='cuda'):
    """
    创建使用特定策略选择的数据集
    
    Args:
        data_dir: 数据目录
        support_size: 支持集大小
        strategy: 选择策略
        model: 分类器模型
        seed: 随机种子
        device: 设备
    
    Returns:
        support_files_dict: {user_id: [support_file_paths]}
        test_files_dict: {user_id: [test_file_paths]}
    """
    data_path = Path(data_dir)
    selector = SmartSampleSelector(model, device)
    
    support_files_dict = {}
    test_files_dict = {}
    
    # 遍历所有用户
    user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for user_dir in user_dirs:
        user_name = user_dir.name
        if not user_name.startswith('ID_'):
            continue
        
        user_id = int(user_name.split('_')[1]) - 1  # ID_1 -> 0
        
        # 获取所有图像
        image_files = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
        if len(image_files) == 0:
            continue
        
        # 排序确保可重复性
        image_files = sorted(image_files)
        
        # 使用策略选择支持集索引
        support_indices = selector.select_samples(
            image_files, user_id, support_size, strategy, seed
        )
        
        # 分离支持集和测试集
        support_files = [image_files[i] for i in support_indices]
        test_files = [img for i, img in enumerate(image_files) 
                     if i not in support_indices]
        
        support_files_dict[user_id] = support_files
        test_files_dict[user_id] = test_files
        
        print(f"  {user_name}: Support {len(support_files)}, Test {len(test_files)}")
    
    return support_files_dict, test_files_dict 
