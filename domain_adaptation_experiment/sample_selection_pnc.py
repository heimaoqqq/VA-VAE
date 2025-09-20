#!/usr/bin/env python3
"""
样本筛选PNC - 验证筛选策略对PNC性能的影响
基于文献调研的不同样本选择方法
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from improved_pnc import ImprovedPNC


class SampleSelector:
    """样本选择策略"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def random_selection(self, dataset, support_size=3):
        """随机选择（baseline）"""
        indices = np.random.choice(len(dataset), support_size, replace=False)
        return Subset(dataset, indices)
    
    def confidence_selection(self, dataset, support_size=3):
        """基于分类器置信度选择"""
        self.model.eval()
        confidences = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if len(dataset[i]) == 3:
                    image, label, _ = dataset[i]
                else:
                    image, label = dataset[i]
                
                image = image.unsqueeze(0).to(self.device)
                output = self.model(image)
                prob = F.softmax(output, dim=1)
                conf = prob.max().item()
                confidences.append((i, conf))
        
        # 选择置信度最高的support_size个样本
        confidences.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in confidences[:support_size]]
        
        return Subset(dataset, selected_indices)
    
    def diversity_selection(self, dataset, support_size=3):
        """基于特征多样性选择"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if len(dataset[i]) == 3:
                    image, label, _ = dataset[i]
                else:
                    image, label = dataset[i]
                
                image = image.unsqueeze(0).to(self.device)
                feature = self.model.backbone(image)
                features.append(feature.cpu().numpy().flatten())
        
        features = np.array(features)
        
        # 使用K-means聚类选择多样化样本
        if len(features) <= support_size:
            return Subset(dataset, list(range(len(dataset))))
        
        kmeans = KMeans(n_clusters=support_size, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # 选择离聚类中心最近的样本
        selected_indices = []
        for i in range(support_size):
            center = kmeans.cluster_centers_[i]
            distances = pairwise_distances([center], features)[0]
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)
        
        return Subset(dataset, selected_indices)
    
    def uncertainty_selection(self, dataset, support_size=3):
        """基于不确定性选择（选择模型最不确定的样本）"""
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if len(dataset[i]) == 3:
                    image, label, _ = dataset[i]
                else:
                    image, label = dataset[i]
                
                image = image.unsqueeze(0).to(self.device)
                output = self.model(image)
                prob = F.softmax(output, dim=1)
                # 使用entropy作为不确定性度量
                entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
                uncertainties.append((i, entropy))
        
        # 选择不确定性最高的support_size个样本
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in uncertainties[:support_size]]
        
        return Subset(dataset, selected_indices)
    
    def balanced_selection(self, dataset, support_size=3):
        """平衡选择：结合置信度和多样性"""
        self.model.eval()
        features = []
        confidences = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if len(dataset[i]) == 3:
                    image, label, _ = dataset[i]
                else:
                    image, label = dataset[i]
                
                image = image.unsqueeze(0).to(self.device)
                feature = self.model.backbone(image)
                output = self.model(image)
                prob = F.softmax(output, dim=1)
                conf = prob.max().item()
                
                features.append(feature.cpu().numpy().flatten())
                confidences.append(conf)
        
        features = np.array(features)
        confidences = np.array(confidences)
        
        # 归一化置信度
        norm_conf = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-8)
        
        # 计算多样性分数（到其他样本的平均距离）
        distances = pairwise_distances(features)
        diversity_scores = distances.mean(axis=1)
        norm_diversity = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # 综合分数：0.7*置信度 + 0.3*多样性
        combined_scores = 0.7 * norm_conf + 0.3 * norm_diversity
        
        # 选择分数最高的样本
        selected_indices = np.argsort(combined_scores)[-support_size:].tolist()
        
        return Subset(dataset, selected_indices)


def test_sample_selection_strategies(model_path, data_dir, support_size=3, seed=42):
    """测试不同样本选择策略"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载完整目标域数据
    full_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=100,  # 使用更多样本用于选择
        mode='support',
        seed=seed
    )
    
    test_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='test',
        seed=seed
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    model = ImprovedClassifier(num_classes=31).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"📊 Testing sample selection with {len(full_dataset)} candidates")
    
    # 初始化样本选择器
    selector = SampleSelector(model, device)
    
    # 按类别组织数据
    class_datasets = {}
    for i in range(len(full_dataset)):
        if len(full_dataset[i]) == 3:
            _, label, _ = full_dataset[i]
        else:
            _, label = full_dataset[i]
        
        if label.item() not in class_datasets:
            class_datasets[label.item()] = []
        class_datasets[label.item()].append(i)
    
    results = {}
    
    # 测试不同选择策略
    strategies = {
        'Random': 'random_selection',
        'High Confidence': 'confidence_selection', 
        'Diversity': 'diversity_selection',
        'High Uncertainty': 'uncertainty_selection',
        'Balanced': 'balanced_selection'
    }
    
    for strategy_name, method_name in strategies.items():
        print(f"\n🎯 Testing {strategy_name} Selection...")
        
        # 对每个类别应用选择策略
        selected_indices = []
        for class_id, indices in class_datasets.items():
            if len(indices) >= support_size:
                class_subset = Subset(full_dataset, indices)
                method = getattr(selector, method_name)
                selected_subset = method(class_subset, support_size)
                # 映射回原始索引
                for idx in selected_subset.indices:
                    selected_indices.append(indices[idx])
            else:
                selected_indices.extend(indices)  # 如果样本不够，全部选择
        
        # 创建支持集
        support_dataset = Subset(full_dataset, selected_indices)
        support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False)
        
        # 使用PNC评估
        pnc = ImprovedPNC(model, device, similarity_tau=0.005)
        pnc.compute_prototypes(support_loader)
        
        acc, conf = pnc.evaluate(test_loader, fusion_alpha=0.7, use_confidence_weight=False)
        results[strategy_name] = {'accuracy': acc, 'confidence': conf, 'samples': len(selected_indices)}
        
        print(f"{strategy_name}: {acc:.2%} (conf: {conf:.3f}, samples: {len(selected_indices)})")
    
    # 总结
    print("\n" + "="*70)
    print("📊 SUMMARY: Sample Selection Strategies")
    print("="*70)
    print(f"{'Strategy':<20} {'Accuracy':<12} {'Confidence':<12} {'Samples':<8} {'vs Random'}")
    print("-"*80)
    
    random_acc = results['Random']['accuracy']
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for strategy, metrics in sorted_results:
        improvement = metrics['accuracy'] - random_acc
        print(f"{strategy:<20} {metrics['accuracy']:<11.2%} {metrics['confidence']:<11.3f} {metrics['samples']:<8d} {improvement:+.2%}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Sample Selection Strategies for PNC')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/working/organized_gait_dataset/Normal_free')
    parser.add_argument('--support-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    test_sample_selection_strategies(
        model_path=args.model_path,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed
    )
