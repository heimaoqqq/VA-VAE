#!/usr/bin/env python3
"""
LCCS + PNC 组合方法
先用LCCS适应BatchNorm，再用PNC原型校准
理论上获得最优性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from tabulate import tabulate
import sys
sys.path.append(str(Path(__file__).parent.parent))

from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CombinedLCCS_PNC:
    """LCCS + PNC 组合适配器"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.original_bn_stats = self._save_bn_stats()
        self.prototypes = None
        
    def _save_bn_stats(self):
        """保存原始BN统计量"""
        bn_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'momentum': module.momentum,
                    'num_batches_tracked': module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None
                }
        return bn_stats
    
    def _restore_bn_stats(self):
        """恢复原始BN统计量"""
        for name, module in self.model.named_modules():
            if name in self.original_bn_stats:
                module.running_mean.data = self.original_bn_stats[name]['running_mean']
                module.running_var.data = self.original_bn_stats[name]['running_var']
                if module.momentum is not None:
                    module.momentum = self.original_bn_stats[name]['momentum']
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.data = self.original_bn_stats[name]['num_batches_tracked']
    
    def step1_lccs_adaptation(self, support_loader, method='progressive', **kwargs):
        """步骤1：LCCS适应BatchNorm"""
        print(f"\n🔧 Step 1: LCCS Adaptation (method={method})")
        
        if method == 'progressive':
            self._lccs_progressive(support_loader, **kwargs)
        elif method == 'weighted':
            self._lccs_weighted(support_loader, **kwargs)
        else:
            raise ValueError(f"Unknown LCCS method: {method}")
    
    def _lccs_progressive(self, support_loader, momentum=0.01, iterations=5):
        """渐进式LCCS更新"""
        print(f"   Progressive update: momentum={momentum}, iterations={iterations}")
        
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 设置小momentum
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.momentum = momentum
        
        # 多次迭代更新
        with torch.no_grad():
            for i in range(iterations):
                for batch in tqdm(support_loader, desc=f"   LCCS iter {i+1}/{iterations}", leave=False):
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        self.model.eval()
        print("   ✅ LCCS adaptation complete")
    
    def _lccs_weighted(self, support_loader, alpha=0.3):
        """加权融合LCCS"""
        print(f"   Weighted fusion: alpha={alpha}")
        
        # 保存源域统计
        source_stats = self._save_bn_stats()
        
        # 收集目标域统计
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
                module.momentum = 1.0
        
        with torch.no_grad():
            for _ in range(10):
                for batch in support_loader:
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        # 保存目标域统计
        target_stats = self._save_bn_stats()
        
        # 加权融合
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in source_stats and name in target_stats:
                    module.running_mean = (1-alpha) * source_stats[name]['running_mean'] + \
                                         alpha * target_stats[name]['running_mean']
                    module.running_var = (1-alpha) * source_stats[name]['running_var'] + \
                                        alpha * target_stats[name]['running_var']
        
        self.model.eval()
        print("   ✅ LCCS weighted fusion complete")
    
    def step2_build_prototypes(self, support_loader):
        """步骤2：在LCCS适应后的模型上构建原型"""
        print(f"\n🎯 Step 2: Building Prototypes on LCCS-adapted model")
        
        self.model.eval()
        
        # 收集特征和标签
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(support_loader, desc="   Extracting features"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                
                # 提取backbone特征（LCCS已经改善了这些特征）
                features = self.model.backbone(images)
                
                all_features.append(features.cpu())
                all_labels.extend(labels.tolist())
        
        # 合并所有特征
        all_features = torch.cat(all_features, dim=0)
        
        # 计算每个类的原型
        num_classes = max(all_labels) + 1
        prototypes = []
        
        for class_id in range(num_classes):
            class_mask = [i for i, l in enumerate(all_labels) if l == class_id]
            if len(class_mask) > 0:
                class_features = all_features[class_mask]
                # 计算均值原型
                prototype = class_features.mean(dim=0)
                # L2归一化
                prototype = prototype / prototype.norm(2)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(all_features.shape[1]))
        
        self.prototypes = torch.stack(prototypes).to(self.device)
        print(f"   ✅ Built prototypes: {self.prototypes.shape}")
    
    def step3_combined_inference(self, test_loader, fusion_alpha=0.6, similarity_tau=0.01):
        """步骤3：组合推理 - LCCS特征 + PNC校准"""
        print(f"\n🚀 Step 3: Combined Inference")
        print(f"   Fusion alpha: {fusion_alpha}")
        print(f"   Similarity tau: {similarity_tau}")
        
        if self.prototypes is None:
            raise ValueError("Prototypes not built! Run step2_build_prototypes first.")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="   Evaluating"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 1. 获取LCCS改善后的分类器输出
                outputs = self.model(images)
                
                # 2. 提取LCCS改善后的特征
                features = self.model.backbone(images)
                
                # 3. L2归一化特征
                features = features / features.norm(2, dim=1, keepdim=True)
                
                # 4. 计算与原型的相似度
                similarities = torch.matmul(features, self.prototypes.T)
                
                # 5. 温度缩放并转为概率
                proto_probs = F.softmax(similarities / similarity_tau, dim=1)
                
                # 6. 分类器概率
                class_probs = F.softmax(outputs, dim=1)
                
                # 7. 融合两种概率
                combined_probs = (1 - fusion_alpha) * class_probs + fusion_alpha * proto_probs
                
                # 8. 最终预测
                confidences, predictions = torch.max(combined_probs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 计算准确率
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        mean_confidence = np.mean(all_confidences)
        
        return accuracy, mean_confidence
    
    def evaluate_baseline(self, test_loader):
        """基线评估（无任何适应）"""
        self.model.eval()
        
        correct = 0
        total = 0
        confidences = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="   Baseline eval"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                confidences.extend(conf.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(confidences)
        return accuracy, mean_confidence


def run_combined_experiment(model_path, data_dir, support_size=3, seed=42,
                          lccs_method='progressive', lccs_momentum=0.01, lccs_iterations=5,
                          fusion_alpha=0.6, similarity_tau=0.01):
    """运行LCCS+PNC组合实验"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("🔬 LCCS + PNC COMBINED EXPERIMENT")
    print("="*80)
    
    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 支持集
    support_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='support',
        seed=seed
    )
    
    support_loader = DataLoader(
        support_dataset,
        batch_size=31,
        shuffle=False,  # 保持顺序以便原型构建
        num_workers=2
    )
    
    # 测试集
    test_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='test',
        seed=seed
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\n📊 Data Configuration:")
    print(f"   Support samples: {len(support_dataset)} ({support_size}/user)")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Seed: {seed}")
    
    # 加载模型
    print(f"\n📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ImprovedClassifier(
        num_classes=checkpoint.get('num_classes', 31),
        backbone=checkpoint.get('backbone', 'resnet18')
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 创建组合适配器
    adapter = CombinedLCCS_PNC(model, device)
    
    # 基线评估
    print("\n" + "-"*60)
    print("📊 BASELINE (No adaptation)")
    baseline_acc, baseline_conf = adapter.evaluate_baseline(test_loader)
    print(f"   Accuracy: {baseline_acc:.2%}")
    print(f"   Confidence: {baseline_conf:.3f}")
    
    # 步骤1：LCCS适应
    print("\n" + "-"*60)
    adapter.step1_lccs_adaptation(
        support_loader,
        method=lccs_method,
        momentum=lccs_momentum,
        iterations=lccs_iterations
    )
    
    # 步骤2：构建原型（在LCCS适应后的模型上）
    print("-"*60)
    adapter.step2_build_prototypes(support_loader)
    
    # 步骤3：组合推理
    print("-"*60)
    combined_acc, combined_conf = adapter.step3_combined_inference(
        test_loader,
        fusion_alpha=fusion_alpha,
        similarity_tau=similarity_tau
    )
    
    # 结果汇总
    print("\n" + "="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    
    results = [
        ["Method", "Accuracy", "Confidence", "Improvement"],
        ["Baseline", f"{baseline_acc:.2%}", f"{baseline_conf:.3f}", "-"],
        ["LCCS+PNC", f"{combined_acc:.2%}", f"{combined_conf:.3f}", 
         f"{combined_acc-baseline_acc:+.2%}"]
    ]
    
    print(tabulate(results, headers="firstrow", tablefmt="grid"))
    
    # 与单独方法比较
    print(f"\n📊 Comparison with individual methods:")
    print(f"   PNC alone: 84.46% (+8.80%)")
    print(f"   LCCS alone: 78.09% (+2.42%)")
    print(f"   LCCS+PNC: {combined_acc:.2%} ({combined_acc-baseline_acc:+.2%})")
    
    # 评估
    improvement = combined_acc - baseline_acc
    if improvement > 0.09:
        print(f"\n🏆 EXCELLENT! Combined approach achieves best performance!")
    elif improvement > 0.08:
        print(f"\n✅ Good! Combined approach is competitive with PNC alone.")
    else:
        print(f"\n⚠️ Combined approach did not exceed PNC alone.")
    
    return {
        'baseline': baseline_acc,
        'combined': combined_acc,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(description='LCCS + PNC Combined Approach')
    
    # 数据和模型
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data')
    parser.add_argument('--support-size', type=int, default=3,
                       help='Support samples per user')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data split')
    
    # LCCS参数
    parser.add_argument('--lccs-method', type=str, default='progressive',
                       choices=['progressive', 'weighted'],
                       help='LCCS adaptation method')
    parser.add_argument('--lccs-momentum', type=float, default=0.01,
                       help='Momentum for progressive LCCS')
    parser.add_argument('--lccs-iterations', type=int, default=5,
                       help='Iterations for progressive LCCS')
    parser.add_argument('--lccs-alpha', type=float, default=0.3,
                       help='Alpha for weighted LCCS')
    
    # PNC参数
    parser.add_argument('--fusion-alpha', type=float, default=0.6,
                       help='Fusion weight for PNC')
    parser.add_argument('--similarity-tau', type=float, default=0.01,
                       help='Temperature for similarity')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_combined_experiment(
        model_path=args.model_path,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed,
        lccs_method=args.lccs_method,
        lccs_momentum=args.lccs_momentum,
        lccs_iterations=args.lccs_iterations,
        fusion_alpha=args.fusion_alpha,
        similarity_tau=args.similarity_tau
    )
    
    print(f"\n🏁 Experiment complete!")


if __name__ == '__main__':
    main()
