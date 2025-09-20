#!/usr/bin/env python3
"""
修复版LCCS - 不完全重置BN统计量，而是微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset


class FixedLCCSAdapter:
    """修复版LCCS：保留原始统计量，仅微调"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.original_bn_stats = self._save_bn_stats()
        
    def _save_bn_stats(self):
        """保存原始BN层的统计量"""
        bn_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
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
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.data = self.original_bn_stats[name]['num_batches_tracked']
    
    def adapt_bn_stats_v1(self, support_loader, alpha=0.3):
        """方法1：加权融合原始和目标域统计量"""
        print(f"🔧 Method 1: Weighted fusion (α={alpha})...")
        
        # 收集目标域统计量（不重置原始统计量）
        self.model.train()
        
        # 保存当前统计量
        source_stats = self._save_bn_stats()
        
        # 临时重置以收集纯目标域统计量
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
                module.momentum = 1.0  # 快速收集
        
        # 收集目标域统计量
        with torch.no_grad():
            for _ in range(10):  # 多次迭代以稳定
                for batch in support_loader:
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        # 保存目标域统计量
        target_stats = self._save_bn_stats()
        
        # 融合源域和目标域统计量
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in source_stats and name in target_stats:
                    # 加权平均
                    module.running_mean = (1-alpha) * source_stats[name]['running_mean'] + \
                                         alpha * target_stats[name]['running_mean']
                    module.running_var = (1-alpha) * source_stats[name]['running_var'] + \
                                        alpha * target_stats[name]['running_var']
        
        self.model.eval()
        print("✅ BN stats adapted via weighted fusion!")
    
    def adapt_bn_stats_v2(self, support_loader, momentum=0.01, iterations=5):
        """方法2：渐进式更新（小momentum）"""
        print(f"🔧 Method 2: Progressive update (momentum={momentum}, iter={iterations})...")
        
        # 设置为训练模式但不更新参数
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 使用小momentum渐进更新
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.momentum = momentum  # 小步更新
                # 不重置统计量！保留原始值
        
        # 多次迭代渐进更新
        with torch.no_grad():
            for iteration in range(iterations):
                for batch in tqdm(support_loader, desc=f"Iter {iteration+1}/{iterations}", leave=False):
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        self.model.eval()
        print("✅ BN stats progressively updated!")
    
    def adapt_bn_stats_v3(self, support_loader):
        """方法3：仅调整均值偏移"""
        print("🔧 Method 3: Mean-shift only...")
        
        self.model.eval()
        
        # 计算支持集特征的均值偏移
        with torch.no_grad():
            source_means = {}
            target_means = {}
            
            # 收集每层的激活均值
            def hook_fn(name):
                def hook(module, input, output):
                    if name not in target_means:
                        target_means[name] = []
                    if isinstance(output, torch.Tensor):
                        # 计算batch均值
                        if len(output.shape) == 4:  # Conv层
                            mean = output.mean(dim=[0,2,3])
                        else:  # Linear层
                            mean = output.mean(dim=0)
                        target_means[name].append(mean.cpu())
                return hook
            
            # 注册钩子
            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # 前向传播收集统计
            for batch in support_loader:
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images, _ = batch
                images = images.to(self.device)
                _ = self.model(images)
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 计算均值偏移并应用
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if name in target_means and len(target_means[name]) > 0:
                        target_mean = torch.stack(target_means[name]).mean(dim=0)
                        source_mean = module.running_mean
                        # 调整均值
                        shift = target_mean.to(self.device) - source_mean
                        module.running_mean = source_mean + 0.3 * shift  # 部分偏移
        
        print("✅ Mean-shift adaptation complete!")
    
    def compute_class_prototypes(self, support_loader):
        """计算每个类的原型（质心）用于NCC分类"""
        print("🔧 Computing class prototypes for NCC...")
        
        self.model.eval()
        class_features = defaultdict(list)
        
        with torch.no_grad():
            for batch in support_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征（backbone输出）
                features = self.model.backbone(images)  # [B, 512] for ResNet18
                
                # 按类收集特征
                for feat, label in zip(features, labels):
                    class_features[label.item()].append(feat)
        
        # 计算每个类的原型（均值）
        self.prototypes = {}
        for class_id, feats in class_features.items():
            if feats:
                class_prototype = torch.stack(feats).mean(dim=0)
                self.prototypes[class_id] = F.normalize(class_prototype, dim=0)
        
        print(f"✅ Computed prototypes for {len(self.prototypes)} classes")
        return self.prototypes
    
    def ncc_predict(self, features):
        """使用最近质心分类器预测"""
        if not hasattr(self, 'prototypes'):
            raise ValueError("Prototypes not computed! Call compute_class_prototypes first.")
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算到每个原型的余弦相似度（更适合对比学习特征）
        similarities = []
        class_ids = []
        for class_id, prototype in self.prototypes.items():
            # 使用余弦相似度替代L2距离
            sim = torch.matmul(features, prototype.unsqueeze(1))  # [batch, 1]
            similarities.append(sim)
            class_ids.append(class_id)
        
        # 找最高相似度的原型
        similarities = torch.cat(similarities, dim=1)  # [batch_size, num_classes]
        argmax_indices = similarities.argmax(dim=1)  # GPU上的索引
        
        # 将class_ids转换为与相似度相同设备的张量
        class_ids_tensor = torch.tensor(class_ids, device=similarities.device)
        predictions = class_ids_tensor[argmax_indices]
        
        return predictions
    
    def ncc_predict_with_confidence(self, features, temperature=0.1):
        """使用最近质心分类器预测（带置信度和温度缩放）"""
        if not hasattr(self, 'prototypes'):
            raise ValueError("Prototypes not computed! Call compute_class_prototypes first.")
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算到每个原型的余弦相似度
        similarities = []
        class_ids = []
        for class_id, prototype in self.prototypes.items():
            # 使用余弦相似度
            sim = torch.matmul(features, prototype.unsqueeze(1))  # [batch, 1]
            similarities.append(sim)
            class_ids.append(class_id)
        
        # 合并相似度
        similarities = torch.cat(similarities, dim=1)  # [batch_size, num_classes]
        
        # 温度缩放后的softmax（解决置信度过低的问题）
        scaled_similarities = similarities / temperature
        probs = F.softmax(scaled_similarities, dim=1)
        
        # 获取预测和置信度
        confidences, indices = probs.max(dim=1)
        
        # 映射到类别ID
        class_ids_tensor = torch.tensor(class_ids, device=similarities.device)
        predictions = class_ids_tensor[indices]
        
        return predictions, confidences
    
    def evaluate(self, test_loader, use_ncc=False, return_confidence=True):
        """评估性能（支持NCC和原始分类器，返回置信度）"""
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
                
                if use_ncc:
                    # 使用NCC分类
                    features = self.model.backbone(images)
                    predicted, confidences = self.ncc_predict_with_confidence(features)
                else:
                    # 使用原始分类器
                    outputs = self.model(images)
                    probs = F.softmax(outputs, dim=1)
                    confidences, predicted = torch.max(probs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_confidences.extend(confidences.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(all_confidences)
        
        if return_confidence:
            return accuracy, mean_confidence
        return accuracy


def load_model(model_path, device, model_type='improved'):
    """加载训练好的分类器模型"""
    if model_type == 'standard':
        # 使用标准ResNet18
        from train_standard_resnet import StandardResNet18Classifier
        model = StandardResNet18Classifier(num_classes=31)
        # 修改backbone访问方式
        model.backbone = model.backbone  # ResNet18已经是backbone
    else:
        # 使用ImprovedClassifier
        model = ImprovedClassifier(num_classes=31)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    return model


def test_all_methods(model_path, data_dir, support_size=3, seed=42, tune_params=False):
    """测试所有LCCS方法（包含NCC）"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        batch_size=31,  # 所有用户一个batch
        shuffle=True,
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
    
    print(f"📊 Data split: {len(support_dataset)} support, {len(test_dataset)} test")
    
    results = {}
    
    # 基线测试（无适应）
    print("\n" + "="*60)
    print("🔬 BASELINE: No adaptation")
    model = load_model(model_path, device)
    adapter = FixedLCCSAdapter(model, device)
    baseline_acc, baseline_conf = adapter.evaluate(test_loader, use_ncc=False)
    results['baseline'] = {'accuracy': baseline_acc, 'confidence': baseline_conf}
    print(f"Baseline accuracy: {baseline_acc:.2%}, confidence: {baseline_conf:.3f}")
    
    # 测试所有方法组合
    print("\n" + "="*60)
    print("🔬 Testing LCCS methods with/without NCC")
    print("="*60)
    
    if tune_params:
        # 参数调优模式：测试多个alpha值
        print("\n🔬 Parameter Tuning Mode")
        alpha_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        momentum_values = [0.001, 0.005, 0.01]
        
        for alpha in alpha_values:
            for use_ncc in [False, True]:
                method_name = f"weighted_a{alpha}{'_NCC' if use_ncc else ''}"
                print(f"\n📊 Testing {method_name}...")
                
                model = load_model(model_path, device)
                adapter = FixedLCCSAdapter(model, device)
                adapter.adapt_bn_stats_v1(support_loader, alpha=alpha)
                
                if use_ncc:
                    adapter.compute_class_prototypes(support_loader)
                
                acc, conf = adapter.evaluate(test_loader, use_ncc=use_ncc)
                results[method_name] = {'accuracy': acc, 'confidence': conf}
                print(f"   Accuracy: {acc:.2%}, Confidence: {conf:.3f}")
                
        for momentum in momentum_values:
            for use_ncc in [False, True]:
                method_name = f"prog_m{momentum}{'_NCC' if use_ncc else ''}"
                print(f"\n📊 Testing {method_name}...")
                
                model = load_model(model_path, device)
                adapter = FixedLCCSAdapter(model, device)
                adapter.adapt_bn_stats_v2(support_loader, momentum=momentum, iterations=3)
                
                if use_ncc:
                    adapter.compute_class_prototypes(support_loader)
                
                acc, conf = adapter.evaluate(test_loader, use_ncc=use_ncc)
                results[method_name] = {'accuracy': acc, 'confidence': conf}
                print(f"   Accuracy: {acc:.2%}, Confidence: {conf:.3f}")
    else:
        # 标准测试模式
        for method in ['weighted_gentle', 'progressive_conservative']:
            for use_ncc in [False, True]:
                method_name = f"{method}{'_NCC' if use_ncc else ''}"
                print(f"\n📊 Testing {method_name}...")
                
                # 重新加载模型
                model = load_model(model_path, device)
                adapter = FixedLCCSAdapter(model, device)
                
                # 应用不同的适应方法
                if method == 'weighted_gentle':
                    adapter.adapt_bn_stats_v1(support_loader, alpha=0.1)
                elif method == 'progressive_conservative':
                    adapter.adapt_bn_stats_v2(support_loader, momentum=0.005, iterations=3)
                
                # 如果使用NCC，计算原型
                if use_ncc:
                    adapter.compute_class_prototypes(support_loader)
                
                # 评估
                acc, conf = adapter.evaluate(test_loader, use_ncc=use_ncc)
                results[method_name] = {'accuracy': acc, 'confidence': conf}
                improvement = acc - baseline_acc
                print(f"   Accuracy: {acc:.2%} ({improvement:+.2%}), Confidence: {conf:.3f}")
    
    # 总结
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':>10} {'Confidence':>12} {'Improvement':>12}")
    print("-"*70)
    
    # 按准确率排序
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else x[1], 
                           reverse=True)
    
    for method, metrics in sorted_results:
        if isinstance(metrics, dict):
            acc = metrics['accuracy']
            conf = metrics['confidence']
        else:
            # 兼容旧格式
            acc = metrics
            conf = 0.0
        
        improvement = acc - baseline_acc
        print(f"{method:<25} {acc:>9.2%} {conf:>11.3f} {improvement:>+11.2%}")
    
    # 找最佳方法
    best_method = max(results.items(), 
                     key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else x[1])
    method_name = best_method[0]
    if isinstance(best_method[1], dict):
        best_acc = best_method[1]['accuracy']
        best_conf = best_method[1]['confidence']
        print(f"\n🏆 Best method: {method_name} with {best_acc:.2%} (confidence: {best_conf:.3f})")
    else:
        best_acc = best_method[1]
        print(f"\n🏆 Best method: {method_name} with {best_acc:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--support-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune-params', action='store_true',
                       help='Enable parameter tuning mode to test multiple alpha/momentum values')
    
    args = parser.parse_args()
    test_all_methods(args.model_path, args.data_dir, args.support_size, args.seed, args.tune_params)


if __name__ == '__main__':
    main()
