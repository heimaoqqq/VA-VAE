#!/usr/bin/env python3
"""
LCCS集成方法 - 结合LCCS的优势避免NCC的问题
核心思想：使用LCCS调整特征，但保留分类器的置信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import sys
import argparse
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from lccs_adapter import FixedLCCSAdapter


class LCCSEnsemble:
    """LCCS集成策略 - 避免NCC问题"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.original_model = deepcopy(model)  # 保留原始模型
        self.adapted_model = model  # LCCS适应后的模型
        
    def apply_lccs(self, support_loader, alpha=0.2):
        """应用LCCS适应"""
        adapter = FixedLCCSAdapter(self.adapted_model, self.device)
        adapter.adapt_bn_stats_v1(support_loader, alpha=alpha)
        return adapter
    
    def ensemble_predict(self, images, method='confidence_weighted'):
        """集成预测"""
        self.original_model.eval()
        self.adapted_model.eval()
        
        with torch.no_grad():
            # 原始模型预测
            orig_outputs = self.original_model(images)
            orig_probs = F.softmax(orig_outputs, dim=1)
            orig_conf, orig_pred = orig_probs.max(dim=1)
            
            # LCCS适应模型预测
            adapt_outputs = self.adapted_model(images)
            adapt_probs = F.softmax(adapt_outputs, dim=1)
            adapt_conf, adapt_pred = adapt_probs.max(dim=1)
            
            if method == 'confidence_weighted':
                # 基于置信度加权
                total_conf = orig_conf + adapt_conf + 1e-8
                orig_weight = orig_conf / total_conf
                adapt_weight = adapt_conf / total_conf
                
                # 加权融合
                final_probs = orig_probs * orig_weight.unsqueeze(1) + \
                             adapt_probs * adapt_weight.unsqueeze(1)
                
            elif method == 'max_confidence':
                # 选择置信度更高的预测
                mask = (adapt_conf > orig_conf).unsqueeze(1)
                final_probs = torch.where(mask, adapt_probs, orig_probs)
                
            elif method == 'average':
                # 简单平均
                final_probs = (orig_probs + adapt_probs) / 2
                
            else:  # 'lccs_only'
                final_probs = adapt_probs
            
            final_conf, final_pred = final_probs.max(dim=1)
            
        return final_pred, final_conf, final_probs
    
    def evaluate(self, test_loader, method='confidence_weighted'):
        """评估集成方法"""
        correct = 0
        total = 0
        all_confidences = []
        
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            predictions, confidences, _ = self.ensemble_predict(images, method)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_confidences.extend(confidences.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(all_confidences)
        
        return accuracy, mean_confidence


def test_lccs_strategies(model_path, data_dir, support_size=3, seed=42):
    """测试不同的LCCS策略"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    support_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
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
    
    support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Data: {len(support_dataset)} support, {len(test_dataset)} test")
    
    # 加载模型
    print("\n📦 Loading model...")
    model = ImprovedClassifier(num_classes=31).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✅ Model loaded")
    
    results = {}
    
    # 1. Baseline (无适应)
    print("\n🎯 Testing Baseline...")
    model.eval()
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            conf, predicted = probs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            confidences.extend(conf.cpu().numpy())
    
    baseline_acc = correct / total
    baseline_conf = np.mean(confidences)
    results['Baseline'] = {'accuracy': baseline_acc, 'confidence': baseline_conf}
    print(f"Baseline: {baseline_acc:.2%} (conf: {baseline_conf:.3f})")
    
    # 2. LCCS (不使用NCC)
    print("\n🎯 Testing LCCS without NCC...")
    adapter = FixedLCCSAdapter(model, device)
    
    # 测试不同的alpha值
    for alpha in [0.1, 0.2, 0.3]:
        # 重新加载模型
        model_lccs = ImprovedClassifier(num_classes=31).to(device)
        if 'model_state_dict' in checkpoint:
            model_lccs.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_lccs.load_state_dict(checkpoint)
        
        adapter_lccs = FixedLCCSAdapter(model_lccs, device)
        adapter_lccs.adapt_bn_stats_v1(support_loader, alpha=alpha)
        
        acc, conf = adapter_lccs.evaluate(test_loader, use_ncc=False)
        results[f'LCCS (α={alpha})'] = {'accuracy': acc, 'confidence': conf}
        print(f"LCCS (α={alpha}): {acc:.2%} (conf: {conf:.3f})")
    
    # 3. LCCS集成方法
    print("\n🎯 Testing LCCS Ensemble Methods...")
    
    # 重新加载用于集成
    model_orig = ImprovedClassifier(num_classes=31).to(device)
    model_adapt = ImprovedClassifier(num_classes=31).to(device)
    
    if 'model_state_dict' in checkpoint:
        model_orig.load_state_dict(checkpoint['model_state_dict'])
        model_adapt.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_orig.load_state_dict(checkpoint)
        model_adapt.load_state_dict(checkpoint)
    
    ensemble = LCCSEnsemble(model_adapt, device)
    ensemble.original_model = model_orig
    ensemble.apply_lccs(support_loader, alpha=0.2)
    
    # 测试不同集成策略
    for method in ['confidence_weighted', 'max_confidence', 'average']:
        acc, conf = ensemble.evaluate(test_loader, method=method)
        results[f'LCCS Ensemble ({method})'] = {'accuracy': acc, 'confidence': conf}
        print(f"LCCS Ensemble ({method}): {acc:.2%} (conf: {conf:.3f})")
    
    # 4. 总结
    print("\n" + "="*70)
    print("📊 SUMMARY: LCCS Strategies (Without NCC)")
    print("="*70)
    print(f"{'Method':<30} {'Accuracy':<12} {'Confidence':<12} {'Improvement'}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), 
                           key=lambda x: (x[1]['confidence'] > 0.7, x[1]['accuracy']), 
                           reverse=True)
    
    for method, metrics in sorted_results:
        improvement = metrics['accuracy'] - baseline_acc
        conf_flag = "✅" if metrics['confidence'] > 0.7 else "⚠️"
        print(f"{method:<30} {metrics['accuracy']:<11.2%} {metrics['confidence']:<11.3f}{conf_flag} {improvement:+.2%}")
    
    # 找出置信度>0.7的最佳方法
    valid_methods = [(m, v) for m, v in results.items() if v['confidence'] > 0.7]
    if valid_methods:
        best = max(valid_methods, key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best method (conf>0.7): {best[0]} - {best[1]['accuracy']:.2%} (conf: {best[1]['confidence']:.3f})")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test LCCS Strategies')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/working/organized_gait_dataset/Normal_free')
    parser.add_argument('--support-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    test_lccs_strategies(
        model_path=args.model_path,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed
    )
