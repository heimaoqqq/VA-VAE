#!/usr/bin/env python3
"""
改进的PNC方法 - 基于实验结果优化
结合LCCS的BN适应和PNC的融合策略
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

sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from lccs_adapter import FixedLCCSAdapter


class ImprovedPNC:
    """改进的PNC：LCCS + 优化的融合策略"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.prototypes = None
        
    def compute_prototypes(self, support_loader):
        """计算高质量原型"""
        self.model.eval()
        features_dict = {}
        
        with torch.no_grad():
            for batch in support_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.model.backbone(images)
                features = F.normalize(features, dim=1)
                
                # 按类别组织
                for feat, label in zip(features, labels):
                    label_id = label.item()
                    if label_id not in features_dict:
                        features_dict[label_id] = []
                    features_dict[label_id].append(feat)
        
        # 计算每个类的原型
        self.prototypes = {}
        for label_id, feats in features_dict.items():
            # 使用加权平均（基于特征范数）
            feats_stack = torch.stack(feats)
            weights = feats_stack.norm(dim=1, keepdim=True)
            weights = F.softmax(weights.squeeze(), dim=0).unsqueeze(1)
            prototype = (feats_stack * weights).sum(dim=0)
            self.prototypes[label_id] = F.normalize(prototype, dim=0)
            
        return self.prototypes
    
    def adaptive_fusion_predict(self, features, outputs, alpha_base=0.5, confidence_weight=True):
        """自适应融合预测"""
        features = F.normalize(features, dim=1)
        
        # 原型预测
        proto_logits = []
        for i in range(31):  # 31个类
            if i in self.prototypes:
                sim = torch.matmul(features, self.prototypes[i].unsqueeze(0).T)
                proto_logits.append(sim)
            else:
                proto_logits.append(torch.zeros((features.shape[0], 1), device=self.device) - 1e10)
        
        proto_logits = torch.cat(proto_logits, dim=1)
        proto_probs = F.softmax(proto_logits / 0.01, dim=1)  # 温度0.01
        
        # 分类器预测
        class_probs = F.softmax(outputs, dim=1)
        
        if confidence_weight:
            # 基于置信度动态调整融合权重
            proto_conf = proto_probs.max(dim=1)[0]
            class_conf = class_probs.max(dim=1)[0]
            
            # 归一化置信度作为权重
            total_conf = proto_conf + class_conf + 1e-8
            proto_weight = proto_conf / total_conf
            class_weight = class_conf / total_conf
            
            # 动态融合
            final_probs = proto_probs * proto_weight.unsqueeze(1) + class_probs * class_weight.unsqueeze(1)
        else:
            # 固定权重融合
            final_probs = alpha_base * proto_probs + (1 - alpha_base) * class_probs
        
        predictions = final_probs.argmax(dim=1)
        confidences = final_probs.max(dim=1)[0]
        
        return predictions, confidences, final_probs
    
    def evaluate(self, test_loader, use_lccs=True, lccs_alpha=0.1):
        """评估改进的PNC"""
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
                
                # 前向传播
                outputs = self.model(images)
                features = self.model.backbone(images)
                
                # 自适应融合预测
                predictions, confidences, _ = self.adaptive_fusion_predict(
                    features, outputs, 
                    alpha_base=0.5, 
                    confidence_weight=True
                )
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                all_confidences.extend(confidences.cpu().numpy())
        
        accuracy = correct / total
        mean_confidence = np.mean(all_confidences)
        
        return accuracy, mean_confidence


def test_improved_pnc(model_path, data_dir, support_size=3, seed=42):
    """测试改进的PNC方法"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
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
    
    # 1. Baseline
    print("\n🎯 Testing Baseline...")
    model.eval()
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for images, labels in test_loader:
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
    
    # 2. 标准PNC
    print("\n🎯 Testing Standard PNC...")
    pnc = ImprovedPNC(model, device)
    pnc.compute_prototypes(support_loader)
    
    acc, conf = pnc.evaluate(test_loader, use_lccs=False)
    results['Standard PNC'] = {'accuracy': acc, 'confidence': conf}
    print(f"Standard PNC: {acc:.2%} (conf: {conf:.3f})")
    
    # 3. LCCS + 改进PNC
    print("\n🎯 Testing LCCS + Improved PNC...")
    # 应用LCCS
    adapter = FixedLCCSAdapter(model, device)
    adapter.adapt_bn_stats_v1(support_loader, alpha=0.2)  # 最佳alpha
    adapter.compute_class_prototypes(support_loader)
    
    # 使用改进的PNC
    pnc_lccs = ImprovedPNC(model, device)
    pnc_lccs.prototypes = adapter.prototypes  # 使用LCCS后的原型
    
    acc, conf = pnc_lccs.evaluate(test_loader, use_lccs=True)
    results['LCCS + Improved PNC'] = {'accuracy': acc, 'confidence': conf}
    print(f"LCCS + Improved PNC: {acc:.2%} (conf: {conf:.3f})")
    
    # 总结
    print("\n" + "="*60)
    print("📊 SUMMARY: Improved PNC Methods")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':<12} {'Confidence':<12} {'Improvement'}")
    print("-"*70)
    
    for method, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        improvement = metrics['accuracy'] - baseline_acc
        print(f"{method:<25} {metrics['accuracy']:<11.2%} {metrics['confidence']:<11.3f} {improvement:+.2%}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Improved PNC Methods')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/working/organized_gait_dataset/Normal_free')
    parser.add_argument('--support-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    test_improved_pnc(
        model_path=args.model_path,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed
    )
