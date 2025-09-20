#!/usr/bin/env python3
"""
评估ImprovedClassifier + PNC - 严格排除支持集，避免数据泄漏
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from tabulate import tabulate
import sys
sys.path.append(str(Path(__file__).parent.parent))

from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms


class NoLeakEvaluator:
    """无数据泄漏的评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """加载模型"""
        print(f"📦 Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = ImprovedClassifier(
            num_classes=checkpoint.get('num_classes', 31),
            backbone=checkpoint.get('backbone', 'resnet18')
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Model loaded successfully")
        return model
    
    def evaluate(self, model, data_dir, support_size, seed, 
                 use_prototypes=False, prototype_path=None,
                 fusion_alpha=0.5, similarity_tau=0.1, use_pure_ncc=False):
        """评估模型 - 使用严格划分的测试集"""
        
        mode = "with PNC" if use_prototypes else "Baseline"
        print(f"\n🎯 Evaluating ({mode})...")
        
        # 创建测试集（排除支持集）
        test_dataset = SplitTargetDomainDataset(
            data_dir=data_dir,
            transform=self.test_transform,
            support_size=support_size,
            mode='test',  # 关键：只使用测试集
            seed=seed
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 加载原型（如果需要）
        prototypes = None
        if use_prototypes:
            if prototype_path is None:
                raise ValueError("Prototype path required for PNC")
            
            print(f"📦 Loading prototypes from: {prototype_path}")
            proto_dict = torch.load(prototype_path, map_location=self.device, weights_only=False)
            prototypes = proto_dict['prototypes'].to(self.device)
            
            # 验证数据划分一致性
            if 'metadata' in proto_dict and proto_dict['metadata'].get('data_split') == 'strict_split':
                print(f"✓ Using strict data split - no overlap between support and test")
            else:
                print(f"⚠️ Warning: Prototypes may not use strict split")
            
            print(f"✓ Loaded prototypes: {prototypes.shape}")
            print(f"   • Fusion weight (α): {fusion_alpha:.2f}")
            print(f"   • Temperature (τ): {similarity_tau:.2f}")
        
        # 评估
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        model.eval()
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 模型预测
                outputs = model(images)
                
                if use_prototypes:
                    # 提取特征
                    features = model.backbone(images)
                    
                    # L2归一化
                    features = features / features.norm(2, dim=1, keepdim=True)
                    
                    # 计算与原型的相似度
                    similarities = torch.matmul(features, prototypes.T)
                    
                    if use_pure_ncc:
                        # 纯NCC：只使用原型相似度
                        predicted = similarities.argmax(dim=1)
                        confidences = torch.softmax(similarities, dim=1).max(dim=1)[0]
                    else:
                        # 原始PNC：原型+分类器融合
                        proto_probs = torch.softmax(similarities / similarity_tau, dim=1)
                        class_probs = torch.softmax(outputs, dim=1)
                        final_probs = fusion_alpha * proto_probs + (1 - fusion_alpha) * class_probs
                        predicted = final_probs.argmax(dim=1)
                        confidences = final_probs.max(dim=1)[0]
                else:
                    predicted = outputs.argmax(dim=1)
                    confidences = torch.softmax(outputs, dim=1).max(dim=1)[0]
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 计算指标
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        mean_confidence = np.mean(all_confidences)
        
        return {
            'accuracy': accuracy,
            'confidence': mean_confidence,
            'total_samples': len(all_labels)
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate without data leakage')
    
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--prototype-path', type=str,
                       default='/kaggle/working/improved_prototypes_split.pt')
    parser.add_argument('--support-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fusion-alpha', type=float, default=0.5)
    parser.add_argument('--similarity-tau', type=float, default=0.01)
    parser.add_argument('--use-pure-ncc', action='store_true', 
                       help='Use pure NCC instead of PNC fusion')
    
    args = parser.parse_args()
    
    evaluator = NoLeakEvaluator()
    model = evaluator.load_model(args.model_path)
    
    print("\n" + "="*80)
    print("📊 EVALUATION WITHOUT DATA LEAKAGE")
    print("="*80)
    
    # 基线评估
    baseline_results = evaluator.evaluate(
        model=model,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed,
        use_prototypes=False
    )
    
    # PNC评估
    pnc_results = evaluator.evaluate(
        model=model,
        data_dir=args.data_dir,
        support_size=args.support_size,
        seed=args.seed,
        use_prototypes=True,
        prototype_path=args.prototype_path,
        fusion_alpha=args.fusion_alpha,
        similarity_tau=args.similarity_tau,
        use_pure_ncc=args.use_pure_ncc
    )
    
    # 结果对比
    print("\n" + "="*80)
    print("📈 REAL RESULTS (NO DATA LEAKAGE)")
    print("="*80)
    
    improvement = pnc_results['accuracy'] - baseline_results['accuracy']
    
    comparison_table = [
        ["Metric", "Baseline", "PNC", "Improvement"],
        ["Accuracy", f"{baseline_results['accuracy']:.2%}", 
         f"{pnc_results['accuracy']:.2%}", f"{improvement:+.2%}"],
        ["Confidence", f"{baseline_results['confidence']:.3f}",
         f"{pnc_results['confidence']:.3f}", 
         f"{pnc_results['confidence']-baseline_results['confidence']:+.3f}"],
        ["Test Samples", baseline_results['total_samples'],
         pnc_results['total_samples'], "-"]
    ]
    
    print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))
    
    # 评估结论
    print(f"\n🏁 HONEST ASSESSMENT:")
    if improvement > 0.05:
        print(f"   ✅ Significant improvement: {improvement:+.2%}")
    elif improvement > 0.02:
        print(f"   🟡 Moderate improvement: {improvement:+.2%}")
    elif improvement > 0:
        print(f"   🟠 Slight improvement: {improvement:+.2%}")
    else:
        print(f"   ❌ No improvement: {improvement:+.2%}")
    
    print(f"\n💡 This is the REAL performance without data leakage!")
    print("="*80)


if __name__ == '__main__':
    main()
