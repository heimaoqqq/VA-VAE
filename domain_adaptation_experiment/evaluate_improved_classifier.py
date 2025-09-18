#!/usr/bin/env python3
"""
使用ImprovedClassifier的原型校准（PNC）评估
适配improved_classifier_training.py训练的分类器
"""

import torch
import argparse
from pathlib import Path
import json
import numpy as np
from tabulate import tabulate
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 导入ImprovedClassifier
from improved_classifier_training import ImprovedClassifier
from cross_domain_evaluator import CrossDomainEvaluator, BackpackWalkingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms


class ImprovedClassifierEvaluator:
    """专用于ImprovedClassifier的原型校准评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_improved_classifier(self, model_path):
        """加载ImprovedClassifier"""
        print(f"📦 Loading ImprovedClassifier from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取模型配置
        num_classes = checkpoint.get('num_classes', 31)
        backbone = checkpoint.get('backbone', 'resnet18')
        
        # 创建模型
        model = ImprovedClassifier(
            num_classes=num_classes,
            backbone=backbone
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print(f"✅ Loaded {backbone} with {num_classes} classes")
        if 'best_val_acc' in checkpoint:
            print(f"   Original validation accuracy: {checkpoint['best_val_acc']:.2f}%")
            
        return model
    
    def evaluate_on_target_domain(self, model, target_data_dir, batch_size=32, 
                                  use_prototypes=False, prototype_path=None,
                                  fusion_alpha=0.4, similarity_tau=0.1):
        """在目标域评估ImprovedClassifier"""
        mode = "with Prototype Calibration" if use_prototypes else "Baseline"
        print(f"\n🎯 Evaluating ImprovedClassifier ({mode}): {target_data_dir}")
        
        # 创建数据集
        target_dataset = BackpackWalkingDataset(
            data_dir=target_data_dir,
            transform=self.test_transform
        )
        
        if len(target_dataset) == 0:
            print("❌ No target domain samples found!")
            return None
        
        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 加载原型（如果启用）
        prototypes = None
        if use_prototypes:
            if prototype_path is None:
                raise ValueError("Prototype path must be provided when use_prototypes=True")
            
            print(f"📦 Loading prototypes from: {prototype_path}")
            proto_dict = torch.load(prototype_path, map_location=self.device, weights_only=False)
            prototypes = proto_dict['prototypes'].to(self.device)
            print(f"✓ Loaded prototypes: {prototypes.shape}")
            print(f"   • Fusion weight (α): {fusion_alpha:.2f}")
            print(f"   • Temperature (τ): {similarity_tau:.2f}")
        
        # 评估
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(target_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # ImprovedClassifier前向传播
                outputs = model(images)
                
                if use_prototypes:
                    # 提取特征：backbone输出
                    features = model.backbone(images)
                    
                    # L2归一化特征
                    features = features / features.norm(2, dim=1, keepdim=True)
                    
                    # 计算与原型的余弦相似度
                    similarities = torch.matmul(features, prototypes.T)
                    
                    # 应用温度缩放并转换为概率
                    proto_probs = torch.softmax(similarities / similarity_tau, dim=1)
                    
                    # 原始分类概率
                    class_probs = torch.softmax(outputs, dim=1)
                    
                    # 融合两种概率分布
                    probabilities = (1 - fusion_alpha) * class_probs + fusion_alpha * proto_probs
                else:
                    # 基线方法：仅使用分类器输出
                    probabilities = torch.softmax(outputs, dim=1)
                
                # 获取预测和置信度
                confidences, predictions = torch.max(probabilities, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 计算评估指标（使用CrossDomainEvaluator的方法）
        evaluator = CrossDomainEvaluator()
        results = evaluator._compute_metrics(all_labels, all_predictions, all_confidences)
        results['total_samples'] = len(all_labels)
        results['num_users'] = len(set(all_labels))
        
        return results


def build_prototypes_for_improved_classifier(model_path, data_dir, output_path, support_size=15):
    """为ImprovedClassifier构建原型"""
    print("\n" + "="*80)
    print("🔧 BUILDING PROTOTYPES FOR IMPROVEDCLASSIFIER")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载ImprovedClassifier
    evaluator = ImprovedClassifierEvaluator(device)
    model = evaluator.load_improved_classifier(model_path)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 从build_target_prototypes.py导入数据集
    sys.path.append(str(Path(__file__).parent))
    from build_target_prototypes import TargetDomainDataset
    
    # 创建支持集
    dataset = TargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"\n📊 Support set statistics:")
    print(f"   • Total samples: {len(dataset)}")
    print(f"   • Samples per user: {support_size}")
    
    # 提取特征
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels, batch_users in tqdm(dataloader, desc="Extracting features"):
            batch_images = batch_images.to(device)
            
            # 使用backbone提取特征（ImprovedClassifier结构）
            feat = model.backbone(batch_images)
            
            features.append(feat.cpu())
            labels.extend(batch_labels.tolist())
    
    features = torch.cat(features, dim=0)
    print(f"✓ Extracted features: {features.shape}")
    
    # 计算原型
    num_classes = max(labels) + 1
    prototypes = []
    
    for class_id in range(num_classes):
        class_mask = [i for i, l in enumerate(labels) if l == class_id]
        if len(class_mask) == 0:
            prototypes.append(torch.zeros(features.shape[1]))
            continue
        
        class_features = features[class_mask]
        prototype = class_features.mean(dim=0)
        
        # L2归一化
        prototype = prototype / prototype.norm(2)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    print(f"✓ Computed prototypes: {prototypes.shape}")
    
    # 保存原型
    save_dict = {
        'prototypes': prototypes,
        'user_ids': list(range(num_classes)),
        'feature_dim': prototypes.shape[1],
        'metadata': {
            'model_type': 'ImprovedClassifier',
            'support_size': support_size,
            'feature_extraction': 'backbone_output'
        }
    }
    
    torch.save(save_dict, output_path)
    print(f"💾 ImprovedClassifier prototypes saved to: {output_path}")
    
    return save_dict


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate ImprovedClassifier with PNC')
    
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/input/best-classifier-model/best_model.pth',
                       help='Path to ImprovedClassifier model')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data')
    parser.add_argument('--prototype-path', type=str,
                       default='/kaggle/working/improved_prototypes.pt',
                       help='Path to prototypes')
    parser.add_argument('--support-size', type=int, default=15,
                       help='Support set size for prototype building')
    parser.add_argument('--build-prototypes', action='store_true',
                       help='Build prototypes first')
    parser.add_argument('--fusion-alpha', type=float, default=0.4,
                       help='Prototype fusion weight')
    parser.add_argument('--similarity-tau', type=float, default=0.1,
                       help='Similarity temperature')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # 构建原型（如果需要）
    if args.build_prototypes:
        build_prototypes_for_improved_classifier(
            args.model_path, args.data_dir, args.prototype_path, args.support_size
        )
        print("\n✅ Prototype building completed!")
    
    # 检查原型文件
    if not Path(args.prototype_path).exists():
        print(f"⚠️ Prototype file not found: {args.prototype_path}")
        print("Please run with --build-prototypes first")
        return
    
    # 创建评估器
    evaluator = ImprovedClassifierEvaluator()
    
    # 加载模型
    model = evaluator.load_improved_classifier(args.model_path)
    
    print("\n" + "="*80)
    print("📊 IMPROVEDCLASSIFIER BASELINE vs PNC COMPARISON")
    print("="*80)
    
    # 基线评估
    print("\n📊 BASELINE EVALUATION")
    baseline_results = evaluator.evaluate_on_target_domain(
        model=model,
        target_data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_prototypes=False
    )
    
    # PNC评估
    print("\n🎯 PROTOTYPE CALIBRATION EVALUATION")
    pnc_results = evaluator.evaluate_on_target_domain(
        model=model,
        target_data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_prototypes=True,
        prototype_path=args.prototype_path,
        fusion_alpha=args.fusion_alpha,
        similarity_tau=args.similarity_tau
    )
    
    # 结果对比
    if baseline_results and pnc_results:
        baseline_acc = baseline_results['overall_accuracy']
        pnc_acc = pnc_results['overall_accuracy']
        improvement = pnc_acc - baseline_acc
        
        print("\n" + "="*80)
        print("📈 COMPARISON RESULTS")
        print("="*80)
        
        comparison_table = [
            ["Metric", "Baseline", "PNC", "Improvement"],
            ["Overall Accuracy", f"{baseline_acc:.2%}", f"{pnc_acc:.2%}", f"{improvement:+.2%}"],
            ["Mean Confidence", 
             f"{baseline_results['confidence_stats']['mean_confidence']:.3f}",
             f"{pnc_results['confidence_stats']['mean_confidence']:.3f}",
             f"{pnc_results['confidence_stats']['mean_confidence'] - baseline_results['confidence_stats']['mean_confidence']:+.3f}"]
        ]
        
        print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))
        
        # 评估结论
        if improvement > 0.05:
            assessment = "🟢 SIGNIFICANT IMPROVEMENT"
        elif improvement > 0.02:
            assessment = "🟡 MODERATE IMPROVEMENT"
        elif improvement > 0:
            assessment = "🟠 SLIGHT IMPROVEMENT"
        else:
            assessment = "🔴 NO IMPROVEMENT"
        
        print(f"\n🏆 ASSESSMENT: {assessment}")
        print(f"💡 Improvement: {improvement:+.2%} on ImprovedClassifier baseline")
        
        # 预测分析
        if baseline_acc >= 0.75:
            print(f"🎯 EXCELLENT! Starting from {baseline_acc:.1%} baseline, PNC achieved {pnc_acc:.1%}")
            print("   This demonstrates strong domain adaptation capabilities!")
    
    print("="*80)


if __name__ == '__main__':
    main()
