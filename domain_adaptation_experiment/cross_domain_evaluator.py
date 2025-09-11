#!/usr/bin/env python3
"""
跨域评估脚本
测试分类器在背包行走数据（目标域）上的识别性能
用于比较基线分类器 vs 生成数据增强分类器的域泛化能力
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 添加父目录到路径以导入分类器
import sys
sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier


class BackpackWalkingDataset(Dataset):
    """背包行走数据集（目标域）"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 加载背包行走数据
        self._load_backpack_data()
        
    def _load_backpack_data(self):
        """加载背包行走数据"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"背包行走数据目录不存在: {self.data_dir}")
        
        # 查找用户目录 (ID_* 或 User_*)
        user_dirs = list(self.data_dir.glob("ID_*")) + list(self.data_dir.glob("User_*"))
        
        if not user_dirs:
            raise ValueError(f"在 {self.data_dir} 中未找到用户目录")
        
        for user_dir in sorted(user_dirs):
            if user_dir.is_dir():
                # 解析用户ID
                if user_dir.name.startswith("ID_"):
                    user_id = int(user_dir.name.split('_')[1]) - 1  # 转换为0-based
                elif user_dir.name.startswith("User_"):
                    user_id = int(user_dir.name.split('_')[1])
                else:
                    continue
                    
                # 加载该用户的所有背包行走图像
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in user_dir.glob(ext):
                        self.samples.append((str(img_path), user_id))
        
        print(f"Loaded {len(self.samples)} backpack walking samples from {len(set(s[1] for s in self.samples))} users")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回零张量作为fallback
            return torch.zeros(3, 224, 224), label


class CrossDomainEvaluator:
    """跨域性能评估器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 测试数据变换（与训练时验证集一致）
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_classifier(self, model_path):
        """加载训练好的分类器"""
        print(f"Loading classifier from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        num_classes = checkpoint.get('num_classes', 31)
        model_name = checkpoint.get('model_name', 'resnet18')
        
        # 创建模型
        model = ImprovedClassifier(
            num_classes=num_classes,
            backbone=model_name
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded {model_name} with {num_classes} classes")
        if 'best_val_acc' in checkpoint:
            print(f"   Original validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        return model, checkpoint
    
    def evaluate_on_target_domain(self, model, target_data_dir, batch_size=32):
        """在目标域（背包行走）数据上评估模型"""
        print(f"\n🎯 Evaluating on target domain: {target_data_dir}")
        
        # 创建目标域数据集
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
        
        # 评估
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(target_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 获取预测和置信度
                confidences, predictions = torch.max(probabilities, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 计算评估指标
        results = self._compute_metrics(all_labels, all_predictions, all_confidences)
        results['total_samples'] = len(all_labels)
        results['num_users'] = len(set(all_labels))
        
        return results
    
    def _compute_metrics(self, true_labels, predictions, confidences):
        """计算详细的评估指标"""
        # 基本准确率
        accuracy = accuracy_score(true_labels, predictions)
        
        # 按用户统计
        user_accuracies = {}
        for user_id in set(true_labels):
            user_mask = np.array(true_labels) == user_id
            if np.sum(user_mask) > 0:
                user_acc = accuracy_score(
                    np.array(true_labels)[user_mask], 
                    np.array(predictions)[user_mask]
                )
                user_accuracies[user_id] = user_acc
        
        # 置信度统计
        confidence_stats = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'correct_confidence': np.mean([conf for i, conf in enumerate(confidences) 
                                         if true_labels[i] == predictions[i]]),
            'incorrect_confidence': np.mean([conf for i, conf in enumerate(confidences) 
                                           if true_labels[i] != predictions[i]])
        }
        
        return {
            'overall_accuracy': accuracy,
            'user_accuracies': user_accuracies,
            'mean_user_accuracy': np.mean(list(user_accuracies.values())),
            'std_user_accuracy': np.std(list(user_accuracies.values())),
            'confidence_stats': confidence_stats,
            'classification_report': classification_report(
                true_labels, predictions, output_dict=True, zero_division=0
            )
        }
    
    def compare_models(self, baseline_results, enhanced_results):
        """比较基线模型和增强模型的性能"""
        print("\n" + "="*80)
        print("🔍 CROSS-DOMAIN GENERALIZATION COMPARISON")
        print("="*80)
        
        # 总体性能对比
        baseline_acc = baseline_results['overall_accuracy']
        enhanced_acc = enhanced_results['overall_accuracy']
        improvement = enhanced_acc - baseline_acc
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Baseline (real data only):     {baseline_acc:.1%}")
        print(f"   • Enhanced (real + generated):   {enhanced_acc:.1%}")
        print(f"   • Improvement:                   {improvement:+.1%}")
        print(f"   • Relative improvement:          {improvement/baseline_acc:+.1%}")
        
        # 用户级别性能对比
        baseline_user_acc = baseline_results['mean_user_accuracy']
        enhanced_user_acc = enhanced_results['mean_user_accuracy']
        user_improvement = enhanced_user_acc - baseline_user_acc
        
        print(f"\n👤 USER-LEVEL PERFORMANCE:")
        print(f"   • Baseline mean user accuracy:   {baseline_user_acc:.1%}")
        print(f"   • Enhanced mean user accuracy:   {enhanced_user_acc:.1%}")
        print(f"   • Mean improvement per user:     {user_improvement:+.1%}")
        
        # 置信度对比
        baseline_conf = baseline_results['confidence_stats']['mean_confidence']
        enhanced_conf = enhanced_results['confidence_stats']['mean_confidence']
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   • Baseline mean confidence:      {baseline_conf:.3f}")
        print(f"   • Enhanced mean confidence:      {enhanced_conf:.3f}")
        print(f"   • Confidence change:             {enhanced_conf-baseline_conf:+.3f}")
        
        # 结果解释
        if improvement > 0.05:  # 5%以上提升
            grade = "🟢 SIGNIFICANT IMPROVEMENT"
            interpretation = "Generated data substantially improves cross-domain generalization"
        elif improvement > 0.02:  # 2%以上提升
            grade = "🟡 MODERATE IMPROVEMENT" 
            interpretation = "Generated data provides modest cross-domain benefits"
        elif improvement > 0:
            grade = "🟠 SLIGHT IMPROVEMENT"
            interpretation = "Generated data provides minimal cross-domain benefits"
        else:
            grade = "🔴 NO IMPROVEMENT"
            interpretation = "Generated data does not improve cross-domain performance"
        
        print(f"\n🏆 DOMAIN ADAPTATION ASSESSMENT: {grade}")
        print(f"💡 INTERPRETATION: {interpretation}")
        print("="*80)
        
        return {
            'baseline_accuracy': baseline_acc,
            'enhanced_accuracy': enhanced_acc,
            'absolute_improvement': improvement,
            'relative_improvement': improvement / baseline_acc,
            'grade': grade,
            'interpretation': interpretation
        }
    
    def save_detailed_results(self, results, output_path):
        """保存详细的评估结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cross-domain evaluation on backpack walking data')
    
    # 模型参数
    parser.add_argument('--baseline_model', required=True, 
                       help='Path to baseline classifier (trained on real data only)')
    parser.add_argument('--enhanced_model', required=True,
                       help='Path to enhanced classifier (trained on real + generated data)')
    
    # 数据参数
    parser.add_argument('--backpack_data_dir', required=True,
                       help='Directory containing backpack walking data')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='Evaluation batch size')
    
    # 输出参数
    parser.add_argument('--output_dir', default='./cross_domain_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CrossDomainEvaluator()
    
    # 评估基线模型
    print("🔵 Evaluating BASELINE classifier...")
    baseline_model, _ = evaluator.load_classifier(args.baseline_model)
    baseline_results = evaluator.evaluate_on_target_domain(
        baseline_model, args.backpack_data_dir, args.batch_size
    )
    
    # 评估增强模型  
    print("\n🟢 Evaluating ENHANCED classifier...")
    enhanced_model, _ = evaluator.load_classifier(args.enhanced_model)
    enhanced_results = evaluator.evaluate_on_target_domain(
        enhanced_model, args.backpack_data_dir, args.batch_size
    )
    
    # 比较结果
    comparison = evaluator.compare_models(baseline_results, enhanced_results)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    all_results = {
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'comparison': comparison,
        'evaluation_config': vars(args)
    }
    
    evaluator.save_detailed_results(all_results, output_dir / 'cross_domain_evaluation.json')
    
    print(f"\n✅ Cross-domain evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
