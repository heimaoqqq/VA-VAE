#!/usr/bin/env python3
"""
综合域适应分析工具
集成了基本评估和高级分析功能
研究问题：正常步态(源域) → 背包步态(目标域) 的域适应效果
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
from sklearn.manifold import TSNE
from scipy.stats import ttest_rel, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

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
            # 返回零张量作为fallback - 尺寸与实际图像一致
            return torch.zeros(3, 256, 256), label


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
        correct_confidences = [conf for i, conf in enumerate(confidences) 
                             if true_labels[i] == predictions[i]]
        incorrect_confidences = [conf for i, conf in enumerate(confidences) 
                               if true_labels[i] != predictions[i]]
        
        confidence_stats = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'median_confidence': np.median(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'high_confidence_samples': np.sum(np.array(confidences) > 0.8) / len(confidences),
            'low_confidence_samples': np.sum(np.array(confidences) < 0.5) / len(confidences)
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
    
    def _advanced_domain_analysis(self, baseline_model, enhanced_model, target_data_dir, source_data_dir, output_path):
        """高级域适应分析"""
        print("\n🔬 Performing advanced domain analysis...")
        
        # 特征表示分析
        if source_data_dir:
            self._feature_analysis(baseline_model, enhanced_model, source_data_dir, target_data_dir, output_path)
        
        # 置信度分布分析
        self._confidence_distribution_analysis(baseline_model, enhanced_model, target_data_dir, output_path)
        
        # 混淆矩阵对比
        self._confusion_matrix_analysis(baseline_model, enhanced_model, target_data_dir, output_path)
    
    def _feature_analysis(self, baseline_model, enhanced_model, source_dir, target_dir, output_path):
        """特征表示t-SNE分析"""
        print("🎨 Analyzing feature representations...")
        
        def extract_features(model, data_dir, max_samples=500):
            # 创建数据集
            if 'backpack' in str(data_dir).lower() or 'target' in str(data_dir).lower():
                dataset = BackpackWalkingDataset(data_dir, self.test_transform)
            else:
                # 假设源域数据格式相同
                dataset = BackpackWalkingDataset(data_dir, self.test_transform)
            
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            features = []
            labels = []
            model.eval()
            
            with torch.no_grad():
                sample_count = 0
                for images, batch_labels in loader:
                    if sample_count >= max_samples:
                        break
                    
                    images = images.to(self.device)
                    # 提取特征 (倒数第二层)
                    feat = model.backbone(images)
                    if hasattr(model, 'classifier'):
                        feat = model.classifier[:-1](feat)  # 除了最后分类层
                    
                    features.append(feat.cpu().numpy())
                    labels.extend(batch_labels.numpy())
                    sample_count += len(batch_labels)
            
            return np.vstack(features), np.array(labels)
        
        # 提取特征
        try:
            source_feat_bl, source_labels = extract_features(baseline_model, source_dir)
            target_feat_bl, target_labels = extract_features(baseline_model, target_dir)
            target_feat_eh, _ = extract_features(enhanced_model, target_dir)
            
            # t-SNE分析
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for idx, (model_name, target_feat) in enumerate([('Baseline', target_feat_bl), ('Enhanced', target_feat_eh)]):
                # 合并源域和目标域特征
                all_features = np.vstack([source_feat_bl, target_feat])
                domain_labels = np.hstack([np.zeros(len(source_feat_bl)), np.ones(len(target_feat))])
                
                # t-SNE降维
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
                features_2d = tsne.fit_transform(all_features)
                
                # 绘制
                colors = ['blue', 'red']
                domains = ['Source (Normal)', 'Target (Backpack)']
                for i, (color, domain) in enumerate(zip(colors, domains)):
                    mask = domain_labels == i
                    axes[idx].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                    c=color, label=domain, alpha=0.6, s=20)
                
                axes[idx].set_title(f'{model_name} Model - Domain Separation')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'feature_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Feature analysis failed: {e}")
    
    def _confidence_distribution_analysis(self, baseline_model, enhanced_model, target_dir, output_path):
        """置信度分布分析"""
        print("📈 Analyzing confidence distributions...")
        
        def get_confidence_stats(model, data_dir):
            dataset = BackpackWalkingDataset(data_dir, self.test_transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            correct_confidences = []
            incorrect_confidences = []
            
            model.eval()
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    for pred, label, prob in zip(preds, labels, probs):
                        confidence = prob[pred].item()
                        if pred.item() == label.item():
                            correct_confidences.append(confidence)
                        else:
                            incorrect_confidences.append(confidence)
            
            return correct_confidences, incorrect_confidences
        
        try:
            # 获取置信度统计
            bl_correct, bl_incorrect = get_confidence_stats(baseline_model, target_dir)
            eh_correct, eh_incorrect = get_confidence_stats(enhanced_model, target_dir)
            
            # 绘制对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            models = [('Baseline', bl_correct, bl_incorrect), ('Enhanced', eh_correct, eh_incorrect)]
            
            for idx, (name, correct, incorrect) in enumerate(models):
                # 置信度分布
                axes[idx, 0].hist(correct, bins=30, alpha=0.7, label='Correct', color='green', density=True)
                axes[idx, 0].hist(incorrect, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
                axes[idx, 0].set_title(f'{name} - Confidence Distribution')
                axes[idx, 0].set_xlabel('Confidence')
                axes[idx, 0].set_ylabel('Density')
                axes[idx, 0].legend()
                axes[idx, 0].grid(True, alpha=0.3)
                
                # 箱线图
                data = [correct, incorrect] if len(correct) > 0 and len(incorrect) > 0 else [[0.5], [0.5]]
                axes[idx, 1].boxplot(data, labels=['Correct', 'Incorrect'])
                axes[idx, 1].set_title(f'{name} - Confidence Box Plot')
                axes[idx, 1].set_ylabel('Confidence')
                axes[idx, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Confidence analysis failed: {e}")
    
    def _confusion_matrix_analysis(self, baseline_model, enhanced_model, target_dir, output_path):
        """混淆矩阵对比分析"""
        print("🔍 Generating confusion matrices...")
        
        def get_predictions(model, data_dir):
            dataset = BackpackWalkingDataset(data_dir, self.test_transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            all_preds = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            return all_labels, all_preds
        
        try:
            # 获取预测结果
            bl_labels, bl_preds = get_predictions(baseline_model, target_dir)
            eh_labels, eh_preds = get_predictions(enhanced_model, target_dir)
            
            # 生成混淆矩阵
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for idx, (name, labels, preds) in enumerate([('Baseline', bl_labels, bl_preds), ('Enhanced', eh_labels, eh_preds)]):
                cm = confusion_matrix(labels, preds)
                cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
                
                im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
                axes[idx].set_title(f'{name} Model - Normalized Confusion Matrix')
                
                # 添加数值标签 (只显示部分，避免过于密集)
                if cm.shape[0] <= 10:
                    thresh = cm_normalized.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            axes[idx].text(j, i, f'{cm_normalized[i, j]:.2f}',
                                         ha="center", va="center",
                                         color="white" if cm_normalized[i, j] > thresh else "black")
                
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Confusion matrix analysis failed: {e}")
    
    def _generate_comprehensive_report(self, baseline_results, enhanced_results, output_path):
        """生成综合域适应报告"""
        improvement = enhanced_results['accuracy'] - baseline_results['accuracy']
        relative_improvement = (improvement / baseline_results['accuracy']) * 100
        
        # 统计显著性检验 (用户级别)
        baseline_user_acc = list(baseline_results['user_accuracies'].values())
        enhanced_user_acc = list(enhanced_results['user_accuracies'].values())
        
        if len(baseline_user_acc) == len(enhanced_user_acc) and len(baseline_user_acc) > 1:
            t_stat, p_value = ttest_rel(enhanced_user_acc, baseline_user_acc)
            significant = p_value < 0.05
        else:
            t_stat, p_value, significant = 0, 1.0, False
        
        # 评估成功程度
        if improvement > 0.05 and relative_improvement > 5 and significant:
            assessment = {"level": "HIGHLY_SUCCESSFUL", "emoji": "🟢"}
        elif improvement > 0.02 and relative_improvement > 2 and significant:
            assessment = {"level": "MODERATELY_SUCCESSFUL", "emoji": "🟡"}
        elif improvement > 0:
            assessment = {"level": "MARGINALLY_SUCCESSFUL", "emoji": "🟠"}
        else:
            assessment = {"level": "NOT_SUCCESSFUL", "emoji": "🔴"}
        
        # 生成报告
        report = {
            "experiment_summary": {
                "research_question": "Can synthetic normal gait data improve recognition of backpack gait?",
                "domain_adaptation": "Normal Gait (Source) → Backpack Gait (Target)"
            },
            "performance_metrics": {
                "baseline_accuracy": float(baseline_results['accuracy']),
                "enhanced_accuracy": float(enhanced_results['accuracy']),
                "absolute_improvement": float(improvement),
                "relative_improvement_percent": float(relative_improvement)
            },
            "statistical_validation": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "statistically_significant": significant
            },
            "assessment": assessment,
            "recommendations": self._generate_recommendations(improvement, relative_improvement, significant)
        }
        
        # 保存并打印报告
        with open(output_path / 'comprehensive_domain_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self._print_comprehensive_summary(report)
        return report
    
    def _generate_recommendations(self, improvement, relative_improvement, significant):
        """生成建议"""
        recommendations = []
        
        if significant and improvement > 0.02:
            recommendations.append("✅ Deploy synthetic data augmentation in production")
            recommendations.append("✅ Focus data collection on normal gait patterns only")
        else:
            recommendations.append("⚠️ Investigate improved generation quality")
            recommendations.append("⚠️ Consider alternative domain adaptation techniques")
        
        if relative_improvement < 5:
            recommendations.append("📈 Generate more diverse synthetic samples")
            recommendations.append("🔬 Explore advanced domain adaptation methods")
        
        recommendations.append("🧪 Validate on additional gait variations")
        recommendations.append("💰 Perform cost-benefit analysis vs. real data collection")
        
        return recommendations
    
    def _print_comprehensive_summary(self, report):
        """打印综合摘要"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE DOMAIN ADAPTATION ANALYSIS")
        print("="*80)
        
        metrics = report['performance_metrics']
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   • Baseline Accuracy:        {metrics['baseline_accuracy']:.1%}")
        print(f"   • Enhanced Accuracy:        {metrics['enhanced_accuracy']:.1%}")
        print(f"   • Absolute Improvement:     {metrics['absolute_improvement']:+.1%}")
        print(f"   • Relative Improvement:     {metrics['relative_improvement_percent']:+.1f}%")
        
        stats = report['statistical_validation']
        print(f"\n📈 STATISTICAL VALIDATION:")
        print(f"   • P-value:                  {stats['p_value']:.4f}")
        print(f"   • Statistically Significant: {'Yes' if stats['statistically_significant'] else 'No'}")
        
        assessment = report['assessment']
        print(f"\n🏆 ASSESSMENT: {assessment['emoji']} {assessment['level']}")
        
        print(f"\n🚀 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        print("="*80)

    def save_detailed_results(self, results, output_path):
        """保存详细的评估结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换numpy int64键为Python int以支持JSON序列化
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k) if hasattr(k, 'item') else k: convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_keys(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
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
    
    # 输出参数
    parser.add_argument('--output_dir', default='./cross_domain_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CrossDomainEvaluator()
    
    # 加载并评估基线模型
    print("📊 Loading and evaluating baseline model...")
    baseline_model, _ = evaluator.load_classifier(args.baseline_model)
    baseline_results = evaluator.evaluate_on_target_domain(baseline_model, args.backpack_data_dir)
    
    # 加载并评估增强模型
    print("📊 Loading and evaluating enhanced model...")
    enhanced_model, _ = evaluator.load_classifier(args.enhanced_model)
    enhanced_results = evaluator.evaluate_on_target_domain(enhanced_model, args.backpack_data_dir)
    
    if baseline_results is None or enhanced_results is None:
        print("❌ Evaluation failed")
        return
    
    # 高级分析
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    evaluator._advanced_domain_analysis(baseline_model, enhanced_model, 
                                       args.backpack_data_dir, None, output_path)
    
    # 生成综合报告
    report = evaluator._generate_comprehensive_report(baseline_results, enhanced_results, output_path)
    
    # 保存详细结果
    all_results = {
        'baseline_results': baseline_results,
        'enhanced_results': enhanced_results,
        'comprehensive_report': report,
        'evaluation_config': vars(args)
    }
    
    evaluator.save_detailed_results(all_results, output_path / 'cross_domain_evaluation.json')
    
    print(f"\n✅ Cross-domain evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
