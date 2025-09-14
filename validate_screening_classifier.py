"""
验证筛选分类器的可靠性
评估只在真实数据训练的分类器是否能胜任合成数据筛选工作
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import entropy
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

class ScreeningClassifierValidator:
    """筛选分类器可靠性验证器"""
    
    def __init__(self, classifier, device):
        self.classifier = classifier
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_calibration(self, test_data_dir):
        """
        评估分类器的概率校准度
        良好校准的分类器：当它说90%置信度时，真的有90%准确率
        """
        print("\n📊 评估分类器校准度...")
        
        all_probs = []
        all_labels = []
        
        # 收集预测概率和真实标签
        total_samples = 0
        correct_predictions = 0
        prediction_samples = []
        
        for class_id in range(31):  # class_id: 0-30
            # 真实数据格式：ID_1到ID_31，jpg格式
            # ID_1对应class_id=0, ID_2对应class_id=1, ..., ID_31对应class_id=30
            user_dir = Path(test_data_dir) / f"ID_{class_id + 1}"
            if not user_dir.exists():
                print(f"   ⚠️ 目录不存在: {user_dir}")
                continue
            
            user_samples = 0
            user_correct = 0
            
            for img_file in list(user_dir.glob("*.jpg"))[:50]:  # 每用户取50个样本
                img = Image.open(img_file).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.classifier(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    
                    # 多分类校准：用最大概率（模型置信度）
                    max_prob, pred = torch.max(prob, dim=1)  # 最高置信度和预测类别
                    pred_label = pred.item()
                    is_correct = (pred_label == class_id)
                    
                    all_probs.append(max_prob.item())  # 用最大概率做校准
                    all_labels.append(int(is_correct))
                    
                    # 统计
                    total_samples += 1
                    user_samples += 1
                    if is_correct:
                        correct_predictions += 1
                        user_correct += 1
                    
                    # 记录前几个样本用于调试
                    if len(prediction_samples) < 10:
                        prediction_samples.append({
                            'class_id': class_id,
                            'file': img_file.name,
                            'predicted': pred_label,
                            'confidence': max_prob.item(),
                            'correct': is_correct
                        })
            
            if user_samples > 0 and class_id < 5:  # 只显示前5个用户
                user_acc = user_correct / user_samples
                print(f"   ID_{class_id + 1}: {user_samples}样本, 准确率={user_acc:.3f}")
        
        # 打印调试信息
        overall_acc = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"   📈 总样本数: {total_samples}")
        print(f"   📈 总体准确率: {overall_acc:.3f}")
        print(f"   📈 前10个预测样例:")
        
        for sample in prediction_samples:
            print(f"      ID_{sample['class_id']+1}/{sample['file']}: 预测={sample['predicted']}, 置信度={sample['confidence']:.3f}, 正确={sample['correct']}")
        
        # 计算校准曲线
        fraction_positives, mean_predicted = calibration_curve(
            all_labels, all_probs, n_bins=10
        )
        
        # 计算ECE (Expected Calibration Error)
        ece = np.abs(fraction_positives - mean_predicted).mean()
        
        # Brier Score (越小越好，0是完美校准)
        brier = brier_score_loss(all_labels, all_probs)
        
        return {
            'ece': ece,
            'brier_score': brier,
            'calibration_curve': (mean_predicted, fraction_positives),
            'is_well_calibrated': ece < 0.1  # ECE < 0.1认为校准良好
        }
    
    def evaluate_confidence_distribution(self, real_data_dir, synthetic_data_dir):
        """
        比较分类器在真实和合成数据上的置信度分布
        如果分布差异过大，说明分类器不适合筛选
        """
        print("\n📊 比较置信度分布...")
        
        real_confidences = self._get_confidence_distribution(real_data_dir, "真实数据")
        synthetic_confidences = self._get_confidence_distribution(synthetic_data_dir, "合成数据")
        
        # 计算KL散度
        # 将置信度离散化到bins
        bins = np.linspace(0, 1, 21)
        real_hist, _ = np.histogram(real_confidences, bins, density=True)
        synth_hist, _ = np.histogram(synthetic_confidences, bins, density=True)
        
        # 避免0值
        real_hist = real_hist + 1e-10
        synth_hist = synth_hist + 1e-10
        
        # KL散度
        kl_divergence = entropy(real_hist, synth_hist)
        
        # 均值和标准差
        real_mean, real_std = np.mean(real_confidences), np.std(real_confidences)
        synth_mean, synth_std = np.mean(synthetic_confidences), np.std(synthetic_confidences)
        
        return {
            'real_confidence': {'mean': real_mean, 'std': real_std},
            'synthetic_confidence': {'mean': synth_mean, 'std': synth_std},
            'kl_divergence': kl_divergence,
            'distribution_shift': abs(real_mean - synth_mean),
            'is_distribution_similar': kl_divergence < 0.5  # KL < 0.5认为分布相似
        }
    
    def _get_confidence_distribution(self, data_dir, desc=""):
        """获取数据集的置信度分布"""
        confidences = []
        
        # 判断是真实数据还是合成数据
        is_real_data = "真实" in desc
        
        for class_id in range(31):  # class_id: 0-30
            if is_real_data:
                # 真实数据格式：ID_1到ID_31，jpg格式
                # ID_1对应class_id=0, ID_2对应class_id=1, ...
                user_dir = Path(data_dir) / f"ID_{class_id + 1}"
                file_pattern = "*.jpg"
            else:
                # 合成数据格式：User_00到User_30，png格式
                # User_00对应class_id=0, User_01对应class_id=1, ...
                user_dir = Path(data_dir) / f"User_{class_id:02d}"
                file_pattern = "*.png"
            
            if not user_dir.exists():
                continue
            
            for img_file in tqdm(list(user_dir.glob(file_pattern))[:50], 
                               desc=f"处理{desc}", leave=False):
                img = Image.open(img_file).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.classifier(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    max_prob, _ = torch.max(prob, dim=1)
                    confidences.append(max_prob.item())
        
        return np.array(confidences)
    
    def evaluate_decision_boundary_stability(self, synthetic_samples, num_augmentations=10):
        """
        测试决策边界稳定性
        对合成样本进行轻微扰动，看预测是否稳定
        """
        print("\n📊 评估决策边界稳定性...")
        
        stability_scores = []
        
        augment = transforms.Compose([
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05)
        ])
        
        for sample_path in synthetic_samples[:100]:  # 测试100个样本
            img = Image.open(sample_path).convert('RGB')
            
            predictions = []
            confidences = []
            
            for _ in range(num_augmentations):
                # 应用轻微增强
                aug_img = augment(img)
                img_tensor = self.transform(aug_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.classifier(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf, pred = torch.max(prob, dim=1)
                    
                    predictions.append(pred.item())
                    confidences.append(conf.item())
            
            # 计算稳定性（预测一致性）
            unique_preds = len(set(predictions))
            stability = 1.0 / unique_preds if unique_preds > 0 else 0
            stability_scores.append(stability)
        
        avg_stability = np.mean(stability_scores)
        
        return {
            'average_stability': avg_stability,
            'stable_samples_ratio': np.mean(np.array(stability_scores) == 1.0),
            'is_stable': avg_stability > 0.8  # 80%以上稳定认为可靠
        }
    
    def evaluate_error_analysis(self, synthetic_data_dir):
        """
        错误分析：分析分类器在合成数据上的错误模式
        """
        print("\n📊 分析错误模式...")
        
        error_patterns = {
            'low_confidence_errors': [],  # 低置信度错误
            'high_confidence_errors': [],  # 高置信度错误
            'confusion_pairs': {}  # 易混淆的用户对
        }
        
        for class_id in range(31):  # class_id: 0-30
            # 合成数据格式：User_00到User_30，png格式
            # User_00对应class_id=0, User_01对应class_id=1, ...
            user_dir = Path(synthetic_data_dir) / f"User_{class_id:02d}"
            if not user_dir.exists():
                continue
            
            for img_file in list(user_dir.glob("*.png"))[:50]:
                img = Image.open(img_file).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.classifier(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf, pred = torch.max(prob, dim=1)
                    
                    if pred.item() != class_id:
                        # 记录错误
                        error_info = {
                            'true_label': class_id,
                            'predicted': pred.item(),
                            'confidence': conf.item()
                        }
                        
                        if conf.item() < 0.5:
                            error_patterns['low_confidence_errors'].append(error_info)
                        else:
                            error_patterns['high_confidence_errors'].append(error_info)
                        
                        # 记录混淆对
                        pair = (class_id, pred.item())
                        if pair not in error_patterns['confusion_pairs']:
                            error_patterns['confusion_pairs'][pair] = 0
                        error_patterns['confusion_pairs'][pair] += 1
        
        # 分析结果
        total_errors = len(error_patterns['low_confidence_errors']) + \
                      len(error_patterns['high_confidence_errors'])
        
        if total_errors > 0:
            low_conf_ratio = len(error_patterns['low_confidence_errors']) / total_errors
        else:
            low_conf_ratio = 0
        
        return {
            'total_errors': total_errors,
            'low_confidence_error_ratio': low_conf_ratio,
            'high_confidence_errors_count': len(error_patterns['high_confidence_errors']),
            'top_confusion_pairs': sorted(
                error_patterns['confusion_pairs'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'is_error_pattern_acceptable': low_conf_ratio > 0.7  # 70%错误是低置信度
        }
    
    def generate_reliability_report(self, results, output_path):
        """生成可靠性评估报告"""
        
        report = {
            'overall_reliability': self._compute_overall_reliability(results),
            'calibration': results['calibration'],
            'distribution_similarity': results['distribution'],
            'decision_stability': results['stability'],
            'error_analysis': results['errors'],
            'recommendations': self._generate_recommendations(results)
        }
        
        # 保存JSON报告
        with open(output_path / 'screening_classifier_reliability.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成可视化
        self._plot_reliability_metrics(results, output_path)
        
        return report
    
    def _compute_overall_reliability(self, results):
        """计算总体可靠性分数"""
        scores = [
            results['calibration']['is_well_calibrated'] * 25,
            results['distribution']['is_distribution_similar'] * 25,
            results['stability']['is_stable'] * 25,
            results['errors']['is_error_pattern_acceptable'] * 25
        ]
        
        total_score = sum(scores)
        
        return {
            'score': total_score,
            'rating': 'Excellent' if total_score >= 80 else 
                     'Good' if total_score >= 60 else 
                     'Fair' if total_score >= 40 else 'Poor',
            'is_reliable': total_score >= 60
        }
    
    def _generate_recommendations(self, results):
        """基于评估结果生成建议"""
        recommendations = []
        
        if not results['calibration']['is_well_calibrated']:
            recommendations.append("分类器需要概率校准（如温度缩放）")
        
        if not results['distribution']['is_distribution_similar']:
            recommendations.append("真实/合成数据置信度分布差异大，考虑域适应技术")
        
        if not results['stability']['is_stable']:
            recommendations.append("决策边界不稳定，增加数据增强或正则化")
        
        if not results['errors']['is_error_pattern_acceptable']:
            recommendations.append("高置信度错误过多，需要重新训练或调整阈值")
        
        if results['distribution']['synthetic_confidence']['mean'] < 0.5:
            recommendations.append("合成数据整体置信度过低，考虑提高生成质量")
        
        return recommendations
    
    def _plot_reliability_metrics(self, results, output_path):
        """绘制可靠性指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 校准曲线
        mean_pred, fraction_pos = results['calibration']['calibration_curve']
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='完美校准')
        axes[0, 0].plot(mean_pred, fraction_pos, 'b-', label='实际校准')
        axes[0, 0].set_xlabel('平均预测概率')
        axes[0, 0].set_ylabel('实际准确率')
        axes[0, 0].set_title(f'校准曲线 (ECE={results["calibration"]["ece"]:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 置信度分布对比
        categories = ['真实数据', '合成数据']
        means = [results['distribution']['real_confidence']['mean'],
                results['distribution']['synthetic_confidence']['mean']]
        stds = [results['distribution']['real_confidence']['std'],
               results['distribution']['synthetic_confidence']['std']]
        
        x = np.arange(len(categories))
        axes[0, 1].bar(x, means, yerr=stds, capsize=5, color=['blue', 'orange'])
        axes[0, 1].set_ylabel('平均置信度')
        axes[0, 1].set_title(f'置信度分布 (KL={results["distribution"]["kl_divergence"]:.3f})')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 稳定性评分
        stability_data = [
            results['stability']['average_stability'],
            results['stability']['stable_samples_ratio']
        ]
        labels = ['平均稳定性', '稳定样本比例']
        
        axes[1, 0].barh(labels, stability_data, color=['green', 'lightgreen'])
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_xlabel('分数')
        axes[1, 0].set_title('决策边界稳定性')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. 错误模式分析
        if results['errors']['total_errors'] > 0:
            sizes = [
                len(results['errors']['low_confidence_errors']),
                len(results['errors']['high_confidence_errors'])
            ]
            labels = ['低置信度错误', '高置信度错误']
            colors = ['lightblue', 'lightcoral']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('错误类型分布')
        else:
            axes[1, 1].text(0.5, 0.5, '无错误', ha='center', va='center', fontsize=16)
            axes[1, 1].set_title('错误类型分布')
        
        plt.tight_layout()
        plt.savefig(output_path / 'screening_reliability_metrics.png', dpi=150)
        plt.close()


def load_classifier(checkpoint_path, device):
    """加载分类器模型"""
    import torchvision.models as models
    
    # 创建与improved_classifier_training.py完全一致的模型结构
    class MicroDopplerModel(nn.Module):
        def __init__(self, num_classes=31, dropout_rate=0.3):
            super().__init__()
            
            # 使用ResNet18作为backbone
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            # 对比学习投影头
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 64)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # 创建模型
    model = MicroDopplerModel(num_classes=31)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 分类器加载完成: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Validate screening classifier reliability')
    parser.add_argument('--classifier_path', type=str, required=True,
                       help='Path to the screening classifier checkpoint')
    parser.add_argument('--real_test_data', type=str, required=True,
                       help='Real test data directory')
    parser.add_argument('--synthetic_data', type=str, required=True,
                       help='Synthetic data directory for validation')
    parser.add_argument('--output_dir', type=str, default='./reliability_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔍 筛选分类器可靠性验证")
    print(f"📦 分类器: {args.classifier_path}")
    print(f"📂 真实测试数据: {args.real_test_data}")
    print(f"📂 合成数据: {args.synthetic_data}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载分类器
    try:
        classifier = load_classifier(args.classifier_path, device)
        validator = ScreeningClassifierValidator(classifier, device)
    except Exception as e:
        print(f"❌ 分类器加载失败: {e}")
        return
    
    # 执行验证
    print("\n🚀 开始验证...")
    
    try:
        # 1. 校准度评估
        print("\n📊 1. 评估分类器校准度...")
        calibration_results = validator.evaluate_calibration(args.real_test_data)
        print(f"   ECE: {calibration_results['ece']:.4f} {'✅' if calibration_results['is_well_calibrated'] else '❌'}")
        print(f"   Brier Score: {calibration_results['brier_score']:.4f}")
        
        # 2. 置信度分布比较
        print("\n📊 2. 比较置信度分布...")
        distribution_results = validator.evaluate_confidence_distribution(
            args.real_test_data, args.synthetic_data
        )
        print(f"   真实数据置信度: {distribution_results['real_confidence']['mean']:.3f} ± {distribution_results['real_confidence']['std']:.3f}")
        print(f"   合成数据置信度: {distribution_results['synthetic_confidence']['mean']:.3f} ± {distribution_results['synthetic_confidence']['std']:.3f}")
        print(f"   KL散度: {distribution_results['kl_divergence']:.4f} {'✅' if distribution_results['is_distribution_similar'] else '❌'}")
        
        # 3. 决策边界稳定性
        print("\n📊 3. 评估决策边界稳定性...")
        synthetic_samples = list(Path(args.synthetic_data).glob("**/*.png"))
        if synthetic_samples:
            stability_results = validator.evaluate_decision_boundary_stability(synthetic_samples)
            print(f"   平均稳定性: {stability_results['average_stability']:.3f}")
            print(f"   稳定样本比例: {stability_results['stable_samples_ratio']:.3f} {'✅' if stability_results['is_stable'] else '❌'}")
        else:
            print("   ⚠️ 未找到合成样本")
            stability_results = {'is_stable': False, 'average_stability': 0, 'stable_samples_ratio': 0}
        
        # 4. 错误模式分析
        print("\n📊 4. 分析错误模式...")
        error_results = validator.evaluate_error_analysis(args.synthetic_data)
        print(f"   总错误数: {error_results['total_errors']}")
        print(f"   低置信度错误比例: {error_results['low_confidence_error_ratio']:.3f} {'✅' if error_results['is_error_pattern_acceptable'] else '❌'}")
        print(f"   高置信度错误数: {error_results['high_confidence_errors_count']}")
        
        # 综合评估
        print("\n📋 综合评估结果:")
        print("=" * 50)
        
        # 计算总体可靠性分数
        results = {
            'calibration': calibration_results,
            'distribution': distribution_results,
            'stability': stability_results,
            'errors': error_results
        }
        
        overall = validator._compute_overall_reliability(results)
        print(f"📈 总体可靠性评分: {overall['score']}/100 ({overall['rating']})")
        print(f"🎯 是否可用于筛选: {'✅ 可用' if overall['is_reliable'] else '❌ 不可用'}")
        
        # 生成建议
        recommendations = validator._generate_recommendations(results)
        if recommendations:
            print("\n💡 改进建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("\n💡 分类器表现良好，无需改进")
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 验证完成！")

if __name__ == "__main__":
    main()
