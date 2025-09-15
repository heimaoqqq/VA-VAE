"""
分析生成样本的筛选指标分布
帮助确定合理的筛选阈值
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json


def load_classifier(model_path, device):
    """加载分类器"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    # 根据checkpoint判断模型类型
    if 'feature_projector.0.weight' in checkpoint['model_state_dict']:
        from train_calibrated_classifier import DomainAdaptiveClassifier
        model = DomainAdaptiveClassifier(
            num_classes=checkpoint['num_classes'],
            dropout_rate=0.3,
            feature_dim=512
        )
    else:
        from improved_classifier_training import ImprovedClassifier
        model = ImprovedClassifier(
            num_classes=checkpoint['num_classes'],
            backbone='resnet18',
            dropout_rate=0.5,
            freeze_layers='minimal'
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def extract_features(img_tensor, classifier):
    """提取特征"""
    with torch.no_grad():
        if hasattr(classifier, 'backbone'):
            features = classifier.backbone(img_tensor)
        else:
            # 如果没有backbone属性，使用模型的特征提取部分
            features = classifier.features(img_tensor)
        return features.cpu().numpy().flatten()


def compute_sample_metrics(image_path, classifier, user_id, device):
    """计算单个样本的所有筛选指标"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取分类器输出
        outputs = classifier(img_tensor)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        probs = F.softmax(logits, dim=1)
        
        # 1. 置信度
        confidence, pred = torch.max(probs, dim=1)
        confidence = confidence.item()
        
        # 2. 决策边界 (margin)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
        
        # 3. 用户特异性
        user_prob = probs[0, user_id].item()
        other_probs = torch.cat([probs[0, :user_id], probs[0, user_id+1:]])
        max_other_prob = torch.max(other_probs).item()
        user_specificity = user_prob - max_other_prob
        
        # 4. 提取特征用于多样性计算
        features = extract_features(img_tensor, classifier)
        
    return {
        'confidence': confidence,
        'margin': margin,
        'user_specificity': user_specificity,
        'predicted_user': pred.item(),
        'correct': pred.item() == user_id,
        'features': features,
        'image_path': str(image_path)
    }


def compute_batch_diversity(features_list):
    """计算批次内特征多样性"""
    if len(features_list) < 2:
        return 0.0
    
    features_array = np.array(features_list)
    cosine_sim_matrix = cosine_similarity(features_array)
    
    # 计算上三角矩阵的平均相似度
    upper_triangle = np.triu(cosine_sim_matrix, k=1)
    n = len(features_list)
    avg_similarity = np.sum(upper_triangle) / (n * (n - 1) / 2)
    
    diversity_score = 1.0 - avg_similarity
    return diversity_score


def analyze_user_samples(sample_dir, classifier, user_id, device):
    """分析单个用户的所有样本"""
    
    user_dir = Path(sample_dir) / f"user_{user_id:02d}"
    if not user_dir.exists():
        user_dir = Path(sample_dir) / f"User_{user_id:02d}"
    if not user_dir.exists():
        print(f"❌ 未找到用户 {user_id} 的样本目录")
        return None
    
    # 获取所有图片文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(user_dir.glob(ext)))
    
    if not image_files:
        print(f"❌ 用户 {user_id} 目录中没有图片文件")
        return None
    
    print(f"📊 分析用户 {user_id} 的 {len(image_files)} 个样本...")
    
    # 计算每个样本的指标
    sample_metrics = []
    features_list = []
    
    for img_path in tqdm(image_files, desc=f"User {user_id}"):
        try:
            metrics = compute_sample_metrics(img_path, classifier, user_id, device)
            sample_metrics.append(metrics)
            features_list.append(metrics['features'])
        except Exception as e:
            print(f"⚠️  处理 {img_path} 时出错: {e}")
            continue
    
    if not sample_metrics:
        return None
    
    # 计算批次多样性
    batch_diversity = compute_batch_diversity(features_list)
    
    # 统计结果
    confidences = [m['confidence'] for m in sample_metrics]
    margins = [m['margin'] for m in sample_metrics]
    user_specificities = [m['user_specificity'] for m in sample_metrics]
    stabilities = [m['stability'] for m in sample_metrics]
    correct_predictions = [m['correct'] for m in sample_metrics]
    
    results = {
        'user_id': user_id,
        'total_samples': len(sample_metrics),
        'accuracy': np.mean(correct_predictions),
        'metrics': {
            'confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'percentiles': {
                    '25': np.percentile(confidences, 25),
                    '50': np.percentile(confidences, 50),
                    '75': np.percentile(confidences, 75),
                    '90': np.percentile(confidences, 90),
                    '95': np.percentile(confidences, 95)
                }
            },
            'margin': {
                'mean': np.mean(margins),
                'std': np.std(margins),
                'min': np.min(margins),
                'max': np.max(margins),
                'percentiles': {
                    '25': np.percentile(margins, 25),
                    '50': np.percentile(margins, 50),
                    '75': np.percentile(margins, 75)
                }
            },
            'user_specificity': {
                'mean': np.mean(user_specificities),
                'std': np.std(user_specificities),
                'min': np.min(user_specificities),
                'max': np.max(user_specificities),
                'percentiles': {
                    '25': np.percentile(user_specificities, 25),
                    '50': np.percentile(user_specificities, 50),
                    '75': np.percentile(user_specificities, 75)
                }
            },
            'stability': {
                'mean': np.mean(stabilities),
                'std': np.std(stabilities),
                'min': np.min(stabilities),
                'max': np.max(stabilities),
                'percentiles': {
                    '25': np.percentile(stabilities, 25),
                    '50': np.percentile(stabilities, 50),
                    '75': np.percentile(stabilities, 75)
                }
            },
            'batch_diversity': batch_diversity
        },
        'raw_data': {
            'confidences': confidences,
            'margins': margins,
            'user_specificities': user_specificities,
            'stabilities': stabilities
        }
    }
    
    return results


def plot_metrics_distribution(results_list, output_dir):
    """绘制指标分布图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有数据
    all_confidences = []
    all_margins = []
    all_user_specificities = []
    all_stabilities = []
    user_ids = []
    
    for result in results_list:
        if result is None:
            continue
        raw_data = result['raw_data']
        n_samples = len(raw_data['confidences'])
        
        all_confidences.extend(raw_data['confidences'])
        all_margins.extend(raw_data['margins'])
        all_user_specificities.extend(raw_data['user_specificities'])
        all_stabilities.extend(raw_data['stabilities'])
        user_ids.extend([result['user_id']] * n_samples)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 置信度分布
    axes[0, 0].hist(all_confidences, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0.75, color='red', linestyle='--', label='推荐阈值: 0.75')
    axes[0, 0].set_xlabel('置信度')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('置信度分布')
    axes[0, 0].legend()
    
    # 2. 决策边界分布
    axes[0, 1].hist(all_margins, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0.3, color='red', linestyle='--', label='推荐阈值: 0.3')
    axes[0, 1].set_xlabel('决策边界 (Margin)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('决策边界分布')
    axes[0, 1].legend()
    
    # 3. 用户特异性分布
    axes[1, 0].hist(all_user_specificities, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0.2, color='red', linestyle='--', label='推荐阈值: 0.2')
    axes[1, 0].set_xlabel('用户特异性')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('用户特异性分布')
    axes[1, 0].legend()
    
    # 4. 稳定性分布
    axes[1, 1].hist(all_stabilities, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0.6, color='red', linestyle='--', label='推荐阈值: 0.6')
    axes[1, 1].set_xlabel('稳定性')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('稳定性分布')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 指标分布图已保存到: {output_dir / 'metrics_distribution.png'}")


def print_summary_report(results_list):
    """打印总结报告"""
    print("\n" + "="*60)
    print("📊 筛选指标分析报告")
    print("="*60)
    
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        print("❌ 没有有效的分析结果")
        return
    
    # 计算各指标的总体统计
    all_metrics = {
        'confidence': [],
        'margin': [],
        'user_specificity': [],
        'batch_diversity': []
    }
    
    for result in valid_results:
        all_metrics['confidence'].extend(result['raw_data']['confidences'])
        all_metrics['margin'].extend(result['raw_data']['margins'])
        all_metrics['user_specificity'].extend(result['raw_data']['user_specificities'])
        all_metrics['batch_diversity'].append(result['metrics']['batch_diversity'])
    
    print(f"\n📈 总体指标统计 (基于 {sum(r['total_samples'] for r in valid_results)} 个样本):")
    
    for metric_name, values in all_metrics.items():
        if not values:
            continue
        print(f"\n🎯 {metric_name}:")
        print(f"   平均值: {np.mean(values):.3f} ± {np.std(values):.3f}")
        print(f"   范围: [{np.min(values):.3f}, {np.max(values):.3f}]")
        print(f"   25%分位: {np.percentile(values, 25):.3f}")
        print(f"   50%分位: {np.percentile(values, 50):.3f}")
        print(f"   75%分位: {np.percentile(values, 75):.3f}")
    
    # 模拟筛选通过率
    print(f"\n🔍 筛选通过率预测:")
    
    thresholds = {
        'confidence': [0.7, 0.75, 0.8, 0.85, 0.9],
        'margin': [0.2, 0.3, 0.4, 0.5],
        'user_specificity': [0.1, 0.2, 0.3, 0.4]
    }
    
    for metric_name, threshold_list in thresholds.items():
        values = all_metrics[metric_name]
        print(f"\n📊 {metric_name} 通过率:")
        for threshold in threshold_list:
            pass_rate = np.mean(np.array(values) >= threshold) * 100
            print(f"   阈值 {threshold}: {pass_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='分析生成样本的筛选指标')
    parser.add_argument('--sample_dir', type=str, default='./generated_samples', 
                       help='生成样本目录')
    parser.add_argument('--classifier_path', type=str, 
                       default='/kaggle/working/VA-VAE/domain_adaptive_classifier/best_calibrated_model.pth',
                       help='分类器路径')
    parser.add_argument('--output_dir', type=str, default='./metrics_analysis',
                       help='输出目录')
    parser.add_argument('--user_ids', type=str, default='all',
                       help='要分析的用户ID，逗号分隔，或"all"分析所有')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 加载分类器
    print("📦 加载分类器...")
    classifier = load_classifier(args.classifier_path, device)
    
    # 确定要分析的用户
    if args.user_ids == 'all':
        sample_dir = Path(args.sample_dir)
        user_dirs = list(sample_dir.glob('user_*')) + list(sample_dir.glob('User_*'))
        user_ids = []
        for user_dir in user_dirs:
            if user_dir.name.startswith('user_'):
                user_id = int(user_dir.name.split('_')[1])
            elif user_dir.name.startswith('User_'):
                user_id = int(user_dir.name.split('_')[1])
            else:
                continue
            user_ids.append(user_id)
        user_ids = sorted(user_ids)
    else:
        user_ids = [int(x.strip()) for x in args.user_ids.split(',')]
    
    print(f"🎯 分析用户: {user_ids}")
    
    # 分析每个用户
    results_list = []
    for user_id in user_ids:
        result = analyze_user_samples(args.sample_dir, classifier, user_id, device)
        if result:
            results_list.append(result)
            print(f"✅ 用户 {user_id}: 准确率 {result['accuracy']:.1%}")
        else:
            results_list.append(None)
    
    # 生成报告
    print_summary_report(results_list)
    
    # 绘制分布图
    if results_list:
        plot_metrics_distribution(results_list, args.output_dir)
    
    # 保存详细结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        # 移除features数据以减小文件大小
        save_results = []
        for result in results_list:
            if result:
                save_result = result.copy()
                save_result['raw_data'] = {k: v for k, v in result['raw_data'].items() if k != 'features'}
                save_results.append(save_result)
            else:
                save_results.append(None)
        json.dump(save_results, f, indent=2)
    
    print(f"\n📁 详细结果已保存到: {output_dir / 'detailed_results.json'}")


if __name__ == "__main__":
    main()
