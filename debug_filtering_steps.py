"""
调试generation_filtering.py的每个筛选步骤
找出0.2%通过率的真正原因
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from collections import defaultdict

def load_classifier(model_path, device):
    """加载分类器（复制自generation_filtering.py）"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
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

def simple_quality_check(images):
    """简化的图像质量检查（复制自generation_filtering.py）"""
    quality_scores = []
    
    for img in images:
        img_array = np.array(img)
        
        # 只检测基本像素值异常
        pixel_mean = np.mean(img_array)
        pixel_std = np.std(img_array)
        
        # 简单的质量分数：只检查是否全黑/全白或无变化
        is_valid = (
            10 < pixel_mean < 245 and  # 不是全黑或全白
            pixel_std > 5              # 有一定变化
        )
        
        quality_score = {
            'pixel_mean': pixel_mean,
            'pixel_std': pixel_std,
            'is_valid': is_valid,
            'overall': 1.0 if is_valid else 0.0
        }
        
        quality_scores.append(quality_score)
    
    return quality_scores

def compute_user_specific_metrics_debug(images, classifier, user_id, device):
    """调试版本的用户指标计算"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    user_metrics_list = []
    
    for img in images:
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取分类器输出和特征
            outputs = classifier(img_tensor)
            
            # 处理分类器输出格式（可能是tuple）
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 通常第一个元素是logits
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=1)
            features = classifier.backbone(img_tensor)
            
            # 1. 基本指标
            confidence, pred = torch.max(probs, dim=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
            
            # 2. 用户特异性分数（与其他用户的区分度）
            user_prob = probs[0, user_id].item()
            other_probs = torch.cat([probs[0, :user_id], probs[0, user_id+1:]])
            max_other_prob = torch.max(other_probs).item()
            # 使用比例而非差值
            user_specificity = user_prob / (user_prob + max_other_prob) if (user_prob + max_other_prob) > 0 else 0.0
            
            metrics = {
                'predicted': pred.item(),
                'confidence': confidence.item(),
                'margin': margin,
                'user_specificity': user_specificity,
                'correct': pred.item() == user_id,
                'features': features.cpu().numpy().flatten()
            }
            
            user_metrics_list.append(metrics)
    
    return user_metrics_list

def debug_filtering_steps(samples_dir, classifier_path, device='cuda:0', user_id=0, num_samples=100):
    """逐步调试筛选过程"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"🔍 调试用户{user_id}的筛选过程")
    print(f"📂 样本目录: {samples_dir}")
    
    # 加载分类器
    classifier = load_classifier(classifier_path, device)
    
    # 获取样本
    samples_path = Path(samples_dir) / f"user_{user_id:02d}"
    if not samples_path.exists():
        print(f"❌ 用户目录不存在: {samples_path}")
        return
    
    image_files = list(samples_path.glob('*.png'))[:num_samples]
    print(f"📷 加载{len(image_files)}张图像进行调试")
    
    # 加载图像
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except:
            continue
    
    if len(images) == 0:
        print("❌ 没有成功加载任何图像")
        return
    
    print(f"✅ 成功加载{len(images)}张图像")
    
    # 第一步：计算指标
    print("\n📊 步骤1: 计算各项指标")
    metrics_list = compute_user_specific_metrics_debug(images, classifier, user_id, device)
    visual_quality_scores = simple_quality_check(images)
    
    # 统计各项指标
    stats = defaultdict(list)
    for i, (metrics, visual) in enumerate(zip(metrics_list, visual_quality_scores)):
        stats['correct'].append(metrics['correct'])
        stats['confidence'].append(metrics['confidence'])
        stats['user_specificity'].append(metrics['user_specificity'])
        stats['margin'].append(metrics['margin'])
        stats['visual_valid'].append(visual['is_valid'])
    
    print("指标统计:")
    for key, values in stats.items():
        if key != 'features':
            pass_rate = np.mean(values) * 100
            mean_val = np.mean(values) if key != 'correct' and key != 'visual_valid' else pass_rate/100
            print(f"   {key}: 均值={mean_val:.3f}, 通过率={pass_rate:.1f}%")
    
    # 第二步：应用各项阈值
    print(f"\n🎯 步骤2: 应用筛选阈值")
    
    thresholds = {
        'confidence': 0.9,
        'user_specificity': 0.7,  # 使用新的比例模式
        'margin': 0.8,
        'diversity': 0.1
    }
    
    # 第一轮筛选（不包括多样性）
    first_stage_candidates = []
    filter_stats = defaultdict(int)
    
    for i, (metrics, visual) in enumerate(zip(metrics_list, visual_quality_scores)):
        filter_stats['total'] += 1
        
        # 检查每个条件
        conditions = {
            'correct': metrics['correct'],
            'confidence': metrics['confidence'] > thresholds['confidence'],
            'user_specificity': metrics['user_specificity'] > thresholds['user_specificity'],
            'margin': metrics['margin'] > thresholds['margin'],
            'visual_valid': visual['is_valid']
        }
        
        # 记录各个条件的通过情况
        for cond, passed in conditions.items():
            if passed:
                filter_stats[f'{cond}_pass'] += 1
            else:
                filter_stats[f'{cond}_fail'] += 1
        
        # 所有条件都满足才进入候选
        if all(conditions.values()):
            first_stage_candidates.append({
                'image': images[i],
                'features': metrics['features'],
                'metrics': metrics,
                'index': i
            })
            filter_stats['first_stage_pass'] += 1
    
    print("第一阶段筛选结果:")
    total = filter_stats['total']
    for key in ['correct', 'confidence', 'user_specificity', 'margin', 'visual_valid']:
        pass_count = filter_stats[f'{key}_pass']
        fail_count = filter_stats[f'{key}_fail']
        pass_rate = pass_count / total * 100
        print(f"   {key}: {pass_count}/{total} ({pass_rate:.1f}%)")
    
    first_pass_rate = filter_stats['first_stage_pass'] / total * 100
    print(f"   📊 第一阶段总通过率: {filter_stats['first_stage_pass']}/{total} ({first_pass_rate:.1f}%)")
    
    # 第三步：多样性筛选
    print(f"\n🌈 步骤3: 多样性筛选")
    
    if len(first_stage_candidates) == 0:
        print("❌ 第一阶段没有候选样本，无法进行多样性筛选")
        return
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    final_candidates = []
    collected_features = []
    
    for candidate in first_stage_candidates:
        # 检查与已收集样本的多样性
        diversity_score = 1.0  # 默认多样性分数（第一个样本）
        if len(collected_features) > 0:
            candidate_features = candidate['features'].reshape(1, -1)
            collected_array = np.array(collected_features)
            
            # 计算与现有样本的最大相似度
            similarities = cosine_similarity(candidate_features, collected_array)[0]
            max_similarity = np.max(similarities)
            diversity_score = 1.0 - max_similarity
        
        # 应用多样性阈值
        if diversity_score >= thresholds['diversity']:
            final_candidates.append(candidate)
            collected_features.append(candidate['features'])
    
    final_pass_rate = len(final_candidates) / total * 100
    diversity_pass_rate = len(final_candidates) / len(first_stage_candidates) * 100 if len(first_stage_candidates) > 0 else 0
    
    print(f"多样性筛选结果:")
    print(f"   输入候选: {len(first_stage_candidates)}")
    print(f"   最终通过: {len(final_candidates)}")
    print(f"   多样性通过率: {diversity_pass_rate:.1f}%")
    print(f"   📊 最终总通过率: {len(final_candidates)}/{total} ({final_pass_rate:.1f}%)")
    
    # 详细分析失败原因
    print(f"\n❌ 失败原因分析:")
    fail_reasons = {}
    for key in ['correct', 'confidence', 'user_specificity', 'margin', 'visual_valid']:
        fail_count = filter_stats[f'{key}_fail']
        fail_reasons[key] = fail_count / total * 100
    
    # 按失败率排序
    sorted_failures = sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)
    for reason, rate in sorted_failures:
        print(f"   {reason}: {rate:.1f}%样本失败")
    
    return {
        'total_samples': total,
        'first_stage_pass': filter_stats['first_stage_pass'],
        'final_pass': len(final_candidates),
        'first_pass_rate': first_pass_rate,
        'final_pass_rate': final_pass_rate,
        'filter_stats': filter_stats,
        'fail_reasons': fail_reasons
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type=str, default='/kaggle/working/VA-VAE/generated_samples2')
    parser.add_argument('--classifier_path', type=str, default='/kaggle/working/VA-VAE/domain_adaptive_classifier/best_calibrated_model.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--user_id', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    debug_filtering_steps(
        args.samples_dir, 
        args.classifier_path, 
        args.device, 
        args.user_id,
        args.num_samples
    )

if __name__ == "__main__":
    main()
