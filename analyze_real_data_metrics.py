"""
分析真实微多普勒数据的筛选指标分布
用于确定合理的筛选阈值，避免合成样本质量污染
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DomainAdaptiveClassifier(nn.Module):
    """与训练时一致的分类器结构"""
    def __init__(self, num_classes=31, dropout_rate=0.3, feature_dim=512):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 用于存储用户原型
        self.register_buffer('feature_bank', torch.zeros(num_classes, feature_dim))
        self.register_buffer('feature_count', torch.zeros(num_classes))

    def forward(self, x):
        backbone_features = self.backbone(x)
        features = self.feature_projector(backbone_features)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """提取特征用于多样性分析"""
        backbone_features = self.backbone(x)
        features = self.feature_projector(backbone_features)
        return features

def load_classifier(checkpoint_path, device):
    """加载训练好的分类器"""
    model = DomainAdaptiveClassifier(num_classes=31)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_real_data(data_dir, max_samples_per_user=100):
    """加载真实微多普勒数据"""
    data_dir = Path(data_dir)
    samples = []
    labels = []
    
    print(f"🔍 扫描真实数据目录: {data_dir}")
    
    # 创建和训练时一致的标签映射
    all_classes = []
    
    for user_id in range(1, 32):  # ID_1 到 ID_31
        user_dir = data_dir / f"ID_{user_id}"
        if not user_dir.exists():
            continue
        all_classes.append(f"ID_{user_id}")
            
    # 数值排序（与训练一致）
    def sort_ids(class_list):
        def extract_number(class_name):
            if class_name.startswith('ID_'):
                try:
                    return int(class_name.split('_')[1])
                except (IndexError, ValueError):
                    return float('inf')
            return float('inf')
        return sorted(class_list, key=extract_number)
    
    sorted_classes = sort_ids(all_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    print(f"✅ 标签映射: {class_to_idx}")
    
    for user_id in range(1, 32):  # ID_1 到 ID_31
        user_dir = data_dir / f"ID_{user_id}"
        if not user_dir.exists():
            continue
            
        user_samples = []
        # 加载所有jpg文件
        for img_path in user_dir.glob("*.jpg"):
            user_samples.append(str(img_path))
        
        # 如果文件数超过限制才截取
        if len(user_samples) > max_samples_per_user:
            user_samples = user_samples[:max_samples_per_user]
        
        samples.extend(user_samples)
        class_name = f"ID_{user_id}"
        correct_label = class_to_idx[class_name]
        labels.extend([correct_label] * len(user_samples))
        print(f"ID_{user_id}: {len(user_samples)} samples -> label {correct_label} ({class_name})")
    
    print(f"📊 总计加载: {len(samples)} 样本")
    return samples, labels

def calculate_metrics_batch(classifier, images, labels, device):
    """批量计算所有筛选指标"""
    classifier.eval()
    
    # 修夏：使用和训练一致的预处理
    print("✅ 使用训练时一致的预处理: 256x256 + ImageNet normalization")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 修夏: 与训练时一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_metrics = {
        'confidence': [],
        'margin': [],
        'user_specificity': [],
        'predicted_user': [],
        'true_user': [],
        'features': []
    }
    
    # 添加调试信息
    print(f"📊 标签分布: {sorted(set(labels))}")
    print(f"📊 标签范围: {min(labels)} - {max(labels)} (应该是 0-30)")
    print(f"📊 总类别数: {len(set(labels))} (应该是 31)")
    
    # 测试几个样本看预测结果
    debug_predictions = []
    
    with torch.no_grad():
        for idx, (img_path, true_label) in enumerate(tqdm(zip(images, labels), total=len(images), desc="计算指标")):
            try:
                # 加载并预处理图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 前向传播
                logits = classifier(img_tensor)
                probs = torch.softmax(logits, dim=1)
                features = classifier.extract_features(img_tensor)
                
                # 1. 置信度 (最大概率)
                confidence = torch.max(probs).item()
                
                # 2. 决策边界 (最大概率 - 次大概率)
                sorted_probs = torch.sort(probs, descending=True)[0]
                margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
                
                # 3. 用户特异性 (预测用户概率 - 其他用户最大概率)
                predicted_user = torch.argmax(probs).item()
                user_prob = probs[0, predicted_user].item()
                other_max = torch.max(torch.cat([probs[0, :predicted_user], 
                                               probs[0, predicted_user+1:]])).item()
                user_specificity = user_prob - other_max
                
                # 存储结果
                batch_metrics['confidence'].append(confidence)
                batch_metrics['margin'].append(margin)
                batch_metrics['user_specificity'].append(user_specificity)
                batch_metrics['predicted_user'].append(predicted_user)
                batch_metrics['true_user'].append(true_label)
                batch_metrics['features'].append(features.cpu().numpy().flatten())
                
                # 调试前10个样本
                if idx < 10:
                    debug_predictions.append({
                        'file': img_path.split('/')[-1],
                        'true_label': true_label,
                        'predicted': predicted_user,
                        'confidence': confidence,
                        'top3_probs': torch.topk(probs, 3)[0].cpu().numpy().flatten(),
                        'top3_users': torch.topk(probs, 3)[1].cpu().numpy().flatten()
                    })
                
            except Exception as e:
                print(f"⚠️ 处理图像 {img_path} 时出错: {e}")
                continue
    
    # 输出调试信息
    print("\n🔍 前10个样本的预测调试信息:")
    for i, debug in enumerate(debug_predictions):
        print(f"样本{i+1}: {debug['file']}")
        # 根据标签映射找到对应的ID
        true_id = [k for k, v in class_to_idx.items() if v == debug['true_label']][0] if debug['true_label'] in class_to_idx.values() else f"Unknown({debug['true_label']})"
        pred_id = [k for k, v in class_to_idx.items() if v == debug['predicted']][0] if debug['predicted'] in class_to_idx.values() else f"Unknown({debug['predicted']})"
        print(f"  真实标签: {debug['true_label']} ({true_id})")
        print(f"  预测标签: {debug['predicted']} ({pred_id})")
        print(f"  置信度: {debug['confidence']:.3f}")
        print(f"  Top3概率: {debug['top3_probs']}")
        top3_ids = []
        for user_idx in debug['top3_users']:
            if user_idx in class_to_idx.values():
                user_id = [k for k, v in class_to_idx.items() if v == user_idx][0]
                top3_ids.append(user_id)
            else:
                top3_ids.append(f"Unknown({user_idx})")
        print(f"  Top3用户: {debug['top3_users']} ({top3_ids})")
        print("")
    
    # 把 class_to_idx 传递给其他函数使用
    batch_metrics['class_to_idx'] = class_to_idx
    return batch_metrics

def analyze_diversity(features):
    """分析特征多样性"""
    if len(features) < 2:
        return []
    
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(features)
    
    # 提取上三角矩阵（避免重复和对角线）
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[triu_indices]
    
    return similarities

def analyze_user_differences(metrics):
    """分析用户间差异"""
    df = pd.DataFrame(metrics)
    
    # 按用户分组统计
    user_stats = df.groupby('true_user').agg({
        'confidence': ['mean', 'std', 'min', 'max'],
        'margin': ['mean', 'std', 'min', 'max'],
        'user_specificity': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    return user_stats

def plot_metric_distributions(metrics, output_dir):
    """绘制指标分布图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 置信度分布
    axes[0, 0].hist(metrics['confidence'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(metrics['confidence']), color='red', linestyle='--', 
                       label=f'均值: {np.mean(metrics["confidence"]):.3f}')
    axes[0, 0].set_title('置信度分布')
    axes[0, 0].set_xlabel('置信度')
    axes[0, 0].legend()
    
    # 2. 决策边界分布
    axes[0, 1].hist(metrics['margin'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(metrics['margin']), color='red', linestyle='--',
                       label=f'均值: {np.mean(metrics["margin"]):.3f}')
    axes[0, 1].set_title('决策边界分布')
    axes[0, 1].set_xlabel('决策边界')
    axes[0, 1].legend()
    
    # 3. 用户特异性分布
    axes[0, 2].hist(metrics['user_specificity'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.mean(metrics['user_specificity']), color='red', linestyle='--',
                       label=f'均值: {np.mean(metrics["user_specificity"]):.3f}')
    axes[0, 2].set_title('用户特异性分布')
    axes[0, 2].set_xlabel('用户特异性')
    axes[0, 2].legend()
    
    # 4. 准确率分析
    accuracy = np.mean(np.array(metrics['predicted_user']) == np.array(metrics['true_user']))
    axes[1, 0].bar(['正确', '错误'], [accuracy, 1-accuracy])
    axes[1, 0].set_title(f'分类准确率: {accuracy:.3f}')
    axes[1, 0].set_ylabel('比例')
    
    # 5. 特征多样性
    if len(metrics['features']) > 1:
        similarities = analyze_diversity(metrics['features'])
        axes[1, 1].hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(similarities), color='red', linestyle='--',
                           label=f'均值: {np.mean(similarities):.3f}')
        axes[1, 1].set_title('特征相似性分布')
        axes[1, 1].set_xlabel('余弦相似度')
        axes[1, 1].legend()
    
    # 6. 置信度vs准确性
    correct_mask = np.array(metrics['predicted_user']) == np.array(metrics['true_user'])
    correct_conf = np.array(metrics['confidence'])[correct_mask]
    wrong_conf = np.array(metrics['confidence'])[~correct_mask]
    
    axes[1, 2].hist([correct_conf, wrong_conf], bins=30, alpha=0.7, 
                    label=['正确预测', '错误预测'], edgecolor='black')
    axes[1, 2].set_title('置信度vs预测准确性')
    axes[1, 2].set_xlabel('置信度')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_data_metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def recommend_thresholds(metrics):
    """基于真实数据分布推荐阈值"""
    print("\n" + "="*60)
    print("📊 基于真实数据的阈值推荐")
    print("="*60)
    
    # 计算分位数
    conf_percentiles = np.percentile(metrics['confidence'], [25, 50, 75, 90, 95])
    margin_percentiles = np.percentile(metrics['margin'], [25, 50, 75, 90, 95])
    spec_percentiles = np.percentile(metrics['user_specificity'], [25, 50, 75, 90, 95])
    
    if len(metrics['features']) > 1:
        similarities = analyze_diversity(metrics['features'])
        sim_percentiles = np.percentile(similarities, [5, 10, 25, 50, 75])
    
    print(f"📈 置信度分位数: P25={conf_percentiles[0]:.3f}, P50={conf_percentiles[1]:.3f}, P75={conf_percentiles[2]:.3f}")
    print(f"📈 决策边界分位数: P25={margin_percentiles[0]:.3f}, P50={margin_percentiles[1]:.3f}, P75={margin_percentiles[2]:.3f}")
    print(f"📈 用户特异性分位数: P25={spec_percentiles[0]:.3f}, P50={spec_percentiles[1]:.3f}, P75={spec_percentiles[2]:.3f}")
    
    if len(metrics['features']) > 1:
        print(f"📈 相似度分位数: P5={sim_percentiles[0]:.3f}, P25={sim_percentiles[2]:.3f}, P50={sim_percentiles[3]:.3f}")
    
    print("\n🎯 推荐的筛选阈值（基于真实数据）:")
    
    # 保守策略：保留75%的真实样本质量
    print("【保守策略 - 保留75%真实样本质量】")
    print(f"  - 置信度: {conf_percentiles[0]:.3f} (P25)")
    print(f"  - 决策边界: {margin_percentiles[0]:.3f} (P25)")  
    print(f"  - 用户特异性: {spec_percentiles[0]:.3f} (P25)")
    if len(metrics['features']) > 1:
        print(f"  - 多样性阈值: {1-sim_percentiles[2]:.3f} (1-P25相似度)")
    
    # 中等策略：保留50%的真实样本质量  
    print("\n【中等策略 - 保留50%真实样本质量】")
    print(f"  - 置信度: {conf_percentiles[1]:.3f} (P50)")
    print(f"  - 决策边界: {margin_percentiles[1]:.3f} (P50)")
    print(f"  - 用户特异性: {spec_percentiles[1]:.3f} (P50)")
    if len(metrics['features']) > 1:
        print(f"  - 多样性阈值: {1-sim_percentiles[3]:.3f} (1-P50相似度)")
    
    # 严格策略：只保留25%最好的真实样本质量
    print("\n【严格策略 - 只保留25%最高质量】")
    print(f"  - 置信度: {conf_percentiles[2]:.3f} (P75)")
    print(f"  - 决策边界: {margin_percentiles[2]:.3f} (P75)")
    print(f"  - 用户特异性: {spec_percentiles[2]:.3f} (P75)")
    if len(metrics['features']) > 1:
        print(f"  - 多样性阈值: {1-sim_percentiles[1]:.3f} (1-P10相似度)")

def main():
    parser = argparse.ArgumentParser(description='分析真实微多普勒数据的筛选指标分布')
    parser.add_argument('--real_data_dir', type=str, required=True,
                       help='真实微多普勒数据目录路径')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                       help='训练好的分类器checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='./real_data_analysis',
                       help='分析结果输出目录')
    parser.add_argument('--max_samples_per_user', type=int, default=100,
                       help='每个用户最大样本数（避免内存溢出）')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载分类器
    print("🔄 加载分类器...")
    classifier = load_classifier(args.classifier_checkpoint, device)
    
    # 加载真实数据
    print("🔄 加载真实数据...")
    images, labels = load_real_data(args.real_data_dir, args.max_samples_per_user)
    
    if len(images) == 0:
        print("❌ 未找到真实数据！请检查数据目录路径")
        return
    
    # 计算指标
    print("🔄 计算筛选指标...")
    metrics = calculate_metrics_batch(classifier, images, labels, device)
    
    # 分析用户间差异
    print("🔄 分析用户间差异...")
    user_stats = analyze_user_differences(metrics)
    print("\n用户统计信息:")
    print(user_stats)
    
    # 绘制分布图
    print("🔄 生成分析图表...")
    plot_metric_distributions(metrics, output_dir)
    
    # 推荐阈值
    recommend_thresholds(metrics)
    
    # 保存分析结果
    results = {
        'total_samples': len(images),
        'accuracy': np.mean(np.array(metrics['predicted_user']) == np.array(metrics['true_user'])),
        'mean_confidence': np.mean(metrics['confidence']),
        'mean_margin': np.mean(metrics['margin']),
        'mean_user_specificity': np.mean(metrics['user_specificity']),
        'user_statistics': user_stats.to_dict()
    }
    
    if len(metrics['features']) > 1:
        similarities = analyze_diversity(metrics['features'])
        results['mean_feature_similarity'] = np.mean(similarities)
    
    # 保存到文件（修复JSON序列化问题）
    import json
    # 转换user_statistics中的tuple keys为string
    if 'user_statistics' in results:
        results['user_statistics'] = {str(k): v for k, v in results['user_statistics'].items()}
    
    with open(output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 分析完成！结果保存到: {output_dir}")
    print(f"📊 分析了 {len(images)} 个真实样本，分类准确率: {results['accuracy']:.3f}")
    
    # 详细分析置信度低的原因
    print("\n" + "="*60)
    print("🔍 置信度分析")
    print("="*60)
    
    correct_mask = np.array(metrics['predicted_user']) == np.array(metrics['true_user'])
    correct_conf = np.array(metrics['confidence'])[correct_mask]
    wrong_conf = np.array(metrics['confidence'])[~correct_mask]
    
    print(f"✅ 正确预测样本置信度: 均值={np.mean(correct_conf):.3f}, 中位数={np.median(correct_conf):.3f}")
    print(f"❌ 错误预测样本置信度: 均值={np.mean(wrong_conf):.3f}, 中位数={np.median(wrong_conf):.3f}")
    print(f"📊 高置信度(>0.8)样本占比: {np.mean(np.array(metrics['confidence']) > 0.8)*100:.1f}%")
    print(f"📊 中等置信度(0.5-0.8)样本占比: {np.mean((np.array(metrics['confidence']) > 0.5) & (np.array(metrics['confidence']) <= 0.8))*100:.1f}%")
    print(f"📊 低置信度(<0.5)样本占比: {np.mean(np.array(metrics['confidence']) < 0.5)*100:.1f}%")
    
    # 分析混淆情况
    predicted = np.array(metrics['predicted_user'])
    true_labels = np.array(metrics['true_user'])
    
    # 计算最容易混淆的用户对
    confusion_pairs = {}
    for i in range(len(predicted)):
        if predicted[i] != true_labels[i]:
            pair = tuple(sorted([true_labels[i], predicted[i]]))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    if confusion_pairs:
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🔄 最容易混淆的用户对（前5）:")
        for (user1, user2), count in top_confusions:
            # 根据标签映射找到ID
            id1 = [k for k, v in class_to_idx.items() if v == user1][0] if user1 in class_to_idx.values() else f"Unknown({user1})"
            id2 = [k for k, v in class_to_idx.items() if v == user2][0] if user2 in class_to_idx.values() else f"Unknown({user2})"
            print(f"   {id1} ↔ {id2}: {count} 次混淆")
    
    # 置信度低的可能原因分析
    print(f"\n💡 置信度低的可能原因:")
    if results['accuracy'] < 0.6:
        print("   1. 🎯 分类器性能不足 - 准确率过低导致不确定性增加")
    if np.mean(np.array(metrics['confidence']) < 0.5) > 0.7:
        print("   2. 👥 用户间相似性过高 - 微多普勒特征差异微小")
    if len(confusion_pairs) > 10:
        print("   3. 🔄 广泛的类间混淆 - 多个用户特征重叠")
    print("   4. 📏 模型校准问题 - 可能需要温度缩放或重新校准")
    print("   5. 🏗️ 架构限制 - ResNet18可能不足以捕获微多普勒细微差异")

if __name__ == '__main__':
    main()
