"""
基于实际分类器性能筛选用户
使用已训练的分类器评估哪些用户最容易区分

理论依据：
1. Feature Subset Selection, Class Separability (Springer 2004)
2. Class-Proportional Coreset Selection (ArXiv 2024)
3. Multi-Class Confusion Matrix Analysis (Analytics Vidhya 2025)

方法：基于混淆矩阵选择最可分离的类别
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# 从训练脚本复制模型定义
class DomainAdaptiveClassifier(nn.Module):
    """针对小数据集和域适应设计的分类器"""
    def __init__(self, num_classes=31, dropout_rate=0.3, feature_dim=512):
        super().__init__()
        
        # 使用ResNet18作为backbone（预训练权重很重要）
        self.backbone = models.resnet18(pretrained=True)
        
        # 冻结前面的层，只微调后面的层（防止小数据集过拟合）
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # 特征提取维度
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 特征投影层（用于对比学习）
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头（简单但有效，避免过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 身份特征记忆库（用于筛选时的特征匹配）
        self.register_buffer('feature_bank', torch.zeros(num_classes, feature_dim))
        self.register_buffer('feature_count', torch.zeros(num_classes))
        
        # 温度参数
        self.temperature = 1.0
    
    def forward(self, x):
        # 特征提取
        backbone_features = self.backbone(x)
        features = self.feature_projector(backbone_features)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, features

def load_and_preprocess_image(img_path, size=256):
    """
    加载并预处理图像
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

def evaluate_user_separability(classifier_path='/kaggle/input/best-calibrated-model-pth/best_calibrated_model.pth',
                               dataset_path='/kaggle/input/dataset'):
    """
    使用训练好的分类器评估用户可分性
    
    核心思想：
    - 即使分类器已经训练好，混淆矩阵仍能反映用户间的天然相似度
    - 找出那些最不容易与其他用户混淆的"独特"用户
    - 这些用户有明显特征，更容易学习和生成
    """
    print("=" * 60)
    print("基于分类器性能评估用户可分性")
    print("=" * 60)
    
    # 先初始化设备，避免作用域问题
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载已训练的分类器
    print(f"尝试从 {classifier_path} 加载分类器...")
    
    try:
        # 直接使用 weights_only=False，因为这是信任的模型
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
        
        # 调试：查看checkpoint的结构
        print(f"加载的checkpoint类型: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint包含的键: {list(checkpoint.keys())}")
            
            # 从 checkpoint 中获取模型参数
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
                print(f"检测到类别数: {num_classes}")
            else:
                num_classes = 31  # 默认值
                print(f"使用默认类别数: {num_classes}")
            
            # 创建模型实例
            classifier = DomainAdaptiveClassifier(num_classes=num_classes)
            
            # 加载模型参数
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("使用 'model_state_dict' 加载参数")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("使用 'state_dict' 加载参数")
            else:
                print("❌ 未找到模型参数")
                return None, None, None
            
            # 加载参数到模型
            classifier.load_state_dict(state_dict)
            classifier = classifier.to(device)
            classifier.eval()
            
            print(f"✅ 成功加载分类器: {classifier_path}")
            print(f"   模型类型: {type(classifier)}")
            print(f"   类别数: {num_classes}")
            
        else:
            print(f"❌ Checkpoint不是字典格式: {type(checkpoint)}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ 加载分类器失败: {e}")
        print("无法进行用户筛选，退出程序")
        return None, None, None
    
    # 2. 对每个用户进行预测，构建混淆矩阵
    all_predictions = []
    all_true_labels = []
    user_accuracies = {}
    user_sample_counts = {}
    
    print("\n开始评估各用户分类性能...")
    
    for user_id in range(1, 32):
        user_dir = Path(dataset_path) / f'ID_{user_id}'
        if not user_dir.exists():
            continue
            
        user_predictions = []
        user_labels = []
        
        # 对该用户所有图像进行预测
        image_files = list(user_dir.glob('*.jpg'))
        user_sample_counts[user_id] = len(image_files)
        
        print(f"处理 ID_{user_id}: {len(image_files)} 张图像", end=" ")
        
        correct = 0
        for img_path in image_files[:50]:  # 限制每用户50张以提高速度
            try:
                # 预处理图像
                img_tensor = load_and_preprocess_image(img_path).to(device)
                
                # 预测
                with torch.no_grad():
                    # DomainAdaptiveClassifier 返回 (logits, features)
                    outputs, features = classifier(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred_class_idx = torch.argmax(probs, dim=1).item()
                    
                    # 分类器输出的是类别索引（0-30），需要映射到用户ID（1-31）
                    pred_user_id = pred_class_idx + 1
                    
                    # 验证预测结果在有效范围内
                    if pred_user_id < 1 or pred_user_id > 31:
                        continue  # 跳过无效预测
                    
                user_predictions.append(pred_user_id)
                user_labels.append(user_id)
                
                if pred_user_id == user_id:
                    correct += 1
                    
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
                continue
        
        if user_labels:
            accuracy = correct / len(user_labels)
            user_accuracies[user_id] = accuracy
            all_predictions.extend(user_predictions)
            all_true_labels.extend(user_labels)
            print(f"→ 准确率: {accuracy:.2%}")
        else:
            print("→ 无有效预测")
    
    if not all_predictions:
        print("❌ 没有获得任何有效预测")
        return None, None, None
    
    print(f"\n总共处理了 {len(all_predictions)} 个预测")
    
    # 3. 构建混淆矩阵分析
    unique_labels = sorted(set(all_true_labels))
    cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_labels)
    
    print(f"\n构建混淆矩阵: {cm.shape}")
    
    # 4. 计算用户间的混淆关系
    print("\n分析用户间的混淆关系...")
    
    # 将混淆矩阵转换为混淆强度矩阵
    confusion_strength = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            # 计算每个用户被误分为其他用户的比例
            confusion_strength[i, :] = cm[i, :] / row_sum
    
    # 计算对称混淆强度（两个用户互相混淆的程度）
    symmetric_confusion = (confusion_strength + confusion_strength.T) / 2
    
    # 5. 计算每个用户的"独特性"指标
    user_metrics = {}
    
    for i, user_id in enumerate(unique_labels):
        if i >= len(cm):
            continue
        
        # 指标1: 自身识别率（对角线值）
        self_recognition = confusion_strength[i, i]
        
        # 指标2: 与其他用户的平均混淆度
        other_confusion = symmetric_confusion[i, :].copy()
        other_confusion[i] = 0  # 排除自己
        avg_confusion_with_others = np.mean(other_confusion)
        max_confusion_with_others = np.max(other_confusion)
        
        # 指标3: 混淆伙伴数（与多少用户有明显混淆，阈值>5%）
        confusion_partners = np.sum(other_confusion > 0.05)
        
        # 指标4: 独特性得分
        # 高自识别 + 低与其他用户混淆 + 少混淆伙伴 = 高独特性
        uniqueness_score = self_recognition - avg_confusion_with_others * 2 - confusion_partners * 0.01
        
        # 指标5: 难度分数（反向指标，值越低越容易学习）
        difficulty_score = 1 - self_recognition + avg_confusion_with_others
        
        user_metrics[user_id] = {
            'self_recognition': self_recognition,
            'avg_confusion': avg_confusion_with_others,
            'max_confusion': max_confusion_with_others,
            'confusion_partners': confusion_partners,
            'uniqueness_score': uniqueness_score,
            'difficulty_score': difficulty_score,
            'sample_count': user_sample_counts.get(user_id, 0)
        }
        
        print(f"ID_{user_id}: 自识别={self_recognition:.2%}, "
              f"平均混淆={avg_confusion_with_others:.2%}, "
              f"混淆伙伴={confusion_partners}, "
              f"独特性={uniqueness_score:.3f}")
    
    # 6. 找出最容易互相混淆的用户组
    print("\n最容易混淆的用户对:")
    confusion_pairs = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            mutual_confusion = symmetric_confusion[i, j]
            if mutual_confusion > 0.1:  # 互相混淆超过10%
                confusion_pairs.append((
                    unique_labels[i], 
                    unique_labels[j], 
                    mutual_confusion
                ))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for user1, user2, conf in confusion_pairs[:5]:
        print(f"  ID_{user1} <-> ID_{user2}: {conf:.2%}")
    
    # 7. 选择最“独特”的用户（多种策略）
    
    # 策略1: 按独特性得分排序（推荐）
    by_uniqueness = sorted(user_metrics.items(), 
                          key=lambda x: x[1]['uniqueness_score'], 
                          reverse=True)
    
    # 策略2: 按最低混淆度排序
    by_low_confusion = sorted(user_metrics.items(), 
                             key=lambda x: x[1]['avg_confusion'], 
                             reverse=False)  # 混淆度越低越好
    
    # 策略3: 按难度排序（选择最容易的）
    by_easiest = sorted(user_metrics.items(),
                       key=lambda x: x[1]['difficulty_score'],
                       reverse=False)  # 难度越低越好
    
    # 策略4: 排除混淆组后选择
    # 先找出所有高度混淆的用户
    confused_users = set()
    for user1, user2, conf in confusion_pairs:
        if conf > 0.15:  # 混淆度超过15%
            confused_users.add(user1)
            confused_users.add(user2)
    
    # 从非混淆用户中选择
    non_confused = [(uid, metrics) for uid, metrics in user_metrics.items() 
                    if uid not in confused_users]
    by_non_confused = sorted(non_confused,
                            key=lambda x: x[1]['uniqueness_score'],
                            reverse=True)
    
    print("\n" + "=" * 60)
    print("用户选择结果")
    print("=" * 60)
    
    strategies = {
        'uniqueness': ([x[0] for x in by_uniqueness[:8]], "基于独特性得分"),
        'low_confusion': ([x[0] for x in by_low_confusion[:8]], "基于低混淆度"),
        'easiest': ([x[0] for x in by_easiest[:8]], "最容易学习的用户"),
        'non_confused': ([x[0] for x in by_non_confused[:8]] if len(by_non_confused) >= 8 
                        else [x[0] for x in by_uniqueness[:8]], 
                        "排除混淆用户后选择")
    }
    
    for strategy_name, (users, description) in strategies.items():
        print(f"\n{description}:")
        print(f"选择的8个用户: {users}")
        if users:
            avg_uniqueness = np.mean([user_metrics[u]['uniqueness_score'] for u in users if u in user_metrics])
            avg_confusion = np.mean([user_metrics[u]['avg_confusion'] for u in users if u in user_metrics])
            print(f"平均独特性: {avg_uniqueness:.3f}")
            print(f"平均混淆度: {avg_confusion:.2%}")
    
    # 推荐最佳策略：选择独特性最高的用户
    best_users = strategies['uniqueness'][0]  
    
    print("\n" + "=" * 60)
    print("最终建议")
    print("=" * 60)
    print("选择独特性最高的8个用户：")
    print(f"  {best_users}")
    print("\n理由：")
    print("  1. 这些用户有明显特征，不容易与其他用户混淆")
    print("  2. 模型更容易学习这些用户的特征")
    print("  3. 生成质量预期更高，通过率更好")
    
    # 警告高度混淆的用户
    if confused_users:
        print("\n⚠️  建议避免的用户（高度混淆）：")
        print(f"  {sorted(list(confused_users)[:5])}")
        print("  这些用户之间相似度太高，难以区分")
    
    return best_users, user_accuracies, user_metrics

def select_by_generation_quality(generated_samples_dir='/kaggle/working/generated'):
    """
    基于生成质量筛选用户
    选择生成样本通过率最高的用户
    """
    print("基于生成质量筛选...")
    
    user_quality = {}
    
    for user_id in range(1, 32):
        user_generated = Path(generated_samples_dir) / f'ID_{user_id}'
        
        if not user_generated.exists():
            continue
            
        # 统计该用户的生成质量
        total_generated = len(list(user_generated.glob('*.jpg')))
        high_quality = len(list(user_generated.glob('*_pass.jpg')))  # 假设高质量样本有特殊标记
        
        if total_generated > 0:
            pass_rate = high_quality / total_generated
            user_quality[user_id] = pass_rate
            print(f"ID_{user_id}: 生成通过率 {pass_rate:.2%}")
    
    # 选择生成质量最高的用户
    sorted_users = sorted(user_quality.items(), key=lambda x: x[1], reverse=True)
    best_users = [user_id for user_id, _ in sorted_users[:8]]
    
    return best_users, user_quality

def select_by_data_distribution():
    """
    基于数据分布特征筛选
    选择特征分布差异最大的用户
    """
    print("基于数据分布筛选...")
    
    # 加载所有用户的latent特征
    user_features = {}
    
    for user_id in range(1, 32):
        latent_path = f'/kaggle/working/latents/ID_{user_id}_latents.npy'
        if Path(latent_path).exists():
            latents = np.load(latent_path)
            # 计算分布特征
            user_features[user_id] = {
                'mean': np.mean(latents, axis=0),
                'std': np.std(latents, axis=0),
                'skew': np.mean((latents - np.mean(latents, axis=0))**3, axis=0)
            }
    
    # 计算用户间的分布距离
    distances = {}
    for i in range(1, 32):
        if i not in user_features:
            continue
        min_dist = float('inf')
        for j in range(1, 32):
            if i == j or j not in user_features:
                continue
            # KL散度或其他分布距离
            dist = np.linalg.norm(user_features[i]['mean'] - user_features[j]['mean'])
            min_dist = min(min_dist, dist)
        distances[i] = min_dist
    
    # 选择分布差异最大的用户
    sorted_users = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    best_users = [user_id for user_id, _ in sorted_users[:8]]
    
    return best_users

def main():
    """
    主函数：执行用户筛选分析
    """
    print("正在进行基于分类器性能的用户筛选...")
    
    # 运行分类器分析
    best_users, accuracies, metrics = evaluate_user_separability()
    
    if best_users is None:
        print("\n" + "=" * 60)
        print("❌ 分类器加载失败，无法进行用户筛选")
        print("请检查分类器文件路径和格式")
        print("=" * 60)
        return None
    
    # 保存结果
    result = {
        'selected_users': best_users,
        'method': 'classifier_based',
        'description': '基于混淆矩阵分析的用户选择',
        'metrics': metrics,
        'accuracies': accuracies
    }
    
    output_path = '/kaggle/working/classifier_based_user_selection.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 60)
    print("最终推荐")
    print("=" * 60)
    print(f"选择的8个用户: {best_users}")
    print("选择依据: 混淆矩阵分离度分析")
    print("理论支撑: Feature Subset Selection & Class Separability")
    
    return best_users

if __name__ == "__main__":
    main()
