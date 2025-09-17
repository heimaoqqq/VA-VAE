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
    基于混淆矩阵分析，选择最容易区分的用户
    
    理论依据：
    - 选择对角线值高（正确分类率高）的类别
    - 选择非对角线值低（误分类率低）的类别
    - 计算每个类别的分离度指标
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
    
    # 4. 计算各种可分性指标
    user_metrics = {}
    
    for i, user_id in enumerate(unique_labels):
        if i >= len(cm):
            continue
            
        # 指标1: 准确率（对角线值 / 行总和）
        accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        
        # 指标2: 精确率（对角线值 / 列总和）
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        
        # 指标3: 混淆度（被误分类的比例）
        confusion_rate = 1 - accuracy
        
        # 指标4: 类别分离度（与最容易混淆类别的区分度）
        max_confusion_with_other = 0
        if cm[i, :].sum() > 0:
            other_class_errors = cm[i, :].copy()
            other_class_errors[i] = 0  # 排除自己
            max_confusion_with_other = np.max(other_class_errors) / cm[i, :].sum()
        
        # 综合得分：高准确率 + 高精确率 + 低混淆度
        separability_score = (accuracy + precision) / 2 - confusion_rate * 0.5
        
        user_metrics[user_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'confusion_rate': confusion_rate,
            'max_confusion': max_confusion_with_other,
            'separability_score': separability_score,
            'sample_count': user_sample_counts.get(user_id, 0)
        }
        
        print(f"ID_{user_id}: 准确率={accuracy:.2%}, 精确率={precision:.2%}, "
              f"混淆度={confusion_rate:.2%}, 分离度={separability_score:.3f}")
    
    # 5. 选择最容易分离的用户（多种策略）
    
    # 策略1: 按分离度得分排序
    by_separability = sorted(user_metrics.items(), 
                            key=lambda x: x[1]['separability_score'], 
                            reverse=True)
    
    # 策略2: 按准确率排序
    by_accuracy = sorted(user_metrics.items(), 
                        key=lambda x: x[1]['accuracy'], 
                        reverse=True)
    
    # 策略3: 混合策略（准确率 + 样本数）
    by_hybrid = sorted(user_metrics.items(), 
                      key=lambda x: x[1]['accuracy'] * 0.7 + 
                                   min(x[1]['sample_count']/50, 1.0) * 0.3, 
                      reverse=True)
    
    print("\n" + "=" * 60)
    print("用户选择结果")
    print("=" * 60)
    
    strategies = {
        'separability': ([x[0] for x in by_separability[:8]], "基于分离度得分"),
        'accuracy': ([x[0] for x in by_accuracy[:8]], "基于准确率"),
        'hybrid': ([x[0] for x in by_hybrid[:8]], "混合策略（准确率+样本数）")
    }
    
    for strategy_name, (users, description) in strategies.items():
        print(f"\n{description}:")
        print(f"选择的8个用户: {users}")
        avg_accuracy = np.mean([user_metrics[u]['accuracy'] for u in users if u in user_metrics])
        print(f"平均准确率: {avg_accuracy:.2%}")
    
    # 推荐最佳策略
    best_users = strategies['hybrid'][0]  # 默认使用混合策略
    
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
