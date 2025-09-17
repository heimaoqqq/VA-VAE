"""
基于数据特征分析选择最有特色的用户
不依赖分类器，直接分析图像/潜在空间的特征分布
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, wasserstein_distance
import torchvision.transforms as transforms
from tqdm import tqdm

def extract_image_features(image_path, method='statistics'):
    """
    提取图像特征
    method: 'statistics' - 基本统计特征
            'histogram' - 颜色直方图
            'gradient' - 梯度特征
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        if method == 'statistics':
            # 基本统计特征
            features = []
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.max(channel_data) - np.min(channel_data)  # 动态范围
                ])
            return np.array(features)
            
        elif method == 'histogram':
            # 颜色直方图
            features = []
            for channel in range(3):
                hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
                hist = hist / hist.sum()  # 归一化
                features.extend(hist)
            return np.array(features)
            
        elif method == 'gradient':
            # 梯度特征（边缘信息）
            gray = np.mean(img_array, axis=2)
            # 简单的Sobel算子
            dx = np.abs(np.diff(gray, axis=1))
            dy = np.abs(np.diff(gray, axis=0))
            gradient_magnitude = np.sqrt(dx[:-1, :]**2 + dy[:, :-1]**2)
            features = [
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.percentile(gradient_magnitude, 90)  # 强边缘
            ]
            return np.array(features)
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_user_characteristics(dataset_path='/kaggle/input/dataset'):
    """
    计算每个用户的数据特征
    """
    print("=" * 60)
    print("分析用户数据特征")
    print("=" * 60)
    
    user_features = {}
    user_stats = {}
    
    # 遍历所有用户
    for user_id in range(1, 32):
        user_dir = Path(dataset_path) / f'ID_{user_id}'
        if not user_dir.exists():
            continue
            
        print(f"\n处理 ID_{user_id}...")
        
        # 收集该用户的所有图像特征
        stat_features = []
        hist_features = []
        grad_features = []
        
        image_files = list(user_dir.glob('*.jpg'))[:50]  # 限制数量以加快速度
        
        for img_path in tqdm(image_files, desc=f"ID_{user_id}", leave=False):
            stat_feat = extract_image_features(img_path, 'statistics')
            hist_feat = extract_image_features(img_path, 'histogram')
            grad_feat = extract_image_features(img_path, 'gradient')
            
            if stat_feat is not None:
                stat_features.append(stat_feat)
            if hist_feat is not None:
                hist_features.append(hist_feat)
            if grad_feat is not None:
                grad_features.append(grad_feat)
        
        if len(stat_features) > 0:
            stat_features = np.array(stat_features)
            hist_features = np.array(hist_features)
            grad_features = np.array(grad_features)
            
            # 计算用户特征的统计量
            user_stats[user_id] = {
                # 特征均值（用户的"中心"特征）
                'mean_stat': np.mean(stat_features, axis=0),
                'mean_hist': np.mean(hist_features, axis=0),
                'mean_grad': np.mean(grad_features, axis=0),
                
                # 特征方差（用户内部的多样性）
                'var_stat': np.var(stat_features, axis=0).mean(),
                'var_hist': np.var(hist_features, axis=0).mean(),
                'var_grad': np.var(grad_features, axis=0).mean(),
                
                # 样本数量
                'num_samples': len(stat_features)
            }
            
            # 保存原始特征用于后续分析
            user_features[user_id] = {
                'stat': stat_features,
                'hist': hist_features,
                'grad': grad_features
            }
            
            print(f"  统计方差: {user_stats[user_id]['var_stat']:.4f}")
            print(f"  直方图方差: {user_stats[user_id]['var_hist']:.6f}")
            print(f"  梯度方差: {user_stats[user_id]['var_grad']:.4f}")
    
    return user_features, user_stats

def analyze_user_separability(user_features, user_stats):
    """
    分析用户间的可分性
    """
    print("\n" + "=" * 60)
    print("计算用户间的差异性")
    print("=" * 60)
    
    user_ids = sorted(user_stats.keys())
    n_users = len(user_ids)
    
    # 1. 计算用户间的距离矩阵
    distance_matrix = np.zeros((n_users, n_users))
    
    for i, uid1 in enumerate(user_ids):
        for j, uid2 in enumerate(user_ids):
            if i == j:
                continue
                
            # 基于直方图的距离（Wasserstein距离）
            hist1 = user_stats[uid1]['mean_hist']
            hist2 = user_stats[uid2]['mean_hist']
            
            # 分别计算RGB三个通道的距离
            distances = []
            for c in range(3):
                h1 = hist1[c*32:(c+1)*32]
                h2 = hist2[c*32:(c+1)*32]
                # 使用JS散度（对称的KL散度）
                m = (h1 + h2) / 2
                js_div = 0.5 * entropy(h1, m) + 0.5 * entropy(h2, m)
                distances.append(js_div)
            
            distance_matrix[i, j] = np.mean(distances)
    
    # 2. 计算每个用户的独特性指标
    user_metrics = {}
    
    for i, uid in enumerate(user_ids):
        # 该用户到其他所有用户的平均距离
        avg_distance = np.mean(distance_matrix[i, :])
        
        # 最近邻距离（最相似用户的距离）
        other_distances = distance_matrix[i, :].copy()
        other_distances[i] = np.inf  # 排除自己
        nearest_distance = np.min(other_distances)
        
        # 用户内部的一致性（方差越小越一致）
        internal_consistency = 1.0 / (1.0 + user_stats[uid]['var_hist'])
        
        # 独特性得分：远离其他用户 + 内部一致
        uniqueness = avg_distance * internal_consistency
        
        # 难易度：最近邻越远越容易区分
        ease_score = nearest_distance
        
        user_metrics[uid] = {
            'avg_distance': avg_distance,
            'nearest_distance': nearest_distance,
            'internal_consistency': internal_consistency,
            'uniqueness_score': uniqueness,
            'ease_score': ease_score,
            'var_stat': user_stats[uid]['var_stat'],
            'var_hist': user_stats[uid]['var_hist'],
            'var_grad': user_stats[uid]['var_grad']
        }
        
        print(f"ID_{uid}: 平均距离={avg_distance:.4f}, "
              f"最近邻={nearest_distance:.4f}, "
              f"独特性={uniqueness:.4f}")
    
    # 3. 找出相似用户组
    print("\n最相似的用户对：")
    similar_pairs = []
    for i in range(n_users):
        for j in range(i+1, n_users):
            similarity = 1.0 / (1.0 + distance_matrix[i, j])
            if similarity > 0.9:  # 高度相似
                similar_pairs.append((user_ids[i], user_ids[j], similarity))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    for uid1, uid2, sim in similar_pairs[:5]:
        print(f"  ID_{uid1} <-> ID_{uid2}: 相似度={sim:.3f}")
    
    return user_metrics, distance_matrix, similar_pairs

def select_best_users(user_metrics, n_select=8):
    """
    选择最佳用户子集
    """
    print("\n" + "=" * 60)
    print("用户选择结果")
    print("=" * 60)
    
    # 策略1：选择独特性最高的
    by_uniqueness = sorted(user_metrics.items(),
                          key=lambda x: x[1]['uniqueness_score'],
                          reverse=True)
    
    # 策略2：选择最容易区分的（最近邻距离大）
    by_ease = sorted(user_metrics.items(),
                    key=lambda x: x[1]['ease_score'],
                    reverse=True)
    
    # 策略3：选择内部一致性高的
    by_consistency = sorted(user_metrics.items(),
                          key=lambda x: x[1]['internal_consistency'],
                          reverse=True)
    
    # 策略4：综合策略
    # 归一化各项指标
    def normalize_scores(metrics_dict, key):
        values = [m[key] for m in metrics_dict.values()]
        min_val, max_val = min(values), max(values)
        if max_val - min_val > 0:
            for uid in metrics_dict:
                normalized = (metrics_dict[uid][key] - min_val) / (max_val - min_val)
                metrics_dict[uid][f'{key}_norm'] = normalized
        else:
            for uid in metrics_dict:
                metrics_dict[uid][f'{key}_norm'] = 0.5
    
    normalize_scores(user_metrics, 'uniqueness_score')
    normalize_scores(user_metrics, 'ease_score')
    normalize_scores(user_metrics, 'internal_consistency')
    
    # 综合得分
    for uid in user_metrics:
        user_metrics[uid]['combined_score'] = (
            0.4 * user_metrics[uid]['uniqueness_score_norm'] +
            0.3 * user_metrics[uid]['ease_score_norm'] +
            0.3 * user_metrics[uid]['internal_consistency_norm']
        )
    
    by_combined = sorted(user_metrics.items(),
                        key=lambda x: x[1]['combined_score'],
                        reverse=True)
    
    # 显示结果
    strategies = {
        'uniqueness': ([x[0] for x in by_uniqueness[:n_select]], "基于独特性"),
        'ease': ([x[0] for x in by_ease[:n_select]], "基于易区分度"),
        'consistency': ([x[0] for x in by_consistency[:n_select]], "基于内部一致性"),
        'combined': ([x[0] for x in by_combined[:n_select]], "综合评分")
    }
    
    for name, (users, desc) in strategies.items():
        print(f"\n{desc}:")
        print(f"  选择的用户: {users}")
        avg_uniqueness = np.mean([user_metrics[u]['uniqueness_score'] for u in users])
        print(f"  平均独特性: {avg_uniqueness:.4f}")
    
    # 推荐最佳策略
    best_users = strategies['combined'][0]
    
    print("\n" + "=" * 60)
    print("最终推荐")
    print("=" * 60)
    print(f"选择综合评分最高的{n_select}个用户：")
    print(f"  {best_users}")
    print("\n选择理由：")
    print("  1. 这些用户具有独特的视觉特征")
    print("  2. 与其他用户的差异明显")
    print("  3. 用户内部数据一致性好")
    print("  4. 预期生成质量和识别率都会较高")
    
    return best_users, strategies

def main():
    """
    主函数
    """
    print("基于数据特征分析的用户筛选")
    print("不依赖分类器，直接分析图像特征分布")
    print()
    
    # 1. 提取特征
    user_features, user_stats = compute_user_characteristics()
    
    # 2. 分析可分性
    user_metrics, distance_matrix, similar_pairs = analyze_user_separability(user_features, user_stats)
    
    # 3. 选择用户
    best_users, all_strategies = select_best_users(user_metrics)
    
    # 4. 保存结果
    result = {
        'selected_users': [int(u) for u in best_users],  # 转为int避免序列化问题
        'strategies': {k: [int(u) for u in users] for k, (users, _) in all_strategies.items()},
        'user_metrics': {int(k): v for k, v in user_metrics.items()},
        'method': 'feature_based',
        'description': '基于图像特征分析的用户选择'
    }
    
    output_path = '/kaggle/working/feature_based_user_selection.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    return best_users

if __name__ == "__main__":
    main()
