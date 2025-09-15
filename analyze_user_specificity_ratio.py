"""
分析比例模式下的user_specificity分布
重新确定合理的阈值
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def simulate_user_specificity_distributions():
    """模拟不同场景下的user_specificity分布"""
    
    # 模拟不同质量的预测结果
    scenarios = {
        'high_quality': {  # 高质量：目标用户明显领先
            'user_probs': np.random.beta(8, 2, 1000),  # 偏向高值
            'other_probs': np.random.beta(2, 8, 1000)  # 偏向低值
        },
        'medium_quality': {  # 中等质量：有一定区分度
            'user_probs': np.random.beta(5, 3, 1000),
            'other_probs': np.random.beta(3, 5, 1000)
        },
        'low_quality': {  # 低质量：区分度不明显
            'user_probs': np.random.beta(3, 5, 1000),
            'other_probs': np.random.beta(4, 4, 1000)
        }
    }
    
    results = {}
    
    for scenario_name, data in scenarios.items():
        user_probs = data['user_probs']
        other_probs = data['other_probs']
        
        # 差值模式
        diff_specificity = user_probs - other_probs
        
        # 比例模式 
        ratio_specificity = user_probs / (user_probs + other_probs)
        
        results[scenario_name] = {
            'diff_mode': diff_specificity,
            'ratio_mode': ratio_specificity,
            'user_probs': user_probs,
            'other_probs': other_probs
        }
    
    return results

def analyze_threshold_effects(results):
    """分析不同阈值的效果"""
    
    print("🔍 不同阈值下的通过率分析\n")
    
    # 差值模式阈值
    diff_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # 比例模式阈值 
    ratio_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    for scenario_name, data in results.items():
        print(f"📊 {scenario_name.upper()}场景:")
        
        print("   差值模式通过率:")
        for thresh in diff_thresholds:
            pass_rate = np.mean(data['diff_mode'] >= thresh) * 100
            print(f"     阈值{thresh:.1f}: {pass_rate:.1f}%")
        
        print("   比例模式通过率:")
        for thresh in ratio_thresholds:
            pass_rate = np.mean(data['ratio_mode'] >= thresh) * 100
            print(f"     阈值{thresh:.2f}: {pass_rate:.1f}%")
        
        # 统计信息
        ratio_mean = np.mean(data['ratio_mode'])
        ratio_25 = np.percentile(data['ratio_mode'], 25)
        ratio_50 = np.percentile(data['ratio_mode'], 50)
        ratio_75 = np.percentile(data['ratio_mode'], 75)
        
        print(f"   比例模式统计: 均值={ratio_mean:.3f}, 25%={ratio_25:.3f}, 50%={ratio_50:.3f}, 75%={ratio_75:.3f}\n")

def convert_evaluation_data_to_ratio():
    """将评估数据中的差值模式转换为比例模式进行对比"""
    
    # 基于之前评估结果，模拟典型的差值分布
    # 评估显示差值模式平均值0.6+
    np.random.seed(42)
    
    # 模拟评估数据：正确预测的样本
    n_samples = 2000
    user_probs = np.random.beta(7, 2, n_samples) * 0.5 + 0.5  # 0.5-1.0范围
    other_probs = np.random.beta(2, 5, n_samples) * 0.4 + 0.0  # 0.0-0.4范围
    
    # 确保user_probs > other_probs (正确预测)
    mask = user_probs > other_probs
    user_probs = user_probs[mask]
    other_probs = other_probs[mask]
    
    # 计算两种模式
    diff_specificity = user_probs - other_probs
    ratio_specificity = user_probs / (user_probs + other_probs)
    
    print("🎯 基于模拟评估数据的分析:")
    print(f"   差值模式: 均值={np.mean(diff_specificity):.3f} (应该接近0.6+)")
    print(f"   比例模式: 均值={np.mean(ratio_specificity):.3f}")
    
    # 分析比例模式的分位数
    percentiles = [25, 50, 75, 90, 95]
    print("   比例模式分位数:")
    for p in percentiles:
        value = np.percentile(ratio_specificity, p)
        print(f"     {p}%分位: {value:.3f}")
    
    return ratio_specificity

def recommend_threshold(ratio_data):
    """基于数据推荐合理阈值"""
    
    # 分析不同阈值的效果
    thresholds = np.arange(0.55, 0.85, 0.01)
    pass_rates = []
    
    for thresh in thresholds:
        pass_rate = np.mean(ratio_data >= thresh) * 100
        pass_rates.append(pass_rate)
    
    # 找到合适的通过率区间 (15-30%)
    target_range = (15, 30)
    suitable_thresholds = []
    
    for i, (thresh, rate) in enumerate(zip(thresholds, pass_rates)):
        if target_range[0] <= rate <= target_range[1]:
            suitable_thresholds.append((thresh, rate))
    
    if suitable_thresholds:
        # 选择中位数阈值
        mid_idx = len(suitable_thresholds) // 2
        recommended_thresh, recommended_rate = suitable_thresholds[mid_idx]
        
        print(f"\n🎯 推荐阈值分析:")
        print(f"   目标通过率范围: {target_range[0]}-{target_range[1]}%")
        print(f"   推荐阈值: {recommended_thresh:.3f}")
        print(f"   对应通过率: {recommended_rate:.1f}%")
        
        # 显示附近的选项
        print("   附近选项:")
        for thresh, rate in suitable_thresholds[max(0, mid_idx-2):mid_idx+3]:
            marker = " ⭐" if abs(thresh - recommended_thresh) < 0.001 else ""
            print(f"     {thresh:.3f}: {rate:.1f}%{marker}")
        
        return recommended_thresh
    else:
        print("⚠️  未找到合适的阈值范围")
        return 0.65

def main():
    print("🔍 User Specificity 阈值优化分析\n")
    
    # 1. 模拟不同场景
    print("=" * 50)
    print("1. 不同质量场景的分布分析")
    print("=" * 50)
    results = simulate_user_specificity_distributions()
    analyze_threshold_effects(results)
    
    # 2. 转换评估数据
    print("=" * 50) 
    print("2. 基于评估数据的分析")
    print("=" * 50)
    ratio_data = convert_evaluation_data_to_ratio()
    
    # 3. 推荐阈值
    print("=" * 50)
    print("3. 阈值推荐")
    print("=" * 50)
    recommended = recommend_threshold(ratio_data)
    
    # 4. 对比0.65
    current_pass_rate = np.mean(ratio_data >= 0.65) * 100
    recommended_pass_rate = np.mean(ratio_data >= recommended) * 100
    
    print(f"\n📊 阈值对比:")
    print(f"   当前0.65阈值: {current_pass_rate:.1f}%通过率")
    print(f"   推荐{recommended:.3f}阈值: {recommended_pass_rate:.1f}%通过率")
    
    if recommended > 0.65:
        print(f"   💡 建议提高阈值到 {recommended:.3f} 以获得更严格的筛选")
    elif recommended < 0.65:
        print(f"   💡 建议降低阈值到 {recommended:.3f} 以获得更合理的通过率")
    else:
        print(f"   ✅ 当前0.65阈值已经合适")

if __name__ == "__main__":
    main()
