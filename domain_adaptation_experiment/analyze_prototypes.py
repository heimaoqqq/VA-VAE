#!/usr/bin/env python3
"""
分析原型质量和分布
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def analyze_prototypes(prototype_path):
    """分析原型的质量和分布"""
    
    print("🔍 Prototype Analysis")
    print("="*50)
    
    # 加载原型
    proto_dict = torch.load(prototype_path, weights_only=False)
    prototypes = proto_dict['prototypes'].numpy()
    
    print(f"📊 基本信息:")
    print(f"   原型形状: {prototypes.shape}")
    print(f"   支持集大小: {proto_dict['metadata']['support_size']}")
    print(f"   总支持样本: {proto_dict['metadata']['num_support_samples']}")
    
    # 计算原型间的相似度
    similarity_matrix = cosine_similarity(prototypes)
    
    # 统计相似度
    # 去除对角线（自相似度）
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    print(f"\n📈 原型间相似度分析:")
    print(f"   平均相似度: {off_diagonal.mean():.3f}")
    print(f"   最大相似度: {off_diagonal.max():.3f}")
    print(f"   最小相似度: {off_diagonal.min():.3f}")
    print(f"   标准差: {off_diagonal.std():.3f}")
    
    # 找出最相似的原型对
    similarity_matrix_copy = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix_copy, -1)  # 忽略对角线
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix_copy), similarity_matrix_copy.shape)
    
    print(f"\n🔝 最相似的原型对:")
    print(f"   ID_{max_sim_idx[0]+1} ↔ ID_{max_sim_idx[1]+1}")
    print(f"   相似度: {similarity_matrix[max_sim_idx]:.3f}")
    
    # 计算原型的范数
    norms = np.linalg.norm(prototypes, axis=1)
    print(f"\n📐 原型范数分析:")
    print(f"   平均范数: {norms.mean():.3f}")
    print(f"   最大范数: {norms.max():.3f}")
    print(f"   最小范数: {norms.min():.3f}")
    print(f"   标准差: {norms.std():.3f}")
    
    # 判断是否有异常
    print(f"\n🚨 异常检测:")
    
    # 检查1：原型过于相似
    if off_diagonal.mean() > 0.7:
        print(f"   ⚠️ 原型间相似度过高 ({off_diagonal.mean():.3f})")
        print(f"      可能原因：特征提取器区分能力不足")
    
    # 检查2：原型分布不均
    if norms.std() > 0.5:
        print(f"   ⚠️ 原型范数方差过大 ({norms.std():.3f})")
        print(f"      可能原因：某些类的支持样本质量不一致")
    
    # 检查3：是否所有原型都已归一化
    if abs(norms.mean() - 1.0) < 0.01:
        print(f"   ✅ 原型已L2归一化")
    else:
        print(f"   ❌ 原型未归一化")
    
    # 极低温度的影响分析
    print(f"\n🌡️ 温度参数影响分析:")
    for tau in [0.005, 0.01, 0.05, 0.1]:
        # 模拟一个查询与原型的相似度
        sim = np.array([0.9, 0.85, 0.8] + [0.6]*28)  # 模拟相似度
        probs = np.exp(sim/tau) / np.sum(np.exp(sim/tau))
        top1_prob = probs[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        print(f"   τ={tau:.3f}: Top-1概率={top1_prob:.3f}, 熵={entropy:.3f}")
    
    print(f"\n💡 关键发现:")
    print(f"   极低温度(τ=0.005)使得决策几乎是one-hot")
    print(f"   这相当于最近邻分类器，完全依赖原型质量")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototype-path', type=str,
                       default='/kaggle/working/improved_prototypes_split.pt')
    args = parser.parse_args()
    
    analyze_prototypes(args.prototype_path)
