#!/usr/bin/env python3
"""
域适应评估配置文件
定义所有超参数搜索空间和评估策略
"""

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'source_domain': 'normal_gait',  # 源域：正常步态
    'target_domain': 'backpack_gait',  # 目标域：背包步态
    'support_sizes': [3, 5, 10],  # 测试不同的few-shot设置
    'random_seeds': [42, 123, 456],  # 多个随机种子确保结果稳定
}

# ==================== PNC配置 ====================
PNC_CONFIG = {
    'fusion_alphas': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 原型权重
    'similarity_taus': [0.005, 0.01, 0.02, 0.05, 0.1],  # 温度参数
    'use_adaptive_fusion': [True, False],  # 是否使用自适应融合
}

# ==================== LCCS配置 ====================
LCCS_CONFIG = {
    'methods': ['progressive', 'weighted'],  # LCCS方法
    
    # Progressive方法参数
    'progressive': {
        'momentums': [0.001, 0.005, 0.01, 0.02],
        'iterations': [3, 5, 10],
    },
    
    # Weighted方法参数
    'weighted': {
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5],  # 目标域权重
    }
}

# ==================== NCC配置 ====================
NCC_CONFIG = {
    'temperatures': [0.005, 0.01, 0.02, 0.05],  # 温度参数
    'distance_metrics': ['cosine', 'euclidean'],  # 距离度量
}

# ==================== 原型构建策略 ====================
PROTOTYPE_STRATEGIES = [
    'simple_mean',     # 简单均值
    'weighted_mean',   # 加权均值（基于置信度）
    'diversity',       # 多样性选择（K-means）
    'uncertainty',     # 基于不确定性选择
]

# ==================== 快速测试配置 ====================
QUICK_TEST_CONFIG = {
    'support_sizes': [3],
    'random_seeds': [42],
} 
