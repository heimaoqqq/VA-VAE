#!/usr/bin/env python3
"""
简化的RMSNorm实现
用于替代fairscale依赖
"""

import torch
import torch.nn as nn
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    简化版本，不依赖fairscale
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: 特征维度
            eps: 数值稳定性的小值
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """计算RMS归一化"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FusedRMSNorm(RMSNorm):
    """
    融合的RMSNorm实现
    为了兼容性，继承自RMSNorm
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim, eps)

# 为了兼容性，提供别名
RMSNormFused = FusedRMSNorm
