"""
简化版RMSNorm实现
当fairscale不可用时的备用方案
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    简化版实现，不依赖fairscale
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: 特征维度
            eps: 数值稳定性参数
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


# 为了兼容性，也提供一个别名
RMSNormSimple = RMSNorm
