#!/usr/bin/env python3
"""
SimplifiedVAVAE: 简化版VA-VAE，禁用VF功能，仅保留重建能力
专门用于条件扩散模型的VAE编码器/解码器
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml

# 添加LightningDiT路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root / 'LightningDiT'))

# 设置taming路径
def setup_taming_path():
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),
        Path.cwd().parent / 'taming-transformers',
    ]
    for location in taming_locations:
        if location.exists() and str(location) not in sys.path:
            sys.path.insert(0, str(location))
            return True
    return False

setup_taming_path()

from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config


class SimplifiedVAVAE(nn.Module):
    """简化VA-VAE：支持VF功能以匹配预训练模型"""
    
    def __init__(self, checkpoint_path=None, use_vf='dinov2'):
        super().__init__()
        self.scale_factor = 1.0  # 默认值，从checkpoint中读取真实值
        self.use_vf = use_vf
        
        # 创建VA-VAE配置（启用VF以匹配预训练模型）
        config = OmegaConf.create({
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'monitor': 'val/rec_loss',
                'embed_dim': 32,
                'ddconfig': {
                    'double_z': True, 
                    'z_channels': 32, 
                    'resolution': 256,
                    'in_channels': 3, 
                    'out_ch': 3, 
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4], 
                    'num_res_blocks': 2,
                    'attn_resolutions': [16], 
                    'dropout': 0.0
                },
                'lossconfig': {
                    'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
                    'params': {
                        'disc_start': 50001,  # 推理时不使用判别器
                        'kl_weight': 1e-6,
                        'pixelloss_weight': 1.0,
                        'perceptual_weight': 1.0,
                        'disc_weight': 0.0,  # 推理时禁用判别器
                    }
                },
                'use_vf': use_vf,  # ✅ 启用VF以匹配预训练模型配置
                'reverse_proj': False,
            }
        })
        
        # 创建模型
        self.vae = instantiate_from_config(config)
        
        # 加载权重（如果提供）
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        
        # 冻结参数
        self.freeze()
    
    def load_checkpoint(self, checkpoint_path):
        """加载VA-VAE权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同格式的checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 提取真实的缩放因子
        if 'scale_factor' in checkpoint:
            self.scale_factor = float(checkpoint['scale_factor'])
            print(f"🔧 从checkpoint读取缩放因子: {self.scale_factor}")
        elif 'state_dict' in checkpoint and any('scale_factor' in k for k in checkpoint['state_dict'].keys()):
            # 寻找包含scale_factor的键
            for k, v in checkpoint['state_dict'].items():
                if 'scale_factor' in k and isinstance(v, torch.Tensor):
                    self.scale_factor = float(v.item())
                    print(f"🔧 从state_dict读取缩放因子: {self.scale_factor}")
                    break
        else:
            # 动态计算缩放因子的备用方案
            print(f"⚠️ 未在checkpoint中找到scale_factor，使用默认值1.0")
            print(f"   建议：训练时添加scale_by_std=True来动态计算")
        
        # 根据VF配置决定是否包含VF权重
        filtered_state_dict = {}
        if self.use_vf:
            # 启用VF时，保留VF相关权重
            excluded_prefixes = ['foundation_model']  # 仅排除foundation_model
        else:
            # 禁用VF时，排除所有VF相关权重
            excluded_prefixes = ['vf_proj', 'vf_model', 'foundation_model']
        
        for k, v in state_dict.items():
            # 检查是否包含需要排除的前缀
            should_exclude = any(prefix in k for prefix in excluded_prefixes)
            if not should_exclude:
                # 移除前缀（如果有）
                clean_key = k.replace('module.', '').replace('vae.', '')
                
                # 特殊处理：修复linear_proj权重形状不匹配
                if 'linear_proj.weight' in clean_key and len(v.shape) == 4:
                    # 检查是否需要转置
                    if hasattr(self.vae, 'linear_proj') and hasattr(self.vae.linear_proj, 'weight'):
                        expected_shape = self.vae.linear_proj.weight.shape
                        if v.shape != expected_shape and v.shape == expected_shape[::-1][:2] + expected_shape[2:]:
                            # 转置前两个维度
                            v = v.transpose(0, 1)
                            print(f"🔧 修复linear_proj权重形状: {expected_shape}")
                
                filtered_state_dict[clean_key] = v
        
        # 加载权重
        missing, unexpected = self.vae.load_state_dict(filtered_state_dict, strict=False)
        
        if missing:
            print(f"⚠️ 缺失的权重: {missing[:5]}...")  # 仅显示前5个
        if unexpected:
            print(f"⚠️ 未预期的权重: {unexpected[:5]}...")
        
        print(f"✅ 成功加载VA-VAE权重: {checkpoint_path}")
        print(f"📏 使用缩放因子: {self.scale_factor}")
    
    def freeze(self):
        """冻结VAE参数"""
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
    
    @torch.no_grad()
    def encode(self, x):
        """
        编码图像到latent空间
        Args:
            x: [B, 3, 256, 256] 图像张量
        Returns:
            z: [B, 32, 16, 16] latent张量（已缩放）
        """
        # 确保输入范围正确
        if x.min() >= 0 and x.max() <= 1:
            x = 2.0 * x - 1.0  # [0,1] -> [-1,1]
        
        # 编码
        posterior = self.vae.encode(x)
        z = posterior.sample()
        
        # 应用缩放
        z = z * self.scale_factor
        
        return z
    
    @torch.no_grad()
    def decode(self, z):
        """
        解码latent到图像
        Args:
            z: [B, 32, 16, 16] latent张量（已缩放）
        Returns:
            x: [B, 3, 256, 256] 图像张量
        """
        # 还原缩放
        z = z / self.scale_factor
        
        # 解码
        x = self.vae.decode(z)
        
        # 转换到[0,1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)
        
        return x
    
    def forward(self, x):
        """前向传播：编码后解码"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
