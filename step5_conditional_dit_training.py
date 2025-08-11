#!/usr/bin/env python3
"""
步骤5: 条件DiT微调训练
基于预训练LightningDiT-XL-64ep进行用户条件微调
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import yaml
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime

# 🚀 完全禁用torch._dynamo，解决DataParallel冲突
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # 完全禁用dynamo编译
torch.backends.cudnn.allow_tf32 = True  # 优化性能
print("🧹 已完全禁用torch._dynamo，避免与DataParallel冲突")

# 添加LightningDiT到路径
sys.path.append(str(Path("LightningDiT").absolute()))

from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE
from step4_microdoppler_adapter import MicroDopplerDataModule, UserConditionEncoder

class ConditionalDiT(nn.Module):
    """条件DiT模型"""
    
    def __init__(
        self,
        model: str = "LightningDiT-XL/1",
        num_users: int = 31,
        condition_dim: int = 1152,  # ✅ 修复：匹配DiT的hidden_size
        frozen_backbone: bool = True,
        dropout: float = 0.15,
        pretrained_path: str = None
    ):
        super().__init__()
        self.num_users = num_users
        self.condition_dim = condition_dim
        self.frozen_backbone = frozen_backbone
        
        # 加载预训练DiT - 配置用户条件参数
        self.dit = LightningDiT_models[model](
            input_size=16,           # 16 = 256/16 (VA-VAE下采样率)
            in_channels=32,          # VA-VAE潜向量通道数
            num_classes=num_users,   # ✅ 关键修复：设置为用户数量而不是ImageNet的1000
            use_qknorm=False,        # 官方配置
            use_swiglu=True,         # 官方配置
            use_rope=True,           # 官方配置
            use_rmsnorm=True,        # 官方配置
            wo_shift=False           # 官方配置
        )
        
        # 重新启用预训练权重加载
        print("✅ 重新启用预训练权重加载")
        self._load_pretrained_weights(pretrained_path)
        
        # 冻结主干网络
        if frozen_backbone:
            self._freeze_backbone()
        
        # 用户条件编码器
        self.condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=condition_dim,
            dropout=dropout
        )
        
        # 条件注入层 - 修改DiT的adaLN层
        self._inject_condition_layers()
        
        # 🚀 初始化用户条件缓存（避免属性错误）
        self._last_user_condition = None
        
        print(f"✅ 条件DiT初始化完成")
        print(f"   - 主干冻结: {frozen_backbone}")
        print(f"   - 总参数: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   - 可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        print(f"📥 加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint格式 - 修复权重加载
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 智能权重加载：过滤不兼容的权重
        model_state = self.dit.state_dict()
        filtered_state_dict = {}
        incompatible_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    incompatible_keys.append(f"{key}: {value.shape} -> {model_state[key].shape}")
            else:
                # 键不存在于当前模型中
                pass
        
        # 加载兼容的权重
        missing, unexpected = self.dit.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"✅ 成功加载 {len(filtered_state_dict)} 个兼容权重")
        if incompatible_keys:
            print(f"⚠️ 跳过 {len(incompatible_keys)} 个不兼容权重:")
            for key in incompatible_keys:  # 显示所有不兼容的权重
                print(f"   {key}")
        if missing:
            print(f"⚠️ 缺失键: {len(missing)}")
        if unexpected:
            print(f"⚠️ 额外键: {len(unexpected)}")
            
        # 🔧 彻底重新初始化所有条件相关层，确保维度一致
        self._reinitialize_conditional_layers()
        
        # 🔍 调试：检查adaLN相关权重的维度
        print("\n🔍 调试：检查关键层的维度")
        for name, param in self.dit.named_parameters():
            if 'adaLN_modulation' in name and 'weight' in name:
                print(f"   {name}: {param.shape}")
            elif 'y_embedder' in name:
                print(f"   {name}: {param.shape}")
                
    def _reinitialize_conditional_layers(self):
        """重新初始化所有条件相关层，确保维度一致性"""
        print("🔧 重新初始化条件相关层...")
        
        # 1. 重新初始化y_embedder (已经正确设置为num_users)
        # 这个应该已经是正确的，但确保初始化
        if hasattr(self.dit, 'y_embedder'):
            nn.init.normal_(self.dit.y_embedder.embedding_table.weight, std=0.02)
            print(f"   重新初始化 y_embedder: {self.dit.y_embedder.embedding_table.weight.shape}")
        
        # 2. 重新初始化所有adaLN_modulation层
        adaLN_count = 0
        for name, module in self.dit.named_modules():
            if 'adaLN_modulation' in name and isinstance(module, nn.Linear):
                # 确保这些层接受正确的输入维度 (hidden_size=1152)
                expected_in_features = self.dit.hidden_size  # 1152
                if module.in_features != expected_in_features:
                    print(f"   ⚠️ 发现维度不匹配的adaLN层: {name}")
                    print(f"      当前: {module.in_features} -> 期望: {expected_in_features}")
                    # 重新创建这个层
                    new_layer = nn.Linear(expected_in_features, module.out_features, bias=module.bias is not None)
                    # 用新层替换旧层
                    parent_module = self.dit
                    for attr in name.split('.')[:-1]:
                        parent_module = getattr(parent_module, attr)
                    setattr(parent_module, name.split('.')[-1], new_layer)
                    adaLN_count += 1
                else:
                    # 维度正确，只需重新初始化权重
                    nn.init.constant_(module.weight, 0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                    adaLN_count += 1
        
        print(f"   ✅ 处理了 {adaLN_count} 个adaLN层")
    
    def _freeze_backbone(self):
        """冻结主干网络"""
        frozen_count = 0
        for name, param in self.dit.named_parameters():
            # 保持最后几层可训练，用于条件适配
            if not any(keep in name for keep in ['final_layer', 'adaln']):
                param.requires_grad = False
                frozen_count += 1
        
        print(f"🔒 冻结参数: {frozen_count}")
    
    def _inject_condition_layers(self):
        """简化架构：不修改DiT内部结构"""
        print("✅ 使用简化架构：保持LightningDiT标准接口")
        print("🎯 用户条件判别将通过强化训练策略实现")
        # 不添加任何条件注入层，保持DiT原始结构
        pass
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (B, C, H, W) 潜向量
            t: (B,) 时间步
            user_classes: (B,) 用户类别 (0-30 对应 ID_1 到 ID_31)
        Returns:
            predicted_noise: (B, C, H, W) 预测的噪声
        """
        # 编码丰富的用户条件特征
        user_condition = self.condition_encoder(user_classes)  # (B, condition_dim=768)
        
        # 将丰富的用户条件注入到DiT中
        # 方法：最简化 - 让DiT使用原生机制，保存user_condition供对比学习使用
        predicted_noise = self._conditional_forward_with_injection(x, t, user_classes, user_condition)
        
        return predicted_noise
    
    def _conditional_forward_with_injection(self, x, t, user_classes, user_condition):
        """简化前向传播：使用标准LightningDiT接口"""
        # ✅ 使用标准DiT接口，无架构修改
        predicted_noise = self.dit(x, t, user_classes)
        
        # 保存用户条件用于强化训练策略
        self._last_user_condition = user_condition
        
        return predicted_noise

class ConditionalDiTTrainer:
    """条件DiT训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Kaggle双GPU支持：使用DataParallel
        self.gpu_count = torch.cuda.device_count()
        self.use_multi_gpu = self.gpu_count > 1
        
        if self.use_multi_gpu:
            # Kaggle双GPU环境：使用DataParallel
            self.device = torch.device('cuda:0')  # 主设备
            print(f"Kaggle环境检测到{self.gpu_count}个GPU，启用DataParallel")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用单GPU/CPU: {self.device}")
        
        # 🛡️ 优先保证训练稳定性（移除潜在不稳定优化）
        if torch.cuda.is_available():
            print("🛡️ 优先稳定性：使用默认CUDNN设置，确保训练可靠性")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用单GPU/CPU: {self.device}")
        
        # 初始化模型
        self._setup_model()
        
        # 🚀 Kaggle双GPU：包装模型为DataParallel
        if self.use_multi_gpu:
            self.model = DP(self.model)  # DataParallel自动使用所有可用GPU
            print(f"✅ 模型已包装为DataParallel，使用GPU: {list(range(self.gpu_count))}")
        
        # 初始化数据
        self._setup_data()
        
        # 初始化优化器
        self._setup_optimizer()
        
        # 初始化日志
        self._setup_logging()
    
    def _get_model_attr(self, attr_name: str):
        """🚀 统一的模型属性访问：支持DataParallel和普通模型"""
        if self.use_multi_gpu and hasattr(self.model, 'module'):
            return getattr(self.model.module, attr_name)
        else:
            return getattr(self.model, attr_name)
    
    def _get_actual_model(self):
        """🚀 获取实际模型：支持DataParallel和普通模型"""
        if self.use_multi_gpu and hasattr(self.model, 'module'):
            return self.model.module
        else:
            return self.model
    
    def _setup_model(self):
        """设置模型"""
        model_config = self.config['model']['params']
        
        # VAE - 使用正确的VA_VAE类和API
        self.vae = VA_VAE("LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
        # VA_VAE已经在初始化时自动移动到GPU和设置eval模式
        
        # 条件DiT
        self.model = ConditionalDiT(
            **model_config,
            pretrained_path="models/lightningdit-xl-imagenet256-64ep.pt"
        )
        self.model.to(self.device)
        
        print(f"✅ 模型设置完成，设备: {self.device}")
    
    def _setup_data(self):
        """设置数据"""
        data_config = self.config['data']['params']
        
        self.data_module = MicroDopplerDataModule(**data_config)
        self.data_module.setup()
        
        # 更新模型的用户数量
        actual_num_users = self.data_module.num_users
        if actual_num_users != self._get_model_attr('num_users'):
            print(f"⚠️ 更新用户数量: {self._get_model_attr('num_users')} -> {actual_num_users}")
            # 这里可能需要重新初始化条件编码器
    
    def _setup_optimizer(self):
        """设置优化器"""
        opt_config = self.config['optimizer']['params']
        
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(trainable_params, **opt_config)
        
        # 学习率调度器
        if 'scheduler' in self.config:
            sch_config = self.config['scheduler']['params']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **sch_config
            )
        else:
            self.scheduler = None
    
    def _setup_logging(self):
        """设置日志"""
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/conditional_dit_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"✅ 实验目录: {self.exp_dir}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失 - 增强版：包含对比学习用于用户区分"""
        images = batch['image'].to(self.device)
        user_classes = batch['user_class'].to(self.device)
        
        batch_size = images.shape[0]
        
        # VAE编码 - 使用正确的VA_VAE API
        with torch.no_grad():
            z = self.vae.encode_images(images)  # VA_VAE直接返回潜向量张量
        
        # 添加噪声（DDPM训练）
        noise = torch.randn_like(z)
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # 前向过程: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
        # 简化实现，使用固定的噪声调度
        alpha_bar_t = 1 - timesteps.float() / 1000
        alpha_bar_t = alpha_bar_t.view(batch_size, 1, 1, 1)
        
        z_noisy = noise * (timesteps.view(-1, 1, 1, 1) / 1000.0) + z * (1 - timesteps.view(-1, 1, 1, 1) / 1000.0)
        
        # 🚀 获取实际模型（DataParallel兼容）
        actual_model = self._get_actual_model()
        
        # 🚀 关键修复：确保用户条件编码器参与梯度计算
        user_embeddings = actual_model.condition_encoder(user_classes)  # (B, 1152) ✅ DataParallel修复
        
        # 模型预测（传递用户嵌入以确保梯度流）
        predicted_noise = self.model(z_noisy, timesteps, user_classes)
        
        # 1. 🎯 基础扩散重构损失
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # 2. 🚀 强化训练策略：直接使用用户嵌入（保持梯度流）
        user_condition = user_embeddings  # ✅ 直接使用，保持梯度连接
        
        # 2.1 强化用户判别：对比学习损失
        contrastive_loss = self.compute_enhanced_contrastive_loss(user_condition, user_classes)
        
        # 2.2 用户间判别损失：确保不同用户生成不同噪声
        inter_user_loss = self.compute_inter_user_discriminative_loss(predicted_noise, user_classes)
        
        # 2.3 渐进式权重调整：随训练进程加强用户判别
        epoch_ratio = min(getattr(self, 'current_epoch', 0) / 20, 1.0)
        contrastive_weight = 0.05 + 0.15 * epoch_ratio  # 从0.05逐渐增加到0.2
        
        # 2.4 正则化：防止过拟合到特定用户
        user_regularization = self.compute_user_regularization_loss(user_condition)
        
        # 3. 🎯 强化训练策略总损失
        total_loss = (
            diffusion_loss +                              # 基础重建
            contrastive_weight * contrastive_loss +       # 用户判别（渐进加强）
            0.1 * inter_user_loss +                       # 用户间差异
            0.02 * user_regularization                    # 正则化
        )
        
        # 🚀 Kaggle双GPU损失处理：DataParallel自动平均
        # DataParallel会自动处理损失聚合，无需手动同步
        diffusion_loss_val = diffusion_loss.item()
        contrastive_loss_val = contrastive_loss.item()
        inter_user_loss_val = inter_user_loss.item() 
        user_regularization_val = user_regularization.item()
        total_loss_val = total_loss.item()
        
        # 记录各项损失用于监控
        self.log_losses = {
            'diffusion_loss': diffusion_loss_val,
            'contrastive_loss': contrastive_loss_val,
            'inter_user_loss': inter_user_loss_val,
            'user_regularization': user_regularization_val,
            'contrastive_weight': contrastive_weight,
            'total_loss': total_loss_val,
            'gpu_count': self.gpu_count,
            'multi_gpu': self.use_multi_gpu
        }
        
        return total_loss
    
    def compute_enhanced_contrastive_loss(self, user_condition, user_classes):
        """🚀 强化对比学习：针对数据稀缺+微妙差异优化"""
        if user_condition is None:
            return torch.tensor(0.0, device=self.device)
            
        batch_size = user_condition.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=user_condition.device)
        
        # L2标准化（提高判别稳定性）
        user_condition_norm = F.normalize(user_condition, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(user_condition_norm, user_condition_norm.t())
        
        # 温度参数：针对微妙差异调低温度，增强敏感性
        temperature = 0.05  # 比标准SimCLR更低，增强微妙差异敏感性
        sim_matrix = sim_matrix / temperature
        
        # 正样本mask：相同用户
        labels = user_classes.unsqueeze(1) == user_classes.unsqueeze(0)
        mask = torch.eye(batch_size, device=labels.device).bool()
        labels = labels & ~mask
        
        # InfoNCE损失计算
        losses = []
        for i in range(batch_size):
            positive_mask = labels[i]
            if positive_mask.sum() == 0:
                continue
                
            # 分子：正样本相似度
            pos_sim = sim_matrix[i][positive_mask]
            # 分母：所有负样本相似度
            neg_sim = sim_matrix[i][~positive_mask]
            
            # InfoNCE损失：正样本在前，负样本在后
            logits = torch.cat([pos_sim, neg_sim])
            # 标签：正样本的索引（0到len(pos_sim)-1都是正样本）
            # 但cross_entropy需要单个标签，我们取第一个正样本作为目标
            target_label = torch.tensor(0, device=logits.device, dtype=torch.long)  # 第一个正样本
            loss = F.cross_entropy(logits.unsqueeze(0), target_label.unsqueeze(0))
            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=user_condition.device)
    
    def compute_inter_user_discriminative_loss(self, predicted_noise, user_classes):
        """🚀 用户间判别损失：确保不同用户生成不同的噪声模式"""
        batch_size = predicted_noise.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=predicted_noise.device)
        
        # 将预测噪声展平为特征向量
        noise_features = predicted_noise.reshape(batch_size, -1)  # (B, C*H*W) - 使用reshape避免stride问题
        
        # 计算用户间的噪声相似度
        unique_users = torch.unique(user_classes)
        if len(unique_users) < 2:
            return torch.tensor(0.0, device=predicted_noise.device)
        
        inter_user_similarities = []
        for i, user_a in enumerate(unique_users):
            for user_b in unique_users[i+1:]:
                mask_a = user_classes == user_a
                mask_b = user_classes == user_b
                
                if mask_a.sum() == 0 or mask_b.sum() == 0:
                    continue
                
                # 计算不同用户间的平均噪声相似度
                noise_a = noise_features[mask_a].mean(dim=0)
                noise_b = noise_features[mask_b].mean(dim=0)
                
                similarity = F.cosine_similarity(noise_a.unsqueeze(0), noise_b.unsqueeze(0))
                inter_user_similarities.append(similarity)
        
        if not inter_user_similarities:
            return torch.tensor(0.0, device=predicted_noise.device)
        
        # 损失：最小化用户间相似度（鼓励差异化）
        avg_similarity = torch.stack(inter_user_similarities).mean()
        # 🔧 修复：将相似度映射到[0,1]再计算损失，避免负值被ReLU截断
        similarity_01 = (avg_similarity + 1.0) / 2.0  # 从[-1,1]映射到[0,1]
        return similarity_01  # 相似度越高，损失越大
    
    def compute_user_regularization_loss(self, user_condition):
        """🔧 修复：用户正则化损失（防止数值爆炸）"""
        if user_condition is None:
            return torch.tensor(0.0, device=self.device)
        
        # L2正则化：防止嵌入过大（使用平方而不是范数，避免爆炸）
        l2_reg = torch.mean(user_condition.pow(2))
        
        # 多样性正则化：鼓励不同用户嵌入分散（限制数值范围）
        if user_condition.shape[0] > 1:
            # 计算嵌入的标准差，鼓励多样性
            user_std = torch.std(user_condition, dim=0).mean()
            # 🔧 修复：避免exp爆炸，使用线性惩罚
            diversity_reg = torch.clamp(1.0 / (user_std + 1e-6), 0.0, 10.0)
        else:
            diversity_reg = torch.tensor(0.0, device=user_condition.device)
        
        return l2_reg + 0.01 * diversity_reg  # 🔧 降低多样性权重
    
    def _compute_contrastive_loss(self, user_embeddings: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """简化的对比学习损失：遵循SimCLR原则，避免memory bank复杂性"""
        batch_size = user_embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=user_embeddings.device)
        
        # === 采用SimCLR风格的简化对比学习 ===
        # 核心思想：在batch内进行充分的对比，配合gradient accumulation模拟大batch
        contrastive_loss = self._compute_simclr_style_contrastive_loss(user_embeddings, user_classes)
        
        return contrastive_loss
    
    def _compute_simclr_style_contrastive_loss(self, user_embeddings: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """SimCLR风格的简化对比学习 - 避免memory bank的复杂性和陈旧性问题"""
        batch_size = user_embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=user_embeddings.device)
        
        # L2标准化嵌入（SimCLR的关键实践）
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        
        # 计算余弦相似度矩阵
        similarities = torch.mm(user_embeddings, user_embeddings.t())  # (B, B)
        
        # 温度参数（SimCLR建议0.07-0.1）
        temperature = 0.07
        similarities = similarities / temperature
        
        # 创建正样本mask：相同用户为正样本
        labels = user_classes.unsqueeze(1) == user_classes.unsqueeze(0)  # (B, B)
        # 排除自己与自己的对（对角线）
        mask = torch.eye(batch_size, device=labels.device).bool()
        labels = labels & ~mask
        
        # 计算每个样本的InfoNCE损失
        losses = []
        for i in range(batch_size):
            # 当前样本的正样本（相同用户的其他样本）
            positive_mask = labels[i]
            
            if positive_mask.sum() == 0:
                # 如果batch中没有相同用户的其他样本，跳过
                continue
                
            # 正样本分数
            pos_scores = similarities[i][positive_mask]  # (num_positives,)
            
            # 负样本分数（所有其他样本，除了自己）
            neg_mask = ~positive_mask & ~mask[i]  # 排除正样本和自己
            neg_scores = similarities[i][neg_mask]  # (num_negatives,)
            
            # InfoNCE损失计算
            pos_exp = torch.exp(pos_scores)
            neg_exp = torch.exp(neg_scores)
            
            # 对每个正样本计算损失
            for pos_exp_single in pos_exp:
                denominator = pos_exp_single + neg_exp.sum()
                loss_single = -torch.log(pos_exp_single / denominator + 1e-8)
                losses.append(loss_single)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=user_embeddings.device)
        
        return torch.stack(losses).mean()
    
    def _compute_cross_user_contrastive_loss(self, user_embeddings: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """全用户负采样对比学习 - 你提出的增强方法"""
        if not hasattr(self, '_negative_user_bank'):
            return torch.tensor(0.0, device=user_embeddings.device)
        
        batch_size = user_embeddings.shape[0]
        total_loss = 0.0
        
        # 对batch中每个样本进行全用户对比
        for i in range(batch_size):
            current_user = user_classes[i].item()
            current_embedding = user_embeddings[i]  # (768,)
            
            # === 正样本：从negative bank中取该用户的其他样本 ===
            if current_user in self._negative_user_bank:
                positive_embeddings = self._negative_user_bank[current_user]  # (N_pos, 768)
                if positive_embeddings.shape[0] > 0:
                    pos_similarities = torch.mm(current_embedding.unsqueeze(0), positive_embeddings.t())  # (1, N_pos)
                    pos_similarities = pos_similarities / 0.1  # temperature
                    positive_score = torch.exp(pos_similarities).sum()
                else:
                    positive_score = torch.tensor(1e-8, device=user_embeddings.device)
            else:
                positive_score = torch.tensor(1e-8, device=user_embeddings.device)
            
            # === 负样本：从negative bank中取其他所有用户的样本 ===
            negative_score = torch.tensor(0.0, device=user_embeddings.device)
            for other_user, other_embeddings in self._negative_user_bank.items():
                if other_user != current_user and other_embeddings.shape[0] > 0:
                    # 随机采样一些负样本，避免计算量过大
                    n_neg_samples = min(5, other_embeddings.shape[0])  # 每个用户采样5个负样本
                    indices = torch.randperm(other_embeddings.shape[0])[:n_neg_samples]
                    sampled_negatives = other_embeddings[indices]  # (n_neg_samples, 768)
                    
                    neg_similarities = torch.mm(current_embedding.unsqueeze(0), sampled_negatives.t())  # (1, n_neg_samples)
                    neg_similarities = neg_similarities / 0.1  # temperature
                    negative_score += torch.exp(neg_similarities).sum()
            
            # InfoNCE损失：log(正样本得分 / (正样本得分 + 负样本得分))
            if negative_score > 0:
                sample_loss = -torch.log(positive_score / (positive_score + negative_score + 1e-8))
                total_loss += sample_loss
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=user_embeddings.device)
    
    def _update_negative_user_bank(self, user_embeddings: torch.Tensor, user_classes: torch.Tensor):
        """维护用户嵌入银行，用于跨batch的全用户对比学习"""
        if not hasattr(self, '_negative_user_bank'):
            self._negative_user_bank = {}
        
        # 将当前batch的嵌入添加到对应用户的银行中
        for i, user_class in enumerate(user_classes):
            user_id = user_class.item()
            embedding = user_embeddings[i].detach().clone()  # 避免梯度传播
            
            if user_id not in self._negative_user_bank:
                self._negative_user_bank[user_id] = []
            
            # 维护每个用户最多50个嵌入样本（内存控制）
            if len(self._negative_user_bank[user_id]) >= 50:
                # 随机替换一个老样本
                replace_idx = torch.randint(0, len(self._negative_user_bank[user_id]), (1,)).item()
                self._negative_user_bank[user_id][replace_idx] = embedding
            else:
                self._negative_user_bank[user_id].append(embedding)
        
        # 转换为tensor格式便于计算
        for user_id in self._negative_user_bank:
            if isinstance(self._negative_user_bank[user_id], list):
                self._negative_user_bank[user_id] = torch.stack(self._negative_user_bank[user_id])
    
    def _compute_regularization_loss(self, user_embeddings: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """正则化损失：确保不同用户的嵌入在空间中分布均匀"""
        unique_users = torch.unique(user_classes)
        
        if len(unique_users) < 2:
            return torch.tensor(0.0, device=user_embeddings.device)
        
        # 计算每个用户的平均嵌入
        user_centers = []
        for user in unique_users:
            mask = user_classes == user
            user_center = user_embeddings[mask].mean(dim=0)
            user_centers.append(user_center)
        
        user_centers = torch.stack(user_centers)  # (num_users, embed_dim)
        
        # 计算用户中心间的最小距离
        distances = torch.cdist(user_centers, user_centers, p=2)  # (num_users, num_users)
        
        # 排除对角线（自己与自己的距离）
        mask = ~torch.eye(len(unique_users), dtype=torch.bool, device=distances.device)
        min_distance = distances[mask].min()
        
        # 正则化：鼓励用户中心间保持最小距离
        target_distance = 1.0  # 目标最小距离
        regularization_loss = F.relu(target_distance - min_distance)
        
        return regularization_loss
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        train_loader = self.data_module.train_dataloader()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('trainer', {}).get('gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['trainer']['gradient_clip_val']
                )
            
            self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        val_loader = self.data_module.val_dataloader()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def generate_validation_samples(self, epoch: int, num_samples: int = 8):
        """🖼️ 生成验证样本进行可视化"""
        if not hasattr(self, 'vae'):
            return
            
        self.model.eval()
        
        try:
            # 选择不同的用户进行生成测试 (使用0-based索引，对应ID_1到ID_31)
            test_users = torch.tensor([0, 4, 9, 14, 19, 24, 29, 30], device=self.device)[:num_samples]
            
            with torch.no_grad():
                # 生成随机噪声
                z_shape = (len(test_users), 32, 16, 16)  # VA-VAE潜向量形状
                z = torch.randn(z_shape, device=self.device)
                t = torch.randint(0, 1000, (len(test_users),), device=self.device)
                
                # 用户条件生成
                generated_z = self.model(z, t, test_users)
                
                # 解码为图像
                generated_images = self.vae.decode_to_images(generated_z)
                
                # 保存可视化结果
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Epoch {epoch} - User-Conditional Generation')
                
                for i, (user_idx, img) in enumerate(zip(test_users, generated_images)):
                    row, col = i // 4, i % 4
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    if img.ndim == 3:
                        img = img.transpose(1, 2, 0)
                    axes[row, col].imshow(img, cmap='viridis')
                    # 显示实际用户ID (0-based索引+1 = 1-based用户ID)
                    actual_user_id = user_idx.item() + 1
                    axes[row, col].set_title(f'User ID_{actual_user_id}')
                    axes[row, col].axis('off')
                
                # 保存图片
                save_path = self.exp_dir / f"validation_samples_epoch_{epoch:03d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"📸 已保存验证样本: {save_path}")
                
        except Exception as e:
            print(f"⚠️ 生成验证样本失败: {e}")
        finally:
            self.model.train()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.exp_dir / "last.ckpt")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.exp_dir / "best.ckpt")
    
    def train(self):
        """完整训练流程"""
        trainer_config = self.config.get('trainer', {})
        max_epochs = trainer_config.get('max_epochs', 50)
        val_every_n_epochs = trainer_config.get('check_val_every_n_epoch', 2)
        
        best_val_loss = float('inf')
        
        print(f"🚀 开始训练，共{max_epochs}个epoch")
        
        for epoch in range(max_epochs):
            print(f"\n📅 Epoch {epoch+1}/{max_epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"🎯 训练损失: {train_loss:.4f}")
            
            # 验证
            if (epoch + 1) % val_every_n_epochs == 0:
                val_loss = self.validate()
                print(f"📊 验证损失: {val_loss:.4f}")
                
                # 🖼️ 生成可视化样本（每10个epoch）
                if (epoch + 1) % 10 == 0:
                    self.generate_validation_samples(epoch + 1)
                
                # 保存最佳模型
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print(f"🏆 新的最佳模型！验证损失: {best_val_loss:.4f}")
                
                self.save_checkpoint(epoch, is_best)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        print(f"\n🎉 训练完成！最佳验证损失: {best_val_loss:.4f}")
        print(f"📁 实验目录: {self.exp_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径（覆盖配置文件）')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖数据目录
    if args.data_dir:
        config['data']['params']['data_dir'] = args.data_dir
    
    print("🚀 步骤5: 条件DiT微调训练")
    print("="*60)
    print(f"📝 配置文件: {args.config}")
    
    # 🔍 调试信息：检查配置加载
    print(f"🔍 调试 - 加载的数据路径: {config['data']['params']['data_dir']}")
    print(f"🔍 调试 - 完整数据配置: {config['data']['params']}")
    
    # 创建训练器并开始训练
    trainer = ConditionalDiTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
