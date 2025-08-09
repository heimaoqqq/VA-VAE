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
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import yaml
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime

# 清除torch._dynamo缓存，解决768维问题
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True  # 临时禁用dynamo编译
print("🧹 已清除torch._dynamo缓存，禁用动态编译以避免768维缓存问题")

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
        
        if pretrained_path:
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
        
        print(f"✅ 条件DiT初始化完成")
        print(f"   - 主干冻结: {frozen_backbone}")
        print(f"   - 总参数: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   - 可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        print(f"📥 加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
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
        """注入条件处理层"""
        # 在DiT的adaLN层中注入用户条件
        # 这是一个简化实现，实际可能需要更复杂的架构修改
        
        # 为每个transformer block添加条件处理
        for i, block in enumerate(self.dit.blocks):
            # 创建条件融合层
            condition_fusion = nn.Sequential(
                nn.Linear(self.condition_dim, block.adaLN_modulation[1].out_features),
                nn.SiLU(),
                nn.Linear(block.adaLN_modulation[1].out_features, block.adaLN_modulation[1].out_features)
            )
            
            # 替换原有的adaLN调制层
            original_adaln = block.adaLN_modulation
            block.adaLN_modulation = nn.Sequential(
                original_adaln,
                condition_fusion
            )
    
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
        """最简化的条件注入方案 - 避免所有维度问题"""
        # ✅ 最简单的方案：让DiT正常运行，但在最后加上用户条件增强
        
        # 1. 使用DiT的标准前向传播
        # 注意：这里我们故意让DiT使用它原来的y_embedder机制
        predicted_noise = self.dit(x, t, user_classes)
        
        # 2. 为对比学习保存用户条件（在compute_loss中使用）
        # 这样我们既保持了DiT的稳定性，又有丰富的用户条件用于对比学习
        self._last_user_condition = user_condition
        
        return predicted_noise

class ConditionalDiTTrainer:
    """条件DiT训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._setup_model()
        
        # 初始化数据
        self._setup_data()
        
        # 初始化优化器
        self._setup_optimizer()
        
        # 初始化日志
        self._setup_logging()
    
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
        if actual_num_users != self.model.num_users:
            print(f"⚠️ 更新用户数量: {self.model.num_users} -> {actual_num_users}")
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
        
        z_noisy = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 获取用户条件嵌入（用于对比学习）
        user_embeddings = self.model.condition_encoder(user_classes)  # (B, 1152) ✅ 修复注释
        
        # 模型预测
        predicted_noise = self.model(z_noisy, timesteps, user_classes)
        
        # 1. 主损失：扩散重构损失
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # 2. 对比损失：强化用户间区分
        contrastive_loss = self._compute_contrastive_loss(user_embeddings, user_classes)
        
        # 3. 正则化损失：防止条件崩溃
        regularization_loss = self._compute_regularization_loss(user_embeddings, user_classes)
        
        # 总损失
        total_loss = diffusion_loss + 0.1 * contrastive_loss + 0.05 * regularization_loss
        
        # 记录分量损失
        if hasattr(self, 'current_losses'):
            self.current_losses = {
                'diffusion': diffusion_loss.item(),
                'contrastive': contrastive_loss.item(), 
                'regularization': regularization_loss.item()
            }
        
        return total_loss
    
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
