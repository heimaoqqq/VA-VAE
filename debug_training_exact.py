#!/usr/bin/env python3
"""
精确复制训练环境的调试测试
用于查找768维问题的确切来源
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
from typing import Optional

# 清除torch._dynamo缓存，解决768维问题
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
print("🧹 已清除torch._dynamo缓存，禁用动态编译")

# 添加LightningDiT到路径
sys.path.append(str(Path("LightningDiT").absolute()))

from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE
from step4_microdoppler_adapter import MicroDopplerDataModule, UserConditionEncoder


class ConditionalDiT(nn.Module):
    """条件DiT模型 - 精确复制training版本"""
    
    def __init__(
        self,
        model_name: str = "LightningDiT-XL/1",
        num_users: int = 31,
        condition_dim: int = 1152,
        frozen_backbone: bool = True,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_users = num_users
        self.condition_dim = condition_dim
        
        print(f"🔧 创建ConditionalDiT: model={model_name}, num_users={num_users}, condition_dim={condition_dim}")
        
        # 创建DiT模型
        self.dit = LightningDiT_models[model_name](
            input_size=16,
            patch_size=1,
            in_channels=32,
            num_classes=num_users,  # ✅ 设置为用户数量
            # 其他LightningDiT-XL参数
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=False,
            use_rmsnorm=False,
            wo_shift=False
        )
        
        print(f"   DiT hidden_size: {self.dit.hidden_size}")
        print(f"   DiT num_classes: {self.dit.num_classes}")
        
        # 创建用户条件编码器
        self.condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=condition_dim,  # ✅ 使用1152
            dropout=dropout
        )
        
        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 冻结主干参数
        if frozen_backbone:
            self._freeze_backbone()
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        print(f"📥 加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 过滤兼容的权重
        model_state_dict = self.dit.state_dict()
        compatible_weights = {}
        incompatible_weights = []
        
        for name, param in state_dict.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    compatible_weights[name] = param
                else:
                    incompatible_weights.append(f"   {name}: {param.shape} -> {model_state_dict[name].shape}")
        
        print(f"✅ 成功加载 {len(compatible_weights)} 个兼容权重")
        if incompatible_weights:
            print(f"⚠️ 跳过 {len(incompatible_weights)} 个不兼容权重:")
            for w in incompatible_weights[:3]:  # 只显示前3个
                print(w)
        
        # 加载兼容权重
        self.dit.load_state_dict(compatible_weights, strict=False)
        
        # 🔧 重新初始化条件相关层
        self._reinitialize_conditional_layers()
    
    def _reinitialize_conditional_layers(self):
        """重新初始化条件相关层"""
        print(f"🔧 重新初始化条件相关层...")
        
        # 重新初始化y_embedder
        if hasattr(self.dit, 'y_embedder'):
            nn.init.normal_(self.dit.y_embedder.embedding_table.weight, std=0.02)
            print(f"   重新初始化 y_embedder: {self.dit.y_embedder.embedding_table.weight.shape}")
        
        # 重新初始化所有adaLN_modulation层
        adaLN_count = 0
        for name, module in self.dit.named_modules():
            if 'adaLN_modulation' in name and isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        # ❌ 这里可能是问题！让我们检查这个层的期望输入维度
                        expected_input = submodule.weight.shape[1]
                        actual_hidden = self.dit.hidden_size
                        
                        print(f"   🔍 {name}: weight={submodule.weight.shape}, 期望输入={expected_input}, 实际hidden={actual_hidden}")
                        
                        if expected_input != actual_hidden:
                            print(f"   ❌ 维度不匹配！重新创建层...")
                            # 重新创建层
                            new_layer = nn.Linear(actual_hidden, submodule.weight.shape[0])
                            nn.init.normal_(new_layer.weight, std=0.02)
                            if submodule.bias is not None:
                                nn.init.zeros_(new_layer.bias)
                            
                            # 替换层
                            parent = module
                            for i, sub in enumerate(parent):
                                if sub is submodule:
                                    parent[i] = new_layer
                                    break
                            
                            print(f"   ✅ 已重新创建层: {new_layer.weight.shape}")
                        else:
                            nn.init.normal_(submodule.weight, std=0.02)
                            if submodule.bias is not None:
                                nn.init.zeros_(submodule.bias)
                            print(f"   ✅ 维度匹配，重新初始化权重")
                        
                        adaLN_count += 1
        
        print(f"   ✅ 处理了 {adaLN_count} 个adaLN层")
    
    def _freeze_backbone(self):
        """冻结主干参数"""
        frozen_count = 0
        for name, param in self.dit.named_parameters():
            if not any(x in name for x in ['y_embedder', 'adaLN_modulation']):
                param.requires_grad = False
                frozen_count += 1
        
        print(f"🔒 冻结参数: {frozen_count}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ 条件DiT初始化完成")
        print(f"   - 主干冻结: True")
        print(f"   - 总参数: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, user_classes: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 生成用户条件嵌入
        user_condition = self.condition_encoder(user_classes)  # (B, 1152)
        
        print(f"🔍 Forward调试:")
        print(f"   x: {x.shape}")
        print(f"   t: {t.shape}")
        print(f"   user_classes: {user_classes.shape}")
        print(f"   user_condition: {user_condition.shape}")
        
        # 调用DiT
        predicted_noise = self.dit(x, t, user_classes)
        
        # 保存用户条件用于对比损失
        self._last_user_condition = user_condition
        
        return predicted_noise


def test_exact_training_reproduction():
    """精确复制训练环境进行测试"""
    print("🚀 开始精确训练复制测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 设备: {device}")
    
    # 1. 创建ConditionalDiT（与训练完全一样）
    print(f"\n📊 1. 创建ConditionalDiT模型:")
    model = ConditionalDiT(
        model_name="LightningDiT-XL/1",
        num_users=31,
        condition_dim=1152,
        frozen_backbone=True,
        pretrained_path="models/lightningdit-xl-imagenet256-64ep.pt",
        dropout=0.1
    ).to(device)
    
    # 2. 创建测试数据（与训练完全一样）
    print(f"\n📊 2. 创建测试数据:")
    batch_size = 2
    x = torch.randn(batch_size, 32, 16, 16).to(device)  # 潜向量
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # 时间步
    user_classes = torch.randint(0, 31, (batch_size,)).to(device)  # 用户类别
    
    print(f"   x: {x.shape}")
    print(f"   t: {t.shape}")
    print(f"   user_classes: {user_classes.shape}")
    
    # 3. 前向传播测试
    print(f"\n📊 3. 前向传播测试:")
    try:
        model.eval()
        with torch.no_grad():
            output = model(x, t, user_classes)
            print(f"   ✅ 前向传播成功！输出: {output.shape}")
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        print(f"   🔍 错误类型: {type(e).__name__}")
        
        # 详细调试
        print(f"\n🔍 详细调试模型状态:")
        
        # 检查condition_encoder输出
        user_condition = model.condition_encoder(user_classes)
        print(f"   condition_encoder输出: {user_condition.shape}")
        
        # 检查DiT的关键层维度
        print(f"   DiT.hidden_size: {model.dit.hidden_size}")
        print(f"   DiT.y_embedder weight: {model.dit.y_embedder.embedding_table.weight.shape}")
        
        # 检查第一个adaLN层
        first_block = model.dit.blocks[0]
        adaLN_layer = first_block.adaLN_modulation[1]  # Linear层
        print(f"   第一个adaLN层weight: {adaLN_layer.weight.shape}")
        print(f"   期望输入维度: {adaLN_layer.weight.shape[1]}")
        
        # 手动测试y_embedder
        print(f"\n🔍 手动测试y_embedder:")
        y_embed = model.dit.y_embedder(user_classes)
        print(f"   y_embed: {y_embed.shape}")
        
        raise e


if __name__ == "__main__":
    test_exact_training_reproduction()
