#!/usr/bin/env python3
"""
LightningDiT维度诊断测试文件
用于empirically验证官方LightningDiT的输入/输出维度要求
"""

import torch
import torch.nn as nn
import sys
import os

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE

def test_official_dit_dimensions():
    """测试官方LightningDiT的维度要求"""
    print("🔍 LightningDiT官方维度诊断测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # 1. 测试官方DiT配置
    print("\n📊 1. 官方DiT模型配置:")
    model_name = "LightningDiT-XL/1"
    
    # 创建官方配置的DiT（ImageNet 1000类）
    print(f"   创建 {model_name} (官方ImageNet配置)")
    dit_official = LightningDiT_models[model_name](
        input_size=16,           # 16x16 patches
        in_channels=32,          # VA-VAE通道数
        # num_classes=1000,      # ImageNet默认（不指定让它使用默认值）
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False
    ).to(device)
    
    # 打印官方模型的关键维度
    print(f"   hidden_size: {dit_official.hidden_size}")
    print(f"   y_embedder.num_classes: {dit_official.y_embedder.num_classes}")
    print(f"   y_embedder.embedding_table.weight.shape: {dit_official.y_embedder.embedding_table.weight.shape}")
    
    # 检查adaLN_modulation的维度
    for name, module in dit_official.named_modules():
        if 'adaLN_modulation' in name and isinstance(module, nn.Linear):
            print(f"   {name}: {module.in_features} -> {module.out_features}")
            break
    
    # 2. 测试用户条件DiT配置  
    print(f"\n📊 2. 用户条件DiT模型配置:")
    print(f"   创建 {model_name} (31用户配置)")
    dit_user = LightningDiT_models[model_name](
        input_size=16,
        in_channels=32,
        num_classes=31,          # 31个用户
        use_qknorm=False,
        use_swiglu=True, 
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False
    ).to(device)
    
    print(f"   hidden_size: {dit_user.hidden_size}")
    print(f"   y_embedder.num_classes: {dit_user.y_embedder.num_classes}")
    print(f"   y_embedder.embedding_table.weight.shape: {dit_user.y_embedder.embedding_table.weight.shape}")
    
    # 检查adaLN_modulation的维度
    for name, module in dit_user.named_modules():
        if 'adaLN_modulation' in name and isinstance(module, nn.Linear):
            print(f"   {name}: {module.in_features} -> {module.out_features}")
            break
    
    # 3. 测试前向传播
    print(f"\n📊 3. 前向传播维度测试:")
    batch_size = 2
    
    # 创建测试输入
    x = torch.randn(batch_size, 32, 16, 16).to(device)  # [B, C, H, W]
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # [B]
    
    print(f"   输入 x: {x.shape}")
    print(f"   时间 t: {t.shape}")
    
    # 测试官方DiT
    print(f"\n   测试官方DiT (ImageNet类别):")
    y_imagenet = torch.randint(0, 1000, (batch_size,)).to(device)  # ImageNet类别
    print(f"   类别 y: {y_imagenet.shape} (值: {y_imagenet})")
    
    try:
        with torch.no_grad():
            # 检查中间维度
            x_embed = dit_official.x_embedder(x) + dit_official.pos_embed
            t_embed = dit_official.t_embedder(t)
            y_embed = dit_official.y_embedder(y_imagenet, False)
            c = t_embed + y_embed
            
            print(f"   x_embed: {x_embed.shape}")
            print(f"   t_embed: {t_embed.shape}")
            print(f"   y_embed: {y_embed.shape}")
            print(f"   c (condition): {c.shape}")
            
            # 测试完整前向传播
            output = dit_official(x, t, y_imagenet)
            print(f"   ✅ 官方DiT输出: {output.shape}")
    except Exception as e:
        print(f"   ❌ 官方DiT错误: {e}")
    
    # 测试用户条件DiT
    print(f"\n   测试用户条件DiT (31用户):")
    y_users = torch.randint(0, 31, (batch_size,)).to(device)  # 用户类别
    print(f"   类别 y: {y_users.shape} (值: {y_users})")
    
    try:
        with torch.no_grad():
            # 检查中间维度
            x_embed = dit_user.x_embedder(x) + dit_user.pos_embed
            t_embed = dit_user.t_embedder(t)
            y_embed = dit_user.y_embedder(y_users, False)
            c = t_embed + y_embed
            
            print(f"   x_embed: {x_embed.shape}")
            print(f"   t_embed: {t_embed.shape}")  
            print(f"   y_embed: {y_embed.shape}")
            print(f"   c (condition): {c.shape}")
            
            # 测试完整前向传播
            output = dit_user(x, t, y_users)
            print(f"   ✅ 用户DiT输出: {output.shape}")
    except Exception as e:
        print(f"   ❌ 用户DiT错误: {e}")
    
    # 4. 测试VA-VAE维度
    print(f"\n📊 4. VA-VAE维度测试:")
    try:
        vae = VA_VAE("configs/vavae.yaml").to(device)
        
        # 测试图像编码
        test_image = torch.randn(batch_size, 3, 256, 256).to(device)
        print(f"   输入图像: {test_image.shape}")
        
        with torch.no_grad():
            z = vae.encode_to_latent(test_image)
            print(f"   ✅ VA-VAE潜向量: {z.shape}")
            
            # 检查是否匹配DiT期望的输入
            if z.shape[1:] == (32, 16, 16):
                print(f"   ✅ VA-VAE输出与DiT输入匹配!")
            else:
                print(f"   ❌ VA-VAE输出与DiT输入不匹配! 期望: (*, 32, 16, 16)")
                
    except Exception as e:
        print(f"   ❌ VA-VAE错误: {e}")
    
    print(f"\n🎯 诊断完成!")

def test_user_condition_encoder():
    """测试用户条件编码器"""
    print(f"\n📊 5. 用户条件编码器测试:")
    
    # 这是我们在训练代码中使用的用户条件编码器
    sys.path.append('.')
    from step4_microdoppler_adapter import UserConditionEncoder
    
    condition_encoder = UserConditionEncoder(
        num_users=31,
        condition_dim=768  # 这个维度！！！
    )
    
    user_ids = torch.tensor([0, 1, 2, 5])
    user_condition = condition_encoder(user_ids)
    print(f"   用户条件编码器输出: {user_condition.shape}")
    print(f"   输出维度: {user_condition.shape[-1]} (这是768！)")
    
    print(f"\n💡 关键发现:")
    print(f"   - DiT期望: hidden_size = 1152")
    print(f"   - 我们的用户条件: condition_dim = 768") 
    print(f"   - 这可能就是维度不匹配的根源!")

if __name__ == "__main__":
    print("🚀 开始LightningDiT维度诊断")
    test_official_dit_dimensions()
    test_user_condition_encoder()
