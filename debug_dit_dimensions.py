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
    
    # 先查看UserConditionEncoder的实际参数
    sys.path.append('.')
    from step4_microdoppler_adapter import UserConditionEncoder
    
    # 检查UserConditionEncoder的__init__签名
    import inspect
    sig = inspect.signature(UserConditionEncoder.__init__)
    print(f"   UserConditionEncoder.__init__参数: {list(sig.parameters.keys())}")
    
    # 用正确的参数创建
    condition_encoder = UserConditionEncoder(num_users=31)
    
    user_ids = torch.tensor([0, 1, 2, 5])
    user_condition = condition_encoder(user_ids)
    print(f"   用户条件编码器输出: {user_condition.shape}")
    print(f"   输出维度: {user_condition.shape[-1]}")
    
    return user_condition

def test_training_simulation():
    """模拟训练代码的DiT调用过程"""
    print(f"\n📊 6. 训练代码模拟测试:")
    print("   模拟step5_conditional_dit_training.py中的实际调用...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 创建和训练代码中完全相同的DiT
    print("   创建ConditionalDiT...")
    from step5_conditional_dit_training import ConditionalDiT
    
    # 不加载预训练权重的版本
    model = ConditionalDiT(
        model="LightningDiT-XL/1",
        num_users=31,
        condition_dim=1152,  # ✅ 使用修复后的参数
        frozen_backbone=True,
        pretrained_path=None  # 不加载权重
    ).to(device)
    
    print(f"   ConditionalDiT创建成功")
    print(f"   DiT hidden_size: {model.dit.hidden_size}")
    
    # 2. 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 32, 16, 16).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    user_classes = torch.randint(0, 31, (batch_size,)).to(device)
    
    # 创建用户条件（这里可能是问题！）
    user_condition = test_user_condition_encoder()[:batch_size].to(device)
    
    print(f"   输入维度:")
    print(f"     x: {x.shape}")
    print(f"     t: {t.shape}")
    print(f"     user_classes: {user_classes.shape}")
    print(f"     user_condition: {user_condition.shape}")
    
    try:
        with torch.no_grad():
            output = model(x, t, user_classes, user_condition)
            print(f"   ✅ ConditionalDiT输出: {output.shape}")
    except Exception as e:
        print(f"   ❌ ConditionalDiT错误: {e}")
        print(f"   这可能就是768维问题的来源！")
    
    # 3. 测试权重加载后的情况
    print(f"\n   测试加载预训练权重的版本...")
    try:
        model_with_weights = ConditionalDiT(
            model="LightningDiT-XL/1", 
            num_users=31,
            condition_dim=1152,  # ✅ 使用修复后的参数
            frozen_backbone=True,
            pretrained_path="models/lightningdit-xl-imagenet256-64ep.pt"
        ).to(device)
        
        with torch.no_grad():
            output = model_with_weights(x, t, user_classes, user_condition)
            print(f"   ✅ 权重加载版本输出: {output.shape}")
            
    except Exception as e:
        print(f"   ❌ 权重加载版本错误: {e}")
        print(f"   🎯 这很可能就是我们在训练中遇到的实际错误!")

def test_deep_768_trace():
    """深度追踪768维的来源"""
    print(f"\n📊 7. 深度追踪768维来源:")
    print("   逐步检查每个可能的768维来源...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 检查UserConditionEncoder的实际输出维度
    print(f"\n   🔍 检查UserConditionEncoder:")
    from step4_microdoppler_adapter import UserConditionEncoder
    
    condition_encoder = UserConditionEncoder(num_users=31)
    user_ids = torch.tensor([0, 1])
    user_condition = condition_encoder(user_ids)
    print(f"   UserConditionEncoder输出: {user_condition.shape}")
    print(f"   这是768维吗？ {'✅ 是' if user_condition.shape[-1] == 768 else '❌ 不是'}")
    
    # 2. 检查预训练权重中是否有768维的残留
    print(f"\n   🔍 检查预训练权重文件:")
    try:
        checkpoint = torch.load("models/lightningdit-xl-imagenet256-64ep.pt", map_location='cpu')
        
        # 检查checkpoint的结构
        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"   预训练权重包含 {len(state_dict)} 个参数")
        
        # 查找所有包含768维度的权重
        weights_with_768 = []
        for key, value in state_dict.items():
            if hasattr(value, 'shape'):
                if 768 in value.shape:
                    weights_with_768.append(f"{key}: {value.shape}")
        
        if weights_with_768:
            print(f"   ⚠️ 发现包含768维的权重:")
            for w in weights_with_768[:5]:  # 只显示前5个
                print(f"     {w}")
            if len(weights_with_768) > 5:
                print(f"     ... 还有 {len(weights_with_768)-5} 个")
        else:
            print(f"   ✅ 预训练权重中没有768维")
            
    except Exception as e:
        print(f"   ❌ 无法加载预训练权重: {e}")
    
    # 3. 测试step-by-step的权重加载过程
    print(f"\n   🔍 逐步测试权重加载过程:")
    
    from step5_conditional_dit_training import ConditionalDiT
    
    # 创建模型（不加载权重）
    model = ConditionalDiT(
        model="LightningDiT-XL/1",
        num_users=31, 
        condition_dim=1152,
        frozen_backbone=True,
        pretrained_path=None
    ).to(device)
    
    print(f"   空模型创建成功，condition_encoder维度:")
    test_input = torch.tensor([0, 1]).to(device)
    with torch.no_grad():
        test_output = model.condition_encoder(test_input)
        print(f"   condition_encoder输出: {test_output.shape}")
        print(f"   这个维度对吗？ {'✅ 对' if test_output.shape[-1] == 1152 else '❌ 错'}")
    
    # 手动加载权重，看看会发生什么
    print(f"\n   🔍 手动加载权重测试:")
    try:
        checkpoint = torch.load("models/lightningdit-xl-imagenet256-64ep.pt", map_location='cpu')
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('ema', checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint
            
        # 只加载DiT的权重，不加载条件编码器
        dit_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('dit.') or not key.startswith('condition_encoder'):
                dit_state_dict[key] = value
        
        print(f"   准备加载 {len(dit_state_dict)} 个DiT权重")
        
        # 检查权重兼容性
        model_state = model.state_dict()
        compatible_count = 0
        incompatible_768 = []
        
        for key, value in dit_state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible_count += 1
            elif key in model_state:
                if 768 in value.shape or 768 in model_state[key].shape:
                    incompatible_768.append(f"{key}: {value.shape} vs {model_state[key].shape}")
        
        print(f"   兼容权重: {compatible_count}")
        if incompatible_768:
            print(f"   ⚠️ 涉及768维的不兼容权重:")
            for w in incompatible_768:
                print(f"     {w}")
                
    except Exception as e:
        print(f"   ❌ 手动权重加载测试失败: {e}")

    print(f"\n💡 深度分析结论:")
    print(f"   - UserConditionEncoder输出: {user_condition.shape[-1]}维")
    print(f"   - 预训练权重检查: {'包含768维残留' if 'weights_with_768' in locals() and weights_with_768 else '无768维残留'}")
    print(f"   - 需要进一步调查权重加载和条件编码器的交互")

if __name__ == "__main__":
    print("🚀 开始LightningDiT维度诊断")
    test_official_dit_dimensions()
    test_user_condition_encoder()
    test_training_simulation()
    test_deep_768_trace()
