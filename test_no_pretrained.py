#!/usr/bin/env python3
"""
🧪 最小化测试：完全不加载预训练权重
用于验证768维问题是否来自预训练权重
"""
import os
import sys
import torch
import torch._dynamo

# 清除dynamo缓存
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_no_pretrained():
    """测试不加载预训练权重的情况"""
    print("🧪 开始测试：完全不加载预训练权重")
    
    try:
        from step5_conditional_dit_training import ConditionalDiT
        
        # 1. 创建模型（不加载预训练权重）
        print("📦 创建ConditionalDiT模型（无预训练权重）...")
        model = ConditionalDiT(
            model_name="LightningDiT-XL/1",
            num_users=31,
            pretrained_path=None,  # 🔧 关键：不加载预训练权重
            frozen_backbone=False,
            condition_dim=1152
        )
        
        # 2. 检查所有条件层维度
        print("🔍 检查条件层维度...")
        print(f"UserConditionEncoder输出维度: {model.condition_encoder.embed_dim}")
        print(f"DiT hidden_size: {model.dit.hidden_size}")
        print(f"DiT y_embedder权重: {model.dit.y_embedder.weight.shape}")
        
        # 3. 检查所有adaLN层
        print("🔍 检查adaLN层...")
        for name, module in model.dit.named_modules():
            if hasattr(module, 'adaLN_modulation') and hasattr(module.adaLN_modulation, 'weight'):
                weight_shape = module.adaLN_modulation.weight.shape
                print(f"  {name}.adaLN_modulation: {weight_shape}")
                if weight_shape[0] != model.dit.hidden_size * 6:  # adaLN应该是hidden_size*6
                    print(f"    ⚠️  警告：维度不匹配！期望 {model.dit.hidden_size * 6}")
        
        # 4. 测试前向传播
        print("🚀 测试前向传播...")
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 创建测试数据
        x = torch.randn(batch_size, 32, 16, 16, device=device)  # [B, C, H, W]
        t = torch.randint(0, 1000, (batch_size,), device=device)
        user_classes = torch.randint(0, 31, (batch_size,), device=device)
        
        with torch.no_grad():
            output = model(x, t, user_classes)
            print(f"✅ 前向传播成功！输出形状: {output.shape}")
            
        print("🎉 测试完成：无预训练权重版本工作正常！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 768维问题根因分析：无预训练权重测试")
    print("=" * 60)
    
    success = test_no_pretrained()
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 结论：问题确实来自预训练权重加载！")
        print("💡 解决方案：需要完善权重加载和层重新初始化逻辑")
    else:
        print("🤔 结论：问题不在预训练权重，需要进一步调试")
    print("=" * 60)

if __name__ == "__main__":
    main()
