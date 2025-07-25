#!/usr/bin/env python3
"""
测试RMSNorm修复
"""

import sys
import os

def test_rmsnorm_import():
    """测试RMSNorm导入"""
    print("🧪 测试RMSNorm导入修复...")
    
    try:
        # 测试简化版本
        from simple_rmsnorm import RMSNorm
        print("✅ simple_rmsnorm导入成功")
        
        # 测试创建RMSNorm实例
        import torch
        rmsnorm = RMSNorm(768)
        print(f"✅ RMSNorm实例创建成功: {rmsnorm}")
        
        # 测试前向传播
        x = torch.randn(2, 10, 768)
        output = rmsnorm(x)
        print(f"✅ RMSNorm前向传播成功: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ simple_rmsnorm测试失败: {e}")
        return False

def test_lightningdit_import():
    """测试LightningDiT导入"""
    print("\n🧪 测试LightningDiT导入...")
    
    try:
        # 添加路径
        sys.path.insert(0, 'LightningDiT')
        
        # 测试导入
        from models.lightningdit import LightningDiT_models
        print("✅ LightningDiT_models导入成功")
        
        # 测试模型创建
        import torch
        model = LightningDiT_models['LightningDiT-B/1'](
            input_size=16,
            num_classes=31,
            in_channels=32,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        print("✅ LightningDiT模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ LightningDiT导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 RMSNorm修复测试")
    print("=" * 30)
    
    success = True
    
    # 测试1: simple_rmsnorm
    if not test_rmsnorm_import():
        success = False
    
    # 测试2: LightningDiT导入
    if not test_lightningdit_import():
        success = False
    
    if success:
        print("\n✅ 所有RMSNorm测试通过!")
        print("现在可以运行:")
        print("python stage3_inference.py --test_imports")
    else:
        print("\n❌ RMSNorm测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
