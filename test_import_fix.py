#!/usr/bin/env python3
"""
测试导入修复
"""

import sys
import os

def test_import():
    """测试导入"""
    print("🧪 测试导入修复...")
    
    try:
        # 添加路径
        sys.path.insert(0, 'LightningDiT')
        print(f"✅ 添加路径: LightningDiT")
        
        # 测试导入
        from models import LightningDiT_models
        print(f"✅ 成功导入 LightningDiT_models")
        
        # 检查可用模型
        available_models = list(LightningDiT_models.keys())
        print(f"✅ 可用模型: {available_models}")
        
        # 测试创建B模型
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
        
        print(f"✅ 成功创建 LightningDiT-B/1 模型")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safetensors():
    """测试safetensors"""
    print("\n🧪 测试safetensors...")
    
    try:
        from safetensors.torch import load_file
        print("✅ safetensors导入成功")
        return True
    except ImportError as e:
        print(f"❌ safetensors导入失败: {e}")
        print("💡 请安装: pip install safetensors")
        return False

def main():
    """主函数"""
    print("🔧 导入修复测试")
    print("=" * 30)
    
    success = True
    
    # 测试1: 模型导入
    if not test_import():
        success = False
    
    # 测试2: safetensors
    if not test_safetensors():
        success = False
    
    if success:
        print("\n✅ 所有导入测试通过!")
        print("现在可以运行:")
        print("python quick_test_fix.py")
    else:
        print("\n❌ 导入测试失败")
        print("请检查环境配置")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
