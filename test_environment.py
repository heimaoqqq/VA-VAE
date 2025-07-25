#!/usr/bin/env python3
"""
环境测试脚本
快速检查所有依赖和导入是否正常
"""

import sys
import os

def test_basic_imports():
    """测试基础库导入"""
    print("🔍 测试基础库...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA不可用")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from PIL import Image
        print("✅ PIL/Pillow")
        
        from accelerate import Accelerator
        print("✅ Accelerate")
        
        from safetensors.torch import load_file, save_file
        print("✅ Safetensors")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

def test_rmsnorm():
    """测试RMSNorm"""
    print("\n🔍 测试RMSNorm...")
    
    try:
        # 测试简化版本
        from simple_rmsnorm import RMSNorm
        print("✅ simple_rmsnorm导入成功")
        
        # 测试创建和使用
        import torch
        rmsnorm = RMSNorm(768)
        x = torch.randn(2, 10, 768)
        output = rmsnorm(x)
        print(f"✅ RMSNorm工作正常: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RMSNorm测试失败: {e}")
        return False

def test_lightningdit():
    """测试LightningDiT导入"""
    print("\n🔍 测试LightningDiT...")
    
    try:
        # 添加路径
        sys.path.insert(0, 'LightningDiT')
        
        # 测试导入
        from models.lightningdit import LightningDiT_models
        print("✅ LightningDiT_models导入成功")
        
        # 测试模型列表
        available_models = list(LightningDiT_models.keys())
        print(f"✅ 可用模型: {available_models}")
        
        # 测试创建小模型
        import torch
        model = LightningDiT_models['LightningDiT-B/1'](
            input_size=16,
            num_classes=5,
            in_channels=32,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        print(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters()):,} 参数")
        
        return True
        
    except Exception as e:
        print(f"❌ LightningDiT测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transport():
    """测试Transport"""
    print("\n🔍 测试Transport...")
    
    try:
        sys.path.insert(0, 'LightningDiT')
        from transport import create_transport
        print("✅ Transport导入成功")
        
        # 测试创建transport
        transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=1e-5,
            sample_eps=1e-3,
        )
        print("✅ Transport创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Transport测试失败: {e}")
        return False

def test_vavae():
    """测试VA-VAE"""
    print("\n🔍 测试VA-VAE...")
    
    try:
        sys.path.insert(0, 'LightningDiT')
        from tokenizer.vavae import VA_VAE
        print("✅ VA_VAE导入成功")
        
        # 检查配置文件
        if os.path.exists('vavae_config.yaml'):
            print("✅ vavae_config.yaml存在")
        else:
            print("⚠️  vavae_config.yaml不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ VA-VAE测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 环境测试")
    print("=" * 50)
    
    tests = [
        ("基础库", test_basic_imports),
        ("RMSNorm", test_rmsnorm),
        ("LightningDiT", test_lightningdit),
        ("Transport", test_transport),
        ("VA-VAE", test_vavae),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    
    all_passed = True
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过! 环境配置正确")
        print("\n可以运行:")
        print("python stage3_inference.py --test_imports")
        print("python stage3_inference.py --vavae_config vavae_config.yaml --output_dir output")
    else:
        print("\n⚠️  部分测试失败，请检查环境配置")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
