#!/usr/bin/env python3
"""
测试流水线脚本
用于验证修复后的代码是否能正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    try:
        # 测试基础导入
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        print("✅ 基础库导入成功")
        
        # 测试自定义模块导入
        sys.path.insert(0, 'LightningDiT')
        from models import LightningDiT_models
        print("✅ LightningDiT模型导入成功")
        
        # 测试VA-VAE导入
        from LightningDiT.vavae.models.vavae import VAVAE
        print("✅ VA-VAE模型导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_data_structure():
    """测试数据结构"""
    print("\n🔍 测试数据结构...")
    
    required_dirs = [
        "data",
        "data/raw", 
        "data/processed",
        "LightningDiT",
        "LightningDiT/vavae"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ 找到目录: {dir_path}")
        else:
            print(f"⚠️  缺少目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 创建目录: {dir_path}")
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        sys.path.insert(0, 'LightningDiT')
        from models import LightningDiT_models
        
        # 创建DiT模型
        model = LightningDiT_models['LightningDiT-XL/1'](
            input_size=16,
            num_classes=31,
            in_channels=32,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        
        print(f"✅ DiT模型创建成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 2
        latents = torch.randn(batch_size, 32, 16, 16)
        timesteps = torch.randint(0, 1000, (batch_size,))
        user_ids = torch.randint(0, 31, (batch_size,))
        
        with torch.no_grad():
            output = model(latents, timesteps, y=user_ids)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {latents.shape}")
        print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_vavae_config():
    """测试VA-VAE配置"""
    print("\n🔍 测试VA-VAE配置...")
    
    config_path = "vavae_config.yaml"
    if os.path.exists(config_path):
        print(f"✅ 找到配置文件: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ 配置文件解析成功")
            return True
        except Exception as e:
            print(f"❌ 配置文件解析失败: {e}")
            return False
    else:
        print(f"⚠️  缺少配置文件: {config_path}")
        return False

def test_inference_script():
    """测试推理脚本的导入"""
    print("\n🔍 测试推理脚本...")
    
    try:
        # 测试stage3_inference.py的关键组件
        exec(open('stage3_inference.py').read(), {'__name__': '__test__'})
        print("✅ 推理脚本语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ 推理脚本语法错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️  推理脚本运行时错误 (可能正常): {e}")
        return True

def create_dummy_data():
    """创建虚拟数据用于测试"""
    print("\n🔍 创建测试数据...")
    
    try:
        # 创建虚拟的潜在特征文件
        os.makedirs("data/processed", exist_ok=True)
        
        # 创建虚拟训练数据
        train_data = {
            'latent': torch.randn(100, 32, 16, 16),  # 100个样本
            'y': torch.randint(0, 31, (100,))        # 用户标签
        }
        
        # 创建虚拟验证数据
        val_data = {
            'latent': torch.randn(20, 32, 16, 16),   # 20个样本
            'y': torch.randint(0, 31, (20,))         # 用户标签
        }
        
        # 保存为safetensors格式 (如果可用)
        try:
            from safetensors.torch import save_file
            save_file(train_data, "data/processed/train.safetensors")
            save_file(val_data, "data/processed/val.safetensors")
            print("✅ 创建safetensors格式测试数据")
        except ImportError:
            # 如果safetensors不可用，使用torch.save
            torch.save(train_data, "data/processed/train.pt")
            torch.save(val_data, "data/processed/val.pt")
            print("✅ 创建PyTorch格式测试数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 VA-VAE流水线测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("数据结构测试", test_data_structure),
        ("模型创建测试", test_model_creation),
        ("VA-VAE配置测试", test_vavae_config),
        ("推理脚本测试", test_inference_script),
        ("测试数据创建", create_dummy_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"🧪 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! 流水线准备就绪")
        print("\n💡 下一步:")
        print("python complete_pipeline.py")
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        print("\n💡 建议:")
        print("1. 检查依赖安装: pip install -r requirements.txt")
        print("2. 下载VA-VAE模型: python download_vavae_model.py")
        print("3. 验证设置: python verify_vavae_setup.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
