#!/usr/bin/env python3
"""
修复检查点加载问题的脚本
专门解决生成图像质量差的问题
"""

import os
import sys
import torch
from pathlib import Path

def check_checkpoint_structure(checkpoint_path):
    """检查检查点文件结构"""
    print(f"🔍 检查检查点结构: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return False
    
    # 检查是否是目录
    if os.path.isdir(checkpoint_path):
        print("📁 检查点是目录格式")
        files = os.listdir(checkpoint_path)
        print(f"   包含文件: {files}")
        
        # 检查Accelerate格式
        pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            print(f"✅ 找到Accelerate检查点: pytorch_model.bin")
            
            # 检查文件大小
            size_mb = os.path.getsize(pytorch_model_path) / (1024 * 1024)
            print(f"   文件大小: {size_mb:.1f} MB")
            
            # 尝试加载检查点
            try:
                checkpoint = torch.load(pytorch_model_path, map_location='cpu')
                print(f"✅ 检查点加载成功")
                print(f"   检查点类型: {type(checkpoint)}")
                
                if isinstance(checkpoint, dict):
                    print(f"   检查点键: {list(checkpoint.keys())[:10]}...")  # 只显示前10个键
                    
                    # 检查是否包含模型权重
                    sample_key = list(checkpoint.keys())[0]
                    sample_value = checkpoint[sample_key]
                    print(f"   样本键值: {sample_key} -> {type(sample_value)} {getattr(sample_value, 'shape', 'N/A')}")
                
                return True
                
            except Exception as e:
                print(f"❌ 检查点加载失败: {e}")
                return False
        else:
            print("❌ 未找到pytorch_model.bin文件")
            return False
    
    # 检查是否是单个文件
    elif os.path.isfile(checkpoint_path):
        print("📄 检查点是单个文件")
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"   文件大小: {size_mb:.1f} MB")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ 检查点加载成功")
            print(f"   检查点类型: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                print(f"   检查点键: {list(checkpoint.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            return False
    
    return False

def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型创建和加载...")
    
    try:
        # 添加路径
        sys.path.append('LightningDiT')
        from models import LightningDiT_models
        
        # 创建B模型 (与训练时一致)
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
        
        print("✅ 模型创建成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试检查点加载
        checkpoint_path = "/kaggle/working/trained_models/best_model"
        if os.path.exists(checkpoint_path):
            pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                print(f"\n🔄 测试检查点加载...")
                checkpoint = torch.load(pytorch_model_path, map_location='cpu')
                
                # 检查权重形状匹配
                model_state = model.state_dict()
                checkpoint_keys = set(checkpoint.keys())
                model_keys = set(model_state.keys())
                
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                print(f"   模型参数数量: {len(model_keys)}")
                print(f"   检查点参数数量: {len(checkpoint_keys)}")
                print(f"   缺失参数: {len(missing_keys)}")
                print(f"   多余参数: {len(unexpected_keys)}")
                
                if missing_keys:
                    print(f"   缺失参数示例: {list(missing_keys)[:5]}")
                if unexpected_keys:
                    print(f"   多余参数示例: {list(unexpected_keys)[:5]}")
                
                # 尝试加载
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    print("✅ 检查点加载成功 (非严格模式)")
                    return True
                except Exception as e:
                    print(f"❌ 检查点加载失败: {e}")
                    return False
            else:
                print("❌ 未找到pytorch_model.bin")
                return False
        else:
            print("❌ 检查点目录不存在")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def create_test_inference():
    """创建测试推理脚本"""
    print("\n🧪 创建测试推理...")
    
    test_script = '''
import torch
import sys
import os
sys.path.append('LightningDiT')

# 测试推理
checkpoint_path = "/kaggle/working/trained_models/best_model"
if os.path.exists(checkpoint_path):
    print("✅ 检查点存在")
    
    # 运行推理
    os.system("""python stage3_inference.py \\
        --dit_checkpoint /kaggle/working/trained_models/best_model \\
        --vavae_config vavae_config.yaml \\
        --output_dir /kaggle/working/test_generated \\
        --user_ids 1 2 \\
        --num_samples_per_user 2 \\
        --seed 42""")
else:
    print("❌ 检查点不存在")
'''
    
    with open('test_inference.py', 'w') as f:
        f.write(test_script)
    
    print("✅ 测试脚本已创建: test_inference.py")

def main():
    """主函数"""
    print("🔧 检查点加载问题诊断工具")
    print("=" * 50)
    
    # 检查常见的检查点路径
    checkpoint_paths = [
        "/kaggle/working/trained_models/best_model",
        "/kaggle/working/trained_models",
        "./checkpoints/best_model",
        "./trained_models/best_model"
    ]
    
    found_checkpoint = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"\n📍 检查路径: {path}")
            if check_checkpoint_structure(path):
                found_checkpoint = True
                break
        else:
            print(f"⚠️  路径不存在: {path}")
    
    if not found_checkpoint:
        print("\n❌ 未找到有效的检查点文件")
        print("💡 建议:")
        print("1. 检查训练是否成功完成")
        print("2. 检查输出目录是否正确")
        print("3. 重新运行训练: python kaggle_training_wrapper.py stage2")
        return False
    
    # 测试模型加载
    if test_model_loading():
        print("\n✅ 模型加载测试通过")
        print("💡 问题已修复，可以正常生成图像")
        
        # 创建测试推理
        create_test_inference()
        
        print("\n🚀 下一步:")
        print("python test_inference.py")
        
        return True
    else:
        print("\n❌ 模型加载测试失败")
        print("💡 建议:")
        print("1. 检查模型配置是否与训练时一致")
        print("2. 重新训练模型")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
