#!/usr/bin/env python3
"""
快速测试修复后的检查点加载
"""

import os
import sys
import torch

def test_safetensors_loading():
    """测试safetensors加载"""
    print("🧪 测试safetensors检查点加载...")
    
    checkpoint_path = "/kaggle/working/trained_models/best_model"
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if not os.path.exists(safetensors_path):
        print(f"❌ 文件不存在: {safetensors_path}")
        return False
    
    try:
        # 添加路径
        sys.path.append('LightningDiT')
        from models import LightningDiT_models
        from safetensors.torch import load_file
        
        # 创建模型
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
        
        print(f"✅ 模型创建成功")
        
        # 加载检查点
        checkpoint = load_file(safetensors_path)
        print(f"✅ safetensors加载成功")
        print(f"   检查点键数量: {len(checkpoint.keys())}")
        
        # 检查键匹配
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        
        print(f"   模型参数: {len(model_keys)}")
        print(f"   检查点参数: {len(checkpoint_keys)}")
        print(f"   缺失: {len(missing)}")
        print(f"   多余: {len(unexpected)}")
        
        # 加载权重
        model.load_state_dict(checkpoint, strict=False)
        print("✅ 权重加载成功!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """测试推理"""
    print("\n🧪 测试推理生成...")
    
    try:
        # 运行简单推理测试
        cmd = '''python stage3_inference.py \
    --dit_checkpoint /kaggle/working/trained_models/best_model \
    --vavae_config vavae_config.yaml \
    --output_dir /kaggle/working/test_fix \
    --user_ids 1 2 \
    --num_samples_per_user 1 \
    --seed 42'''
        
        print(f"运行命令: {cmd}")
        result = os.system(cmd)
        
        if result == 0:
            print("✅ 推理测试成功!")
            
            # 检查输出
            output_dir = "/kaggle/working/test_fix"
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"   生成文件: {files}")
                
                # 检查文件大小
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(output_dir, file)
                        size_kb = os.path.getsize(file_path) / 1024
                        print(f"   {file}: {size_kb:.1f} KB")
            
            return True
        else:
            print("❌ 推理测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 推理测试异常: {e}")
        return False

def main():
    """主函数"""
    print("🔧 快速修复验证")
    print("=" * 40)
    
    # 测试1: safetensors加载
    if test_safetensors_loading():
        print("\n✅ 检查点加载修复成功!")
        
        # 测试2: 推理
        if test_inference():
            print("\n🎉 完全修复成功!")
            print("现在生成的图像应该是高质量的微多普勒信号，而不是噪声")
            
            print("\n🚀 运行完整推理:")
            print('''python stage3_inference.py \\
    --dit_checkpoint /kaggle/working/trained_models/best_model \\
    --vavae_config vavae_config.yaml \\
    --output_dir /kaggle/working/generated_images \\
    --user_ids 1 2 3 4 5 \\
    --num_samples_per_user 4 \\
    --seed 42''')
            
            return True
        else:
            print("\n⚠️  检查点加载成功，但推理仍有问题")
            return False
    else:
        print("\n❌ 检查点加载仍有问题")
        print("💡 建议:")
        print("1. 检查safetensors库是否安装: pip install safetensors")
        print("2. 检查训练是否完全成功")
        print("3. 重新运行训练")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
