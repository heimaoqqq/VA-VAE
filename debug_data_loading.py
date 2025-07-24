#!/usr/bin/env python3
"""
调试数据加载问题的脚本
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

def test_data_loading():
    """测试数据加载和维度转换"""
    print("🔍 测试数据加载和维度转换...")
    
    # 添加项目路径
    sys.path.append('.')
    
    try:
        from minimal_micro_doppler_dataset import MicroDopplerDataset
        print("✅ 成功导入MicroDopplerDataset")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 测试数据集
    data_dir = "/kaggle/working/data_split/train"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    print(f"📁 测试数据目录: {data_dir}")
    
    # 创建数据集
    try:
        dataset = MicroDopplerDataset(data_dir, split='train')
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False
    
    # 测试单个样本
    try:
        sample = dataset[0]
        image = sample['image']
        user_id = sample['user_id']
        
        print(f"📊 样本信息:")
        print(f"  - 图像维度: {image.shape}")
        print(f"  - 图像类型: {image.dtype}")
        print(f"  - 图像范围: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  - 用户ID: {user_id}")
        
        # 验证维度
        if image.dim() == 3 and image.shape[0] == 3:
            print("✅ 图像维度正确: (C, H, W)")
            return True
        else:
            print(f"❌ 图像维度错误: 期望(3, 256, 256)，得到{image.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 样本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    # 添加LightningDiT路径
    sys.path.append('LightningDiT')
    
    try:
        from tokenizer.autoencoder import AutoencoderKL
        print("✅ 成功导入AutoencoderKL")
    except Exception as e:
        print(f"❌ 导入AutoencoderKL失败: {e}")
        return False
    
    # 测试模型创建
    try:
        model = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=None,
            model_type='vavae'
        )
        print("✅ 模型创建成功")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model.encode(dummy_input)
            print(f"✅ 编码测试成功，输出类型: {type(output)}")
            
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pretrained_loading():
    """测试预训练模型加载"""
    print("\n🔍 测试预训练模型加载...")
    
    pretrained_path = "/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt"
    if not os.path.exists(pretrained_path):
        print(f"❌ 预训练模型文件不存在: {pretrained_path}")
        return False
    
    print(f"📁 预训练模型路径: {pretrained_path}")
    
    # 添加LightningDiT路径
    sys.path.append('LightningDiT')
    
    try:
        from tokenizer.autoencoder import AutoencoderKL
        
        # 使用ckpt_path参数加载
        model = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=pretrained_path,
            model_type='vavae'
        )
        print("✅ 预训练模型加载成功")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model.encode(dummy_input)
            print(f"✅ 预训练模型编码测试成功")
            
        return True
        
    except Exception as e:
        print(f"❌ 预训练模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始调试测试...")
    print("=" * 50)
    
    # 检查工作目录
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📁 Python路径: {sys.path[:3]}...")
    
    # 测试数据加载
    data_ok = test_data_loading()
    
    # 测试模型加载
    model_ok = test_model_loading()
    
    # 测试预训练模型加载
    pretrained_ok = test_pretrained_loading()
    
    # 总结
    print("\n" + "=" * 50)
    print("🎯 测试总结:")
    print(f"  - 数据加载: {'✅ 通过' if data_ok else '❌ 失败'}")
    print(f"  - 模型创建: {'✅ 通过' if model_ok else '❌ 失败'}")
    print(f"  - 预训练加载: {'✅ 通过' if pretrained_ok else '❌ 失败'}")
    
    if data_ok and model_ok and pretrained_ok:
        print("🎉 所有测试通过！可以开始训练")
    else:
        print("⚠️ 存在问题，需要修复")

if __name__ == "__main__":
    main()
