#!/usr/bin/env python3
"""
调试训练脚本 - 直接在Python中运行训练，捕获详细错误
"""

import sys
import os
import inspect
from pathlib import Path

# 修复兼容性
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# 设置路径
taming_path = str(Path("taming-transformers").absolute())
if taming_path not in sys.path:
    sys.path.insert(0, taming_path)

vavae_path = str(Path("LightningDiT/vavae").absolute())
if vavae_path not in sys.path:
    sys.path.insert(0, vavae_path)

print(f"🔧 Python路径: {sys.path[:3]}...")

def debug_config_loading():
    """调试配置文件加载"""
    print("\n🔍 测试配置文件加载...")
    
    try:
        from omegaconf import OmegaConf
        
        config_path = "configs/stage1_custom_data.yaml"
        config = OmegaConf.load(config_path)
        
        print(f"✅ 配置文件加载成功")
        print(f"📋 模型类型: {config.model.target}")
        print(f"📋 数据类型: {config.data.target}")
        print(f"📋 预训练模型: {config.get('weight_init', 'NOT FOUND')}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_model_instantiation(config):
    """调试模型实例化"""
    print("\n🔍 测试模型实例化...")
    
    try:
        # 切换到正确目录
        os.chdir("LightningDiT/vavae")
        
        # 导入必要模块
        from ldm.util import instantiate_from_config
        
        # 尝试实例化模型
        print("📦 实例化模型...")
        model = instantiate_from_config(config.model)
        print(f"✅ 模型实例化成功: {type(model)}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_data_loading(config):
    """调试数据加载"""
    print("\n🔍 测试数据模块加载...")
    
    try:
        from ldm.util import instantiate_from_config
        
        # 尝试实例化数据模块
        print("📦 实例化数据模块...")
        data = instantiate_from_config(config.data)
        print(f"✅ 数据模块实例化成功: {type(data)}")
        
        # 准备数据
        print("📦 准备数据...")
        data.prepare_data()
        print("✅ 数据准备成功")
        
        # 设置数据
        print("📦 设置数据...")
        data.setup()
        print("✅ 数据设置成功")
        
        # 获取数据加载器
        train_loader = data.train_dataloader()
        print(f"✅ 训练数据加载器创建成功，批次数: {len(train_loader)}")
        
        # 测试第一个批次
        first_batch = next(iter(train_loader))
        print(f"✅ 第一个批次加载成功，形状: {first_batch['image'].shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主调试函数"""
    print("🐛 开始训练调试...")
    
    # 1. 测试配置加载
    config = debug_config_loading()
    if not config:
        return False
    
    # 2. 测试模型实例化
    model = debug_model_instantiation(config)
    if not model:
        return False
    
    # 3. 测试数据加载
    data = debug_data_loading(config)
    if not data:
        return False
    
    print("\n✅ 所有组件调试通过！")
    print("💡 问题可能在于PyTorch Lightning训练器或其他运行时问题")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 建议：问题可能在训练循环中，不是组件初始化")
    else:
        print("\n❌ 发现了组件初始化问题，需要修复")
