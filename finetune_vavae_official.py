#!/usr/bin/env python3
"""
VA-VAE官方单阶段微调脚本
基于官方LightningDiT方法，在预训练模型基础上微调
"""

import os
import sys
import subprocess
import inspect
from pathlib import Path

def setup_environment():
    """设置环境和路径"""
    print("🔧 设置环境...")
    
    # 修复兼容性
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
        print("✅ 已应用 getargspec 兼容性修复")
    
    # 设置taming路径
    taming_path = str(Path("taming-transformers").absolute())
    if taming_path not in sys.path:
        sys.path.insert(0, taming_path)
    
    # 设置环境变量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = f"{taming_path}:{current_pythonpath}" if current_pythonpath else taming_path
    os.environ['PYTHONPATH'] = new_pythonpath
    
    print(f"🔧 PYTHONPATH: {new_pythonpath}")

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import taming
        print("✅ taming-transformers 已安装")
    except ImportError:
        print("❌ taming-transformers 未安装")
        return False
    
    try:
        import pytorch_lightning
        print(f"✅ pytorch-lightning {pytorch_lightning.__version__} 已安装")
    except ImportError:
        print("❌ pytorch-lightning 未安装")
        return False
    
    return True

def check_files():
    """检查必要文件"""
    print("🔍 检查文件...")
    
    # 检查预训练模型
    model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    if os.path.exists(model_path):
        print(f"✅ 预训练模型存在: {model_path}")
    else:
        print(f"❌ 预训练模型不存在: {model_path}")
        return False
    
    # 检查数据目录
    data_path = "/kaggle/input/dataset"
    if os.path.exists(data_path):
        print(f"✅ 数据目录存在: {data_path}")
    else:
        print(f"❌ 数据目录不存在: {data_path}")
        return False
    
    # 检查配置文件
    config_path = "configs/vavae_finetune_custom.yaml"
    if os.path.exists(config_path):
        print(f"✅ 配置文件存在: {config_path}")
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    return True

def copy_custom_loader():
    """复制自定义数据加载器到LightningDiT目录"""
    print("📋 复制自定义数据加载器...")
    
    source_path = "custom_data_loader.py"
    target_path = "LightningDiT/vavae/custom_data_loader.py"
    
    if os.path.exists(source_path):
        import shutil
        shutil.copy2(source_path, target_path)
        print(f"✅ 已复制: {source_path} -> {target_path}")
        return True
    else:
        print(f"❌ 源文件不存在: {source_path}")
        return False

def run_finetune():
    """执行微调"""
    print("\n🚀 开始VA-VAE微调...")
    print("📋 配置文件: configs/vavae_finetune_custom.yaml")
    
    # 切换到正确目录
    vavae_dir = "LightningDiT/vavae"
    config_path = "../../configs/vavae_finetune_custom.yaml"
    
    # 构建命令
    cmd = [
        sys.executable, "main.py",
        "--base", config_path,
        "--train"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"📁 工作目录: {vavae_dir}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = os.environ.get('PYTHONPATH', '')
    
    try:
        # 执行训练
        process = subprocess.Popen(
            cmd,
            cwd=vavae_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ 微调完成")
            return True
        else:
            print(f"❌ 微调失败，退出码: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def main():
    """主函数"""
    print("============================================================")
    print("🎯 VA-VAE官方单阶段微调")
    print("============================================================")
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败")
        return False
    
    # 3. 检查文件
    if not check_files():
        print("❌ 文件检查失败")
        return False
    
    # 4. 复制自定义加载器
    if not copy_custom_loader():
        print("❌ 自定义加载器复制失败")
        return False
    
    # 5. 执行微调
    success = run_finetune()
    
    if success:
        print("\n🎉 微调成功完成！")
        print("💡 检查点保存在: LightningDiT/vavae/logs/*/checkpoints/")
        print("💡 使用 evaluate_vavae.py 评估微调后的模型")
    else:
        print("\n❌ 微调失败")
    
    return success

if __name__ == "__main__":
    main()
