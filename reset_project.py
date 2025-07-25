#!/usr/bin/env python3
"""
重置项目：删除现有LightningDiT，重新克隆纯净版本
确保代码100%纯净，避免任何修改导致的问题
"""

import os
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"💻 执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print(f"错误: {e.stderr.strip()}")
        return False

def backup_models():
    """备份已下载的模型文件"""
    print("📦 备份模型文件...")
    
    models_dir = Path("official_models")
    backup_dir = Path("/tmp/backup_models")
    
    if models_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(models_dir, backup_dir)
        print(f"✅ 模型文件已备份到: {backup_dir}")
        return True
    else:
        print("⚠️ 没有找到模型文件目录")
        return False

def restore_models():
    """恢复模型文件"""
    print("📦 恢复模型文件...")
    
    backup_dir = Path("/tmp/backup_models")
    models_dir = Path("official_models")
    
    if backup_dir.exists():
        if models_dir.exists():
            shutil.rmtree(models_dir)
        shutil.copytree(backup_dir, models_dir)
        print(f"✅ 模型文件已恢复到: {models_dir}")
        return True
    else:
        print("❌ 没有找到备份文件")
        return False

def remove_lightningdit():
    """删除现有的LightningDiT目录"""
    print("🗑️ 删除现有LightningDiT目录...")
    
    lightning_dir = Path("LightningDiT")
    if lightning_dir.exists():
        shutil.rmtree(lightning_dir)
        print("✅ LightningDiT目录已删除")
        return True
    else:
        print("⚠️ LightningDiT目录不存在")
        return True

def clone_lightningdit():
    """重新克隆纯净的LightningDiT项目"""
    print("📥 重新克隆LightningDiT项目...")
    
    # 尝试克隆
    if run_command("git clone https://github.com/hustvl/LightningDiT.git", "克隆LightningDiT"):
        print("✅ LightningDiT克隆成功")
        return True
    
    # 如果失败，尝试浅克隆
    print("🔄 尝试浅克隆...")
    if run_command("git clone --depth 1 https://github.com/hustvl/LightningDiT.git", "浅克隆LightningDiT"):
        print("✅ LightningDiT浅克隆成功")
        return True
    
    print("❌ 克隆失败，请检查网络连接")
    return False

def verify_clone():
    """验证克隆是否成功"""
    print("🔍 验证克隆结果...")
    
    lightning_dir = Path("LightningDiT")
    if not lightning_dir.exists():
        print("❌ LightningDiT目录不存在")
        return False
    
    # 检查关键文件
    key_files = [
        "inference.py",
        "run_fast_inference.sh",
        "tokenizer/vavae.py",
        "models/lightningdit.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        full_path = lightning_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: 缺失")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 发现 {len(missing_files)} 个缺失文件")
        return False
    else:
        print("✅ 所有关键文件都存在")
        return True

def main():
    """主函数"""
    print("🔄 重置项目：获取纯净的LightningDiT代码")
    print("=" * 60)
    print("🎯 目标：确保代码100%纯净，避免任何修改导致的问题")
    
    # 1. 备份模型文件
    has_models = backup_models()
    
    # 2. 删除现有LightningDiT
    if not remove_lightningdit():
        print("❌ 删除失败")
        return False
    
    # 3. 重新克隆
    if not clone_lightningdit():
        print("❌ 克隆失败")
        return False
    
    # 4. 验证克隆
    if not verify_clone():
        print("❌ 验证失败")
        return False
    
    # 5. 恢复模型文件
    if has_models:
        if not restore_models():
            print("⚠️ 模型文件恢复失败，需要重新下载")
    
    print("\n✅ 项目重置完成！")
    print("🎉 现在拥有100%纯净的LightningDiT代码")
    print("\n📋 接下来的步骤:")
    print("1. 如果模型文件丢失，运行: python step1_download_models.py")
    print("2. 生成配置文件: python step2_setup_configs.py")
    print("3. 运行推理: python step3_run_inference.py")
    
    return True

if __name__ == "__main__":
    main()
