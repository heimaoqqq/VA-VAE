#!/usr/bin/env python3
"""
完整的微多普勒信号生成流水线
包含特征提取、模型训练和推理生成
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n🚀 {description}")
    print(f"命令: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} 完成")
        if result.stdout:
            print("输出:")
            print(result.stdout)
    else:
        print(f"❌ {description} 失败")
        print("错误:")
        print(result.stderr)
        return False
    
    return True

def check_data_structure():
    """检查数据结构"""
    print("🔍 检查数据结构...")
    
    required_paths = [
        "data/raw",
        "data/processed", 
        "LightningDiT/vavae"
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"❌ 缺少目录: {path}")
            return False
        print(f"✅ 找到目录: {path}")
    
    return True

def stage1_extract_features():
    """阶段1: 提取特征"""
    print("\n" + "="*60)
    print("🎯 阶段1: 提取VA-VAE特征")
    print("="*60)
    
    # 检查是否已有特征文件
    train_features = "data/processed/train.safetensors"
    val_features = "data/processed/val.safetensors"
    
    if os.path.exists(train_features) and os.path.exists(val_features):
        print("✅ 特征文件已存在，跳过提取")
        return True
    
    cmd = "python stage1_extract_features.py"
    return run_command(cmd, "特征提取")

def stage2_train_model(max_epochs=50, output_dir="./checkpoints"):
    """阶段2: 训练DiT模型"""
    print("\n" + "="*60)
    print("🎯 阶段2: 训练DiT模型")
    print("="*60)
    
    # 检查是否已有训练好的模型
    best_model_path = os.path.join(output_dir, "best_model")
    if os.path.exists(best_model_path):
        print("✅ 训练好的模型已存在，跳过训练")
        return True, best_model_path
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python stage2_train_dit.py --latent_dir ./data/processed --output_dir {output_dir} --max_epochs {max_epochs} --batch_size 16"
    
    success = run_command(cmd, f"DiT模型训练 ({max_epochs} epochs)")
    
    if success and os.path.exists(best_model_path):
        return True, best_model_path
    else:
        print("⚠️  训练完成但未找到最佳模型，将使用随机模型")
        return True, None

def stage3_generate_samples(checkpoint_path=None, output_dir="./generated_samples"):
    """阶段3: 生成样本"""
    print("\n" + "="*60)
    print("🎯 阶段3: 生成微多普勒样本")
    print("="*60)
    
    # 构建命令
    cmd_parts = [
        "python stage3_inference.py",
        f"--vavae_config vavae_config.yaml",
        f"--output_dir {output_dir}",
        "--user_ids 1 2 3 4 5",
        "--num_samples_per_user 4",
        "--guidance_scale 4.0",
        "--num_steps 250"
    ]
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd_parts.append(f"--dit_checkpoint {checkpoint_path}")
        print(f"📥 使用训练好的模型: {checkpoint_path}")
    else:
        print("⚠️  没有找到训练好的模型，将使用随机初始化模型进行演示")
        # 创建一个虚拟检查点路径，让脚本知道要使用随机模型
        cmd_parts.append("--dit_checkpoint dummy_path")
    
    cmd = " ".join(cmd_parts)
    return run_command(cmd, "样本生成")

def main():
    parser = argparse.ArgumentParser(description='完整的微多普勒信号生成流水线')
    parser.add_argument('--skip_extract', action='store_true', help='跳过特征提取')
    parser.add_argument('--skip_train', action='store_true', help='跳过模型训练')
    parser.add_argument('--skip_generate', action='store_true', help='跳过样本生成')
    parser.add_argument('--max_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点目录')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help='生成样本输出目录')
    parser.add_argument('--force_retrain', action='store_true', help='强制重新训练')
    
    args = parser.parse_args()
    
    print("🎯 微多普勒信号生成完整流水线")
    print("=" * 60)
    print(f"特征提取: {'跳过' if args.skip_extract else '执行'}")
    print(f"模型训练: {'跳过' if args.skip_train else f'执行 ({args.max_epochs} epochs)'}")
    print(f"样本生成: {'跳过' if args.skip_generate else '执行'}")
    print(f"检查点目录: {args.checkpoint_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 检查数据结构
    if not check_data_structure():
        print("❌ 数据结构检查失败，请确保数据目录正确")
        return 1
    
    checkpoint_path = None
    
    # 阶段1: 特征提取
    if not args.skip_extract:
        if not stage1_extract_features():
            print("❌ 特征提取失败")
            return 1
    
    # 阶段2: 模型训练
    if not args.skip_train:
        # 如果强制重新训练，删除现有检查点
        if args.force_retrain:
            best_model_path = os.path.join(args.checkpoint_dir, "best_model")
            if os.path.exists(best_model_path):
                import shutil
                shutil.rmtree(best_model_path)
                print("🗑️  删除现有模型，强制重新训练")
        
        success, checkpoint_path = stage2_train_model(args.max_epochs, args.checkpoint_dir)
        if not success:
            print("❌ 模型训练失败")
            return 1
    else:
        # 检查是否有现有的检查点
        best_model_path = os.path.join(args.checkpoint_dir, "best_model")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
    
    # 阶段3: 样本生成
    if not args.skip_generate:
        if not stage3_generate_samples(checkpoint_path, args.output_dir):
            print("❌ 样本生成失败")
            return 1
    
    print("\n" + "="*60)
    print("🎉 流水线执行完成!")
    print("="*60)
    
    if checkpoint_path:
        print(f"📥 使用的模型: {checkpoint_path}")
    else:
        print("⚠️  使用随机初始化模型")
    
    if not args.skip_generate:
        print(f"📁 生成的样本保存在: {args.output_dir}")
    
    print("\n💡 提示:")
    print("- 如果生成质量不好，尝试增加训练轮数: --max_epochs 100")
    print("- 如果要重新训练: --force_retrain")
    print("- 如果只想生成样本: --skip_extract --skip_train")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
