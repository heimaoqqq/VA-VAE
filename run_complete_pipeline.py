#!/usr/bin/env python3
"""
完整的微多普勒用户条件化生成流程
遵循LightningDiT原项目的正确方法
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n🚀 {description}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ 错误: {description} 失败")
        sys.exit(1)
    else:
        print(f"✅ {description} 完成")

def main():
    parser = argparse.ArgumentParser(description='完整的微多普勒用户条件化生成流程')
    
    # 数据路径
    parser.add_argument('--data_dir', type=str, required=True,
                       help='微多普勒数据目录 (包含train和val子目录)')
    parser.add_argument('--vavae_path', type=str, required=True,
                       help='预训练VA-VAE模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出根目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--devices', type=int, default=1,
                       help='GPU数量')
    
    # 生成参数
    parser.add_argument('--generate_user_ids', type=int, nargs='+', 
                       default=[1, 2, 3, 4, 5],
                       help='要生成的用户ID列表')
    parser.add_argument('--num_samples_per_user', type=int, default=4,
                       help='每个用户生成的样本数')
    
    # 流程控制
    parser.add_argument('--skip_extraction', action='store_true',
                       help='跳过特征提取阶段')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练阶段')
    parser.add_argument('--skip_generation', action='store_true',
                       help='跳过生成阶段')
    
    args = parser.parse_args()
    
    print("🎯 微多普勒用户条件化生成 - 完整流程")
    print("=" * 60)
    print("基于LightningDiT原项目的正确实现方法")
    print("=" * 60)
    
    # 创建输出目录结构
    output_dir = Path(args.output_dir)
    latent_dir = output_dir / "latent_features"
    model_dir = output_dir / "trained_models"
    generation_dir = output_dir / "generated_images"
    
    for dir_path in [latent_dir, model_dir, generation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输出目录结构:")
    print(f"  根目录: {output_dir}")
    print(f"  潜在特征: {latent_dir}")
    print(f"  训练模型: {model_dir}")
    print(f"  生成图像: {generation_dir}")
    
    # 阶段1: 特征提取
    if not args.skip_extraction:
        train_latent_file = latent_dir / "train.safetensors"
        val_latent_file = latent_dir / "val.safetensors"
        
        if not (train_latent_file.exists() and val_latent_file.exists()):
            cmd = [
                "python", "stage1_extract_features.py",
                "--data_dir", args.data_dir,
                "--vavae_path", args.vavae_path,
                "--output_path", str(latent_dir),
                "--batch_size", str(args.batch_size)
            ]
            run_command(cmd, "阶段1: 特征提取")
        else:
            print("✅ 潜在特征已存在，跳过提取阶段")
    
    # 阶段2: DiT训练
    if not args.skip_training:
        cmd = [
            "python", "stage2_train_dit.py",
            "--latent_dir", str(latent_dir),
            "--output_dir", str(model_dir),
            "--batch_size", str(args.batch_size),
            "--max_epochs", str(args.max_epochs),
            "--lr", str(args.lr),
            "--devices", str(args.devices),
            "--precision", "16-mixed"
        ]
        run_command(cmd, "阶段2: DiT训练")
    
    # 阶段3: 图像生成
    if not args.skip_generation:
        # 查找最新的检查点
        checkpoint_dir = model_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                # 选择最新的检查点
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                
                cmd = [
                    "python", "stage3_inference.py",
                    "--dit_checkpoint", str(latest_checkpoint),
                    "--vavae_path", args.vavae_path,
                    "--output_dir", str(generation_dir),
                    "--user_ids"] + [str(uid) for uid in args.generate_user_ids] + [
                    "--num_samples_per_user", str(args.num_samples_per_user),
                    "--guidance_scale", "4.0",
                    "--num_steps", "250"
                ]
                run_command(cmd, "阶段3: 图像生成")
            else:
                print("❌ 未找到训练好的模型检查点")
        else:
            print("❌ 检查点目录不存在")
    
    print("\n" + "=" * 60)
    print("🎉 完整流程执行完成!")
    print("=" * 60)
    
    # 显示结果摘要
    print("\n📊 结果摘要:")
    
    if latent_dir.exists():
        train_file = latent_dir / "train.safetensors"
        val_file = latent_dir / "val.safetensors"
        if train_file.exists() and val_file.exists():
            print(f"✅ 潜在特征: {train_file}, {val_file}")
    
    checkpoint_dir = model_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            print(f"✅ 训练模型: {len(checkpoints)} 个检查点")
    
    if generation_dir.exists():
        generated_images = list(generation_dir.glob("*.png"))
        if generated_images:
            print(f"✅ 生成图像: {len(generated_images)} 张图像")
    
    print(f"\n📁 所有结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
