#!/usr/bin/env python3
"""
步骤3: 准备微多普勒数据集
- 使用data_split.py进行数据集划分
- 验证数据集结构
- 生成数据集统计信息
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

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

def check_input_dataset(input_dir):
    """检查输入数据集结构"""
    print("\n📁 检查输入数据集结构...")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 检查ID_1到ID_31目录
    user_dirs = []
    for i in range(1, 32):  # ID_1 到 ID_31
        user_dir = input_path / f"ID_{i}"
        if user_dir.exists():
            user_dirs.append(user_dir)
            
            # 统计图像文件
            image_files = list(user_dir.glob("*.png")) + \
                         list(user_dir.glob("*.jpg")) + \
                         list(user_dir.glob("*.jpeg"))
            
            print(f"✅ ID_{i}: {len(image_files)} 张图像")
            
            if len(image_files) == 0:
                print(f"⚠️ 警告: ID_{i} 目录为空")
        else:
            print(f"❌ 缺失: ID_{i}")
    
    if len(user_dirs) == 0:
        print("❌ 未找到任何用户目录 (ID_1, ID_2, ...)")
        print("💡 请确保数据集结构为:")
        print("   input_dir/")
        print("   ├── ID_1/")
        print("   ├── ID_2/")
        print("   └── ... ID_31/")
        return False
    
    print(f"✅ 找到 {len(user_dirs)} 个用户目录")
    return True

def run_data_split(input_dir, output_dir):
    """运行数据集划分"""
    print("\n🔄 运行数据集划分...")
    
    # 构建data_split.py命令
    cmd = f"python data_split.py " \
          f"--input_dir {input_dir} " \
          f"--output_dir {output_dir} " \
          f"--train_ratio 0.8 " \
          f"--val_ratio 0.2 " \
          f"--seed 42 " \
          f"--min_samples_per_user 10 " \
          f"--image_extensions png,jpg,jpeg"
    
    return run_command(cmd, "数据集划分")

def verify_output_dataset(output_dir):
    """验证输出数据集结构"""
    print("\n🔍 验证输出数据集结构...")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return False
    
    # 检查train和val目录
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    if not train_dir.exists():
        print("❌ 训练集目录不存在")
        return False
    
    if not val_dir.exists():
        print("❌ 验证集目录不存在")
        return False
    
    # 统计每个用户的数据
    print("\n📊 数据集统计:")
    print("用户ID | 训练集 | 验证集 | 总计")
    print("-" * 35)
    
    total_train = 0
    total_val = 0
    
    for user_id in range(1, 32):  # 用户1到31
        train_user_dir = train_dir / f"user{user_id}"
        val_user_dir = val_dir / f"user{user_id}"
        
        train_count = len(list(train_user_dir.glob("*.png"))) + \
                     len(list(train_user_dir.glob("*.jpg"))) + \
                     len(list(train_user_dir.glob("*.jpeg"))) if train_user_dir.exists() else 0
        
        val_count = len(list(val_user_dir.glob("*.png"))) + \
                   len(list(val_user_dir.glob("*.jpg"))) + \
                   len(list(val_user_dir.glob("*.jpeg"))) if val_user_dir.exists() else 0
        
        if train_count > 0 or val_count > 0:
            print(f"用户{user_id:2d}  |  {train_count:3d}   |  {val_count:3d}   | {train_count + val_count:3d}")
            total_train += train_count
            total_val += val_count
    
    print("-" * 35)
    print(f"总计    |  {total_train:3d}   |  {total_val:3d}   | {total_train + total_val:3d}")
    
    # 检查比例
    if total_train + total_val > 0:
        train_ratio = total_train / (total_train + total_val)
        val_ratio = total_val / (total_train + total_val)
        print(f"\n📈 实际比例: 训练 {train_ratio:.1%}, 验证 {val_ratio:.1%}")
    
    return True

def create_dataset_config(output_dir):
    """创建数据集配置文件"""
    print("\n📝 创建数据集配置文件...")
    
    config_content = f"""# 微多普勒数据集配置
# 生成时间: {Path().cwd()}

dataset:
  name: "micro_doppler_gait"
  num_users: 31
  image_size: 256
  channels: 3
  
  # 数据路径
  train_dir: "{output_dir}/train"
  val_dir: "{output_dir}/val"
  
  # 数据统计
  total_samples: "见split_info.txt"
  train_val_ratio: "8:2"
  
  # 用户信息
  user_format: "user1, user2, ..., user31"
  image_format: "image_001.png, image_002.png, ..."
  
# 训练配置建议
training:
  batch_size: 2  # 适合T4×2 GPU
  num_workers: 4
  pin_memory: true
  
# VA-VAE微调配置
vavae_finetune:
  base_lr: 1.0e-05
  max_epochs: 100
  warmup_epochs: 10
"""
    
    config_file = Path(output_dir) / "dataset_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已保存: {config_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='步骤3: 准备微多普勒数据集')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='输入数据目录 (包含ID_1, ID_2, ..., ID_31)')
    parser.add_argument('--output_dir', type=str, default='micro_doppler_dataset',
                       help='输出数据目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 步骤3: 准备微多普勒数据集")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 检查输入数据集
    if not check_input_dataset(args.input_dir):
        print("\n❌ 输入数据集检查失败")
        return False
    
    # 2. 运行数据集划分
    if not run_data_split(args.input_dir, args.output_dir):
        print("\n❌ 数据集划分失败")
        return False
    
    # 3. 验证输出数据集
    if not verify_output_dataset(args.output_dir):
        print("\n❌ 输出数据集验证失败")
        return False
    
    # 4. 创建配置文件
    create_dataset_config(args.output_dir)
    
    print("\n✅ 步骤3完成！微多普勒数据集准备就绪")
    print(f"📁 数据集位置: {args.output_dir}")
    print("📋 下一步: python step4_finetune_vavae.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
