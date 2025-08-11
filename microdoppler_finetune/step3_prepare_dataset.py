#!/usr/bin/env python3
"""
步骤3: 准备微多普勒数据集
划分训练集和验证集，为每个用户创建80/20划分
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

def split_user_data(user_folder: Path, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """为单个用户划分训练和验证数据"""
    random.seed(seed)
    
    # 获取该用户的所有图像
    images = sorted([str(img) for img in user_folder.glob("*.jpg")])
    
    if not images:
        print(f"⚠️ {user_folder.name} 没有找到图像")
        return [], []
    
    # 随机打乱并划分
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    return train_images, val_images

def prepare_microdoppler_dataset():
    """准备微多普勒数据集"""
    print("📊 准备微多普勒数据集")
    print("="*60)
    
    # 检测环境
    if os.path.exists('/kaggle/input/dataset'):
        data_root = Path('/kaggle/input/dataset')
        output_root = Path('/kaggle/working/data_split')
        print("📍 Kaggle环境：使用/kaggle/input/dataset")
    else:
        # 本地测试环境
        data_root = Path('G:/micro-doppler-dataset')  # 请根据实际路径修改
        output_root = Path.cwd() / 'data_split'
        print(f"📍 本地环境：数据路径 {data_root}")
    
    # 检查数据目录
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        print("请确保数据集位于正确位置")
        print("Kaggle: /kaggle/input/dataset/")
        print("本地: 修改脚本中的data_root路径")
        return None
    
    # 创建输出目录
    output_root.mkdir(exist_ok=True)
    
    # 统计数据
    user_stats = {}
    total_images = 0
    
    print("\n📂 扫描用户数据...")
    for user_id in range(1, 32):
        user_folder = data_root / f"ID_{user_id}"
        if user_folder.exists():
            image_count = len(list(user_folder.glob("*.jpg")))
            user_stats[f"ID_{user_id}"] = image_count
            total_images += image_count
            print(f"   ID_{user_id}: {image_count} 张图像")
    
    print(f"\n📊 数据统计:")
    print(f"   总用户数: {len(user_stats)}")
    print(f"   总图像数: {total_images}")
    print(f"   平均每用户: {total_images/len(user_stats):.1f} 张")
    
    # 为每个用户划分数据
    print("\n✂️ 划分训练/验证集 (80/20)...")
    
    dataset_split = {
        "train": {},
        "val": {},
        "test": {},  # 预留测试集
        "statistics": {
            "total_users": len(user_stats),
            "total_images": total_images,
            "train_images": 0,
            "val_images": 0,
            "split_ratio": "80/20",
            "random_seed": 42
        }
    }
    
    # 每个用户内部划分（80%训练，20%验证）
    print(f"\n🎯 每个用户内部划分 80/20")
    
    for user_id in range(1, 32):
        user_folder = data_root / f"ID_{user_id}"
        if not user_folder.exists():
            continue
        
        user_key = f"ID_{user_id}"
        
        # 每个用户内部80/20划分
        train_images, val_images = split_user_data(user_folder, train_ratio=0.8)
        dataset_split["train"][user_key] = train_images
        dataset_split["val"][user_key] = val_images
        dataset_split["statistics"]["train_images"] += len(train_images)
        dataset_split["statistics"]["val_images"] += len(val_images)
        print(f"   {user_key}: {len(train_images)} 训练, {len(val_images)} 验证")
    
    # 保存划分信息
    split_file = output_root / "dataset_split.json"
    with open(split_file, 'w') as f:
        json.dump(dataset_split, f, indent=2)
    print(f"\n✅ 数据划分已保存到: {split_file}")
    
    # 创建用户标签映射
    user_labels = {f"ID_{i}": i-1 for i in range(1, 32)}  # 0-30
    labels_file = output_root / "user_labels.json"
    with open(labels_file, 'w') as f:
        json.dump(user_labels, f, indent=2)
    print(f"✅ 用户标签已保存到: {labels_file}")
    
    # 创建训练配置
    train_config = {
        "data_root": str(data_root),
        "split_file": str(split_file),
        "labels_file": str(labels_file),
        "num_users": len(user_stats),
        "split_strategy": "per_user_80_20",  # 每个用户内部80/20划分
        "image_size": 256,
        "batch_size": 8,
        "num_workers": 4
    }
    
    config_file = output_root / "train_config.json"
    with open(config_file, 'w') as f:
        json.dump(train_config, f, indent=2)
    print(f"✅ 训练配置已保存到: {config_file}")
    
    # 显示总结
    print("\n" + "="*60)
    print("📊 数据集准备完成!")
    print(f"   训练图像: {dataset_split['statistics']['train_images']}")
    print(f"   验证图像: {dataset_split['statistics']['val_images']}")
    print(f"   总用户数: {len(user_stats)} 个")
    print(f"   每个用户都有训练和验证数据（80/20划分）")
    
    print("\n下一步:")
    print("1. 运行 python step4_train_stage1.py 开始第一阶段训练（语义对齐）")
    print("2. 完成后运行 python step5_train_stage2.py 进行第二阶段训练（整体微调）")
    
    return output_root

def verify_dataset_structure(data_root: Path):
    """验证数据集结构"""
    print("\n🔍 验证数据集结构...")
    
    issues = []
    
    # 检查用户文件夹
    for user_id in range(1, 32):
        user_folder = data_root / f"ID_{user_id}"
        if not user_folder.exists():
            issues.append(f"缺少用户文件夹: ID_{user_id}")
        else:
            images = list(user_folder.glob("*.jpg"))
            if len(images) == 0:
                issues.append(f"ID_{user_id} 没有图像")
            elif len(images) < 50:
                issues.append(f"ID_{user_id} 图像过少: {len(images)}")
    
    if issues:
        print("⚠️ 发现以下问题:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ 数据集结构正常")
    
    return len(issues) == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None, help='数据集路径')
    parser.add_argument('--verify_only', action='store_true', help='仅验证数据集')
    args = parser.parse_args()
    
    if args.data_root:
        data_root = Path(args.data_root)
        if args.verify_only:
            verify_dataset_structure(data_root)
        else:
            # 自定义路径
            prepare_microdoppler_dataset()
    else:
        # 自动检测
        prepare_microdoppler_dataset()
