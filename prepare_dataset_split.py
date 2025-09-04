#!/usr/bin/env python3
"""
准备31用户微多普勒数据集划分
从/kaggle/input/dataset读取31个用户文件夹，按8:2划分训练/验证集
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import argparse

def create_dataset_split(dataset_root, output_file, train_ratio=0.8, seed=42):
    """
    创建数据集划分
    
    Args:
        dataset_root: 数据集根目录 (/kaggle/input/dataset)
        output_file: 输出JSON文件路径
        train_ratio: 训练集比例
        seed: 随机种子
    """
    
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    
    print(f"🔍 扫描数据集: {dataset_root}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 扫描用户文件夹
    user_folders = [d for d in dataset_root.iterdir() if d.is_dir()]
    user_folders.sort()  # 确保顺序一致
    
    print(f"📊 发现 {len(user_folders)} 个用户文件夹")
    
    if len(user_folders) != 31:
        print(f"⚠️ 预期31个用户，实际发现{len(user_folders)}个")
    
    # 统计和划分数据
    train_data = {}
    val_data = {}
    total_images = 0
    
    for user_folder in user_folders:
        user_id = user_folder.name
        print(f"🔄 处理用户: {user_id}")
        
        # 获取所有jpg图像
        image_files = list(user_folder.glob("*.jpg")) + list(user_folder.glob("*.JPG"))
        image_paths = [str(img_file) for img_file in image_files]
        
        if not image_paths:
            print(f"⚠️ 用户 {user_id} 没有找到图像文件")
            continue
        
        print(f"   找到 {len(image_paths)} 张图像")
        total_images += len(image_paths)
        
        # 随机打乱
        random.shuffle(image_paths)
        
        # 按比例划分
        n_train = int(len(image_paths) * train_ratio)
        
        train_data[user_id] = image_paths[:n_train]
        val_data[user_id] = image_paths[n_train:]
        
        print(f"   训练集: {len(train_data[user_id])} 张")
        print(f"   验证集: {len(val_data[user_id])} 张")
    
    # 创建完整的数据划分
    data_split = {
        'train': train_data,
        'val': val_data,
        'metadata': {
            'total_users': len(user_folders),
            'total_images': total_images,
            'train_ratio': train_ratio,
            'seed': seed,
            'dataset_root': str(dataset_root)
        }
    }
    
    # 保存到文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data_split, f, indent=2)
    
    # 统计信息
    train_total = sum(len(images) for images in train_data.values())
    val_total = sum(len(images) for images in val_data.values())
    
    print(f"\n📊 数据集划分完成:")
    print(f"   用户数: {len(user_folders)}")
    print(f"   总图像: {total_images}")
    print(f"   训练集: {train_total} 张 ({train_total/total_images*100:.1f}%)")
    print(f"   验证集: {val_total} 张 ({val_total/total_images*100:.1f}%)")
    print(f"   平均每用户训练样本: {train_total/len(user_folders):.1f}")
    print(f"   平均每用户验证样本: {val_total/len(user_folders):.1f}")
    print(f"   保存路径: {output_file}")
    
    return output_file

def validate_split(split_file):
    """验证数据划分文件"""
    print(f"\n🔍 验证数据划分: {split_file}")
    
    with open(split_file, 'r') as f:
        data_split = json.load(f)
    
    train_data = data_split['train']
    val_data = data_split['val']
    metadata = data_split['metadata']
    
    print(f"📊 元数据:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    # 检查每个用户
    print(f"\n👥 用户详情:")
    for user_id in sorted(train_data.keys()):
        train_count = len(train_data[user_id])
        val_count = len(val_data.get(user_id, []))
        total_count = train_count + val_count
        
        print(f"   {user_id}: {total_count} 总计 ({train_count} 训练, {val_count} 验证)")
        
        # 检查文件是否存在
        missing_files = []
        for img_path in train_data[user_id][:3]:  # 只检查前3个文件
            if not Path(img_path).exists():
                missing_files.append(img_path)
        
        if missing_files:
            print(f"     ⚠️ 发现缺失文件: {missing_files}")
    
    print(f"\n✅ 验证完成")

def main():
    parser = argparse.ArgumentParser(description='创建31用户微多普勒数据集划分')
    parser.add_argument('--dataset_root', type=str, default='/kaggle/input/dataset',
                       help='数据集根目录')
    parser.add_argument('--output_file', type=str, default='/kaggle/working/dataset_split.json',
                       help='输出JSON文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--validate', action='store_true',
                       help='验证生成的划分文件')
    
    args = parser.parse_args()
    
    try:
        # 创建数据划分
        output_file = create_dataset_split(
            dataset_root=args.dataset_root,
            output_file=args.output_file,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # 验证划分
        if args.validate:
            validate_split(output_file)
        
        print(f"\n🎉 数据集准备完成！")
        print(f"📝 下一步: 使用 {output_file} 开始训练")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
