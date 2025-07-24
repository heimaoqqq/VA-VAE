"""
数据集划分程序
将未划分的微多普勒数据集按用户进行训练/验证/测试划分
"""

import os
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='微多普勒数据集划分')

    parser.add_argument('--input_dir', type=str, required=True, help='输入数据目录')
    parser.add_argument('--output_dir', type=str, default='data_split', help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--use_test_set', action='store_true', help='是否创建测试集')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例（如果使用）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--min_samples_per_user', type=int, default=3, help='每个用户最少样本数')
    parser.add_argument('--image_extensions', type=str, default='png,jpg,jpeg', help='图像文件扩展名')

    return parser.parse_args()


def scan_data_files(data_dir, image_extensions):
    """
    扫描数据文件，按用户组织
    适配 ID_1, ID_2, ..., ID_31 的目录结构

    Args:
        data_dir: 数据目录 (包含 ID_1, ID_2, ... 子目录)
        image_extensions: 图像文件扩展名列表

    Returns:
        dict: {user_id: [file_paths]}
    """
    data_path = Path(data_dir)
    user_files = defaultdict(list)

    # 解析图像扩展名
    extensions = [ext.strip().lower() for ext in image_extensions.split(',')]

    # 扫描 ID_1 到 ID_31 目录
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                # 解析用户ID: ID_1 -> 1, ID_2 -> 2, ...
                user_id = int(user_dir.name.split('_')[1])

                # 扫描该用户目录下的所有图像文件
                for ext in extensions:
                    for file_path in user_dir.glob(f"*.{ext}"):
                        user_files[user_id].append(file_path)

                print(f"用户 {user_id}: 找到 {len(user_files[user_id])} 个图像文件")

            except (ValueError, IndexError):
                print(f"警告: 无法解析目录名 {user_dir.name}")

    return dict(user_files)


def split_user_data(user_files, train_ratio, val_ratio, test_ratio, seed, min_samples_per_user, use_test_set):
    """
    按用户划分数据

    Args:
        user_files: {user_id: [file_paths]}
        train_ratio, val_ratio, test_ratio: 划分比例
        seed: 随机种子
        min_samples_per_user: 每个用户最少样本数
        use_test_set: 是否使用测试集

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]} 或 {'train': [...], 'val': [...]}
    """
    np.random.seed(seed)

    if use_test_set:
        splits = {'train': [], 'val': [], 'test': []}
        # 检查比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试比例之和必须等于1.0")
        print(f"数据划分比例 - 训练: {train_ratio}, 验证: {val_ratio}, 测试: {test_ratio}")
    else:
        splits = {'train': [], 'val': []}
        # 重新归一化比例
        total_ratio = train_ratio + val_ratio
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        print(f"数据划分比例 - 训练: {train_ratio:.2f}, 验证: {val_ratio:.2f}")

    print(f"每个用户最少样本数: {min_samples_per_user}")
    
    for user_id, files in user_files.items():
        num_files = len(files)
        
        # 检查样本数是否足够
        if num_files < min_samples_per_user:
            print(f"警告: 用户 {user_id} 只有 {num_files} 个样本，少于最少要求 {min_samples_per_user}")
            # 将所有样本都放入训练集
            splits['train'].extend(files)
            continue
        
        # 打乱文件顺序
        files_shuffled = files.copy()
        np.random.shuffle(files_shuffled)
        
        # 计算划分点
        train_end = int(num_files * train_ratio)

        if use_test_set:
            val_end = train_end + int(num_files * val_ratio)

            # 确保每个集合至少有一个样本（如果总样本数足够）
            if train_end == 0 and num_files > 0:
                train_end = 1
            if val_end == train_end and num_files > train_end:
                val_end = train_end + 1

            # 划分数据
            splits['train'].extend(files_shuffled[:train_end])
            splits['val'].extend(files_shuffled[train_end:val_end])
            splits['test'].extend(files_shuffled[val_end:])

            print(f"用户 {user_id:2d}: 总计 {num_files:3d} -> "
                  f"训练 {train_end:3d}, 验证 {val_end-train_end:3d}, 测试 {num_files-val_end:3d}")
        else:
            # 只有训练集和验证集
            # 确保每个集合至少有一个样本
            if train_end == 0 and num_files > 0:
                train_end = 1
            if train_end == num_files and num_files > 1:
                train_end = num_files - 1

            # 划分数据
            splits['train'].extend(files_shuffled[:train_end])
            splits['val'].extend(files_shuffled[train_end:])

            print(f"用户 {user_id:2d}: 总计 {num_files:3d} -> "
                  f"训练 {train_end:3d}, 验证 {num_files-train_end:3d}")
    
    return splits


def copy_files_to_splits(splits, output_dir):
    """
    将文件复制到对应的划分目录，并重命名为统一格式

    Args:
        splits: {'train': [...], 'val': [...], 'test': [...]}
        output_dir: 输出目录
    """
    output_path = Path(output_dir)

    for split_name, files in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n复制 {split_name} 集文件...")

        # 按用户组织文件以便重命名
        user_files = {}
        for file_path in files:
            # 从路径中提取用户ID
            if 'ID_' in str(file_path):
                user_id = int(file_path.parent.name.split('_')[1])
            else:
                # 如果已经是重命名格式
                parts = file_path.stem.split('_')
                user_id = int(parts[1])

            if user_id not in user_files:
                user_files[user_id] = []
            user_files[user_id].append(file_path)

        # 重命名并复制文件
        file_count = 0
        for user_id, user_file_list in user_files.items():
            for sample_idx, file_path in enumerate(user_file_list, 1):
                # 生成新文件名: user_XX_sample_XXX.ext
                new_name = f"user_{user_id:02d}_sample_{sample_idx:03d}{file_path.suffix}"
                dest_path = split_dir / new_name

                shutil.copy2(file_path, dest_path)
                file_count += 1

                if file_count % 100 == 0:
                    print(f"  已复制 {file_count}/{len(files)} 个文件")

        print(f"{split_name} 集完成: {len(files)} 个文件")


def create_split_info(splits, user_files, output_dir, use_test_set):
    """
    创建划分信息文件

    Args:
        splits: 划分结果
        user_files: 原始用户文件
        output_dir: 输出目录
        use_test_set: 是否使用测试集
    """
    output_path = Path(output_dir)
    
    # 统计信息
    info = {
        'total_files': sum(len(files) for files in splits.values()),
        'total_users': len(user_files),
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val'])
        },
        'user_distribution': {}
    }

    # 如果使用测试集，添加测试集信息
    if use_test_set and 'test' in splits:
        info['splits']['test'] = len(splits['test'])
    
    # 按用户统计分布
    for user_id in user_files.keys():
        user_splits = {'train': 0, 'val': 0, 'test': 0} if use_test_set else {'train': 0, 'val': 0}

        for split_name, split_files in splits.items():
            user_splits[split_name] = sum(1 for f in split_files
                                        if f.stem.startswith(f'user_{user_id:02d}_'))

        info['user_distribution'][f'user_{user_id:02d}'] = user_splits
    
    # 保存信息文件
    info_file = output_path / 'split_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("微多普勒数据集划分信息\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总文件数: {info['total_files']}\n")
        f.write(f"总用户数: {info['total_users']}\n\n")
        
        f.write("数据集划分:\n")
        for split_name, count in info['splits'].items():
            ratio = count / info['total_files'] * 100
            f.write(f"  {split_name}: {count} 个文件 ({ratio:.1f}%)\n")
        
        f.write("\n用户分布:\n")
        for user_id, user_splits in info['user_distribution'].items():
            total = sum(user_splits.values())
            if use_test_set and 'test' in user_splits:
                f.write(f"  {user_id}: 总计 {total:3d} -> "
                       f"训练 {user_splits['train']:3d}, "
                       f"验证 {user_splits['val']:3d}, "
                       f"测试 {user_splits['test']:3d}\n")
            else:
                f.write(f"  {user_id}: 总计 {total:3d} -> "
                       f"训练 {user_splits['train']:3d}, "
                       f"验证 {user_splits['val']:3d}\n")
    
    print(f"\n划分信息已保存到: {info_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return
    
    print("=" * 60)
    print("微多普勒数据集划分程序")
    print("=" * 60)
    
    # 调整比例（如果不使用测试集）
    if not args.use_test_set:
        # 重新归一化训练和验证比例
        total_ratio = args.train_ratio + args.val_ratio
        args.train_ratio = args.train_ratio / total_ratio
        args.val_ratio = args.val_ratio / total_ratio
        args.test_ratio = 0.0

    # 1. 扫描数据文件
    print("1. 扫描数据文件...")
    user_files = scan_data_files(args.input_dir, args.image_extensions)

    if not user_files:
        print("错误: 未找到符合格式的数据文件")
        print("目录结构应为: ID_1/, ID_2/, ..., ID_31/")
        return

    total_files = sum(len(files) for files in user_files.values())
    print(f"找到 {len(user_files)} 个用户的 {total_files} 个文件")

    # 2. 划分数据
    print("\n2. 划分数据...")
    splits = split_user_data(
        user_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.min_samples_per_user,
        args.use_test_set
    )
    
    # 3. 复制文件
    print(f"\n3. 复制文件到 {args.output_dir}...")
    copy_files_to_splits(splits, args.output_dir)
    
    # 4. 创建信息文件
    print("\n4. 创建划分信息...")
    create_split_info(splits, user_files, args.output_dir, args.use_test_set)
    
    print(f"\n数据集划分完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"  - train/: {len(splits['train'])} 个文件")
    print(f"  - val/:   {len(splits['val'])} 个文件")
    if args.use_test_set:
        print(f"  - test/:  {len(splits['test'])} 个文件")


if __name__ == '__main__':
    main()
