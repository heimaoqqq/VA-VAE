"""
雷达步态数据集重组工具
将按用户组织的数据集重组为按步态类型组织的数据集
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def organize_radar_gait_dataset(source_dir, target_dir):
    """
    重组雷达步态数据集
    
    Args:
        source_dir (str): 源数据集目录路径 (包含01-31用户文件夹)
        target_dir (str): 目标数据集目录路径 (按步态类型组织)
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 定义8种步态类型
    gait_types = [
        'Backpack_free',
        'Backpack_line', 
        'Bag_free',
        'Bag_line',
        'Bag_Phone_free',
        'Bag_Phone_line',
        'Normal_free',
        'Normal_line'
    ]
    
    # 为每种步态类型创建目标文件夹
    for gait_type in gait_types:
        gait_folder = target_path / gait_type
        gait_folder.mkdir(exist_ok=True)
        print(f"创建步态文件夹: {gait_folder}")
    
    # 统计信息
    total_files = 0
    copied_files = 0
    error_files = []
    
    # 遍历31个用户文件夹
    for user_id in range(1, 32):  # 01到31
        user_folder_name = f"{user_id:02d}"  # 格式化为两位数字，如01, 02, ...
        user_source_path = source_path / user_folder_name
        
        if not user_source_path.exists():
            print(f"警告: 用户文件夹不存在 - {user_source_path}")
            continue
            
        print(f"处理用户文件夹: {user_folder_name} -> ID_{user_id}")
        
        # 遍历该用户的8种步态
        for gait_type in gait_types:
            gait_source_path = user_source_path / gait_type
            
            if not gait_source_path.exists():
                print(f"  警告: 步态文件夹不存在 - {gait_source_path}")
                continue
            
            # 获取该步态文件夹下的所有jpg文件
            jpg_files = list(gait_source_path.glob("*.jpg"))
            total_files += len(jpg_files)
            
            if not jpg_files:
                print(f"  警告: 步态文件夹为空 - {gait_source_path}")
                continue
            
            # 复制文件到目标位置
            target_gait_folder = target_path / gait_type
            target_user_folder = target_gait_folder / f"ID_{user_id}"
            target_user_folder.mkdir(exist_ok=True)
            
            for jpg_file in tqdm(jpg_files, 
                               desc=f"  复制 {gait_type}/ID_{user_id}", 
                               leave=False):
                try:
                    target_file = target_user_folder / jpg_file.name
                    shutil.copy2(jpg_file, target_file)
                    copied_files += 1
                except Exception as e:
                    error_msg = f"复制失败: {jpg_file} -> {target_file}, 错误: {str(e)}"
                    error_files.append(error_msg)
                    print(f"    错误: {error_msg}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据集重组完成!")
    print(f"总文件数: {total_files}")
    print(f"成功复制: {copied_files}")
    print(f"失败文件: {len(error_files)}")
    
    if error_files:
        print("\n失败文件列表:")
        for error in error_files:
            print(f"  - {error}")
    
    # 打印目标目录结构
    print(f"\n目标目录结构: {target_path}")
    for gait_type in gait_types:
        gait_folder = target_path / gait_type
        if gait_folder.exists():
            user_folders = list(gait_folder.glob("ID_*"))
            print(f"  {gait_type}/ ({len(user_folders)} 个用户)")
            for user_folder in sorted(user_folders):
                jpg_count = len(list(user_folder.glob("*.jpg")))
                print(f"    {user_folder.name}/ ({jpg_count} 张图片)")


def verify_dataset_structure(dataset_dir):
    """
    验证重组后的数据集结构
    
    Args:
        dataset_dir (str): 重组后的数据集目录
    """
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"错误: 数据集目录不存在 - {dataset_path}")
        return
    
    gait_types = [
        'Backpack_free', 'Backpack_line', 'Bag_free', 'Bag_line',
        'Bag_Phone_free', 'Bag_Phone_line', 'Normal_free', 'Normal_line'
    ]
    
    print(f"验证数据集结构: {dataset_path}")
    print("="*60)
    
    total_users = 0
    total_images = 0
    
    for gait_type in gait_types:
        gait_folder = dataset_path / gait_type
        if not gait_folder.exists():
            print(f"❌ 缺少步态文件夹: {gait_type}")
            continue
            
        user_folders = list(gait_folder.glob("ID_*"))
        user_count = len(user_folders)
        
        # 统计图片数量
        gait_images = 0
        for user_folder in user_folders:
            jpg_files = list(user_folder.glob("*.jpg"))
            gait_images += len(jpg_files)
        
        print(f"✅ {gait_type}: {user_count} 个用户, {gait_images} 张图片")
        total_users += user_count
        total_images += gait_images
    
    print("="*60)
    print(f"总计: 平均每种步态 {total_users/8:.1f} 个用户, 总共 {total_images} 张图片")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='重组雷达步态数据集')
    parser.add_argument('--source', '-s', type=str, required=True,
                      help='源数据集目录路径 (包含01-31用户文件夹)')
    parser.add_argument('--target', '-t', type=str, required=True,
                      help='目标数据集目录路径 (按步态类型组织)')
    parser.add_argument('--verify', '-v', action='store_true',
                      help='验证重组后的数据集结构')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset_structure(args.target)
    else:
        organize_radar_gait_dataset(args.source, args.target)
        print("\n运行验证请使用: python organize_radar_gait_dataset.py --verify -t [目标目录]")


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("雷达步态数据集重组工具")
        print("="*60)
        print("用法示例:")
        print("1. 重组数据集:")
        print("   python organize_radar_gait_dataset.py -s '/kaggle/input/gait-dataset' -t '/kaggle/working/organized_gait_dataset'")
        print("")
        print("2. 验证数据集结构:")
        print("   python organize_radar_gait_dataset.py --verify -t '/kaggle/working/organized_gait_dataset'")
        print("")
        print("数据集结构转换:")
        print("  原始结构: /kaggle/input/gait-dataset/01/Backpack_line/*.jpg")
        print("  目标结构: /kaggle/working/organized_gait_dataset/Backpack_line/ID_1/*.jpg")
        print("")
        
        # Kaggle环境默认路径
        default_source = "/kaggle/input/gait-dataset"
        default_target = "/kaggle/working/organized_gait_dataset"
        
        print(f"检测到可能是Kaggle环境，使用默认路径:")
        print(f"  源路径: {default_source}")
        print(f"  目标路径: {default_target}")
        
        use_default = input("使用默认路径? (y/n, 默认y): ").strip().lower()
        
        if use_default in ['', 'y', 'yes']:
            source_dir = default_source
            target_dir = default_target
        else:
            # 交互式模式
            source_dir = input("请输入源数据集目录路径: ").strip()
            target_dir = input("请输入目标数据集目录路径: ").strip()
        
        if source_dir and target_dir:
            organize_radar_gait_dataset(source_dir, target_dir)
        else:
            print("路径不能为空!")
    else:
        main()
