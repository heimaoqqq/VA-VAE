"""
Kaggle雷达步态数据集重组工具
专门为Kaggle环境优化的版本
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_kaggle_gait_dataset():
    """
    在Kaggle环境中重组雷达步态数据集
    源路径: /kaggle/input/gait-dataset
    目标路径: /kaggle/working/organized_gait_dataset
    """
    
    # Kaggle环境路径
    source_path = Path("/kaggle/input/gait-dataset")
    target_path = Path("/kaggle/working/organized_gait_dataset")
    
    print(f"源数据集路径: {source_path}")
    print(f"目标数据集路径: {target_path}")
    
    # 检查源路径是否存在
    if not source_path.exists():
        print(f"❌ 错误: 源数据集路径不存在 - {source_path}")
        print("请确认数据集已正确上传到Kaggle")
        return False
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建目标目录: {target_path}")
    
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
        print(f"📁 创建步态文件夹: {gait_type}")
    
    # 统计信息
    total_files = 0
    copied_files = 0
    error_files = []
    
    print("\n开始重组数据集...")
    print("="*60)
    
    # 遍历31个用户文件夹
    for user_id in range(1, 32):  # 01到31
        user_folder_name = f"{user_id:02d}"  # 格式化为两位数字，如01, 02, ...
        user_source_path = source_path / user_folder_name
        
        if not user_source_path.exists():
            print(f"⚠️  警告: 用户文件夹不存在 - {user_folder_name}")
            continue
            
        print(f"👤 处理用户: {user_folder_name} -> ID_{user_id}")
        
        # 遍历该用户的8种步态
        for gait_type in gait_types:
            gait_source_path = user_source_path / gait_type
            
            if not gait_source_path.exists():
                print(f"  ⚠️  步态文件夹不存在: {gait_type}")
                continue
            
            # 获取该步态文件夹下的所有jpg文件
            jpg_files = list(gait_source_path.glob("*.jpg"))
            total_files += len(jpg_files)
            
            if not jpg_files:
                print(f"  ⚠️  步态文件夹为空: {gait_type}")
                continue
            
            # 复制文件到目标位置
            target_gait_folder = target_path / gait_type
            target_user_folder = target_gait_folder / f"ID_{user_id}"
            target_user_folder.mkdir(exist_ok=True)
            
            # 使用tqdm显示进度
            for jpg_file in tqdm(jpg_files, 
                               desc=f"  📸 {gait_type}/ID_{user_id}", 
                               leave=False,
                               ncols=80):
                try:
                    target_file = target_user_folder / jpg_file.name
                    shutil.copy2(jpg_file, target_file)
                    copied_files += 1
                except Exception as e:
                    error_msg = f"复制失败: {jpg_file.name} -> {target_file}, 错误: {str(e)}"
                    error_files.append(error_msg)
                    print(f"    ❌ {error_msg}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("🎉 数据集重组完成!")
    print(f"📊 总文件数: {total_files}")
    print(f"✅ 成功复制: {copied_files}")
    print(f"❌ 失败文件: {len(error_files)}")
    
    if error_files:
        print("\n失败文件列表:")
        for error in error_files[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个错误")
    
    # 验证重组后的结构
    print(f"\n📁 重组后的目录结构:")
    for gait_type in gait_types:
        gait_folder = target_path / gait_type
        if gait_folder.exists():
            user_folders = list(gait_folder.glob("ID_*"))
            user_count = len(user_folders)
            
            # 统计该步态类型的总图片数
            total_images = sum(len(list(user_folder.glob("*.jpg"))) 
                             for user_folder in user_folders)
            
            print(f"  📂 {gait_type}: {user_count} 个用户, {total_images} 张图片")
    
    print(f"\n✅ 重组完成! 数据保存在: {target_path}")
    return True


def check_dataset_structure():
    """检查源数据集结构"""
    
    source_path = Path("/kaggle/input/gait-dataset")
    
    if not source_path.exists():
        print(f"❌ 数据集路径不存在: {source_path}")
        return False
    
    print(f"🔍 检查数据集结构: {source_path}")
    print("="*60)
    
    # 检查用户文件夹
    user_folders = []
    for user_id in range(1, 32):
        user_folder_name = f"{user_id:02d}"
        user_path = source_path / user_folder_name
        if user_path.exists():
            user_folders.append(user_folder_name)
    
    print(f"👥 找到用户文件夹: {len(user_folders)}/31")
    if len(user_folders) < 31:
        missing = [f"{i:02d}" for i in range(1, 32) 
                  if f"{i:02d}" not in user_folders]
        print(f"⚠️  缺少用户文件夹: {missing}")
    
    # 检查步态类型
    gait_types = [
        'Backpack_free', 'Backpack_line', 'Bag_free', 'Bag_line',
        'Bag_Phone_free', 'Bag_Phone_line', 'Normal_free', 'Normal_line'
    ]
    
    print(f"\n🚶 检查步态类型分布:")
    for gait_type in gait_types:
        count = 0
        total_images = 0
        for user_folder in user_folders[:5]:  # 检查前5个用户作为样本
            gait_path = source_path / user_folder / gait_type
            if gait_path.exists():
                count += 1
                images = list(gait_path.glob("*.jpg"))
                total_images += len(images)
        
        avg_images = total_images / max(count, 1)
        print(f"  📊 {gait_type}: {count}/5 用户有此步态, 平均 {avg_images:.1f} 张图片")
    
    return True


if __name__ == "__main__":
    print("🚀 Kaggle雷达步态数据集重组工具")
    print("="*60)
    
    # 首先检查数据集结构
    if check_dataset_structure():
        print("\n" + "="*60)
        
        # 询问是否继续重组
        proceed = input("数据集检查完成，是否开始重组? (y/n, 默认y): ").strip().lower()
        
        if proceed in ['', 'y', 'yes']:
            success = organize_kaggle_gait_dataset()
            
            if success:
                print("\n🎯 下一步建议:")
                print("1. 检查 /kaggle/working/organized_gait_dataset 目录")
                print("2. 使用重组后的数据进行训练")
                print("3. 如需要可以将数据打包下载")
        else:
            print("👋 重组已取消")
    else:
        print("❌ 数据集检查失败，请检查数据集路径和结构")
