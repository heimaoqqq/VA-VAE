#!/usr/bin/env python3
"""
检查数据结构脚本
验证微多普勒数据的目录结构和文件
"""

import os
from pathlib import Path

def check_data_structure(data_dir):
    """检查数据目录结构"""
    print(f"🔍 检查数据目录: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        return False
    
    print(f"✅ 数据目录存在: {data_path}")
    
    # 列出所有子目录
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"📁 找到 {len(subdirs)} 个子目录:")
    
    for subdir in sorted(subdirs):
        print(f"  📂 {subdir.name}")
        
        # 检查是否是用户目录
        if subdir.name.startswith('user'):
            # 统计图像文件
            image_files = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg'))
            print(f"    🖼️  {len(image_files)} 个图像文件")
            
            # 显示前几个文件名
            for i, img_file in enumerate(image_files[:3]):
                print(f"      - {img_file.name}")
            if len(image_files) > 3:
                print(f"      - ... 还有 {len(image_files) - 3} 个文件")
        else:
            print(f"    ⚠️  不是用户目录 (不以'user'开头)")
    
    # 统计总的图像数量
    total_images = 0
    user_dirs = 0
    
    for subdir in subdirs:
        if subdir.name.startswith('user'):
            user_dirs += 1
            image_files = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg'))
            total_images += len(image_files)
    
    print(f"\n📊 统计信息:")
    print(f"  用户目录数: {user_dirs}")
    print(f"  总图像数: {total_images}")
    
    if total_images == 0:
        print("❌ 没有找到任何图像文件!")
        print("请检查:")
        print("1. 数据目录路径是否正确")
        print("2. 子目录是否以'user'开头")
        print("3. 图像文件是否为.png/.jpg/.jpeg格式")
        return False
    
    return True

def suggest_data_structure():
    """建议的数据结构"""
    print("\n💡 建议的数据结构:")
    print("""
/kaggle/working/data_split/
├── train/
│   ├── user1/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── user2/
│   │   ├── image1.png
│   │   └── ...
│   └── ...
└── val/
    ├── user1/
    │   ├── image1.png
    │   └── ...
    └── ...
    """)

def main():
    """主函数"""
    print("🎯 微多普勒数据结构检查")
    print("=" * 50)
    
    # 检查常见的数据路径
    possible_paths = [
        "/kaggle/working/data_split/train",
        "/kaggle/working/data_split/val",
        "/kaggle/input/dataset/train",
        "/kaggle/input/dataset/val",
        "/kaggle/working/train",
        "/kaggle/working/val"
    ]
    
    found_data = False
    
    for path in possible_paths:
        print(f"\n检查路径: {path}")
        if check_data_structure(path):
            found_data = True
        print("-" * 30)
    
    if not found_data:
        print("\n❌ 没有找到有效的数据!")
        suggest_data_structure()
        
        print("\n🔧 可能的解决方案:")
        print("1. 运行数据分割脚本:")
        print("   python data_split.py --input_dir /kaggle/input/dataset --output_dir /kaggle/working/data_split")
        print("\n2. 检查数据是否已上传到Kaggle")
        print("\n3. 确认目录结构符合要求")
    else:
        print("\n✅ 找到有效的数据结构!")

if __name__ == "__main__":
    main()
