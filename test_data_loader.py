#!/usr/bin/env python3
"""
测试自定义数据加载器是否能正确工作
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path("LightningDiT/vavae").absolute()))

def test_custom_loader():
    """测试自定义数据加载器"""
    print("🧪 测试自定义数据加载器...")
    
    try:
        # 导入自定义数据加载器
        from custom_data_loader import CustomImageTrain
        
        # 创建数据集实例
        dataset = CustomImageTrain(
            data_root="/kaggle/input/dataset",
            size=256
        )
        
        print(f"✅ 数据集创建成功")
        print(f"📊 数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试加载第一个样本
            sample = dataset[0]
            print(f"✅ 样本加载成功")
            print(f"📋 样本键: {list(sample.keys())}")
            print(f"🖼️ 图像形状: {sample['image'].shape}")
            print(f"🏷️ 类别: {sample['class_name']}")
            return True
        else:
            print("❌ 数据集为空")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_directory():
    """测试数据目录结构"""
    print("\n🗂️ 测试数据目录结构...")
    
    data_root = Path("/kaggle/input/dataset")
    if not data_root.exists():
        print(f"❌ 数据根目录不存在: {data_root}")
        return False
    
    # 列出所有子目录
    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"📁 发现 {len(subdirs)} 个子目录:")
    
    total_images = 0
    for subdir in sorted(subdirs)[:5]:  # 只显示前5个
        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        total_images += len(images)
        print(f"   {subdir.name}: {len(images)} 张图像")
    
    if len(subdirs) > 5:
        print(f"   ... 还有 {len(subdirs) - 5} 个目录")
    
    print(f"📊 总计约 {total_images} 张图像（仅统计前5个目录）")
    return total_images > 0

if __name__ == "__main__":
    print("🔍 开始数据加载测试...")
    
    # 测试数据目录
    dir_ok = test_data_directory()
    
    if dir_ok:
        # 测试自定义加载器
        loader_ok = test_custom_loader()
        
        if loader_ok:
            print("\n✅ 所有测试通过！数据加载器工作正常")
        else:
            print("\n❌ 数据加载器测试失败")
    else:
        print("\n❌ 数据目录测试失败")
