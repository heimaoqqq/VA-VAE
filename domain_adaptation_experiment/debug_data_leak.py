#!/usr/bin/env python3
"""
检查数据泄漏和异常情况
"""

import torch
from pathlib import Path
from cross_domain_evaluator import BackpackWalkingDataset
from build_improved_prototypes import TargetDomainDataset
import torchvision.transforms as transforms

def check_data_leak():
    """检查原型构建数据与测试数据是否有重叠"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 构建支持集（原型用）
    support_dataset = TargetDomainDataset(
        data_dir='/kaggle/input/backpack/backpack',
        transform=transform,
        support_size=10,
        seed=42
    )
    
    # 完整测试集
    test_dataset = BackpackWalkingDataset(
        data_dir='/kaggle/input/backpack/backpack',
        transform=transform
    )
    
    # 获取文件路径
    support_paths = set()
    for sample in support_dataset.samples:
        support_paths.add(str(sample['path']))
    
    test_paths = set()
    # BackpackWalkingDataset的数据格式不同，直接访问内部属性
    if hasattr(test_dataset, 'samples') and len(test_dataset.samples) > 0:
        for sample in test_dataset.samples:
            if isinstance(sample, dict):
                test_paths.add(str(sample['path']))
            else:
                # 如果是其他格式，尝试通过数据加载器获取
                print(f"⚠️ BackpackWalkingDataset格式不同，尝试手动获取路径...")
                break
    
    # 如果上面的方法不行，手动扫描目录
    if len(test_paths) == 0:
        data_dir = Path('/kaggle/input/backpack/backpack')
        for user_dir in data_dir.iterdir():
            if user_dir.is_dir():
                for img_file in user_dir.glob('*.png'):
                    test_paths.add(str(img_file))
                for img_file in user_dir.glob('*.jpg'):
                    test_paths.add(str(img_file))
    
    # 检查重叠
    overlap = support_paths & test_paths
    
    print(f"📊 数据集分析:")
    print(f"   支持集样本: {len(support_paths)}")
    print(f"   测试集样本: {len(test_paths)}")
    print(f"   重叠样本: {len(overlap)}")
    print(f"   重叠比例: {len(overlap)/len(support_paths)*100:.1f}%")
    
    if len(overlap) > 0:
        print(f"⚠️ 发现数据泄漏！")
        print(f"   前5个重叠文件:")
        for i, path in enumerate(list(overlap)[:5]):
            print(f"   {i+1}. {path}")
        return True
    else:
        print(f"✅ 无数据泄漏")
        return False

def analyze_baseline_performance():
    """分析基线性能是否正常"""
    
    print(f"\n📈 基线性能分析:")
    print(f"   当前基线: 75.5%")
    print(f"   预期基线（相似域）: >80%")
    
    if 75.5 < 80:
        print(f"⚠️ 基线偏低，可能原因:")
        print(f"   1. 域差异被低估")
        print(f"   2. 模型在目标域泛化能力弱")
        print(f"   3. 数据质量问题")

def check_extreme_improvement():
    """检查极端提升是否合理"""
    
    improvement = 11.37
    print(f"\n🎯 提升幅度分析:")
    print(f"   当前提升: +{improvement}%")
    print(f"   典型PNC提升: 2-5%")
    print(f"   异常阈值: >8%")
    
    if improvement > 8:
        print(f"🚨 提升异常大！可能原因:")
        print(f"   1. 数据泄漏")
        print(f"   2. 过拟合到支持集")
        print(f"   3. 评估方法错误")
        print(f"   4. τ=0.01过于极端")

if __name__ == '__main__':
    print("🔍 PNC异常结果诊断")
    print("="*50)
    
    # 检查数据泄漏
    has_leak = check_data_leak()
    
    # 分析基线性能
    analyze_baseline_performance()
    
    # 检查提升幅度
    check_extreme_improvement()
    
    print(f"\n🏁 诊断结论:")
    if has_leak:
        print(f"❌ 存在数据泄漏，结果不可信")
    else:
        print(f"✅ 无明显数据泄漏")
        print(f"⚠️ 但提升幅度仍然异常，需要进一步调查")
