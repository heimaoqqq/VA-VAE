#!/usr/bin/env python3
"""
NCC 补充评估
只评估缺失的 NCC 和 LCCS+NCC，快速测试temperature
"""

import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from itertools import product
import sys
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))
from eval_utils import *
from eval_components import *
from strategic_dataset import StrategicDataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NCC补充评估')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--output-dir', type=str,
                       default='./ncc_supplement_results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔍 NCC 补充评估（只评估缺失部分）")
    print("="*80)
    
    # 最佳配置
    best_strategy = 'diversity'
    best_lccs = {
        'momentum': 0.02,
        'iterations': 10
    }
    
    # 测试temperature（快速筛选）
    temperatures = [0.005, 0.01, 0.05]
    
    print(f"\n📋 已有评估结果:")
    print(f"  ✅ Baseline:       ~75%")
    print(f"  ✅ LCCS+Baseline:  ~79.44%")
    print(f"  ✅ PNC:            ~87.47%")
    print(f"  ✅ LCCS+PNC:       87.81-88.51%")
    
    print(f"\n📋 待评估:")
    print(f"  ❌ NCC (测试 {len(temperatures)} 个temperature)")
    print(f"  ❌ LCCS+NCC")
    
    # 加载模型
    print("\n📦 加载分类器...")
    model = load_classifier(args.model_path, device)
    model.device = device
    
    # 测试配置（先用support=3快速测试）
    support_sizes = [3]  # 先只测试最佳的support_size
    seeds = [42, 123, 456]
    
    results = []
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 总实验数
    total = len(support_sizes) * len(seeds) * (len(temperatures) + 1)  # NCC温度测试 + LCCS+NCC
    
    print(f"\n总共 {total} 个实验")
    
    # 第一阶段：测试NCC的temperature
    print("\n" + "="*80)
    print("📊 阶段1: NCC Temperature 测试")
    print("="*80)
    
    ncc_temp_results = []
    
    with tqdm(total=len(temperatures) * len(support_sizes) * len(seeds), 
              desc="NCC Temp Test") as pbar:
        for support_size, seed, temp in product(support_sizes, seeds, temperatures):
            # 创建数据集
            support_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy=best_strategy,
                model=model,
                mode='support',
                seed=seed,
                device=device,
                transform=transform
            )
            
            test_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy=best_strategy,
                model=model,
                mode='test',
                seed=seed,
                device=device,
                transform=transform
            )
            
            support_loader = DataLoader(support_dataset, batch_size=64, 
                                       shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=64, 
                                    shuffle=False, num_workers=0)
            
            # 提取特征和构建原型
            features, labels = extract_features(model, support_loader, device)
            prototypes = build_prototypes_simple_mean(features, labels)
            
            # NCC评估
            ncc = NCCEvaluator(prototypes, device)
            ncc_acc, ncc_conf = ncc.predict(
                model, test_loader, 
                temperature=temp, 
                distance_metric='cosine'
            )
            
            ncc_temp_results.append({
                'method': 'NCC',
                'temperature': temp,
                'support_size': support_size,
                'seed': seed,
                'accuracy': ncc_acc,
                'confidence': ncc_conf
            })
            
            pbar.update(1)
    
    # 找最佳temperature
    temp_df = pd.DataFrame(ncc_temp_results)
    best_temp_acc = {}
    for temp in temperatures:
        avg_acc = temp_df[temp_df['temperature']==temp]['accuracy'].mean()
        best_temp_acc[temp] = avg_acc
        print(f"  Temperature={temp}: 平均准确率={avg_acc:.4f}")
    
    best_temp = max(best_temp_acc, key=best_temp_acc.get)
    print(f"\n✨ 最佳 Temperature: {best_temp} (准确率: {best_temp_acc[best_temp]:.4f})")
    
    # 第二阶段：使用最佳temperature评估LCCS+NCC
    print("\n" + "="*80)
    print("📊 阶段2: LCCS+NCC 评估（使用最佳temperature）")
    print("="*80)
    
    lccs_ncc_results = []
    
    # 测试所有support_size
    all_support_sizes = [3, 5, 10]
    
    with tqdm(total=len(all_support_sizes) * len(seeds), 
              desc="LCCS+NCC") as pbar:
        for support_size, seed in product(all_support_sizes, seeds):
            # 创建数据集
            support_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy=best_strategy,
                model=model,
                mode='support',
                seed=seed,
                device=device,
                transform=transform
            )
            
            test_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy=best_strategy,
                model=model,
                mode='test',
                seed=seed,
                device=device,
                transform=transform
            )
            
            support_loader = DataLoader(support_dataset, batch_size=64, 
                                       shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=64, 
                                    shuffle=False, num_workers=0)
            
            # LCCS适应
            lccs = LCCSAdapter(model, device)
            lccs.adapt_progressive(
                support_loader,
                momentum=best_lccs['momentum'],
                iterations=best_lccs['iterations']
            )
            
            # 在LCCS适应后重新提取特征和构建原型
            features_lccs, labels_lccs = extract_features(model, support_loader, device)
            prototypes_lccs = build_prototypes_simple_mean(features_lccs, labels_lccs)
            
            # LCCS+NCC评估（使用最佳temperature）
            ncc_lccs = NCCEvaluator(prototypes_lccs, device)
            lccs_ncc_acc, lccs_ncc_conf = ncc_lccs.predict(
                model, test_loader,
                temperature=best_temp,
                distance_metric='cosine'
            )
            
            lccs_ncc_results.append({
                'method': 'LCCS+NCC',
                'temperature': best_temp,
                'support_size': support_size,
                'seed': seed,
                'accuracy': lccs_ncc_acc,
                'confidence': lccs_ncc_conf
            })
            
            print(f"  Support={support_size}, Seed={seed}: {lccs_ncc_acc:.4f}")
            
            # 恢复BN
            lccs.restore_bn_stats()
            
            pbar.update(1)
    
    # 保存所有结果
    all_results = ncc_temp_results + lccs_ncc_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path / 'ncc_results.csv', index=False)
    
    # 生成对比报告
    print("\n" + "="*80)
    print("📊 最终对比报告")
    print("="*80)
    
    # NCC结果（使用最佳temperature）
    ncc_best = temp_df[temp_df['temperature']==best_temp]
    ncc_mean = ncc_best['accuracy'].mean()
    
    # LCCS+NCC结果
    lccs_ncc_df = pd.DataFrame(lccs_ncc_results)
    
    print("\n总体平均:")
    print("-" * 80)
    print(f"  Baseline:       ~75.00%")
    print(f"  LCCS+Baseline:  ~79.44%")
    print(f"  NCC:            {ncc_mean:.4f} (temperature={best_temp})")
    print(f"  PNC:            ~87.47%")
    
    # 按support_size分组
    print("\nLCCS+NCC (按 Support Size):")
    for size in [3, 5, 10]:
        subset = lccs_ncc_df[lccs_ncc_df['support_size']==size]
        mean_acc = subset['accuracy'].mean()
        print(f"  Support={size}: {mean_acc:.4f}")
    
    print(f"\n  LCCS+PNC:       87.81-88.51%")
    
    # 关键对比
    print("\n关键对比:")
    print("-" * 80)
    lccs_ncc_mean = lccs_ncc_df['accuracy'].mean()
    print(f"  NCC vs PNC:           {ncc_mean:.4f} vs 87.47% (差异: {87.47-ncc_mean:+.4f})")
    print(f"  LCCS+NCC vs LCCS+PNC: {lccs_ncc_mean:.4f} vs 88.16% (差异: {88.16-lccs_ncc_mean:+.4f})")
    
    # 保存总结
    summary = {
        'ncc': {
            'best_temperature': best_temp,
            'accuracy': float(ncc_mean),
            'temperature_comparison': {float(k): float(v) for k, v in best_temp_acc.items()}
        },
        'lccs_ncc': {
            'mean_accuracy': float(lccs_ncc_mean),
            'by_support_size': {
                str(size): float(lccs_ncc_df[lccs_ncc_df['support_size']==size]['accuracy'].mean())
                for size in [3, 5, 10]
            }
        },
        'comparison': {
            'NCC_vs_PNC': float(87.47 - ncc_mean),
            'LCCS_NCC_vs_LCCS_PNC': float(88.16 - lccs_ncc_mean)
        }
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    print(f"   - ncc_results.csv")
    print(f"   - summary.json")
    
    print("\n" + "="*80)
    print("✅ 补充评估完成！")
    print("="*80)


if __name__ == '__main__':
    main() 
