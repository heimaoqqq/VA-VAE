#!/usr/bin/env python3
"""
快速评估脚本
直接使用已找到的最佳配置运行 PNC+LCCS 组合评估
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
    parser = argparse.ArgumentParser(description='使用最佳配置快速评估')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--output-dir', type=str,
                       default='./best_config_results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 使用最佳配置快速评估 PNC+LCCS")
    print("="*80)
    print(f"💻 设备: {device}")
    print(f"📂 输出目录: {output_path}")
    
    # 最佳配置（从之前的结果中获得）
    best_strategy = 'diversity'
    best_pnc = {
        'prototype_strategy': 'simple_mean',
        'fusion_alpha': 0.7,
        'similarity_tau': 0.005,
        'use_adaptive': False
    }
    best_lccs = {
        'lccs_method': 'progressive',
        'momentum': 0.02,
        'iterations': 10
    }
    
    print(f"\n📋 使用配置:")
    print(f"   样本选择策略: {best_strategy}")
    print(f"   PNC 原型策略: {best_pnc['prototype_strategy']}")
    print(f"   PNC Fusion α: {best_pnc['fusion_alpha']}")
    print(f"   PNC Temperature τ: {best_pnc['similarity_tau']}")
    print(f"   LCCS 方法: {best_lccs['lccs_method']}")
    print(f"   LCCS Momentum: {best_lccs['momentum']}")
    print(f"   LCCS Iterations: {best_lccs['iterations']}")
    
    # 1. 加载模型
    print("\n" + "="*80)
    print("📦 加载分类器")
    print("="*80)
    model = load_classifier(args.model_path, device)
    model.device = device
    print("✅ 模型加载完成")
    
    # 2. PNC+LCCS 组合评估
    print("\n" + "="*80)
    print("🎯 PNC+LCCS 组合评估")
    print("="*80)
    
    results = []
    
    support_sizes = [3, 5, 10]
    seeds = [42, 123, 456]
    
    total = len(support_sizes) * len(seeds)
    
    print(f"总共 {total} 个实验")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    with tqdm(total=total, desc="PNC+LCCS") as pbar:
        for support_size, seed in product(support_sizes, seeds):
            print(f"\n📊 Support={support_size}, Seed={seed}")
            
            # 创建数据集（使用最佳策略）
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
            
            support_loader = DataLoader(
                support_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0
            )
            
            # 创建 LCCS 适配器
            lccs = LCCSAdapter(model, device)
            
            # 步骤1：LCCS 适应
            lccs.adapt_progressive(
                support_loader,
                momentum=best_lccs['momentum'],
                iterations=best_lccs['iterations']
            )
            
            # 步骤2：构建原型（在 LCCS 适应后的模型上）
            if best_pnc['prototype_strategy'] == 'simple_mean':
                features, labels = extract_features(model, support_loader, device)
                prototypes = build_prototypes_simple_mean(features, labels)
            else:
                prototypes = build_prototypes_weighted(model, support_loader, device)
            
            # 步骤3：PNC 评估
            pnc = PNCEvaluator(model, prototypes, device)
            acc, conf = pnc.predict(
                test_loader,
                fusion_alpha=best_pnc['fusion_alpha'],
                similarity_tau=best_pnc['similarity_tau'],
                use_adaptive=best_pnc['use_adaptive']
            )
            
            # 保存结果
            results.append({
                'support_size': support_size,
                'seed': seed,
                'strategy': best_strategy,
                'accuracy': acc,
                'confidence': conf
            })
            
            print(f"  准确率: {acc:.4f}, 置信度: {conf:.4f}")
            
            # 恢复 BN 统计量
            lccs.restore_bn_stats()
            
            pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'pnc_lccs_combined_results.csv', index=False)
    
    # 生成总结
    print("\n" + "="*80)
    print("📊 最终结果总结")
    print("="*80)
    
    # 按 support_size 分组统计
    print("\n按 Support Size 分组:")
    for size in [3, 5, 10]:
        subset = df[df['support_size'] == size]
        print(f"  Support={size}: "
              f"准确率={subset['accuracy'].mean():.4f} ± {subset['accuracy'].std():.4f}, "
              f"最大={subset['accuracy'].max():.4f}")
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"  平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  最大准确率: {df['accuracy'].max():.4f}")
    print(f"  最小准确率: {df['accuracy'].min():.4f}")
    
    # 找最佳配置
    best_row = df.loc[df['accuracy'].idxmax()]
    print(f"\n🏆 最佳配置:")
    print(f"  Support Size: {best_row['support_size']}")
    print(f"  Seed: {best_row['seed']}")
    print(f"  准确率: {best_row['accuracy']:.4f}")
    
    # 保存总结
    summary = {
        'best_config': {
            'strategy': best_strategy,
            'pnc': best_pnc,
            'lccs': best_lccs
        },
        'results': {
            'mean_accuracy': float(df['accuracy'].mean()),
            'std_accuracy': float(df['accuracy'].std()),
            'max_accuracy': float(df['accuracy'].max()),
            'min_accuracy': float(df['accuracy'].min())
        },
        'by_support_size': {
            str(size): {
                'mean': float(df[df['support_size']==size]['accuracy'].mean()),
                'std': float(df[df['support_size']==size]['accuracy'].std()),
                'max': float(df[df['support_size']==size]['accuracy'].max())
            }
            for size in [3, 5, 10]
        }
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    print(f"   - pnc_lccs_combined_results.csv")
    print(f"   - summary.json")
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == '__main__':
    main() 
