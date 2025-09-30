#!/usr/bin/env python3
"""
NCC 完整超参数调优
测试所有参数组合，找到NCC的极限性能
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
    parser = argparse.ArgumentParser(description='NCC完整超参数调优')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--output-dir', type=str,
                       default='./ncc_tuning_results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🎯 NCC 完整超参数调优")
    print("="*80)
    
    # 加载模型
    print("\n📦 加载分类器...")
    model = load_classifier(args.model_path, device)
    model.device = device
    
    # 超参数搜索空间
    strategies = ['diversity']  # 已知最佳
    support_sizes = [3, 5, 10]
    seeds = [42, 123, 456]
    
    # NCC超参数
    temperatures = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    distance_metrics = ['cosine', 'euclidean']
    prototype_methods = ['simple_mean', 'weighted_mean']
    
    # LCCS参数（已知最佳）
    lccs_config = {
        'momentum': 0.02,
        'iterations': 10
    }
    
    total_configs = (len(temperatures) * len(distance_metrics) * 
                    len(prototype_methods) * len(support_sizes) * len(seeds))
    
    print(f"\n超参数搜索空间:")
    print(f"  Temperature: {temperatures}")
    print(f"  Distance: {distance_metrics}")
    print(f"  Prototype: {prototype_methods}")
    print(f"  Support sizes: {support_sizes}")
    print(f"  Seeds: {seeds}")
    print(f"\n总计: {total_configs} 个NCC配置")
    print(f"每个配置测试: NCC 和 LCCS+NCC")
    print(f"总实验数: {total_configs * 2}")
    
    results = []
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    with tqdm(total=total_configs * 2, desc="NCC Tuning") as pbar:
        for (temp, metric, proto_method, support_size, seed) in product(
            temperatures, distance_metrics, prototype_methods, 
            support_sizes, seeds
        ):
            # 创建数据集
            support_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy='diversity',
                model=model,
                mode='support',
                seed=seed,
                device=device,
                transform=transform
            )
            
            test_dataset = StrategicDataset(
                data_dir=args.data_dir,
                support_size=support_size,
                strategy='diversity',
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
            
            # 构建原型（根据方法）
            if proto_method == 'simple_mean':
                features, labels = extract_features(model, support_loader, device)
                prototypes = build_prototypes_simple_mean(features, labels)
            else:  # weighted_mean
                prototypes = build_prototypes_weighted(model, support_loader, device)
            
            # ========== 1. NCC（无LCCS）==========
            ncc = NCCEvaluator(prototypes, device)
            ncc_acc, ncc_conf = ncc.predict(
                model, test_loader,
                temperature=temp,
                distance_metric=metric
            )
            
            results.append({
                'method': 'NCC',
                'temperature': temp,
                'distance_metric': metric,
                'prototype_method': proto_method,
                'support_size': support_size,
                'seed': seed,
                'accuracy': ncc_acc,
                'confidence': ncc_conf,
                'lccs_applied': False
            })
            
            pbar.update(1)
            
            # ========== 2. LCCS+NCC ==========
            lccs = LCCSAdapter(model, device)
            lccs.adapt_progressive(
                support_loader,
                momentum=lccs_config['momentum'],
                iterations=lccs_config['iterations']
            )
            
            # 重新构建原型（在LCCS适应后）
            if proto_method == 'simple_mean':
                features_lccs, labels_lccs = extract_features(model, support_loader, device)
                prototypes_lccs = build_prototypes_simple_mean(features_lccs, labels_lccs)
            else:
                prototypes_lccs = build_prototypes_weighted(model, support_loader, device)
            
            ncc_lccs = NCCEvaluator(prototypes_lccs, device)
            lccs_ncc_acc, lccs_ncc_conf = ncc_lccs.predict(
                model, test_loader,
                temperature=temp,
                distance_metric=metric
            )
            
            results.append({
                'method': 'LCCS+NCC',
                'temperature': temp,
                'distance_metric': metric,
                'prototype_method': proto_method,
                'support_size': support_size,
                'seed': seed,
                'accuracy': lccs_ncc_acc,
                'confidence': lccs_ncc_conf,
                'lccs_applied': True
            })
            
            # 恢复BN
            lccs.restore_bn_stats()
            
            pbar.update(1)
    
    # 保存详细结果
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'ncc_tuning_detailed.csv', index=False)
    
    # ==================== 分析结果 ====================
    print("\n" + "="*80)
    print("📊 超参数调优结果分析")
    print("="*80)
    
    # 1. NCC最佳配置
    print("\n【NCC】最佳配置:")
    print("-" * 80)
    ncc_df = df[df['method'] == 'NCC']
    best_ncc = ncc_df.loc[ncc_df['accuracy'].idxmax()]
    print(f"  ✨ 最高准确率: {best_ncc['accuracy']:.4f}")
    print(f"  ✨ 平均置信度: {best_ncc['confidence']:.4f}")
    print(f"  Temperature: {best_ncc['temperature']}")
    print(f"  Distance metric: {best_ncc['distance_metric']}")
    print(f"  Prototype method: {best_ncc['prototype_method']}")
    print(f"  Support size: {best_ncc['support_size']}")
    print(f"  Seed: {best_ncc['seed']}")
    
    # NCC平均性能（按配置分组）
    print("\n【NCC】按超参数平均性能:")
    print("-" * 80)
    
    # 按temperature
    print("\n  Temperature:")
    for temp in temperatures:
        subset = ncc_df[ncc_df['temperature'] == temp]
        avg_acc = subset['accuracy'].mean()
        avg_conf = subset['confidence'].mean()
        print(f"    {temp:6.3f}: Acc={avg_acc:.4f}, Conf={avg_conf:.4f}")
    
    # 按distance_metric
    print("\n  Distance Metric:")
    for metric in distance_metrics:
        subset = ncc_df[ncc_df['distance_metric'] == metric]
        avg_acc = subset['accuracy'].mean()
        avg_conf = subset['confidence'].mean()
        print(f"    {metric:10s}: Acc={avg_acc:.4f}, Conf={avg_conf:.4f}")
    
    # 按prototype_method
    print("\n  Prototype Method:")
    for method in prototype_methods:
        subset = ncc_df[ncc_df['prototype_method'] == method]
        avg_acc = subset['accuracy'].mean()
        avg_conf = subset['confidence'].mean()
        print(f"    {method:15s}: Acc={avg_acc:.4f}, Conf={avg_conf:.4f}")
    
    # 2. LCCS+NCC最佳配置
    print("\n" + "="*80)
    print("【LCCS+NCC】最佳配置:")
    print("-" * 80)
    lccs_ncc_df = df[df['method'] == 'LCCS+NCC']
    best_lccs_ncc = lccs_ncc_df.loc[lccs_ncc_df['accuracy'].idxmax()]
    print(f"  ✨ 最高准确率: {best_lccs_ncc['accuracy']:.4f}")
    print(f"  ✨ 平均置信度: {best_lccs_ncc['confidence']:.4f}")
    print(f"  Temperature: {best_lccs_ncc['temperature']}")
    print(f"  Distance metric: {best_lccs_ncc['distance_metric']}")
    print(f"  Prototype method: {best_lccs_ncc['prototype_method']}")
    print(f"  Support size: {best_lccs_ncc['support_size']}")
    print(f"  Seed: {best_lccs_ncc['seed']}")
    
    # 按support_size统计
    print("\n【LCCS+NCC】按 Support Size:")
    print("-" * 80)
    for size in [3, 5, 10]:
        subset = lccs_ncc_df[lccs_ncc_df['support_size'] == size]
        avg_acc = subset['accuracy'].mean()
        std_acc = subset['accuracy'].std()
        max_acc = subset['accuracy'].max()
        avg_conf = subset['confidence'].mean()
        print(f"  Support={size:2d}: Acc={avg_acc:.4f}±{std_acc:.4f} "
              f"(最大={max_acc:.4f}), Conf={avg_conf:.4f}")
    
    # 3. 总体对比
    print("\n" + "="*80)
    print("📈 总体性能对比")
    print("="*80)
    
    ncc_mean = ncc_df['accuracy'].mean()
    ncc_max = ncc_df['accuracy'].max()
    ncc_conf_mean = ncc_df['confidence'].mean()
    
    lccs_ncc_mean = lccs_ncc_df['accuracy'].mean()
    lccs_ncc_max = lccs_ncc_df['accuracy'].max()
    lccs_ncc_conf_mean = lccs_ncc_df['confidence'].mean()
    
    print(f"\nNCC:")
    print(f"  平均准确率: {ncc_mean:.4f}")
    print(f"  最大准确率: {ncc_max:.4f}")
    print(f"  平均置信度: {ncc_conf_mean:.4f}")
    
    print(f"\nLCCS+NCC:")
    print(f"  平均准确率: {lccs_ncc_mean:.4f}")
    print(f"  最大准确率: {lccs_ncc_max:.4f}")
    print(f"  平均置信度: {lccs_ncc_conf_mean:.4f}")
    
    print(f"\nLCCS 带来的提升:")
    print(f"  平均: +{lccs_ncc_mean - ncc_mean:.4f}")
    print(f"  最大: +{lccs_ncc_max - ncc_max:.4f}")
    print(f"  置信度: +{lccs_ncc_conf_mean - ncc_conf_mean:.4f}")
    
    # 保存最佳配置
    summary = {
        'best_ncc': {
            'accuracy': float(best_ncc['accuracy']),
            'confidence': float(best_ncc['confidence']),
            'config': {
                'temperature': float(best_ncc['temperature']),
                'distance_metric': str(best_ncc['distance_metric']),
                'prototype_method': str(best_ncc['prototype_method']),
                'support_size': int(best_ncc['support_size']),
                'seed': int(best_ncc['seed'])
            }
        },
        'best_lccs_ncc': {
            'accuracy': float(best_lccs_ncc['accuracy']),
            'confidence': float(best_lccs_ncc['confidence']),
            'config': {
                'temperature': float(best_lccs_ncc['temperature']),
                'distance_metric': str(best_lccs_ncc['distance_metric']),
                'prototype_method': str(best_lccs_ncc['prototype_method']),
                'support_size': int(best_lccs_ncc['support_size']),
                'seed': int(best_lccs_ncc['seed']),
                'lccs': lccs_config
            }
        },
        'statistics': {
            'ncc': {
                'mean_accuracy': float(ncc_mean),
                'max_accuracy': float(ncc_max),
                'mean_confidence': float(ncc_conf_mean)
            },
            'lccs_ncc': {
                'mean_accuracy': float(lccs_ncc_mean),
                'max_accuracy': float(lccs_ncc_max),
                'mean_confidence': float(lccs_ncc_conf_mean)
            },
            'lccs_improvement': {
                'accuracy': float(lccs_ncc_mean - ncc_mean),
                'confidence': float(lccs_ncc_conf_mean - ncc_conf_mean)
            }
        }
    }
    
    with open(output_path / 'best_configs.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    print(f"   - ncc_tuning_detailed.csv    # 详细结果")
    print(f"   - best_configs.json          # 最佳配置")
    
    print("\n" + "="*80)
    print("🎉 NCC超参数调优完成！")
    print("="*80)
    
    # 最终推荐
    print(f"\n🏆 最终推荐配置:")
    print(f"  方法: LCCS+NCC")
    print(f"  准确率: {best_lccs_ncc['accuracy']:.4f}")
    print(f"  置信度: {best_lccs_ncc['confidence']:.4f}")
    print(f"  Temperature: {best_lccs_ncc['temperature']}")
    print(f"  Distance: {best_lccs_ncc['distance_metric']}")
    print(f"  Prototype: {best_lccs_ncc['prototype_method']}")
    print(f"  Support size: {best_lccs_ncc['support_size']}")


if __name__ == '__main__':
    main() 
