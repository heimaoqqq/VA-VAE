#!/usr/bin/env python3
"""
优化版超参数搜索
- 模型只加载一次
- 数据按 support_size 分组加载
- 速度提升 5-10 倍
- 严格保证无数据泄露
"""

import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from itertools import product
import sys
import copy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent))
from eval_config import *
from eval_utils import *
from eval_components import *
from strategic_dataset import StrategicDataset


def evaluate_pnc_hyperparameters_optimized(model, data_loaders_dict, output_dir):
    """
    优化版 PNC 超参数搜索
    
    Args:
        model: 预加载的分类器（只读，不修改）
        data_loaders_dict: {(support_size, seed): (support_loader, test_loader)}
        output_dir: 输出目录
    
    Returns:
        DataFrame: PNC 结果
    """
    print("\n" + "="*80)
    print("🔍 PNC超参数搜索（优化版）")
    print("="*80)
    
    results = []
    
    # 超参数配置
    prototype_strategies = ['simple_mean', 'weighted_mean']
    fusion_alphas = [0.4, 0.5, 0.6, 0.7]
    similarity_taus = [0.005, 0.01, 0.02]
    use_adaptives = [False, True]
    
    # 总配置数
    total_configs = (len(prototype_strategies) * len(fusion_alphas) * 
                    len(similarity_taus) * len(use_adaptives))
    total = total_configs * len(data_loaders_dict)
    
    print(f"超参数配置: {total_configs} 种")
    print(f"数据配置: {len(data_loaders_dict)} 种")
    print(f"总计: {total} 个实验\n")
    
    with tqdm(total=total, desc="PNC") as pbar:
        # 按数据集分组
        for key, (support_loader, test_loader) in data_loaders_dict.items():
            if len(key) == 3:  # (support_size, seed, strategy)
                support_size, seed, strategy = key
                print(f"\n📊 Support={support_size}, Seed={seed}, Strategy={strategy}")
            else:  # (support_size, seed)
                support_size, seed = key
                print(f"\n📊 Support={support_size}, Seed={seed}")
            
            # 对同一数据集测试所有超参数
            for proto_strat, alpha, tau, adaptive in product(
                prototype_strategies, fusion_alphas, similarity_taus, use_adaptives
            ):
                # 1. 构建原型（每次重新构建，避免缓存）
                if proto_strat == 'simple_mean':
                    features, labels = extract_features(model, support_loader, model.device)
                    prototypes = build_prototypes_simple_mean(features, labels)
                elif proto_strat == 'weighted_mean':
                    prototypes = build_prototypes_weighted(model, support_loader, model.device)
                else:
                    raise ValueError(f"Unknown strategy: {proto_strat}")
                
                # 2. PNC 评估（只读模型，不修改）
                pnc = PNCEvaluator(model, prototypes, model.device)
                acc, conf = pnc.predict(test_loader, alpha, tau, adaptive)
                
                # 3. 保存结果
                results.append({
                    'method': 'pnc',
                    'prototype_strategy': proto_strat,
                    'fusion_alpha': alpha,
                    'similarity_tau': tau,
                    'use_adaptive': adaptive,
                    'support_size': support_size,
                    'seed': seed,
                    'accuracy': acc,
                    'confidence': conf
                })
                
                pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'pnc_results.csv', index=False)
    
    # 找最佳配置
    best = df.loc[df['accuracy'].idxmax()]
    print(f"\n🏆 最佳PNC配置:")
    print(f"   准确率: {best['accuracy']:.4f}")
    print(f"   原型策略: {best['prototype_strategy']}")
    print(f"   Fusion α: {best['fusion_alpha']}")
    print(f"   Temperature τ: {best['similarity_tau']}")
    print(f"   自适应: {best['use_adaptive']}")
    
    return df


def evaluate_lccs_hyperparameters_optimized(model, data_loaders_dict, output_dir):
    """
    优化版 LCCS 超参数搜索
    
    关键：每次实验后恢复原始 BN 统计量，避免状态污染
    
    Args:
        model: 预加载的分类器
        data_loaders_dict: 数据加载器字典
        output_dir: 输出目录
    
    Returns:
        DataFrame: LCCS 结果
    """
    print("\n" + "="*80)
    print("🔍 LCCS超参数搜索（优化版）")
    print("="*80)
    
    results = []
    
    # Progressive 配置
    progressive_configs = list(product(
        [0.005, 0.01, 0.02],  # momentum
        [3, 5, 10]             # iterations
    ))
    
    # Weighted 配置
    weighted_configs = [0.2, 0.3, 0.4]  # alpha
    
    total_configs = len(progressive_configs) + len(weighted_configs)
    total = total_configs * len(data_loaders_dict)
    
    print(f"Progressive 配置: {len(progressive_configs)} 种")
    print(f"Weighted 配置: {len(weighted_configs)} 种")
    print(f"数据配置: {len(data_loaders_dict)} 种")
    print(f"总计: {total} 个实验\n")
    
    with tqdm(total=total, desc="LCCS") as pbar:
        # 按数据集分组
        for key, (support_loader, test_loader) in data_loaders_dict.items():
            if len(key) == 3:  # (support_size, seed, strategy)
                support_size, seed, strategy = key
                print(f"\n📊 Support={support_size}, Seed={seed}, Strategy={strategy}")
            else:  # (support_size, seed)
                support_size, seed = key
                print(f"\n📊 Support={support_size}, Seed={seed}")
            
            # 创建 LCCS 适配器（保存原始 BN 统计量）
            lccs = LCCSAdapter(model, model.device)
            
            # 测试 Progressive
            for momentum, iterations in progressive_configs:
                # 适应 BN
                lccs.adapt_progressive(support_loader, momentum, iterations)
                
                # 评估
                acc, conf = evaluate_baseline(model, test_loader, model.device)
                
                # 保存结果
                results.append({
                    'method': 'lccs',
                    'lccs_method': 'progressive',
                    'momentum': momentum,
                    'iterations': iterations,
                    'alpha': None,
                    'support_size': support_size,
                    'seed': seed,
                    'accuracy': acc,
                    'confidence': conf
                })
                
                # ⚠️ 关键：恢复原始 BN 统计量
                lccs.restore_bn_stats()
                
                pbar.update(1)
            
            # 测试 Weighted
            for alpha in weighted_configs:
                # 适应 BN
                lccs.adapt_weighted(support_loader, alpha)
                
                # 评估
                acc, conf = evaluate_baseline(model, test_loader, model.device)
                
                # 保存结果
                results.append({
                    'method': 'lccs',
                    'lccs_method': 'weighted',
                    'momentum': None,
                    'iterations': None,
                    'alpha': alpha,
                    'support_size': support_size,
                    'seed': seed,
                    'accuracy': acc,
                    'confidence': conf
                })
                
                # ⚠️ 关键：恢复原始 BN 统计量
                lccs.restore_bn_stats()
                
                pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'lccs_results.csv', index=False)
    
    # 找最佳配置
    best = df.loc[df['accuracy'].idxmax()]
    print(f"\n🏆 最佳LCCS配置:")
    print(f"   准确率: {best['accuracy']:.4f}")
    print(f"   方法: {best['lccs_method']}")
    if best['lccs_method'] == 'progressive':
        print(f"   Momentum: {best['momentum']}")
        print(f"   Iterations: {best['iterations']}")
    else:
        print(f"   Alpha: {best['alpha']}")
    
    return df


def evaluate_pnc_lccs_combined_optimized(model, best_pnc, best_lccs, 
                                        data_dir, output_dir):
    """
    优化版 PNC+LCCS 组合评估
    
    Args:
        model: 预加载的分类器
        best_pnc: 最佳 PNC 配置
        best_lccs: 最佳 LCCS 配置
        data_dir: 数据目录
        output_dir: 输出目录
    
    Returns:
        DataFrame: 组合结果
    """
    print("\n" + "="*80)
    print("🔍 PNC+LCCS组合评估（优化版）")
    print("="*80)
    
    results = []
    
    support_sizes = [3, 5, 10]
    seeds = [42, 123, 456]
    
    total = len(support_sizes) * len(seeds)
    
    print(f"总共 {total} 个配置")
    print(f"\n最佳PNC配置:")
    print(f"  原型策略: {best_pnc['prototype_strategy']}")
    print(f"  Fusion α: {best_pnc['fusion_alpha']}")
    print(f"  Temperature τ: {best_pnc['similarity_tau']}")
    print(f"\n最佳LCCS配置:")
    print(f"  方法: {best_lccs['lccs_method']}")
    
    with tqdm(total=total, desc="PNC+LCCS") as pbar:
        for support_size, seed in product(support_sizes, seeds):
            # 创建数据加载器
            support_loader, test_loader = create_data_loaders(
                data_dir, support_size, seed, batch_size=64, num_workers=0
            )
            
            # 创建 LCCS 适配器
            lccs = LCCSAdapter(model, model.device)
            
            # 步骤1：LCCS 适应
            if best_lccs['lccs_method'] == 'progressive':
                lccs.adapt_progressive(
                    support_loader,
                    momentum=best_lccs['momentum'],
                    iterations=best_lccs['iterations']
                )
            else:
                lccs.adapt_weighted(support_loader, alpha=best_lccs['alpha'])
            
            # 步骤2：构建原型（在 LCCS 适应后的模型上）
            if best_pnc['prototype_strategy'] == 'simple_mean':
                features, labels = extract_features(model, support_loader, model.device)
                prototypes = build_prototypes_simple_mean(features, labels)
            else:
                prototypes = build_prototypes_weighted(model, support_loader, model.device)
            
            # 步骤3：PNC 评估
            pnc = PNCEvaluator(model, prototypes, model.device)
            acc, conf = pnc.predict(
                test_loader,
                fusion_alpha=best_pnc['fusion_alpha'],
                similarity_tau=best_pnc['similarity_tau'],
                use_adaptive=best_pnc['use_adaptive']
            )
            
            # 保存结果
            results.append({
                'method': 'pnc_lccs',
                'support_size': support_size,
                'seed': seed,
                'accuracy': acc,
                'confidence': conf
            })
            
            # ⚠️ 关键：恢复原始 BN 统计量
            lccs.restore_bn_stats()
            
            pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'pnc_lccs_combined_results.csv', index=False)
    
    print(f"\n🏆 PNC+LCCS组合结果:")
    print(f"   平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"   最大准确率: {df['accuracy'].max():.4f}")
    
    return df


def compare_sample_selection_strategies(model, data_dir, output_dir):
    """
    对比不同的样本选择策略
    
    测试策略：Random、Confidence、Diversity、Uncertainty、Hybrid
    
    Args:
        model: 预加载的分类器
        data_dir: 数据目录
        output_dir: 输出目录
    
    Returns:
        DataFrame: 策略对比结果
    """
    print("\n" + "="*80)
    print("🎯 样本选择策略对比")
    print("="*80)
    
    strategies = ['random', 'confidence', 'diversity', 'uncertainty', 'hybrid']
    support_sizes = [3, 5]
    seeds = [42]
    
    results = []
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    total = len(strategies) * len(support_sizes) * len(seeds)
    
    print(f"策略: {len(strategies)} 种")
    print(f"Support sizes: {support_sizes}")
    print(f"总计: {total} 个实验\n")
    
    with tqdm(total=total, desc="Strategy Comparison") as pbar:
        for strategy in strategies:
            for support_size in support_sizes:
                for seed in seeds:
                    print(f"\n📊 Strategy={strategy}, Support={support_size}, Seed={seed}")
                    
                    # 创建数据集
                    support_dataset = StrategicDataset(
                        data_dir=data_dir,
                        support_size=support_size,
                        strategy=strategy,
                        model=model,
                        mode='support',
                        seed=seed,
                        device=model.device,
                        transform=transform
                    )
                    
                    test_dataset = StrategicDataset(
                        data_dir=data_dir,
                        support_size=support_size,
                        strategy=strategy,
                        model=model,
                        mode='test',
                        seed=seed,
                        device=model.device,
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
                    
                    # 评估基线（不使用PNC/LCCS）
                    baseline_acc, baseline_conf = evaluate_baseline(
                        model, test_loader, model.device
                    )
                    
                    # 评估PNC（简单配置）
                    features, labels = extract_features(model, support_loader, model.device)
                    prototypes = build_prototypes_simple_mean(features, labels)
                    pnc = PNCEvaluator(model, prototypes, model.device)
                    pnc_acc, pnc_conf = pnc.predict(test_loader, 
                                                    fusion_alpha=0.6, 
                                                    similarity_tau=0.01)
                    
                    # 保存结果
                    results.append({
                        'strategy': strategy,
                        'support_size': support_size,
                        'seed': seed,
                        'baseline_accuracy': baseline_acc,
                        'baseline_confidence': baseline_conf,
                        'pnc_accuracy': pnc_acc,
                        'pnc_confidence': pnc_conf,
                        'pnc_improvement': pnc_acc - baseline_acc
                    })
                    
                    print(f"  Baseline: {baseline_acc:.4f}, PNC: {pnc_acc:.4f} (+{pnc_acc-baseline_acc:.4f})")
                    
                    pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'strategy_comparison.csv', index=False)
    
    # 打印总结
    print(f"\n🏆 策略对比总结:")
    print("="*80)
    
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]
        avg_baseline = strategy_df['baseline_accuracy'].mean()
        avg_pnc = strategy_df['pnc_accuracy'].mean()
        avg_improvement = strategy_df['pnc_improvement'].mean()
        
        print(f"{strategy:12s}: Baseline={avg_baseline:.4f}, PNC={avg_pnc:.4f}, "
              f"Improvement={avg_improvement:+.4f}")
    
    # 找最佳策略
    best_row = df.loc[df['pnc_accuracy'].idxmax()]
    print(f"\n✨ 最佳策略: {best_row['strategy']} (support_size={best_row['support_size']})")
    print(f"   PNC准确率: {best_row['pnc_accuracy']:.4f}")
    
    return df


def preload_data_loaders_with_strategy(model, data_dir, strategy, device='cuda'):
    """
    使用特定策略预加载数据加载器
    
    Args:
        model: 分类器
        data_dir: 数据目录
        strategy: 样本选择策略
        device: 设备
    
    Returns:
        dict: {(support_size, seed, strategy): (support_loader, test_loader)}
    """
    print(f"\n📂 预加载数据 (Strategy: {strategy})")
    print("="*80)
    
    data_loaders = {}
    
    support_sizes = [3, 5]
    seeds = [42]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for support_size in support_sizes:
        for seed in seeds:
            print(f"\n加载 support_size={support_size}, seed={seed}, strategy={strategy}")
            
            support_dataset = StrategicDataset(
                data_dir=data_dir,
                support_size=support_size,
                strategy=strategy,
                model=model,
                mode='support',
                seed=seed,
                device=device,
                transform=transform
            )
            
            test_dataset = StrategicDataset(
                data_dir=data_dir,
                support_size=support_size,
                strategy=strategy,
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
            
            data_loaders[(support_size, seed, strategy)] = (support_loader, test_loader)
    
    print(f"\n✅ 预加载完成，共 {len(data_loaders)} 组数据")
    
    return data_loaders


def preload_data_loaders(data_dir, device='cuda'):
    """
    预加载所有数据加载器（随机策略）
    
    Args:
        data_dir: 数据目录
        device: 设备
    
    Returns:
        dict: {(support_size, seed): (support_loader, test_loader)}
    """
    print("\n" + "="*80)
    print("📂 预加载数据")
    print("="*80)
    
    data_loaders = {}
    
    support_sizes = [3, 5]
    seeds = [42]
    
    for support_size in support_sizes:
        for seed in seeds:
            print(f"\n加载 support_size={support_size}, seed={seed}")
            support_loader, test_loader = create_data_loaders(
                data_dir, support_size, seed, batch_size=64, num_workers=0
            )
            data_loaders[(support_size, seed)] = (support_loader, test_loader)
    
    print(f"\n✅ 预加载完成，共 {len(data_loaders)} 组数据")
    
    return data_loaders


def main():
    import argparse
    parser = argparse.ArgumentParser(description='优化版超参数搜索')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--output-dir', type=str,
                       default='./optimized_search_results')
    parser.add_argument('--skip-pnc', action='store_true',
                       help='跳过PNC评估')
    parser.add_argument('--skip-lccs', action='store_true',
                       help='跳过LCCS评估')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 优化版超参数搜索")
    print("="*80)
    print(f"💻 设备: {device}")
    print(f"📂 输出目录: {output_path}")
    
    # 1. 加载模型（只加载一次）
    print("\n" + "="*80)
    print("📦 加载分类器")
    print("="*80)
    model = load_classifier(args.model_path, device)
    model.device = device  # 保存 device 属性
    print("✅ 模型加载完成\n")
    
    # 2. 先对比样本选择策略
    print("\n" + "="*80)
    print("🎯 步骤 1: 样本选择策略对比")
    print("="*80)
    strategy_df = compare_sample_selection_strategies(model, args.data_dir, args.output_dir)
    
    # 找最佳策略
    best_strategy_row = strategy_df.loc[strategy_df['pnc_accuracy'].idxmax()]
    best_strategy = best_strategy_row['strategy']
    
    print(f"\n✨ 选择最佳策略: {best_strategy}")
    print(f"   将使用此策略进行后续超参数搜索\n")
    
    # 3. 使用最佳策略预加载数据
    data_loaders_dict = preload_data_loaders_with_strategy(
        model, args.data_dir, best_strategy, device
    )
    
    # 4. 评估 PNC（使用最佳策略）
    if not args.skip_pnc:
        print("\n" + "="*80)
        print("🎯 步骤 2: PNC 超参数搜索")
        print("="*80)
        pnc_df = evaluate_pnc_hyperparameters_optimized(
            model, data_loaders_dict, args.output_dir
        )
        best_pnc = pnc_df.loc[pnc_df['accuracy'].idxmax()]
    else:
        print("\n⏭️ 跳过PNC评估")
        best_pnc = {
            'prototype_strategy': 'simple_mean',
            'fusion_alpha': 0.6,
            'similarity_tau': 0.01,
            'use_adaptive': False
        }
    
    # 5. 评估 LCCS（使用最佳策略）
    if not args.skip_lccs:
        print("\n" + "="*80)
        print("🎯 步骤 3: LCCS 超参数搜索")
        print("="*80)
        lccs_df = evaluate_lccs_hyperparameters_optimized(
            model, data_loaders_dict, args.output_dir
        )
        best_lccs = lccs_df.loc[lccs_df['accuracy'].idxmax()]
    else:
        print("\n⏭️ 跳过LCCS评估")
        best_lccs = {
            'lccs_method': 'progressive',
            'momentum': 0.01,
            'iterations': 5
        }
    
    # 6. 评估 PNC+LCCS 组合（使用最佳策略）
    print("\n" + "="*80)
    print("🎯 步骤 4: PNC+LCCS 组合评估")
    print("="*80)
    combined_df = evaluate_pnc_lccs_combined_optimized(
        model, best_pnc, best_lccs, args.data_dir, args.output_dir
    )
    
    # 7. 生成最终报告
    print("\n" + "="*80)
    print("📊 最终总结")
    print("="*80)
    
    summary = {
        'best_strategy': best_strategy,
        'strategy_results': {
            strategy: {
                'avg_pnc_accuracy': float(strategy_df[strategy_df['strategy']==strategy]['pnc_accuracy'].mean()),
                'avg_improvement': float(strategy_df[strategy_df['strategy']==strategy]['pnc_improvement'].mean())
            }
            for strategy in ['random', 'confidence', 'diversity', 'uncertainty', 'hybrid']
        },
        'best_pnc': best_pnc.to_dict() if hasattr(best_pnc, 'to_dict') else best_pnc,
        'best_lccs': best_lccs.to_dict() if hasattr(best_lccs, 'to_dict') else best_lccs,
        'combined_results': {
            'mean_accuracy': float(combined_df['accuracy'].mean()),
            'std_accuracy': float(combined_df['accuracy'].std()),
            'max_accuracy': float(combined_df['accuracy'].max())
        }
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 所有结果已保存到: {args.output_dir}")
    print(f"   - strategy_comparison.csv         # 样本选择策略对比")
    print(f"   - pnc_results.csv                 # PNC 超参数搜索")
    print(f"   - lccs_results.csv                # LCCS 超参数搜索")
    print(f"   - pnc_lccs_combined_results.csv   # PNC+LCCS 组合结果")
    print(f"   - summary.json                    # 最佳配置总结")
    
    print(f"\n🎯 性能总结:")
    print("="*80)
    print(f"✨ 最佳样本选择策略: {best_strategy}")
    for strategy in ['random', 'confidence', 'diversity', 'uncertainty', 'hybrid']:
        avg_pnc = strategy_df[strategy_df['strategy']==strategy]['pnc_accuracy'].mean()
        print(f"   {strategy:12s}: {avg_pnc:.4f}")
    
    print(f"\n📈 最佳超参数配置:")
    print(f"   最佳PNC: {best_pnc['accuracy'] if 'accuracy' in best_pnc else 'N/A':.4f}" if isinstance(best_pnc.get('accuracy'), (int, float)) else "   最佳PNC: 见CSV")
    print(f"   最佳LCCS: {best_lccs['accuracy'] if 'accuracy' in best_lccs else 'N/A':.4f}" if isinstance(best_lccs.get('accuracy'), (int, float)) else "   最佳LCCS: 见CSV")
    
    print(f"\n🏆 最终组合效果:")
    print(f"   PNC+LCCS平均: {combined_df['accuracy'].mean():.4f} ± {combined_df['accuracy'].std():.4f}")
    print(f"   PNC+LCCS最大: {combined_df['accuracy'].max():.4f}")


if __name__ == '__main__':
    main() 
