#!/usr/bin/env python3
"""
完整超参数搜索评估
测试所有方法和超参数组合
"""

import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from itertools import product
import sys

sys.path.append(str(Path(__file__).parent))
from eval_config import *
from eval_utils import *
from eval_components import *
from run_comprehensive_eval import run_single_experiment


def evaluate_pnc_hyperparameters(model_path, data_dir, output_dir):
    """评估PNC的超参数"""
    print("\n" + "="*80)
    print("🔍 PNC超参数搜索")
    print("="*80)
    
    results = []
    
    # 测试配置
    prototype_strategies = ['simple_mean', 'weighted_mean']
    fusion_alphas = [0.4, 0.5, 0.6, 0.7]
    similarity_taus = [0.005, 0.01, 0.02]
    use_adaptives = [False, True]
    support_sizes = [3, 5]
    seeds = [42]
    
    total = (len(prototype_strategies) * len(fusion_alphas) * 
             len(similarity_taus) * len(use_adaptives) * 
             len(support_sizes) * len(seeds))
    
    print(f"总共 {total} 个配置")
    
    with tqdm(total=total, desc="PNC") as pbar:
        for (proto_strat, alpha, tau, adaptive, support_size, seed) in product(
            prototype_strategies, fusion_alphas, similarity_taus,
            use_adaptives, support_sizes, seeds
        ):
            params = {
                'prototype_strategy': proto_strat,
                'fusion_alpha': alpha,
                'similarity_tau': tau,
                'use_adaptive': adaptive
            }
            
            result = run_single_experiment(
                model_path, data_dir, support_size, seed,
                method='pnc', params=params, device='cuda'
            )
            
            results.append({
                'method': 'pnc',
                'prototype_strategy': proto_strat,
                'fusion_alpha': alpha,
                'similarity_tau': tau,
                'use_adaptive': adaptive,
                'support_size': support_size,
                'seed': seed,
                'accuracy': result['accuracy'],
                'confidence': result['confidence']
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


def evaluate_lccs_hyperparameters(model_path, data_dir, output_dir):
    """评估LCCS的超参数"""
    print("\n" + "="*80)
    print("🔍 LCCS超参数搜索")
    print("="*80)
    
    results = []
    
    support_sizes = [3, 5]
    seeds = [42]
    
    # Progressive方法
    prog_configs = list(product(
        [0.005, 0.01, 0.02],  # momentum
        [3, 5, 10]  # iterations
    ))
    
    # Weighted方法
    weighted_configs = [0.2, 0.3, 0.4]  # alpha
    
    total = (len(prog_configs) + len(weighted_configs)) * len(support_sizes) * len(seeds)
    
    print(f"总共 {total} 个配置")
    
    with tqdm(total=total, desc="LCCS") as pbar:
        # Progressive
        for (momentum, iterations, support_size, seed) in product(
            *zip(*prog_configs), support_sizes, seeds
        ):
            params = {
                'lccs_method': 'progressive',
                'lccs_params': {
                    'momentum': momentum,
                    'iterations': iterations
                }
            }
            
            result = run_single_experiment(
                model_path, data_dir, support_size, seed,
                method='lccs', params=params, device='cuda'
            )
            
            results.append({
                'method': 'lccs',
                'lccs_method': 'progressive',
                'momentum': momentum,
                'iterations': iterations,
                'alpha': None,
                'support_size': support_size,
                'seed': seed,
                'accuracy': result['accuracy'],
                'confidence': result['confidence']
            })
            
            pbar.update(1)
        
        # Weighted
        for (alpha, support_size, seed) in product(
            weighted_configs, support_sizes, seeds
        ):
            params = {
                'lccs_method': 'weighted',
                'lccs_params': {'alpha': alpha}
            }
            
            result = run_single_experiment(
                model_path, data_dir, support_size, seed,
                method='lccs', params=params, device='cuda'
            )
            
            results.append({
                'method': 'lccs',
                'lccs_method': 'weighted',
                'momentum': None,
                'iterations': None,
                'alpha': alpha,
                'support_size': support_size,
                'seed': seed,
                'accuracy': result['accuracy'],
                'confidence': result['confidence']
            })
            
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


def evaluate_pnc_lccs_combined(model_path, data_dir, output_dir,
                               best_pnc, best_lccs):
    """
    评估PNC+LCCS组合
    使用最佳超参数配置
    """
    print("\n" + "="*80)
    print("🔍 PNC+LCCS组合评估（使用最佳配置）")
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
            # 准备LCCS参数
            if best_lccs['lccs_method'] == 'progressive':
                lccs_params = {
                    'momentum': best_lccs['momentum'],
                    'iterations': best_lccs['iterations']
                }
            else:
                lccs_params = {'alpha': best_lccs['alpha']}
            
            params = {
                'prototype_strategy': best_pnc['prototype_strategy'],
                'fusion_alpha': best_pnc['fusion_alpha'],
                'similarity_tau': best_pnc['similarity_tau'],
                'use_adaptive': best_pnc['use_adaptive'],
                'lccs_method': best_lccs['lccs_method'],
                'lccs_params': lccs_params
            }
            
            result = run_single_experiment(
                model_path, data_dir, support_size, seed,
                method='pnc_lccs', params=params, device='cuda'
            )
            
            results.append({
                'method': 'pnc_lccs',
                'support_size': support_size,
                'seed': seed,
                'accuracy': result['accuracy'],
                'confidence': result['confidence']
            })
            
            pbar.update(1)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'pnc_lccs_combined_results.csv', index=False)
    
    print(f"\n🏆 PNC+LCCS组合结果:")
    print(f"   平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='完整超参数搜索')
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--output-dir', type=str,
                       default='./hyperparameter_search_results')
    parser.add_argument('--skip-pnc', action='store_true',
                       help='跳过PNC评估')
    parser.add_argument('--skip-lccs', action='store_true',
                       help='跳过LCCS评估')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 评估PNC
    if not args.skip_pnc:
        pnc_df = evaluate_pnc_hyperparameters(
            args.model_path, args.data_dir, args.output_dir
        )
        best_pnc = pnc_df.loc[pnc_df['accuracy'].idxmax()]
    else:
        print("⏭️ 跳过PNC评估")
        best_pnc = {
            'prototype_strategy': 'simple_mean',
            'fusion_alpha': 0.6,
            'similarity_tau': 0.01,
            'use_adaptive': False
        }
    
    # 2. 评估LCCS
    if not args.skip_lccs:
        lccs_df = evaluate_lccs_hyperparameters(
            args.model_path, args.data_dir, args.output_dir
        )
        best_lccs = lccs_df.loc[lccs_df['accuracy'].idxmax()]
    else:
        print("⏭️ 跳过LCCS评估")
        best_lccs = {
            'lccs_method': 'progressive',
            'momentum': 0.01,
            'iterations': 5
        }
    
    # 3. 评估PNC+LCCS组合
    combined_df = evaluate_pnc_lccs_combined(
        args.model_path, args.data_dir, args.output_dir,
        best_pnc, best_lccs
    )
    
    # 4. 生成最终报告
    print("\n" + "="*80)
    print("📊 最终总结")
    print("="*80)
    
    summary = {
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
    print(f"   - pnc_results.csv")
    print(f"   - lccs_results.csv")
    print(f"   - pnc_lccs_combined_results.csv")
    print(f"   - summary.json")


if __name__ == '__main__':
    main() 
