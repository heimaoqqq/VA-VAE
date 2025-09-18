#!/usr/bin/env python3
"""
使用原型校准（PNC）评估域适应效果
对比基线方法与原型校准方法在背包步态识别上的性能
适配Kaggle环境
"""

import torch
import argparse
from pathlib import Path
import json
import numpy as np
from tabulate import tabulate
import sys
sys.path.append(str(Path(__file__).parent.parent))
from cross_domain_evaluator import CrossDomainEvaluator


def evaluate_prototype_calibration(args):
    """评估原型校准的效果"""
    
    print("\n" + "="*80)
    print("🔬 PROTOTYPE CALIBRATION EVALUATION")
    print("="*80)
    
    # 创建评估器
    evaluator = CrossDomainEvaluator()
    
    # 加载分类器
    print(f"\n📦 Loading classifier: {args.model_path}")
    model, checkpoint = evaluator.load_classifier(args.model_path)
    
    # 1. 基线评估（不使用原型）
    print("\n" + "-"*60)
    print("📊 BASELINE EVALUATION (without prototypes)")
    print("-"*60)
    baseline_results = evaluator.evaluate_on_target_domain(
        model=model,
        target_data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_prototypes=False
    )
    
    # 2. 原型校准评估（使用原型）
    print("\n" + "-"*60)
    print("🎯 PROTOTYPE CALIBRATION EVALUATION")
    print("-"*60)
    pnc_results = evaluator.evaluate_on_target_domain(
        model=model,
        target_data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_prototypes=True,
        prototype_path=args.prototype_path,
        fusion_alpha=args.fusion_alpha,
        similarity_tau=args.similarity_tau
    )
    
    # 3. 对比分析
    print("\n" + "="*80)
    print("📈 COMPARISON RESULTS")
    print("="*80)
    
    if baseline_results and pnc_results:
        # 准确率对比
        baseline_acc = baseline_results['overall_accuracy']
        pnc_acc = pnc_results['overall_accuracy']
        improvement = pnc_acc - baseline_acc
        
        # 置信度对比
        baseline_conf = baseline_results['confidence_stats']['mean_confidence']
        pnc_conf = pnc_results['confidence_stats']['mean_confidence']
        
        # 创建对比表格
        comparison_table = [
            ["Metric", "Baseline", "PNC", "Improvement"],
            ["Overall Accuracy", f"{baseline_acc:.2%}", f"{pnc_acc:.2%}", f"{improvement:+.2%}"],
            ["Mean Confidence", f"{baseline_conf:.3f}", f"{pnc_conf:.3f}", f"{pnc_conf-baseline_conf:+.3f}"],
            ["Total Samples", baseline_results['total_samples'], pnc_results['total_samples'], "-"],
        ]
        
        print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))
        
        # 用户级别分析
        if 'per_user_accuracy' in baseline_results and 'per_user_accuracy' in pnc_results:
            print("\n📊 PER-USER ACCURACY ANALYSIS:")
            
            user_improvements = []
            for user_id in baseline_results['per_user_accuracy']:
                if user_id in pnc_results['per_user_accuracy']:
                    baseline_user = baseline_results['per_user_accuracy'][user_id]
                    pnc_user = pnc_results['per_user_accuracy'][user_id]
                    improv = pnc_user - baseline_user
                    user_improvements.append((user_id, baseline_user, pnc_user, improv))
            
            # 排序：按改进幅度从大到小
            user_improvements.sort(key=lambda x: x[3], reverse=True)
            
            # 显示前5个改进最大的用户
            print("\nTop 5 Most Improved Users:")
            top_table = [["User", "Baseline", "PNC", "Improvement"]]
            for user_id, base, pnc, improv in user_improvements[:5]:
                top_table.append([f"User_{user_id+1}", f"{base:.1%}", f"{pnc:.1%}", f"{improv:+.1%}"])
            print(tabulate(top_table, headers="firstrow", tablefmt="simple"))
            
            # 统计改进情况
            improvements = [x[3] for x in user_improvements]
            improved_count = sum(1 for x in improvements if x > 0)
            
            print(f"\n📈 IMPROVEMENT STATISTICS:")
            print(f"   • Users improved: {improved_count}/{len(user_improvements)} ({improved_count/len(user_improvements):.1%})")
            print(f"   • Mean improvement: {np.mean(improvements):+.2%}")
            print(f"   • Std improvement: {np.std(improvements):.2%}")
            print(f"   • Max improvement: {max(improvements):+.2%}")
            print(f"   • Min change: {min(improvements):+.2%}")
        
        # 评估结论
        print("\n" + "="*80)
        if improvement > 0.05:
            assessment = "🟢 SIGNIFICANT IMPROVEMENT"
            message = "Prototype calibration provides substantial gains for domain adaptation"
        elif improvement > 0.02:
            assessment = "🟡 MODERATE IMPROVEMENT"
            message = "Prototype calibration offers meaningful benefits"
        elif improvement > 0:
            assessment = "🟠 SLIGHT IMPROVEMENT"
            message = "Prototype calibration provides marginal gains"
        else:
            assessment = "🔴 NO IMPROVEMENT"
            message = "Prototype calibration does not improve performance"
        
        print(f"🏆 ASSESSMENT: {assessment}")
        print(f"💡 {message}")
        
        # 保存结果
        if args.save_results:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = {
                'baseline': baseline_results,
                'pnc': pnc_results,
                'comparison': {
                    'accuracy_improvement': improvement,
                    'confidence_change': pnc_conf - baseline_conf,
                    'assessment': assessment,
                    'message': message
                },
                'config': {
                    'model_path': args.model_path,
                    'data_dir': args.data_dir,
                    'prototype_path': args.prototype_path,
                    'fusion_alpha': args.fusion_alpha,
                    'similarity_tau': args.similarity_tau
                }
            }
            
            # 转换numpy int64键为Python int以支持JSON序列化
            def convert_keys(obj):
                if isinstance(obj, dict):
                    return {str(k) if hasattr(k, 'item') else k: convert_keys(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_keys(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_keys(results)
            save_path = output_path / 'prototype_calibration_results.json'
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {save_path}")
    
    else:
        print("❌ Evaluation failed")
    
    print("="*80)


def grid_search_hyperparameters(args):
    """网格搜索最优超参数"""
    
    print("\n" + "="*80)
    print("🔍 HYPERPARAMETER GRID SEARCH")
    print("="*80)
    
    # 创建评估器
    evaluator = CrossDomainEvaluator()
    
    # 加载模型
    model, _ = evaluator.load_classifier(args.model_path)
    
    # 定义搜索网格
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    taus = [0.05, 0.1, 0.15, 0.2]
    
    best_acc = 0
    best_params = {}
    results_grid = []
    
    print(f"\nSearching over {len(alphas)} alphas × {len(taus)} taus = {len(alphas)*len(taus)} combinations")
    
    for alpha in alphas:
        for tau in taus:
            print(f"\nTesting α={alpha:.1f}, τ={tau:.2f}...")
            
            results = evaluator.evaluate_on_target_domain(
                model=model,
                target_data_dir=args.data_dir,
                batch_size=args.batch_size,
                use_prototypes=True,
                prototype_path=args.prototype_path,
                fusion_alpha=alpha,
                similarity_tau=tau
            )
            
            if results:
                acc = results['overall_accuracy']
                results_grid.append({
                    'alpha': alpha,
                    'tau': tau,
                    'accuracy': acc
                })
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'alpha': alpha, 'tau': tau}
                
                print(f"   Accuracy: {acc:.2%}")
    
    # 显示结果
    print("\n" + "="*80)
    print("🏆 BEST PARAMETERS:")
    print(f"   • Alpha (α): {best_params['alpha']:.1f}")
    print(f"   • Tau (τ): {best_params['tau']:.2f}")
    print(f"   • Best Accuracy: {best_acc:.2%}")
    
    # 创建结果表格
    print("\n📊 FULL RESULTS:")
    table_data = [["α \\ τ"] + [f"{tau:.2f}" for tau in taus]]
    for alpha in alphas:
        row = [f"{alpha:.1f}"]
        for tau in taus:
            acc = next((r['accuracy'] for r in results_grid 
                       if r['alpha'] == alpha and r['tau'] == tau), None)
            if acc:
                row.append(f"{acc:.1%}")
            else:
                row.append("-")
        table_data.append(row)
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    return best_params


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate prototype calibration for domain adaptation')
    
    # 模型和数据路径（Kaggle环境）
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/input/best-calibrated-model-pth/best_calibrated_model.pth',
                       help='Path to trained classifier')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data (背包步态)')
    parser.add_argument('--prototype-path', type=str,
                       default='/kaggle/working/target_prototypes.pt',
                       help='Path to computed prototypes')
    
    # 原型校准超参数
    parser.add_argument('--fusion-alpha', type=float, default=0.4,
                       help='Prototype fusion weight (0-1)')
    parser.add_argument('--similarity-tau', type=float, default=0.1,
                       help='Similarity temperature parameter')
    
    # 其他参数
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform hyperparameter grid search')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save evaluation results')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working/pnc_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 检查原型文件是否存在
    if not Path(args.prototype_path).exists():
        print(f"⚠️ Prototype file not found: {args.prototype_path}")
        print("Please run build_target_prototypes.py first to create prototypes")
        return
    
    # 执行评估
    if args.grid_search:
        # 网格搜索最优参数
        best_params = grid_search_hyperparameters(args)
        
        # 使用最优参数重新评估
        print("\n" + "="*80)
        print("🎯 FINAL EVALUATION WITH BEST PARAMETERS")
        args.fusion_alpha = best_params['alpha']
        args.similarity_tau = best_params['tau']
    
    # 主评估
    evaluate_prototype_calibration(args)


if __name__ == '__main__':
    main()
