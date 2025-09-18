#!/usr/bin/env python3
"""
修复版LCCS - 不完全重置BN统计量，而是微调
"""

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from improved_classifier_training import ImprovedClassifier
from build_improved_prototypes_with_split import SplitTargetDomainDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class FixedLCCSAdapter:
    """修复版LCCS：保留原始统计量，仅微调"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.original_bn_stats = self._save_bn_stats()
        
    def _save_bn_stats(self):
        """保存原始BN层的统计量"""
        bn_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'momentum': module.momentum,
                    'num_batches_tracked': module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None
                }
        return bn_stats
    
    def _restore_bn_stats(self):
        """恢复原始BN统计量"""
        for name, module in self.model.named_modules():
            if name in self.original_bn_stats:
                module.running_mean.data = self.original_bn_stats[name]['running_mean']
                module.running_var.data = self.original_bn_stats[name]['running_var']
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.data = self.original_bn_stats[name]['num_batches_tracked']
    
    def adapt_bn_stats_v1(self, support_loader, alpha=0.3):
        """方法1：加权融合原始和目标域统计量"""
        print(f"🔧 Method 1: Weighted fusion (α={alpha})...")
        
        # 收集目标域统计量（不重置原始统计量）
        self.model.train()
        
        # 保存当前统计量
        source_stats = self._save_bn_stats()
        
        # 临时重置以收集纯目标域统计量
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
                module.momentum = 1.0  # 快速收集
        
        # 收集目标域统计量
        with torch.no_grad():
            for _ in range(10):  # 多次迭代以稳定
                for batch in support_loader:
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        # 保存目标域统计量
        target_stats = self._save_bn_stats()
        
        # 融合源域和目标域统计量
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in source_stats and name in target_stats:
                    # 加权平均
                    module.running_mean = (1-alpha) * source_stats[name]['running_mean'] + \
                                         alpha * target_stats[name]['running_mean']
                    module.running_var = (1-alpha) * source_stats[name]['running_var'] + \
                                        alpha * target_stats[name]['running_var']
        
        self.model.eval()
        print("✅ BN stats adapted via weighted fusion!")
    
    def adapt_bn_stats_v2(self, support_loader, momentum=0.01, iterations=5):
        """方法2：渐进式更新（小momentum）"""
        print(f"🔧 Method 2: Progressive update (momentum={momentum}, iter={iterations})...")
        
        # 设置为训练模式但不更新参数
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 使用小momentum渐进更新
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.momentum = momentum  # 小步更新
                # 不重置统计量！保留原始值
        
        # 多次迭代渐进更新
        with torch.no_grad():
            for iteration in range(iterations):
                for batch in tqdm(support_loader, desc=f"Iter {iteration+1}/{iterations}", leave=False):
                    if len(batch) == 3:
                        images, _, _ = batch
                    else:
                        images, _ = batch
                    images = images.to(self.device)
                    _ = self.model(images)
        
        self.model.eval()
        print("✅ BN stats progressively updated!")
    
    def adapt_bn_stats_v3(self, support_loader):
        """方法3：仅调整均值偏移"""
        print("🔧 Method 3: Mean-shift only...")
        
        self.model.eval()
        
        # 计算支持集特征的均值偏移
        with torch.no_grad():
            source_means = {}
            target_means = {}
            
            # 收集每层的激活均值
            def hook_fn(name):
                def hook(module, input, output):
                    if name not in target_means:
                        target_means[name] = []
                    if isinstance(output, torch.Tensor):
                        # 计算batch均值
                        if len(output.shape) == 4:  # Conv层
                            mean = output.mean(dim=[0,2,3])
                        else:  # Linear层
                            mean = output.mean(dim=0)
                        target_means[name].append(mean.cpu())
                return hook
            
            # 注册钩子
            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # 前向传播收集统计
            for batch in support_loader:
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images, _ = batch
                images = images.to(self.device)
                _ = self.model(images)
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 计算均值偏移并应用
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if name in target_means and len(target_means[name]) > 0:
                        target_mean = torch.stack(target_means[name]).mean(dim=0)
                        source_mean = module.running_mean
                        # 调整均值
                        shift = target_mean.to(self.device) - source_mean
                        module.running_mean = source_mean + 0.3 * shift  # 部分偏移
        
        print("✅ Mean-shift adaptation complete!")
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy


def test_all_methods(model_path, data_dir, support_size=3, seed=42):
    """测试所有LCCS方法"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 支持集
    support_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='support',
        seed=seed
    )
    
    support_loader = DataLoader(
        support_dataset,
        batch_size=31,  # 所有用户一个batch
        shuffle=True,
        num_workers=2
    )
    
    # 测试集
    test_dataset = SplitTargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size,
        mode='test',
        seed=seed
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    print(f"📊 Data split: {len(support_dataset)} support, {len(test_dataset)} test")
    
    results = {}
    
    # 基线测试
    print("\n" + "="*60)
    print("📊 BASELINE (No adaptation)")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ImprovedClassifier(
        num_classes=checkpoint.get('num_classes', 31),
        backbone=checkpoint.get('backbone', 'resnet18')
    ).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    adapter = FixedLCCSAdapter(model, device)
    baseline_acc = adapter.evaluate(test_loader)
    results['baseline'] = baseline_acc
    print(f"Baseline accuracy: {baseline_acc:.2%}")
    
    # 方法1：加权融合
    print("\n" + "="*60)
    print("🔬 METHOD 1: Weighted Fusion")
    adapter._restore_bn_stats()  # 恢复原始
    adapter.adapt_bn_stats_v1(support_loader, alpha=0.3)
    method1_acc = adapter.evaluate(test_loader)
    results['weighted_fusion'] = method1_acc
    print(f"Method 1 accuracy: {method1_acc:.2%} ({method1_acc-baseline_acc:+.2%})")
    
    # 方法2：渐进更新
    print("\n" + "="*60)
    print("🔬 METHOD 2: Progressive Update")
    adapter._restore_bn_stats()  # 恢复原始
    adapter.adapt_bn_stats_v2(support_loader, momentum=0.01, iterations=5)
    method2_acc = adapter.evaluate(test_loader)
    results['progressive'] = method2_acc
    print(f"Method 2 accuracy: {method2_acc:.2%} ({method2_acc-baseline_acc:+.2%})")
    
    # 方法3：均值偏移
    print("\n" + "="*60)
    print("🔬 METHOD 3: Mean-shift Only")
    adapter._restore_bn_stats()  # 恢复原始
    adapter.adapt_bn_stats_v3(support_loader)
    method3_acc = adapter.evaluate(test_loader)
    results['mean_shift'] = method3_acc
    print(f"Method 3 accuracy: {method3_acc:.2%} ({method3_acc-baseline_acc:+.2%})")
    
    # 总结
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for method, acc in results.items():
        improvement = acc - results['baseline'] if method != 'baseline' else 0
        print(f"{method:20s}: {acc:.2%} ({improvement:+.2%})")
    
    best_method = max(results, key=results.get)
    print(f"\n🏆 Best method: {best_method} ({results[best_method]:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack')
    parser.add_argument('--support-size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    test_all_methods(args.model_path, args.data_dir, args.support_size, args.seed)


if __name__ == '__main__':
    main()
