#!/usr/bin/env python3
"""
LCCS (Label-Conditional Channel Statistics) Adapter
基于Few-Shot Adaptation论文的BatchNorm适应方法
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
from build_improved_prototypes import TargetDomainDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class LCCSAdapter:
    """LCCS适配器：通过少量目标域样本更新BatchNorm统计量"""
    
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
    
    def adapt_bn_stats(self, support_loader, momentum=0.1):
        """使用支持集更新BN统计量"""
        print("🔧 Adapting BatchNorm statistics with support set...")
        
        # 设置BN层为训练模式以更新统计量
        self.model.train()
        
        # 但保持所有参数冻结（只更新BN统计量）
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 重置BN统计量
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.momentum = momentum
                module.reset_running_stats()
        
        # 在支持集上前向传播以收集统计量
        with torch.no_grad():
            for _ in range(3):  # 多次遍历以稳定统计量
                for images, _ in tqdm(support_loader, desc="Collecting BN stats"):
                    images = images.to(self.device)
                    _ = self.model(images)
        
        # 恢复评估模式
        self.model.eval()
        print("✅ BatchNorm statistics adapted!")
    
    def evaluate_with_lccs(self, test_loader):
        """使用适应后的模型评估"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating with LCCS"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy


def apply_lccs_and_evaluate(model_path, data_dir, support_size=50):
    """应用LCCS并评估"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print(f"📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ImprovedClassifier(
        num_classes=checkpoint.get('num_classes', 31),
        backbone=checkpoint.get('backbone', 'resnet18')
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建支持集（用于BN适应）
    support_dataset = TargetDomainDataset(
        data_dir=data_dir,
        transform=transform,
        support_size=support_size  # 更多样本用于BN统计
    )
    
    support_loader = DataLoader(
        support_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # 创建测试集（完整数据）
    from cross_domain_evaluator import BackpackWalkingDataset
    test_dataset = BackpackWalkingDataset(
        data_dir=data_dir,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # 基线评估
    print("\n📊 Baseline evaluation (without LCCS)...")
    model.eval()
    adapter = LCCSAdapter(model, device)
    baseline_acc = adapter.evaluate_with_lccs(test_loader)
    print(f"Baseline accuracy: {baseline_acc:.2%}")
    
    # LCCS适应
    adapter.adapt_bn_stats(support_loader, momentum=0.1)
    
    # LCCS评估
    print("\n🎯 Evaluation with LCCS adaptation...")
    lccs_acc = adapter.evaluate_with_lccs(test_loader)
    print(f"LCCS accuracy: {lccs_acc:.2%}")
    
    # 结果对比
    improvement = lccs_acc - baseline_acc
    print(f"\n📈 Improvement: {improvement:+.2%}")
    
    return baseline_acc, lccs_acc


def main():
    parser = argparse.ArgumentParser(description='LCCS Adapter for domain adaptation')
    
    parser.add_argument('--model-path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth',
                       help='Path to model')
    parser.add_argument('--data-dir', type=str,
                       default='/kaggle/input/backpack/backpack',
                       help='Path to target domain data')
    parser.add_argument('--support-size', type=int, default=50,
                       help='Support set size for BN adaptation')
    
    args = parser.parse_args()
    
    baseline_acc, lccs_acc = apply_lccs_and_evaluate(
        args.model_path,
        args.data_dir,
        args.support_size
    )
    
    print("\n" + "="*60)
    print("🏆 LCCS ADAPTATION COMPLETE")
    print(f"   Baseline: {baseline_acc:.2%}")
    print(f"   LCCS: {lccs_acc:.2%}")
    print(f"   Gain: {lccs_acc - baseline_acc:+.2%}")
    print("="*60)


if __name__ == '__main__':
    main()
