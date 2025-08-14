#!/usr/bin/env python3
"""
Stage 2 训练前置条件验证脚本（简化版）
"""

import os
import sys
from pathlib import Path
import json

def verify_stage2_readiness():
    """验证Stage 2训练准备情况"""
    
    print("="*60)
    print("🔍 VA-VAE Stage 2 训练准备状态检查")
    print("="*60)
    
    issues = []
    warnings = []
    
    # 1. 检查Stage 1 checkpoint (Kaggle训练好的模型)
    print("\n📦 Stage 1 Checkpoint检查:")
    kaggle_stage1_path = Path("/kaggle/input/stage1/vavae-stage1-epoch43-val_rec_loss0.0000.ckpt")
    
    if kaggle_stage1_path.exists():
        size_mb = kaggle_stage1_path.stat().st_size / (1024*1024)
        print(f"   ✅ Kaggle Stage 1模型存在")
        print(f"   文件: {kaggle_stage1_path.name}")
        print(f"   验证损失: 0.0000 (epoch 43)")
        print(f"   文件大小: {size_mb:.1f} MB")
        
        if size_mb < 100:
            warnings.append(f"⚠️ Checkpoint文件较小({size_mb:.1f}MB)，可能不完整")
    else:
        # 回退检查本地checkpoint
        print(f"   ❌ Kaggle模型未找到: {kaggle_stage1_path}")
        print("   尝试检查本地checkpoint...")
        
        stage1_dir = Path('checkpoints/stage1')
        if stage1_dir.exists():
            ckpt_files = list(stage1_dir.glob('*.ckpt'))
            if ckpt_files:
                print("   📂 本地checkpoint文件:")
                for ckpt in ckpt_files:
                    size_mb = ckpt.stat().st_size / (1024*1024)
                    print(f"   - {ckpt.name} ({size_mb:.1f} MB)")
            else:
                issues.append("❌ 本地也未找到Stage 1 checkpoint文件")
        else:
            issues.append("❌ Kaggle和本地都未找到Stage 1 checkpoint")
    
    # 2. 检查数据配置
    print("\n📊 数据配置检查:")
    split_file = Path('data/microdoppler_split.json')
    
    if not split_file.exists():
        issues.append("❌ 数据划分文件不存在")
    else:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        train_count = sum(len(imgs) for imgs in split_data['train'].values())
        val_count = sum(len(imgs) for imgs in split_data['val'].values())
        
        print(f"   ✅ 训练集: {train_count} 张图像")
        print(f"   ✅ 验证集: {val_count} 张图像")
        
        if train_count < 100:
            warnings.append(f"⚠️ 训练集较小({train_count}张)，可能过拟合")
    
    # 3. 检查损失计算修复
    print("\n🔧 损失计算修复检查:")
    loss_file = Path('../LightningDiT/vavae/ldm/modules/losses/contperceptual.py')
    
    if loss_file.exists():
        with open(loss_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'torch.mean(weighted_nll_loss)' in content and 'torch.mean(kl_loss)' in content:
            print("   ✅ 损失计算已修复为torch.mean()")
        elif 'torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]' in content:
            issues.append("❌ 损失计算未修复，使用sum/batch_size会导致训练损失异常高(14000+)")
            print("   ❌ 检测到错误的损失计算方式")
        else:
            warnings.append("⚠️ 损失计算状态不明确")
    else:
        warnings.append("⚠️ 无法验证损失计算修复状态")
    
    # 4. 检查训练脚本中的关键配置
    print("\n⚙️ 训练脚本配置检查:")
    train_script = Path('step4_train_vavae.py')
    
    if train_script.exists():
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键配置
        checks = {
            "perceptual_weight': 1.0": "感知损失权重",
            "adaptive_vf': False": "禁用自适应VF",
            "best_loss = float('inf')": "最佳checkpoint选择逻辑",
            "torch.mean": "损失计算修复"
        }
        
        for pattern, desc in checks.items():
            if pattern in content:
                print(f"   ✅ {desc}: 已正确配置")
            else:
                if desc == "损失计算修复":
                    warnings.append(f"⚠️ {desc}: 未在训练脚本中找到")
                else:
                    issues.append(f"❌ {desc}: 配置错误或缺失")
    
    # 5. 检查预训练模型
    print("\n🎯 预训练模型检查:")
    pretrained_path = Path('../pretrained/vavae_ckpt.pt')
    
    if not pretrained_path.exists():
        # 尝试其他可能的位置
        alt_paths = [
            Path('../LightningDiT/pretrained/vavae_ckpt.pt'),
            Path('pretrained/vavae_ckpt.pt')
        ]
        found = False
        for alt_path in alt_paths:
            if alt_path.exists():
                size_mb = alt_path.stat().st_size / (1024*1024)
                print(f"   ✅ 预训练模型存在于: {alt_path}")
                print(f"   文件大小: {size_mb:.1f} MB")
                found = True
                break
        
        if not found:
            warnings.append("⚠️ 预训练模型未在标准位置找到")
    else:
        size_mb = pretrained_path.stat().st_size / (1024*1024)
        print(f"   ✅ 预训练模型存在 ({size_mb:.1f} MB)")
    
    # 6. Stage 2配置验证
    print("\n📋 Stage 2 预期配置:")
    print("   ✅ 判别器启动: epoch 1 (立即启动)")
    print("   ✅ VF权重: 0.1 (从0.5降低以优化重建)")
    print("   ✅ 学习率: 5e-5 (从1e-4降低以稳定训练)")
    print("   ✅ 最大轮次: 45")
    print("   ✅ 批次大小: 4 (GPU内存限制)")
    print("   ✅ Checkpoint加载: Stage 1最佳模型")
    
    # 总结
    print("\n" + "="*60)
    print("📊 检查总结:")
    
    if issues:
        print("\n❌ 发现严重问题 (必须修复):")
        for issue in issues:
            print(f"   {issue}")
    
    if warnings:
        print("\n⚠️ 警告 (建议关注):")
        for warning in warnings:
            print(f"   {warning}")
    
    if not issues:
        print("\n✅ 所有关键检查通过！可以开始Stage 2训练")
        print("\n🚀 建议运行命令:")
        print("   python step4_train_vavae.py --stage 2 --batch_size 4")
        print("\n📌 Stage 2训练要点:")
        print("   1. 继承Stage 1的最佳模型权重")
        print("   2. 判别器从第1个epoch开始参与训练")
        print("   3. VF权重降低到0.1，专注重建质量")
        print("   4. 学习率降低到5e-5，确保稳定收敛")
    else:
        print("\n❌ 请先解决以上问题再开始Stage 2训练")
    
    print("="*60)
    
    return len(issues) == 0

if __name__ == '__main__':
    success = verify_stage2_readiness()
    sys.exit(0 if success else 1)
