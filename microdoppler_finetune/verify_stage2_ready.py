#!/usr/bin/env python3
"""
Stage 2 训练前置条件验证脚本
确保所有配置和checkpoint正确
"""

import os
import sys
from pathlib import Path
import torch
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
    # 检查Kaggle数据路径
    kaggle_split_file = Path("/kaggle/working/data_split/dataset_split.json")
    local_split_file = Path('data/microdoppler_split.json')
    
    split_file = kaggle_split_file if kaggle_split_file.exists() else local_split_file
    
    if split_file.exists():
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        train_count = sum(len(imgs) for imgs in split_data['train'].values())
        val_count = sum(len(imgs) for imgs in split_data['val'].values())
        
        print(f"   ✅ 数据划分文件: {split_file}")
        print(f"   ✅ 训练集: {train_count} 张图像")
        print(f"   ✅ 验证集: {val_count} 张图像")
        
        if train_count < 100:
            warnings.append(f"⚠️ 训练集较小({train_count}张)，可能过拟合")
    else:
        issues.append("❌ 数据划分文件不存在 (检查了Kaggle和本地路径)")
    
    # 3. 检查损失计算修复
    print("\n🔧 损失计算修复检查:")
    loss_file = Path('../LightningDiT/vavae/ldm/modules/losses/contperceptual.py')
    
    if loss_file.exists():
        with open(loss_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'torch.mean(weighted_nll_loss)' in content:
            print("   ✅ 损失计算已修复为torch.mean()")
        else:
            issues.append("❌ 损失计算未修复，将导致训练损失异常高")
    else:
        warnings.append("⚠️ 无法验证损失计算修复状态")
    
    # 4. 检查预训练模型
    print("\n🎯 预训练模型检查:")
    pretrained_path = Path('../pretrained/vavae_ckpt.pt')
    
    if not pretrained_path.exists():
        issues.append("❌ 预训练模型不存在")
    else:
        size_mb = pretrained_path.stat().st_size / (1024*1024)
        print(f"   ✅ 预训练模型存在 ({size_mb:.1f} MB)")
    
    # 5. Stage 2配置验证
    print("\n⚙️ Stage 2 配置验证:")
    print("   预期配置:")
    print("   - 判别器启动: epoch 1 (立即启动)")
    print("   - VF权重: 0.1 (降低以优化重建)")
    print("   - 学习率: 5e-5 (降低以稳定训练)")
    print("   - 最大轮次: 45")
    print("   - 批次大小: 建议4 (GPU内存限制)")
    
    # 总结
    print("\n" + "="*60)
    print("📋 检查总结:")
    
    if issues:
        print("\n❌ 发现严重问题 (必须修复):")
        for issue in issues:
            print(f"   {issue}")
    
    if warnings:
        print("\n⚠️ 警告 (建议关注):")
        for warning in warnings:
            print(f"   {warning}")
    
    if not issues:
        print("\n✅ 所有检查通过！可以开始Stage 2训练")
        print("\n建议运行命令:")
        print("python step4_train_vavae.py --stage 2 --batch_size 4")
    else:
        print("\n❌ 请先解决以上问题再开始Stage 2训练")
    
    print("="*60)
    
    return len(issues) == 0

if __name__ == '__main__':
    success = verify_stage2_readiness()
    sys.exit(0 if success else 1)
