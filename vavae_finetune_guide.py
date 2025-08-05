#!/usr/bin/env python3
"""
VA-VAE微调完整指南
专注于VA-VAE部分的微调，提供详细的步骤说明
"""

import os
import sys
from pathlib import Path

def show_vavae_finetune_guide():
    """显示VA-VAE微调完整指南"""
    print("🎯 VA-VAE专项微调完整指南")
    print("="*60)
    
    print("📚 背景说明:")
    print("   VA-VAE是本项目的核心创新，通过DINOv2对齐提升VAE性能")
    print("   微调VA-VAE可以让模型更好地适应微多普勒数据特征")
    print("   这是独立的微调过程，不涉及后续的扩散模型训练")
    
    print("\n🔍 微调前评估:")
    print("   1. 运行基础评估了解当前效果:")
    print("      !python evaluate_vae_quality.py")
    print("   2. 查看MSE和FID指标，决定是否需要微调")
    print("   3. 当前预期: MSE≈0.006, FID≈16.24")
    
    print("\n⚙️ 微调方案选择:")
    print("   运行环境检查和方案选择:")
    print("   !python run_vavae_finetune.py")
    
    print("\n📋 方案A: 原项目官方3阶段微调 (推荐)")
    print("   命令: !python finetune_vavae_official.py")
    print("   特点:")
    print("   - 基于f16d32_vfdinov2_long.yaml的官方策略")
    print("   - 3阶段训练: 对齐(50) + 重建(15) + 边距(15)")
    print("   - 完整LDM框架: 判别器 + LPIPS + DINOv2对齐")
    print("   - 时间: 4-8小时")
    print("   - 效果: 最佳，经过原项目验证")
    
    print("\n🔧 方案B: 简化版微调 (快速验证)")
    print("   命令: !python finetune_vavae.py")
    print("   特点:")
    print("   - 集成DINOv2对齐的简化实现")
    print("   - 单阶段训练: 80 epochs + 早停")
    print("   - 基于原项目参数的简化版本")
    print("   - 时间: 3-6小时")
    print("   - 效果: 良好，易于调试")
    
    print("\n📊 微调后评估:")
    print("   命令: !python evaluate_finetuned_vae.py")
    print("   功能:")
    print("   - 对比微调前后的MSE和FID")
    print("   - 生成可视化对比图像")
    print("   - 量化改善幅度")
    print("   - 预期改善: MSE 15-30%, FID 20-40%")
    
    print("\n💡 微调建议:")
    print("   1. 如果追求最佳效果且时间充足: 选择方案A")
    print("   2. 如果快速验证或资源受限: 选择方案B")
    print("   3. 两个方案都包含VA-VAE的核心创新 (DINOv2对齐)")
    print("   4. 微调是可选的，当前预训练模型效果已经很好")
    
    print("\n🎯 预期效果:")
    print("   微调成功后，您将获得:")
    print("   - 更适应微多普勒数据的VA-VAE模型")
    print("   - 更好的重建质量 (MSE和FID改善)")
    print("   - 更强的语义保持能力")
    print("   - 为后续任务提供更好的特征表示")

def check_prerequisites():
    """检查微调前置条件"""
    print("\n🔍 检查微调前置条件:")
    print("-" * 40)
    
    # 检查数据
    data_dir = Path("/kaggle/input/dataset")
    if data_dir.exists():
        user_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
        total_images = sum(len(list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))) 
                          for user_dir in user_dirs)
        print(f"✅ 数据检查: {len(user_dirs)} 用户, {total_images} 张图像")
    else:
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 检查模型
    model_path = Path("models/vavae-imagenet256-f16d32-dinov2.pt")
    if model_path.exists():
        print(f"✅ 预训练模型: {model_path}")
    else:
        print(f"❌ 预训练模型不存在: {model_path}")
        print("💡 请先运行: !python step2_download_models.py")
        return False
    
    # 检查依赖
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import timm
        print(f"✅ TIMM (DINOv2支持): {timm.__version__}")
    except ImportError:
        print("⚠️ TIMM未安装，DINOv2对齐将被禁用")
        print("💡 建议运行: !pip install timm")
    
    print("✅ 前置条件检查完成")
    return True

def show_next_steps():
    """显示后续步骤"""
    print("\n🚀 开始VA-VAE微调:")
    print("-" * 40)
    print("1. 运行环境检查: !python run_vavae_finetune.py")
    print("2. 选择微调方案:")
    print("   - 方案A (推荐): !python finetune_vavae_official.py")
    print("   - 方案B (快速): !python finetune_vavae.py")
    print("3. 评估微调效果: !python evaluate_finetuned_vae.py")
    print("4. 根据评估结果决定是否使用微调模型")

def main():
    """主函数"""
    show_vavae_finetune_guide()
    
    # 检查前置条件
    if check_prerequisites():
        show_next_steps()
        print("\n🎉 准备就绪！可以开始VA-VAE微调了")
        return True
    else:
        print("\n❌ 前置条件不满足，请先解决上述问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
