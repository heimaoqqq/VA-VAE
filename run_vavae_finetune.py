#!/usr/bin/env python3
"""
专注VA-VAE微调 - 基于原项目官方方案
提供官方3阶段和简化版本两种选择
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查微调环境"""
    print("🔍 检查VA-VAE微调环境...")

    # 检查数据
    data_dir = Path("/kaggle/input/dataset")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    # 统计数据
    user_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
    total_images = 0
    for user_dir in user_dirs:
        images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
        total_images += len(images)

    print(f"✅ 数据检查通过: {len(user_dirs)} 用户, {total_images} 张图像")

    # 检查模型
    vae_model_path = Path("models/vavae-imagenet256-f16d32-dinov2.pt")
    if not vae_model_path.exists():
        print(f"❌ 预训练模型不存在: {vae_model_path}")
        print("💡 请先运行: !python step2_download_models.py")
        return False

    print(f"✅ 预训练模型检查通过: {vae_model_path}")

    # 检查依赖
    try:
        import timm
        print("✅ DINOv2支持可用 (timm已安装)")
        dinov2_available = True
    except ImportError:
        print("⚠️ DINOv2支持不可用 (timm未安装)")
        print("💡 建议运行: !pip install timm")
        dinov2_available = False

    return True, dinov2_available

def show_options():
    """显示微调选项"""
    print("\n🎯 VA-VAE微调方案选择:")
    print("="*50)

    print("📚 方案A: 原项目官方3阶段微调 (推荐)")
    print("   - 基于f16d32_vfdinov2_long.yaml的官方策略")
    print("   - 3阶段训练: 对齐(50) + 重建(15) + 边距(15)")
    print("   - 完整LDM框架: 判别器 + LPIPS + DINOv2对齐")
    print("   - 时间: 4-8小时")
    print("   - 命令: !python finetune_vavae_official.py")

    print("\n🔧 方案B: 简化版微调 (快速)")
    print("   - 集成DINOv2对齐的简化实现")
    print("   - 单阶段训练: 80 epochs + 早停")
    print("   - 基于原项目参数的简化版本")
    print("   - 时间: 3-6小时")
    print("   - 命令: !python finetune_vavae.py")

    print("\n💡 建议:")
    print("   - 如果追求最佳效果: 选择方案A")
    print("   - 如果快速验证: 选择方案B")
    print("   - 两个方案都包含DINOv2对齐 (VA-VAE核心创新)")

def main():
    """主函数"""
    print("🎯 VA-VAE专项微调工具")
    print("="*50)

    # 环境检查
    env_ok, dinov2_available = check_environment()
    if not env_ok:
        return False

    # 显示选项
    show_options()
    
    print("⚙️ 微调配置:")
    print(f"   批次大小: {config['batch_size']}")
    print(f"   阶段1 (解码器): {config['stage1_epochs']} epochs, lr={config['stage1_lr']}")
    print(f"   阶段2 (全模型): {config['stage2_epochs']} epochs, lr={config['stage2_lr']}")
    print(f"   预计时间: 2-5小时")
    
    # 创建微调器
    device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    print(f"🔥 使用设备: {device}")
    
    tuner = VAEFineTuner(vae_model_path, device)
    if tuner.vae is None:
        print("❌ VA-VAE模型加载失败")
        return False
    
    # 开始微调
    try:
        final_model_path = tuner.finetune(data_dir, output_dir, config)
        
        print(f"\n🎉 微调完成！")
        print(f"📁 微调后的模型: {final_model_path}")
        print(f"📊 训练日志: {output_dir}/training_curves.png")
        
        # 建议下一步
        print(f"\n💡 下一步建议:")
        print(f"1. 运行评估脚本验证微调效果:")
        print(f"   !python evaluate_finetuned_vae.py")
        print(f"2. 如果效果满意，进入阶段2 UNet扩散模型训练")
        
        return True
        
    except Exception as e:
        print(f"❌ 微调失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ VA-VAE微调成功完成！")
    else:
        print("\n❌ VA-VAE微调失败！")
    
    sys.exit(0 if success else 1)
