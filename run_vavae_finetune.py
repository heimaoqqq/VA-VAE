#!/usr/bin/env python3
"""
一键运行VA-VAE微调
"""

import os
import sys
from pathlib import Path
from finetune_vavae import VAEFineTuner

def main():
    """主函数"""
    print("🚀 一键VA-VAE微调")
    print("="*50)
    
    # 检查环境
    data_dir = "/kaggle/input/dataset"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    output_dir = "vavae_finetuned"
    
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    if not Path(vae_model_path).exists():
        print(f"❌ 模型文件不存在: {vae_model_path}")
        print("💡 请先运行 step2_download_models.py")
        return False
    
    # 微调配置
    config = {
        'batch_size': 4,        # 适合Kaggle GPU内存
        'stage1_epochs': 2,     # 解码器微调
        'stage1_lr': 5e-5,      # 较大学习率
        'stage2_epochs': 3,     # 全模型微调
        'stage2_lr': 1e-5,      # 较小学习率
    }
    
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
