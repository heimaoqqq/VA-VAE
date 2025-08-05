#!/usr/bin/env python3
"""
步骤3: 设置LightningDiT推理配置
基于官方reproduction配置，适配Kaggle环境
"""

import yaml
import sys
from pathlib import Path

def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")
    
    # 检查LightningDiT目录
    lightningdit_dir = Path("LightningDiT")
    if not lightningdit_dir.exists():
        print("❌ LightningDiT目录不存在")
        return False
    
    # 检查模型文件
    models_dir = Path("models")
    required_models = [
        "vavae-imagenet256-f16d32-dinov2.pt",
        "lightningdit-xl-imagenet256-800ep.pt",
        "latents_stats.pt"
    ]
    
    for model in required_models:
        model_path = models_dir / model
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        else:
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"✅ {model}: {size_mb:.1f} MB")
    
    print("✅ 前置条件检查通过")
    return True

def setup_inference_config():
    """设置推理配置 - 直接修改官方配置文件"""
    print("\n⚙️ 设置推理配置...")

    # 使用官方reproduction配置作为基础
    lightningdit_dir = Path("LightningDiT")
    original_config_path = lightningdit_dir / "configs" / "reproductions" / "lightningdit_xl_vavae_f16d32_800ep_cfg.yaml"

    if not original_config_path.exists():
        print(f"❌ 官方配置文件不存在: {original_config_path}")
        return False

    print(f"📋 修改官方配置: {original_config_path}")

    # 读取官方配置
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 适配Kaggle环境 - 更新模型路径
    models_dir = Path("models").absolute()

    # 更新模型路径
    config['ckpt_path'] = str(models_dir / "lightningdit-xl-imagenet256-800ep.pt")
    config['data']['data_path'] = str(models_dir / "latents_stats.pt")

    # Kaggle环境优化
    config['sample']['per_proc_batch_size'] = 2  # 降低批次大小适应Kaggle GPU
    config['sample']['num_sampling_steps'] = 50  # 降低采样步数加快推理

    # 直接更新官方配置文件
    with open(original_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ 官方配置已更新: {original_config_path}")

    # 显示更新后的配置
    print("📝 更新后的配置:")
    print(f"   ckpt_path: {config['ckpt_path']}")
    print(f"   data_path: {config['data']['data_path']}")
    print(f"   batch_size: {config['sample']['per_proc_batch_size']}")
    print(f"   sampling_steps: {config['sample']['num_sampling_steps']}")

    return True

def setup_vavae_config():
    """设置VA-VAE配置"""
    print("\n⚙️ 设置VA-VAE配置...")
    
    lightningdit_dir = Path("LightningDiT")
    vavae_config_path = lightningdit_dir / "tokenizer" / "configs" / "vavae_f16d32.yaml"
    
    if not vavae_config_path.exists():
        print(f"❌ VA-VAE配置文件不存在: {vavae_config_path}")
        return False
    
    # 读取配置
    with open(vavae_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新模型路径
    models_dir = Path("models").absolute()
    config['ckpt_path'] = str(models_dir / "vavae-imagenet256-f16d32-dinov2.pt")
    
    # 写回配置
    with open(vavae_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ VA-VAE配置已更新: {vavae_config_path}")
    
    # 显示配置内容
    print("📝 VA-VAE配置内容:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    return True



def verify_configuration():
    """验证配置"""
    print("\n🔍 验证配置...")

    # 检查官方配置文件
    official_config = Path("LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml")
    if official_config.exists():
        print(f"✅ 官方配置文件: 存在")

        # 验证YAML格式和路径
        try:
            with open(official_config, 'r') as f:
                config = yaml.safe_load(f)

            # 检查关键路径
            if 'ckpt_path' in config and config['ckpt_path']:
                print(f"✅ 模型路径: {config['ckpt_path']}")
            else:
                print("❌ 模型路径未设置")
                return False

            if 'data' in config and 'data_path' in config['data'] and config['data']['data_path']:
                print(f"✅ 数据路径: {config['data']['data_path']}")
            else:
                print("❌ 数据路径未设置")
                return False

        except yaml.YAMLError as e:
            print(f"❌ 配置文件YAML格式错误: {e}")
            return False
    else:
        print(f"❌ 官方配置文件不存在: {official_config}")
        return False

    # 检查VA-VAE配置
    vavae_config = Path("LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
    if vavae_config.exists():
        print(f"✅ VA-VAE配置: 存在")
    else:
        print(f"❌ VA-VAE配置不存在: {vavae_config}")
        return False

    # 检查推理脚本
    inference_script = Path("step4_run_inference.py")
    if inference_script.exists():
        print(f"✅ 推理脚本: 存在")
    else:
        print(f"❌ 推理脚本: 不存在")
        return False

    print("✅ 配置验证通过")
    return True

def main():
    """主函数"""
    print("🚀 步骤3: 设置LightningDiT推理配置")
    print("="*60)
    
    # 检查前置条件
    if not check_prerequisites():
        print("❌ 前置条件检查失败")
        return False
    
    # 设置推理配置
    if not setup_inference_config():
        print("❌ 推理配置设置失败")
        return False
    
    # 设置VA-VAE配置
    if not setup_vavae_config():
        print("❌ VA-VAE配置设置失败")
        return False
    
    # 验证配置
    if not verify_configuration():
        print("❌ 配置验证失败")
        return False
    
    print("\n✅ 步骤3完成！配置设置完成")
    print("📋 下一步: !python step4_run_inference.py")
    print("📝 配置文件:")
    print("   - LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml (主配置)")
    print("   - LightningDiT/tokenizer/configs/vavae_f16d32.yaml (VA-VAE配置)")
    print("💡 使用官方accelerate launch + --demo模式进行推理")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
