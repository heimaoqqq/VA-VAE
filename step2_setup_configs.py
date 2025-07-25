#!/usr/bin/env python3
"""
步骤2: 设置配置文件
严格按照LightningDiT README和tutorial步骤
"""

import os
import yaml
from pathlib import Path

def create_inference_config():
    """创建推理配置文件 - 基于官方reproductions配置"""
    
    config_path = "inference_config.yaml"
    models_dir = Path("./official_models")
    
    # 基于官方configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml
    config = {
        'ckpt_path': str(models_dir / "lightningdit-xl-imagenet256-800ep.pt"),
        'data': {
            'data_path': str(models_dir / "latents_stats.pt"),
            'fid_reference_file': 'path/to/your/VIRTUAL_imagenet256_labeled.npz',
            'image_size': 256,
            'num_classes': 1000,
            'num_workers': 8,
            'latent_norm': True,
            'latent_multiplier': 1.0
        },
        'vae': {
            'model_name': 'vavae_f16d32',
            'downsample_ratio': 16
        },
        'model': {
            'model_type': 'LightningDiT-XL/1',
            'use_qknorm': False,
            'use_swiglu': True,
            'use_rope': True,
            'use_rmsnorm': True,
            'wo_shift': False,
            'in_chans': 32
        },
        'train': {
            'max_steps': 80000,
            'global_batch_size': 1024,
            'global_seed': 0,  # 这是缺失的关键字段
            'output_dir': 'output',
            'exp_name': 'lightningdit_xl_vavae_f16d32',
            'ckpt': None,
            'log_every': 100,
            'ckpt_every': 20000
        },
        'optimizer': {
            'lr': 0.0002,
            'beta2': 0.95
        },
        'transport': {
            'path_type': 'Linear',
            'prediction': 'velocity',
            'loss_weight': None,
            'sample_eps': None,
            'train_eps': None,
            'use_cosine_loss': True,
            'use_lognorm': True
        },
        'sample': {
            'mode': 'ODE',
            'sampling_method': 'euler',
            'atol': 0.000001,
            'rtol': 0.001,
            'reverse': False,
            'likelihood': False,
            'num_sampling_steps': 250,
            'cfg_scale': 6.7,  # 官方800ep模型使用6.7
            'per_proc_batch_size': 4,
            'cfg_interval_start': 0.125,
            'timestep_shift': 0.3
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ 推理配置文件已创建: {config_path}")
    return config_path

def update_vavae_config():
    """更新VA-VAE配置 - 官方tutorial要求的步骤"""
    
    vavae_config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
    models_dir = Path("./official_models")
    
    if not os.path.exists(vavae_config_path):
        print(f"❌ VA-VAE配置文件不存在: {vavae_config_path}")
        return False
    
    # 读取现有配置
    with open(vavae_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新检查点路径 (官方tutorial步骤)
    old_path = config.get('ckpt_path', 'N/A')
    new_path = str(models_dir / "vavae-imagenet256-f16d32-dinov2.pt")
    config['ckpt_path'] = new_path
    
    # 保存更新后的配置
    with open(vavae_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ VA-VAE配置已更新:")
    print(f"   旧路径: {old_path}")
    print(f"   新路径: {new_path}")
    
    return True

def main():
    """步骤2: 设置配置文件"""
    
    print("⚙️ 步骤2: 设置配置文件")
    print("=" * 50)
    
    # 检查模型文件是否存在
    models_dir = Path("./official_models")
    required_files = [
        "vavae-imagenet256-f16d32-dinov2.pt",
        "lightningdit-xl-imagenet256-800ep.pt",
        "latents_stats.pt"
    ]
    
    missing_files = []
    for filename in required_files:
        if not (models_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print("❌ 缺少模型文件:")
        for filename in missing_files:
            print(f"   - {filename}")
        print("💡 请先运行: python step1_download_models.py")
        return
    
    print("✅ 所有模型文件已存在")
    
    # 创建推理配置文件
    print("\n📝 创建推理配置文件...")
    config_path = create_inference_config()
    
    # 更新VA-VAE配置
    print("\n🔧 更新VA-VAE配置...")
    if update_vavae_config():
        print("\n✅ 步骤2完成！配置文件已设置")
        print(f"📄 推理配置: {config_path}")
        print("📄 VA-VAE配置: LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
        print("\n🎯 下一步: 运行 python step3_run_inference.py")
    else:
        print("\n❌ VA-VAE配置更新失败")

if __name__ == "__main__":
    main()
