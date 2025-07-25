#!/usr/bin/env python3
"""
修复配置文件：添加缺失的global_seed等字段
解决KeyError: 'global_seed'问题
"""

import yaml
from pathlib import Path

def fix_config():
    """修复配置文件，添加缺失字段"""
    
    print("🔧 修复配置文件：添加缺失的global_seed字段")
    print("=" * 50)
    
    config_path = "inference_config.yaml"
    models_dir = Path("./official_models")
    
    # 完全按照官方configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml
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
            'global_seed': 0,  # 这是缺失的关键字段！
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
    
    # 保存修复后的配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ 配置文件已修复: {config_path}")
    print("🔑 添加了缺失的字段:")
    print("   - train.global_seed: 0")
    print("   - train.max_steps: 80000")
    print("   - train.global_batch_size: 1024")
    print("   - optimizer.lr: 0.0002")
    print("   - optimizer.beta2: 0.95")
    print("   - transport.loss_weight: null")
    print("   - transport.sample_eps: null")
    print("   - transport.train_eps: null")
    
    return config_path

def main():
    """主函数"""
    print("🐛 修复KeyError: 'global_seed'问题")
    print("=" * 40)
    
    config_path = fix_config()
    
    print(f"\n✅ 配置修复完成！")
    print(f"📄 配置文件: {config_path}")
    print("\n🚀 现在可以重新运行推理:")
    print("   python step3_run_inference.py")

if __name__ == "__main__":
    main()
