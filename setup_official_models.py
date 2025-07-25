#!/usr/bin/env python3
"""
严格按照LightningDiT官方README方法
下载预训练模型并运行推理
"""

import os
import requests
import yaml
import subprocess
from pathlib import Path

def download_file(url, local_path):
    """下载文件"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ 下载完成")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    """主函数：按照官方README步骤执行"""

    print("🚀 按照LightningDiT官方README方法")
    print("=" * 50)

    # 步骤1: 下载官方预训练模型
    print("\n📥 步骤1: 下载官方预训练模型")
    models_dir = download_official_models()

    # 步骤2: 修改配置文件 (官方要求)
    print("\n⚙️ 步骤2: 修改配置文件")
    config_path = create_config(models_dir)
    update_vavae_config(models_dir)

    # 步骤3: 运行官方推理脚本
    print("\n🚀 步骤3: 运行官方推理")
    run_official_inference(config_path)

def download_official_models():
    """下载官方预训练模型"""

    models_dir = Path("./official_models")
    models_dir.mkdir(exist_ok=True)

    # 官方README中的下载链接
    models = {
        "VA-VAE": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
        "LightningDiT-XL-800ep": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
        "Latent Statistics": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt"
    }

    for name, url in models.items():
        filename = url.split('/')[-1]
        filepath = models_dir / filename

        if filepath.exists():
            print(f"✅ {name}: 已存在")
        else:
            print(f"📥 下载 {name}...")
            download_file(url, str(filepath))

    return models_dir

def create_config(models_dir):
    """创建配置文件 - 基于官方reproductions配置"""

    config_path = "inference_config.yaml"

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
            'output_dir': 'output',
            'exp_name': 'lightningdit_xl_vavae_f16d32'
        },
        'transport': {
            'path_type': 'Linear',
            'prediction': 'velocity',
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
            'cfg_scale': 6.7,
            'per_proc_batch_size': 4,
            'cfg_interval_start': 0.125,
            'timestep_shift': 0.3
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ 配置文件: {config_path}")
    return config_path

def update_vavae_config(models_dir):
    """更新VA-VAE配置 (官方教程要求)"""

    vavae_config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"

    with open(vavae_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 更新检查点路径
    config['ckpt_path'] = str(models_dir / "vavae-imagenet256-f16d32-dinov2.pt")

    with open(vavae_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ VA-VAE配置已更新")

def run_official_inference(config_path):
    """运行官方推理脚本"""

    print("🚀 运行官方推理脚本")

    # 切换到LightningDiT目录
    os.chdir("LightningDiT")

    # 运行官方命令: bash run_fast_inference.sh config_path
    cmd = f"bash run_fast_inference.sh ../{config_path}"

    print(f"🎯 执行: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("✅ 推理完成!")
        print("📁 输出: demo_images/demo_samples.png")

    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {e}")
        print("💡 请检查环境和依赖")

if __name__ == "__main__":
    main()
