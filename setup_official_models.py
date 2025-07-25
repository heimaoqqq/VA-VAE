#!/usr/bin/env python3
"""
按照LightningDiT官方方法设置预训练模型
下载官方预训练权重并配置推理环境
"""

import os
import sys
import requests
import yaml
from pathlib import Path

def download_file(url, local_path, description=""):
    """下载文件"""
    print(f"📥 下载 {description}: {url}")
    
    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 检查文件是否已存在
    if os.path.exists(local_path):
        print(f"✅ 文件已存在: {local_path}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def setup_official_models():
    """设置官方预训练模型"""
    
    print("🚀 设置LightningDiT官方预训练模型")
    print("=" * 50)
    
    # 创建模型目录
    models_dir = Path("./official_models")
    models_dir.mkdir(exist_ok=True)
    
    # 官方模型下载链接
    models = {
        "VA-VAE": {
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
            "path": models_dir / "vavae-imagenet256-f16d32-dinov2.pt"
        },
        "LightningDiT-XL-800ep": {
            "url": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
            "path": models_dir / "lightningdit-xl-imagenet256-800ep.pt"
        },
        "Latent Statistics": {
            "url": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
            "path": models_dir / "latents_stats.pt"
        }
    }
    
    # 下载模型
    success_count = 0
    for name, info in models.items():
        if download_file(info["url"], str(info["path"]), name):
            success_count += 1
    
    print(f"\n📊 下载结果: {success_count}/{len(models)} 个文件成功")
    
    # 创建配置文件
    config_path = create_inference_config(models_dir)
    
    # 创建推理脚本
    create_inference_script(config_path)
    
    print("\n✅ 设置完成!")
    print(f"📁 模型文件位置: {models_dir.absolute()}")
    print(f"⚙️  配置文件: {config_path}")
    print(f"🚀 运行推理: python run_official_inference.py")

def create_inference_config(models_dir):
    """创建推理配置文件"""
    
    config_path = "official_inference_config.yaml"
    
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
            'cfg_scale': 6.7,
            'per_proc_batch_size': 4,
            'fid_num': 50000,
            'cfg_interval_start': 0.125,
            'timestep_shift': 0.3
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ 配置文件已创建: {config_path}")
    return config_path

def create_inference_script(config_path):
    """创建推理脚本"""
    
    script_content = f'''#!/usr/bin/env python3
"""
使用官方预训练模型进行推理
"""

import os
import sys
import subprocess

def main():
    print("🚀 使用LightningDiT官方预训练模型进行推理")
    
    # 检查环境
    try:
        import torch
        print(f"✅ PyTorch: {{torch.__version__}}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 切换到LightningDiT目录
    os.chdir("LightningDiT")
    
    # 更新VA-VAE配置
    vavae_config = "tokenizer/configs/vavae_f16d32.yaml"
    update_vavae_config(vavae_config)
    
    # 运行官方推理脚本
    config_path = "../{config_path}"
    cmd = f"bash run_fast_inference.sh {{config_path}}"
    
    print(f"🎯 执行命令: {{cmd}}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ 推理完成!")
        print("📁 生成的图像保存在: demo_images/demo_samples.png")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {{e}}")
        print(f"错误输出: {{e.stderr}}")

def update_vavae_config(config_path):
    """更新VA-VAE配置文件"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新检查点路径
    config['ckpt_path'] = '../official_models/vavae-imagenet256-f16d32-dinov2.pt'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ 已更新VA-VAE配置: {{config_path}}")

if __name__ == "__main__":
    main()
'''
    
    with open("run_official_inference.py", 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("run_official_inference.py", 0o755)
    
    print("✅ 推理脚本已创建: run_official_inference.py")

if __name__ == "__main__":
    setup_official_models()
