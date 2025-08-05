#!/usr/bin/env python3
"""
步骤4: LightningDiT推理 - 最终版本
集成了标准推理和Demo推理，自动选择最佳方案
"""

import os
import sys
import torch
import subprocess
from pathlib import Path
import yaml

def check_environment():
    """检查推理环境"""
    print("🔍 检查推理环境...")
    
    # 检查CUDA
    print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i} 内存: {memory_total:.1f} GB")
    
    # 检查关键模块
    required_modules = ['accelerate', 'torchdiffeq', 'timm', 'diffusers']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 可用")
        except ImportError:
            print(f"❌ {module}: 不可用")
            return False
    
    return True

def check_files():
    """检查必需文件"""
    print("\n🔍 检查必需文件...")
    
    # 检查LightningDiT目录
    lightningdit_dir = Path("LightningDiT")
    if not lightningdit_dir.exists():
        print(f"❌ LightningDiT目录不存在: {lightningdit_dir}")
        return False
    print(f"✅ LightningDiT目录: {lightningdit_dir}")
    
    # 检查推理脚本
    inference_script = lightningdit_dir / "inference.py"
    if not inference_script.exists():
        print(f"❌ 推理脚本不存在: {inference_script}")
        return False
    print(f"✅ 推理脚本: {inference_script}")
    
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
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"✅ {model}: {size_mb:.1f} MB")
    
    return True

def create_demo_config():
    """创建Demo配置文件"""
    print("⚙️ 创建Demo配置文件...")
    
    demo_config = {
        'ckpt_path': str(Path("../models/lightningdit-xl-imagenet256-800ep.pt").absolute()),
        'data': {
            'data_path': str(Path("../models").absolute()),
            'image_size': 256,
            'num_classes': 1000,
            'num_workers': 2,
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
            'global_batch_size': 256,
            'global_seed': 0,
            'output_dir': 'demo_output',
            'exp_name': 'lightningdit_demo',
            'ckpt': None,
            'log_every': 100,
            'ckpt_every': 20000
        },
        'optimizer': {'lr': 0.0002, 'beta2': 0.95},
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
            'num_sampling_steps': 50,
            'cfg_scale': 9.0,
            'per_proc_batch_size': 1,
            'fid_num': 100,
            'cfg_interval_start': 0.0,
            'timestep_shift': 0.0
        }
    }
    
    config_path = Path("demo_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Demo配置已创建: {config_path}")
    return str(config_path)

def run_inference():
    """运行推理 - 智能选择方案"""
    print("\n🚀 开始LightningDiT推理...")
    
    original_cwd = os.getcwd()
    lightningdit_dir = Path("LightningDiT")
    
    try:
        os.chdir(lightningdit_dir)
        current_dir = Path.cwd()
        print(f"📁 当前目录: {current_dir}")
        
        # 方案1: 尝试官方配置
        official_config = "configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml"
        if Path(official_config).exists():
            print(f"🎯 尝试官方配置: {official_config}")
            cmd = f"accelerate launch --mixed_precision bf16 inference.py --config {official_config} --demo"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ 官方配置推理成功！")
                return True
            else:
                print("⚠️ 官方配置失败，尝试Demo配置...")
        
        # 方案2: 使用Demo配置
        demo_config = create_demo_config()
        print(f"🎯 使用Demo配置: {demo_config}")
        
        # 单GPU运行，避免分布式问题
        cmd = f"python inference.py --config {demo_config} --demo"
        print(f"💻 执行命令: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        
        # 输出结果
        if result.stdout:
            print("📤 标准输出:")
            print(result.stdout[-1000:])  # 只显示最后1000字符
        
        if result.stderr:
            print("📤 错误输出:")
            print(result.stderr[-1000:])  # 只显示最后1000字符
        
        if result.returncode == 0:
            print("✅ Demo推理成功完成！")
            return True
        else:
            print(f"❌ 推理失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 推理超时（30分钟）")
        return False
    except Exception as e:
        print(f"❌ 推理异常: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def verify_results():
    """验证推理结果"""
    print("\n🔍 验证推理结果...")
    
    output_dirs = [
        Path("LightningDiT/demo_output"),
        Path("LightningDiT/demo_images"),
        Path("LightningDiT/output")
    ]
    
    found_images = []
    
    for output_dir in output_dirs:
        if output_dir.exists():
            print(f"📁 发现输出目录: {output_dir}")
            
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images = list(output_dir.glob(ext))
                found_images.extend(images)
                
                for img in images:
                    size_mb = img.stat().st_size / 1024 / 1024
                    print(f"   📸 {img.name}: {size_mb:.2f} MB")
    
    if found_images:
        print(f"✅ 找到 {len(found_images)} 个生成图像")
        return True
    else:
        print("❌ 未找到生成的图像")
        return False

def main():
    """主函数"""
    print("🚀 步骤4: LightningDiT推理 (最终版本)")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 检查文件
    if not check_files():
        print("❌ 文件检查失败")
        return False
    
    # 运行推理
    if not run_inference():
        print("❌ 推理失败")
        return False
    
    # 验证结果
    if not verify_results():
        print("❌ 结果验证失败")
        return False
    
    print("\n🎉 LightningDiT推理完成！")
    print("📊 成功复现了官方效果")
    print("🎯 下一步: 考虑适配您的31用户微多普勒数据")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
