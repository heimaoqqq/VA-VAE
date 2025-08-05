#!/usr/bin/env python3
"""
步骤4: LightningDiT Demo推理 - 简化版本
专门为Kaggle环境设计，避免数据集路径问题
"""

import os
import sys
import torch
import subprocess
from pathlib import Path
import yaml

def create_demo_config():
    """创建专门的demo配置文件"""
    print("⚙️ 创建demo专用配置...")
    
    # 基础配置
    demo_config = {
        'ckpt_path': str(Path("../models/lightningdit-xl-imagenet256-800ep.pt").absolute()),
        
        'data': {
            'data_path': str(Path("../models").absolute()),  # 指向models目录
            'image_size': 256,
            'num_classes': 1000,
            'num_workers': 2,  # 降低worker数量
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
            'global_batch_size': 256,  # 降低批次大小
            'global_seed': 0,
            'output_dir': 'demo_output',
            'exp_name': 'lightningdit_demo',
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
            'num_sampling_steps': 50,  # 降低采样步数
            'cfg_scale': 9.0,  # demo模式固定值
            'per_proc_batch_size': 1,  # 最小批次大小
            'fid_num': 100,  # 降低生成数量
            'cfg_interval_start': 0.0,  # demo模式简化
            'timestep_shift': 0.0
        }
    }
    
    # 保存配置
    config_path = Path("demo_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Demo配置已创建: {config_path}")
    return str(config_path)

def run_simple_demo():
    """运行简化的demo推理"""
    print("\n🚀 开始简化Demo推理...")
    
    # 切换到LightningDiT目录
    original_cwd = os.getcwd()
    lightningdit_dir = Path("LightningDiT")
    
    try:
        os.chdir(lightningdit_dir)
        current_dir = Path.cwd()
        print(f"📁 当前目录: {current_dir}")
        
        # 创建demo配置
        config_path = create_demo_config()
        
        # 验证关键文件
        model_path = Path("../models/lightningdit-xl-imagenet256-800ep.pt")
        vae_path = Path("../models/vavae-imagenet256-f16d32-dinov2.pt")
        stats_path = Path("../models/latents_stats.pt")
        
        print(f"🔍 验证文件:")
        print(f"   模型: {model_path.exists()} - {model_path}")
        print(f"   VAE: {vae_path.exists()} - {vae_path}")
        print(f"   统计: {stats_path.exists()} - {stats_path}")
        
        if not all([model_path.exists(), vae_path.exists(), stats_path.exists()]):
            print("❌ 关键文件缺失")
            return False
        
        # 使用单GPU运行，避免分布式问题
        cmd = f"python inference.py --config {config_path} --demo"
        print(f"💻 执行命令: {cmd}")
        
        # 运行推理
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=1800  # 30分钟超时
        )
        
        # 输出结果
        if result.stdout:
            print("📤 标准输出:")
            print(result.stdout)
        
        if result.stderr:
            print("📤 错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Demo推理成功完成！")
            return True
        else:
            print(f"❌ Demo推理失败，返回码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 推理超时（30分钟）")
        return False
    except Exception as e:
        print(f"❌ 推理异常: {e}")
        return False
    finally:
        # 恢复原始目录
        os.chdir(original_cwd)

def check_environment():
    """检查环境"""
    print("🔍 检查Demo推理环境...")
    
    # 检查CUDA
    print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU数量: {torch.cuda.device_count()}")
    
    # 检查关键目录
    dirs_to_check = [
        Path("LightningDiT"),
        Path("models"),
        Path("LightningDiT/inference.py")
    ]
    
    for path in dirs_to_check:
        if path.exists():
            print(f"✅ {path}: 存在")
        else:
            print(f"❌ {path}: 不存在")
            return False
    
    return True

def verify_results():
    """验证结果"""
    print("\n🔍 验证Demo推理结果...")
    
    # 检查可能的输出目录
    output_dirs = [
        Path("LightningDiT/demo_output"),
        Path("LightningDiT/demo_images"),
        Path("LightningDiT/output")
    ]
    
    found_images = []
    
    for output_dir in output_dirs:
        if output_dir.exists():
            print(f"📁 发现输出目录: {output_dir}")
            
            # 查找图像文件
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
    print("🚀 步骤4: LightningDiT Demo推理 (简化版)")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 运行demo推理
    if not run_simple_demo():
        print("❌ Demo推理失败")
        return False
    
    # 验证结果
    if not verify_results():
        print("❌ 结果验证失败")
        return False
    
    print("\n🎉 Demo推理完成！")
    print("💡 这是简化版本，避免了复杂的数据集配置问题")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
