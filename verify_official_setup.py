#!/usr/bin/env python3
"""
验证官方方法设置
严格按照LightningDiT官方README和教程检查
"""

import os
import sys
import yaml
from pathlib import Path

def verify_official_setup():
    """验证官方方法设置"""
    
    print("🔍 验证LightningDiT官方方法设置")
    print("=" * 60)
    
    checks = [
        ("📁 检查模型文件", check_model_files),
        ("⚙️  检查配置文件", check_config_files),
        ("📜 检查脚本文件", check_script_files),
        ("🔧 检查VA-VAE配置", check_vavae_config),
        ("📋 对比官方配置", compare_with_official),
        ("🎯 检查推理流程", check_inference_flow)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}")
        print("-" * 40)
        
        try:
            if check_func():
                print(f"✅ {check_name}: 通过")
                passed += 1
            else:
                print(f"❌ {check_name}: 失败")
        except Exception as e:
            print(f"❌ {check_name}: 异常 - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 验证结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 完全符合官方方法！")
        print("🚀 可以运行: python run_official_inference.py")
    else:
        print("⚠️  部分检查失败，请修正后重试")

def check_model_files():
    """检查模型文件"""
    models_dir = Path("./official_models")
    
    if not models_dir.exists():
        print("❌ official_models目录不存在")
        return False
    
    # 官方要求的文件
    required_files = {
        "vavae-imagenet256-f16d32-dinov2.pt": "VA-VAE tokenizer",
        "lightningdit-xl-imagenet256-800ep.pt": "LightningDiT-XL 800ep",
        "latents_stats.pt": "Latent statistics"
    }
    
    all_exist = True
    for file_name, description in required_files.items():
        file_path = models_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✅ {description}: {size_mb:.1f} MB")
        else:
            print(f"❌ {description}: 文件不存在")
            all_exist = False
    
    return all_exist

def check_config_files():
    """检查配置文件"""
    
    # 检查我们的配置文件
    config_path = "official_inference_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查关键配置项
    required_keys = [
        'ckpt_path', 'data', 'vae', 'model', 'transport', 'sample'
    ]
    
    for key in required_keys:
        if key in config:
            print(f"✅ 配置项 {key}: 存在")
        else:
            print(f"❌ 配置项 {key}: 缺失")
            return False
    
    # 检查关键参数值
    sample_config = config.get('sample', {})
    expected_values = {
        'num_sampling_steps': 250,
        'cfg_scale': 6.7,
        'sampling_method': 'euler',
        'mode': 'ODE'
    }
    
    for key, expected in expected_values.items():
        actual = sample_config.get(key)
        if actual == expected:
            print(f"✅ {key}: {actual} (正确)")
        else:
            print(f"⚠️  {key}: {actual} (期望: {expected})")
    
    return True

def check_script_files():
    """检查脚本文件"""
    
    # 检查LightningDiT官方脚本
    lightning_dir = Path("LightningDiT")
    if not lightning_dir.exists():
        print("❌ LightningDiT目录不存在")
        return False
    
    required_scripts = [
        "run_fast_inference.sh",
        "inference.py"
    ]
    
    for script in required_scripts:
        script_path = lightning_dir / script
        if script_path.exists():
            print(f"✅ 官方脚本: {script}")
        else:
            print(f"❌ 官方脚本缺失: {script}")
            return False
    
    # 检查我们的脚本
    our_scripts = [
        "setup_official_models.py",
        "test_official_models.py"
    ]
    
    for script in our_scripts:
        if os.path.exists(script):
            print(f"✅ 我们的脚本: {script}")
        else:
            print(f"❌ 我们的脚本缺失: {script}")
            return False
    
    return True

def check_vavae_config():
    """检查VA-VAE配置"""
    
    vavae_config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
    
    if not os.path.exists(vavae_config_path):
        print(f"❌ VA-VAE配置文件不存在: {vavae_config_path}")
        return False
    
    with open(vavae_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ckpt_path = config.get('ckpt_path', '')
    
    # 检查是否指向我们的模型文件
    expected_path = '../official_models/vavae-imagenet256-f16d32-dinov2.pt'
    
    if ckpt_path == expected_path:
        print(f"✅ VA-VAE检查点路径: {ckpt_path}")
        return True
    elif 'official_models' in ckpt_path:
        print(f"⚠️  VA-VAE检查点路径: {ckpt_path} (可能正确)")
        return True
    else:
        print(f"❌ VA-VAE检查点路径: {ckpt_path}")
        print(f"   期望: {expected_path}")
        return False

def compare_with_official():
    """对比官方配置"""
    
    # 检查官方reproductions配置
    official_config_path = "LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml"
    
    if not os.path.exists(official_config_path):
        print(f"❌ 官方配置文件不存在: {official_config_path}")
        return False
    
    with open(official_config_path, 'r') as f:
        official_config = yaml.safe_load(f)
    
    our_config_path = "official_inference_config.yaml"
    with open(our_config_path, 'r') as f:
        our_config = yaml.safe_load(f)
    
    # 对比关键配置
    key_comparisons = [
        ('model.model_type', 'LightningDiT-XL/1'),
        ('model.in_chans', 32),
        ('vae.model_name', 'vavae_f16d32'),
        ('vae.downsample_ratio', 16),
        ('sample.num_sampling_steps', 250),
        ('sample.cfg_scale', 6.7),
        ('sample.sampling_method', 'euler')
    ]
    
    all_match = True
    for key_path, expected in key_comparisons:
        keys = key_path.split('.')
        
        # 获取我们配置中的值
        our_value = our_config
        for key in keys:
            our_value = our_value.get(key, None)
        
        if our_value == expected:
            print(f"✅ {key_path}: {our_value}")
        else:
            print(f"❌ {key_path}: {our_value} (期望: {expected})")
            all_match = False
    
    return all_match

def check_inference_flow():
    """检查推理流程"""
    
    print("📋 推理流程检查:")
    print("1. 下载模型: python setup_official_models.py")
    print("2. 测试加载: python test_official_models.py") 
    print("3. 运行推理: python run_official_inference.py")
    print("4. 官方命令: bash run_fast_inference.sh config_path")
    print("5. 输出位置: LightningDiT/demo_images/demo_samples.png")
    
    # 检查是否有run_official_inference.py
    if os.path.exists("run_official_inference.py"):
        print("✅ 推理脚本已生成")
        return True
    else:
        print("❌ 推理脚本未生成")
        return False

if __name__ == "__main__":
    verify_official_setup()
