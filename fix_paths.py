#!/usr/bin/env python3
"""
修复路径问题：推理脚本在LightningDiT/目录下运行
需要将模型路径调整为相对于LightningDiT/的路径
"""

import yaml
import os
from pathlib import Path

def fix_paths():
    """修复配置文件中的路径问题"""
    
    print("🔧 修复路径问题")
    print("=" * 40)
    print("问题：推理脚本在LightningDiT/目录下运行")
    print("解决：将路径调整为相对于LightningDiT/的路径")
    
    # 检查配置文件
    config_file = "inference_config.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 读取配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修复路径
    print("\n🔧 修复路径...")
    
    # 修复主配置中的路径
    old_ckpt = config.get('ckpt_path', '')
    old_data = config.get('data', {}).get('data_path', '')
    
    config['ckpt_path'] = '../official_models/lightningdit-xl-imagenet256-800ep.pt'
    config['data']['data_path'] = '../official_models'  # 指向目录，不是文件
    
    print(f"✅ ckpt_path: {old_ckpt} -> {config['ckpt_path']}")
    print(f"✅ data_path: {old_data} -> {config['data']['data_path']}")
    
    # 保存修复后的配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ 配置文件已修复: {config_file}")
    
    # 修复VA-VAE配置
    vavae_config_file = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
    if os.path.exists(vavae_config_file):
        print(f"\n🔧 修复VA-VAE配置: {vavae_config_file}")
        
        with open(vavae_config_file, 'r') as f:
            vavae_config = yaml.safe_load(f)
        
        old_vavae_path = vavae_config.get('ckpt_path', '')
        vavae_config['ckpt_path'] = '../official_models/vavae-imagenet256-f16d32-dinov2.pt'
        
        with open(vavae_config_file, 'w') as f:
            yaml.dump(vavae_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ VA-VAE路径: {old_vavae_path} -> {vavae_config['ckpt_path']}")
    
    return True

def verify_files():
    """验证模型文件是否存在"""
    print("\n📁 验证模型文件...")
    
    models_dir = Path("./official_models")
    required_files = [
        "vavae-imagenet256-f16d32-dinov2.pt",
        "lightningdit-xl-imagenet256-800ep.pt",
        "latents_stats.pt"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024*1024)
            print(f"✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"❌ {filename}: 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("🐛 解决FileNotFoundError路径问题")
    print("=" * 50)
    
    # 验证文件存在
    if not verify_files():
        print("\n❌ 模型文件缺失")
        print("💡 请先运行: python step1_download_models.py")
        return
    
    # 修复路径
    if fix_paths():
        print("\n✅ 路径修复完成！")
        print("🚀 现在可以运行推理: python step3_run_inference.py")
    else:
        print("\n❌ 路径修复失败")

if __name__ == "__main__":
    main()
