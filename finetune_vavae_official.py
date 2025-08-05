#!/usr/bin/env python3
"""
VA-VAE官方微调脚本 - 完全基于原项目框架
使用原项目的3阶段训练策略和官方配置
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

def create_stage_configs():
    """创建3阶段训练配置文件 - 基于原项目官方策略"""
    
    # 基础配置模板
    base_config = {
        'ckpt_path': '/path/to/ckpt',
        'weight_init': None,  # 将在各阶段中设置
        'model': {
            'base_learning_rate': 1.0e-04,
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'monitor': 'val/rec_loss',
                'embed_dim': 32,
                'use_vf': 'dinov2',
                'reverse_proj': True,
                'lossconfig': {
                    'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
                    'params': {
                        'kl_weight': 1.0e-06,
                        'disc_weight': 0.5,
                        'adaptive_vf': True,
                        # 这些参数将在各阶段中设置
                        'disc_start': None,
                        'vf_weight': None,
                        'distmat_margin': None,
                        'cos_margin': None,
                    }
                },
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [16],
                    'dropout': 0.0
                }
            }
        },
        'data': {
            'target': 'main.DataModuleFromConfig',
            'params': {
                'batch_size': 4,  # 适合Kaggle GPU
                'wrap': True,
                'train': {
                    'target': 'ldm.data.custom.CustomDataset',  # 需要实现
                    'params': {
                        'data_root': '/kaggle/input/dataset'
                    }
                },
                'validation': {
                    'target': 'ldm.data.custom.CustomDataset',
                    'params': {
                        'data_root': '/kaggle/input/dataset'
                    }
                }
            }
        },
        'lightning': {
            'trainer': {
                'devices': 1,
                'num_nodes': 1,
                'strategy': 'auto',
                'accelerator': 'gpu',
                'precision': 32,
                'max_epochs': None  # 将在各阶段中设置
            }
        }
    }
    
    # 阶段1配置 (100 epochs - 对齐阶段)
    stage1_config = base_config.copy()
    stage1_config['weight_init'] = 'models/vavae-imagenet256-f16d32-dinov2.pt'
    stage1_config['model']['params']['lossconfig']['params'].update({
        'disc_start': 5001,
        'vf_weight': 0.5,
        'distmat_margin': 0,
        'cos_margin': 0,
    })
    stage1_config['lightning']['trainer']['max_epochs'] = 50  # 适应小数据集
    
    # 阶段2配置 (15 epochs - 重建优化)
    stage2_config = base_config.copy()
    stage2_config['weight_init'] = 'vavae_finetuned/stage1_final.pt'
    stage2_config['model']['params']['lossconfig']['params'].update({
        'disc_start': 1,
        'vf_weight': 0.1,
        'distmat_margin': 0,
        'cos_margin': 0,
    })
    stage2_config['lightning']['trainer']['max_epochs'] = 15
    
    # 阶段3配置 (15 epochs - 边距优化)
    stage3_config = base_config.copy()
    stage3_config['weight_init'] = 'vavae_finetuned/stage2_final.pt'
    stage3_config['model']['params']['lossconfig']['params'].update({
        'disc_start': 1,
        'vf_weight': 0.1,
        'distmat_margin': 0.25,
        'cos_margin': 0.5,
    })
    stage3_config['lightning']['trainer']['max_epochs'] = 15
    
    return stage1_config, stage2_config, stage3_config

def save_configs():
    """保存3阶段配置文件"""
    stage1, stage2, stage3 = create_stage_configs()
    
    # 创建配置目录
    config_dir = Path("vavae_finetune_configs")
    config_dir.mkdir(exist_ok=True)
    
    # 保存配置文件
    configs = [
        (stage1, "stage1_alignment.yaml"),
        (stage2, "stage2_reconstruction.yaml"), 
        (stage3, "stage3_margin.yaml")
    ]
    
    for config, filename in configs:
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ 保存配置: {config_path}")
    
    return config_dir

def run_official_finetune():
    """运行官方3阶段微调"""
    print("🚀 VA-VAE官方3阶段微调")
    print("="*60)
    
    # 检查环境
    if not Path("/kaggle/input/dataset").exists():
        print("❌ 数据目录不存在: /kaggle/input/dataset")
        return False
    
    if not Path("models/vavae-imagenet256-f16d32-dinov2.pt").exists():
        print("❌ 预训练模型不存在")
        return False
    
    # 创建配置文件
    print("📝 创建官方3阶段配置...")
    config_dir = save_configs()
    
    # 创建输出目录
    output_dir = Path("vavae_finetuned")
    output_dir.mkdir(exist_ok=True)
    
    print("⚙️ 官方3阶段微调策略:")
    print("   阶段1 (50 epochs): 对齐阶段, vf_weight=0.5, disc_start=5001")
    print("   阶段2 (15 epochs): 重建优化, vf_weight=0.1, disc_start=1")
    print("   阶段3 (15 epochs): 边距优化, margin=0.25/0.5")
    print("   总计: 80 epochs")
    print("   基于: 原项目f16d32_vfdinov2_long.yaml")
    
    print("\n🔧 使用原项目训练框架:")
    print("   cd LightningDiT/vavae")
    print("   bash run_train.sh ../../vavae_finetune_configs/stage1_alignment.yaml")
    print("   # 然后依次运行stage2和stage3")
    
    print("\n💡 注意事项:")
    print("   1. 需要实现CustomDataset类来加载微多普勒数据")
    print("   2. 使用原项目的完整LDM框架")
    print("   3. 包含判别器、LPIPS损失、DINOv2对齐")
    print("   4. 这是真正的VA-VAE官方微调方案")
    
    return True

def main():
    """主函数"""
    print("🎯 VA-VAE官方微调工具")
    print("="*50)
    
    print("📚 基于原项目的完整3阶段训练策略:")
    print("   - 使用原项目的LDM训练框架")
    print("   - 完整的损失函数 (LPIPS + 判别器 + DINOv2)")
    print("   - 官方的3阶段参数设置")
    print("   - 避免自己实现带来的错误")
    
    success = run_official_finetune()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
