#!/usr/bin/env python3
"""
步骤4: VA-VAE微调
- 基于预训练的vavae-imagenet256-f16d32-dinov2.pt进行微调
- 适配微多普勒时频图数据
- 针对T4×2 GPU优化
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
import argparse

def create_micro_doppler_dataset_class():
    """创建微多普勒数据集类"""
    dataset_code = '''
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class MicroDopplerDataset(Dataset):
    """微多普勒时频图数据集"""
    
    def __init__(self, data_root, size=256, user_conditioning=True):
        self.data_root = data_root
        self.size = size
        self.user_conditioning = user_conditioning
        
        # 加载31个用户的时频图文件
        self.samples = []
        data_path = Path(data_root)
        
        for user_id in range(1, 32):  # 用户1到31
            user_dir = data_path / f"user{user_id}"
            if user_dir.exists():
                for img_file in user_dir.iterdir():
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.samples.append({
                            'path': img_file,
                            'user_id': user_id - 1  # 转换为0-30索引
                        })
        
        print(f"Loaded {len(self.samples)} micro-Doppler samples from {data_root}")
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载时频图 (256x256x3)
        image = Image.open(sample['path']).convert('RGB')
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        result = {'image': image}
        
        if self.user_conditioning:
            result['user_id'] = sample['user_id']
            result['class_label'] = sample['user_id']  # 用于条件生成
            
        return result
    
    def __len__(self):
        return len(self.samples)
'''
    
    # 保存到LightningDiT/vavae/ldm/data/目录
    dataset_file = Path("LightningDiT/vavae/ldm/data/micro_doppler.py")
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write(dataset_code)
    
    print(f"✅ 微多普勒数据集类已创建: {dataset_file}")

def create_vavae_config(dataset_dir, output_dir):
    """创建VA-VAE微调配置文件"""
    
    config = {
        'ckpt_path': str(Path("official_models/vavae-imagenet256-f16d32-dinov2.pt").absolute()),
        'weight_init': str(Path("official_models/vavae-imagenet256-f16d32-dinov2.pt").absolute()),
        
        'model': {
            'base_learning_rate': 1.0e-05,  # 微调学习率
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'monitor': 'val/rec_loss',
                'embed_dim': 32,
                'use_vf': 'dinov2',  # 保持DINOv2特征对齐
                'reverse_proj': True,
                'lossconfig': {
                    'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
                    'params': {
                        'disc_start': 1000,  # 延迟判别器启动
                        'kl_weight': 1.0e-06,
                        'disc_weight': 0.3,  # 降低判别器权重
                        'vf_weight': 0.05,   # 降低视觉特征权重
                        'adaptive_vf': True,
                        'distmat_margin': 0.25,
                        'cos_margin': 0.5,
                    }
                },
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,    # 彩色时频图
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
                'batch_size': 1,  # T4×2 GPU优化
                'wrap': False,    # 不重复数据
                'train': {
                    'target': 'ldm.data.micro_doppler.MicroDopplerDataset',
                    'params': {
                        'data_root': str(Path(dataset_dir) / "train"),
                        'size': 256,
                        'user_conditioning': True
                    }
                },
                'validation': {
                    'target': 'ldm.data.micro_doppler.MicroDopplerDataset',
                    'params': {
                        'data_root': str(Path(dataset_dir) / "val"),
                        'size': 256,
                        'user_conditioning': True
                    }
                }
            }
        },
        
        'lightning': {
            'trainer': {
                'devices': 2,  # T4×2
                'num_nodes': 1,
                'strategy': 'ddp_find_unused_parameters_true',
                'accelerator': 'gpu',
                'max_epochs': 100,  # 微调轮数
                'precision': 16,    # 混合精度
                'check_val_every_n_epoch': 5,
                'log_every_n_steps': 10,
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 4,  # 梯度累积
            }
        }
    }
    
    # 保存配置文件
    config_file = Path("LightningDiT/vavae/configs/micro_doppler_vavae.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ VA-VAE配置文件已创建: {config_file}")
    return config_file

def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")
    
    # 检查预训练模型
    vavae_model = Path("official_models/vavae-imagenet256-f16d32-dinov2.pt")
    if not vavae_model.exists():
        print(f"❌ 预训练VA-VAE模型不存在: {vavae_model}")
        print("💡 请先运行: python step1_download_models.py")
        return False
    
    print(f"✅ 预训练VA-VAE模型: {vavae_model}")
    
    # 检查LightningDiT目录
    vavae_dir = Path("LightningDiT/vavae")
    if not vavae_dir.exists():
        print(f"❌ VA-VAE训练目录不存在: {vavae_dir}")
        return False
    
    print(f"✅ VA-VAE训练目录: {vavae_dir}")
    
    # 检查Taming-Transformers
    taming_dir = Path("taming-transformers")
    if not taming_dir.exists():
        print(f"❌ Taming-Transformers未安装")
        print("💡 请先运行:")
        print("   git clone https://github.com/CompVis/taming-transformers.git")
        print("   cd taming-transformers && pip install -e .")
        return False
    
    print(f"✅ Taming-Transformers: {taming_dir}")
    
    return True

def start_training(config_file):
    """启动VA-VAE微调训练"""
    print("\n🚀 启动VA-VAE微调训练...")
    
    # 切换到vavae目录
    vavae_dir = Path("LightningDiT/vavae")
    original_dir = Path.cwd()
    
    try:
        os.chdir(vavae_dir)
        
        # 构建训练命令
        config_name = config_file.name
        cmd = f"bash run_train.sh configs/{config_name}"
        
        print(f"💻 执行命令: {cmd}")
        print(f"📁 工作目录: {vavae_dir.absolute()}")
        print("\n🔥 开始训练...")
        print("=" * 60)
        
        # 执行训练
        os.system(cmd)
        
    finally:
        # 恢复原目录
        os.chdir(original_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='步骤4: VA-VAE微调')
    parser.add_argument('--dataset_dir', type=str, default='micro_doppler_dataset',
                       help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='vavae_finetuned',
                       help='输出目录')
    parser.add_argument('--dry_run', action='store_true',
                       help='只准备配置，不启动训练')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 步骤4: VA-VAE微调")
    print("=" * 60)
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败")
        return False
    
    # 2. 检查数据集
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"❌ 数据集目录不存在: {args.dataset_dir}")
        print("💡 请先运行: python step3_prepare_micro_doppler_dataset.py")
        return False
    
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    if not train_dir.exists() or not val_dir.exists():
        print("❌ 数据集结构不完整，缺少train或val目录")
        return False
    
    print(f"✅ 数据集检查通过: {args.dataset_dir}")
    
    # 3. 创建微多普勒数据集类
    create_micro_doppler_dataset_class()
    
    # 4. 创建VA-VAE配置
    config_file = create_vavae_config(args.dataset_dir, args.output_dir)
    
    # 5. 启动训练（除非是dry run）
    if args.dry_run:
        print("\n✅ 配置准备完成（dry run模式）")
        print(f"📝 配置文件: {config_file}")
        print("💡 要启动训练，请运行:")
        print(f"   python {sys.argv[0]} --dataset_dir {args.dataset_dir}")
    else:
        start_training(config_file)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
