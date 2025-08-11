#!/usr/bin/env python3
"""
Step 4: VA-VAE 微多普勒微调训练
基于LightningDiT原项目的完整实现
包含三阶段训练策略和Vision Foundation对齐
"""

import os
import sys
import argparse
from pathlib import Path
import json
import yaml
from datetime import datetime

# 添加LightningDiT路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))  # 添加根目录以导入自定义数据集

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
from main import DataModuleFromConfig  # 从原项目正确导入


class MicroDopplerDataset(Dataset):
    """微多普勒数据集 - 兼容原项目格式"""
    
    def __init__(self, data_root, split_file, split='train', image_size=256):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        
        # 加载数据划分
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.samples = []
        data_list = split_data['train'] if split == 'train' else split_data['val']
        
        for item in data_list:
            img_path = self.data_root / item['path']
            if img_path.exists():
                self.samples.append({
                    'path': img_path,
                    'user_id': item['user_id']
                })
        
        print(f"✅ {split}集: {len(self.samples)} 张图像")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        img = Image.open(sample['path']).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1, 1] (原项目标准)
        img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # 返回原项目格式
        return {'image': img_tensor}


# 注意：原VA-VAE不包含用户对比损失
# VF对齐机制（DINOv2）已经提供了足够的语义区分能力
# 添加额外的用户对比损失可能会干扰原始训练目标


def get_training_strategy(args):
    """根据GPU配置选择训练策略"""
    if not torch.cuda.is_available():
        return 'auto'
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        return 'auto'
    elif num_gpus == 2 and args.kaggle_t4:
        # Kaggle T4×2特殊配置
        print("🔧 使用Kaggle T4×2 DDP策略")
        return DDPStrategy(
            find_unused_parameters=True,
            static_graph=False,  # T4可能需要动态图
            gradient_as_bucket_view=True
        )
    else:
        # 通用多GPU配置
        return 'ddp_find_unused_parameters_true'


def create_stage_config(args, stage, checkpoint_path=None):
    """创建阶段配置 - 完全兼容原项目"""
    
    # 基于原项目的三阶段参数
    stage_params = {
        1: {  # 语义对齐阶段
            'disc_start': 5001,  # 原项目默认值，延迟判别器启动
            'disc_weight': 0.5,  # 原项目默认值
            'vf_weight': 0.5,  # 高权重进行语义对齐
            'distmat_margin': 0.0,
            'cos_margin': 0.0,
            'learning_rate': 1e-4,
            'max_epochs': 30  # 适应小数据集
        },
        2: {  # 重建优化阶段
            'disc_start': 1,  # 启用判别器
            'disc_weight': 0.5,  # 原项目默认值
            'vf_weight': 0.1,  # 降低VF权重
            'distmat_margin': 0.0,
            'cos_margin': 0.0,
            'learning_rate': 5e-5,
            'max_epochs': 15
        },
        3: {  # 边距优化阶段
            'disc_start': 1,
            'disc_weight': 0.5,  # 原项目默认值
            'vf_weight': 0.1,
            'distmat_margin': 0.25,  # 原项目默认值
            'cos_margin': 0.5,  # 原项目默认值
            'learning_rate': 2e-5,
            'max_epochs': 10
        }
    }
    
    params = stage_params[stage]
    
    config = OmegaConf.create({
        'model': {
            'base_learning_rate': params['learning_rate'],
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'monitor': 'val/rec_loss',
                'embed_dim': 32,
                'ckpt_path': args.pretrained_path if stage == 1 else checkpoint_path,
                
                # Vision Foundation配置 - 原项目核心
                'use_vf': 'dinov2',
                'reverse_proj': True,  # 32D -> 1024D投影
                
                # 架构配置 - 与原项目一致
                'ddconfig': {
                    'double_z': True,  # KL-VAE需要
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [16],
                    'dropout': 0.0
                },
                
                # 损失配置 - 原项目核心
                'lossconfig': {
                    'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                    'params': {
                        # 判别器参数 - 与原项目完全一致
                        'disc_start': params['disc_start'],
                        'disc_num_layers': 3,
                        'disc_weight': params['disc_weight'],  # 使用阶段特定值
                        'disc_factor': 1.0,
                        'disc_in_channels': 3,
                        'disc_conditional': False,
                        'disc_loss': 'hinge',  # 原项目默认
                        
                        # 重建损失 - 与原项目一致
                        'pixelloss_weight': 1.0,
                        'perceptual_weight': 0.0,  # 原项目VA-VAE不用感知损失
                        'kl_weight': 1e-6,  # 原项目值
                        'logvar_init': 0.0,  # 原项目默认
                        
                        # VF对齐损失 - 原项目核心参数
                        'vf_weight': params['vf_weight'],
                        'adaptive_vf': True,  # 自适应权重平衡
                        'distmat_weight': 1.0,  # 距离矩阵权重
                        'cos_weight': 1.0,  # 余弦相似度权重
                        'distmat_margin': params['distmat_margin'],
                        'cos_margin': params['cos_margin'],
                        'pp_style': False,  # 原项目默认
                        'use_actnorm': False  # 原项目默认
                    }
                }
            }
        },
        
        'data': {
            'target': 'main.DataModuleFromConfig',
            'params': {
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'wrap': False,  # 原项目参数
                'train': {
                    'target': 'microdoppler_finetune.step4_train_vavae.MicroDopplerDataset',
                    'params': {
                        'data_root': args.data_root,
                        'split_file': args.split_file,
                        'split': 'train'
                    }
                },
                'validation': {
                    'target': 'microdoppler_finetune.step4_train_vavae.MicroDopplerDataset',
                    'params': {
                        'data_root': args.data_root,
                        'split_file': args.split_file,
                        'split': 'val'
                    }
                }
            }
        },
        
        'lightning': {
            'trainer': {
                'devices': args.devices if args.devices else 'auto',
                'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                'max_epochs': params['max_epochs'],
                'precision': 32,  # 原项目使用32位，避免FP16的NaN问题
                'strategy': get_training_strategy(args),  # 根据GPU配置选择策略
                'accumulate_grad_batches': args.gradient_accumulation,
                'gradient_clip_val': 0.5,  # 更保守的梯度裁剪
                'log_every_n_steps': 10,
                'val_check_interval': 0.5,  # 减少验证频率以加速训练
                'num_sanity_val_steps': 0,
                'detect_anomaly': args.detect_anomaly  # 调试NaN问题
            }
        }
    })
    
    return config


def train_stage(args, stage):
    """训练单个阶段"""
    
    print(f"\n{'='*60}")
    print(f"🚀 VA-VAE 第{stage}阶段训练")
    print(f"{'='*60}")
    
    # 设置随机种子
    seed_everything(args.seed, workers=True)
    
    # 获取上一阶段checkpoint
    checkpoint_path = None
    if stage > 1:
        prev_ckpt_dir = Path(f'checkpoints/stage{stage-1}')
        if prev_ckpt_dir.exists():
            # 查找最新的checkpoint
            ckpt_files = list(prev_ckpt_dir.glob('*.ckpt'))
            if ckpt_files:
                checkpoint_path = str(max(ckpt_files, key=lambda x: x.stat().st_mtime))
                print(f"📦 加载第{stage-1}阶段checkpoint: {checkpoint_path}")
    
    # 创建配置
    config = create_stage_config(args, stage, checkpoint_path)
    
    # 保存配置
    config_dir = Path('checkpoints') / f'stage{stage}'
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_dir / 'config.yaml')
    
    # 实例化模型
    model = instantiate_from_config(config.model)
    
    # 实例化数据模块 - 使用原项目的DataModuleFromConfig
    data_module = instantiate_from_config(config.data)
    
    # 配置回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/stage{stage}',
        filename=f'vavae-stage{stage}-{{epoch:02d}}-{{val_rec_loss:.4f}}',
        monitor='val/rec_loss',
        mode='min',
        save_top_k=2,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 配置训练器
    trainer = pl.Trainer(
        **config.lightning.trainer,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    
    # 开始训练
    print(f"\n🎯 开始第{stage}阶段训练...")
    print(f"   判别器启动: {config.model.params.lossconfig.params.disc_start}")
    print(f"   VF权重: {config.model.params.lossconfig.params.vf_weight}")
    print(f"   距离边距: {config.model.params.lossconfig.params.distmat_margin}")
    print(f"   余弦边距: {config.model.params.lossconfig.params.cos_margin}")
    
    trainer.fit(model, data_module)
    
    return trainer.checkpoint_callback.best_model_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='/kaggle/input/micro-doppler-data',
                       help='微多普勒数据集根目录')
    parser.add_argument('--split_file', type=str, default='dataset_split.json',
                       help='数据划分文件')
    
    # 模型参数
    parser.add_argument('--pretrained_path', type=str,
                       default='/kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt',
                       help='预训练VA-VAE模型路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    
    # GPU配置
    parser.add_argument('--devices', type=str, default=None,
                       help='GPU设备，例如"0,1"或"1"')
    parser.add_argument('--kaggle_t4', action='store_true',
                       help='使用Kaggle T4×2配置')
    parser.add_argument('--detect_anomaly', action='store_true',
                       help='启用异常检测（调试NaN）')
    
    # 阶段选择
    parser.add_argument('--stages', type=str, default='1,2,3',
                       help='要训练的阶段，逗号分隔')
    parser.add_argument('--kaggle', action='store_true',
                       help='Kaggle环境标志')
    
    args = parser.parse_args()
    
    # 验证环境
    if torch.cuda.is_available():
        print(f"🖥️ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   GPU数量: {torch.cuda.device_count()}")
    
    if args.kaggle:
        print("🌐 Kaggle环境检测")
        kaggle_input = Path('/kaggle/input')
        if kaggle_input.exists():
            print("✅ 检测到Kaggle环境")
            # 自动设置路径
            if (kaggle_input / 'micro-doppler-data').exists():
                args.data_root = '/kaggle/input/micro-doppler-data'
            if (kaggle_input / 'vavae-pretrained').exists():
                args.pretrained_path = '/kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt'
    
    # 设置种子
    seed_everything(args.seed)
    
    # 解析阶段
    stages_to_train = [int(s) for s in args.stages.split(',')]
    
    print("="*60)
    print("🚀 VA-VAE 微多普勒微调 - LightningDiT兼容版")
    print("="*60)
    print(f"📊 数据集: {args.data_root}")
    print(f"📦 预训练模型: {args.pretrained_path}")
    print(f"🎯 训练阶段: {stages_to_train}")
    print(f"⚙️  设置:")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Gradient Accumulation: {args.gradient_accumulation}")
    print(f"   - 有效Batch Size: {args.batch_size * args.gradient_accumulation}")
    print("="*60)
    
    # 训练各阶段
    best_checkpoints = []
    for stage in stages_to_train:
        best_ckpt = train_stage(args, stage)
        best_checkpoints.append(best_ckpt)
        print(f"\n✅ 第{stage}阶段完成")
        print(f"📦 最佳checkpoint: {best_ckpt}")
    
    # 保存最终模型
    if best_checkpoints:
        final_ckpt = best_checkpoints[-1]
        checkpoint = torch.load(final_ckpt, map_location='cpu')
        
        # 提取state_dict
        state_dict = checkpoint['state_dict']
        
        # 保存为.pt格式（兼容原项目）
        final_path = Path('checkpoints') / 'vavae_microdoppler_final.pt'
        torch.save({
            'state_dict': state_dict,
            'stages_trained': stages_to_train,
            'config': {
                'embed_dim': 32,
                'use_vf': 'dinov2',
                'reverse_proj': True,
                'resolution': 256
            }
        }, final_path)
        
        print(f"\n{'='*60}")
        print(f"✅ 训练完成!")
        print(f"📦 最终模型: {final_path}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
