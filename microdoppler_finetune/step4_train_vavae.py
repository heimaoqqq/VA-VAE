#!/usr/bin/env python3
"""
VA-VAE训练脚本 - 微多普勒数据微调
基于官方LightningDiT项目的三阶段训练策略
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import argparse

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT', 'vavae'))

from ldm.util import instantiate_from_config
from ldm.data.microdoppler import MicroDopplerDataset
from ldm.models.autoencoder import AutoencoderKL
from torch.utils.data import DataLoader, random_split

class MicroDopplerDataModule(pl.LightningDataModule):
    """PyTorch Lightning数据模块"""
    def __init__(self, data_dir, batch_size=4, num_workers=2, val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
    def setup(self, stage=None):
        if not hasattr(self, 'train_dataset'):
            dataset = MicroDopplerDataset(self.data_dir)
            
            # 分割训练和验证集
            total_size = len(dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"数据集分割: 训练{train_size}, 验证{val_size}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

class TrainingMonitor(Callback):
    """训练监控回调"""
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}: 训练损失 = {trainer.callback_metrics.get('train/aeloss', 'N/A')}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}: 验证损失 = {trainer.callback_metrics.get('val/aeloss', 'N/A')}")

def create_stage_config(stage, checkpoint_path=None):
    """创建训练阶段配置"""
    
    # 三阶段训练配置
    stage_configs = {
        1: {  # Stage 1: 语义对齐
            'max_epochs': 50,
            'learning_rate': 1e-4,
            'disc_start': 5001,
            'disc_weight': 0.5,
            'vf_weight': 0.5,
            'distmat_margin': 0.0,
            'cos_margin': 0.0
        },
        2: {  # Stage 2: 重建优化
            'max_epochs': 15,
            'learning_rate': 5e-5,
            'disc_start': 1,
            'disc_weight': 0.5,
            'vf_weight': 0.1,
            'distmat_margin': 0.0,
            'cos_margin': 0.0
        },
        3: {  # Stage 3: 边界优化
            'max_epochs': 15,
            'learning_rate': 1e-5,
            'disc_start': 1,
            'disc_weight': 0.5,
            'vf_weight': 0.1,
            'distmat_margin': 0.25,
            'cos_margin': 0.5
        }
    }
    
    params = stage_configs[stage]
    
    config = {
        'base_learning_rate': params['learning_rate'],
        'target': 'ldm.models.autoencoder.AutoencoderKL',
        'params': {
            'embed_dim': 32,
            'monitor': 'val/rec_loss',
            'use_vf': 'dinov2',
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
            },
            'lossconfig': {
                'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                'params': {
                    'disc_start': params['disc_start'],
                    'disc_weight': params['disc_weight'],
                    'disc_num_layers': 3,
                    'kl_weight': 1e-6,
                    'pixelloss_weight': 1.0,
                    'perceptual_weight': 1.0,
                    'disc_in_channels': 3,
                    'disc_conditional': False,
                    'vf_weight': params['vf_weight'],
                    'adaptive_vf': True,  # 原始设置
                    'distmat_weight': 1.0,
                    'cos_weight': 1.0,
                    'distmat_margin': params['distmat_margin'],
                    'cos_margin': params['cos_margin'],
                    'use_actnorm': False,
                    'pp_style': False
                }
            },
            'ckpt_path': checkpoint_path,
            'ignore_keys': [],
            'image_key': 'image',
            'colorize_nlabels': None,
            'proj_fix': False
        }
    }
    
    return config, params

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kaggle', action='store_true')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"🚀 VA-VAE 训练 - Stage {args.stages}")
    print(f"{'='*60}")
    print(f"📊 配置:")
    print(f"   数据目录: {args.data_root}")
    print(f"   预训练模型: {args.pretrained_path}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   随机种子: {args.seed}")
    print(f"{'='*60}\n")
    
    # 确定检查点路径
    checkpoint_path = None
    if args.stages == 1:
        checkpoint_path = args.pretrained_path
    elif args.stages > 1:
        # 自动查找前一阶段的最佳检查点
        prev_stage = args.stages - 1
        ckpt_dir = f'logs/stage{prev_stage}/checkpoints'
        if os.path.exists(ckpt_dir):
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if checkpoints:
                checkpoint_path = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
                print(f"✅ 自动加载Stage {prev_stage}检查点: {checkpoint_path}")
    
    # 创建配置
    config, params = create_stage_config(args.stages, checkpoint_path)
    
    # 创建模型
    model = instantiate_from_config(config)
    
    # 创建数据模块
    data_module = MicroDopplerDataModule(
        data_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'logs/stage{args.stages}/checkpoints',
        filename='epoch{epoch:02d}-val_loss{val/rec_loss:.4f}',
        monitor='val/rec_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    
    monitor_callback = TrainingMonitor()
    
    # 创建logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=f'stage{args.stages}',
        version=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        devices='auto',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=params['max_epochs'],
        precision=32,
        callbacks=[checkpoint_callback, monitor_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    print(f"\n🎯 开始Stage {args.stages}训练")
    print(f"   学习率: {config['base_learning_rate']:.2e}")
    print(f"   VF权重: {params['vf_weight']}")
    print(f"   判别器起始: {params['disc_start']}")
    print(f"   最大轮数: {params['max_epochs']}")
    print(f"{'='*60}\n")
    
    # 开始训练
    trainer.fit(model, data_module)
    
    print(f"\n✅ Stage {args.stages}训练完成!")
    print(f"   最佳检查点保存在: logs/stage{args.stages}/checkpoints/")
    
    # 如果不是最后阶段，提示下一步
    if args.stages < 3:
        print(f"\n📋 下一步:")
        print(f"   运行Stage {args.stages + 1}: --stages {args.stages + 1}")

if __name__ == "__main__":
    main()
