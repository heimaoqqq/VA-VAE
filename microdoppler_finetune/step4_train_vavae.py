#!/usr/bin/env python3
"""
VA-VAE 训练脚本 - 微多普勒数据微调

关键特性：
1. 固定VF权重（adaptive_vf=False）避免域差异导致的权重失控
   - Stage 1: vf_weight=0.5 (初始对齐)
   - Stage 2/3: vf_weight=0.1 (精细调整)
2. 详细损失监控 - 显示所有损失分量明细
3. VF语义对齐检测 - 监控特征相似度
4. 自动可视化 - 每个epoch保存重建对比图

使用方法：
python step4_train_vavae.py --stage 1 --batch_size 4
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT', 'vavae'))

# 添加taming-transformers路径（解决taming模块导入问题）
def setup_taming_path():
    """设置taming-transformers路径"""
    # 检查多个可能的taming位置
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'taming-transformers'),  # 相对路径
        '/kaggle/working/taming-transformers',  # Kaggle环境
        os.path.join(os.getcwd(), 'taming-transformers'),  # 当前目录
    ]
    
    # 检查.taming_path文件
    taming_path_file = os.path.join(os.path.dirname(__file__), '..', '.taming_path')
    if os.path.exists(taming_path_file):
        with open(taming_path_file, 'r') as f:
            possible_paths.insert(0, f.read().strip())
    
    for taming_path in possible_paths:
        if os.path.exists(taming_path):
            if taming_path not in sys.path:
                sys.path.insert(0, taming_path)
                print(f"✅ 已添加taming路径: {taming_path}")
            return True
    
    print("❌ 未找到taming-transformers，请先运行step1_setup_environment.py")
    return False

# 设置taming路径
setup_taming_path()

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
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = MicroDopplerDataset(self.data_dir)
            
            # 分割训练和验证集
            total_size = len(self.dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size],
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

class DetailedLossMonitor(Callback):
    """详细损失监控回调"""
    def __init__(self, stage, save_dir='logs/reconstructions'):
        self.stage = stage
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.epoch_losses = {
            'train': {'rec': [], 'kl': [], 'vf': [], 'disc': [], 'g': [], 'total': []},
            'val': {'rec': [], 'kl': [], 'vf': [], 'total': []}
        }
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录训练批次的详细损失"""
        # 从日志中提取详细损失
        logs = trainer.logged_metrics
        
        if 'train/rec_loss' in logs:
            self.epoch_losses['train']['rec'].append(logs['train/rec_loss'].item())
        if 'train/kl_loss' in logs:
            self.epoch_losses['train']['kl'].append(logs['train/kl_loss'].item())
        if 'train/vf_loss' in logs:
            self.epoch_losses['train']['vf'].append(logs['train/vf_loss'].item())
        if 'train/disc_loss' in logs:
            self.epoch_losses['train']['disc'].append(logs['train/disc_loss'].item())
        if 'train/g_loss' in logs:
            self.epoch_losses['train']['g'].append(logs['train/g_loss'].item())
        if 'train/aeloss' in logs:
            self.epoch_losses['train']['total'].append(logs['train/aeloss'].item())
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录验证批次的详细损失"""
        logs = trainer.logged_metrics
        
        if 'val/rec_loss' in logs:
            self.epoch_losses['val']['rec'].append(logs['val/rec_loss'].item())
        if 'val/kl_loss' in logs:
            self.epoch_losses['val']['kl'].append(logs['val/kl_loss'].item())
        if 'val/vf_loss' in logs:
            self.epoch_losses['val']['vf'].append(logs['val/vf_loss'].item())
        if 'val/total_loss' in logs:
            self.epoch_losses['val']['total'].append(logs['val/total_loss'].item())
    
    def on_epoch_end(self, trainer, pl_module):
        """Epoch结束时打印详细损失并生成重建图像"""
        epoch = trainer.current_epoch
        
        print(f"\n{'='*60}")
        print(f"📊 Stage {self.stage} - Epoch {epoch} 详细损失报告")
        print(f"{'='*60}")
        
        # 打印训练损失
        print(f"\n🎯 训练损失明细:")
        if self.epoch_losses['train']['rec']:
            print(f"   重建损失: {np.mean(self.epoch_losses['train']['rec']):.4f}")
        if self.epoch_losses['train']['kl']:
            print(f"   KL损失: {np.mean(self.epoch_losses['train']['kl']):.6f}")
        if self.epoch_losses['train']['vf']:
            vf_loss = np.mean(self.epoch_losses['train']['vf'])
            print(f"   VF损失: {vf_loss:.4f}")
            if vf_loss > 0:
                print(f"   ✅ VF正在工作 - 语义对齐损失: {vf_loss:.4f}")
            else:
                print(f"   ⚠️ VF未激活或损失为0")
        if self.epoch_losses['train']['disc']:
            print(f"   判别器损失: {np.mean(self.epoch_losses['train']['disc']):.4f}")
        if self.epoch_losses['train']['g']:
            print(f"   生成器损失: {np.mean(self.epoch_losses['train']['g']):.4f}")
        if self.epoch_losses['train']['total']:
            total_loss = np.mean(self.epoch_losses['train']['total'])
            print(f"   总损失: {total_loss:.2f}")
            if total_loss > 1000:
                print(f"   ⚠️ 总损失异常高，可能是VF权重问题")
        
        # 打印验证损失
        print(f"\n📊 验证损失明细:")
        if self.epoch_losses['val']['rec']:
            print(f"   重建损失: {np.mean(self.epoch_losses['val']['rec']):.4f}")
        if self.epoch_losses['val']['kl']:
            print(f"   KL损失: {np.mean(self.epoch_losses['val']['kl']):.6f}")
        if self.epoch_losses['val']['vf']:
            print(f"   VF损失: {np.mean(self.epoch_losses['val']['vf']):.4f}")
        if self.epoch_losses['val']['total']:
            print(f"   总损失: {np.mean(self.epoch_losses['val']['total']):.4f}")
        
        # 生成重建图像
        self.generate_reconstructions(trainer, pl_module, epoch)
        
        # 清空损失记录
        for split in self.epoch_losses:
            for key in self.epoch_losses[split]:
                self.epoch_losses[split][key] = []
        
        print(f"{'='*60}\n")
    
    def generate_reconstructions(self, trainer, pl_module, epoch):
        """生成并保存重建图像"""
        pl_module.eval()
        dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        
        with torch.no_grad():
            # 获取一个批次的数据
            for batch in dataloader:
                inputs = pl_module.get_input(batch, pl_module.image_key)
                inputs = inputs[:8].to(pl_module.device)  # 只处理前8个样本
                
                # 生成重建
                reconstructions, posterior, z, aux_feature = pl_module(inputs)
                
                # 检查VF特征
                if aux_feature is not None:
                    vf_norm = torch.norm(aux_feature, dim=1).mean().item()
                    z_norm = torch.norm(z, dim=1).mean().item()
                    similarity = torch.nn.functional.cosine_similarity(
                        aux_feature.view(aux_feature.size(0), -1),
                        z.view(z.size(0), -1)
                    ).mean().item()
                    print(f"\n🔍 VF语义对齐检查:")
                    print(f"   VF特征范数: {vf_norm:.4f}")
                    print(f"   潜在编码范数: {z_norm:.4f}")
                    print(f"   余弦相似度: {similarity:.4f}")
                    if similarity > 0.5:
                        print(f"   ✅ VF语义对齐良好")
                    else:
                        print(f"   ⚠️ VF语义对齐较差，需要更多训练")
                
                # 创建可视化
                fig, axes = plt.subplots(2, 8, figsize=(16, 4))
                
                for i in range(min(8, inputs.shape[0])):
                    # 原始图像
                    orig = inputs[i].cpu().numpy()
                    if orig.shape[0] == 3:  # RGB
                        orig = np.transpose(orig, (1, 2, 0))
                    else:  # 单通道
                        orig = orig[0]
                    
                    # 重建图像
                    recon = reconstructions[i].cpu().numpy()
                    if recon.shape[0] == 3:  # RGB
                        recon = np.transpose(recon, (1, 2, 0))
                    else:  # 单通道
                        recon = recon[0]
                    
                    # 显示
                    axes[0, i].imshow(orig, cmap='viridis' if orig.ndim == 2 else None)
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_title('原始')
                    
                    axes[1, i].imshow(recon, cmap='viridis' if recon.ndim == 2 else None)
                    axes[1, i].axis('off')
                    if i == 0:
                        axes[1, i].set_title('重建')
                
                plt.suptitle(f'Stage {self.stage} - Epoch {epoch} 重建效果')
                save_path = os.path.join(self.save_dir, f'stage{self.stage}_epoch{epoch:03d}.png')
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"   💾 重建图像已保存: {save_path}")
                
                break  # 只处理第一个批次
        
        pl_module.train()

def create_stage_config(stage, checkpoint_path=None):
    """创建阶段配置 - 修复版"""
    # 三阶段训练配置
    stage_configs = {
        1: {
            'disc_start': 5001, 
            'disc_weight': 0.5, 
            'vf_weight': 0.5,  # 保持原始值
            'distmat_margin': 0.0, 
            'cos_margin': 0.0, 
            'learning_rate': 1e-4, 
            'max_epochs': 50
        },
        2: {
            'disc_start': 1, 
            'disc_weight': 0.5, 
            'vf_weight': 0.1,  # Stage 2降低VF权重
            'distmat_margin': 0.0, 
            'cos_margin': 0.0, 
            'learning_rate': 5e-5, 
            'max_epochs': 15
        },
        3: {
            'disc_start': 1, 
            'disc_weight': 0.5, 
            'vf_weight': 0.1, 
            'distmat_margin': 0.25,  # Stage 3启用margin
            'cos_margin': 0.5, 
            'learning_rate': 2e-5, 
            'max_epochs': 15
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
                    'perceptual_weight': 1.0,  # 启用LPIPS
                    'disc_in_channels': 3,
                    'disc_conditional': False,
                    'vf_weight': params['vf_weight'], 
                    'adaptive_vf': False,  # 关键修复：禁用自适应VF避免权重失控
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='data/microdoppler_dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--pretrained', type=str, 
                       default='pretrained_models/vavae/dinov2_f16d32_res256x256.pth',
                       help='预训练模型路径')
    args = parser.parse_args()
    
    # 打印训练信息
    print(f"\n{'='*60}")
    print(f"🚀 VA-VAE 增强版训练 - Stage {args.stage}")
    print(f"{'='*60}")
    print(f"📊 配置:")
    print(f"   批次大小: {args.batch_size}")
    print(f"   数据目录: {args.data_dir}")
    print(f"   预训练模型: {args.pretrained}")
    print(f"   检查点: {args.checkpoint}")
    print(f"   ⚠️ adaptive_vf: False (避免权重失控)")
    print(f"{'='*60}\n")
    
    # 确定检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.stage == 1:
        checkpoint_path = args.pretrained
    elif checkpoint_path is None and args.stage > 1:
        # 自动查找前一阶段的最佳检查点
        prev_stage = args.stage - 1
        ckpt_dir = f'logs/stage{prev_stage}/checkpoints'
        if os.path.exists(ckpt_dir):
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if checkpoints:
                checkpoint_path = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
                print(f"✅ 自动加载Stage {prev_stage}检查点: {checkpoint_path}")
    
    # 创建配置
    config, params = create_stage_config(args.stage, checkpoint_path)
    
    # 创建模型
    model = instantiate_from_config(config)
    
    # 创建数据模块
    data_module = MicroDopplerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'logs/stage{args.stage}/checkpoints',
        filename='epoch{epoch:02d}-val_loss{val/rec_loss:.4f}',
        monitor='val/rec_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    
    loss_monitor = DetailedLossMonitor(
        stage=args.stage,
        save_dir=f'logs/stage{args.stage}/reconstructions'
    )
    
    # 创建logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=f'stage{args.stage}',
        version=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        devices='auto',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=params['max_epochs'],
        precision=32,  # 使用FP32保证稳定性
        callbacks=[checkpoint_callback, loss_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # 梯度裁剪防止爆炸
        enable_progress_bar=True
    )
    
    print(f"\n🎯 开始Stage {args.stage}训练")
    print(f"   学习率: {config['base_learning_rate']:.2e}")
    print(f"   VF权重: {params['vf_weight']}")
    print(f"   判别器起始: {params['disc_start']}")
    print(f"   最大轮数: {params['max_epochs']}")
    print(f"{'='*60}\n")
    
    # 开始训练
    trainer.fit(model, data_module)
    
    print(f"\n✅ Stage {args.stage}训练完成!")
    print(f"   最佳检查点保存在: logs/stage{args.stage}/checkpoints/")
    print(f"   重建图像保存在: logs/stage{args.stage}/reconstructions/")
    
    # 如果不是最后阶段，提示下一步
    if args.stage < 3:
        print(f"\n📌 下一步:")
        print(f"   python step4_train_vavae_enhanced.py --stage {args.stage + 1}")

if __name__ == '__main__':
    main()
