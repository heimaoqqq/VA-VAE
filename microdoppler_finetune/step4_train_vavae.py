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

# 关键：在导入ldm之前设置taming路径！
def setup_taming_path():
    """设置taming路径，必须在导入ldm之前调用"""
    # 按优先级检查taming位置
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),  # Kaggle标准位置
        Path('/kaggle/working/.taming_path'),  # 路径文件
        Path.cwd().parent / 'taming-transformers',  # 项目根目录
        Path.cwd() / '.taming_path'  # 当前目录路径文件
    ]
    
    for location in taming_locations:
        if location.name == '.taming_path' and location.exists():
            # 读取路径文件
            try:
                with open(location, 'r') as f:
                    taming_path = f.read().strip()
                if Path(taming_path).exists() and taming_path not in sys.path:
                    sys.path.insert(0, taming_path)
                    print(f"📂 已加载taming路径: {taming_path}")
                    return True
            except Exception as e:
                continue
        elif location.name == 'taming-transformers' and location.exists():
            # 直接路径
            taming_path = str(location.absolute())
            if taming_path not in sys.path:
                sys.path.insert(0, taming_path)
                print(f"📂 发现并加载taming: {taming_path}")
                return True
    
    # 静默失败，因为可能已经通过其他方式加载
    return False

# 在任何导入ldm之前设置taming路径
setup_taming_path()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
# from main import DataModuleFromConfig  # 使用自定义数据模块


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
        data_dict = split_data['train'] if split == 'train' else split_data['val']
        
        # step3生成格式：{"ID_1": [img_paths], "ID_2": [img_paths], ...}
        for user_id, img_paths in data_dict.items():
            for img_path in img_paths:
                if Path(img_path).exists():
                    self.samples.append({
                        'path': Path(img_path),
                        'user_id': user_id
                    })
        
        print(f"✅ {split}集: {len(self.samples)} 张图像")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        img = Image.open(sample['path']).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 最终解决方案：模型get_input需要HWC格式！
        # 验证：get_input会调用permute(0,3,1,2)将BHWC转为BCHW
        
        # 方法1：使用numpy直接创建HWC格式
        img_array = np.array(img).astype(np.float32)  # HWC格式 [256,256,3]
        img_array = img_array / 127.5 - 1.0  # 归一化到[-1,1]
        
        # 转为tensor，保持HWC格式
        img_tensor = torch.from_numpy(img_array)  # [256,256,3]
        
        return {
            'image': img_tensor,  # HWC格式，正确匹配get_input期望
            'user_id': int(sample['user_id'].split('_')[1])
        }


class TrainingMonitorCallback(Callback):
    """训练监控回调 - 增强版"""
    
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.best_val_loss = float('inf')
        self.loss_history = []
        # 创建重建图像保存目录
        self.save_dir = Path(f'logs/stage{stage}/reconstructions')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # 调试：显示所有可用的训练指标
        if epoch <= 2:  # 前3个epoch显示调试信息
            train_keys = [k for k in metrics.keys() if k.startswith('train/')]
            print(f"🔍 调试 - 可用的训练指标: {train_keys}")
        
        # 获取关键损失指标 - 检查损失函数实际返回的指标名称
        val_rec_loss = metrics.get('val/rec_loss', 0)
        val_kl_loss = metrics.get('val/kl_loss', 0)
        val_vf_loss = metrics.get('val/vf_loss', 0)  # 修正为实际名称
        
        # VA-VAE实际记录的训练损失指标
        train_ae_loss = metrics.get('train/aeloss', 0)  # AutoEncoder总损失
        train_disc_loss = metrics.get('train/discloss', 0)  # 判别器损失
        
        # 尝试获取详细损失分解 - 使用损失函数实际的指标名称
        train_total_loss = metrics.get('train/total_loss', 0)  # 总损失
        train_rec_loss = metrics.get('train/rec_loss', 0)      # 重建损失
        train_kl_loss = metrics.get('train/kl_loss', 0)        # KL损失
        train_vf_loss = metrics.get('train/vf_loss', 0)        # VF对齐损失
        train_g_loss = metrics.get('train/g_loss', 0)          # 生成器损失
        
        # 调试：如果损失异常高，打印所有可用的metrics
        if train_ae_loss > 1000:
            print(f"\n⚠️ 检测到异常高损失，详细metrics:")
            for key, value in metrics.items():
                if 'train/' in key and value != 0:
                    print(f"   {key}: {value:.4f}")
        
        # 获取学习率
        current_lr = 0
        if hasattr(pl_module, 'optimizers'):
            opts = pl_module.optimizers()
            if opts and len(opts) > 0:
                current_lr = opts[0].param_groups[0]['lr']
        
        # 判断训练稳定性
        is_stable = self._check_training_stability(val_rec_loss, train_ae_loss)
        stability_icon = "✅" if is_stable else "⚠️"
        
        # 更新最佳损失
        if val_rec_loss < self.best_val_loss:
            self.best_val_loss = val_rec_loss
            best_icon = "🏆"
        else:
            best_icon = ""
        
        print(f"\n{stability_icon} Stage {self.stage} - Epoch {epoch + 1} {best_icon}")
        print(f"📊 验证损失:")
        print(f"   重建: {val_rec_loss:.4f} | KL: {val_kl_loss:.4f} | VF: {val_vf_loss:.4f}")
        print(f"🎯 训练损失:")
        print(f"   AutoEncoder: {train_ae_loss:.4f} | 判别器: {train_disc_loss:.4f}")
        
        # 显示详细损失分解 - 用于诊断 (始终尝试显示，即使是第一个epoch)
        print(f"\n📊 训练损失详情 (高精度):")
        
        # 检查是否有任何详细损失被记录
        has_detailed_loss = (train_total_loss != 0 or train_rec_loss != 0 or 
                           train_kl_loss != 0 or train_vf_loss != 0 or train_g_loss != 0)
        
        if has_detailed_loss:
            print(f"   - Total Loss: {train_total_loss:.6f}")
            print(f"   - Rec Loss: {train_rec_loss:.6f}")
            
            # 显示KL损失的精确值（显示12位小数以观察微小变化）
            if train_kl_loss == 0:
                print(f"   - KL Loss: 0.000000000000 (完全为零)")
            else:
                # 显示实际KL值和加权后的值
                raw_kl = train_kl_loss / 1e-6 if train_kl_loss > 0 else 0
                print(f"   - KL Loss: {train_kl_loss:.12f} (原始KL={raw_kl:.6f}, 权重=1e-6)")
                
            # 显示VF损失的精确值（显示12位小数）
            if train_vf_loss == 0:
                print(f"   - VF Loss: 0.000000000000 (完全为零)")
            else:
                vf_weight = metrics.get('train/vf_weight', 0.5)
                raw_vf = train_vf_loss / vf_weight if vf_weight > 0 else train_vf_loss
                print(f"   - VF Loss: {train_vf_loss:.12f} (原始VF={raw_vf:.6f}, 权重={vf_weight})")
                
            print(f"   - Disc Loss: {train_disc_loss:.6f}")
            print(f"   - Generator Loss: {train_g_loss:.6f}")
        else:
            # 如果没有详细损失，尝试从autoencoder和discriminator损失推断
            print(f"   - AE Loss (聚合): {train_ae_loss:.6f}")
            print(f"   - Disc Loss (聚合): {train_disc_loss:.6f}")
            print(f"   ℹ️ 详细损失分解将在下个epoch开始记录")
        
        print(f"⚙️  学习率: {current_lr:.2e}")
        
        # 阶段特定关注点
        if self.stage == 1:
            if train_vf_loss > 0:
                print(f"🎨 Stage 1 重点: VF对齐效果 = {train_vf_loss:.4f}")
            else:
                print(f"🎨 Stage 1 重点: AE损失(含VF) = {train_ae_loss:.4f}")
        elif self.stage == 2:
            print(f"🏗️  Stage 2 重点: 判别器平衡 = {train_disc_loss:.4f}")
        elif self.stage == 3:
            print(f"🎯 Stage 3 重点: 用户区分优化")
        
        # 异常警告
        self._check_anomalies(val_rec_loss, train_ae_loss, current_lr)
        
        # 🎯 新增功能1: VF语义对齐检查
        self._check_vf_alignment(trainer, pl_module)
        
        # 🎯 新增功能2: 每个epoch生成重建图像
        self._generate_reconstruction_images(trainer, pl_module, epoch)
        
        print("-" * 50)
        
    def _check_training_stability(self, val_loss, train_loss):
        """检查训练稳定性"""
        if torch.isnan(torch.tensor([val_loss, train_loss])).any():
            return False
        if val_loss > 10.0 or train_loss > 10.0:  # 损失过大
            return False
        return True
        
    def _check_anomalies(self, val_loss, train_loss, lr):
        """检查训练异常"""
        warnings = []
        
        if torch.isnan(torch.tensor([val_loss, train_loss])).any():
            warnings.append("🚨 检测到NaN损失!")
        if val_loss > 5.0:
            warnings.append("⚠️  验证损失过高，可能过拟合")
        if train_loss > 10.0:
            warnings.append("⚠️  训练损失异常高")
        if lr < 1e-7:
            warnings.append("⚠️  学习率过低，训练可能停滞")
        if len(self.loss_history) > 5:
            recent_losses = self.loss_history[-5:]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-5 for i in range(1, 5)):
                warnings.append("⚠️  损失收敛停滞")
                
        self.loss_history.append(val_loss)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
            
        for warning in warnings:
            print(warning)
    
    def _check_vf_alignment(self, trainer, pl_module):
        """检查VF语义对齐质量"""
        try:
            if not hasattr(pl_module, 'foundation_model') or pl_module.foundation_model is None:
                print("⚠️ VF模块未初始化")
                return
            
            # 获取验证数据批次进行VF检查
            val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = pl_module.get_input(batch, pl_module.image_key)
                    inputs = inputs[:4].to(pl_module.device)  # 只用前4个样本
                    
                    # 前向传播获取特征
                    reconstructions, posterior, z, aux_feature = pl_module(inputs)
                    
                    if aux_feature is not None and z is not None:
                        # 计算VF特征范数
                        vf_norm = torch.norm(aux_feature, dim=1).mean().item()
                        z_norm = torch.norm(z, dim=1).mean().item()
                        
                        # 计算余弦相似度 - 使用reshape避免tensor stride问题
                        aux_flat = aux_feature.reshape(aux_feature.size(0), -1)
                        z_flat = z.reshape(z.size(0), -1)
                        similarity = torch.nn.functional.cosine_similarity(aux_flat, z_flat, dim=1).mean().item()
                        
                        print(f"\n🔍 VF语义对齐检查:")
                        print(f"   VF特征范数: {vf_norm:.4f}")
                        print(f"   潜在编码范数: {z_norm:.4f}")
                        print(f"   余弦相似度: {similarity:.4f}")
                        
                        if similarity > 0.3:
                            print(f"   ✅ VF语义对齐良好 (相似度 > 0.3)")
                        elif similarity > 0.1:
                            print(f"   ⚠️ VF语义对齐中等 (需要更多训练)")
                        else:
                            print(f"   ❌ VF语义对齐较差 (需要检查配置)")
                        
                        if vf_norm > 0.1:
                            print(f"   ✅ VF特征正常工作 (范数 > 0.1)")
                        else:
                            print(f"   ❌ VF特征可能未激活")
                    else:
                        print("⚠️ VF特征或潜在编码为None")
                    
                    break  # 只检查第一个批次
                    
        except Exception as e:
            print(f"⚠️ VF对齐检查失败: {e}")
    
    def _generate_reconstruction_images(self, trainer, pl_module, epoch):
        """生成并保存重建图像可视化"""
        try:
            pl_module.eval()
            val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = pl_module.get_input(batch, pl_module.image_key)
                    inputs = inputs[:8].to(pl_module.device)  # 只处理前8个样本
                    
                    # 使用正确的方式生成重建：先编码后解码
                    posterior = pl_module.encode(inputs)
                    z = posterior.sample()
                    reconstructions = pl_module.decode(z)
                    
                    # 确保重建不是None且形状正确
                    if reconstructions is None:
                        print(f"   ⚠️ 解码器返回None，尝试使用完整前向传播")
                        # 备用方案：使用完整的前向传播
                        outputs, posterior = pl_module(inputs)
                        reconstructions = outputs
                    
                    # 调试：打印张量形状和范围
                    print(f"   📐 输入形状: {inputs.shape}, 范围: [{inputs.min():.2f}, {inputs.max():.2f}]")
                    print(f"   📐 潜在编码形状: {z.shape}, 范围: [{z.min():.2f}, {z.max():.2f}]")
                    print(f"   📐 重建形状: {reconstructions.shape}, 范围: [{reconstructions.min():.2f}, {reconstructions.max():.2f}]")
                    
                    # 验证重建是否真的不同
                    input_mean = inputs.mean().item()
                    recon_mean = reconstructions.mean().item()
                    mse = ((inputs - reconstructions) ** 2).mean().item()
                    print(f"   🔍 输入均值: {input_mean:.4f}, 重建均值: {recon_mean:.4f}")
                    print(f"   🔍 MSE差异: {mse:.6f}")
                    
                    # 根据实际batch大小创建可视化
                    num_samples = min(8, inputs.shape[0])  # 最多显示8张
                    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
                    fig.suptitle(f'Stage {self.stage} - Epoch {epoch + 1} 重建效果对比')
                    
                    # 确保axes是二维数组，即使只有一列
                    if num_samples == 1:
                        axes = axes.reshape(2, 1)
                    
                    for i in range(num_samples):
                        # 原始图像 (转换为numpy显示格式)
                        orig = inputs[i].cpu().detach().numpy()
                        if orig.shape[0] == 3:  # RGB
                            orig = np.transpose(orig, (1, 2, 0))
                            orig = (orig + 1.0) / 2.0  # 从[-1,1]转为[0,1]
                            orig = np.clip(orig, 0, 1)
                        else:  # 单通道
                            orig = orig[0]
                            orig = (orig + 1.0) / 2.0
                            orig = np.clip(orig, 0, 1)
                        
                        # 重建图像 - 确保使用正确的重建结果
                        recon = reconstructions[i].cpu().detach().numpy()
                        if recon.shape[0] == 3:  # RGB
                            recon = np.transpose(recon, (1, 2, 0))
                            recon = (recon + 1.0) / 2.0
                            recon = np.clip(recon, 0, 1)
                        else:  # 单通道
                            recon = recon[0]
                            recon = (recon + 1.0) / 2.0
                            recon = np.clip(recon, 0, 1)
                        
                        # 显示原始图像（第一行）
                        axes[0, i].imshow(orig, cmap='viridis' if orig.ndim == 2 else None)
                        axes[0, i].axis('off')
                        if i == 0:
                            axes[0, i].set_title('原始图像', fontsize=10)
                        
                        # 显示重建图像（第二行）- 确保是重建而不是原始
                        axes[1, i].imshow(recon, cmap='viridis' if recon.ndim == 2 else None)
                        axes[1, i].axis('off')
                        if i == 0:
                            axes[1, i].set_title('重建图像', fontsize=10)
                        
                        # 调试：检查是否真的不同
                        if i == 0:
                            diff = np.abs(orig - recon).mean()
                            print(f"   📊 第一张图的平均差异: {diff:.4f}")
                    
                    # 保存图像
                    save_path = self.save_dir / f'stage{self.stage}_epoch{epoch + 1:03d}.png'
                    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    print(f"💾 重建图像已保存: {save_path}")
                    break  # 只处理第一个批次
                    
            pl_module.train()
        except Exception as e:
            print(f"⚠️ 图像生成失败: {e}")


class MicroDopplerDataModule(pl.LightningDataModule):
    """微多普勒数据模块"""
    
    def __init__(self, data_root, split_file, batch_size=8, num_workers=4, image_size=256):
        super().__init__()
        self.data_root = data_root
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MicroDopplerDataset(
                data_root=self.data_root,
                split_file=self.split_file,
                split='train',
                image_size=self.image_size
            )
            self.val_dataset = MicroDopplerDataset(
                data_root=self.data_root,
                split_file=self.split_file,
                split='val',
                image_size=self.image_size
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )


def create_stage_config(args, stage, checkpoint_path=None):
    """创建阶段配置"""
    
    stage_params = {
        1: {'disc_start': 5001, 'disc_weight': 0.5, 'vf_weight': 0.5, 'distmat_margin': 0.0, 'cos_margin': 0.0, 'learning_rate': 1e-4, 'max_epochs': 50},
        2: {'disc_start': 1, 'disc_weight': 0.5, 'vf_weight': 0.1, 'distmat_margin': 0.0, 'cos_margin': 0.0, 'learning_rate': 5e-5, 'max_epochs': 15},
        3: {'disc_start': 1, 'disc_weight': 0.5, 'vf_weight': 0.1, 'distmat_margin': 0.25, 'cos_margin': 0.5, 'learning_rate': 2e-5, 'max_epochs': 15}
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
                'use_vf': 'dinov2',
                'reverse_proj': True,
                'ddconfig': {
                    'double_z': True, 'z_channels': 32, 'resolution': 256,
                    'in_channels': 3, 'out_ch': 3, 'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2,
                    'attn_resolutions': [16], 'dropout': 0.0
                },
                'lossconfig': {
                    'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                    'params': {
                        'disc_start': params['disc_start'], 'disc_num_layers': 3,
                        'disc_weight': params['disc_weight'], 'disc_factor': 1.0,
                        'disc_in_channels': 3, 'disc_conditional': False, 'disc_loss': 'hinge',
                        'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,  # 重要：原项目使用感知损失！
                        'kl_weight': 1e-6, 'logvar_init': 0.0,
                        'use_actnorm': False,  # 判别器中不使用ActNorm
                        'pp_style': False,  # 不使用pp_style的nll损失计算
                        'vf_weight': params['vf_weight'], 'adaptive_vf': False,  # 禁用自适应避免权重失控
                        'distmat_weight': 1.0, 'cos_weight': 1.0,
                        'distmat_margin': params['distmat_margin'],
                        'cos_margin': params['cos_margin']
                    }
                }
            }
        }
    })
    
    return config


def train_stage(args, stage):
    """训练阶段"""
    
    seed_everything(args.seed, workers=True)
    
    checkpoint_path = None
    if stage > 1:
        prev_ckpt_dir = Path(f'checkpoints/stage{stage-1}')
        if prev_ckpt_dir.exists():
            ckpt_files = list(prev_ckpt_dir.glob('*.ckpt'))
            if ckpt_files:
                checkpoint_path = str(max(ckpt_files, key=lambda x: x.stat().st_mtime))
                print(f"加载checkpoint: {checkpoint_path}")

    config = create_stage_config(args, stage, checkpoint_path)
    params = config.model.params.lossconfig.params

    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate

    # 全面验证VF模块和权重加载
    if hasattr(model, 'use_vf'):
        print(f"🔍 VF模块状态: use_vf={model.use_vf}")
        if model.use_vf and hasattr(model, 'foundation_model'):
            print(f"✅ DINOv2模型已加载")
            # 检查关键权重是否存在
            has_vf_weights = any('foundation_model' in k for k in model.state_dict().keys())
            has_proj_weights = any('linear_proj' in k for k in model.state_dict().keys())
            print(f"   - Foundation权重: {'✅ 已加载' if has_vf_weights else '❌ 缺失'}")
            print(f"   - Projection权重: {'✅ 已加载' if has_proj_weights else '❌ 缺失'}")
            
            if not has_vf_weights:
                print(f"⚠️  警告：DINOv2权重未从预训练模型加载！")
                print(f"   这会导致VF损失为0，Stage 1训练无效")
                print(f"   请确保预训练文件包含foundation_model权重")
        else:
            print(f"⚠️  DINOv2模型未正确初始化！")
    else:
        print(f"❌ 模型缺少use_vf属性！")
    print(f"学习率: {model.learning_rate:.2e}")

    data_module = MicroDopplerDataModule(
        data_root=args.data_root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=256
    )
    
    checkpoint_dir = Path(f'checkpoints/stage{stage}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'vavae-stage{stage}-{{epoch:02d}}-{{val_rec_loss:.4f}}',
        monitor='val/rec_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=True
    )
    
    training_monitor = TrainingMonitorCallback(stage)
    
    trainer = pl.Trainer(
        devices='auto',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=params.get('max_epochs', 50),
        precision=32,
        callbacks=[checkpoint_callback, training_monitor],
        enable_progress_bar=True,  # 启用默认进度条
        enable_model_summary=False,  # 禁用模型摘要输出
        log_every_n_steps=50,  # 增加日志步长减少输出频率
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # 跳过sanity check避免额外的验证输出
        logger=False  # 禁用默认logger减少输出
    )
    
    print(f"\n第{stage}阶段训练 - LR: {config.model.base_learning_rate:.2e}")
    
    trainer.fit(model, data_module)
    
    return trainer.checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/kaggle/input/micro-doppler-data')
    parser.add_argument('--split_file', type=str, default='/kaggle/working/data_split/dataset_split.json')
    parser.add_argument('--pretrained_path', type=str, default='/kaggle/input/vavae-pretrained/vavae-imagenet256-f16d32-dinov2.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stages', type=str, default='1,2,3')
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--kaggle', action='store_true')
    
    args = parser.parse_args()
    stages_to_train = [int(s) for s in args.stages.split(',')]
    
    print("开始VA-VAE微多普勒微调训练")
    print(f"数据: {args.data_root}")
    print(f"训练阶段: {stages_to_train}")
    
    data_module = MicroDopplerDataModule(
        data_root=args.data_root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=256
    )
    data_module.setup()
    
    print(f"训练集: {len(data_module.train_dataset)} 张图像, 验证集: {len(data_module.val_dataset)} 张图像")
    
    best_checkpoints = []
    for stage in stages_to_train:
        best_ckpt = train_stage(args, stage)
        best_checkpoints.append(best_ckpt)
        print(f"第{stage}阶段完成")
    
    print(f"训练完成! 最佳checkpoints: {best_checkpoints}")
    
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
