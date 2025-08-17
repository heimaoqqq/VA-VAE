#!/usr/bin/env python3
"""
Step 6: LightningDiT 微多普勒条件生成训练 (重构版)
基于官方LightningDiT项目，针对微多普勒小数据集优化
支持Kaggle T4×2 GPU环境
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import shutil

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from PIL import Image
import yaml

# 添加LightningDiT路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MicroDopplerConfig:
    """微多普勒训练配置 - 基于官方LightningDiT格式"""
    
    # 训练配置 - 针对小数据集优化
    num_epochs: int = 80
    batch_size: int = 2  # T4×2显存限制
    gradient_accumulation_steps: int = 4  # 有效batch_size=8
    learning_rate: float = 5e-5  # 小数据集保守学习率
    weight_decay: float = 0.0  # 官方不使用
    beta2: float = 0.95  # 官方推荐
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 200
    ema_decay: float = 0.9999
    
    # 早停配置 - 防止小数据集过拟合
    patience: int = 15
    min_improvement: float = 1e-4
    
    # 模型配置
    model_type: str = "L/2"  # LightningDiT-L/2 (正确的模型键名)
    num_classes: int = 31  # 31个用户
    input_size: int = 16  # 256/16=16 (VA-VAE下采样比例)
    
    # 损失函数配置 - 官方推荐
    use_cosine_loss: bool = True
    use_lognorm: bool = True
    
    # 采样配置 - 针对微多普勒优化
    sampling_method: str = "euler"
    num_steps: int = 200
    cfg_scale: float = 8.0  # 适中CFG，避免过度引导
    cfg_interval_start: float = 0.11
    timestep_shift: float = 0.1  # 保守设置，保留细节
    
    # 数据配置
    num_workers: int = 2  # Kaggle环境
    pin_memory: bool = True
    persistent_workers: bool = True
    latent_norm: bool = True  # 官方强烈推荐
    latent_multiplier: float = 1.0
    
    # 路径配置
    data_dir: str = "/kaggle/input/dataset"
    vae_checkpoint: str = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    split_file: str = "/kaggle/working/data_split/dataset_split.json"
    output_dir: str = "/kaggle/working"
    
    # Kaggle T4×2 GPU配置
    use_multi_gpu: bool = True
    gpu_strategy: str = "DataParallel"  # Kaggle推荐DataParallel而非DDP


class MicroDopplerDataManager:
    """微多普勒数据管理器 - 兼容官方LightningDiT数据格式"""
    
    def __init__(self, config: MicroDopplerConfig):
        self.config = config
        self.split_data = None
        self.latents_file = Path(config.output_dir) / "latents_microdoppler.safetensors"
        self.stats_file = Path(config.output_dir) / "latents_stats.pt"
        
    def load_split_info(self):
        """加载step3生成的数据划分信息"""
        split_file = Path(self.config.split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"数据划分文件不存在: {split_file}")
        
        with open(split_file, 'r') as f:
            self.split_data = json.load(f)
        
        logger.info(f"✅ 加载数据划分信息:")
        logger.info(f"   训练图像: {self.split_data['statistics']['train_images']}")
        logger.info(f"   验证图像: {self.split_data['statistics']['val_images']}")
        logger.info(f"   总用户数: {self.split_data['statistics']['total_users']}")
        
    def encode_to_latents(self, vae, device):
        """将数据集编码到潜空间并保存为safetensors格式"""
        if self.latents_file.exists() and self.stats_file.exists():
            logger.info(f"✅ 潜空间数据已存在: {self.latents_file}")
            return
            
        logger.info("🔄 开始编码数据集到潜空间...")
        
        all_latents = []
        all_labels = []
        total_processed = 0
        
        # 处理训练集和验证集
        for split_name in ['train', 'val']:
            logger.info(f"\n📊 处理 {split_name} 集...")
            split_images = self.split_data[split_name]
            
            for user_key, image_paths in split_images.items():
                user_id = int(user_key.split('_')[1]) - 1  # ID_1 -> 0
                
                if not image_paths:
                    continue
                    
                logger.info(f"  编码用户 {user_key}: {len(image_paths)} 张图像")
                
                for img_path in image_paths:
                    try:
                        img_path = Path(img_path)
                        if not img_path.exists():
                            continue
                        
                        # 加载并预处理图像
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((256, 256), Image.LANCZOS)
                        
                        # 转换为tensor并归一化到[-1,1]
                        img_array = np.array(img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                        img_tensor = img_tensor * 2.0 - 1.0
                        img_tensor = img_tensor.to(device)
                        
                        # 编码到潜空间
                        with torch.no_grad():
                            # VA_VAE使用encode_images方法
                            latent = vae.encode_images(img_tensor)

                            # 调试信息：检查第一个潜空间的形状
                            if total_processed == 0:
                                logger.info(f"第一个潜空间形状: {latent.shape}")

                            all_latents.append(latent.cpu())
                            all_labels.append(user_id)
                            total_processed += 1
                            
                    except Exception as e:
                        logger.warning(f"编码失败 {img_path}: {e}")
                        continue
        
        if not all_latents:
            raise ValueError("没有成功编码任何图像！")
        
        # 合并数据
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        # 计算统计信息 - 修复维度问题
        logger.info(f"计算潜空间统计信息，形状: {all_latents.shape}")

        # 计算每个通道的均值和标准差
        latent_mean = all_latents.mean(dim=0, keepdim=True)  # [1, C, H, W]
        latent_std = all_latents.std(dim=0, keepdim=True)    # [1, C, H, W]

        logger.info(f"统计信息形状 - mean: {latent_mean.shape}, std: {latent_std.shape}")
        
        # 保存为safetensors格式（兼容官方ImgLatentDataset）
        try:
            import safetensors.torch
            safetensors.torch.save_file({
                'latents': all_latents,
                'labels': all_labels
            }, self.latents_file)
        except ImportError:
            # 如果safetensors不可用，使用torch格式
            logger.warning("⚠️ safetensors不可用，使用torch格式保存")
            torch.save({
                'latents': all_latents,
                'labels': all_labels
            }, self.latents_file.with_suffix('.pt'))
        
        # 保存统计信息
        torch.save({
            'mean': latent_mean,
            'std': latent_std
        }, self.stats_file)
        
        logger.info(f"\n✅ 编码完成!")
        logger.info(f"   处理图像: {total_processed} 张")
        logger.info(f"   潜空间样本: {len(all_latents)} 个")
        logger.info(f"   潜空间形状: {all_latents.shape}")
        logger.info(f"   保存到: {self.latents_file}")
        
    def create_dataloaders(self):
        """创建训练和验证数据加载器"""
        # 使用简化的数据集类
        dataset = MicroDopplerLatentDataset(
            latents_file=self.latents_file,
            stats_file=self.stats_file,
            latent_norm=self.config.latent_norm,
            latent_multiplier=self.config.latent_multiplier
        )

        # 划分训练和验证集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

        logger.info(f"✅ 数据加载器创建完成:")
        logger.info(f"   训练集: {len(train_dataset)} 样本")
        logger.info(f"   验证集: {len(val_dataset)} 样本")

        return train_loader, val_loader


class MicroDopplerLatentDataset(Dataset):
    """微多普勒潜空间数据集"""

    def __init__(self, latents_file, stats_file, latent_norm=True, latent_multiplier=1.0):
        self.latents_file = Path(latents_file)
        self.stats_file = Path(stats_file)
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        # 加载数据
        if self.latents_file.suffix == '.safetensors':
            try:
                import safetensors.torch
                with safetensors.torch.safe_open(self.latents_file, framework="pt", device="cpu") as f:
                    self.latents = f.get_tensor('latents')
                    self.labels = f.get_tensor('labels')
            except ImportError:
                # 回退到torch格式
                data = torch.load(self.latents_file.with_suffix('.pt'))
                self.latents = data['latents']
                self.labels = data['labels']
        else:
            data = torch.load(self.latents_file)
            self.latents = data['latents']
            self.labels = data['labels']

        # 加载统计信息
        if latent_norm and self.stats_file.exists():
            stats = torch.load(self.stats_file)
            self.latent_mean = stats['mean']
            self.latent_std = stats['std']
            logger.info(f"加载统计信息 - mean: {self.latent_mean.shape}, std: {self.latent_std.shape}")
        else:
            self.latent_mean = None
            self.latent_std = None

        logger.info(f"数据集大小: {len(self.latents)}, 潜空间形状: {self.latents[0].shape if len(self.latents) > 0 else 'N/A'}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx].clone()
        label = self.labels[idx].clone()

        # 归一化 - 修复维度不匹配问题
        if self.latent_norm and self.latent_mean is not None:
            # 确保统计信息的维度与潜空间数据匹配
            if self.latent_mean.dim() == 4:  # [1, C, H, W]
                mean = self.latent_mean.squeeze(0)  # [C, H, W]
                std = self.latent_std.squeeze(0)    # [C, H, W]
            else:
                mean = self.latent_mean
                std = self.latent_std

            # 确保维度匹配
            if latent.shape != mean.shape:
                logger.warning(f"维度不匹配: latent {latent.shape} vs mean {mean.shape}")
                # 如果统计信息是通道维度的，需要广播到空间维度
                if mean.dim() == 1:  # 只有通道维度
                    mean = mean.view(-1, 1, 1)
                    std = std.view(-1, 1, 1)
                elif mean.dim() == 3 and latent.dim() == 3:
                    # 都是3维，检查空间维度
                    if mean.shape[1:] != latent.shape[1:]:
                        # 只使用通道维度的统计信息
                        mean = mean.mean(dim=[1, 2], keepdim=True)
                        std = std.mean(dim=[1, 2], keepdim=True)

            latent = (latent - mean) / (std + 1e-8)  # 添加小值避免除零

        latent = latent * self.latent_multiplier

        return latent, label


def setup_kaggle_multi_gpu():
    """设置Kaggle T4×2 GPU环境"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请在Kaggle中启用GPU加速器")

    num_gpus = torch.cuda.device_count()
    logger.info(f"🖥️ 检测到 {num_gpus} 个GPU")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    if num_gpus < 2:
        logger.warning("⚠️ 只检测到1个GPU，将使用单GPU训练")
        return False, torch.device("cuda:0")

    # Kaggle T4×2环境设置
    # 设置主设备为cuda:0，DataParallel会自动使用所有可用GPU
    torch.cuda.set_device(0)  # 设置主GPU
    device = torch.device("cuda:0")

    # 清理GPU缓存
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

    logger.info("✅ 将使用DataParallel进行双GPU训练")
    logger.info(f"   主设备: {device}")
    logger.info(f"   可用GPU: {list(range(num_gpus))}")
    return True, device


class DiTModelManager:
    """DiT模型管理器 - 使用官方LightningDiT组件"""

    def __init__(self, config: MicroDopplerConfig):
        self.config = config

    def load_vae(self, device):
        """加载微调后的VA-VAE模型"""
        logger.info("🔄 加载微调后的VA-VAE模型...")

        # 设置taming路径
        self._setup_taming_path()

        # 创建临时配置文件
        vae_config = self._create_vae_config()
        temp_config_path = Path(self.config.output_dir) / "temp_vae_config.yaml"

        with open(temp_config_path, 'w') as f:
            yaml.dump(vae_config, f)

        # 加载VA-VAE
        from tokenizer.vavae import VA_VAE
        try:
            # 尝试使用配置文件路径
            vae = VA_VAE(
                config=str(temp_config_path),
                img_size=256,
                horizon_flip=0.0,  # 不使用数据增强
                fp16=True
            )
        except Exception as e:
            logger.warning(f"使用配置文件加载失败: {e}")
            # 尝试直接使用checkpoint路径
            try:
                vae = VA_VAE(
                    config=self.config.vae_checkpoint,
                    img_size=256,
                    horizon_flip=0.0,
                    fp16=True
                )
            except Exception as e2:
                logger.error(f"VA-VAE加载失败: {e2}")
                raise RuntimeError(f"无法加载VA-VAE模型: {e2}")

        # 清理临时文件
        temp_config_path.unlink()

        logger.info(f"✅ VA-VAE加载完成")
        return vae

    def create_dit_model(self, device, use_multi_gpu=False):
        """创建LightningDiT模型并进行XL→L权重迁移"""
        logger.info("🔄 创建LightningDiT模型...")

        from models.lightningdit import LightningDiT_models

        # 创建L模型
        dit_model = LightningDiT_models[f"LightningDiT-{self.config.model_type}"](
            input_size=self.config.input_size,
            num_classes=self.config.num_classes,
        )

        # XL→L权重迁移
        self._transfer_xl_to_l_weights(dit_model)

        # 确保模型在正确设备上
        dit_model = dit_model.to(device)

        # Kaggle T4×2使用DataParallel
        if use_multi_gpu and torch.cuda.device_count() > 1:
            # 确保模型在cuda:0上，然后包装DataParallel
            dit_model = dit_model.to('cuda:0')
            dit_model = nn.DataParallel(dit_model, device_ids=list(range(torch.cuda.device_count())))
            logger.info(f"✅ 启用DataParallel多GPU训练，使用GPU: {list(range(torch.cuda.device_count()))}")
        else:
            logger.info(f"✅ 单GPU训练，使用设备: {device}")

        # 统计参数
        total_params = sum(p.numel() for p in dit_model.parameters())
        trainable_params = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)

        logger.info(f"✅ LightningDiT-{self.config.model_type} 创建完成")
        logger.info(f"   模型架构: 24层, 1024隐藏维度, patch_size=2")
        logger.info(f"   总参数: {total_params / 1e6:.1f}M")
        logger.info(f"   可训练参数: {trainable_params / 1e6:.1f}M")

        return dit_model

    def setup_training_components(self, dit_model):
        """设置训练组件：优化器、调度器、transport"""
        from transport import create_transport

        # 创建transport - 使用官方推荐配置
        transport = create_transport(
            path_type='Linear',
            prediction='velocity',
            loss_weight=None,
            train_eps=None,
            sample_eps=None,
            use_cosine_loss=self.config.use_cosine_loss,
            use_lognorm=self.config.use_lognorm
        )

        # 优化器 - 官方推荐配置
        optimizer = torch.optim.AdamW(
            dit_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, self.config.beta2),
            eps=1e-8
        )

        # 学习率调度器
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            # 余弦退火
            progress = float(current_step - self.config.warmup_steps) / float(max(1, 10000 - self.config.warmup_steps))
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        logger.info("✅ 训练组件设置完成")
        logger.info(f"   优化器: AdamW (lr={self.config.learning_rate}, beta2={self.config.beta2})")
        logger.info(f"   调度器: 预热{self.config.warmup_steps}步 + 余弦退火")
        logger.info(f"   Transport: Linear path + velocity prediction")

        return optimizer, scheduler, transport

    def _setup_taming_path(self):
        """设置taming-transformers路径"""
        taming_locations = [
            Path('/kaggle/working/taming-transformers'),
            Path('/kaggle/working/.taming_path'),
        ]

        for location in taming_locations:
            if location.name == '.taming_path' and location.exists():
                with open(location, 'r') as f:
                    taming_path = f.read().strip()
                if Path(taming_path).exists() and taming_path not in sys.path:
                    sys.path.insert(0, taming_path)
                    return
            elif location.exists():
                taming_path = str(location.absolute())
                if taming_path not in sys.path:
                    sys.path.insert(0, taming_path)
                    return

    def _create_vae_config(self):
        """创建VA-VAE配置 - 基于官方格式"""
        return {
            'ckpt_path': self.config.vae_checkpoint,
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
                            'disc_start': 1,
                            'kl_weight': 1.0e-06,
                            'disc_weight': 0.5,
                            'vf_weight': 0.1,
                            'adaptive_vf': True,
                            'vf_loss_type': 'combined_v3',
                            'distmat_margin': 0.25,
                            'cos_margin': 0.5
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
            }
        }

    def _transfer_xl_to_l_weights(self, dit_model):
        """XL→L权重智能迁移"""
        # 检查可能的XL权重路径
        possible_paths = [
            Path("/kaggle/input/lightningdit-xl/lightningdit-xl-imagenet256-64ep.pt"),
            Path("/kaggle/input/models/lightningdit-xl-imagenet256-64ep.pt"),
            Path("/kaggle/working/models/lightningdit-xl-imagenet256-64ep.pt"),
        ]

        xl_checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                xl_checkpoint_path = path
                break

        if xl_checkpoint_path is None:
            logger.warning("⚠️ XL预训练权重不存在，使用随机初始化")
            return

        logger.info("🔄 进行XL→L权重迁移...")

        try:
            xl_state = torch.load(xl_checkpoint_path, map_location='cpu')
            if 'model' in xl_state:
                xl_state = xl_state['model']
            elif 'state_dict' in xl_state:
                xl_state = xl_state['state_dict']

            # 智能权重映射
            l_state = dit_model.state_dict()
            transferred = 0

            for name, param in l_state.items():
                if name in xl_state and param.shape == xl_state[name].shape:
                    l_state[name] = xl_state[name]
                    transferred += 1
                elif name.startswith('blocks.') and transferred < 24:  # L模型24层
                    # 层级映射逻辑
                    layer_idx = int(name.split('.')[1])
                    if layer_idx < 28:  # XL模型28层
                        xl_name = name
                        if xl_name in xl_state and param.shape == xl_state[xl_name].shape:
                            l_state[name] = xl_state[xl_name]
                            transferred += 1

            dit_model.load_state_dict(l_state)
            logger.info(f"✅ 权重迁移完成，成功迁移 {transferred} 个参数")

        except Exception as e:
            logger.warning(f"⚠️ 权重迁移失败: {e}，使用随机初始化")


class MicroDopplerTrainer:
    """微多普勒DiT训练器 - 基于官方训练循环"""

    def __init__(self, config: MicroDopplerConfig, data_manager: MicroDopplerDataManager,
                 model_manager: DiTModelManager):
        self.config = config
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.device = None
        self.use_multi_gpu = False

    def _safe_transport_call(self, transport, dit_model, latents, model_kwargs):
        """安全的transport调用，确保设备一致性"""
        device = latents.device

        # 临时设置默认设备
        original_device = torch.cuda.current_device()
        try:
            if device.type == 'cuda':
                torch.cuda.set_device(device)

            # 确保所有输入在同一设备
            if hasattr(dit_model, 'module'):
                # DataParallel情况
                latents = latents.to('cuda:0')
                if 'y' in model_kwargs:
                    model_kwargs['y'] = model_kwargs['y'].to('cuda:0')

            return transport.training_losses(dit_model, latents, model_kwargs)
        finally:
            # 恢复原始设备
            torch.cuda.set_device(original_device)

    def train(self):
        """主训练流程"""
        logger.info("\n" + "="*60)
        logger.info("🚀 开始微多普勒DiT训练")
        logger.info("="*60)

        # 1. 设置GPU环境
        self.use_multi_gpu, self.device = setup_kaggle_multi_gpu()

        # 2. 加载数据划分信息
        self.data_manager.load_split_info()

        # 3. 加载VA-VAE并编码数据
        vae = self.model_manager.load_vae(self.device)
        self.data_manager.encode_to_latents(vae, self.device)
        del vae  # 释放显存
        torch.cuda.empty_cache()

        # 4. 创建数据加载器
        train_loader, val_loader = self.data_manager.create_dataloaders()

        # 5. 创建DiT模型
        dit_model = self.model_manager.create_dit_model(self.device, self.use_multi_gpu)

        # 6. 设置训练组件
        optimizer, scheduler, transport = self.model_manager.setup_training_components(dit_model)

        # 7. 训练循环
        self._training_loop(dit_model, train_loader, val_loader, optimizer, scheduler, transport)

    def _training_loop(self, dit_model, train_loader, val_loader, optimizer, scheduler, transport):
        """训练循环"""
        best_val_loss = float('inf')
        patience_counter = 0
        # 修复GradScaler弃用警告
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            # 回退到旧版本
            scaler = torch.cuda.amp.GradScaler()

        logger.info(f"\n🎯 开始训练循环:")
        logger.info(f"   总轮数: {self.config.num_epochs}")
        logger.info(f"   早停耐心: {self.config.patience}")
        logger.info(f"   梯度累积: {self.config.gradient_accumulation_steps}")

        for epoch in range(self.config.num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(dit_model, train_loader, optimizer, scheduler, transport, scaler, epoch)

            # 验证阶段
            val_loss = self._validate_epoch(dit_model, val_loader, transport, epoch)

            # 早停检查
            if val_loss < best_val_loss - self.config.min_improvement:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model(dit_model, optimizer, scheduler, epoch, val_loss)
            else:
                patience_counter += 1

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                       f"Patience: {patience_counter}/{self.config.patience}")

            # 早停
            if patience_counter >= self.config.patience:
                logger.info(f"\n🛑 早停触发！连续{self.config.patience}轮验证损失未改善")
                break

        logger.info(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")

    def _train_epoch(self, dit_model, train_loader, optimizer, scheduler, transport, scaler, epoch):
        """单个训练轮次"""
        dit_model.train()
        total_loss = 0
        num_batches = 0

        optimizer.zero_grad()

        with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
            for batch_idx, (latents, labels) in enumerate(pbar):
                # 确保数据在正确设备上
                if hasattr(dit_model, 'module'):
                    # DataParallel情况下，数据必须在cuda:0
                    latents = latents.to('cuda:0')
                    labels = labels.to('cuda:0')
                else:
                    latents = latents.to(self.device)
                    labels = labels.to(self.device)

                # 修复autocast弃用警告
                try:
                    autocast_context = torch.amp.autocast('cuda')
                except AttributeError:
                    autocast_context = torch.cuda.amp.autocast()

                with autocast_context:
                    # 模型预测 - 确保所有张量在同一设备
                    model_kwargs = {"y": labels}

                    # 确保dit_model在正确设备上
                    if hasattr(dit_model, 'module'):
                        # DataParallel情况下，确保输入在cuda:0
                        latents = latents.to('cuda:0')
                        labels = labels.to('cuda:0')
                        model_kwargs = {"y": labels}

                    # 使用安全的transport调用
                    loss_dict = self._safe_transport_call(transport, dit_model, latents, model_kwargs)

                    # 损失计算
                    if 'cos_loss' in loss_dict and self.config.use_cosine_loss:
                        mse_loss = loss_dict["loss"].mean()
                        cos_loss = loss_dict["cos_loss"].mean()
                        loss = cos_loss + mse_loss
                    else:
                        loss = loss_dict["loss"].mean()

                    loss = loss / self.config.gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                # 梯度累积
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(dit_model.parameters(), self.config.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / num_batches

    def _validate_epoch(self, dit_model, val_loader, transport, epoch):
        """验证轮次"""
        dit_model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]") as pbar:
                for latents, labels in pbar:
                    # 确保所有张量在正确设备上
                    if hasattr(dit_model, 'module'):
                        # DataParallel情况下，使用cuda:0
                        latents = latents.to('cuda:0')
                        labels = labels.to('cuda:0')
                    else:
                        latents = latents.to(self.device)
                        labels = labels.to(self.device)

                    model_kwargs = {"y": labels}

                    # 使用安全的transport调用
                    loss_dict = self._safe_transport_call(transport, dit_model, latents, model_kwargs)

                    if 'cos_loss' in loss_dict and self.config.use_cosine_loss:
                        mse_loss = loss_dict["loss"].mean()
                        cos_loss = loss_dict["cos_loss"].mean()
                        loss = cos_loss + mse_loss
                    else:
                        loss = loss_dict["loss"].mean()

                    total_loss += loss.item()
                    num_batches += 1

                    pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        return total_loss / num_batches

    def _save_best_model(self, dit_model, optimizer, scheduler, epoch, val_loss):
        """保存最佳模型"""
        # 删除旧的最佳模型
        old_models = list(Path(self.config.output_dir).glob("best_dit_epoch_*.pt"))
        for old_model in old_models:
            old_model.unlink()

        # 保存新模型
        model_path = Path(self.config.output_dir) / f"best_dit_epoch_{epoch+1}.pt"

        # 获取模型状态字典
        if hasattr(dit_model, 'module'):  # DataParallel
            model_state = dit_model.module.state_dict()
        else:
            model_state = dit_model.state_dict()

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, model_path)

        logger.info(f"✅ 保存最佳模型: {model_path} (val_loss: {val_loss:.6f})")


def test_setup():
    """测试设置是否正常"""
    logger.info("🧪 测试环境设置...")

    # 测试GPU
    use_multi_gpu, device = setup_kaggle_multi_gpu()
    logger.info(f"   多GPU支持: {use_multi_gpu}")
    logger.info(f"   主设备: {device}")

    # 测试配置
    config = MicroDopplerConfig()
    logger.info(f"✅ 配置创建成功")

    # 测试数据管理器
    data_manager = MicroDopplerDataManager(config)
    logger.info(f"✅ 数据管理器创建成功")

    # 测试模型管理器
    model_manager = DiTModelManager(config)
    logger.info(f"✅ 模型管理器创建成功")

    # 避免未使用变量警告
    _ = data_manager, model_manager

    # 检查必要文件
    required_files = [
        Path(config.data_dir),
        Path(config.vae_checkpoint),
        Path(config.split_file)
    ]

    for file_path in required_files:
        if file_path.exists():
            logger.info(f"✅ 找到必要文件: {file_path}")
        else:
            logger.warning(f"⚠️ 缺少文件: {file_path}")

    logger.info("🎯 环境测试完成！")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='微多普勒DiT训练')
    parser.add_argument('--test', action='store_true', help='只运行测试，不开始训练')
    args = parser.parse_args()

    if args.test:
        test_setup()
        return

    logger.info("🎯 微多普勒DiT条件生成训练")

    # 创建配置
    config = MicroDopplerConfig()

    # 显示配置信息
    logger.info("\n📋 训练配置:")
    logger.info(f"   模型: LightningDiT-{config.model_type} (24层, 1024维)")
    logger.info(f"   用户类别: {config.num_classes}")
    logger.info(f"   批次大小: {config.batch_size}")
    logger.info(f"   梯度累积: {config.gradient_accumulation_steps}")
    logger.info(f"   有效批次: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"   学习率: {config.learning_rate}")
    logger.info(f"   总轮数: {config.num_epochs}")
    logger.info(f"   早停耐心: {config.patience}")

    # 创建管理器
    data_manager = MicroDopplerDataManager(config)
    model_manager = DiTModelManager(config)
    trainer = MicroDopplerTrainer(config, data_manager, model_manager)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"\n❌ 训练出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
