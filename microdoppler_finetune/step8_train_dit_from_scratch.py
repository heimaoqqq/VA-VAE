#!/usr/bin/env python3
"""
步骤8: 从头训练LightningDiT-Base模型用于微多普勒条件生成
基于官方train.py复制并修改，确保最大程度与官方一致
Training LightningDiT-Base with VA-VAE on micro-Doppler dataset
"""

import torch
# import torch.distributed as dist  # 使用accelerator替代
import torch.backends.cuda
import torch.backends.cudnn
# from torch.nn.parallel import DistributedDataParallel as DDP  # 使用accelerator替代
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

import math
import yaml
import json
import numpy as np
import logging
import os
import sys
import argparse
import time
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from pathlib import Path
import torchvision

# 添加LightningDiT路径
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')
sys.path.append('/kaggle/working/LightningDiT')

from utils_scheduler import create_scheduler
from utils_regularization import (
    LabelSmoothing, mixup_data, orthogonal_regularization,
    ContrastiveRegularizer, EarlyStopping
)

# 禁用torch.compile以避免Kaggle环境问题
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
torch._dynamo.disable()

from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator

# 导入我们自己的微多普勒数据集类
sys.path.append('/kaggle/working/microdoppler_finetune')

# 直接定义简化的数据集类，避免导入问题
from safetensors.torch import load_file
from torch.utils.data import Dataset

class MicroDopplerLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = Path(data_dir)
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 获取所有latent文件
        self.latent_files = sorted(list(self.data_dir.glob("*.safetensors")))
        
        if len(self.latent_files) == 0:
            raise ValueError(f"No safetensors files found in {data_dir}")
        
        print(f"Found {len(self.latent_files)} latent files in {data_dir}")
        
        # 预加载所有数据到内存
        self.latents = []
        self.labels = []
        
        for file_path in self.latent_files:
            data = load_file(str(file_path))
            
            # 获取latent和labels（支持我们的batch格式）
            latent = data['latents']  # [B, C, H, W]
            labels_batch = data['labels']  # [B]
            
            # 添加每个样本
            for i in range(latent.shape[0]):
                sample_latent = latent[i]  # [C, H, W]
                sample_label = labels_batch[i]
                
                # 暂时不在这里归一化，保持原始latent
                # 归一化将在计算全局统计后进行
                self.latents.append(sample_latent)
                self.labels.append(sample_label.long())
        
        # 统计类别分布
        unique_labels = torch.stack(self.labels).unique()
        print(f"Dataset contains {len(self.latents)} samples with {len(unique_labels)} unique classes")
        
        # 计算latent统计信息用于采样
        self._compute_latent_stats()
        
    def _compute_latent_stats(self):
        """计算数据集中latent的统计信息并进行归一化"""
        if len(self.latents) > 0:
            # 将所有latent堆叠并计算统计
            all_latents = torch.stack(self.latents)  # [N, C, H, W]
            # 计算原始数据的统计信息（用于反归一化）
            # 使用keepdim=True保持维度[1, C, 1, 1]与官方一致
            self.latent_mean = all_latents.mean(dim=[0, 2, 3], keepdim=True)  # [1, C, 1, 1]
            self.latent_std = all_latents.std(dim=[0, 2, 3], keepdim=True)    # [1, C, 1, 1]
            
            # 添加原始空间统计显示
            original_mean = all_latents.mean().item()
            original_std = all_latents.std().item()
            print(f"\n📊 原始latent空间统计 (加载后):")
            print(f"   Mean: {original_mean:.4f}")
            print(f"   Std:  {original_std:.4f}")
            
            # 按照官方LightningDiT方式处理
            for i in range(len(self.latents)):
                if self.latent_norm:
                    # 归一化到N(0,1): (x - mean) / std
                    # self.latents[i]形状是[C, H, W]，需要正确广播
                    # latent_mean/std形状是[1, C, 1, 1]，squeeze(0)后是[C, 1, 1]
                    mean_broadcast = self.latent_mean.squeeze(0)  # [C, 1, 1]
                    std_broadcast = self.latent_std.squeeze(0)    # [C, 1, 1] 
                    self.latents[i] = (self.latents[i] - mean_broadcast) / (std_broadcast + 1e-8)
                    
                    # 调试：检查归一化结果
                    if i == 0:  # 只检查第一个样本避免过多输出
                        norm_mean = self.latents[i].mean().item()
                        norm_std = self.latents[i].std().item()
                        print(f"\n✅ 归一化验证（训练数据样本0）:")
                        print(f"   归一化后: mean={norm_mean:.4f}, std={norm_std:.4f}")
                        print(f"   状态: {'正常' if abs(norm_mean) < 0.1 and abs(norm_std - 1.0) < 0.1 else '异常'}")
                
                # 官方总是乘multiplier（无论是否归一化）
                self.latents[i] = self.latents[i] * self.latent_multiplier
                
            # 检查训练空间的统计（缩放后）
            all_scaled = torch.stack(self.latents)  # 重新堆叠缩放后的数据
            train_mean = all_scaled.mean().item()
            train_std = all_scaled.std().item()
            print(f"\n🎯 训练空间统计 (multiplier={self.latent_multiplier}应用后):")
            print(f"   Mean: {train_mean:.4f}")
            print(f"   Std:  {train_std:.4f}")
            print(f"   {'✅ 接近目标' if abs(train_std - 1.0) < 0.2 else '⚠️ 偏离目标'} (目标std≈1.0)")
        else:
            # 默认值，保持[1, C, 1, 1]形状
            self.latent_mean = torch.zeros(1, 32, 1, 1)  # 假设32维latent
            self.latent_std = torch.ones(1, 32, 1, 1)
    
    def get_latent_stats(self):
        """返回latent的均值和标准差统计"""
        return self.latent_mean, self.latent_std
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        label = self.labels[idx]
        
        # 移除数据增强 - 微多普勒时频图需要保持精确的物理意义
        # 噪声可能破坏关键的时频特征和用户间的细微差别
        
        return latent, label

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT-Base for micro-Doppler generation.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, accelerator)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 16 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        class_dropout_prob=0.1,  # 官方默认值，用于训练时随机dropout类别条件
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    ).to(device)  # Move model to device immediately after creation

    # Create EMA model - must be on same device as model
    ema = deepcopy(model)
    
    # Explicitly move both models to device and ensure all parameters are there
    model = model.to(device)
    ema = ema.to(device)
    
    # load pretrained model (if provided)
    if 'weight_init' in train_config['train'] and train_config['train']['weight_init'] is not None:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    
    requires_grad(ema, False)
    
    # 创建并加载我们微调的VA-VAE
    from tokenizer.vavae import VA_VAE
    # 使用专门为DiT训练准备的配置文件（不包含adaptive_vf）
    vae_config_path = '/kaggle/working/VA-VAE/microdoppler_finetune/vavae_config_for_dit.yaml'
    vae = VA_VAE(vae_config_path)
    
    # 加载微调的VA-VAE权重
    vae_checkpoint_path = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    if os.path.exists(vae_checkpoint_path):
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        
        # 从checkpoint中提取state_dict
        if 'state_dict' in vae_checkpoint:
            state_dict = vae_checkpoint['state_dict']
        else:
            state_dict = vae_checkpoint
        
        # 过滤出只属于VAE的权重（排除loss、foundation_model等）
        vae_state_dict = {}
        for key, value in state_dict.items():
            # 只保留不以loss、foundation_model、linear_proj开头的权重
            if not key.startswith('loss.') and not key.startswith('foundation_model.') and key != 'linear_proj.weight':
                vae_state_dict[key] = value
        
        # 加载过滤后的权重
        vae.model.load_state_dict(vae_state_dict, strict=True)
        
        if accelerator.is_main_process:
            logger.info(f"Loaded custom VA-VAE from {vae_checkpoint_path}")
            logger.info(f"Filtered {len(state_dict) - len(vae_state_dict)} non-VAE keys from checkpoint")
    else:
        if accelerator.is_main_process:
            logger.warning(f"VA-VAE checkpoint not found at {vae_checkpoint_path}, using default weights")
    
    # VA_VAE类内部已经处理了.cuda()和.eval()，不需要额外调用
    # 冻结VA-VAE权重，仅用于编码解码
    for param in vae.model.parameters():
        param.requires_grad = False
    if accelerator.is_main_process:
        logger.info("VA-VAE weights frozen for inference only")
    
    # 创建transport和优化器（在accelerator.prepare之前）
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    
    # 设置优化器参数 - 匹配官方默认值
    train_config['optimizer']['lr'] = train_config.get('optimizer', {}).get('lr', 1e-4)
    train_config['optimizer']['beta2'] = train_config.get('optimizer', {}).get('beta2', 0.999)  # 从config读取
    train_config['optimizer']['max_grad_norm'] = train_config.get('optimizer', {}).get('max_grad_norm', 1.0)
    
    # 加强权重衰减防止过拟合
    weight_decay = train_config.get('optimizer', {}).get('weight_decay', 1e-3)  # 大幅增加L2正则化
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config['optimizer']['lr'], 
        weight_decay=weight_decay,  # 使用权重衰减防止过拟合
        betas=(0.9, train_config['optimizer']['beta2'])  # 官方beta1固定为0.9
    )
    
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}, weight_decay={weight_decay}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
        logger.info(f"🛡️ Regularization: weight_decay={weight_decay}")
    
    # Setup data - 使用兼容的微多普勒latent数据集
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],  # 使用官方参数名data_dir
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data'].get('latent_multiplier', 1.0),  # 使用1.0作为默认值
    )
    # dataset.training = True  # 移除数据增强 - 保持微多普勒信号的精确性
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # 创建学习率调度器（现在loader已经定义）
    total_steps = train_config['train']['max_epochs'] * len(loader)
    scheduler = None  # 初始化scheduler变量
    if 'scheduler' in train_config:
        scheduler_config = train_config['scheduler']
        if scheduler_config['type'] == 'cosine':
            # 余弦退火，带线性warmup
            def lr_lambda(current_step):
                warmup_steps = scheduler_config.get('warmup_steps', 500)  # 标准warmup
                min_lr_ratio = float(scheduler_config.get('min_lr', 1e-6)) / float(train_config['optimizer']['lr'])
                
                if current_step < warmup_steps:
                    # 线性warmup
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # 余弦退火到min_lr
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            print(f"📊 Using cosine scheduler with {scheduler_config.get('warmup_steps', 500)} warmup steps")               
        else:
            scheduler = None
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
        if scheduler:
            logger.info(f"Using learning rate scheduler: {train_config['train'].get('scheduler', {}).get('type', 'cosine')}")
            logger.info(f"Total training steps: {total_steps:,}")
    
    # 验证集
    if 'valid_path' in train_config['data']:
        valid_dataset = MicroDopplerLatentDataset(
            data_dir=train_config['data']['valid_path'],  # 使用官方参数名data_dir
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data'].get('latent_multiplier', 1.0),  # 使用1.0作为默认值
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,  # 官方验证集也shuffle
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True  # 官方训练和验证都drop_last
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
                logger.info(f"Starting from epoch {start_epoch}, step {train_steps}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
    
    model.train()
    running_loss = 0.0
    step_count = 0
    start_time = time.time()
    
    best_loss = float('inf')
    
    # Gradient accumulation设置 - 修复配置
    gradient_accumulation_steps = train_config['train'].get('gradient_accumulation_steps', 1)  # 默认无累积
    global_batch_size = train_config['train']['global_batch_size']  # 16
    world_size = accelerator.num_processes  # 2 (T4x2)
    per_device_batch_size = global_batch_size // world_size  # 8 per GPU
    
    if accelerator.is_main_process:
        logger.info(f"Batch size配置:")
        logger.info(f"  Global batch size: {global_batch_size}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Per device batch size: {per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {global_batch_size * gradient_accumulation_steps}")
    model, opt, loader = accelerator.prepare(model, opt, loader)
    if 'valid_path' in train_config['data']:
        valid_loader = accelerator.prepare(valid_loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    start_epoch = 0

    # 重新创建EMA模型确保设备同步（accelerator.prepare后）
    with accelerator.main_process_first():
        ema = deepcopy(accelerator.unwrap_model(model))
        ema = ema.to(accelerator.device)
        requires_grad(ema, False)
        # Initialize EMA with model weights
        update_ema(ema, accelerator.unwrap_model(model), decay=0)
        if accelerator.is_main_process:
            logger.info(f"Using checkpointing: {train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True}")

    # 早停参数 - 加强过拟合控制
    patience = train_config['train'].get('patience', 15)  # 降低patience
    min_delta = train_config['train'].get('min_delta', 5e-4)  # 增加最小改善阈值
    overfitting_threshold = train_config['train'].get('overfitting_threshold', 1.2)  # 过拟合阈值
    num_epochs = train_config['train'].get('max_epochs', 200)
    
    # 学习率调度器 - 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, 
        T_max=num_epochs, 
        eta_min=1e-6
    )
    if accelerator.is_main_process:
        logger.info(f"📉 Learning rate scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min=1e-6)")
    steps_per_epoch = len(loader)
    
    # 训练循环 - 改为基于epoch
    for epoch in range(start_epoch, num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
                
        # 每个epoch开始时验证数据分布
        if epoch == 0:
            print(f"\n🔍 首个Epoch数据分布验证:")
            # 从dataloader取一个批次进行验证
            for batch_idx, (z_check, y_check) in enumerate(loader):
                if batch_idx == 0:
                    z_mean = z_check.mean().item()
                    z_std = z_check.std().item()
                    print(f"   首批数据: mean={z_mean:.4f}, std={z_std:.4f}")
                    print(f"   状态: {'✅ 理想范围' if abs(z_std - 1.0) < 0.2 else '⚠️ 需要调整multiplier'}")
                    break
        
        # 只有DistributedSampler才有set_epoch方法
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        print(f"🔄 Training loader has {len(loader)} batches")
            
        epoch_loss = 0
        epoch_steps = 0
        running_loss = 0
        log_steps = 0
        start_time = time.time()
            
        if accelerator.is_main_process:
            print(f"🚀 Starting training loop for epoch {epoch}")
            
        # 使用tqdm显示训练进度
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process)
        
        for batch_idx, batch in pbar:
            if batch_idx == 0 and accelerator.is_main_process:
                logger.info(f"Training epoch {epoch+1}, batch shape: {batch[0].shape}, labels: {batch[1].shape}")
        
            try:
                device = accelerator.device
                    
                # Unpack batch - DataLoader返回(latents, labels)元组
                x, y = batch
                x = x.to(device)
                y = y.to(device, dtype=torch.long)
                    
                # 前向传播 (考虑梯度累积) - 修复损失计算与官方一致
                with accelerator.autocast():
                    loss_dict = transport.training_losses(model, x, model_kwargs=dict(y=y))
                    if 'cos_loss' in loss_dict:
                        # 官方组合损失：cosine + mse
                        mse_loss = loss_dict["loss"].mean()
                        cos_loss = loss_dict["cos_loss"].mean()
                        raw_loss = cos_loss + mse_loss
                        if accelerator.is_main_process and batch_idx % 100 == 0:
                            print(f"  MSE Loss: {mse_loss:.4f}, Cosine Loss: {cos_loss:.4f}, Total: {raw_loss:.4f}")
                    else:
                        raw_loss = loss_dict["loss"].mean()
                    loss = raw_loss / gradient_accumulation_steps  # 归一化损失用于反向传播
                    
                # 反向传播
                accelerator.backward(loss)
                
                # 只在累积足够梯度后更新
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪 - 完全匹配官方：使用sync_gradients条件
                    if 'max_grad_norm' in train_config['optimizer']:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
                    
                    # 优化器更新
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    
                    # EMA更新 - 使用accelerator.unwrap_model确保获取正确的模型
                    update_ema(ema, accelerator.unwrap_model(model))
                    
                    step_count += 1
                    
                # Log loss values:
                running_loss += raw_loss.item()  # 使用原始损失值
                epoch_loss += raw_loss.item()     # 使用原始损失值
                    
                log_steps += 1
                train_steps += 1
                epoch_steps += 1
                
                # 更新进度条显示当前batch的损失
                if accelerator.is_main_process:
                    pbar.set_postfix({'loss': f"{raw_loss.item():.4f}", 'lr': f"{opt.param_groups[0]['lr']:.2e}"})
                
                # 定期记录 (只在优化器更新后)
                if (batch_idx + 1) % gradient_accumulation_steps == 0 and train_steps % train_config['train']['log_every'] == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss = accelerator.gather(avg_loss).mean()  # 使用accelerator替代dist
                    avg_loss = avg_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"[Epoch {epoch+1}/{num_epochs}, Step {train_steps:07d}] Train Loss: {avg_loss:.4f}, LR: {opt.param_groups[0]['lr']:.2e}, Speed: {steps_per_sec:.2f} steps/sec")
                        writer.add_scalar('Loss/train', avg_loss, train_steps)
                        writer.add_scalar('Learning_rate', opt.param_groups[0]['lr'], train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time.time()
                    
                # 训练批次统计检查（每100步）
                if train_steps % 100 == 0:
                    batch_mean = x.mean().item()
                    batch_std = x.std().item()
                    if accelerator.is_main_process:
                        logger.info(f"📊 Batch统计 [Step {train_steps}]: mean={batch_mean:.4f}, std={batch_std:.4f}")
                        # 添加到tensorboard
                        writer.add_scalar('train/batch_mean', batch_mean, train_steps)
                        writer.add_scalar('train/batch_std', batch_std, train_steps)
            
            except Exception as e:
                if accelerator.is_main_process:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                continue

        # 关闭进度条以确保后续输出可见
        if accelerator.is_main_process:
            pbar.close()
        
        # Epoch结束，计算平均损失
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        
        # 更新学习率调度器
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"   📉 Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"   🔢 Total Steps: {train_steps}")
            print(f"   📚 Learning Rate: {opt.param_groups[0]['lr']:.2e}")
            logger.info(f"Epoch {epoch+1}/{num_epochs} Summary - Train Loss: {avg_epoch_loss:.4f}, LR: {opt.param_groups[0]['lr']:.2e}")
            
        # 每个epoch结束后在验证集上评估
        if 'valid_path' in train_config['data']:
            if accelerator.is_main_process:
                logger.info(f"Evaluating at epoch {epoch+1}")
            
            model.eval()
            val_loss = evaluate(model, valid_loader, device, transport, accelerator)
            model.train()
            
            if accelerator.is_main_process:
                print(f"   🧪 Validation Loss: {val_loss:.4f}")
                writer.add_scalar('Loss/validation', val_loss, epoch+1)
                
                # 计算过拟合指标
                overfitting_ratio = val_loss / avg_epoch_loss if avg_epoch_loss > 0 else 1.0
                print(f"   📈 Overfitting Ratio (val/train): {overfitting_ratio:.3f}")
                
                # 过拟合状态判断
                if overfitting_ratio < 1.1:
                    print(f"   ✅ Good fit - model generalizing well")
                elif overfitting_ratio < 1.3:
                    print(f"   ⚠️  Slight overfitting - still acceptable")
                else:
                    print(f"   🚨 Overfitting detected - consider regularization")
                
                # 检查是否是最佳模型 - 更严格的过拟合控制
                # 只在没有过拟合时保存模型
                if val_loss < best_val_loss - min_delta and overfitting_ratio < overfitting_threshold:
                    improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 100
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"   🏆 New best model! Improvement: {improvement:.2f}%")
                    logger.info(f"New best model - Val Loss: {val_loss:.4f}, Improvement: {improvement:.2f}%")
                    
                    # 删除旧的最佳模型
                    best_checkpoint_path = f"{checkpoint_dir}/best_model.pt"
                    if os.path.exists(best_checkpoint_path):
                        os.remove(best_checkpoint_path)
                        print(f"   🗑️  Removed old best model")
                    
                    # 保存新的最佳模型
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                    }
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"   💾 New best model saved: {best_checkpoint_path}")
                else:
                    patience_counter += 1
                    gap = val_loss - best_val_loss
                    logger.info(f"  ⚠️ No improvement. Patience: {patience_counter}/{patience}, Gap: {gap:.5f}")
                    
                    # 更早的过拟合干预
                    if overfitting_ratio > overfitting_threshold and patience_counter == 5:
                        # 手动降低学习率
                        for param_group in opt.param_groups:
                            param_group['lr'] *= 0.3  # 更大幅度降低
                        logger.info(f"  📉 Early overfitting intervention - LR reduced by 70%")
                    
                    # 早停条件：更严格的过拟合控制
                    if patience_counter >= patience:
                        if overfitting_ratio > 1.4:
                            logger.info(f"  🛑 Early stopping - Overfitting ratio {overfitting_ratio:.3f} > 1.4")
                            break
                        else:
                            logger.info(f"  🛑 Early stopping - No validation improvement for {patience} epochs")
                            break
                    
                    # 额外的过拟合检查
                    elif overfitting_ratio > 1.5:
                        logger.info(f"  🛑 Emergency stop - Severe overfitting ratio {overfitting_ratio:.3f}")
                        break
                
                logger.info(f"{'='*50}")
        
        # 生成demo样本
        if True:  # 每个epoch都生成
            sample_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/demo_samples"
            # 使用EMA模型进行生成（关键修复：之前错误地使用了model而不是ema）
            generate_demo_samples(ema, vae, transport, device, accelerator, train_config, epoch, sample_dir)
            if accelerator.is_main_process:
                print(f"{'='*60}")
        
        # 定期保存checkpoint
        if (epoch + 1) % train_config['train'].get('ckpt_every_epoch', 10) == 0:
            if accelerator.is_main_process:
                # 清理旧的epoch checkpoint（只保留最近3个）
                epoch_checkpoints = glob(f"{checkpoint_dir}/epoch_*.pt")
                epoch_checkpoints.sort()
                if len(epoch_checkpoints) >= 3:
                    for old_ckpt in epoch_checkpoints[:-2]:  # 保留最后2个
                        os.remove(old_ckpt)
                        logger.info(f"Removed old checkpoint: {old_ckpt}")
                
                checkpoint = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "config": train_config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()  # 使用accelerator替代dist.barrier()

    if accelerator.is_main_process:
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return accelerator


@torch.no_grad()
def generate_demo_samples(model, vae, transport, device, accelerator, train_config, epoch, sample_dir):
    """生成演示样本，基于官方inference.py实现
    
    注意：model参数应该是EMA模型，不是训练模型
    """
    model.eval()
    
    # 采样配置 - 从配置文件读取
    cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0.11)
    timestep_shift = train_config['sample'].get('timestep_shift', 0.0)
    num_samples = 8  # 生成8个样本做成2x4网格
    
    # 创建sampler
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['sample']['sampling_method'],
        num_steps=train_config['sample']['num_sampling_steps'],
        atol=train_config['sample']['atol'],
        rtol=train_config['sample']['rtol'],
        reverse=train_config['sample']['reverse'],
        timestep_shift=timestep_shift,
    )
    
    # 获取潜在空间统计信息（用于反归一化）
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'],
        latent_multiplier=train_config['data'].get('latent_multiplier', 1.0),
    )
    
    # 获取统计信息 - 注意dataset返回的是原始统计（未应用multiplier）
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data'].get('latent_multiplier', 1.0)
    
    if latent_mean is None or latent_std is None:
        print("⚠️ 警告：无法获取latent统计信息，将无法正确反归一化")
        latent_mean = torch.zeros(1, 32, 1, 1, device=device)  # VA-VAE f16d32是32通道
        latent_std = torch.ones(1, 32, 1, 1, device=device)
    else:
        # 将统计信息移到正确设备
        latent_mean = latent_mean.to(device)  # [1, C, 1, 1]
        latent_std = latent_std.to(device)    # [1, C, 1, 1]
        
        # 验证统计信息的正确性
        # dataset.get_latent_stats()返回的是原始latent统计，无需额外计算
        orig_mean = latent_mean.mean().item()
        orig_std = latent_std.mean().item()
        
        # 验证训练空间数据统计（经过归一化+multiplier处理后）
        sample_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        sample_batch = next(iter(sample_loader))
        actual_mean = sample_batch[0].mean().item()  # 训练空间实际值
        actual_std = sample_batch[0].std().item()    # 训练空间实际值
        
        print(f"\n💾 生成时统计信息验证:")
        print(f"   原始VAE空间: mean={orig_mean:.4f}, std={orig_std:.4f}")
        print(f"   训练空间（实测）: mean={actual_mean:.4f}, std={actual_std:.4f}")
        print(f"   配置: latent_norm={train_config['data']['latent_norm']}, multiplier={latent_multiplier}")
        
        if train_config['data']['latent_norm']:
            # 归一化模式：训练空间应该接近N(0,1)*multiplier
            expected_mean = 0.0 * latent_multiplier  # 0
            expected_std = 1.0 * latent_multiplier   # 1.0
            print(f"   预期训练空间: mean≈{expected_mean:.1f}, std≈{expected_std:.1f}")
            print(f"   统计验证: {'✅' if abs(actual_mean) < 0.2 and abs(actual_std - expected_std) < 0.2 else '⚠️ 异常'}")
        else:
            # 非归一化模式：仅应用multiplier
            print(f"   模式: 仅缩放（原始×{latent_multiplier}）")
            print(f"   统计验证: {'✅' if abs(actual_std - orig_std * latent_multiplier) < 0.1 else '⚠️ 异常'}")
        
    
    if accelerator.is_main_process:
        print(f"🎨 Generating demo samples for epoch {epoch+1}...")
        
        images = []
        # 生成不同类别的样本
        demo_labels = [0, 1, 5, 10, 15, 20, 25, 30]  # 选择8个不同类别
        
        for label in demo_labels:
            # 创建噪声和标签
            latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
            # 注意：这里的model已经是EMA模型
            unwrapped_model = model if not hasattr(model, 'module') else model.module
            z = torch.randn(1, unwrapped_model.in_channels, latent_size, latent_size, device=device)
            y = torch.tensor([label], device=device)
            
            # CFG设置
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([train_config['data']['num_classes']], device=device)  # 使用num_classes作为null类别（对于31个用户，这里是31）
            y = torch.cat([y, y_null], 0)
            
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
            model_fn = unwrapped_model.forward_with_cfg
            
            # 采样
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)  # 移除null class样本
            
            # 解码流程 - 根据官方img_latent_dataset.py代码分析
            # 训练时: (latent - mean) / std * multiplier
            # 推理时: 使用官方inference.py的公式（数学等价但写法不同）
            
            if train_config['data']['latent_norm']:
                # 官方标准流程：反归一化从训练空间还原到VAE latent空间
                # 官方公式：(samples * std) / multiplier + mean
                samples_for_decode = (samples * latent_std) / latent_multiplier + latent_mean
                
                # 首次生成时显示反归一化验证
                if label == demo_labels[0]:
                    print(f"\n🔄 反归一化验证（首个生成样本）:")
                    print(f"   生成样本（训练空间）: mean={samples.mean():.4f}, std={samples.std():.4f}")
                    print(f"   还原后（VAE空间）: mean={samples_for_decode.mean():.4f}, std={samples_for_decode.std():.4f}")
            else:
                # 如果禁用归一化，仅应用multiplier的逆操作
                samples_for_decode = samples / latent_multiplier
                
                # 首次生成时显示验证信息
                if label == demo_labels[0]:
                    print(f"\n🔄 解码验证（首个样本）:")
                    print(f"   生成样本: mean={samples.mean():.4f}, std={samples.std():.4f}")
                    print(f"   还原后: mean={samples_for_decode.mean():.4f}, std={samples_for_decode.std():.4f}")
                    print(f"   模式: 仅反缩放（/multiplier={latent_multiplier}）")
            
            # VAE解码为图像
            with torch.no_grad():
                # 使用VA_VAE的decode_to_images方法，直接返回numpy数组
                images_decoded = vae.decode_to_images(samples_for_decode)  # 返回[B, H, W, C] numpy数组
                image = images_decoded[0]  # 取第一个图像 [H, W, C]
            images.append(image)
        
        # 创建2x4网格
        h, w = images[0].shape[:2]
        grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
        for idx, image in enumerate(images):
            i, j = divmod(idx, 4)  # 2x4网格位置
            grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
        
        # 保存网格图像
        os.makedirs(sample_dir, exist_ok=True)
        grid_path = f"{sample_dir}/epoch_{epoch+1:03d}_samples.png"
        Image.fromarray(grid).save(grid_path)
        print(f"💾 Demo samples saved to: {grid_path}")
    
    model.train()

@torch.no_grad()
def evaluate(model, valid_loader, device, transport, accelerator):
    """评估模型在验证集上的性能"""
    total_loss = 0
    total_steps = 0
    
    for x, y in tqdm(valid_loader, desc="Evaluating", disable=not accelerator.is_main_process):
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        
        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        
        if 'cos_loss' in loss_dict:
            # 与训练一致的组合损失计算
            mse_loss = loss_dict["loss"].mean()
            cos_loss = loss_dict["cos_loss"].mean()
            loss = cos_loss + mse_loss
        else:
            loss = loss_dict["loss"].mean()
            
        total_loss += loss.item()
        total_steps += 1
    
    avg_loss = total_loss / total_steps
    
    # All-reduce across processes using accelerator
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss_tensor = accelerator.gather(avg_loss_tensor).mean()
    avg_loss = avg_loss_tensor.item()
    
    return avg_loss


def load_weights_with_shape_check(model, checkpoint, rank=0):
    """加载权重并检查形状匹配"""
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first channels the same
                weight = torch.zeros_like(model_state_dict[name])
                min_channels = min(param.shape[1], weight.shape[1])
                weight[:, :min_channels] = param[:, :min_channels]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    官方LightningDiT的EMA更新策略
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        name = name.replace("module.", "")
        # 指数移动平均更新
        if name in ema_params:
            # Ensure both tensors are on the same device
            ema_param = ema_params[name]
            if ema_param.device != param.data.device:
                ema_param.data = ema_param.data.to(param.data.device)
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger(logging_dir, accelerator):
    """
    Create a logger that writes to a log file and stdout.
    使用accelerator判断是否为主进程，避免分布式初始化问题
    """
    if accelerator.is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_dit_base.yaml')
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='bf16')  # 启用混合精度和多GPU
    train_config = load_config(args.config)
    
    # 打印accelerator信息
    if accelerator.is_main_process:
        print(f"🚀 Accelerator setup - Devices: {accelerator.num_processes}, Mixed precision: {accelerator.mixed_precision}")
    
    do_train(train_config, accelerator)
