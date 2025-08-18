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

import math
import yaml
import json
import numpy as np
import logging
import os
import sys
import argparse
from time import time
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
                
                # 应用归一化和缩放
                if self.latent_norm:
                    sample_latent = (sample_latent - sample_latent.mean()) / (sample_latent.std() + 1e-8)
                sample_latent = sample_latent * self.latent_multiplier
                
                self.latents.append(sample_latent)
                self.labels.append(sample_label.long())
        
        # 统计类别分布
        unique_labels = torch.stack(self.labels).unique()
        print(f"Dataset contains {len(self.latents)} samples with {len(unique_labels)} unique classes")
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

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
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model (if provided)
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
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
    
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    
    # Setup data - 使用兼容的微多普勒latent数据集
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],  # 使用官方参数名data_dir
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0,
    )
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
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    # 验证集
    if 'valid_path' in train_config['data']:
        valid_dataset = MicroDopplerLatentDataset(
            data_dir=train_config['data']['valid_path'],  # 使用官方参数名data_dir
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,  # 验证集不shuffle
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
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
    model, opt, loader = accelerator.prepare(model, opt, loader)
    if 'valid_path' in train_config['data']:
        valid_loader = accelerator.prepare(valid_loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0

    # 重新创建EMA模型确保设备同步（accelerator.prepare后）
    ema = deepcopy(model).to(accelerator.device)
    requires_grad(ema, False)
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True}")

    # 早停参数
    patience = train_config['train'].get('patience', 20)
    max_epochs = train_config['train'].get('max_epochs', 200)
    steps_per_epoch = len(loader)
    
    # 训练循环 - 改为基于epoch
    for epoch in range(start_epoch, max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Starting Epoch {epoch+1}/{max_epochs}")
            print(f"🔄 Training loader has {len(loader)} batches")
            
        epoch_loss = 0
        epoch_steps = 0
        running_loss = 0
        log_steps = 0
        start_time = time()
        
        if accelerator.is_main_process:
            print(f"🚀 Starting training loop for epoch {epoch+1}")
        
        # 使用tqdm显示训练进度
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{max_epochs}", disable=not accelerator.is_main_process)
        
        for batch_idx, (x, y) in pbar:
            if batch_idx == 0 and accelerator.is_main_process:
                logger.info(f"Training epoch {epoch+1}, batch shape: {x.shape}, labels: {y.shape}")
                
            try:
                if accelerator.mixed_precision == 'no':
                    x = x.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.long)  # 确保标签是long类型
                else:
                    x = x.to(device)
                    y = y.to(device, dtype=torch.long)
                
                model_kwargs = dict(y=y)
                # 使用accelerator包装后的模型
                loss_dict = transport.training_losses(model, x, model_kwargs)
                if 'cos_loss' in loss_dict:
                    mse_loss = loss_dict["loss"].mean()
                    loss = loss_dict["cos_loss"].mean() + mse_loss
                else:
                    loss = loss_dict["loss"].mean()
                
                opt.zero_grad()
                accelerator.backward(loss)
                if 'max_grad_norm' in train_config['optimizer']:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
                opt.step()
                update_ema(ema, model)

                # Log loss values:
                if 'cos_loss' in loss_dict:
                    running_loss += mse_loss.item()
                    epoch_loss += mse_loss.item()
                else:
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    
                log_steps += 1
                train_steps += 1
                epoch_steps += 1
                
                # 更新进度条显示当前batch的损失
                if accelerator.is_main_process:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{opt.param_groups[0]['lr']:.2e}"})
                
                if train_steps % train_config['train']['log_every'] == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss = accelerator.gather(avg_loss).mean()  # 使用accelerator替代dist
                    avg_loss = avg_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"[Epoch {epoch+1}/{max_epochs}, Step {train_steps:07d}] Train Loss: {avg_loss:.4f}, LR: {opt.param_groups[0]['lr']:.2e}, Speed: {steps_per_sec:.2f} steps/sec")
                        writer.add_scalar('Loss/train', avg_loss, train_steps)
                        writer.add_scalar('Learning_rate', opt.param_groups[0]['lr'], train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()
                    
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                raise e

        # 关闭进度条以确保后续输出可见
        if accelerator.is_main_process:
            pbar.close()
        
        # Epoch结束，计算平均损失
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch+1}/{max_epochs} Summary:")
            print(f"   📉 Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"   🔢 Total Steps: {train_steps}")
            print(f"   📚 Learning Rate: {opt.param_groups[0]['lr']:.2e}")
            logger.info(f"Epoch {epoch+1}/{max_epochs} Summary - Train Loss: {avg_epoch_loss:.4f}, LR: {opt.param_groups[0]['lr']:.2e}")
            
        # 每个epoch结束进行验证
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
                
                # 检查是否是最佳模型
                if val_loss < best_val_loss:
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
                    logger.info(f"  ⚠️ No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        logger.info("  🛑 Early stopping triggered!")
                        break
                
                logger.info(f"{'='*50}")
        
        # 每10个epoch生成演示样本
        if (epoch + 1) % 10 == 0:
            sample_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/demo_samples"
            generate_demo_samples(model, vae, transport, device, accelerator, train_config, epoch, sample_dir)
            if accelerator.is_main_process:
                print(f"{'='*60}")
        
        # 定期保存checkpoint
        if (epoch + 1) % train_config['train'].get('ckpt_every_epoch', 10) == 0:
            if accelerator.is_main_process:
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
    """生成演示样本，基于官方inference.py实现"""
    model.eval()
    
    # 采样配置
    cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0.11)
    timestep_shift = train_config['sample'].get('timestep_shift', 0.1)
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
    
    # 获取潜在空间统计信息
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'],
        latent_multiplier=train_config['data']['latent_multiplier'],
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    
    # 如果get_latent_stats不存在，使用默认统计值
    if not hasattr(dataset, 'get_latent_stats'):
        # 使用训练时计算的统计值或默认值
        latent_mean = torch.zeros(train_config['model']['in_chans'], device=device)
        latent_std = torch.ones(train_config['model']['in_chans'], device=device)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    latent_multiplier = train_config['data']['latent_multiplier']
    
    if accelerator.is_main_process:
        print(f"🎨 Generating demo samples for epoch {epoch+1}...")
        
        images = []
        # 生成不同类别的样本
        demo_labels = [0, 1, 5, 10, 15, 20, 25, 30]  # 选择8个不同类别
        
        for label in demo_labels:
            # 创建噪声和标签
            latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
            unwrapped_model = accelerator.unwrap_model(model)
            z = torch.randn(1, unwrapped_model.in_channels, latent_size, latent_size, device=device)
            y = torch.tensor([label], device=device)
            
            # CFG设置
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([train_config['data']['num_classes']], device=device)
            y = torch.cat([y, y_null], 0)
            
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
            model_fn = unwrapped_model.forward_with_cfg
            
            # 采样
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)  # 移除null class样本
            
            # 反归一化
            samples = (samples * latent_std) / latent_multiplier + latent_mean
            
            # VAE解码为图像
            with torch.no_grad():
                image = vae.decode(samples)[0]  # 取第一个图像tensor
                # 转换为numpy数组 (0-1范围)
                image = torch.clamp(image, 0, 1)
                image = (image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
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
    return

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
            loss = loss_dict["loss"].mean()
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
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # 确保EMA参数与模型参数在同一设备上
        if name in ema_params:
            ema_param = ema_params[name]
            # 如果设备不匹配，将EMA参数移动到模型参数的设备
            if ema_param.device != param.device:
                ema_param.data = ema_param.data.to(param.device)
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
