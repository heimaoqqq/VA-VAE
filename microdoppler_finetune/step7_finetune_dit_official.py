#!/usr/bin/env python3
"""
步骤7: 基于官方LightningDiT train.py的微调脚本
专门用于微多普勒数据集的条件生成训练
使用DataParallel实现Kaggle T4x2双GPU训练
"""

import os
import sys
import argparse
import yaml
import logging
import math
import numpy as np
from glob import glob
from copy import deepcopy
from time import time
from pathlib import Path
from collections import OrderedDict

# Suppress CUDA library registration warnings in Kaggle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Disable torch.compile and FX tracing to avoid conflicts with gradient checkpointing
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Memory optimization for Kaggle T4x2
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
import safetensors.torch as st
from safetensors import safe_open

# Completely disable torch compile and dynamo
torch._dynamo.disable()
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

# Disable torch._inductor optimizations that conflict with checkpointing
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.append('/kaggle/working/VA-VAE/LightningDiT')
from models.lightningdit import LightningDiT_models
from transport import create_transport

class MicroDopplerLatentDataset(Dataset):
    """微多普勒latent数据集 - 处理step6创建的批量格式数据"""
    def __init__(self, data_dir, latent_norm=False, latent_multiplier=0.18215):
        self.data_dir = Path(data_dir)
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 加载所有文件并构建索引
        print(f"Loading batched latent files from: {self.data_dir}")
        self.latent_files = list(self.data_dir.glob('*.safetensors'))
        
        if not self.latent_files:
            raise ValueError(f"No latent files found in {data_dir}")
        
        # 构建全局索引: (file_idx, sample_idx)
        self.sample_indices = []
        self.total_samples = 0
        
        for file_idx, latent_file in enumerate(self.latent_files):
            with safe_open(str(latent_file), framework="pt", device="cpu") as f:
                if 'latents' in f.keys():
                    latents = f.get_tensor("latents")
                    num_samples = latents.shape[0]
                    
                    # 添加该文件中的所有样本索引
                    for sample_idx in range(num_samples):
                        self.sample_indices.append((file_idx, sample_idx))
                    
                    self.total_samples += num_samples
                    print(f"File {latent_file.name}: {num_samples} samples")
        
        print(f"Dataset initialized with {self.total_samples} total samples from {len(self.latent_files)} files")
        
        # 加载统计信息
        stats_file = self.data_dir / 'latents_stats.pt'
        if stats_file.exists():
            stats = torch.load(stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            self.mean = torch.zeros(32)  # VA-VAE f16d32 has 32 channels
            self.std = torch.ones(32)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 获取对应的文件和样本索引
        file_idx, sample_idx = self.sample_indices[idx]
        latent_file = self.latent_files[file_idx]
        
        # 加载该文件的数据
        with safe_open(str(latent_file), framework="pt", device="cpu") as f:
            latents = f.get_tensor("latents")  # [N, 32, 16, 16]
            labels = f.get_tensor("labels")    # [N]
            
            # 提取单个样本
            latent = latents[sample_idx]  # [32, 16, 16] 
            user_id = labels[sample_idx].item()  # scalar
        
        # 应用归一化
        if self.latent_norm:
            latent = (latent - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
            latent = latent * self.latent_multiplier
        
        return latent, user_id

def do_train_dataparallel(train_config, device, gpu_count):
    """
    使用DataParallel的LightningDiT训练函数
    """
    print(f"Setting up training with {gpu_count} GPU(s)")
    
    # Setup experiment folder
    os.makedirs(train_config['train']['output_dir'], exist_ok=True)
    experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
    model_string_name = train_config['model']['model_type'].replace("/", "-")
    exp_name = train_config['train']['exp_name'] if 'exp_name' in train_config['train'] else f'{experiment_index:03d}-{model_string_name}'
    experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Setup TensorBoard
    tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
    os.makedirs(tensorboard_dir_log, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir_log)

    # Model setup
    downsample_ratio = 16  # VA-VAE f16d32
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

    # Load pretrained weights
    if train_config['train']['weight_init'] is not None:
        logger.info(f"Loading pretrained weights: {train_config['train']['weight_init']}")
        checkpoint = torch.load(train_config['train']['weight_init'], map_location='cpu')
        if "ema" in checkpoint:
            checkpoint = checkpoint["ema"]
        
        # Handle class number mismatch (ImageNet 1001 vs MicroDoppler 32)
        if 'y_embedder.embedding_table.weight' in checkpoint:
            checkpoint_classes = checkpoint['y_embedder.embedding_table.weight'].shape[0]
            model_classes = model.y_embedder.embedding_table.weight.shape[0]
            
            if checkpoint_classes != model_classes:
                logger.info(f"Class mismatch: checkpoint {checkpoint_classes} vs model {model_classes}, skipping y_embedder")
                del checkpoint['y_embedder.embedding_table.weight']
        
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Pretrained weights loaded successfully")

    # Move model to GPU
    model = model.to('cuda:0')  # 明确指定主设备
    
    if gpu_count > 1:
        logger.info(f"Using DataParallel on {gpu_count} GPUs with device_ids=[0,1]")
        # 显式指定设备ID，确保负载均衡
        model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    
    # Create EMA model
    ema = deepcopy(model.module if gpu_count > 1 else model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model.module if gpu_count > 1 else model, decay=0)
    
    model.train()
    ema.eval()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer and scaler for mixed precision training
    opt = torch.optim.AdamW(model.parameters(), 
                           lr=train_config['optimizer']['lr'], 
                           weight_decay=train_config['optimizer'].get('weight_decay', 0.0))
    scaler = torch.amp.GradScaler(device_type='cuda')

    # Setup datasets
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=train_config['train']['global_batch_size'],
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Training dataset: {len(dataset):,} samples")

    # Setup validation
    valid_loader = None
    if 'valid_path' in train_config['data'] and train_config['data']['valid_path'] is not None:
        valid_dataset = MicroDopplerLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_config['train']['global_batch_size'],
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Validation dataset: {len(valid_dataset):,} samples")

    # Setup transport (diffusion process)
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )
    
    # Setup training state
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    # Resume from checkpoint if requested
    if train_config['train'].get('resume', False):
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            
            if gpu_count > 1:
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
            
            ema.load_state_dict(checkpoint['ema'])
            opt.load_state_dict(checkpoint['opt'])
            train_steps = int(os.path.basename(latest_checkpoint).split('.')[0])
            logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    # Clear memory and set memory fraction
    torch.cuda.empty_cache()
    
    # Set memory fraction for balanced GPU usage - 更保守的内存分配
    if gpu_count > 1:
        torch.cuda.set_per_process_memory_fraction(0.40, device=0)  # 主GPU更保守
        torch.cuda.set_per_process_memory_fraction(0.45, device=1)  # 从GPU也保守一些
        
        # 预分配内存以避免碎片化
        torch.cuda.empty_cache()
        with torch.cuda.device(0):
            torch.cuda.memory.set_per_process_memory_fraction(0.40)
        with torch.cuda.device(1):
            torch.cuda.memory.set_per_process_memory_fraction(0.45)  
    logger.info("Starting training loop...")
    
    # Training loop
    while train_steps < train_config['train']['max_steps']:
        for x, y in loader:
            if train_steps >= train_config['train']['max_steps']:
                break
                
            # Memory management
            torch.cuda.empty_cache()
                
            # Move tensors to primary device (cuda:0) for DataParallel
            # Use FP16 to save memory
            x = x.to('cuda:0', dtype=torch.float16)
            y = y.to('cuda:0')
            
            model_kwargs = dict(y=y)
            
            # Use proper mixed precision training with GradScaler
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            
            # Scale loss for backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            if 'max_grad_norm' in train_config['optimizer']:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            
            # Step with scaler
            scaler.step(opt)
            scaler.update()
            
            # Update EMA
            if gpu_count > 1:
                update_ema(ema, model.module)
            else:
                update_ema(ema, model)
            
            # Clear cache after each step
            torch.cuda.empty_cache()

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % train_config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                writer.add_scalar('Loss/train', avg_loss, train_steps)
                
                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if gpu_count > 1:
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                    
                checkpoint = {
                    "model": model_state,
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "config": train_config,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Validation evaluation (optional)
                if valid_loader is not None:
                    logger.info(f"Evaluating at step {train_steps}")
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_x, val_y in valid_loader:
                            val_x = val_x.to(device)
                            val_y = val_y.to(device)
                            val_model_kwargs = dict(y=val_y)
                            val_loss_dict = transport.training_losses(model, val_x, val_model_kwargs)
                            val_loss += val_loss_dict["loss"].mean().item()
                            val_steps += 1
                            if val_steps >= 10:  # Limited validation steps
                                break
                    
                    avg_val_loss = val_loss / val_steps
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar('Loss/validation', avg_val_loss, train_steps)
                    model.train()
            
            if train_steps >= train_config['train']['max_steps']:
                break
        
        if train_steps >= train_config['train']['max_steps']:
            break

    logger.info("Training completed!")
    writer.close()
    
    # Save final checkpoint
    if gpu_count > 1:
        final_model_state = model.module.state_dict()
    else:
        final_model_state = model.state_dict()
        
    final_checkpoint = {
        "model": final_model_state,
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "config": train_config,
    }
    final_path = f"{checkpoint_dir}/final.pt"
    torch.save(final_checkpoint, final_path)
    logger.info(f"Final model saved to {final_path}")
    
    return model, ema

def load_weights_with_shape_check(model, checkpoint, rank=0):
    """从官方train.py复制的权重加载函数"""
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
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

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
    """
    if accelerator.process_index == 0:  # real logger
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

@torch.no_grad()
def evaluate(model, loader, device, transport, t_range):
    """Evaluate the model on validation set"""
    model.eval()
    val_loss = 0.0
    num_samples = 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        val_loss += loss_dict["loss"].mean().item() * x.shape[0]
        num_samples += x.shape[0]
    
    model.train()
    return torch.tensor(val_loss / num_samples, device=device)

if __name__ == "__main__":
    # 读取配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_microdoppler_finetune.yaml')
    args = parser.parse_args()

    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count > 1:
        print(f"Using DataParallel on {gpu_count} GPUs")
        device = torch.device('cuda')
    else:
        print("Using single GPU")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    train_config = load_config(args.config)
    do_train_dataparallel(train_config, device, gpu_count)
