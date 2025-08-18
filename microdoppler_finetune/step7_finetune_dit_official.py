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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import math
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from omegaconf import OmegaConf
from safetensors import safe_open
from PIL import Image
from pathlib import Path

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

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        rank=rank, 
        world_size=world_size
    )
    
    # 设置当前进程的GPU设备
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def create_logger(logging_dir, rank):
    """创建日志记录器（仅主进程输出到文件）"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

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

def do_train_ddp(rank, world_size, train_config):
    """
    使用DDP的LightningDiT训练函数
    """
    # 设置分布式环境
    setup_distributed(rank, world_size)
    device = f'cuda:{rank}'
    
    if rank == 0:
        print(f"Setting up DDP training with {world_size} GPU(s)")
    
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
    logger = create_logger(experiment_dir, rank)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Setup TensorBoard
    tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
    os.makedirs(tensorboard_dir_log, exist_ok=True)
    if rank == 0:
        writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
    
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
    # 移动模型到GPU并包装为DDP
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    if rank == 0:
        logger.info(f"Using DDP on {world_size} GPUs")
    
    # Create EMA model (只在主进程创建)
    if rank == 0:
        ema = deepcopy(model.module).to(device)
        requires_grad(ema, False)
        update_ema(ema, model.module, decay=0)
        ema.eval()
    else:
        ema = None
    
    model.train()
    if rank == 0 and ema is not None:
        ema.eval()
    
    if rank == 0:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer with CPU offloading to save GPU memory
    # Use SGD instead of Adam to save memory (no momentum/variance buffers)
    opt = torch.optim.SGD(model.parameters(), 
                         lr=train_config['optimizer']['lr'],
                         momentum=0.9,
                         weight_decay=train_config['optimizer'].get('weight_decay', 0.0))
    scaler = torch.amp.GradScaler('cuda')
    
    if rank == 0:
        logger.info("Using SGD optimizer to save GPU memory (instead of AdamW)")

    # Setup datasets
    dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    
    # 使用DistributedSampler确保数据在进程间正确分布
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    loader = DataLoader(
        dataset,
        batch_size=train_config['train']['global_batch_size'],
        sampler=sampler,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    if rank == 0:
        logger.info(f"Training dataset: {len(dataset):,} samples")
        logger.info(f"Samples per GPU: {len(dataset) // world_size}")

    # Setup validation
    valid_loader = None
    if 'valid_path' in train_config['data'] and train_config['data']['valid_path'] is not None:
        # 验证数据集（仅主进程）
        if rank == 0:
            val_dataset = MicroDopplerLatentDataset(
                data_dir=train_config['data']['valid_path'],
                latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
                latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_config['train']['global_batch_size'],
                shuffle=False,
                num_workers=train_config['data']['num_workers'],
                pin_memory=True,
                drop_last=False
            )
            logger.info(f"Validation dataset: {len(val_dataset):,} samples")

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
            
            if rank == 0:
                ema.load_state_dict(checkpoint['ema'])
            opt.load_state_dict(checkpoint['opt'])
            train_steps = int(os.path.basename(latest_checkpoint).split('.')[0])
            if rank == 0:
                logger.info(f"Resumed from checkpoint: {latest_checkpoint}")
        else:
            if rank == 0:
                logger.info("No checkpoint found, starting from scratch")

    # Clear memory
    torch.cuda.empty_cache()
    
    if rank == 0:
        logger.info("Starting DDP training loop...")
        logger.info(f"Will log every {train_config['train']['log_every']} steps")
        logger.info(f"Gradient accumulation: {train_config['train'].get('gradient_accumulation_steps', 1)} steps")
    
    # DDP training loop with gradient accumulation
    gradient_accumulation_steps = train_config['train'].get('gradient_accumulation_steps', 1)
    
    while train_steps < train_config['train']['max_steps']:
        sampler.set_epoch(train_steps // len(loader))  # 确保每个epoch的数据不同
        
        for batch_idx, (x, y) in enumerate(loader):
            if train_steps >= train_config['train']['max_steps']:
                break
                
            # Log first few batches to confirm training started
            if rank == 0 and batch_idx < 5:
                logger.info(f"Processing batch {batch_idx}, train_steps={train_steps}")
                
            # Move data to device
            x = x.to(device, dtype=torch.float16)
            y = y.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = transport.training_losses(model, x, dict(y=y))
                loss = loss_dict["loss"].mean() / gradient_accumulation_steps  # 归一化梯度
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['optimizer'].get('grad_clip_norm', 1.0))
                
                # Clear cache before optimizer step to free memory
                torch.cuda.empty_cache()
                
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)  # More aggressive memory cleanup
                
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
                
                # Update EMA after optimizer step (仅主进程)
                if rank == 0 and ema is not None:
                    update_ema(ema, model.module, decay=0.9999)
                
                train_steps += 1
            
            # Logging (accumulate loss for all micro-batches)
            running_loss += loss.item() * gradient_accumulation_steps  # 反归一化用于显示
            log_steps += 1
            
            # Only log after optimizer step (仅主进程)
            if rank == 0 and (batch_idx + 1) % gradient_accumulation_steps == 0 and train_steps % train_config['train']['log_every'] == 0:
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

            # Save checkpoint and validate (仅主进程)
            if rank == 0 and train_steps > 0 and train_steps % train_config['train']['ckpt_every'] == 0:
                if gpu_count > 1:
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                checkpoint_data = {
                    'model': model_state,
                    'ema': ema.state_dict() if ema is not None else None,
                    'opt': opt.state_dict(),
                    'config': train_config
                }
                torch.save(checkpoint_data, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
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

    if rank == 0:
        logger.info("Training completed!")
        writer.close()
        
        # Save final checkpoint (仅主进程)
        final_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_final.pt"
        final_checkpoint_data = {
            'model': model.module.state_dict(),
            'ema': ema.state_dict() if ema is not None else None,
            'opt': opt.state_dict(),
            'config': train_config
        }
        torch.save(final_checkpoint_data, final_checkpoint_path)
        logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
        logger.info(f"Training completed! Total steps: {train_steps}")
        
        writer.close()
    
    # 清理分布式环境
    cleanup_distributed()

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

def main():
    """主函数 - 启动多进程DDP训练"""
    parser = argparse.ArgumentParser(description="LightningDiT微调脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    
    args = parser.parse_args()
    
    # Load configuration
    train_config = OmegaConf.load(args.config)
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size >= 2:
        print("Using DDP on multiple GPUs")
        # 启动多进程训练
        mp.spawn(do_train_ddp, args=(world_size, train_config), nprocs=world_size, join=True)
    else:
        print("Using single GPU training")
        do_train_ddp(0, 1, train_config)

if __name__ == "__main__":
    main()
