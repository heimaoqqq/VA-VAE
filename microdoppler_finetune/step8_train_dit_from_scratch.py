#!/usr/bin/env python3
"""
步骤8: 从头训练LightningDiT-Base模型用于微多普勒条件生成
基于官方train.py复制并修改，确保最大程度与官方一致
Training LightningDiT-Base with VA-VAE on micro-Doppler dataset
"""

import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
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
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from pathlib import Path

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
# from datasets.img_latent_dataset import ImgLatentDataset  # 官方数据集

# 导入我们自己的微多普勒数据集类
sys.path.append('/kaggle/working/microdoppler_finetune')
from step6_encode_dataset import MicroDopplerLatentDataset

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
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data - 使用我们的微多普勒latent数据集
    dataset = MicroDopplerLatentDataset(
        data_path=train_config['data']['data_path'],  # 使用data_path而不是data_dir
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
            data_path=train_config['data']['valid_path'],  # 使用data_path而不是data_dir
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
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
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

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    # 早停参数
    patience = train_config['train'].get('patience', 20)
    max_epochs = train_config['train'].get('max_epochs', 200)
    steps_per_epoch = len(loader)
    
    # 训练循环 - 改为基于epoch
    for epoch in range(start_epoch, max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Starting Epoch {epoch+1}/{max_epochs}")
            
        epoch_loss = 0
        epoch_steps = 0
        
        for x, y in loader:
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)  # 确保标签是long类型
            else:
                x = x.to(device)
                y = y.to(device, dtype=torch.long)
            
            model_kwargs = dict(y=y)
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
            update_ema(ema, model.module)

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
            
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f"(epoch={epoch+1:03d}, step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Epoch结束，计算平均损失
        avg_epoch_loss = epoch_loss / epoch_steps
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
        # 每个epoch结束进行验证
        if 'valid_path' in train_config['data']:
            if accelerator.is_main_process:
                logger.info(f"Evaluating at epoch {epoch+1}")
            
            model.eval()
            val_loss = evaluate(model, valid_loader, device, transport, accelerator)
            model.train()
            
            if accelerator.is_main_process:
                logger.info(f"Validation Loss: {val_loss:.4f}")
                writer.add_scalar('Loss/validation', val_loss, epoch+1)
                
                # 检查是否是最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                    }
                    best_checkpoint_path = f"{checkpoint_dir}/best_model.pt"
                    torch.save(checkpoint, best_checkpoint_path)
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                    
                # 早停检查
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # 定期保存checkpoint
        if (epoch + 1) % train_config['train'].get('ckpt_every_epoch', 10) == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.module.state_dict(),
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
            dist.barrier()

    if accelerator.is_main_process:
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return accelerator


class MicroDopplerLatentDataset(torch.utils.data.Dataset):
    """微多普勒潜在编码数据集"""
    def __init__(self, data_dir, split='train', latent_norm=True, latent_multiplier=1.0):
        self.data_dir = Path(data_dir)
        self.split = split
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 加载split信息
        split_file = self.data_dir / 'data_split.json'
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits = json.load(f)
                self.file_list = splits[split]
        else:
            # 如果没有split文件，使用所有文件
            import safetensors.torch
            self.file_list = list(self.data_dir.glob('latents_*.safetensors'))
        
        # 加载统计信息
        stats_file = self.data_dir / 'latents_stats.pt'
        if stats_file.exists() and latent_norm:
            stats = torch.load(stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            self.mean = 0.0
            self.std = 1.0
        
        # 构建数据索引
        self.data = []
        self.labels = []
        
        for file_path in self.file_list:
            # 从文件名提取用户ID
            # 假设文件名格式: userX_frameY.safetensors
            filename = file_path.stem
            if 'user' in filename:
                user_id = int(filename.split('user')[1].split('_')[0])
            else:
                user_id = 0  # 默认标签
            
            self.data.append(file_path)
            self.labels.append(user_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 加载潜在编码
        import safetensors.torch
        file_path = self.data[idx]
        latent_dict = safetensors.torch.load_file(str(file_path))
        latent = latent_dict['latent']
        
        # 归一化
        if self.latent_norm:
            latent = (latent - self.mean) * self.latent_multiplier / self.std
        
        return latent, self.labels[idx]


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
    
    # All-reduce across processes
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    
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

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)
