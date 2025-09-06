"""
Training DiT-S from scratch for micro-Doppler dataset
Based on official LightningDiT train.py with minimal modifications
保持与官方流程完全一致，仅修改必要部分
"""

import torch
import torch.backends.cuda
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import sys

# 添加LightningDiT路径
sys.path.append('./LightningDiT')

# 直接文件导入LightningDiT模块
import importlib.util
import os

# 检查LightningDiT路径
lightningdit_path = '/kaggle/working/VA-VAE/LightningDiT'
if not os.path.exists(lightningdit_path):
    lightningdit_path = './LightningDiT'

# 创建必需的__init__.py文件
init_files = [
    os.path.join(lightningdit_path, '__init__.py'),
    os.path.join(lightningdit_path, 'datasets', '__init__.py'),
    os.path.join(lightningdit_path, 'models', '__init__.py'),
    os.path.join(lightningdit_path, 'transport', '__init__.py')
]

for init_file in init_files:
    if not os.path.exists(init_file):
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        with open(init_file, 'w') as f:
            f.write("# Auto-generated __init__.py\n")

try:
    # 导入数据集模块
    dataset_path = os.path.join(lightningdit_path, 'datasets', 'img_latent_dataset.py')
    spec_dataset = importlib.util.spec_from_file_location("img_latent_dataset", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec_dataset)
    spec_dataset.loader.exec_module(dataset_module)
    ImgLatentDataset = dataset_module.ImgLatentDataset
    
    # 尝试标准导入其他模块
    from models.lightningdit import LightningDiT_models
    from transport import create_transport, Sampler
    from accelerate import Accelerator
    print("✅ 所有模块导入成功")
    
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    raise

# 我们自己的模块  
from simplified_vavae import SimplifiedVAVAE

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT-S model from scratch.
    保持与官方train.py完全一致的流程
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, accelerator)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str = json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model (DiT-S):
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0
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

    # Create optimizer before preparing with Accelerate
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config['optimizer']['lr'], 
        weight_decay=0, 
        betas=(0.9, train_config['optimizer']['beta2'])
    )
    
    # Create transport (与官方完全一致)
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm=train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )
    
    # Setup data (与官方完全一致的数据加载方式)
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
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
    
    # Prepare model, optimizer, and data loader with Accelerate (替代DDP)
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Create EMA after preparing with Accelerate
    ema = deepcopy(accelerator.unwrap_model(model)).to(accelerator.device)
    requires_grad(ema, False)
    
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in accelerator.unwrap_model(model).parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    # Validation loader (如果有)
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 1.0,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    # Prepare models for training:
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Initialize EMA
    model.train()
    ema.eval()
    
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Starting training from scratch")
        logger.info(f"Entering training loop...")

    # 训练循环 (与官方完全一致)
    while True:
        if accelerator.is_main_process:
            logger.info(f"Starting epoch, loading first batch...")
        for x, y in loader:
            if accelerator.is_main_process and train_steps == 0:
                logger.info(f"First batch loaded, shape: {x.shape}, starting forward pass...")
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            
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
            update_ema(ema, accelerator.unwrap_model(model))

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.gather(avg_loss).mean()
                avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

                # Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                    val_loss = evaluate(model, valid_loader, device, transport, (0.0, 1.0))
                    val_loss = accelerator.gather(val_loss).mean()
                    val_loss = val_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model.train()
            
            if train_steps >= train_config['train']['max_steps']:
                break
        
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def evaluate(model, loader, device, transport, time_range):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs, time_range=time_range)
            total_loss += loss_dict["loss"].mean().item()
    model.train()
    return torch.tensor(total_loss / len(loader))

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir, accelerator=None):
    """
    Create a logger that writes to a log file and stdout.
    使用Accelerate而非原生PyTorch分布式
    """
    # 使用Accelerate的is_main_process替代dist.get_rank() == 0
    if accelerator is None or accelerator.is_main_process:  # real logger
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dit_s_microdoppler.yaml')
    args = parser.parse_args()

    # 明确启用多GPU训练
    print(f"🔍 检测到 {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    accelerator = Accelerator(
        mixed_precision='no',  # T4可以使用fp16，但先确保稳定性
        gradient_accumulation_steps=1,
        log_with="tensorboard" if torch.cuda.device_count() == 1 else None
    )
    
    print(f"🚀 Accelerator使用 {accelerator.num_processes} 个进程")
    print(f"📱 当前进程在设备: {accelerator.device}")
    
    train_config = load_config(args.config)
    do_train(train_config, accelerator)
