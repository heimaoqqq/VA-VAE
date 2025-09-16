"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL

Modified for micro-Doppler dataset training
"""

import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
import sys  # 添加sys用于强制刷新输出
import os  # 确保os在顶部导入
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 修复datasets模块命名冲突：确保HuggingFace datasets能被正确导入
import importlib
try:
    # 提前导入HuggingFace datasets以避免与本地LightningDiT/datasets冲突
    datasets_module = importlib.import_module('datasets')
    sys.modules['datasets'] = datasets_module
except ImportError:
    pass

import math
import yaml
import json
import numpy as np
import logging
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('LightningDiT')
from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
# 修改点1：使用简化版数据集（完全匹配官方格式）
from microdoppler_latent_dataset_simple import MicroDopplerLatentDataset

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
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
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    else:
        # 非主进程也需要logger变量（空的）
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        writer = None
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Load VAE for decoding (needed for sampling)
    vae = None
    try:
        # 添加LightningDiT路径到系统路径
        lightningdit_path = os.path.join(os.getcwd(), 'LightningDiT')
        if lightningdit_path not in sys.path:
            sys.path.insert(0, lightningdit_path)
        
        from tokenizer.vavae import VA_VAE
        import yaml
        import tempfile
        
        # 动态指定我们微调的VAE模型路径
        custom_vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
        
        # 创建与step4_train_vavae.py完全一致的配置
        vae_config = {
            'ckpt_path': custom_vae_checkpoint,
            'model': {
                'base_learning_rate': 2.0e-05,  # Stage 3学习率
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'monitor': 'val/rec_loss',
                    'embed_dim': 32,
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
                            'disc_start': 1, 'disc_num_layers': 3, 'disc_weight': 0.5,
                            'disc_factor': 1.0, 'disc_in_channels': 3, 'disc_conditional': False,
                            'disc_loss': 'hinge', 'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,
                            'kl_weight': 1e-6, 'logvar_init': 0.0, 'use_actnorm': False,
                            'pp_style': False, 'vf_weight': 0.1, 'adaptive_vf': False,
                            'distmat_weight': 1.0, 'cos_weight': 1.0,
                            'distmat_margin': 0.25, 'cos_margin': 0.5
                        }
                    }
                }
            }
        }
        
        # 写入临时配置文件
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(vae_config, f, default_flow_style=False)
        os.close(temp_config_fd)
        
        try:
            # 使用官方VA_VAE类加载，配置完全匹配step4_train_vavae.py
            vae = VA_VAE(temp_config_path)
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
        
        if accelerator.is_main_process:
            print(f"[SETUP] Successfully loaded custom VAE model from: {custom_vae_checkpoint}")
            print(f"[SETUP] VAE config: {train_config['vae']['model_name']}")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[WARNING] Failed to load VAE: {e}")
            import traceback
            traceback.print_exc()
            print(f"[WARNING] VAE will be None, cannot decode images during training")
        vae = None
    
    # Create model:
    # VA-VAE实际输出16x16 latent (不是标准VAE的32x32)
    downsample_ratio = train_config['vae']['downsample_ratio']
    assert train_config['data']['image_size'] % downsample_ratio == 0, f"Image size must be divisible by {downsample_ratio}"
    latent_size = train_config['data']['image_size'] // downsample_ratio
    
    # 创建DiT模型
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        class_dropout_prob=train_config['model'].get('class_dropout_prob', 0.1),  # CFG dropout正则化
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model (支持ckpt和weight_init两种配置)
    pretrained_path = train_config['train'].get('ckpt') or train_config['train'].get('weight_init')
    if accelerator.is_main_process:
        print(f"[DEBUG] Looking for pretrained weights at: {pretrained_path}")
        print(f"[DEBUG] Model in_channels: {train_config['model'].get('in_chans', 4)}")
    
    if pretrained_path:
        # 检查文件是否存在
        if os.path.exists(pretrained_path):
            if accelerator.is_main_process:
                print(f"[PRETRAINED] Loading weights from: {pretrained_path}")
                print(f"[PRETRAINED] File size: {os.path.getsize(pretrained_path) / 1024 / 1024:.2f} MB")
            
            checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            if accelerator.is_main_process:
                print(f"[PRETRAINED] Checkpoint keys: {checkpoint.keys()}")
                if 'model' in checkpoint:
                    print(f"[PRETRAINED] Model has {len(checkpoint['model'])} parameters")
            
            # remove the prefix 'module.' from the keys
            if 'model' in checkpoint:
                checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
                
                # 打印详细调试信息
                if accelerator.is_main_process:
                    # 检查关键参数
                    x_embedder_weight = checkpoint['model'].get('x_embedder.proj.weight')
                    if x_embedder_weight is not None:
                        print(f"[CHECKPOINT] x_embedder.proj.weight shape: {x_embedder_weight.shape}")
                    else:
                        print(f"[CHECKPOINT] x_embedder.proj.weight NOT FOUND")
                        print(f"[CHECKPOINT] First 10 keys: {list(checkpoint['model'].keys())[:10]}")
                    
                    model_x_embedder = model.state_dict().get('x_embedder.proj.weight')
                    if model_x_embedder is not None:
                        print(f"[MODEL] x_embedder.proj.weight shape: {model_x_embedder.shape}")
                    
                    # 检查模型类型
                    print(f"[MODEL TYPE] {train_config['model']['model_type']}")
                    print(f"[MODEL] Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
                
                model = load_weights_with_shape_check(model, checkpoint, rank=rank)
                ema = load_weights_with_shape_check(ema, checkpoint, rank=rank) 
                if accelerator.is_main_process:
                    print(f"[PRETRAINED] Successfully loaded pretrained weights")
            elif 'ema' in checkpoint:
                # 可能权重保存在ema字段中
                if accelerator.is_main_process:
                    print(f"[INFO] Loading from 'ema' field instead of 'model'")
                checkpoint['model'] = checkpoint['ema']
                model = load_weights_with_shape_check(model, checkpoint, rank=rank)
                ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
            else:
                if accelerator.is_main_process:
                    print(f"[ERROR] Checkpoint doesn't contain 'model' or 'ema' key!")
                    print(f"[ERROR] Available keys: {checkpoint.keys()}")
        else:
            if accelerator.is_main_process:
                print(f"[WARNING] Pretrained checkpoint not found: {pretrained_path}")
                print(f"[WARNING] Training from scratch with random initialization!")
    else:
        if accelerator.is_main_process:
            print("[INFO] No pretrained path specified, training from scratch")
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
        partitial_train = train_config['transport']['partitial_train'] if 'partitial_train' in train_config['transport'] else None,
        partial_ratio = train_config['transport']['partial_ratio'] if 'partial_ratio' in train_config['transport'] else 1.0,
        shift_lg = train_config['transport']['shift_lg'] if 'shift_lg' in train_config['transport'] else False,
    )  # 完整的官方transport配置 
    if accelerator.is_main_process:
        print(f"[MAIN PROCESS] LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"[MAIN PROCESS] Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        print(f'[MAIN PROCESS] Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        print(f'[MAIN PROCESS] Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
        # 打印采样配置信息
        sample_config = train_config.get('sample', {})
        print(f'[MAIN PROCESS] Sampling method: {sample_config.get("sampling_method", "euler")}')
        print(f'[MAIN PROCESS] Sampling steps: {sample_config.get("num_sampling_steps", "dynamic")}')
        print(f'[MAIN PROCESS] CFG scale: {sample_config.get("cfg_scale", "dynamic")}')
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    # 使用权重衰减正则化（不同于官方的weight_decay=0）
    weight_decay = train_config['optimizer'].get('weight_decay', 0.01)
    opt = torch.optim.AdamW(model.parameters(), 
                           lr=train_config['optimizer']['lr'], 
                           weight_decay=weight_decay, 
                           betas=(0.9, train_config['optimizer']['beta2']))
    
    # 创建数据加载器 - 使用现有的latent数据集
    train_dataset = MicroDopplerLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'],
        latent_multiplier=train_config['data']['latent_multiplier']
    )
    
    # 统计每个用户的图像数量
    if accelerator.is_main_process:
        from pathlib import Path
        data_path = Path(train_config['data']['data_path'])
        print("\n" + "="*60)
        print("📊 数据集统计信息")
        print("="*60)
        
        # 统计每个用户的图像数量
        user_counts = {}
        total_images = 0
        for user_dir in sorted(data_path.glob("User_*")):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1])
                jpg_files = list(user_dir.glob("*.jpg"))
                png_files = list(user_dir.glob("*.png"))
                user_total = len(jpg_files) + len(png_files)
                user_counts[user_id] = {
                    'total': user_total,
                    'jpg': len(jpg_files),
                    'png': len(png_files)
                }
                total_images += user_total
        
        # 打印详细统计
        for user_id in sorted(user_counts.keys()):
            counts = user_counts[user_id]
            print(f"User_{user_id:02d}: {counts['total']:3d} 张 (JPG: {counts['jpg']:3d}, PNG: {counts['png']:3d})")
        
        print(f"\n📈 总计: {total_images} 张图像")
        print(f"📁 数据路径: {train_config['data']['data_path']}")
        print("="*60 + "\n")
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    
    # 标准随机采样（DiT/ImageNet标准做法）
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} latents from {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
        # 打印标签统计
        if hasattr(train_dataset, 'labels') and len(train_dataset.labels) > 0:
            import collections
            label_counts = collections.Counter(train_dataset.labels)
            print(f"\n📊 标签分布统计:")
            for label, count in sorted(label_counts.items()):
                print(f"  User_{label:02d}: {count:3d} 个样本")
            print(f"\n📈 标签范围: {min(train_dataset.labels)} - {max(train_dataset.labels)}")
            print(f"📈 总训练样本: {len(train_dataset)} 个")
    
    if 'valid_path' in train_config['data'] and train_config['data']['valid_path']:
        # 使用独立验证集
        valid_dataset = MicroDopplerLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'],
            latent_multiplier=train_config['data']['latent_multiplier']
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
    else:
        valid_loader = None
        valid_dataset = None
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🎯 数据集加载完成:")
        print(f"  训练集: {len(train_dataset):,} 个样本")
        if valid_dataset is not None:
            print(f"  验证集: {len(valid_dataset):,} 个样本")
            train_ratio = len(train_dataset) / (len(train_dataset) + len(valid_dataset)) * 100
            print(f"  训练/验证比例: {train_ratio:.1f}% / {100-train_ratio:.1f}%")
        else:
            print(f"  验证集: 未设置")
        print(f"={'='*60}\n")

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
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
            train_steps = 0  # 初始化train_steps为0
    else:
        # resume=False的情况
        train_steps = 0
    
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = train_config['train'].get('early_stopping_patience', 12)
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    # 打印训练开始信息
    if accelerator.is_main_process:
        print(f"[TRAINING START] Starting training loop...", flush=True)
        print(f"[TRAINING INFO] Total steps: {train_config['train']['max_steps']}", flush=True)
        print(f"[TRAINING INFO] Log every: {train_config['train']['log_every']} steps", flush=True)
        print(f"[TRAINING INFO] Checkpoint every: {train_config['train']['ckpt_every']} steps", flush=True)
    
    while True:
        for x, y in loader:
            
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
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
            update_ema(ema, model.module, decay=train_config['train']['ema_decay'])

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
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # 直接print确保能看到输出
                if accelerator.is_main_process:
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    if writer is not None:
                        writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint and generate samples (every ckpt_every steps)
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # 生成样本图像进行可视化
                    print(f"[SAMPLING] Generating samples at step {train_steps}...")
                    generate_samples(ema, vae, transport, device, train_steps, experiment_dir)
                dist.barrier()

                # Evaluate on validation set
                if accelerator.is_main_process:
                    print(f"[VALIDATION] Evaluating at step {train_steps}...")
                    logger.info(f"Start evaluating at step {train_steps}")
                val_loss = evaluate(model, valid_loader, device, transport, train_config['transport']['sample_eps'])
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss = val_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    print(f"[VALIDATION] Step {train_steps}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    if writer is not None:
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    
                    # Early stopping logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        print(f"[EARLY STOPPING] New best validation loss: {best_val_loss:.4f}")
                        logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        print(f"[EARLY STOPPING] No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                        logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                        
                        if patience_counter >= early_stopping_patience:
                            print(f"[EARLY STOPPING] Early stopping triggered after {train_steps} steps")
                            logger.info(f"Early stopping triggered after {train_steps} steps")
                            return accelerator
                model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    loaded_params = 0
    skipped_params = 0
    
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
                loaded_params += 1
            else:
                # 不要对x_embedder.proj.weight做特殊处理，32通道权重应该完全匹配
                if rank == 0:
                    print(f"[WARNING] Shape mismatch for '{name}': "
                        f"checkpoint {param.shape} vs model {model_state_dict[name].shape}")
                skipped_params += 1
        else:
            if rank == 0:
                print(f"[WARNING] Parameter '{name}' not found in model")
            skipped_params += 1
    
    if rank == 0:
        print(f"[LOAD SUMMARY] Loaded {loaded_params} params, skipped {skipped_params} params")
    
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def generate_samples(ema_model, vae, transport, device, step, output_dir, num_samples=8):
    """
    生成样本并保存latent可视化
    """
    from torchvision.utils import save_image
    import torch
    
    ema_model.eval()
    with torch.no_grad():
        # 生成随机噪声 - 修正通道数为32（VA-VAE的latent维度）
        z = torch.randn(num_samples, 32, 16, 16, device=device)  # 32通道，16x16大小
        # 条件生成：随机选择用户ID (0-30，对应31个用户)
        y = torch.randint(0, 31, (num_samples,), dtype=torch.long, device=device)
        print(f"[SAMPLING DEBUG] Generating for user IDs: {y.cpu().numpy()}")
        
        # 打印调试信息
        print(f"[SAMPLING DEBUG] Initial noise stats: mean={z.mean():.3f}, std={z.std():.3f}")
        
        # 使用配置文件中的采样设置（而非硬编码）
        sample_config = train_config.get('sample', {})
        num_steps = sample_config.get('num_sampling_steps', 50 if step < 5000 else 250)
        cfg_scale = sample_config.get('cfg_scale', 10.0 if step >= 10000 else 1.0)
        cfg_interval_start = sample_config.get('cfg_interval_start', 0.11)
        timestep_shift = sample_config.get('timestep_shift', 0.1)
        sampling_method = sample_config.get('sampling_method', 'euler')
        atol = sample_config.get('atol', 1e-6)
        rtol = sample_config.get('rtol', 1e-3)
        using_cfg = cfg_scale > 1.0
        
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(
            sampling_method=sampling_method,   # 从配置读取（dopri5或euler）
            num_steps=num_steps,               # 从配置读取（300步）
            atol=atol,                         # 从配置读取
            rtol=rtol,                         # 从配置读取  
            reverse=sample_config.get('reverse', False),
            timestep_shift=timestep_shift,     # 从配置读取
        )
        
        # 标准CFG实现
        if using_cfg:
            # CFG需要双倍batch处理条件和无条件
            z_cfg = torch.cat([z, z], 0)
            y_null = torch.tensor([31] * num_samples, device=device)  # null class
            y_cfg = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
            
            # 使用CFG前向传播
            samples = sample_fn(z_cfg, ema_model.forward_with_cfg, **model_kwargs)
            samples = samples[-1]  # 获取最终时间步的样本
            samples, _ = samples.chunk(2, dim=0)  # 去掉null class样本
            print(f"[SAMPLING DEBUG] Using CFG with scale={cfg_scale}, method={sampling_method}, steps={num_steps}")
        else:
            # 标准采样
            samples = sample_fn(z, ema_model, **dict(y=y))
            samples = samples[-1]  # 获取最终时间步的样本
            print(f"[SAMPLING DEBUG] Using standard sampling (no CFG), method={sampling_method}, steps={num_steps}")
        
        # 修复维度问题：确保samples是4D张量 [batch, channels, height, width]
        if samples.dim() == 5:
            # 如果是5维 [1, batch, channels, height, width]，去掉第一个维度
            samples = samples.squeeze(0)
        elif samples.dim() != 4:
            raise ValueError(f"Unexpected samples shape: {samples.shape}, expected 4D tensor")
            
        print(f"[SAMPLING DEBUG] Generated samples stats: mean={samples.mean():.3f}, std={samples.std():.3f}")
        print(f"[SAMPLING DEBUG] Samples shape after fixing: {samples.shape}")
        
        # 关键：反归一化！训练时做了channel-wise归一化，生成后需要反归一化
        # 官方公式：samples = (samples * latent_std) / latent_multiplier + latent_mean
        # 因为训练时：feature = (feature - mean) / std * latent_multiplier
        # 所以推理时：samples = samples / latent_multiplier * std + mean
        latent_stats_path = '/kaggle/working/VA-VAE/latents_safetensors/train/latent_stats.pt'
        if os.path.exists(latent_stats_path):
            stats = torch.load(latent_stats_path)
            mean = stats['mean'].to(device)  # [32, 1, 1]
            std = stats['std'].to(device)     # [32, 1, 1]
            latent_multiplier = 1.0  # VA-VAE使用1.0，不是0.18215
            # 反归一化：按照官方公式，不需要unsqueeze(0)因为样本已经是4D
            samples_denorm = (samples * std) / latent_multiplier + mean
            print(f"[SAMPLING DEBUG] After denorm: mean={samples_denorm.mean():.3f}, std={samples_denorm.std():.3f}")
            print(f"[SAMPLING DEBUG] samples_denorm shape: {samples_denorm.shape}")
        else:
            samples_denorm = samples
        
        # 使用VAE解码latent为真实图像
        if vae is not None:
            try:
                # 创建保存目录
                os.makedirs(output_dir, exist_ok=True)
                print(f"[SAMPLING DEBUG] Output directory: {output_dir}")
                
                # 确保VAE在正确设备上
                if hasattr(vae, 'to'):
                    vae = vae.to(device)
                
                # 使用VA-VAE解码latent为图像
                print(f"[SAMPLING DEBUG] Decoding {samples_denorm.shape} latents to images...")
                decoded_images = vae.decode_to_images(samples_denorm)
                print(f"[SAMPLING DEBUG] Decoded {len(decoded_images)} images, each of shape {decoded_images[0].shape if len(decoded_images) > 0 else 'N/A'}")
                
                # 按照官方方式保存单个图像文件
                from PIL import Image
                saved_files = []
                for i, image in enumerate(decoded_images):
                    save_path = f"{output_dir}/sample_{step:07d}_{i:02d}.png"
                    Image.fromarray(image).save(save_path)
                    saved_files.append(save_path)
                    print(f"[SAMPLING DEBUG] Saved: {save_path}")
                
                print(f"[SAMPLING] Successfully saved {len(saved_files)} decoded images to {output_dir}")
                
            except Exception as e:
                print(f"[ERROR] VAE decoding failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Cannot save images without VAE decoding")
        else:
            print(f"[ERROR] VAE is None, cannot decode latents to images")
        
        # 重要观察：如果Generated samples和Initial noise几乎相同，说明模型还没学到东西
        if abs(samples.mean().item()) < 0.1 and abs(samples.std().item() - 1.0) < 0.1:
            print(f"[WARNING] Model output is almost identical to input noise - model needs more training!")
        
        print(f"[SAMPLING] Saved visualizations to {output_dir}")
    
    ema_model.train()
    return samples

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

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
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
def evaluate(model, loader, device, transport, sample_eps_range):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    num_samples = 0
    
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y
        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        loss = loss_dict["loss"].mean()
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
    
    avg_loss = total_loss / num_samples
    return torch.tensor(avg_loss, device=device)

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dit_s_microdoppler.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)
