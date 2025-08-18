#!/usr/bin/env python3
"""
LightningDiT微调脚本 - 使用DistributedDataParallel (DDP)
针对Kaggle T4x2环境优化，解决DataParallel负载不均问题
"""
import os
import sys
import logging
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from time import time
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from safetensors import safe_open
from PIL import Image

# 禁用torch.compile和FX tracing
os.environ['TORCH_COMPILE_DISABLE'] = '1'  
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append('/kaggle/working/VA-VAE')
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')

try:
    torch._dynamo.disable()
    torch._inductor.config.debug = False
    torch._inductor.config.triton.unique_kernel_names = False
    torch._inductor.config.coordinate_descent_tuning = False
except:
    pass

from models.lightningdit import LightningDiT_models
from transport import create_transport

def requires_grad(model, flag=True):
    """设置模型参数的requires_grad"""
    for p in model.parameters():
        p.requires_grad = flag

def update_ema(target, source, decay):
    """更新EMA模型"""
    target_params = dict(target.named_parameters())
    source_params = dict(source.named_parameters())
    
    for key in target_params:
        if key in source_params:
            target_params[key].data.mul_(decay).add_(source_params[key].data, alpha=1 - decay)

class MicroDopplerLatentDataset(torch.utils.data.Dataset):
    """微多普勒潜在数据集加载器"""
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        
        # 收集所有批量潜在文件
        self.latent_files = []
        self.samples_per_file = []
        self.cumulative_samples = [0]
        
        print(f"Loading batched latent files from: {data_dir}")
        
        for file_name in sorted(os.listdir(data_dir)):
            if file_name.endswith('.safetensors'):
                file_path = os.path.join(data_dir, file_name)
                
                # 检查文件中的样本数量
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    latents = f.get_tensor("latent")
                    labels = f.get_tensor("label")  
                    num_samples = latents.shape[0]
                    
                    print(f"File {file_name}: {num_samples} samples")
                    
                    self.latent_files.append(file_path)
                    self.samples_per_file.append(num_samples)
                    self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)
        
        self.total_samples = self.cumulative_samples[-1]
        print(f"Dataset initialized with {self.total_samples} total samples from {len(self.latent_files)} files")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 找到对应的文件和文件内索引
        file_idx = 0
        while file_idx < len(self.cumulative_samples) - 1 and idx >= self.cumulative_samples[file_idx + 1]:
            file_idx += 1
        
        local_idx = idx - self.cumulative_samples[file_idx]
        file_path = self.latent_files[file_idx]
        
        # 加载指定样本
        with safe_open(file_path, framework="pt", device="cpu") as f:
            latent = f.get_tensor("latent")[local_idx]  # [32, 16, 16]
            label = f.get_tensor("label")[local_idx]    # scalar
        
        # 应用归一化和缩放
        if self.latent_norm:
            latent = latent * self.latent_multiplier
            
        return latent, label.long()

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

def do_train_ddp(rank, world_size, train_config):
    """DDP训练主函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)
    device = f'cuda:{rank}'
    
    # 创建输出目录
    os.makedirs(train_config['train']['output_dir'], exist_ok=True)
    os.makedirs(f"{train_config['train']['output_dir']}/checkpoints", exist_ok=True)
    
    # 创建日志记录器
    logger = create_logger(train_config['train']['output_dir'], rank)
    
    if rank == 0:
        logger.info(f"Setting up DDP training with {world_size} GPU(s)")
        writer = SummaryWriter(log_dir=f"{train_config['train']['output_dir']}/tensorboard")
    
    # 创建transport
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps']
    )
    
    # 创建模型
    model = LightningDiT_models[train_config['model']['model_type']](
        in_channels=train_config['model']['in_chans'],
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'],
        use_rope=train_config['model']['use_rope'],
        use_rmsnorm=train_config['model']['use_rmsnorm'],
        wo_shift=train_config['model']['wo_shift'],
        use_checkpoint=train_config['model']['use_checkpoint'],
    )
    
    # 加载预训练权重
    if 'weight_init' in train_config['train'] and train_config['train']['weight_init']:
        if rank == 0:
            logger.info(f"Loading pretrained weights from {train_config['train']['weight_init']}")
        
        checkpoint = torch.load(train_config['train']['weight_init'], map_location='cpu')
        
        # 处理类别数量不匹配
        if 'y_embedder.embedding_table.weight' in checkpoint:
            checkpoint_classes = checkpoint['y_embedder.embedding_table.weight'].shape[0]
            model_classes = model.y_embedder.embedding_table.weight.shape[0]
            
            if checkpoint_classes != model_classes:
                if rank == 0:
                    logger.info(f"Class mismatch: checkpoint {checkpoint_classes} vs model {model_classes}, skipping y_embedder")
                del checkpoint['y_embedder.embedding_table.weight']
        
        model.load_state_dict(checkpoint, strict=False)
        if rank == 0:
            logger.info("Pretrained weights loaded successfully")

    # 移动模型到GPU并包装为DDP
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # 创建EMA模型（仅在主进程）
    if rank == 0:
        ema = deepcopy(model.module).to(device)
        requires_grad(ema, False)
        update_ema(ema, model.module, decay=0)
        ema.eval()
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()

    # 设置优化器和scaler
    opt = torch.optim.AdamW(model.parameters(), 
                           lr=train_config['optimizer']['lr'], 
                           weight_decay=train_config['optimizer'].get('weight_decay', 0.0))
    scaler = torch.cuda.amp.GradScaler()

    # 设置数据集和分布式采样器
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
        logger.info(f"Samples per GPU: {len(loader.dataset) // world_size}")

    # 验证数据集（仅主进程）
    if rank == 0:
        val_dataset = MicroDopplerLatentDataset(
            data_dir=train_config['data']['val_data_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['train']['global_batch_size'],
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True
        )

    # 恢复检查点
    train_steps = 0
    start_time = time()
    running_loss = 0
    log_steps = 0
    
    if train_config['train'].get('resume', False):
        checkpoint_dir = f"{train_config['train']['output_dir']}/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                model.module.load_state_dict(checkpoint['model'])
                if rank == 0:
                    ema.load_state_dict(checkpoint['ema'])
                opt.load_state_dict(checkpoint['opt'])
                train_steps = int(os.path.basename(latest_checkpoint_path).split('.')[0])
                if rank == 0:
                    logger.info(f"Resumed from checkpoint: {latest_checkpoint_path}")
            else:
                if rank == 0:
                    logger.info("No checkpoint found, starting from scratch")

    if rank == 0:
        logger.info("Starting DDP training loop...")
    
    # DDP训练循环
    gradient_accumulation_steps = train_config['train'].get('gradient_accumulation_steps', 1)
    
    while train_steps < train_config['train']['max_steps']:
        sampler.set_epoch(train_steps // len(loader))  # 确保每个epoch的数据不同
        
        for batch_idx, (x, y) in enumerate(loader):
            if train_steps >= train_config['train']['max_steps']:
                break
                
            # 移动数据到设备
            x = x.to(device, dtype=torch.float16)
            y = y.to(device)
            
            # 前向传播（混合精度）
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = transport.training_losses(model, x, dict(y=y))
                loss = loss_dict["loss"].mean() / gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积完成后更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['optimizer'].get('grad_clip_norm', 1.0))
                
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
                # 更新EMA（仅主进程）
                if rank == 0:
                    update_ema(ema, model.module, decay=0.9999)
                
                train_steps += 1
            
            # 累积损失用于日志
            running_loss += loss.item() * gradient_accumulation_steps
            log_steps += 1
            
            # 日志记录（仅主进程）
            if rank == 0 and (batch_idx + 1) % gradient_accumulation_steps == 0 and train_steps % train_config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                writer.add_scalar('Loss/train', avg_loss, train_steps)
                
                # 重置监控变量
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # 保存检查点（仅主进程）
            if rank == 0 and train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                checkpoint_path = f"{train_config['train']['output_dir']}/checkpoints/{train_steps:07d}.pt"
                torch.save({
                    'model': model.module.state_dict(),
                    'ema': ema.state_dict(),
                    'opt': opt.state_dict(),
                    'config': train_config
                }, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                # 验证评估
                model.eval()
                ema.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_steps = 0
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(device, dtype=torch.float16)
                        val_y = val_y.to(device)
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            val_loss_dict = transport.training_losses(ema, val_x, dict(y=val_y))
                            val_loss += val_loss_dict["loss"].mean().item()
                            val_steps += 1
                            
                            if val_steps >= 10:  # 限制验证步数
                                break
                    
                    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar('Loss/val', avg_val_loss, train_steps)
                
                model.train()
    
    # 保存最终检查点
    if rank == 0:
        final_checkpoint_path = f"{train_config['train']['output_dir']}/checkpoints/{train_steps:07d}_final.pt"
        torch.save({
            'model': model.module.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            'config': train_config
        }, final_checkpoint_path)
        logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
        logger.info(f"Training completed! Total steps: {train_steps}")
        writer.close()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    """主函数 - 启动多进程DDP训练"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    train_config = OmegaConf.load(args.config)
    
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size < 2:
        print("Warning: DDP requires at least 2 GPUs, falling back to single GPU")
        world_size = 1
    
    # 启动多进程训练
    if world_size > 1:
        mp.spawn(do_train_ddp, args=(world_size, train_config), nprocs=world_size, join=True)
    else:
        do_train_ddp(0, 1, train_config)

if __name__ == "__main__":
    main()
