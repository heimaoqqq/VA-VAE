#!/usr/bin/env python3
"""
DiT XL完整微调 - Kaggle T4x2 DDP版本
基于官方预训练权重的分布式数据并行微调
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
import sys
import os
from pathlib import Path
import time
import yaml
from tqdm import tqdm
import numpy as np
import safetensors.torch as safetensors
import argparse

# 🔧 关键显存优化设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 添加路径
sys.path.append('/kaggle/working/VA-VAE')
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')

from models.lightningdit import LightningDiT_models

class LatentDataset(torch.utils.data.Dataset):
    """Latent向量数据集"""
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

def load_dit_xl_for_full_finetune(checkpoint_path, rank=None):
    """DDP优化版DiT XL模型加载 - CPU先加载再转移GPU"""
    if rank is None or rank == 0:
        print(f"📂 加载DiT XL模型: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        if rank is None or rank == 0:
            print(f"❌ 模型文件不存在: {checkpoint_path}")
        return None
    
    # 🔧 DDP关键优化：完全在CPU上初始化，避免多进程GPU竞争
    if rank is None or rank == 0:
        print(f"📊 开始CPU加载，避免DDP显存竞争...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if rank is None or rank == 0:
        print(f"✅ Checkpoint已在CPU加载 ({os.path.getsize(checkpoint_path)/1e9:.1f}GB)")
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    
    # 处理DataParallel权重键
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace('module.', '') if key.startswith('module.') else key
        clean_state_dict[clean_key] = value
    
    # 推断参数
    pos_embed_shape = None
    y_embed_shape = None
    final_layer_shape = None
    has_swiglu = False
    
    for key, tensor in clean_state_dict.items():
        if key == 'pos_embed':
            pos_embed_shape = tensor.shape
        elif key == 'y_embedder.embedding_table.weight':
            y_embed_shape = tensor.shape
        elif key == 'final_layer.linear.weight':
            final_layer_shape = tensor.shape
        elif 'mlp.w12' in key:
            has_swiglu = True
    
    input_size = int(pos_embed_shape[1]**0.5) if pos_embed_shape else 16
    num_classes = y_embed_shape[0] if y_embed_shape else 1001
    out_channels = final_layer_shape[0] if final_layer_shape else 32
    
    print(f"📋 模型配置:")
    print(f"   输入尺寸: {input_size}x{input_size}")
    print(f"   类别数量: {num_classes}")
    print(f"   输出通道: {out_channels}")
    print(f"   MLP类型: {'SwiGLU' if has_swiglu else 'GELU'}")
    
    # 🔧 关键：在CPU上完成所有模型初始化 + 启用梯度检查点
    model = LightningDiT_models['LightningDiT-XL/1'](
        input_size=input_size,
        num_classes=num_classes,
        class_dropout_prob=0.0,
        use_qknorm=False,
        use_swiglu=has_swiglu,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        in_channels=out_channels,
        use_checkpoint=True,  # 🔧 启用梯度检查点节省显存！
    )
    
    # 确保模型在CPU上
    model = model.cpu()
    
    # 在CPU上加载权重
    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    
    if (rank is None or rank == 0) and missing_keys:
        print(f"⚠️ 缺失权重键: {len(missing_keys)}")
    if (rank is None or rank == 0) and unexpected_keys:
        print(f"⚠️ 多余权重键: {len(unexpected_keys)}")
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank is None or rank == 0:
        print(f"✅ DiT XL模型在CPU初始化完成")
        print(f"   总参数: {param_count/1e6:.1f}M")
        print(f"   可训练参数: {trainable_count/1e6:.1f}M ({trainable_count/param_count*100:.1f}%)")
        print(f"📦 模型保持在CPU，等待安全转移到GPU...")
    
    # 返回CPU上的模型，稍后再转移到指定GPU
    return model

def setup_ddp(rank, world_size):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前GPU
    torch.cuda.set_device(rank)
    
    # 🔧 显存优化配置
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.reset_peak_memory_stats(rank)

def cleanup_ddp():
    """清理DDP进程组"""
    dist.destroy_process_group()

def evaluate_model(model, vae_model, device, rank):
    """验证模型性能"""
    model.eval()
    val_loss = 0.0
    num_samples = 20  # 验证样本数
    
    with torch.no_grad():
        # 使用正确的验证集数据路径
        from pathlib import Path
        official_val_dir = Path("/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_val_256")
        latent_path = official_val_dir / "latents_rank00_shard000.safetensors"
        
        if not latent_path.exists():
            if rank == 0:
                print(f"⚠️ 验证数据不存在，跳过验证: {latent_path}")
            return 0.0
            
        data = safetensors.load_file(str(latent_path))
        latents = data['latents'][:num_samples]
        labels = data['labels'][:num_samples]
        
        # 🔧 应用与训练数据相同的官方LightningDiT归一化
        # 检查是否有latents_stats.pt文件
        stats_file = official_val_dir / "latents_stats.pt"
        train_stats_file = Path("/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256/latents_stats.pt")
        
        if stats_file.exists():
            stats = torch.load(stats_file, map_location='cpu')
            latent_mean = stats['mean']
            latent_std = stats['std']
        elif train_stats_file.exists():
            # 使用训练集的统计信息
            stats = torch.load(train_stats_file, map_location='cpu')
            latent_mean = stats['mean']
            latent_std = stats['std']
        else:
            # 计算当前验证数据的统计信息
            latent_mean = latents.mean(dim=0, keepdim=True)
            latent_std = latents.std(dim=0, keepdim=True)
        
        # 应用官方归一化: (latent - mean) / std * multiplier
        latents = (latents - latent_mean) / latent_std * 1.0
        
        # 移动到设备
        latents = latents.to(device)
        labels = labels.to(device)
        
        for i in range(num_samples):
            # 模拟扩散过程
            noise = torch.randn_like(latents[i:i+1])
            timesteps = torch.randint(0, 1000, (1,), device=device).float()
            
            alpha_t = 1 - timesteps / 1000
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            noisy_latents = torch.sqrt(alpha_t) * latents[i:i+1] + torch.sqrt(1 - alpha_t) * noise
            
            # 前向传播
            pred_noise = model(noisy_latents, timesteps, labels[i:i+1])
            loss = nn.MSELoss()(pred_noise, noise)
            val_loss += loss.item()
    
    model.train()
    return val_loss / num_samples

def generate_validation_samples(model, vae_model, device, epoch):
    """快速生成条件扩散样本进行可视化验证"""
    model.eval()
    
    with torch.no_grad():
        print(f"   🔍 开始生成验证样本，设备: {device}")
        
        # 生成4个不同类别的样本
        num_samples = 4
        sample_labels = torch.arange(num_samples, device=device)
        print(f"   🏷️ 样本标签: {sample_labels.tolist()}")
        
        # 从纯噪声开始 - 确保在正确设备上
        latent_shape = (num_samples, 32, 16, 16)
        z = torch.randn(latent_shape, device=device, dtype=torch.float16)
        print(f"   📐 噪声形状: {z.shape}, 设备: {z.device}, 类型: {z.dtype}")
        
        # 官方LightningDiT ODE采样设置
        cfg_scale = 10.0  # 官方标准CFG强度
        timestep_shift = 0.3  # 官方标准时间步偏移
        
        # ODE求解 - 官方dopri5风格
        t_start, t_end = 1.0, 1e-4
        num_steps = 50  # 平衡质量和速度
        
        def ode_fn(t, x):
            """ODE函数：dx/dt = f(x,t) - 官方LightningDiT实现"""
            print(f"     ODE步骤: t={t:.4f}")
            
            # 确保所有张量在GPU上
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.float32)
            
            if cfg_scale > 1.0:
                # CFG: 条件 + 无条件预测 (官方实现)
                uncond_labels = torch.full_like(sample_labels, -1)  # -1表示无条件
                x_cond = torch.cat([x, x], dim=0)
                t_cond = torch.cat([t_batch, t_batch], dim=0) 
                labels_cond = torch.cat([sample_labels, uncond_labels], dim=0)
                
                with torch.cuda.amp.autocast():
                    pred_both = model(x_cond, t_cond, labels_cond)
                pred_cond, pred_uncond = pred_both.chunk(2, dim=0)
                
                # CFG组合
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                with torch.cuda.amp.autocast():
                    pred = model(x, t_batch, sample_labels)
            
            return -pred  # 负号：从噪声到干净图像
        
        print(f"   ⏱️ 开始官方ODE采样 ({num_steps}步)")
        
        # 手动Euler积分（dopri5简化版，官方风格）
        dt = (t_end - t_start) / num_steps
        t = t_start
        x = z.clone()
        
        for step in range(num_steps):
            print(f"     步骤 {step+1}/{num_steps}, t={t:.4f}")
            
            # 时间步偏移调整 (官方实现)
            t_shifted = t + timestep_shift
            t_shifted = max(t_shifted, t_end)
            
            # Euler步骤
            dx_dt = ode_fn(t_shifted, x)
            x = x + dt * dx_dt
            t = t + dt
            
            if t <= t_end:
                break
        
        latents = x.float()  # 转换为float32用于VAE解码
        print(f"   🎯 最终latents形状: {latents.shape}, 设备: {latents.device}")
        
        # VA-VAE解码
        print(f"   🔄 开始VA-VAE解码...")
        generated_images = vae_model.decode(latents)
        print(f"   📷 解码后图像: {generated_images.shape}, 范围: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
        
        # 分析VA-VAE输出范围并正确归一化
        # 大多数VAE输出范围是 [-1, 1]，但需要确认
        if generated_images.min() >= -1.1 and generated_images.max() <= 1.1:
            # 范围是 [-1, 1]，转换到 [0, 1]
            generated_images = (generated_images + 1.0) / 2.0
            print(f"   ✅ 检测到[-1,1]范围，转换到[0,1]")
        elif generated_images.min() >= -0.1 and generated_images.max() <= 1.1:
            # 范围已经是 [0, 1] 左右
            print(f"   ✅ 检测到[0,1]范围，直接使用")
        else:
            # 其他范围，使用min-max归一化
            min_val = generated_images.min()
            max_val = generated_images.max()
            generated_images = (generated_images - min_val) / (max_val - min_val)
            print(f"   ⚠️ 异常范围[{min_val:.3f},{max_val:.3f}]，使用min-max归一化")
        
        # 确保最终范围正确
        generated_images = torch.clamp(generated_images, 0.0, 1.0)
        print(f"   📊 最终图像范围: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
        
        # 保存样本
        save_path = f"/kaggle/working/validation_samples_epoch_{epoch}.png"
        import torchvision.utils as vutils
        vutils.save_image(generated_images, save_path, nrow=2, normalize=False)
        print(f"   ✅ 保存验证样本: {save_path}")
    
    model.train()

def train_ddp_worker(rank, world_size, config):
    """DDP工作进程 - 每个GPU运行此函数"""
    print(f"🚀 启动DDP Worker - Rank {rank}/{world_size}")
    
    # 初始化DDP
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # 训练参数
    num_epochs = config.get('num_epochs', 20)
    learning_rate = config.get('learning_rate', 1e-5) 
    save_interval = config.get('save_interval', 5)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    batch_size = config.get('batch_size', 1)
    
    if rank == 0:
        print(f"📋 DDP训练配置:")
        print(f"   设备: {device} (Rank {rank})")
        print(f"   训练轮数: {num_epochs}")
        print(f"   学习率: {learning_rate}")
        print(f"   批次大小: {batch_size}")
        print(f"   梯度累积步数: {gradient_accumulation_steps}")
    
    # 🔧 DDP优化加载流程：CPU → GPU → DDP
    dit_xl_path = "/kaggle/working/VA-VAE/models/lightningdit-xl-imagenet256-64ep.pt"
    
    if rank == 0:
        print(f"🔄 第1步：在CPU加载DiT XL模型...")
    
    # Step 1: 在CPU加载模型
    model = load_dit_xl_for_full_finetune(dit_xl_path, rank)
    if model is None:
        cleanup_ddp()
        return
    
    if rank == 0:
        print(f"🔄 第2步：转移模型到GPU{rank}...")
        
    # Step 2: 分阶段模型转移到GPU (减少进程资源竞争)
    torch.cuda.empty_cache()  # 清理显存
    pre_gpu_memory = torch.cuda.memory_allocated(device) / (1024**3)
    
    # 🔄 错峰转移策略：减少系统资源竞争
    if world_size > 1:
        # DDP环境：按rank错峰转移
        time.sleep(rank * 3)  # rank 0先转，rank 1等3秒后转
        if rank == 0:
            print(f"🔄 Rank {rank}: 开始转移模型到GPU...")
        model = model.to(device)
        if rank == 0:
            print(f"✅ Rank {rank}: 模型转移完成")
        # 等待所有rank完成转移
        dist.barrier()
    else:
        # 单GPU环境
        model = model.to(device)
    
    post_gpu_memory = torch.cuda.memory_allocated(device) / (1024**3)
    
    if rank == 0:
        print(f"✅ 模型转移完成 - GPU{rank}显存: {pre_gpu_memory:.1f}GB → {post_gpu_memory:.1f}GB")
        print(f"🔄 第3步：加载VA-VAE模型...")
    
    # Step 3: 加载VA-VAE (每个rank独立加载)
    vae_model = load_vae_model(device)
    
    if rank == 0:
        print(f"🔄 第4步：DDP包装模型...")
    
    # Step 4: 包装为DDP - 修复梯度检查点冲突
    model = DDP(
        model, 
        device_ids=[device], 
        find_unused_parameters=True,  # 解决未使用参数错误
        gradient_as_bucket_view=True  # 优化梯度同步
    )
    # 🔧 关键修复：解决梯度检查点+DDP冲突
    model._set_static_graph()  # 告诉DDP图结构不变
    
    if rank == 0:
        print(f"✅ DDP包装完成 - 模型准备就绪")
    
    # 创建分布式数据加载器
    dataloader = create_distributed_dataloader(config, rank, world_size)
    
    # 🔧 激进显存优化配置
    scaler = torch.amp.GradScaler('cuda',
        init_scale=2**10,    # 更低初始缩放
        growth_factor=1.5,   # 更保守增长
        backoff_factor=0.25, # 更激进回退
        growth_interval=2000 # 更长检查间隔
    )
    
    # 🔧 显存优化的优化器配置
    # 注意：Kaggle环境可能没有bitsandbytes，使用标准AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-4
    )
    if rank == 0:
        print("⚠️ 使用标准AdamW优化器 (Kaggle环境限制)")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        effective_batch = batch_size * gradient_accumulation_steps * world_size
        print(f"✅ DDP模型就绪 - 参数量: {param_count/1e6:.1f}M")
        print(f"🔧 梯度检查点: 已启用 (节省显存)")
        print(f"🔧 混合精度: 稳定型FP16 (节省显存)")
        print(f"📊 批次配置: {batch_size}×{gradient_accumulation_steps}×{world_size} = {effective_batch}")
        if vae_model is not None:
            print(f"✅ VA-VAE模型就绪")
        else:
            print(f"⚠️ VA-VAE不可用")
    
    # 显存统计
    torch.cuda.empty_cache()
    memory_used = torch.cuda.memory_allocated(device) / (1024**3)
    
    if rank == 0:
        print(f"📊 显存使用: {memory_used:.1f}GB (GPU{rank})")
    
    # 最佳模型跟踪变量
    best_val_loss = float('inf')
    best_model_path = None
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        if dataloader is None:
            print("   ❌ 无数据加载器，退出训练")
            break
            
        # 设置sampler的epoch (重要!)
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}') if rank == 0 else dataloader
        
        for step, (latents, labels) in enumerate(progress_bar):
            try:
                # 🔧 数据类型优化 (保持输入FP32提高稳定性)
                latents = latents.to(device, dtype=torch.float32)  # 保持FP32输入
                labels = labels.to(device)
                
                batch_size = latents.shape[0]
                
                # 🔧 混合精度前向传播
                with torch.amp.autocast('cuda'):  # 修复废弃API警告
                    # 扩散过程
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
                    
                    # 添加噪声 (DDPM forward process)
                    alpha_t = 1 - timesteps / 1000
                    alpha_t = alpha_t.view(-1, 1, 1, 1)
                    noisy_latents = torch.sqrt(alpha_t) * latents + torch.sqrt(1 - alpha_t) * noise
                    
                    # 前向传播 - FP16自动混合精度
                    pred_noise = model(noisy_latents, timesteps, labels)
                    loss = criterion(pred_noise, noise) / gradient_accumulation_steps
                
                # 🔧 强化的数值稳定性检查
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                    print(f"\n⚠️ 检测到不稳定损失: {loss.item():.4f}，跳过批次")
                    # 降低缩放因子
                    scaler._scale = scaler._scale * 0.5
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                # 🔧 混合精度反向传播
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 🔧 稳定的混合精度梯度更新
                    scaler.unscale_(optimizer)
                    
                    # 🔧 激进梯度裁剪和稳定性检查
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
                    
                    # 🛡️ 梯度爆炸检查（放宽阈值）
                    if grad_norm > 100.0 or torch.isnan(grad_norm):
                        if rank == 0:
                            print(f"⚠️ 梯度爆炸: {grad_norm:.2f}, 跳过更新")
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)  # 释放梯度内存
                        continue
                    
                    # 🧹 定期清理显存碎片
                    if step % config['empty_cache_freq'] == 0:
                        torch.cuda.empty_cache()
                        if rank == 0 and step % 50 == 0:
                            mem_used = torch.cuda.memory_allocated(device) / 1024**3
                            print(f"🧹 显存清理 - 当前使用: {mem_used:.1f}GB")
                    
                    # 📈 执行优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # set_to_none=True释放更多内存
                    
                    # 更新进度条和梯度监控 (只在rank 0)
                    if rank == 0:
                        current_scale = scaler.get_scale()
                        progress_bar.set_postfix({
                            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                            'grad_norm': f'{grad_norm:.2f}',
                            'scale': f'{current_scale:.0f}',
                            'gpu_mem': f'{torch.cuda.memory_allocated(device) / (1024**3):.1f}GB'
                        })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ 显存不足，执行紧急清理: {e}")
                    # 🔧 紧急显存清理
                    if 'pred_noise' in locals(): del pred_noise
                    if 'noisy_latents' in locals(): del noisy_latents
                    if 'noise' in locals(): del noise
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Epoch完成 - 同步所有进程
        avg_loss = epoch_loss / max(len(dataloader), 1)
        scheduler.step()
        
        # 全局损失平均 (跨所有GPUs)
        if world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        if rank == 0:
            print(f"   平均损失: {avg_loss:.4f}")
            print(f"   学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 🔍 每个epoch验证评估 
        if rank == 0:
            print(f"\n🔍 Epoch {epoch+1} 验证评估:")
            validation_loss = evaluate_model(model, vae_model, device, rank)
            print(f"   验证损失: {validation_loss:.4f}")
            print(f"   训练损失: {avg_loss:.4f}")
            print(f"   差异: {abs(validation_loss - avg_loss):.4f}")
            
            # 🏆 检查是否为最佳模型并保存
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                
                # 🗑️ 删除旧的最佳模型
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    print(f"🗑️ 删除旧最佳模型: {os.path.basename(best_model_path)}")
                
                # 💾 保存新的最佳模型
                best_model_path = f"/kaggle/working/best_dit_xl_ddp_epoch_{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.module.state_dict(),  # 注意: model.module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': validation_loss,
                    'best_val_loss': best_val_loss,
                }, best_model_path)
                print(f"🏆 新最佳模型! 验证损失: {validation_loss:.4f}")
                print(f"💾 保存最佳模型: {os.path.basename(best_model_path)}")
            
            # 🖼️ 每个epoch生成条件扩散样本
            print(f"🎨 生成验证样本...")
            generate_validation_samples(model, vae_model, device, epoch+1)
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
        
        torch.cuda.empty_cache()
    
    # 训练完成
    if rank == 0:
        print("✅ DDP训练完成")
        
        # 保存最终模型
        final_save_path = "/kaggle/working/dit_xl_ddp_final.pt"
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'config': config,
        }, final_save_path)
        print(f"💾 最终模型: {final_save_path}")
    
    # 清理DDP
    cleanup_ddp()
    return model

def load_vae_model(device):
    """加载VA-VAE模型 - 优化版"""
    rank = device.index if hasattr(device, 'index') else 0
    if rank == 0:
        print(f"📂 加载VA-VAE模型:")
    
    # 导入VA-VAE
    try:
        from LightningDiT.tokenizer.autoencoder import AutoencoderKL
        if rank == 0:
            print(f"   📋 使用真正的VA-VAE实现")
    except ImportError as e:
        if rank == 0:
            print(f"   ❌ 无法导入AutoencoderKL: {e}")
        return None
    
    vae_checkpoint_path = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    if not os.path.exists(vae_checkpoint_path):
        if rank == 0:
            print(f"   ❌ 未找到VA-VAE权重文件: {vae_checkpoint_path}")
        return None
    
    try:
        # VA-VAE也采用CPU先加载策略
        if rank == 0:
            print(f"   🔄 CPU先加载VA-VAE，再转移到GPU{rank}...")
            
        vae_model = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            use_variational=True,
            ckpt_path=vae_checkpoint_path,
            model_type='vavae'
        )
        vae_model.eval().to(device)  # CPU → GPU转移
        
        if rank == 0:
            print(f"   ✅ VA-VAE模型加载成功")
            print(f"   📊 模型参数: embed_dim=32, 下采样=16x")
        return vae_model
    except Exception as e:
        if rank == 0:
            print(f"   ❌ VA-VAE模型加载失败: {e}")
        return None

def create_distributed_dataloader(config, rank, world_size):
    """创建分布式数据加载器"""
    if rank == 0:
        print(f"📊 创建分布式数据加载器:")
    
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 2)  # 每GPU 2个worker
    
    # 加载数据
    official_train_dir = Path("/kaggle/working/latents_official/vavae_config_for_dit/microdoppler_train_256")
    official_train_file = official_train_dir / "latents_rank00_shard000.safetensors"
    
    if not official_train_file.exists():
        if rank == 0:
            print(f"   ❌ 未找到训练数据: {official_train_file}")
        return None
        
    try:
        latents_data = safetensors.load_file(str(official_train_file))
        latents = latents_data['latents']  
        labels = latents_data['labels']

        # 🔧 应用官方LightningDiT归一化策略
        # latent_norm=true: 标准化到mean=0, std=1
        # latent_multiplier=1.0: 保持标准差为1.0

        # 检查是否有latents_stats.pt文件（官方预处理的统计信息）
        stats_file = official_train_dir / "latents_stats.pt"
        if stats_file.exists():
            if rank == 0:
                print(f"   📊 加载官方latent统计信息: {stats_file}")
            stats = torch.load(stats_file, map_location='cpu')
            latent_mean = stats['mean']
            latent_std = stats['std']
        else:
            # 计算当前数据的统计信息
            if rank == 0:
                print(f"   📊 计算latent统计信息...")
            latent_mean = latents.mean(dim=0, keepdim=True)
            latent_std = latents.std(dim=0, keepdim=True)

        # 应用官方归一化: (latent - mean) / std * multiplier
        if rank == 0:
            print(f"   🔧 应用官方latent_norm=true, latent_multiplier=1.0")
            print(f"   📈 原始latent范围: [{latents.min():.4f}, {latents.max():.4f}]")

        latents = (latents - latent_mean) / latent_std  # 标准化
        latents = latents * 1.0  # latent_multiplier=1.0

        if rank == 0:
            print(f"   📈 归一化后范围: [{latents.min():.4f}, {latents.max():.4f}]")
            print(f"   ✅ 成功加载并归一化训练数据")
            print(f"   📊 Latents形状: {latents.shape}")
            print(f"   📊 Labels形状: {labels.shape}")
            print(f"   📊 类别范围: {labels.min().item()} - {labels.max().item()}")

        # 创建数据集和数据加载器
        dataset = MicroDopplerLatentDataset(latents, labels)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None), 
            sampler=sampler,
            num_workers=0, 
            pin_memory=True
        )
        return dataloader

    except Exception as e:
        if rank == 0:
            print(f"   ⚠️ 数据加载失败: {e}")
        return None

def main():
    """主函数 - 启动DDP训练"""
    print("🎯 DiT XL DDP微调脚本 (Kaggle T4x2)")
    print("="*50)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA")
        return
    
    world_size = torch.cuda.device_count()
    print(f"🔧 检测到 {world_size} 个GPU")
    
    if world_size < 2:
        print("⚠️ 建议使用2个GPU进行DDP训练")
        world_size = 1  # 单GPU回退
    
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU{i}: {props.name} - {props.total_memory/1024**3:.1f}GB")
    
    # 🔧 稳定性优先的DDP配置
    config = {
        'learning_rate': 5e-6,  # 稳定的学习率
        'num_epochs': 15,
        'batch_size': 1,        # 每GPU批次大小
        'save_interval': 5,     # 减少IO开销
        'gradient_accumulation_steps': 12, # 更大累积步数减少显存压力
        'num_workers': 0,       # 禁用多进程减少内存
        'warmup_steps': 100,    # 添加热身阶段
        'use_fp32': False,      # 设为True即可回退FP32训练
        'max_grad_norm': 5.0,   # 放宽梯度裁剪阈值
        'empty_cache_freq': 10, # 每10步清理缓存
    }
    
    print(f"\n📋 DDP配置 (稳定性优先):")
    print(f"   World Size: {world_size}")
    print(f"   Backend: NCCL")
    effective_batch = config['batch_size'] * config['gradient_accumulation_steps'] * world_size
    print(f"   有效批次大小: {effective_batch} ({config['batch_size']}×{config['gradient_accumulation_steps']}×{world_size})")
    for key, value in config.items():
        if key != 'batch_size' and key != 'gradient_accumulation_steps':
            print(f"   {key}: {value}")
    
    if world_size > 1:
        # 启动DDP多进程训练
        print(f"\n🚀 启动DDP多进程训练...")
        mp.spawn(
            train_ddp_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # 单GPU训练
        print(f"\n🚀 启动单GPU训练...")
        train_ddp_worker(0, 1, config)
    
    print(f"\n🎉 DDP训练完成！")

if __name__ == "__main__":
    main()
