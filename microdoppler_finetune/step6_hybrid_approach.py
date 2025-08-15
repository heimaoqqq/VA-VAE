#!/usr/bin/env python3
"""
混合策略：标准DiT训练 + 可选的用户区分增强
兼顾原项目思想和任务特定需求
"""

import os
import sys
import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# 添加LightningDiT到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT'))

# 先处理fairscale依赖问题
import torch.nn as nn

# 创建fairscale的mock替代，避免导入错误
class MockFairscaleModule:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return None

# Mock fairscale模块
import sys
if 'fairscale' not in sys.modules:
    import types
    fairscale_mock = types.ModuleType('fairscale')
    fairscale_mock.nn = types.ModuleType('nn')
    fairscale_mock.nn.model_parallel = types.ModuleType('model_parallel')
    fairscale_mock.nn.model_parallel.initialize = MockFairscaleModule()
    fairscale_mock.nn.model_parallel.layers = types.ModuleType('layers')
    fairscale_mock.nn.model_parallel.layers.ColumnParallelLinear = nn.Linear
    fairscale_mock.nn.model_parallel.layers.RowParallelLinear = nn.Linear
    fairscale_mock.nn.model_parallel.layers.ParallelEmbedding = nn.Embedding
    sys.modules['fairscale'] = fairscale_mock
    sys.modules['fairscale.nn'] = fairscale_mock.nn
    sys.modules['fairscale.nn.model_parallel'] = fairscale_mock.nn.model_parallel
    sys.modules['fairscale.nn.model_parallel.initialize'] = fairscale_mock.nn.model_parallel.initialize
    sys.modules['fairscale.nn.model_parallel.layers'] = fairscale_mock.nn.model_parallel.layers

try:
    from models.lightningdit import LightningDiT_models
    from tokenizer.vavae import VA_VAE
    from transport import create_transport
    print("Successfully imported LightningDiT models")
except ImportError as e:
    print(f"Error importing LightningDiT models: {e}")
    exit(1)

# 直接在此文件中定义，避免导入依赖问题
# from step6_standard_dit_training import MicroDopplerLatentDataset, create_logger

from torch.utils.data import Dataset
import logging


class MicroDopplerLatentDataset(Dataset):
    """微多普勒潜在空间数据集"""
    
    def __init__(self, data_root, vae_model, split='train', device='cuda'):
        self.data_root = Path(data_root)
        self.vae = vae_model
        self.device = device
        self.split = split
        
        # 收集所有数据
        self.samples = []
        
        for user_id in range(1, 32):  # ID_1 到 ID_31
            user_folder = self.data_root / f'ID_{user_id}'
            if user_folder.exists():
                images = list(user_folder.glob('*.jpg')) + list(user_folder.glob('*.png'))
                
                # 划分训练/验证集
                n_train = int(len(images) * 0.8)
                if split == 'train':
                    selected = images[:n_train]
                else:
                    selected = images[n_train:]
                
                for img_path in selected:
                    self.samples.append({
                        'path': str(img_path),
                        'class_id': user_id - 1,  # 0-30，作为类别ID
                    })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        img = Image.open(sample['path']).convert('RGB')
        
        # 转换为tensor：HWC -> CHW，[0,1] -> [-1,1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        # VAE编码到潜在空间
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            posterior = self.vae.encode(img_tensor)
            latent = posterior.sample().squeeze(0).cpu()  # (32, 16, 16)
        
        return {
            'latent': latent,
            'class_id': sample['class_id']
        }


def create_logger(logging_dir):
    """创建日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt")
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


class UserDiscriminationLoss(torch.nn.Module):
    """可选的用户区分损失 - 轻量级实现"""
    
    def __init__(self, temperature=0.1, weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, features, user_ids):
        """
        简单的InfoNCE损失
        features: (B, D) 特征向量
        user_ids: (B,) 用户ID
        """
        if features.dim() > 2:
            features = features.flatten(1)  # 展平特征
        
        # 归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # 创建正样本mask（同用户）
        user_ids = user_ids.unsqueeze(0)
        pos_mask = (user_ids == user_ids.t()).float()
        
        # 移除对角线（自己和自己）
        pos_mask.fill_diagonal_(0)
        
        # 如果batch中没有同用户的其他样本，返回0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 简化的对比损失
        pos_logits = sim_matrix * pos_mask
        neg_logits = sim_matrix * (1 - pos_mask)
        
        # 计算损失
        pos_loss = -torch.log(torch.exp(pos_logits).sum() / 
                             (torch.exp(pos_logits).sum() + torch.exp(neg_logits).sum()))
        
        return pos_loss * self.weight


def hybrid_dit_train(config_path='../configs/microdoppler_finetune.yaml', 
                     use_user_loss=True, user_loss_weight=0.1):
    """
    混合训练策略：
    - 主体：标准LightningDiT训练
    - 可选：轻量级用户区分损失
    """
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"User discrimination loss: {'ON' if use_user_loss else 'OFF'}")
    
    # 创建实验目录
    exp_name = f"hybrid_dit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(f"experiments/{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"User loss enabled: {use_user_loss}, weight: {user_loss_weight}")
    
    # 初始化VA-VAE
    logger.info("=== 初始化VA-VAE ===")
    vae = VA_VAE("../LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
    
    vae_checkpoint_path = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    if os.path.exists(vae_checkpoint_path):
        logger.info(f"Loading VA-VAE checkpoint: {vae_checkpoint_path}")
        vae.load_from_checkpoint(vae_checkpoint_path)
    else:
        logger.info("⚠️ VA-VAE checkpoint not found")
    
    vae.to(device)
    vae.eval()
    
    # 初始化DiT - 标准方式
    logger.info("=== 初始化LightningDiT ===")
    model = LightningDiT_models["LightningDiT-XL/1"](
        input_size=16,
        num_classes=31,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        in_channels=32,
    )
    
    # 加载预训练权重
    pretrained_path = "models/lightningdit-xl-imagenet256-64ep.pt"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model.to(device)
    
    # 可选的用户区分损失
    user_loss_fn = UserDiscriminationLoss(weight=user_loss_weight) if use_user_loss else None
    
    # 创建EMA
    from copy import deepcopy
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # Transport
    transport = create_transport(
        path_type="Linear",
        prediction="velocity", 
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        use_cosine_loss=True,
        use_lognorm=True,
    )
    
    # 数据集
    logger.info("=== 准备数据集 ===")
    train_dataset = MicroDopplerLatentDataset(
        data_root=config['data']['params']['data_dir'],
        vae_model=vae,
        split='train',
        device=device
    )
    
    val_dataset = MicroDopplerLatentDataset(
        data_root=config['data']['params']['data_dir'],
        vae_model=vae,
        split='val', 
        device=device
    )
    
    batch_size = 8  # 降低batch_size适应数据量少的情况
    logger.info(f"Using batch_size={batch_size} (reduced for small dataset)")
    
    # 特殊采样：确保每个batch包含多个用户（对比损失需要）
    if use_user_loss:
        from torch.utils.data.sampler import BatchSampler, RandomSampler
        
        # 创建平衡采样器
        class BalancedBatchSampler:
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
                
                # 按用户分组样本索引
                self.user_to_indices = {}
                for idx, sample in enumerate(dataset.samples):
                    user_id = sample['class_id']
                    if user_id not in self.user_to_indices:
                        self.user_to_indices[user_id] = []
                    self.user_to_indices[user_id].append(idx)
                
                self.num_users = len(self.user_to_indices)
            
            def __iter__(self):
                while True:
                    # 每个batch尝试包含多个不同用户
                    batch_indices = []
                    users_in_batch = min(self.batch_size, self.num_users)
                    selected_users = np.random.choice(list(self.user_to_indices.keys()), 
                                                    users_in_batch, replace=False)
                    
                    samples_per_user = self.batch_size // users_in_batch
                    for user_id in selected_users:
                        user_indices = self.user_to_indices[user_id]
                        selected = np.random.choice(user_indices, samples_per_user, replace=True)
                        batch_indices.extend(selected)
                    
                    # 填充到batch_size
                    while len(batch_indices) < self.batch_size:
                        random_user = np.random.choice(list(self.user_to_indices.keys()))
                        random_idx = np.random.choice(self.user_to_indices[random_user])
                        batch_indices.append(random_idx)
                    
                    yield batch_indices[:self.batch_size]
            
            def __len__(self):
                return len(self.dataset) // self.batch_size
        
        train_sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        # 标准随机采样
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 优化器 - 按原项目标准设置
    lr = 2e-4  # 原项目标准学习率，比config中7e-6更合适全参数训练
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0,  # 原项目标准：weight_decay=0
        betas=(0.9, 0.95)  # 原项目标准beta2=0.95
    )
    logger.info(f"Optimizer: AdamW, lr={lr}, beta2=0.95, weight_decay=0")
    
    # 训练循环
    logger.info("=== 开始训练 ===")
    num_epochs = config['trainer']['max_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        # 训练
        model.train()
        train_losses = {'total': [], 'diffusion': [], 'user': []}
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            latents = batch['latent'].to(device)
            class_ids = batch['class_id'].to(device)
            
            # 标准扩散训练
            t = torch.rand(latents.shape[0], device=device)
            x_1 = torch.randn_like(latents)
            x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * latents
            target = x_1 - latents
            
            model_output = model(x_t, t * 1000, class_ids)
            diffusion_loss = F.mse_loss(model_output, target)
            
            # 可选的用户区分损失
            total_loss = diffusion_loss
            user_loss_value = 0.0
            
            if use_user_loss and user_loss_fn is not None:
                # 使用模型的中间特征计算用户区分损失
                # 这里简化为使用模型输出作为特征
                user_loss_value = user_loss_fn(model_output, class_ids)
                total_loss = diffusion_loss + user_loss_value
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # EMA更新
            with torch.no_grad():
                for ema_param, param in zip(ema.parameters(), model.parameters()):
                    ema_param.data.mul_(0.9999).add_(param.data, alpha=0.0001)
            
            # 记录损失
            train_losses['total'].append(total_loss.item())
            train_losses['diffusion'].append(diffusion_loss.item())
            train_losses['user'].append(user_loss_value.item() if isinstance(user_loss_value, torch.Tensor) else user_loss_value)
            
            if batch_idx % 50 == 0:
                if use_user_loss:
                    logger.info(f"  Batch {batch_idx}: Total={total_loss.item():.4f} "
                              f"(Diff={diffusion_loss.item():.4f}, User={user_loss_value:.4f})")
                else:
                    logger.info(f"  Batch {batch_idx}: Loss={diffusion_loss.item():.4f}")
        
        # 训练统计
        avg_total = np.mean(train_losses['total'])
        avg_diff = np.mean(train_losses['diffusion'])
        avg_user = np.mean(train_losses['user'])
        
        logger.info(f"Train - Total: {avg_total:.4f}, Diffusion: {avg_diff:.4f}, User: {avg_user:.4f}")
        
        # 验证（仅使用扩散损失）
        if epoch % config['trainer']['check_val_every_n_epoch'] == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    latents = batch['latent'].to(device)
                    class_ids = batch['class_id'].to(device)
                    
                    t = torch.rand(latents.shape[0], device=device)
                    x_1 = torch.randn_like(latents)
                    x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * latents
                    target = x_1 - latents
                    
                    model_output = model(x_t, t * 1000, class_ids)
                    loss = F.mse_loss(model_output, target)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'config': config,
                    'use_user_loss': use_user_loss,
                    'user_loss_weight': user_loss_weight
                }, exp_dir / 'best_model.pt')
                logger.info(f"✅ Saved best model (val_loss={avg_val_loss:.4f})")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    logger.info(f"✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/microdoppler_finetune.yaml')
    parser.add_argument('--use_user_loss', action='store_true', help='Enable user discrimination loss')
    parser.add_argument('--user_loss_weight', type=float, default=0.1, help='Weight for user loss')
    args = parser.parse_args()
    
    hybrid_dit_train(args.config, args.use_user_loss, args.user_loss_weight)
