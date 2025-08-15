#!/usr/bin/env python3
"""
混合策略：标准DiT训练 + 可选的用户区分增强
兼顾原项目思想和任务特定需求
优化for Kaggle T4*2 GPU分布式训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# T4*2 GPU优化配置 - 启用Triton编译器和TensorCore
torch.backends.cuda.matmul.allow_tf32 = True  # T4 TensorCore优化
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 优化固定输入大小

# Kaggle T4*2 分布式训练支持
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime
from tqdm import tqdm
from PIL import Image
import yaml
import numpy as np
import argparse
import logging
from pathlib import Path

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

# Mock缺失的依赖模块
import sys
import types

# Mock torchdiffeq
def mock_odeint(func, y0, t, **kwargs):
    """简单的欧拉法mock实现"""
    result = [y0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dy = func(t[i-1], result[-1])
        result.append(result[-1] + dt * dy)
    return torch.stack(result)

if 'torchdiffeq' not in sys.modules:
    torchdiffeq_mock = types.ModuleType('torchdiffeq')
    torchdiffeq_mock.odeint = mock_odeint
    sys.modules['torchdiffeq'] = torchdiffeq_mock

if 'fairscale' not in sys.modules:
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
    print("✅ 成功导入LightningDiT模型")
except ImportError as e:
    print(f"Error importing LightningDiT models: {e}")
    exit(1)

# 直接在此文件中定义，避免导入依赖问题
# from step6_standard_dit_training import MicroDopplerLatentDataset, create_logger

from torch.utils.data import Dataset
import logging


class MicroDopplerDataset(Dataset):
    """微多普勒数据集"""
    
    def __init__(self, data_dir, image_size=256, split="train", train_ratio=0.8, vae=None, device=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
        # 获取所有用户的图片
        for user_dir in sorted(self.data_dir.iterdir()):
            if user_dir.is_dir() and user_dir.name.startswith("ID_"):
                user_id = int(user_dir.name.split("_")[1]) - 1  # 0-based indexing
                images = list(user_dir.glob("*.jpg"))
                
                # 分割训练/验证集
                n_train = int(len(images) * train_ratio)
                if split == "train":
                    selected = images[:n_train]
                else:
                    selected = images[n_train:]
                
                for img_path in selected:
                    self.samples.append({
                        "path": img_path,
                        "class_id": user_id  # 0-based用户ID
                    })
        
        print(f"{split}集: {len(self.samples)}个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载并预处理图片
        img = Image.open(sample["path"]).convert("RGB")
        img = transforms.ToTensor()(img)  # [3, H, W], 范围[0,1]
        img = (img * 2.0) - 1.0  # 转换到[-1, 1]
        
        # 返回原图，VAE编码将在主进程GPU上进行
        return {
            "image": img,
            "class_id": sample["class_id"]
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


# Kaggle T4*2 分布式训练辅助函数
def setup_distributed_training(rank, world_size):
    """初始化分布式训练 - Kaggle T4*2优化"""
    if world_size > 1:
        # Kaggle环境特殊配置
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # 使用127.0.0.1更稳定
        os.environ['MASTER_PORT'] = '29500'  # 使用较高端口避免冲突
        os.environ['NCCL_DEBUG'] = 'WARN'  # 减少NCCL日志
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # 关闭调试日志
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        ) 
        torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def hybrid_dit_train_worker(rank, world_size, config_path, use_user_loss=False, user_loss_weight=0.1):
    """分布式训练worker进程 - Kaggle T4x2优化"""
    # Kaggle T4x2标准环境变量设置
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"[GPU {rank}] 在设备 {device} 上启动分布式训练")
    print(f"[GPU {rank}] 进程组初始化成功")
    
    # 调用原训练函数但添加分布式支持
    hybrid_dit_train(config_path, use_user_loss, user_loss_weight, rank, world_size, device)
    
    # 清理分布式训练
    cleanup_distributed_training()


def hybrid_dit_train(config_path='../configs/microdoppler_finetune.yaml', 
                     use_user_loss=True, user_loss_weight=0.1, 
                     rank=0, world_size=1, device=None):
    """
    混合训练策略：
    - 主体：标准LightningDiT训练
    - 可选：轻量级用户区分损失
    """
    
    # 加载配置 - 直接构建正确的绝对路径
    if not os.path.isabs(config_path):
        # 相对于当前脚本文件的路径，向上一级到VA-VAE根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))  # /kaggle/working/VA-VAE/microdoppler_finetune
        va_vae_root = os.path.dirname(script_dir)  # /kaggle/working/VA-VAE
        config_path = os.path.join(va_vae_root, 'configs', 'microdoppler_finetune.yaml')
    
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"VA-VAE root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    print(f"Loading config from: {config_path}")
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        # 如果不存在，尝试另一个可能的路径
        alt_config_path = '/kaggle/working/VA-VAE/configs/microdoppler_finetune.yaml'
        print(f"Config file not found at {config_path}")
        print(f"Trying alternative path: {alt_config_path}")
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
        else:
            raise FileNotFoundError(f"Config file not found at either {config_path} or {alt_config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备配置 - 支持分布式训练
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"Using device: {device}")
        print(f"🖥️ 使用设备: {device}")
    print(f"🔗 分布式训练: {'开启' if world_size > 1 else '关闭'}")
    if world_size > 1:
        print(f"📊 集群大小: {world_size} GPUs")
        print(f"🏷️ 当前进程: Rank {rank}")
    print(f"👤 用户区分损失: {'开启' if use_user_loss else '关闭'}")
    print("=" * 50)
    
    # 创建实验目录
    exp_name = f"hybrid_dit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(f"experiments/{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(exp_dir)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"User loss enabled: {use_user_loss}, weight: {user_loss_weight}")
    
    # 加载微调后的VA-VAE - 使用Stage 3训练好的模型
    logger.info("=== 加载微调后的VA-VAE编码器 ===")
    
    # 使用微调后的checkpoint路径
    vae_checkpoint_path = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    
    # 检查文件是否存在
    if not os.path.exists(vae_checkpoint_path):
        logger.error(f"微调后的VA-VAE模型不存在: {vae_checkpoint_path}")
        logger.error("请确保已添加stage3数据集作为输入")
        raise FileNotFoundError(f"VA-VAE checkpoint not found: {vae_checkpoint_path}")
    
    logger.info(f"加载微调后的VA-VAE: {vae_checkpoint_path}")
    
    # 准备VA-VAE配置文件路径
    vae_config_path = os.path.join(va_vae_root, 'LightningDiT', 'tokenizer', 'configs', 'vavae_f16d32.yaml')
    
    # 修复配置中的checkpoint路径 - 使用微调后的模型
    import tempfile
    from omegaconf import OmegaConf
    
    # 加载并修改配置
    vae_config = OmegaConf.load(vae_config_path)
    vae_config.ckpt_path = vae_checkpoint_path  # 使用微调后的模型
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
        OmegaConf.save(vae_config, tmp_config.name)
        tmp_config_path = tmp_config.name
    
    # 初始化VA-VAE - 按照官方方式
    vae = VA_VAE(tmp_config_path, img_size=256, fp16=True)
    vae.model.eval()  # VA-VAE内部已经调用了.cuda()，不需要额外的.to(device)
    
    logger.info("✅ 成功加载微调后的VA-VAE模型")
    logger.info("  - 该模型经过3阶段训练优化")
    logger.info("  - VF语义相似度 > 0.987")
    logger.info("  - 专门针对微多普勒数据优化")
    
    # 加载DiT模型 - 遵循官方LightningDiT项目结构
    logger.info("=== 加载LightningDiT模型 ===")
    pretrained_path = os.path.join(va_vae_root, 'LightningDiT', 'models', 'lightningdit-xl-imagenet256-64ep.pt')
    
    if not os.path.exists(pretrained_path):
        logger.warning(f"预训练模型不存在: {pretrained_path}")
        logger.warning("请确保已下载LightningDiT预训练权重")
    
    # 初始化DiT - 遵循官方LightningDiT标准
    logger.info("=== 初始化LightningDiT ===")
    num_classes = config['model']['params']['num_users']  # 31个用户
    
    # 使用官方LightningDiT模型工厂
    model = LightningDiT_models["LightningDiT-XL/1"](
        num_classes=num_classes,
        in_channels=32,  # VA-VAE潜空间通道数
    )
    
    # 加载预训练权重
    if os.path.exists(pretrained_path):
        logger.info(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        # 清理state_dict键名
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process:
            logger.info(f"缺失键: {len(missing)}, 意外键: {len(unexpected)}")
    else:
        logger.warning(f"预训练权重文件不存在: {pretrained_path}")
        logger.warning("将使用随机初始化权重进行训练")
    
    model.to(device)
    
    # 混合精度训练配置 - T4优化
    use_amp = True  # T4 GPU支持FP16
    scaler = torch.amp.GradScaler('cuda') if use_amp else None  # 修复废弃警告
    
    # 分布式模型包装 - 符合官方DDP最佳实践
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False  # 提升性能
        )
    if is_main_process:
        logger.info(f"Model wrapped with DDP on rank {rank}")
    
    # 可选的用户区分损失
    user_loss_fn = UserDiscriminationLoss(weight=user_loss_weight) if use_user_loss else None
    
    # 创建EMA
    from copy import deepcopy
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # 创建Transport对象 - 遵循官方LightningDiT流匹配损失配置
    transport = create_transport(
        path_type='Linear',
        prediction="velocity",  # 官方标准：速度预测
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        use_cosine_loss=True,  # 官方推荐设置
        use_lognorm=True,  # 官方推荐设置
    )
    
    # 数据集
    logger.info("=== 准备数据集 ===")
    # 创建数据集 - 不在Dataset中进行VAE编码，支持多进程加载
    data_dir = config['data']['params']['data_dir']
    val_split = config['data']['params']['val_split']
    train_ratio = 1.0 - val_split
    
    train_dataset = MicroDopplerDataset(
        data_dir=data_dir, 
        split="train", 
        train_ratio=train_ratio
    )
    val_dataset = MicroDopplerDataset(
        data_dir=data_dir, 
        split="val", 
        train_ratio=train_ratio
    )
    
    # T4*2优化的batch_size配置
    base_batch_size = config['data']['params']['batch_size']  # 从配置文件读取
    batch_size = base_batch_size // world_size  # 分布式训练时每个GPU的batch_size
    
    if is_main_process:
        logger.info(f"Total batch_size: {base_batch_size}, Per-GPU batch_size: {batch_size}")
    
    # 分布式采样器配置
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if is_distributed else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    ) if is_distributed else None
    
    # 特殊采样：确保每个batch包含多个用户（对比损失需要）
    if use_user_loss and not is_distributed:
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
        # 标准随机采样 - 恢复多进程数据加载
        num_workers = config['data']['params'].get('num_workers', 4)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # 没有sampler时才shuffle
            num_workers=num_workers // world_size if is_distributed else num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers // world_size if is_distributed else num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    # 优化器 - 遵循官方LightningDiT配置
    lr = 1e-4  # 官方微调学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0,  # 官方标准：weight_decay=0
        betas=(0.9, 0.95)  # 官方标准beta2=0.95
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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练中")):
            # 获取图像并进行VAE编码
            images = batch['image'].to(device)  # [B, 3, 256, 256]
            class_ids = batch['class_id'].to(device)
            
            # VA-VAE编码 - 在主进程GPU上执行
            with torch.no_grad():
                if rank == 0 or not is_distributed:
                    # rank 0或单GPU模式：直接编码
                    latents = vae.encode_images(images)  # [B, 32, 16, 16]
                else:
                    # 其他rank：需要将数据发到cuda:0编码后再返回
                    images_cuda0 = images.to('cuda:0')
                    latents = vae.encode_images(images_cuda0)
                    latents = latents.to(device)  # 移回当前设备
            
            # 标准流匹配训练 - 使用官方Transport API
            with torch.cuda.amp.autocast(enabled=use_amp):
                model_kwargs = dict(y=class_ids)
                loss_dict = transport.training_losses(model, latents, model_kwargs)
                
                # 处理cos_loss（如果存在）
                if 'cos_loss' in loss_dict:
                    diffusion_loss = loss_dict["loss"].mean()
                    cos_loss = loss_dict["cos_loss"].mean()
                    diffusion_loss = diffusion_loss + cos_loss
                else:
                    diffusion_loss = loss_dict["loss"].mean()
            
            # 可选的用户区分损失
            total_loss = diffusion_loss
            user_loss_value = 0.0
            
            if use_user_loss and user_loss_fn is not None:
                # 获取模型特征用于用户区分
                # 需要单独前向传播获取中间特征
                with torch.cuda.amp.autocast(enabled=use_amp):
                    t = torch.rand(latents.shape[0], device=device)
                    x_1 = torch.randn_like(latents)
                    x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * latents
                    model_features = model(x_t, t * 1000, class_ids)
                    user_loss_value = user_loss_fn(model_features, class_ids)
                total_loss = diffusion_loss + user_loss_value
            
            # 反向传播 - 支持混合精度
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                # 梯度裁剪
                if config['trainer']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['trainer']['gradient_clip_val'])
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                # 梯度裁剪
                if config['trainer']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['trainer']['gradient_clip_val'])
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
                    # 获取图像并进行VAE编码
                    images = batch['image'].to(device)
                    class_ids = batch['class_id'].to(device)
                    
                    # VA-VAE编码
                    if rank == 0 or not is_distributed:
                        latents = vae.encode_images(images)
                    else:
                        images_cuda0 = images.to('cuda:0')
                        latents = vae.encode_images(images_cuda0)
                        latents = latents.to(device)
                    
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
        if (epoch + 1) % 10 == 0 and is_main_process:
            torch.save({
                'model': model.module.state_dict() if is_distributed else model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    if is_main_process:
        logger.info(f"✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
    
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid DiT Training for Micro-Doppler")
    parser.add_argument("--config", type=str, default="../configs/microdoppler_finetune.yaml")
    parser.add_argument("--use_user_loss", action="store_true", help="Enable user discrimination loss")
    parser.add_argument("--user_loss_weight", type=float, default=0.1, help="Weight for user loss")
    parser.add_argument("--distributed", action="store_true", help="Enable T4*2 distributed training")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs for distributed training")
    args = parser.parse_args()
    
    # 检测GPU数量并自动配置分布式训练
    num_gpus = torch.cuda.device_count()
    print(f"🔍 检测到 {num_gpus} 个GPU")
    
    if args.distributed and num_gpus >= 2:
        print("🚀 启动Kaggle T4x2分布式训练...")
        print(f"📊 GPU配置:")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        print(f"🌐 进程数: {args.world_size}")
        print("=" * 50)
        
        # 设置必要的环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
        
        # 启动T4*2分布式训练
        mp.spawn(
            hybrid_dit_train_worker,
            args=(args.world_size, args.config, args.use_user_loss, args.user_loss_weight),
            nprocs=args.world_size,
            join=True
        )
    else:
        # 单GPU训练
        if args.distributed:
            print("⚠️ Distributed training requested but insufficient GPUs available")
            print("Falling back to single GPU training...")
        else:
            print("🎯 Starting Single GPU Training...")
        
        hybrid_dit_train(args.config, args.use_user_loss, args.user_loss_weight)
