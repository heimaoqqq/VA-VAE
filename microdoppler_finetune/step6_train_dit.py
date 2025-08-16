"""
Step 6: 训练条件LightningDiT模型
使用微调后的VA-VAE潜空间进行micro-Doppler图像生成
针对Kaggle T4×2 GPU环境优化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

# 添加LightningDiT路径（参考step4）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))  # 添加根目录以导入自定义数据集

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kaggle T4内存优化设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# 禁用torch.compile
import torch._dynamo
torch._dynamo.disable()

# 分布式训练相关函数
def setup(rank, world_size):
    """初始化分布式训练 - Kaggle优化版本"""
    # Kaggle环境配置
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'  # 使用标准端口
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(seconds=60)
    )
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    
    # 打印分布式信息
    if rank == 0:
        logger.info(f"Initialized DDP with {world_size} GPUs")
        for i in range(world_size):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()

def is_main_process():
    """是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

# 导入LightningDiT模块
from transport import create_transport
from models.lightningdit import LightningDiT_models

# 导入VA-VAE
from tokenizer.vavae import VA_VAE

# ==================== 数据集定义 ====================
class MicroDopplerLatentDataset(Dataset):
    """微调后的潜空间数据集，包含用户条件"""
    
    def __init__(self, data_dir, split='train', val_ratio=0.2, latent_norm=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.latent_norm = latent_norm
        
        # 加载潜空间数据和标签 - 从可写目录加载
        latents_file = Path("/kaggle/working") / 'latents_microdoppler.npz'
        stats_file = Path("/kaggle/working") / 'latents_stats.pt'
        
        # 加载微多普勒数据集自己的统计信息（不使用官方ImageNet统计）
        if stats_file.exists():
            stats = torch.load(stats_file)
            self.latent_mean = stats['mean'].float()
            self.latent_std = stats['std'].float()
            logger.info(f"加载微多普勒潜空间统计: mean={self.latent_mean.shape}, std={self.latent_std.shape}")
        else:
            self.latent_mean = None
            self.latent_std = None
            
        if latents_file.exists():
            data = np.load(latents_file)
            self.latents = torch.from_numpy(data['latents']).float()
            self.user_ids = torch.from_numpy(data['user_ids']).long()
            
            # 如果没有统计信息，从数据中计算
            if self.latent_mean is None:
                self.latent_mean = self.latents.mean(dim=[0, 2, 3], keepdim=True)
                self.latent_std = self.latents.std(dim=[0, 2, 3], keepdim=True)
                logger.info("从数据计算潜空间统计信息")
            
            logger.info(f"Loaded {len(self.latents)} latent samples from {latents_file}")
            
            # 调试信息
            if len(self.latents) == 0:
                logger.error("潜空间文件存在但为空！")
                raise ValueError("Empty latents file")
        else:
            # 如果预计算的潜空间不存在，实时编码
            logger.warning(f"Pre-computed latents not found at {latents_file}")
            self.latents = None
            self.load_images()
        
        # 分割训练/验证集
        if self.latents is not None:
            n_samples = len(self.latents)
        elif hasattr(self, 'image_paths'):
            n_samples = len(self.image_paths)
        else:
            logger.error("没有数据可用于创建数据集")
            n_samples = 0
            
        if n_samples == 0:
            logger.error(f"数据集为空！latents={self.latents is not None}, "
                        f"has_image_paths={hasattr(self, 'image_paths')}")
            self.indices = np.array([])
        else:
            indices = np.arange(n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            n_val = int(n_samples * val_ratio)
            if split == 'train':
                self.indices = indices[n_val:]
            else:
                self.indices = indices[:n_val]
        
        logger.info(f"{split} dataset: {len(self.indices)} samples (total: {n_samples})")
    
    def load_images(self):
        """加载原始图像路径"""
        self.image_paths = []
        self.user_ids = []
        
        data_path = self.data_dir / 'processed_microdoppler'
        for user_dir in sorted(data_path.glob('ID_*')):
            user_id = int(user_dir.name.split('_')[1]) - 1
            for img_path in user_dir.glob('*.png'):
                self.image_paths.append(img_path)
                self.user_ids.append(user_id)
        
        self.user_ids = torch.tensor(self.user_ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        if self.latents is not None:
            latent = self.latents[real_idx]
            user_id = self.user_ids[real_idx]
            
            # 潜空间归一化
            if self.latent_norm and self.latent_mean is not None:
                # 确保维度匹配
                if latent.dim() == 3 and self.latent_mean.dim() == 4:
                    latent = latent.unsqueeze(0)  # 添加batch维度
                latent = (latent - self.latent_mean) / self.latent_std
                if latent.dim() == 4 and latent.shape[0] == 1:
                    latent = latent.squeeze(0)  # 移除临时batch维度
        else:
            # 实时编码（需要VAE模型）
            img_path = self.image_paths[real_idx]
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1)
            latent = image  # 这里需要VAE编码，暂时返回原图
            user_id = self.user_ids[real_idx]
        
        return latent, user_id


# ==================== 训练函数 ====================
def train_dit(rank=0, world_size=1):
    """主训练函数 - 支持分布式训练"""
    
    # 初始化分布式训练
    if world_size > 1:
        setup(rank, world_size)
    
    # 设备设置
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(rank)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(rank).total_memory / 1e9:.1f} GB")
    
    # ===== 1. 初始化VA-VAE（仅用于编码） =====
    logger.info("=== 初始化VA-VAE编码器 ===")
    # 创建临时配置文件，指向我们的checkpoint
    vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    
    # 加载原始配置并修改checkpoint路径
    vae_config_path = project_root / 'LightningDiT' / 'tokenizer' / 'configs' / 'vavae_f16d32.yaml'
    vae_config = OmegaConf.load(str(vae_config_path))
    vae_config.ckpt_path = vae_checkpoint  # 设置checkpoint路径
    
    # 保存修改后的配置到临时文件
    temp_config_path = project_root / 'temp_vavae_config.yaml'
    OmegaConf.save(vae_config, str(temp_config_path))
    
    # 使用修改后的配置初始化VA-VAE
    vae = VA_VAE(
        config=str(temp_config_path),
        img_size=256,
        horizon_flip=0.0,  # 训练时不需要水平翻转（数据增强对微多普勒时频图效果很差）
        fp16=True
    )
    
    # VA-VAE已经通过配置文件加载了checkpoint
    if os.path.exists(vae_checkpoint):
        logger.info("✓ VA-VAE loaded successfully with checkpoint")
    else:
        logger.warning(f"⚠️ VA-VAE checkpoint not found at {vae_checkpoint}")
        logger.warning("Using randomly initialized weights")
    
    # VA-VAE的model已经在初始化时调用了.cuda()，不需要.to(device)
    # vae.model已经是eval模式
    
    # ===== 2. 初始化LightningDiT-B模型 =====
    logger.info("=== 初始化LightningDiT-B ===")
    latent_size = 16  # 256/16 = 16
    num_users = 31
    
    model = LightningDiT_models["LightningDiT-B/1"](
        input_size=latent_size,
        num_classes=num_users,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        in_channels=32  # VA-VAE f16d32
    )
    
    logger.info(f"Model: LightningDiT-B/1 (768-dim, 12 layers)")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 加载预训练权重
    pretrained_base = "models/lightningdit-b-imagenet256.pt"
    pretrained_xl = "models/lightningdit-xl-imagenet256-64ep.pt"
    
    if os.path.exists(pretrained_base):
        logger.info(f"Loading Base model weights: {pretrained_base}")
        checkpoint = torch.load(pretrained_base, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif os.path.exists(pretrained_xl):
        logger.info(f"Base weights not found, trying partial XL loading: {pretrained_xl}")
        checkpoint = torch.load(pretrained_xl, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 只加载兼容的权重
        compatible = {}
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible[k] = v
        
        if compatible:
            model.load_state_dict(compatible, strict=False)
            logger.info(f"Loaded {len(compatible)} compatible weights from XL")
    else:
        logger.warning("No pretrained weights found, using random init")
    
    model.to(device)
    
    # 包装为分布式模型（Kaggle优化）
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=False,  # 设为False以提高性能
            broadcast_buffers=True,
            gradient_as_bucket_view=True  # 内存优化
        )
        if is_main_process():
            logger.info(f"Model wrapped with DDP on rank {rank}")
    
    # 创建EMA模型
    from copy import deepcopy
    ema_model = deepcopy(model.module if world_size > 1 else model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    
    # ===== 3. 创建Transport（扩散过程） =====
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None
    )
    
    # ===== 4. 准备数据集 =====
    if is_main_process():
        logger.info("=== 准备数据集 ===")
    
    # 首先尝试生成潜空间（如果需要）
    # 修改为正确的数据集路径 - 与step3_prepare_dataset.py一致
    data_dir = Path("/kaggle/input/dataset")
    latents_file = Path("/kaggle/working") / 'latents_microdoppler.npz'  # 保存到可写目录
    
    # 只在主进程中预计算潜空间，避免多进程冲突
    if not latents_file.exists() and is_main_process():
        logger.info("预计算潜空间表示...")
        encode_dataset_to_latents(vae, data_dir, device)
        logger.info("潜空间编码完成，等待文件写入...")
        # 确保文件已写入磁盘
        import time
        time.sleep(2)
    
    # 同步所有进程，确保潜空间数据已生成
    if world_size > 1:
        dist.barrier()
        logger.info(f"Rank {rank} 同步完成")
    
    # 创建数据采样器（分布式）
    train_dataset = MicroDopplerLatentDataset(data_dir, split='train')
    val_dataset = MicroDopplerLatentDataset(data_dir, split='val')
    
    # 分布式采样器（Kaggle优化）
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True  # 确保批次大小一致
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Kaggle环境数据加载器优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 每GPU批次大小
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=1,  # Kaggle环境减少worker数量
        pin_memory=True,
        persistent_workers=False,  # 避免Kaggle环境中的内存问题
        prefetch_factor=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        sampler=val_sampler,
        num_workers=1,  # 保持一致
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1
    )
    
    # ===== 5. 设置优化器 =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=25,
        eta_min=1e-7
    )
    
    # ===== 6. 训练循环 =====
    logger.info("=== 开始训练 ===")
    
    num_epochs = 25
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # 设置epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        # 只在主进程显示进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', disable=not is_main_process())
        for batch in pbar:
            latents = batch[0].to(device)
            user_ids = batch[1].to(device)
            
            # 前向传播（混合精度）
            with torch.cuda.amp.autocast():
                # Transport内部自动采样时间
                model_kwargs = {"y": user_ids}
                # 分布式训练时使用module
                dit_model = model.module if world_size > 1 else model
                loss_dict = transport.training_losses(dit_model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            
            # 定期清理显存缓存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # 更新EMA（只在主进程）
            if is_main_process():
                ema_decay = 0.9999
                model_params = model.module.parameters() if world_size > 1 else model.parameters()
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model_params):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1-ema_decay)
            
            train_loss += loss.item()
            train_steps += 1
            
            if is_main_process():
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', disable=not is_main_process())
            for batch in val_pbar:
                latents = batch[0].to(device)
                user_ids = batch[1].to(device)
                
                with torch.cuda.amp.autocast():
                    model_kwargs = {"y": user_ids}
                    dit_model = model.module if world_size > 1 else model
                    loss_dict = transport.training_losses(dit_model, latents, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        # 更新学习率
        scheduler.step()
        
        # 同步所有进程的损失（Kaggle优化）
        if world_size > 1:
            # 使用更安全的损失同步方式
            train_loss_tensor = torch.tensor(avg_train_loss, device=device, dtype=torch.float32)
            val_loss_tensor = torch.tensor(avg_val_loss, device=device, dtype=torch.float32)
            
            try:
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_train_loss = train_loss_tensor.item()
                avg_val_loss = val_loss_tensor.item()
            except Exception as e:
                if is_main_process():
                    logger.warning(f"Loss synchronization failed: {e}, using local loss")
        
        # 日志记录（只在主进程）
        if is_main_process():
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 早停检查（只在主进程）
        should_stop = False
        if is_main_process():
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                save_path = f"outputs/dit_best_epoch{epoch+1}_val{avg_val_loss:.4f}.pt"
                os.makedirs("outputs", exist_ok=True)
                model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, save_path)
                    logger.info(f"  ✓ Saved best model to {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save model: {e}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    should_stop = True
        
        # 在分布式训练中同步早停决策
        if world_size > 1:
            stop_tensor = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
        
        if should_stop:
            break
        
        # 定期生成样本（只在主进程）
        if (epoch + 1) % 5 == 0 and is_main_process():
            logger.info("Generating samples...")
            generate_samples(ema_model, vae, transport, device, epoch+1)
        
        # 内存清理
        torch.cuda.empty_cache()
    
    if is_main_process():
        logger.info("=== 训练完成 ===")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # 清理分布式训练
    if world_size > 1:
        cleanup()


def encode_dataset_to_latents(vae, data_dir, device, stats_path=None):
    """预计算并保存整个数据集的潜空间表示"""
    logger.info("编码数据集到潜空间...")
    
    latents_list = []
    user_ids_list = []
    
    # 遍历所有用户文件夹
    for user_dir in sorted(data_dir.glob('ID_*')):
        user_id = int(user_dir.name.split('_')[1]) - 1  # ID_1 -> 0
        
        # 收集该用户的所有图像（修正为.jpg格式）
        image_files = sorted(list(user_dir.glob('*.jpg')))
        
        if not image_files:
            logger.warning(f"用户 {user_dir.name} 没有找到.jpg图像文件")
            continue
            
        logger.info(f"编码用户 {user_dir.name}: {len(image_files)} 张图像")
        
        for img_path in image_files:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 编码到潜空间 - 使用VA_VAE的encode_images方法
            with torch.no_grad():
                latent = vae.encode_images(image)
            
            latents_list.append(latent.cpu().numpy())
            user_ids_list.append(user_id)
    
    # 检查是否有数据
    if not latents_list:
        logger.error("没有收集到任何潜空间数据！")
        raise ValueError("No latent data collected")
    
    # 保存潜空间数据
    latents = np.concatenate(latents_list, axis=0)
    user_ids = np.array(user_ids_list)
    
    logger.info(f"准备保存 {len(latents)} 个潜空间样本")
    logger.info(f"潜空间形状: {latents.shape}")
    logger.info(f"用户ID数量: {len(user_ids)}, 唯一用户: {len(np.unique(user_ids))}")
    
    # 计算潜空间统计信息（用于归一化）
    mean = latents.mean(axis=(0, 2, 3), keepdims=True)  # [1, C, 1, 1]
    std = latents.std(axis=(0, 2, 3), keepdims=True)
    
    # 保存到可写目录
    save_path = Path("/kaggle/working") / 'latents_microdoppler.npz'
    stats_save_path = Path("/kaggle/working") / 'latents_stats.pt'
    
    np.savez(save_path, latents=latents, user_ids=user_ids, mean=mean, std=std)
    torch.save({'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std)}, stats_save_path)
    
    logger.info(f"成功保存潜空间数据到 {save_path}")
    logger.info(f"成功保存统计信息到 {stats_save_path}")
    logger.info(f"潜空间均值形状: {mean.shape}, 标准差形状: {std.shape}")
    
    # 验证保存
    test_data = np.load(save_path)
    logger.info(f"验证: 加载了 {len(test_data['latents'])} 个样本")


def get_training_config():
    """获取训练配置参数"""
    return {
        'batch_size': 2,           # 每GPU批次大小（减少显存占用）
        'num_epochs': 25,          # 训练轮数
        'learning_rate': 1e-4,     # 学习率
        'weight_decay': 0.01,      # 权重衰减
        'num_workers': 1,          # 数据加载器worker数（减少CPU内存）
        'patience': 5,             # 早停耐心
        'gradient_clip_norm': 1.0, # 梯度裁剪
        'warmup_steps': 100,       # 学习率预热步数
        'gradient_checkpointing': True,  # 启用梯度检查点
    }

def print_training_config(model, optimizer, scheduler, config, 
                         train_size, val_size, world_size, train_dataset=None):
    """输出详细的训练配置信息"""
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 获取模型配置
    if hasattr(model, 'module'):  # DDP wrapped
        model_config = model.module
    else:
        model_config = model
    
    # 动态获取用户类别数
    num_classes = 31  # 默认值
    if train_dataset is not None:
        try:
            # 尝试从数据集获取用户数量
            if hasattr(train_dataset, 'user_ids'):
                num_classes = len(torch.unique(train_dataset.user_ids))
            elif hasattr(train_dataset, 'num_classes'):
                num_classes = train_dataset.num_classes
        except:
            pass  # 使用默认值
    
    print("\n" + "="*80)
    print("🚀 DiT微调训练配置")
    print("="*80)
    
    print(f"📊 数据配置:")
    print(f"  训练样本数: {train_size:,}")
    print(f"  验证样本数: {val_size:,}")
    print(f"  每GPU批量大小: {config['batch_size']}")
    print(f"  总批量大小: {config['batch_size'] * world_size}")
    print(f"  用户类别数: {num_classes}")
    
    # 动态获取模型信息
    model_type = getattr(model_config, '__class__', type(model_config)).__name__
    input_channels = getattr(model_config, 'in_channels', 'Unknown')
    input_size = getattr(model_config, 'input_size', 'Unknown')
    
    print(f"\n🏗️  模型配置:")
    print(f"  模型类型: {model_type}")
    if input_size != 'Unknown':
        print(f"  输入尺寸: {input_size}×{input_size} (潜空间)")
    else:
        print(f"  输入尺寸: 推断为16×16 (潜空间)")
    print(f"  输入通道: {input_channels}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    if hasattr(model_config, 'depth'):
        print(f"  Transformer层数: {model_config.depth}")
    if hasattr(model_config, 'hidden_size'):
        print(f"  隐藏层维度: {model_config.hidden_size}")
    if hasattr(model_config, 'num_heads'):
        print(f"  注意力头数: {model_config.num_heads}")
    
    print(f"\n⚙️  训练配置:")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  优化器: {optimizer.__class__.__name__}")
    print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  权重衰减: {optimizer.param_groups[0]['weight_decay']:.2e}")
    print(f"  梯度裁剪: {config['gradient_clip_norm']}")
    print(f"  数据加载器worker数: {config['num_workers']}")
    print(f"  调度器: {scheduler.__class__.__name__}")
    print(f"  混合精度: 启用")
    
    print(f"\n🔧 硬件配置:")
    print(f"  GPU数量: {world_size}")
    print(f"  并行方式: {'DistributedDataParallel' if world_size > 1 else 'Single GPU'}")
    if world_size > 1:
        print(f"  通信后端: NCCL")
    
    print(f"\n📈 评估指标:")
    print(f"  • 训练/验证损失")
    print(f"  • 梯度范数")
    print(f"  • 学习率变化")
    print(f"  • 每秒处理样本数")
    print(f"  • GPU内存使用率")
    
    print("="*80 + "\n")


def calculate_metrics(model, loss, optimizer):
    """计算训练质量评估指标"""
    metrics = {}
    
    # 基础损失
    metrics['loss'] = loss.item()
    
    # 梯度范数
    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        metrics['grad_norm'] = total_norm ** (1. / 2)
    else:
        metrics['grad_norm'] = 0.0
    
    # 学习率
    metrics['lr'] = optimizer.param_groups[0]['lr']
    
    return metrics


def train_dit_kaggle():
    """Kaggle环境下的DiT微调训练"""
    logger.info("开始Kaggle DiT微调训练...")
    
    # 检查GPU状态
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU进行训练")
    
    world_size = torch.cuda.device_count()
    logger.info(f"检测到 {world_size} 个GPU")
    
    # 强制检查是否真的有2个GPU
    if world_size == 1:
        # 尝试手动检测第二个GPU
        try:
            torch.cuda.get_device_name(1)
            logger.warning("发现第二个GPU但torch.cuda.device_count()只返回1，强制设置world_size=2")
            world_size = 2
        except:
            logger.info("确认只有1个GPU可用")
    
    # 显示GPU信息
    for i in range(world_size):
        try:
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")
            # 测试GPU可用性
            torch.cuda.set_device(i)
            test_tensor = torch.randn(10, device=f'cuda:{i}')
            logger.info(f"GPU {i} 可用性测试: 通过")
            del test_tensor
        except Exception as e:
            logger.error(f"GPU {i} 不可用: {e}")
            if i == 1:  # 如果GPU 1不可用，回退到单GPU
                world_size = 1
                break
    
    # 预处理：确保潜空间数据准备好
    latents_file = Path("/kaggle/working") / 'latents_microdoppler.npz'
    if not latents_file.exists():
        logger.info("预计算潜空间数据...")
        prepare_latents_for_training()
        logger.info("潜空间预计算完成")
    
    # 配置多进程环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 使用IP而非localhost
    os.environ['MASTER_PORT'] = '12457'      # 换一个端口避免冲突
    # Kaggle环境优化
    os.environ['NCCL_DEBUG'] = 'INFO'        # 更详细的调试信息
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_TIMEOUT'] = '1800'      # 增加超时时间
    os.environ['NCCL_BLOCKING_WAIT'] = '1'   # 阻塞等待
    
    logger.info(f"最终world_size: {world_size}")
    
    if world_size > 1:
        logger.info(f"启动 {world_size} 进程分布式训练...")
        # 多GPU DDP训练
        try:
            # 显式检查两个GPU是否真的可用
            for i in range(world_size):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                memory_free = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, 总显存: {memory_free:.1f}GB")
            
            # 测试进程间通信
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', int(os.environ['MASTER_PORT'])))
                sock.close()
                logger.info(f"端口 {os.environ['MASTER_PORT']} 可用")
            except Exception as port_e:
                logger.warning(f"端口测试失败: {port_e}")
                # 尝试其他端口
                for test_port in range(12400, 12500):
                    try:
                        sock.bind(('127.0.0.1', test_port))
                        sock.close()
                        os.environ['MASTER_PORT'] = str(test_port)
                        logger.info(f"使用端口 {test_port}")
                        break
                    except:
                        continue
                else:
                    raise Exception("无法找到可用端口")
            
            # 启动多进程训练
            logger.info(f"启动多进程，端口: {os.environ['MASTER_PORT']}")
            torch.multiprocessing.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            logger.error(f"多GPU训练失败: {e}")
            logger.info("回退到单GPU训练...")
            world_size = 1
            train_worker(0, 1)
    else:
        logger.info("使用单GPU训练...")
        train_worker(0, 1)


def prepare_latents_for_training():
    """预处理：只在主进程中准备潜空间数据"""
    # 检查GPU状态和显存
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    logger.info("=== 准备潜空间数据 ===")
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"初始显存使用: {initial_memory:.2f}GB")
    
    # 修正配置中的checkpoint路径为微调模型
    logger.info("准备VA-VAE配置...")
    vae_config_path = Path("/kaggle/working/VA-VAE/LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
    
    with open(vae_config_path, 'r') as f:
        config_content = f.read()
    
    # 使用微调后的模型  
    finetuned_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    if finetuned_checkpoint not in config_content:
        config_content = config_content.replace(
            'ckpt_path: /path/to/checkpoint.pt',
            f'ckpt_path: {finetuned_checkpoint}'
        )
        with open(vae_config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"已更新VA-VAE checkpoint路径: {finetuned_checkpoint}")
    
    # 初始化VA-VAE进行潜空间编码
    from tokenizer.vavae import VA_VAE
    logger.info("加载VA-VAE模型...")
    vae = VA_VAE(str(vae_config_path), img_size=256, horizon_flip=False, fp16=True)
    vae_memory = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"VA-VAE加载完成，显存: {vae_memory:.2f}GB")
    
    # 预计算潜空间
    data_dir = Path("/kaggle/input/dataset")
    latents_file = Path("/kaggle/working") / 'latents_microdoppler.npz'
    if not latents_file.exists():
        logger.info("开始编码数据集到潜空间...")
        encode_dataset_to_latents(vae, data_dir, device)
        logger.info("潜空间编码完成")
    else:
        logger.info("潜空间文件已存在，跳过编码")
    
    # 释放VA-VAE显存
    del vae
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"VA-VAE释放后显存: {final_memory:.2f}GB")


def train_worker(rank, world_size):
    """DDP训练工作进程 - 不加载VA-VAE，只处理DiT训练"""
    
    # 获取训练配置
    config = get_training_config()
    
    # 初始化分布式环境
    if world_size > 1:
        # 从环境变量获取配置
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '12457')
        
        logger.info(f"进程{rank}: 连接到 {master_addr}:{master_port}")
        
        # 初始化进程组，指定设备
        torch.distributed.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size,
            init_method=f'tcp://{master_addr}:{master_port}'
        )
    
    # 设备设置
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    
    # 清理CUDA缓存
    torch.cuda.empty_cache()
    
    # 显存使用优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    logger.info(f"进程 {rank}/{world_size}: 使用设备 {device}")
    
    # 检查初始显存状态（所有进程都报告）
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"进程{rank}初始显存使用: {initial_memory:.2f}GB")
    
    # 验证进程分配
    if world_size > 1:
        logger.info(f"进程{rank}: 等待所有进程同步...")
        torch.distributed.barrier()
        logger.info(f"进程{rank}: 同步完成")
    
    # 创建数据集 - 只使用预计算的潜空间
    data_dir = Path("/kaggle/input/dataset")
    train_dataset = MicroDopplerLatentDataset(data_dir, split='train')
    val_dataset = MicroDopplerLatentDataset(data_dir, split='val')
    
    # DDP数据采样器
    batch_size = config['batch_size']  # 每个GPU的batch size
    
    if world_size > 1:
        logger.info(f"进程{rank}: 配置DistributedSampler，总进程={world_size}")
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False
        
        # 报告数据分片情况
        logger.info(f"进程{rank}: 训练数据分片 {len(train_sampler)} samples")
        logger.info(f"进程{rank}: 验证数据分片 {len(val_sampler)} samples") 
    else:
        logger.info("单GPU模式: 使用标准DataLoader")
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    if rank == 0:
        logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 初始化模型
    if rank == 0:
        logger.info("初始化DiT模型...")
    
    from transport import create_transport
    from models.lightningdit import LightningDiT
    
    model = LightningDiT(
        input_size=16,
        patch_size=1,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=31,
        learn_sigma=False
    ).to(device)
    
    # 启用梯度检查点以节省显存
    if config.get('gradient_checkpointing', False):
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        # 为transformer层启用梯度检查点
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                if hasattr(block, 'checkpoint'):
                    block.checkpoint = True
    
    # DDP包装
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )
        if rank == 0:
            logger.info(f"使用DistributedDataParallel，{world_size}个GPU")
    else:
        if rank == 0:
            logger.info("使用单GPU训练")
    
    # Transport配置
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        use_cosine_loss=True,
        use_lognorm=True
    )
    
    # 优化器和调度器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs']
    )
    
    # 输出详细训练配置
    if rank == 0:
        print_training_config(model, optimizer, scheduler, config, 
                             len(train_dataset), len(val_dataset), world_size, train_dataset)
    
    # 混合精度训练（使用更积极的缩放策略）
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1024.0,  # 降低初始缩放
        growth_factor=1.1,   # 减少增长因子
        backoff_factor=0.8   # 增加回退因子
    )
    
    # 显存监控
    if rank == 0:
        model_memory = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"模型加载后显存: {model_memory:.2f}GB")
    
    # 训练循环
    best_val_loss = float('inf')
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # DDP sampler需要设置epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        train_grad_norm = 0
        train_samples = 0
        
        # 只在主进程显示进度条
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            try:
                batch_start_time = time.time()
                
                # 调试：第一个batch的详细信息
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"进程{rank}: 开始处理第一个batch")
                
                latents = batch[0].to(device)
                user_ids = batch[1].to(device)
                
                # 调试：检查第一个batch的shape（仅rank 0）
                if epoch == 0 and batch_idx == 0 and rank == 0:
                    logger.info(f"\n📋 数据格式检查:")
                    logger.info(f"  潜空间shape: {latents.shape}")
                    logger.info(f"  用户ID shape: {user_ids.shape}")
                    logger.info(f"  潜空间数据类型: {latents.dtype}")
                    logger.info(f"  潜空间数值范围: [{latents.min():.3f}, {latents.max():.3f}]")
                    print()
                
                # 前向传播
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"进程{rank}: 开始前向传播")
                
                with torch.cuda.amp.autocast():
                    model_kwargs = {"y": user_ids}
                    loss_dict = transport.training_losses(model, latents, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"进程{rank}: 前向传播完成，loss={loss.item():.4f}")
            
            except Exception as e:
                logger.error(f"进程{rank}: batch {batch_idx} 出错: {e}")
                raise
            
            # 反向传播
            if epoch == 0 and batch_idx == 0:
                logger.info(f"进程{rank}: 开始反向传播")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            if epoch == 0 and batch_idx == 0:
                logger.info(f"进程{rank}: backward完成，开始梯度裁剪")
            
            scaler.unscale_(optimizer)
            
            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            
            if epoch == 0 and batch_idx == 0:
                logger.info(f"进程{rank}: 梯度裁剪完成，grad_norm={grad_norm:.4f}")
            
            scaler.step(optimizer)
            scaler.update()
            
            if epoch == 0 and batch_idx == 0:
                logger.info(f"进程{rank}: 优化器step完成")
            
            # 定期清理显存缓存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # 统计
            train_loss += loss.item()
            train_grad_norm += grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            train_steps += 1
            train_samples += latents.size(0)
            
            if rank == 0:
                # 计算当前指标
                current_lr = optimizer.param_groups[0]['lr']
                samples_per_sec = latents.size(0) / (time.time() - batch_start_time)
                
                # 显存监控
                current_memory = torch.cuda.memory_allocated(device) / 1024**3
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'sps': f'{samples_per_sec:.1f}',
                    'mem': f'{current_memory:.1f}GB'
                })
                
                # 显存警告
                if current_memory > 13.0:  # T4总显存约14.7GB，警告阈值13GB
                    logger.warning(f"显存使用过高: {current_memory:.2f}GB")
                    
            # 第一个batch完成后的同步
            if epoch == 0 and batch_idx == 0 and world_size > 1:
                logger.info(f"进程{rank}: 第一个batch完成，等待同步...")
                torch.distributed.barrier()
                logger.info(f"进程{rank}: 同步完成")
        
        # 训练阶段统计
        avg_train_loss = train_loss / train_steps
        avg_train_grad_norm = train_grad_norm / train_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        val_samples = 0
        
        with torch.no_grad():
            if rank == 0:
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            else:
                val_pbar = val_loader
                
            for batch in val_pbar:
                latents = batch[0].to(device)
                user_ids = batch[1].to(device)
                
                with torch.cuda.amp.autocast():
                    model_kwargs = {"y": user_ids}
                    loss_dict = transport.training_losses(model, latents, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                val_loss += loss.item()
                val_steps += 1
                val_samples += latents.size(0)
                
                if rank == 0:
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_steps
        scheduler.step()
        
        # 计算epoch统计
        epoch_time = time.time() - epoch_start_time
        train_samples_per_sec = train_samples / epoch_time * world_size  # 总吞吐量
        current_lr = optimizer.param_groups[0]['lr']
        
        # GPU内存统计（仅rank 0）
        if rank == 0:
            gpu_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            
            # 输出详细的epoch统计
            print(f"\n📊 Epoch {epoch+1}/{config['num_epochs']} 训练统计")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🎯 损失指标:")
            print(f"  训练损失: {avg_train_loss:.6f}")
            print(f"  验证损失: {avg_val_loss:.6f}")
            print(f"  损失变化: {avg_val_loss - avg_train_loss:+.6f}")
            
            print(f"\n⚡ 训练动态:")
            print(f"  梯度范数: {avg_train_grad_norm:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  训练时间: {epoch_time:.1f}s")
            print(f"  训练吞吐: {train_samples_per_sec:.1f} samples/sec")
            
            print(f"\n💾 资源使用:")
            print(f"  GPU显存峰值: {gpu_memory:.2f}GB")
            print(f"  训练样本数: {train_samples}")
            print(f"  验证样本数: {val_samples}")
            
            # 训练质量评估
            is_improving = avg_val_loss < best_val_loss
            improvement_rate = (best_val_loss - avg_val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
            
            print(f"\n📈 质量评估:")
            print(f"  模型改进: {'✅ 是' if is_improving else '❌ 否'}")
            if is_improving:
                print(f"  改进幅度: {improvement_rate:.2f}%")
                print(f"  最佳损失: {avg_val_loss:.6f}")
            else:
                print(f"  最佳损失: {best_val_loss:.6f}")
            
            # 过拟合检测
            overfitting_gap = avg_train_loss - avg_val_loss
            if overfitting_gap < -0.01:
                print(f"  ⚠️  可能过拟合 (gap: {overfitting_gap:.4f})")
            elif overfitting_gap > 0.05:
                print(f"  ⚠️  可能欠拟合 (gap: {overfitting_gap:.4f})")
            else:
                print(f"  ✅ 拟合良好 (gap: {overfitting_gap:.4f})")
            
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            # 最佳模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if rank == 0:
                    best_model_path = Path("/kaggle/working") / f"best_dit_epoch_{epoch+1}.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': avg_val_loss,
                        'config': config
                    }, best_model_path)
                    logger.info(f"保存最佳模型到 {best_model_path}")
        
        # 更新学习率
        scheduler.step()
    
    # 训练完成
    if rank == 0:
        logger.info("🎉 训练完成!")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存最终模型
        final_model_path = Path("/kaggle/working") / "final_dit_model.pt"
        torch.save({
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'training_history': {
                'train_metrics': train_metrics_history,
                'val_metrics': val_metrics_history
            }
        }, final_model_path)
        logger.info(f"保存最终模型到 {final_model_path}")
    
    # 清理分布式环境
    if world_size > 1:
        logger.info(f"进程{rank}: 清理分布式环境...")
        torch.distributed.destroy_process_group()
        logger.info(f"进程{rank}: 分布式环境清理完成")
        logger.info("训练完成！")


def generate_samples(model, vae, transport, device, epoch):
    """生成样本用于可视化"""
    model.eval()
    vae.eval()
    
    with torch.no_grad():
        # 为每个用户生成一个样本
        num_samples = min(8, 31)  # 生成8个用户的样本
        user_ids = torch.arange(num_samples, device=device)
        
        # 采样潜空间
        z = torch.randn(num_samples, 32, 16, 16, device=device)
        model_kwargs = {"y": user_ids}
        
        # 使用DDIM采样
        samples = transport.sample_with_model_kwargs(
            model, z, model_kwargs, 
            steps=50,  # 采样步数
            eta=0.0,   # DDIM确定性采样
            guidance_scale=2.0  # CFG强度
        )
        
        # 解码到图像空间
        images = vae.decode(samples)
        images = (images + 1) / 2  # [-1,1] -> [0,1]
        images = images.clamp(0, 1)
        
        # 保存图像
        save_dir = Path("outputs/samples")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            img_pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            img_pil.save(save_dir / f"epoch{epoch}_user{user_ids[i].item()}.png")
        
        logger.info(f"Saved {num_samples} samples to {save_dir}")


def main():
    """主函数 - Kaggle T4x2优化版本"""
    # Kaggle环境GPU检查
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Please enable GPU accelerator in Kaggle.")
        return
    
    world_size = torch.cuda.device_count()
    logger.info(f"Detected {world_size} GPU(s)")
    
    # Kaggle T4x2使用DataParallel，不使用分布式训练
    train_dit_kaggle()

if __name__ == "__main__":
    main()
