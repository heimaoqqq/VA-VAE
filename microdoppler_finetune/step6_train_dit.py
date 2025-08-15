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
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
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
        timeout=datetime.timedelta(seconds=60)
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
    
    def __init__(self, data_dir, split='train', val_ratio=0.2):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 加载潜空间数据和标签
        latents_file = self.data_dir / 'latents_microdoppler.npz'
        if latents_file.exists():
            data = np.load(latents_file)
            self.latents = torch.from_numpy(data['latents']).float()
            self.user_ids = torch.from_numpy(data['user_ids']).long()
            logger.info(f"Loaded {len(self.latents)} latent samples")
        else:
            # 如果预计算的潜空间不存在，实时编码
            logger.warning("Pre-computed latents not found, will encode on-the-fly")
            self.latents = None
            self.load_images()
        
        # 分割训练/验证集
        n_samples = len(self.user_ids) if self.latents is not None else len(self.image_paths)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        n_val = int(n_samples * val_ratio)
        if split == 'train':
            self.indices = indices[n_val:]
        else:
            self.indices = indices[:n_val]
        
        logger.info(f"{split} dataset: {len(self.indices)} samples")
    
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
        else:
            # 实时编码（需要VAE模型）
            img_path = self.image_paths[real_idx]
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1)
            latent = image  # 这里需要VAE编码，暂时返回原图
            user_id = self.user_ids[real_idx]
        
        return {
            'latent': latent,
            'user_id': user_id
        }


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
    vae = VA_VAE(
        in_channels=3,
        out_channels=3,
        latent_dim=32,
        downsample_factor=16,
        num_res_blocks=2,
        norm='group',
        num_groups=32,
        learned_variance=False,
        use_vae_ema=True
    )
    
    # 加载微调后的VA-VAE权重
    vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    if os.path.exists(vae_checkpoint):
        logger.info(f"Loading VA-VAE checkpoint: {vae_checkpoint}")
        checkpoint = torch.load(vae_checkpoint, map_location='cpu')
        
        # 处理权重键名
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.')}
        
        vae.load_state_dict(state_dict, strict=False)
        logger.info("✓ VA-VAE loaded successfully")
    else:
        logger.warning("⚠️ VA-VAE checkpoint not found")
    
    vae.to(device)
    vae.eval()
    
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
    data_dir = Path("/kaggle/input/dataset")
    latents_file = data_dir / 'latents_microdoppler.npz'
    
    # 只在主进程中预计算潜空间，避免多进程冲突
    if not latents_file.exists() and is_main_process():
        logger.info("预计算潜空间表示...")
        encode_dataset_to_latents(vae, data_dir, device)
    
    # 同步所有进程，确保潜空间数据已生成
    if world_size > 1:
        dist.barrier()
    
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
            latents = batch['latent'].to(device)
            user_ids = batch['user_id'].to(device)
            
            # 采样时间步
            t = torch.randint(0, transport.num_timesteps, (latents.shape[0],), device=device)
            
            # 前向传播（混合精度）
            with torch.cuda.amp.autocast():
                # 添加噪声
                model_kwargs = {"y": user_ids}
                # 分布式训练时使用module
                dit_model = model.module if world_size > 1 else model
                loss_dict = transport.training_losses(dit_model, latents, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
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
                latents = batch['latent'].to(device)
                user_ids = batch['user_id'].to(device)
                
                t = torch.randint(0, transport.num_timesteps, (latents.shape[0],), device=device)
                
                with torch.cuda.amp.autocast():
                    model_kwargs = {"y": user_ids}
                    dit_model = model.module if world_size > 1 else model
                    loss_dict = transport.training_losses(dit_model, latents, t, model_kwargs)
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


def encode_dataset_to_latents(vae, data_dir, device):
    """预计算数据集的潜空间表示"""
    logger.info("Encoding dataset to latent space...")
    
    data_path = data_dir / 'processed_microdoppler'
    latents_list = []
    user_ids_list = []
    
    for user_dir in sorted(data_path.glob('ID_*')):
        user_id = int(user_dir.name.split('_')[1]) - 1
        
        for img_path in tqdm(list(user_dir.glob('*.png')), desc=f"Encoding {user_dir.name}"):
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 编码到潜空间
            with torch.no_grad():
                posterior = vae.encode(image)
                latent = posterior.sample() if hasattr(posterior, 'sample') else posterior
            
            latents_list.append(latent.cpu().numpy())
            user_ids_list.append(user_id)
    
    # 保存潜空间数据
    latents = np.concatenate(latents_list, axis=0)
    user_ids = np.array(user_ids_list)
    
    save_path = data_dir / 'latents_microdoppler.npz'
    np.savez(save_path, latents=latents, user_ids=user_ids)
    logger.info(f"Saved {len(latents)} latent samples to {save_path}")


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
    """主函数 - 处理分布式训练启动（Kaggle优化）"""
    # Kaggle环境GPU检查
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Please enable GPU accelerator in Kaggle.")
        return
    
    world_size = torch.cuda.device_count()
    logger.info(f"Detected {world_size} GPU(s)")
    
    # 显示GPU信息
    for i in range(world_size):
        gpu_props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
    
    if world_size > 1:
        logger.info(f"Starting distributed training with {world_size} GPUs")
        # Kaggle环境优化：使用spawn方法启动多进程
        try:
            mp.spawn(
                train_dit, 
                args=(world_size,), 
                nprocs=world_size, 
                join=True
            )
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            logger.info("Falling back to single GPU training...")
            train_dit(rank=0, world_size=1)
    else:
        logger.info("Starting single GPU training")
        train_dit(rank=0, world_size=1)

if __name__ == "__main__":
    main()
