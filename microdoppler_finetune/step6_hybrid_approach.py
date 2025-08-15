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
from torch.utils.data import Dataset, DataLoader
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

# 设置路径
va_vae_root = Path("/kaggle/working/VA-VAE")
sys.path.append(str(va_vae_root))
sys.path.append(str(va_vae_root / "LightningDiT"))

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

# 导入必要的模块
from models.transport import create_transport
from models import LightningDiT_models
from vae.model import VA_VAE

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
def train_dit():
    """主训练函数"""
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
    
    # 创建EMA模型
    from copy import deepcopy
    ema_model = deepcopy(model).to(device)
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
    logger.info("=== 准备数据集 ===")
    
    # 首先尝试生成潜空间（如果需要）
    data_dir = Path("/kaggle/input/dataset")
    latents_file = data_dir / 'latents_microdoppler.npz'
    
    if not latents_file.exists():
        logger.info("预计算潜空间表示...")
        encode_dataset_to_latents(vae, data_dir, device)
    
    # 创建数据加载器
    train_dataset = MicroDopplerLatentDataset(data_dir, split='train')
    val_dataset = MicroDopplerLatentDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # T4内存限制
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
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
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            latents = batch['latent'].to(device)
            user_ids = batch['user_id'].to(device)
            
            # 采样时间步
            t = torch.randint(0, transport.num_timesteps, (latents.shape[0],), device=device)
            
            # 前向传播（混合精度）
            with torch.cuda.amp.autocast():
                # 添加噪声
                model_kwargs = {"y": user_ids}
                loss_dict = transport.training_losses(model, latents, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新EMA
            ema_decay = 0.9999
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1-ema_decay)
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                latents = batch['latent'].to(device)
                user_ids = batch['user_id'].to(device)
                
                t = torch.randint(0, transport.num_timesteps, (latents.shape[0],), device=device)
                
                with torch.cuda.amp.autocast():
                    model_kwargs = {"y": user_ids}
                    loss_dict = transport.training_losses(model, latents, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        # 更新学习率
        scheduler.step()
        
        # 日志记录
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            save_path = f"outputs/dit_best_epoch{epoch+1}_val{avg_val_loss:.4f}.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            logger.info(f"  ✓ Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 定期生成样本
        if (epoch + 1) % 5 == 0:
            logger.info("Generating samples...")
            generate_samples(ema_model, vae, transport, device, epoch+1)
        
        # 内存清理
        torch.cuda.empty_cache()
    
    logger.info("=== 训练完成 ===")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


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


if __name__ == "__main__":
    train_dit()
