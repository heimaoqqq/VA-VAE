#!/usr/bin/env python3
"""
Step 6 V2: LightningDiT条件生成训练
- 基于LightningDiT原项目架构
- 适配微多普勒数据集（31用户）
- Kaggle T4×2 GPU优化
- 实时VAE编码 + 每epoch生成样本
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 环境设置 ====================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))

# Kaggle T4内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 设置taming路径（必须在导入VAE之前）
def setup_taming_path():
    """设置taming路径"""
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),
        Path('/kaggle/working/.taming_path'),
        project_root / 'taming-transformers',
    ]
    
    for location in taming_locations:
        if location.name == '.taming_path' and location.exists():
            with open(location, 'r') as f:
                taming_path = f.read().strip()
            if Path(taming_path).exists():
                sys.path.insert(0, taming_path)
                return True
        elif location.exists():
            sys.path.insert(0, str(location))
            return True
    return False

setup_taming_path()

# 导入LightningDiT模块
from transport import create_transport, Sampler
from models.lightningdit import LightningDiT_models, LightningDiT_L_2

# 导入VA-VAE（需要先添加vavae路径）
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
from ldm.models.autoencoder import AutoencoderKL

# ==================== 数据集定义 ====================
class MicroDopplerDataset(Dataset):
    """微多普勒数据集 - 兼容step3_prepare_dataset.py输出"""
    
    def __init__(self, data_dir, split='train', split_file=None):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 使用step3_prepare_dataset.py的输出
        if split_file is None:
            split_file = Path('/kaggle/working/data_split/dataset_split.json')
        
        self.data_list = []
        
        if split_file.exists():
            print(f"📥 加载数据划分: {split_file}")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            # 加载用户标签映射
            labels_file = split_file.parent / 'user_labels.json'
            if labels_file.exists():
                with open(labels_file, 'r') as f:
                    user_labels = json.load(f)
            else:
                # 如果没有标签文件，创建默认映射 ID_1->0, ID_2->1, ...
                user_labels = {f"ID_{i}": i-1 for i in range(1, 32)}
            
            # 处理step3格式的数据 
            if split in split_data and isinstance(split_data[split], dict):
                for user_key, image_paths in split_data[split].items():
                    user_id = user_labels.get(user_key, 0)
                    for img_path in image_paths:
                        self.data_list.append({
                            'path': img_path,
                            'user_id': user_id
                        })
            else:
                print(f"⚠️ 未找到{split}数据，尝试自动扫描...")
                # 回退到自动扫描
                self._auto_scan_data()
        else:
            print(f"⚠️ 数据划分文件不存在: {split_file}")
            print("建议先运行: python step3_prepare_dataset.py")
            self._auto_scan_data()
        
        print(f"📊 {split}集: {len(self.data_list)}张图像")
    
    def _auto_scan_data(self):
        """自动扫描数据（兼容ID_1到ID_31格式）"""
        all_data = []
        for user_id in range(1, 32):  # ID_1 到 ID_31
            user_dir = self.data_dir / f"ID_{user_id}"
            if user_dir.exists():
                images = list(user_dir.glob("*.jpg"))  # step3使用.jpg
                for img_path in images:
                    all_data.append({
                        'path': str(img_path),
                        'user_id': user_id - 1  # 转换为0-30
                    })
        
        # 简单80/20划分
        np.random.seed(42)
        np.random.shuffle(all_data)
        split_idx = int(len(all_data) * 0.8)
        
        if self.split == 'train':
            self.data_list = all_data[:split_idx]
        else:
            self.data_list = all_data[split_idx:]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载图像
        image = Image.open(item['path']).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # 归一化到[-1, 1]
        image = image * 2.0 - 1.0
        
        return image, item['user_id']

# ==================== 模型初始化 ====================
def create_dit_model(device='cuda'):
    """创建并初始化LightningDiT-L模型"""
    
    # 创建L模型
    model = LightningDiT_L_2(
        input_size=16,  # 16×16潜空间
        num_classes=31,  # 31个用户
        in_channels=32,  # VA-VAE的32通道
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    
    # 智能加载XL预训练权重
    xl_checkpoint = Path('/kaggle/working/VA-VAE/models/lightningdit-xl-imagenet256-64ep.pt')
    if not xl_checkpoint.exists():
        xl_checkpoint = project_root / 'models' / 'lightningdit-xl-imagenet256-64ep.pt'
    
    if xl_checkpoint.exists():
        print("📥 加载XL预训练权重并智能映射到L模型...")
        checkpoint = torch.load(xl_checkpoint, map_location='cpu')
        
        # 获取XL状态字典
        if 'ema' in checkpoint:
            xl_state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            xl_state_dict = checkpoint['model']
        else:
            xl_state_dict = checkpoint
        
        # 移除module.前缀
        xl_state_dict = {k.replace('module.', ''): v for k, v in xl_state_dict.items()}
        
        # 智能映射到L模型
        l_state_dict = model.state_dict()
        loaded_keys = []
        
        for k, v in l_state_dict.items():
            if k in xl_state_dict:
                # 处理transformer blocks
                if 'blocks.' in k:
                    block_idx = int(k.split('.')[1])
                    if block_idx < 24:  # L模型只有24层
                        if xl_state_dict[k].shape == v.shape:
                            l_state_dict[k] = xl_state_dict[k]
                            loaded_keys.append(k)
                # 处理其他层
                elif xl_state_dict[k].shape == v.shape:
                    l_state_dict[k] = xl_state_dict[k]
                    loaded_keys.append(k)
        
        model.load_state_dict(l_state_dict, strict=False)
        print(f"✅ 成功加载 {len(loaded_keys)}/{len(l_state_dict)} 个权重")
    else:
        print("⚠️ 未找到预训练权重，使用随机初始化")
    
    return model.to(device)

def load_vae(checkpoint_path, device='cuda'):
    """加载微调后的VA-VAE"""
    
    # VA-VAE配置
    vae_config = {
        'embed_dim': 32,
        'double_z': True,
        'z_channels': 32,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.0
    }
    
    # 损失配置（VAE需要）
    loss_config = {
        'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
        'params': {
            'disc_factor': 0.5,
            'perceptual_weight': 1.0,
            'disc_weight': 0.5
        }
    }
    
    # 创建VAE模型
    vae = AutoencoderKL(
        ddconfig=vae_config, 
        lossconfig=loss_config,
        embed_dim=vae_config['embed_dim']
    )
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        print(f"📥 加载VAE权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 移除前缀
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        vae.load_state_dict(state_dict, strict=False)
        print("✅ VAE加载成功")
    else:
        print(f"❌ VAE权重文件不存在: {checkpoint_path}")
    
    vae = vae.to(device).eval()
    return vae

# ==================== 训练配置 ====================
class TrainingConfig:
    """训练配置 - 针对微多普勒任务优化"""
    
    # 模型配置
    model_type = 'LightningDiT-L'
    num_classes = 31
    
    # 训练参数（适配小数据集）
    batch_size = 4  # 每GPU 4张，总共8张
    learning_rate = 2e-5  # 较小学习率
    weight_decay = 0.01  # 适度正则化
    num_epochs = 100  # 更多轮数
    gradient_clip = 1.0
    ema_decay = 0.9999
    
    # 优化器
    warmup_steps = 500
    
    # 采样配置（与官方一致）
    sampling_method = 'euler'
    num_sampling_steps = 250
    cfg_scale = 10.0
    cfg_interval_start = 0.11
    timestep_shift = 0.1
    
    # 验证配置
    val_batch_size = 8
    sample_users = [0, 4, 8, 12, 16, 20, 24, 28]  # 8个用户
    
    # 路径配置
    data_dir = '/kaggle/input/dataset'
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    output_dir = '/kaggle/working/dit_outputs'
    
    # 设备配置
    device = 'cuda'
    num_workers = 2

# ==================== 训练函数 ====================
def train_epoch(model, vae, dataloader, optimizer, transport, config, epoch):
    """训练一个epoch"""
    model.train()
    vae.eval()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(config.device)
        labels = labels.to(config.device)
        
        # VAE编码
        with torch.no_grad():
            # 处理双重z
            posterior = vae.encode(images)
            if hasattr(posterior, 'sample'):
                latents = posterior.sample()
            else:
                latents = posterior
            
            # 归一化潜空间
            latents = latents * 0.13025
        
        # 训练步骤
        model_kwargs = dict(y=labels)
        loss_dict = transport.training_losses(model, latents, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        # 更新EMA
        if hasattr(model, 'module'):
            ema_model = model.module
        else:
            ema_model = model
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate_and_sample(model, vae, val_loader, transport, sampler, config, epoch):
    """验证并生成样本"""
    model.eval()
    vae.eval()
    
    # 验证损失
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            # VAE编码
            posterior = vae.encode(images)
            if hasattr(posterior, 'sample'):
                latents = posterior.sample()
            else:
                latents = posterior
            latents = latents * 0.13025
            
            # 计算损失
            model_kwargs = dict(y=labels)
            loss_dict = transport.training_losses(model, latents, model_kwargs)
            val_loss += loss_dict["loss"].mean().item()
    
    val_loss = val_loss / len(val_loader)
    
    # 生成样本
    print(f"🎨 生成样本...")
    samples = []
    
    with torch.no_grad():
        for user_id in config.sample_users:
            # 生成单个样本
            y = torch.tensor([user_id], device=config.device)
            z = torch.randn(1, 32, 16, 16, device=config.device)
            
            # 采样
            model_kwargs = dict(y=y, cfg_scale=config.cfg_scale)
            sample_fn = sampler.sample_ode(
                sampling_method=config.sampling_method,
                num_steps=config.num_sampling_steps,
                atol=1e-6,
                rtol=1e-3,
            )
            
            sample = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            
            # VAE解码
            sample = sample / 0.13025
            sample = vae.decode(sample)
            
            # 转换为图像
            sample = (sample + 1) / 2
            sample = sample.clamp(0, 1)
            sample = sample.cpu()
            samples.append(sample)
    
    # 保存样本图像
    samples = torch.cat(samples, dim=0)
    save_samples(samples, epoch, config)
    
    return val_loss

def save_samples(samples, epoch, config):
    """保存生成的样本"""
    output_dir = Path(config.output_dir) / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建网格图像
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=4, padding=2)
    
    # 转换为PIL图像
    grid = grid.permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    image = Image.fromarray(grid)
    
    # 保存
    image.save(output_dir / f'epoch_{epoch+1:03d}.png')
    print(f"💾 样本已保存: {output_dir / f'epoch_{epoch+1:03d}.png'}")

# ==================== 主训练循环 ====================
def main():
    """主训练函数"""
    
    config = TrainingConfig()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"🖥️ 检测到 {gpu_count} 个GPU")
    
    # 加载VAE
    vae = load_vae(config.vae_checkpoint, config.device)
    
    # 创建模型
    model = create_dit_model(config.device)
    
    # 使用DataParallel（Kaggle双GPU）
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print("✅ 启用DataParallel多GPU训练")
    
    # 创建Transport和Sampler
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )
    
    sampler = Sampler(transport)
    
    # 创建数据集和加载器
    train_dataset = MicroDopplerDataset(config.data_dir, split='train')
    val_dataset = MicroDopplerDataset(config.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size * gpu_count,  # 总batch size
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    print("\n" + "="*60)
    print("🚀 开始训练")
    print("="*60)
    
    for epoch in range(config.num_epochs):
        # 训练
        train_loss = train_epoch(model, vae, train_loader, optimizer, transport, config, epoch)
        
        # 验证和生成样本
        val_loss = validate_and_sample(model, vae, val_loader, transport, sampler, config, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 打印epoch总结
        print(f"\n📊 Epoch {epoch+1}/{config.num_epochs} 总结:")
        print(f"   训练损失: {train_loss:.4f}")
        print(f"   验证损失: {val_loss:.4f}")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 显存监控
        if torch.cuda.is_available():
            for i in range(gpu_count):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU{i}: {mem_allocated:.1f}GB/{mem_reserved:.1f}GB")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 删除旧的最佳模型
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
                print(f"🗑️ 删除旧模型: {best_checkpoint_path}")
            
            # 保存新的最佳模型
            best_checkpoint_path = f"{config.output_dir}/checkpoints/best_model_epoch{epoch+1}_val{val_loss:.4f}.pt"
            
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__,
            }
            
            torch.save(checkpoint, best_checkpoint_path)
            print(f"💾 保存最佳模型: {best_checkpoint_path}")
        
        print("-"*60)
    
    print("\n✅ 训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型: {best_checkpoint_path}")

if __name__ == "__main__":
    main()
