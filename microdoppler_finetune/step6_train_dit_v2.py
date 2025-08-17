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

# 彻底禁用torch.compile和Dynamo（解决DataParallel FX tracing错误）
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
try:
    import torch._dynamo
    torch._dynamo.disable()
    torch._dynamo.config.suppress_errors = True
except:
    pass
try:
    import torch._inductor
    torch._inductor.config.disable_progress = True
    torch._inductor.config.disable_cpp_codegen = True
except:
    pass

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

# 导入VA-VAE（官方方式）
from tokenizer.autoencoder import AutoencoderKL

# ==================== 预编码功能 ====================
@torch.no_grad()
def create_preencoded_dataset(vae, data_dir, output_dir='/kaggle/working/encoded_dataset', device='cuda'):
    """创建预编码数据集"""
    
    print("🚀 开始创建预编码数据集")
    print("="*50)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集划分信息
    split_file = Path('/kaggle/working/data_split/dataset_split.json')
    if not split_file.exists():
        split_file = Path('data_split/dataset_split.json')
    
    if not split_file.exists():
        print(f"❌ 数据划分文件不存在: {split_file}")
        return False
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 加载用户标签映射
    labels_file = split_file.parent / 'user_labels.json'
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            user_labels = json.load(f)
    else:
        user_labels = {f"ID_{i}": i-1 for i in range(1, 32)}
    
    # 统计总图像数
    total_images = 0
    for split in ['train', 'val']:
        if split in split_data:
            for user_key, image_paths in split_data[split].items():
                total_images += len(image_paths)
    
    print(f"📊 总图像数量: {total_images}")
    
    # 预编码每个数据集
    vae.eval()
    
    for split in ['train', 'val']:
        if split not in split_data:
            continue
        
        print(f"\n📦 编码{split}集...")
        split_data_list = []
        
        # 计算分割的图像数量
        split_total = sum(len(paths) for paths in split_data[split].values())
        progress_bar = tqdm(total=split_total, desc=f'🔄 编码{split}集')
        
        for user_key, image_paths in split_data[split].items():
            user_id = user_labels.get(user_key, 0)
            
            for img_path in image_paths:
                try:
                    # 加载并预处理图像
                    image = Image.open(img_path).convert('RGB')
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
                    image = image * 2.0 - 1.0  # 归一化到[-1, 1]
                    image = image.unsqueeze(0).to(device)  # 添加batch维度
                    
                    # VAE编码
                    posterior = vae.encode(image)
                    if hasattr(posterior, 'sample'):
                        latent = posterior.sample()
                    else:
                        latent = posterior
                    
                    # 归一化潜空间（与训练时保持一致）
                    latent = latent * 0.13025
                    
                    # 添加到列表
                    split_data_list.append({
                        'latent': latent.squeeze(0).cpu(),  # [32, 16, 16]
                        'user_id': user_id,
                        'user_key': user_key,
                        'original_path': img_path
                    })
                    
                    progress_bar.update(1)
                    
                except Exception as e:
                    print(f"⚠️ 编码失败 {img_path}: {e}")
                    continue
        
        progress_bar.close()
        
        # 保存编码结果
        if split_data_list:
            split_file = output_dir / f'{split}_encoded.pt'
            torch.save(split_data_list, split_file)
            print(f"✅ {split}集已保存: {split_file} ({len(split_data_list)} 样本)")
    
    # 保存元数据
    metadata = {
        'vae_scaling_factor': 0.13025,
        'latent_shape': [32, 16, 16],
        'num_users': 31,
        'user_labels': user_labels,
        'created_time': time.time()
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ 元数据已保存: {metadata_file}")
    
    # 计算存储信息
    total_size = 0
    for file in output_dir.glob('*.pt'):
        size = file.stat().st_size
        total_size += size
    
    print(f"\n📊 预编码完成:")
    print(f"   总大小: {total_size/1024/1024:.1f} MB")
    print(f"   存储位置: {output_dir}")
    
    return True

# ==================== 数据集定义 ====================
class MicroDopplerDataset(Dataset):
    """微多普勒数据集 - 支持原图像和预编码数据"""
    
    def __init__(self, data_dir, split='train', split_file=None, use_preencoded=True, encoded_dir='/kaggle/working/encoded_dataset'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_preencoded = use_preencoded
        self.encoded_dir = Path(encoded_dir)
        
        # 检查是否有预编码数据
        encoded_file = self.encoded_dir / f'{split}_encoded.pt'
        
        if use_preencoded and encoded_file.exists():
            print(f"🚀 使用预编码数据: {encoded_file}")
            self._load_preencoded_data(encoded_file)
        else:
            if use_preencoded:
                print(f"⚠️ 预编码文件不存在: {encoded_file}")
                print("   回退到实时VAE编码模式")
            print(f"📥 使用原图像数据")
            self.use_preencoded = False
            self._load_image_data(split_file)
        
        print(f"📊 {split}集: {len(self.data_list)}个样本")
    
    def _load_preencoded_data(self, encoded_file):
        """加载预编码数据"""
        print(f"📥 加载预编码数据: {encoded_file}")
        self.data_list = torch.load(encoded_file, map_location='cpu')
        
        if len(self.data_list) > 0:
            sample = self.data_list[0]
            print(f"   潜在表示形状: {sample['latent'].shape}")
            print(f"   用户数量: {len(set([item['user_id'] for item in self.data_list]))}")
    
    def _load_image_data(self, split_file):
        """加载原图像数据"""
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
            if self.split in split_data and isinstance(split_data[self.split], dict):
                for user_key, image_paths in split_data[self.split].items():
                    user_id = user_labels.get(user_key, 0)
                    for img_path in image_paths:
                        self.data_list.append({
                            'path': img_path,
                            'user_id': user_id
                        })
            else:
                print(f"⚠️ 未找到{self.split}数据，尝试自动扫描...")
                # 回退到自动扫描
                self._auto_scan_data()
        else:
            print(f"⚠️ 数据划分文件不存在: {split_file}")
            print("建议先运行: python step3_prepare_dataset.py")
            self._auto_scan_data()
    
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
        
        if self.use_preencoded:
            # 返回预编码的潜在表示
            latent = item['latent']  # [32, 16, 16]
            user_id = item['user_id']
            return latent, user_id
        else:
            # 加载并预处理原图像
            image = Image.open(item['path']).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            # 归一化到[-1, 1]
            image = image * 2.0 - 1.0
            
            return image, item['user_id']

# ==================== 模型初始化 ====================
def create_dit_model(device='cuda', use_base_model=False):
    """创建并初始化LightningDiT模型
    
    Args:
        device: 设备
        use_base_model: 是否使用Base模型（768维，内存更小，与XL部分兼容）
    """
    
    if use_base_model:
        # 使用Base模型 - 内存更小，可能与XL有部分兼容性
        from models.lightningdit import LightningDiT_B_2
        print("📦 创建LightningDiT-Base模型（768维，12层）")
        model = LightningDiT_B_2(
            input_size=16,
            num_classes=31,
            in_channels=32,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
        )
    else:
        # 创建L模型
        print("📦 创建LightningDiT-L模型（1024维，24层）")
        model = LightningDiT_L_2(
            input_size=16,  # 16×16潜空间
            num_classes=31,  # 31个用户
            in_channels=32,  # VA-VAE的32通道
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
        )
    
    # 尝试智能加载XL预训练权重
    xl_checkpoint = Path('/kaggle/working/VA-VAE/models/lightningdit-xl-imagenet256-64ep.pt')
    if not xl_checkpoint.exists():
        xl_checkpoint = project_root / 'models' / 'lightningdit-xl-imagenet256-64ep.pt'
    
    if xl_checkpoint.exists():
        model_name = "Base" if use_base_model else "L"
        print(f"📥 尝试从XL预训练权重智能初始化{model_name}模型...")
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
        
        # 智能权重转换策略
        target_state_dict = model.state_dict()
        converted_weights = {}
        
        print(f"🔄 开始智能权重转换（XL→{model_name}）...")
        
        for key, target_tensor in target_state_dict.items():
            # 1. 尝试直接匹配
            if key in xl_state_dict and xl_state_dict[key].shape == target_tensor.shape:
                converted_weights[key] = xl_state_dict[key]
                continue
            
            # 2. 对于不同维度的权重，使用智能初始化策略
            if key in xl_state_dict:
                xl_tensor = xl_state_dict[key]
                
                # 位置编码：裁剪或插值
                if 'pos_embed' in key:
                    # XL: [1, 256, 1152], L: [1, 64, 1024], B: [1, 64, 768]
                    if len(xl_tensor.shape) == 3 and len(target_tensor.shape) == 3:
                        # 裁剪序列长度
                        seq_len = min(xl_tensor.shape[1], target_tensor.shape[1])
                        # 裁剪特征维度
                        feat_dim = min(xl_tensor.shape[2], target_tensor.shape[2])
                        converted_weights[key] = xl_tensor[:, :seq_len, :feat_dim].contiguous()
                        print(f"  ✂️ 裁剪位置编码: {key} {xl_tensor.shape}→{converted_weights[key].shape}")
                
                # 线性层权重：裁剪或填充
                elif 'weight' in key and len(xl_tensor.shape) == 2:
                    out_dim = min(xl_tensor.shape[0], target_tensor.shape[0])
                    in_dim = min(xl_tensor.shape[1], target_tensor.shape[1])
                    
                    # 初始化目标张量
                    new_weight = torch.nn.init.xavier_uniform_(target_tensor.clone())
                    # 复制可用部分
                    new_weight[:out_dim, :in_dim] = xl_tensor[:out_dim, :in_dim]
                    converted_weights[key] = new_weight
                    
                # 偏置：裁剪
                elif 'bias' in key and len(xl_tensor.shape) == 1:
                    dim = min(xl_tensor.shape[0], target_tensor.shape[0])
                    converted_weights[key] = xl_tensor[:dim]
                
                # Layer Norm参数：裁剪
                elif ('ln' in key or 'norm' in key) and len(xl_tensor.shape) == 1:
                    dim = min(xl_tensor.shape[0], target_tensor.shape[0])
                    converted_weights[key] = xl_tensor[:dim]
        
        # 加载转换后的权重
        if converted_weights:
            # 只更新成功转换的权重
            for k, v in converted_weights.items():
                if v.shape == target_state_dict[k].shape:
                    target_state_dict[k] = v
            
            model.load_state_dict(target_state_dict, strict=False)
            print(f"✅ 成功智能初始化 {len(converted_weights)}/{len(target_state_dict)} 个权重")
            print(f"   包括：位置编码、线性层、归一化层等关键组件")
        else:
            print("⚠️ 无法从XL权重初始化，使用随机初始化")
            print("   建议：考虑使用Base模型或从头训练")
    else:
        print("⚠️ 未找到XL预训练权重，使用随机初始化")
        print(f"   期望路径: {xl_checkpoint}")
    
    return model.to(device)

def load_vae(checkpoint_path, device='cuda'):
    """使用官方方式加载VA-VAE（与tokenizer.autoencoder兼容）"""
    
    print(f"📥 加载VAE权重: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ VAE权重文件不存在: {checkpoint_path}")
        return None
    
    # 使用官方AutoencoderKL（与tokenizer一致）
    vae = AutoencoderKL(
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=checkpoint_path
    ).to(device).eval()
    
    print("✅ VAE加载成功，已设置为eval模式")
    return vae

# ==================== 训练配置 ====================
class TrainingConfig:
    """训练配置 - 针对微多普勒任务优化"""
    
    # 模型配置
    model_type = 'LightningDiT-L'
    num_classes = 31
    
    # 训练参数（适配小数据集+GPU利用率优化）
    batch_size = 6  # 每GPU 6张，总共12张（从8张提升）
    learning_rate = 3e-5  # 相应提升学习率（batch size增加50%，lr增加50%）
    weight_decay = 0.01  # 适度正则化
    num_epochs = 80   # batch size增大后可以减少轮数
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
def train_epoch(model, vae, dataloader, optimizer, transport, config, epoch, use_preencoded=False):
    """训练一个epoch"""
    model.train()
    if not use_preencoded and vae is not None:
        vae.eval()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data = data.to(config.device)
        labels = labels.to(config.device)
        
        if use_preencoded:
            # 数据已经是潜在表示
            latents = data  # [B, 32, 16, 16]
        else:
            # 实时VAE编码
            with torch.no_grad():
                posterior = vae.encode(data)
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
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'mode': 'preenc' if use_preencoded else 'realtime'
        })
    
    return total_loss / len(dataloader)

def validate_and_sample(model, vae, val_loader, transport, sampler, config, epoch, use_preencoded=False):
    """验证并生成样本"""
    model.eval()
    if vae is not None:
        vae.eval()
    
    # 验证损失
    val_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(config.device)
            labels = labels.to(config.device)
            
            if use_preencoded:
                # 数据已经是潜在表示
                latents = data
            else:
                # 实时VAE编码
                posterior = vae.encode(data)
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
            
            # 处理DataParallel包装的模型
            if hasattr(model, 'module'):
                model_fn = model.module.forward_with_cfg
            else:
                model_fn = model.forward_with_cfg
            
            sample = sample_fn(z, model_fn, **model_kwargs)[-1]
            
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
    
    # 加载VAE（用于预编码和验证）
    vae = load_vae(config.vae_checkpoint, config.device)
    
    # 检查预编码数据
    encoded_dir = Path('/kaggle/working/encoded_dataset')
    use_preencoded = (encoded_dir / 'train_encoded.pt').exists() and (encoded_dir / 'val_encoded.pt').exists()
    
    if not use_preencoded:
        print("⚠️ 预编码数据不存在，开始创建预编码数据集...")
        print("   这是一次性操作，完成后训练将显著加速")
        
        # 创建预编码数据集
        success = create_preencoded_dataset(vae, config.data_dir, encoded_dir, config.device)
        
        if success:
            use_preencoded = True
            print("✅ 预编码数据创建成功！")
        else:
            print("❌ 预编码失败，回退到实时VAE编码模式")
            use_preencoded = False
    
    if use_preencoded:
        print("🚀 使用预编码数据，启用加速训练模式")
        print("   - 无需实时VAE编码")
        print("   - 显著提升训练速度")
    else:
        print("📥 使用实时VAE编码模式")
    
    # 创建模型（默认使用L模型，与step2下载的checkpoint匹配）
    # 只有在明确需要节省内存时才使用Base模型
    use_base = False  # 默认使用L模型
    # 如果需要节省内存，可以手动设置为True使用Base模型
    # use_base = True  # 内存不足时启用
    model = create_dit_model(config.device, use_base_model=use_base)
    
    # 使用DataParallel（Kaggle双GPU）
    if gpu_count > 1:
        # 确保模型在包装前已经在正确设备上
        model = model.to(config.device)
        model = nn.DataParallel(model)
        print("✅ 启用DataParallel多GPU训练")
        print("   注意：已禁用torch.compile以避免FX tracing错误")
    
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
    train_dataset = MicroDopplerDataset(config.data_dir, split='train', use_preencoded=use_preencoded)
    val_dataset = MicroDopplerDataset(config.data_dir, split='val', use_preencoded=use_preencoded)
    
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
        train_loss = train_epoch(model, vae, train_loader, optimizer, transport, config, epoch, use_preencoded)
        
        # 验证和生成样本
        val_loss = validate_and_sample(model, vae, val_loader, transport, sampler, config, epoch, use_preencoded)
        
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
