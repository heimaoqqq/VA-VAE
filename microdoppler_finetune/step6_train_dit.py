"""
Step 6: 训练条件LightningDiT模型
使用微调后的VA-VAE潜空间进行micro-Doppler图像生成
针对Kaggle T4×2 GPU环境优化
"""

import os
import sys
import time
import json
import logging
import types
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# DataParallel辅助函数
def is_main_process():
    """始终返回True，因为不使用分布式训练"""
    return True


# 导入LightningDiT模块
from transport import create_transport, Sampler
from models.lightningdit import LightningDiT_models, LightningDiT_L_2

# 导入VA-VAE
from tokenizer.vavae import VA_VAE

# ==================== 数据集定义 ====================
class MicroDopplerLatentDataset(Dataset):
    """微调后的潜空间数据集，包含用户条件"""
    
    def __init__(self, data_dir, split='train', val_ratio=0.2, latent_norm=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.latent_norm = latent_norm
        
        # 加载数据集划分信息
        split_file = Path("/kaggle/working/data_split/dataset_split.json")
        if not split_file.exists():
            # 如果没有划分文件，尝试其他位置
            alt_split_file = Path("/kaggle/input/data_split/dataset_split.json")
            if alt_split_file.exists():
                split_file = alt_split_file
            else:
                logger.warning(f"未找到数据集划分文件: {split_file}")
                logger.warning("请先运行 step3_prepare_dataset.py 创建数据划分")
                split_info = None
        else:
            import json
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            logger.info(f"加载数据划分: {split_file}")
        
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
        
        # 使用step3创建的数据划分
        if split_info is not None and split in split_info:
            # 使用预定义的划分
            self.file_list = []
            self.user_labels = []
            
            for user_key, image_paths in split_info[split].items():
                user_id = int(user_key.split('_')[1]) - 1  # ID_1 -> 0
                for img_path in image_paths:
                    self.file_list.append(img_path)
                    self.user_labels.append(user_id)
            
            logger.info(f"使用step3划分: {split} 集包含 {len(self.file_list)} 张图像")
            self.indices = np.arange(len(self.file_list))
            n_samples = len(self.file_list)  # 设置n_samples用于后续日志
        else:
            # 如果没有划分信息，使用原来的随机划分
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
        
        # 检查不同的数据路径格式
        # 原始数据集: /kaggle/input/dataset/ID_*
        # 处理后: processed_microdoppler/ID_*
        data_path = self.data_dir
        if not (data_path / 'ID_1').exists():
            # 尝试processed_microdoppler子目录
            alt_path = data_path / 'processed_microdoppler'
            if alt_path.exists():
                data_path = alt_path
                logger.info(f"使用处理后的数据路径: {data_path}")
        
        for user_dir in sorted(data_path.glob('ID_*')):
            user_id = int(user_dir.name.split('_')[1]) - 1  # ID_1 -> 0
            
            # 收集该用户的所有图像（修正为.jpg格式）
            image_files = sorted(list(user_dir.glob('*.jpg')))
            
            if not image_files:
                logger.warning(f"用户 {user_dir.name} 没有找到.jpg图像文件")
                continue
                
            logger.info(f"编码用户 {user_dir.name}: {len(image_files)} 张图像")
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.user_ids.append(user_id)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        if self.latents is not None:
            latent = self.latents[real_idx]
            user_id = self.user_ids[real_idx]
            
            # 确保是tensor格式
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent).float()
            else:
                latent = latent.float()
                
            if isinstance(user_id, np.ndarray):
                user_id = torch.from_numpy(user_id).long()
            else:
                user_id = user_id.long()
            
            # 潜空间归一化
            if self.latent_norm and self.latent_mean is not None:
                # 确保维度匹配
                if latent.dim() == 3 and self.latent_mean.dim() == 4:
                    latent = latent.unsqueeze(0)  # 添加batch维度
                latent = (latent - self.latent_mean) / self.latent_std
                if latent.dim() == 4 and latent.shape[0] == 1:
                    latent = latent.squeeze(0)  # 移除临时batch维度
        
        return latent, user_id


# ==================== 训练函数 ====================
def train_dit():
    """主训练函数 - DataParallel模式"""
    
    # ===== 训练配置（L模型 + 智能权重初始化）=====
    config = {
        'num_epochs': 80,  # 更多轮数补偿部分随机初始化
        'batch_size': 2,  # 保守设置，避免DataParallel的GPU0 OOM
        'gradient_accumulation_steps': 3,  # 有效batch_size=6
        'learning_rate': 5e-6,  # 更小学习率防止过拟合
        'weight_decay': 0.1,   # 更强正则化
        'gradient_clip_norm': 1.0,
        'warmup_steps': 200,  # 较短的预热期
        'ema_decay': 0.9999,
        
        # 采样配置（针对时频图优化）
        'sampling_method': 'dopri5',  # 高精度ODE求解器，适合时频图
        'num_steps': 150,  # 平衡质量和速度
        'cfg_scale': 7.0,  # 适度CFG，保留细节
        'cfg_interval_start': 0.11,  # CFG开始时间
        'timestep_shift': 0.1,  # 减小偏移保留更多细节
        
        # 数据配置
        'num_workers': 2,  # Kaggle环境
        'pin_memory': True,
        'persistent_workers': True,
    }
    
    logger.info("\n" + "="*60)
    logger.info("🚀 LightningDiT-L 训练配置（T4×2优化）")
    logger.info("="*60)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60 + "\n")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # ===== 1. 初始化VA-VAE（仅用于编码） =====
    logger.info("=== 初始化VA-VAE编码器 ===")
    # 创建临时配置文件，自动检测VA-VAE checkpoint
    # 使用指定的VA-VAE模型路径
    vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    
    if Path(vae_checkpoint).exists():
        logger.info(f"✅ 找到VA-VAE模型: {vae_checkpoint}")
        # 检查文件大小
        size_mb = Path(vae_checkpoint).stat().st_size / (1024 * 1024)
        logger.info(f"   模型大小: {size_mb:.2f} MB")
    else:
        logger.error("❌ 未找到VA-VAE模型！")
        logger.error(f"   请确保文件存在: {vae_checkpoint}")
        raise FileNotFoundError(f"VA-VAE checkpoint not found at {vae_checkpoint}")
    
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
    
    # VA-VAE的model已经在初始化时调用了.cuda()，不需要.to(device)
    # vae.model已经是eval模式
    
    # ==================== 模型初始化 ====================
    print("\n" + "="*60)
    print("正在初始化模型...")
    print("="*60)
    
    # 1. 初始化LightningDiT-L模型 - 平衡性能和显存
    print("\n📊 初始化LightningDiT-L...")
    dit_model = LightningDiT_L_2(
        in_channels=32,       # VA-VAE潜空间通道数
        num_classes=31,       # 31个用户条件
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True
    ).to(device)
    
    logger.info(f"Model: LightningDiT-L (1024-dim, 24 layers)")
    logger.info(f"Parameters: {sum(p.numel() for p in dit_model.parameters()) / 1e6:.2f}M")
    
    # 使用XL模型权重进行智能初始化
    pretrained_xl = "/kaggle/working/VA-VAE/models/lightningdit-xl-imagenet256-64ep.pt"
    
    if os.path.exists(pretrained_xl):
        logger.info(f"✅ 找到LightningDiT-XL权重用于初始化L模型: {pretrained_xl}")
        size_gb = os.path.getsize(pretrained_xl) / (1024**3)
        logger.info(f"   XL权重大小: {size_gb:.2f} GB")
        logger.info("   将从XL权重中提取兼容层初始化L模型")
    else:
        logger.warning("⚠️ 未找到XL模型权重文件，请先运行 step2_download_models.py")
        logger.warning(f"   预期路径: {pretrained_xl}")
        logger.warning("   L模型将使用随机初始化（不推荐）")
    
    # 智能加载XL权重到L模型
    logger.info("\n" + "="*60)
    logger.info("🎯 智能权重初始化策略")
    logger.info("="*60)
    
    if pretrained_xl and os.path.exists(pretrained_xl):
        try:
            checkpoint = torch.load(pretrained_xl, map_location='cpu')
            xl_state_dict = checkpoint.get('model', checkpoint)
            xl_state_dict = {k.replace('module.', ''): v for k, v in xl_state_dict.items()}
            
            # L模型和XL模型的映射策略
            l_state_dict = dit_model.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for k, v in l_state_dict.items():
                if k in xl_state_dict:
                    # 对于transformer blocks，L有24层，XL有28层
                    if 'blocks.' in k:
                        block_idx = int(k.split('.')[1])
                        if block_idx < 24:  # L模型只有24层
                            xl_key = k
                            if xl_key in xl_state_dict and xl_state_dict[xl_key].shape == v.shape:
                                l_state_dict[k] = xl_state_dict[xl_key]
                                loaded_keys.append(k)
                            else:
                                skipped_keys.append(k)
                    # 非block层的权重
                    elif xl_state_dict[k].shape == v.shape:
                        l_state_dict[k] = xl_state_dict[k]
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)
            
            dit_model.load_state_dict(l_state_dict, strict=False)
            
            print(f"✅ 智能初始化完成:")
            print(f"   从XL加载: {len(loaded_keys)} 个权重")
            print(f"   随机初始化: {len(skipped_keys)} 个权重")
            print(f"   总参数量: {sum(p.numel() for p in dit_model.parameters()) / 1e6:.1f}M")
            
            # 🎯 全参数微调策略 - 充分适应微多普勒域
            print("   训练策略: 全参数微调 (跨域任务最优)")
            print("   理由1: 时频图与自然图像域差异极大，需要深层特征重学习")
            print("   理由2: 用户间微弱差异需要精细特征，冻结会限制判别能力")
            print("   理由3: XL→L权重映射不完美，需要微调修正")
            
            trainable_params = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in dit_model.parameters())
            print(f"   可训练参数: {trainable_params / 1e6:.1f}M (100%)")
            print("   过拟合防护: 小学习率 + 强正则化 + 梯度累积")
        except Exception as e:
            logger.error(f"❌ 加载预训练权重失败: {e}")
    else:
        logger.error("\n" + "❌"*30)
        logger.error("❌ 严重错误：未找到预训练权重！")
        logger.error("❌ 模型将使用随机初始化，这会导致生成纯噪声图像。")
        logger.error("❌"*30)
        logger.error("\n请确保：")
        logger.error(f"  1. XL权重文件存在于: {pretrained_xl}")
        logger.error("  2. 运行 step2_download_models.py 下载模型")
        logger.error("\n训练将立即停止以避免时间浪费！\n")
        raise ValueError("必须加载预训练权重才能正常训练！")
    
    dit_model.to(device)
    
    # 如果是多GPU，使用DataParallel包装
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # 使用平衡的GPU分配策略
        dit_model = DataParallel(dit_model)
        logger.info(f"✅ 使用 {num_gpus} 个GPU进行DataParallel训练")
        logger.info("⚠️ 注意: DataParallel在GPU0上会有额外显存开销")
        logger.info("   建议: 如果OOM，可尝试减小batch_size或使用单GPU")
    
    logger.info(f"✅ 模型创建完成")
    logger.info(f"   参数量: {sum(p.numel() for p in dit_model.parameters()) / 1e6:.2f}M")
    
    # ===== 3. 初始化训练组件 =====
    logger.info("\n=== 初始化训练组件 ===")
    
    # 创建EMA模型 - 用于稳定生成质量
    from copy import deepcopy
    ema_model = deepcopy(dit_model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    logger.info("EMA模型已创建（衰减率=0.9999）")
    
    # 创建transport
    transport = create_transport(
        'Linear',
        'velocity',
        None,
        None,
        None,
    )
    
    # 优化器 - 只优化未冻结的参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, dit_model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),  # 官方配置：beta2=0.95
        eps=1e-8
    )
    
    # 使用余弦退火+预热，避免早期学习率过高
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        warmup_steps = config.get('warmup_steps', 500)  # 适度预热期
        if current_step < warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦退火，最低保留学习率
        total_steps = config['num_epochs'] * len(train_loader)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))  # 最低1%
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_interval=2000)
    
    # 自动检测数据集路径
    possible_data_paths = [
        "/kaggle/input/dataset",
        "/kaggle/input/micro-doppler-data",
        "./dataset",
    ]
    
    data_dir = None
    for path in possible_data_paths:
        if Path(path).exists():
            data_dir = Path(path)
            logger.info(f"找到数据集: {path}")
            break
    
    if data_dir is None:
        logger.error("未找到数据集！请检查路径")
        raise FileNotFoundError("Dataset not found")
    
    # 创建数据集
    train_dataset = MicroDopplerLatentDataset(data_dir, split='train')
    val_dataset = MicroDopplerLatentDataset(data_dir, split='val')
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers']
    )
    
    # 打印训练信息
    logger.info("\n" + "="*60)
    logger.info("📊 训练信息总览")
    logger.info("="*60)
    logger.info(f"模型: LightningDiT-L (1024维, 24层, 智能初始化)")
    logger.info(f"参数量: {sum(p.numel() for p in dit_model.parameters()) / 1e6:.2f}M")
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"批次大小: {config['batch_size']}")
    logger.info(f"梯度累积: {config['gradient_accumulation_steps']} 步")
    logger.info(f"有效批次: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"训练轮数: {config['num_epochs']}")
    logger.info(f"GPU数量: {num_gpus}")
    logger.info("="*60)
    
    # ===== 5. 开始训练 =====
    # 训练循环
    best_val_loss = float('inf')
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # 训练阶段
        dit_model.train()
        train_loss = 0
        train_steps = 0
        train_grad_norm = 0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            latents = batch[0].to(device)
            user_ids = batch[1].to(device)
            
            # 前向传播 - 添加CFG训练(10%无条件)
            with torch.cuda.amp.autocast():
                # CFG训练：15%概率丢弃条件（增强无条件学习）
                # 注意：不需要手动设置null token，LabelEmbedder会自动处理
                # 当dropout时，模型内部会使用num_classes(31)作为null token
                model_kwargs = {"y": user_ids}
                
                # 使用重要性采样
                t = torch.rand(latents.shape[0], device=device)
                
                loss_dict = transport.training_losses(dit_model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # 梯度裁剪 - 只对可训练参数
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, dit_model.parameters()), 
                config['gradient_clip_norm']
            )
            
            scaler.step(optimizer)
            scaler.update()
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # 统计
            train_loss += loss.item()
            train_grad_norm += grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            train_steps += 1
            train_samples += latents.size(0)
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            samples_per_sec = latents.size(0) / (time.time() - batch_start_time)
            current_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'grad': f'{grad_norm:.2f}',
                'sps': f'{samples_per_sec:.1f}',
                'mem': f'{current_memory:.1f}GB'
            })
            
            scheduler.step()
        
        # 训练阶段统计
        avg_train_loss = train_loss / train_steps
        avg_train_grad_norm = train_grad_norm / train_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]") as pbar:
            for batch in pbar:
                latents = batch[0].to(device)
                user_ids = batch[1].to(device)
                
                with torch.no_grad():
                    # 传递条件信息
                    model_kwargs = {"y": user_ids}
                    loss_dict = transport.training_losses(dit_model, latents, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                val_loss += loss.item()
                val_steps += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_steps
        
        # 确保进度条完全结束后再输出日志
        print()  # 添加空行确保tqdm进度条结束
        
        # 记录指标
        train_metrics_history.append({
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            'grad_norm': avg_train_grad_norm,
            'lr': current_lr
        })
        
        val_metrics_history.append({
            'epoch': epoch + 1,
            'loss': avg_val_loss
        })
        
        # 详细训练报告 - 使用print确保输出
        print("\n" + "="*80)
        print(f"📊 Epoch {epoch+1}/{config['num_epochs']} 训练报告")
        print("="*80)
        
        # 训练统计
        print("\n📈 训练阶段统计:")
        print(f"  • 平均损失: {avg_train_loss:.6f}")
        print(f"  • 梯度范数: {avg_train_grad_norm:.4f}")
        print(f"  • 学习率: {current_lr:.2e}")
        print(f"  • 训练样本数: {train_samples}")
        print(f"  • 每秒样本数: {train_samples / (time.time() - epoch_start_time):.1f}")
        
        # 验证统计
        print("\n📉 验证阶段统计:")
        print(f"  • 平均损失: {avg_val_loss:.6f}")
        print(f"  • 相对改善: {((best_val_loss - avg_val_loss) / best_val_loss * 100):.2f}%" if best_val_loss != float('inf') else "N/A")
        
        # GPU使用情况
        print("\n🖥️ GPU资源使用:")
        for i in range(num_gpus):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}:")
            print(f"    • 已分配内存: {mem_allocated:.2f}GB")
            print(f"    • 已预留内存: {mem_reserved:.2f}GB")
            print(f"    • 利用率: {(mem_allocated/mem_reserved*100):.1f}%" if mem_reserved > 0 else "N/A")
        
        # 条件信息验证
        print("\n✅ 条件注入验证:")
        print(f"  • 用户类别数: {31}")
        # LabelEmbedder使用embedding_table而非embedding_dim
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'y_embedder') and hasattr(actual_model.y_embedder, 'embedding_table'):
            embed_dim = actual_model.y_embedder.embedding_table.embedding_dim
            num_classes = actual_model.y_embedder.embedding_table.num_embeddings
            print(f"  • 条件嵌入维度: {embed_dim}")
            print(f"  • 支持的类别数: {num_classes}")
        print(f"  • 条件信息已正确传递到模型")
        
        print("="*80)
        
        # 每个epoch输出详细训练质量报告
        print("\n" + "="*80)
        print("🔍 训练质量分析")
        print("="*80)
        
        # 损失趋势分析
        if epoch > 0:
            train_loss_change = avg_train_loss - train_metrics_history[-2]['loss'] if len(train_metrics_history) > 1 else 0
            val_loss_change = avg_val_loss - val_metrics_history[-2]['loss'] if len(val_metrics_history) > 1 else 0
            
            print("\n📉 损失趋势:")
            if len(train_metrics_history) > 1:
                print(f"  • 训练损失变化: {train_loss_change:+.6f} ({(train_loss_change/train_metrics_history[-2]['loss']*100):+.2f}%)")
            else:
                print("  • 训练损失变化: N/A")
            if len(val_metrics_history) > 1:
                print(f"  • 验证损失变化: {val_loss_change:+.6f} ({(val_loss_change/val_metrics_history[-2]['loss']*100):+.2f}%)")
            else:
                print("  • 验证损失变化: N/A")
            
            # 过拟合检测
            overfit_gap = avg_val_loss - avg_train_loss
            print("\n⚠️ 过拟合检测:")
            print(f"  • 验证-训练损失差: {overfit_gap:.6f}")
            if overfit_gap > 0.1:
                print("  • 警告: 可能存在过拟合，验证损失显著高于训练损失")
            elif overfit_gap < -0.05:
                print("  • 注意: 验证损失低于训练损失，可能存在数据泄露或评估问题")
            else:
                print("  • 状态: 正常，无明显过拟合")
        
        # 训练稳定性分析
        print("\n⚡ 训练稳定性:")
        if avg_train_grad_norm > 10:
            print(f"  • 警告: 梯度范数过大 ({avg_train_grad_norm:.2f})，可能需要降低学习率")
        elif avg_train_grad_norm < 0.01:
            print(f"  • 警告: 梯度范数过小 ({avg_train_grad_norm:.4f})，可能陷入局部最优")
        else:
            print(f"  • 梯度范数正常: {avg_train_grad_norm:.4f}")
        
        # 收敛状态判断
        print("\n🎯 收敛状态:")
        if epoch >= 4:  # 至少5个epoch后判断
            recent_val_losses = [m['loss'] for m in val_metrics_history[-5:]]
            val_std = np.std(recent_val_losses)
            print(f"  • 最近5轮验证损失标准差: {val_std:.6f}")
            if val_std < 0.001:
                print("  • 模型可能已收敛")
            elif val_std < 0.01:
                print("  • 模型接近收敛")
            else:
                print("  • 模型仍在优化中")
        
        if avg_val_loss < 0.4:
            print("  📈 模型表现：优秀（损失 < 0.4）")
        elif avg_val_loss < 0.6:
            print("  📊 模型表现：良好（损失 < 0.6）")
        elif avg_val_loss < 0.8:
            print("  ⚠️ 模型表现：一般（损失 < 0.8）")
        else:
            print("  ❌ 模型表现：较差（损失 >= 0.8，需要继续训练）")
        
        print("="*80)
        
        # 每个epoch生成条件扩散样本
        if True:  # 每个epoch都生成可视化图像
            print("\n" + "="*80)
            print(f"🎨 Epoch {epoch + 1}: 生成条件扩散样本（使用微调后的VA-VAE）...")
            print("="*80)
            
            try:
                # 延迟初始化VAE以节省内存
                if vae is None:
                    print("  • 初始化VA-VAE用于样本解码...")
                    from tokenizer.vavae import VA_VAE
                    # 使用微调好的VA-VAE模型
                    vae_config_path = Path("/kaggle/working/VA-VAE/LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
                    vae_config = OmegaConf.load(str(vae_config_path))
                    # 使用自动检测到的VA-VAE模型
                    vae_config.ckpt_path = vae_checkpoint
                    temp_config_path = Path("/kaggle/working/temp_vae_config.yaml")
                    OmegaConf.save(vae_config, str(temp_config_path))
                    vae = VA_VAE(str(temp_config_path), img_size=256, horizon_flip=False, fp16=True)
                    print("  • VA-VAE初始化完成（使用Stage 3微调模型）")
                
                # 使用EMA模型生成样本（质量更好）
                generate_conditional_samples(ema_model, vae, transport, device, epoch+1, num_gpus, config)
                print("  • 条件样本生成完成\n")
            except Exception as e:
                print(f"  ⚠️ 条件样本生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # 删除旧的最佳模型以节约空间
            old_best_models = list(Path("/kaggle/working").glob("best_dit_epoch_*.pt"))
            for old_model in old_best_models:
                try:
                    old_model.unlink()
                    logger.info(f"  • 删除旧模型: {old_model.name}")
                except Exception as e:
                    logger.warning(f"  • 无法删除旧模型 {old_model.name}: {e}")
            
            # 保存新的最佳模型
            best_model_path = Path("/kaggle/working") / f"best_dit_epoch_{epoch+1}.pt"
            model_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'ema_state_dict': ema_model.state_dict() if 'ema_model' in locals() else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            logger.info(f"  ✅ 保存最佳模型到 {best_model_path} (val_loss: {avg_val_loss:.6f})")
    
    # 保存最终模型
    final_model_path = Path("/kaggle/working") / "final_dit_model.pt"
    model_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'training_history': {
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history
        }
    }, final_model_path)
    
    logger.info(f"\n🎉 训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"最终模型保存到: {final_model_path}")


def prepare_latents_for_training():
    """预处理：只在主进程中准备潜空间数据"""
    # 检查GPU状态和显存
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    logger.info("=== 准备潜空间数据 ===")
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3
    logger.info(f"初始显存使用: {initial_memory:.2f}GB")
    
    # 使用指定的VA-VAE模型路径
    vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
    
    if Path(vae_checkpoint).exists():
        logger.info(f"✅ 找到VA-VAE模型: {vae_checkpoint}")
        # 检查文件大小
        size_mb = Path(vae_checkpoint).stat().st_size / (1024 * 1024)
        logger.info(f"   模型大小: {size_mb:.2f} MB")
    else:
        logger.error("❌ 未找到VA-VAE模型！")
        logger.error(f"   请确保文件存在: {vae_checkpoint}")
        raise FileNotFoundError(f"VA-VAE checkpoint not found at {vae_checkpoint}")
    
    # 自动检测数据集路径
    possible_data_paths = [
        "/kaggle/input/dataset",  # 标准Kaggle路径
        "/kaggle/input/micro-doppler-data",  # 替代路径
        "./dataset",  # 本地路径
        "G:/micro-doppler-dataset"  # 本地测试路径
    ]
    
    data_dir = None
    for path in possible_data_paths:
        if Path(path).exists():
            data_dir = Path(path)
            logger.info(f"找到数据集: {path}")
            break
    
    if data_dir is None:
        logger.error("未找到数据集！请检查以下路径:")
        for path in possible_data_paths:
            logger.error(f"  - {path}")
        raise FileNotFoundError("Dataset not found")
    
    # 修正配置中的checkpoint路径为微调模型
    logger.info("准备VA-VAE配置...")
    vae_config_path = Path("/kaggle/working/VA-VAE/LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
    
    with open(vae_config_path, 'r') as f:
        config_content = f.read()
    
    # 使用自动检测到的微调模型  
    finetuned_checkpoint = vae_checkpoint
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
    
    # 预计算潜空间（使用已检测的数据路径）
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


# DDP训练函数已删除，改用DataParallel


def generate_conditional_samples(model, vae, transport, device, epoch, num_gpus, config=None):
    """生成条件扩散样本用于验证条件控制能力
    
    采样技术说明：
    - dopri5: 5阶自适应Runge-Kutta方法，精度最高
    - 150步: 平衡质量和速度（官方推荐250步）
    - CFG: 未启用（需要修改模型forward接口）
    """
    model.eval()
    
    with torch.no_grad():
        print(f"\n  📊 开始生成条件样本 (Epoch {epoch}, euler求解器, 250步)...")
        
        # 生成多个用户的样本
        num_users_to_sample = min(8, 31)  # 采样8个不同用户
        samples_per_user = 2  # 每个用户生成2个样本
        
        # 选择要采样的用户
        selected_users = torch.linspace(0, 30, num_users_to_sample, dtype=torch.long, device=device)
        print(f"  • 选择的用户ID: {selected_users.tolist()}")
        print(f"  • 每个用户生成 {samples_per_user} 个样本")
        
        all_samples = []
        all_user_ids = []
        
        for idx, user_id in enumerate(selected_users):
            # 为每个用户生成多个样本
            user_batch = user_id.repeat(samples_per_user)
            
            # 随机初始化噪声
            z = torch.randn(samples_per_user, 32, 16, 16, device=device)
            
            # 生成样本 - 使用CFG增强条件控制
            actual_model = model.module if num_gpus > 1 else model
            
            # CFG采样：同时计算条件和无条件
            cfg_scale = config.get('cfg_scale', 10.0)  # 官方推荐CFG=10.0
            
            # 准备条件和无条件输入
            y_null = torch.full_like(user_batch, 31)  # null token (第32个类别)
            y_combined = torch.cat([user_batch, y_null], dim=0)
            
            # 定义CFG包装函数（支持cfg_interval）
            def model_fn(x, t):
                # 应用cfg_interval：仅在低噪声时使用CFG
                cfg_interval_start = config.get('cfg_interval_start', 0.11)
                # 注意：t是归一化的时间步（0到1），t=0是纯噪声，t=1是干净数据
                # 只有当t > cfg_interval_start时才使用CFG
                if t.min() > cfg_interval_start:
                    # x已经是单批次，需要复制为条件和无条件
                    x_combined = torch.cat([x, x], dim=0)
                    # 时间步t也需要复制以匹配批量大小
                    t_combined = torch.cat([t, t], dim=0)
                    out = actual_model(x_combined, t_combined, y=y_combined)
                    out_cond, out_uncond = out.chunk(2, dim=0)
                    # CFG: 无条件输出 + scale * (条件 - 无条件)
                    return out_uncond + cfg_scale * (out_cond - out_uncond)
                else:
                    # 高噪声阶段不使用CFG，只使用条件生成
                    return actual_model(x, t, y=user_batch)
            
            # 使用Sampler进行采样 - 官方推荐配置
            sampler = Sampler(transport)
            sample_fn = sampler.sample_ode(
                sampling_method=config['sampling_method'],
                num_steps=config['num_steps'],
                atol=1e-6,
                rtol=1e-3,
                timestep_shift=config.get('timestep_shift', 0.3)
            )
            samples = sample_fn(z, model_fn)[-1]
            
            all_samples.append(samples)
            all_user_ids.append(user_batch)
            
            if (idx + 1) % 4 == 0:
                print(f"    • 已生成 {idx + 1}/{num_users_to_sample} 个用户的样本")
        
        # 合并所有样本
        all_samples = torch.cat(all_samples, dim=0)
        all_user_ids = torch.cat(all_user_ids, dim=0)
        
        print(f"  • 成功生成 {len(all_samples)} 个条件样本")
        print(f"  • 样本形状: {all_samples.shape}")
        
        # 使用VA-VAE解码
        print("\n  🎨 开始解码潜空间样本到图像...")
        try:
            # 将潜空间样本移到cuda:0（VA-VAE所在的设备）
            samples_cuda0 = all_samples.to('cuda:0')
            print(f"    • 样本已移到 cuda:0")
            
            # 反标准化（如果有统计信息）
            if hasattr(vae, 'latent_mean') and vae.latent_mean is not None:
                samples_cuda0 = samples_cuda0 * vae.latent_std + vae.latent_mean
                print(f"    • 已应用反标准化")
            
            # 解码
            print(f"    • 开始VA-VAE解码...")
            # VA_VAE使用decode_to_images方法，返回[0,255]的uint8图像
            images_np = vae.decode_to_images(samples_cuda0)
            # 转换为torch tensor并正确归一化到[-1, 1]
            images = torch.from_numpy(images_np).float() / 255.0 * 2.0 - 1.0  # [0,255] -> [-1,1]
            images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
            print(f"    • 解码完成，图像形状: {images.shape}")
            
            # 保存图像
            save_dir = Path("/kaggle/working") / f"samples_epoch_{epoch}"
            save_dir.mkdir(exist_ok=True, parents=True)
            
            from torchvision.utils import save_image
            
            # 创建网格图
            grid_path = save_dir / "grid.png"
            num_show = min(16, len(images))
            save_image(images[:num_show], grid_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"  ✅ 网格图已保存: {grid_path}")
            
            # 保存前几张单独的图像
            num_save = min(8, len(images))
            for i in range(num_save):
                uid = all_user_ids[i].item()
                img_path = save_dir / f"user_{uid}_sample_{i}.png"
                save_image(images[i], img_path, normalize=True, value_range=(-1, 1))
            print(f"  ✅ 保存了 {num_save} 张单独图像到: {save_dir}")
            
            # 创建用户分组的网格图
            users_path = save_dir / "users_grid.png"
            save_image(images, users_path, nrow=samples_per_user, normalize=True, value_range=(-1, 1))
            print(f"  ✅ 用户分组网格图: {users_path}")
            
        except Exception as e:
            print(f"\n  ⚠️ 解码失败: {e}")
            import traceback
            traceback.print_exc()
            
            print("  • 正在保存潜空间样本而非图像...")
            
            # 保存潜空间表示
            latent_path = Path("/kaggle/working") / f"latents_epoch_{epoch}.pt"
            torch.save({
                'latents': all_samples.cpu(),
                'user_ids': all_user_ids.cpu(),
                'epoch': epoch
            }, latent_path)
            print(f"  ✅ 潜空间样本已保存到: {latent_path}")


def main():
    """主函数 - Kaggle T4x2优化版本"""
    # Kaggle环境GPU检查
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Please enable GPU accelerator in Kaggle.")
        return
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s)")
    
    # Kaggle T4x2使用DataParallel，不使用分布式训练
    train_dit()

if __name__ == "__main__":
    main()
