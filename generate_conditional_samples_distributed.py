"""
分布式条件扩散样本生成脚本 (torchrun版本)
为每个用户生成200张条件样本，用于评估用户区分度
"""
import torch
import torch.distributed as dist
import sys
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from simplified_vavae import SimplifiedVAVAE

def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def load_model_and_config(checkpoint_path, config_path, local_rank):
    """加载模型和配置"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    device = torch.device(f'cuda:{local_rank}')
    
    # 创建DiT模型
    latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']
    model = LightningDiT_models[config['model']['model_type']](
        input_size=latent_size,
        num_classes=config['data']['num_classes'],
        class_dropout_prob=config['model']['class_dropout_prob'],
        use_qknorm=config['model']['use_qknorm'],
        use_swiglu=config['model'].get('use_swiglu', False),
        use_rope=config['model'].get('use_rope', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False),
        wo_shift=config['model'].get('wo_shift', False),
        in_channels=config['model']['in_chans'],
        use_checkpoint=config['model'].get('use_checkpoint', False),
    ).to(device)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理EMA权重
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
        print("📦 使用EMA权重")
    else:
        state_dict = checkpoint.get('model', checkpoint)
        print("📦 使用标准权重")
    
    # 清理键名
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned_state_dict[clean_key] = v
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    
    # 创建VAE
    vae = SimplifiedVAVAE(config['vae']['model_name']).to(device)
    vae.eval()
    
    # 创建transport
    transport = create_transport(
        config['transport']['path_type'],
        config['transport']['prediction'],
        config['transport']['loss_weight'],
        config['transport']['train_eps'],
        config['transport']['sample_eps'],
        use_cosine_loss=config['transport'].get('use_cosine_loss', False),
        use_lognorm=config['transport'].get('use_lognorm', False),
        partitial_train=config['transport'].get('partitial_train', None),
        partial_ratio=config['transport'].get('partial_ratio', 1.0),
        shift_lg=config['transport'].get('shift_lg', False),
    )
    
    return model, vae, transport, config, device

def generate_samples_for_user_distributed(model, vae, transport, sampler, user_id, num_samples, 
                                        output_dir, cfg_scale=10.0, seed=42, batch_size=16, 
                                        rank=0, world_size=1):
    """分布式生成指定用户的条件样本"""
    # 创建采样函数（与官方train_dit_s_official.py保持一致）
    sample_fn = sampler.sample_ode(
        sampling_method="dopri5",  # 高精度ODE求解器
        num_steps=300,             # 使用300步以获得高质量
        atol=1e-6,                 # 更严格的误差容限
        rtol=1e-3,                 
        reverse=False,
        timestep_shift=0.1,        # 与官方配置一致
    )
    
    # 计算每个进程处理的样本数
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    if rank == world_size - 1:  # 最后一个进程处理剩余样本
        end_idx = num_samples
    
    actual_samples = end_idx - start_idx
    
    torch.manual_seed(seed + user_id + rank * 1000)
    np.random.seed(seed + user_id + rank * 1000)
    
    device = next(model.parameters()).device
    user_dir = Path(output_dir) / f"user_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print(f"🎨 生成用户 {user_id} 的样本 (总计{num_samples}张, {world_size}卡并行)...")
    
    with torch.no_grad():
        num_batches = (actual_samples + batch_size - 1) // batch_size
        sample_idx = start_idx
        
        for batch_idx in tqdm(range(num_batches), desc=f"Rank{rank}-User{user_id}", 
                             disable=(rank != 0)):
            # 计算当前batch大小
            current_batch_size = min(batch_size, end_idx - sample_idx)
            if current_batch_size <= 0:
                break
            
            # 准备条件
            y = torch.tensor([user_id] * current_batch_size, device=device)
            
            # 创建随机噪声
            z = torch.randn(current_batch_size, 32, 16, 16, device=device)
            
            # CFG采样
            if cfg_scale > 1.0:
                # 构建CFG batch
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31] * current_batch_size, device=device)
                y_cfg = torch.cat([y, y_null], 0)
                
                # 手动实现CFG（与官方train_dit_s_official.py一致）
                def model_fn(x, t):
                    # 使用模型的forward_with_cfg方法（如果存在）
                    if hasattr(model, 'forward_with_cfg'):
                        return model.forward_with_cfg(x, t, y=y_cfg, cfg_scale=cfg_scale)
                    else:
                        # 手动实现CFG
                        pred = model(x, t, y=y_cfg)
                        pred_cond, pred_uncond = pred.chunk(2, dim=0)
                        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                
                samples = sample_fn(z_cfg, model_fn)
                samples = samples[-1]
                samples, _ = samples.chunk(2, dim=0)
            else:
                def model_fn(x, t):
                    return model(x, t, y=y)
                
                samples = sample_fn(z, model_fn)
                samples = samples[-1]
            
            # 检查latent范围 (仅rank 0输出)
            if rank == 0 and batch_idx == 0:
                print(f"🔍 生成的Latent范围: [{samples.min():.3f}, {samples.max():.3f}], 标准差: {samples.std():.3f}")
            
            # 🔴 关键步骤：反归一化！
            # 因为训练配置中 latent_norm: true
            # 训练时做了: feature = (feature - mean) / std * 1.0
            # 所以推理时需要: samples = samples * std + mean
            # 调整为用户实际的latent目录路径
            latent_stats_path = './latents_safetensors/train/latent_stats.pt'
            if os.path.exists(latent_stats_path):
                stats = torch.load(latent_stats_path, map_location=device)
                mean = stats['mean'].to(device)  # [32, 1, 1]
                std = stats['std'].to(device)     # [32, 1, 1]
                
                # 反归一化公式（因为latent_multiplier=1.0）
                samples_denorm = samples * std + mean
                
                if rank == 0 and batch_idx == 0:
                    print(f"🔍 反归一化后范围: [{samples_denorm.min():.3f}, {samples_denorm.max():.3f}], 标准差: {samples_denorm.std():.3f}")
                    print(f"📊 使用统计信息: mean shape={mean.shape}, std shape={std.shape}")
            else:
                if rank == 0:
                    print(f"⚠️ 警告: 找不到latent统计文件 {latent_stats_path}")
                    print(f"⚠️ 跳过反归一化步骤，可能导致生成噪声！")
                    print(f"💡 尝试从数据集直接计算统计信息...")
                    # 如果统计文件不存在，尝试从数据集直接计算
                    try:
                        from LightningDiT.datasets.img_latent_dataset import ImgLatentDataset
                        train_dataset = ImgLatentDataset('./latents_safetensors/train', latent_norm=True)
                        stats = train_dataset.compute_latent_stats()
                        mean = stats['mean'].to(device)  # [1, 32, 1, 1]
                        std = stats['std'].to(device)    # [1, 32, 1, 1]
                        # 去掉batch维度
                        mean = mean.squeeze(0)  # [32, 1, 1]
                        std = std.squeeze(0)    # [32, 1, 1]
                        
                        # 保存统计文件供下次使用
                        os.makedirs('./latents_safetensors/train', exist_ok=True)
                        torch.save({'mean': mean, 'std': std}, './latents_safetensors/train/latent_stats.pt')
                        print(f"✅ 从数据集计算统计完成，已保存到 ./latents_safetensors/train/latent_stats.pt")
                        
                        # 反归一化
                        samples_denorm = samples * std + mean
                        
                        if rank == 0 and batch_idx == 0:
                            print(f"🔍 反归一化后范围: [{samples_denorm.min():.3f}, {samples_denorm.max():.3f}], 标准差: {samples_denorm.std():.3f}")
                            print(f"📊 使用统计信息: mean shape={mean.shape}, std shape={std.shape}")
                    except Exception as e:
                        if rank == 0:
                            print(f"❌ 无法计算统计信息: {e}")
                            print(f"💡 请先运行: python prepare_latent_stats.py --data_dir ./latents_safetensors/train")
                        samples_denorm = samples
                else:
                    samples_denorm = samples
            
            # VAE解码（使用反归一化后的latent）
            images = vae.decode(samples_denorm)
            
            # 检查解码后范围 (仅rank 0输出)
            if rank == 0 and batch_idx == 0:
                print(f"🔍 解码后图像范围: [{images.min():.3f}, {images.max():.3f}], 标准差: {images.std():.3f}")
            
            # 后处理：[0,1] -> [0,255]
            images = torch.clamp(images, 0, 1)
            images = (images * 255).round().byte()
            
            # 保存每个图像
            for i in range(current_batch_size):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                if image.shape[2] == 1:  # 灰度图
                    image = image.squeeze(2)
                    pil_image = Image.fromarray(image, mode='L')
                elif image.shape[2] == 3:  # RGB图
                    pil_image = Image.fromarray(image, mode='RGB')
                else:
                    if rank == 0:
                        print(f"⚠️ 未知图像格式: {image.shape}")
                    continue
                
                filename = user_dir / f"sample_{sample_idx:03d}.png"
                pil_image.save(filename)
                sample_idx += 1
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"✅ 完成用户 {user_id}: {num_samples} 个样本已保存到 {user_dir}")

def main():
    parser = argparse.ArgumentParser(description='分布式生成条件扩散样本')
    parser.add_argument('--checkpoint', required=True, help='模型checkpoint路径')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--output_dir', default='./generated_samples', help='输出目录')
    parser.add_argument('--samples_per_user', type=int, default=200, help='每用户生成样本数')
    parser.add_argument('--cfg_scale', type=float, default=10.0, help='CFG引导强度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_users', type=int, default=31, help='用户数量')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"🚀 分布式推理: {world_size} GPUs")
        print(f"📂 输出目录: {args.output_dir}")
    
    # 加载模型
    model, vae, transport, config, device = load_model_and_config(
        args.checkpoint, args.config, local_rank
    )
    
    # 创建采样器
    sampler = Sampler(transport)
    
    # 创建输出目录
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()  # 等待目录创建完成
    
    # 为每个用户生成样本
    for user_id in range(args.num_users):
        generate_samples_for_user_distributed(
            model, vae, transport, sampler, 
            user_id, args.samples_per_user, args.output_dir,
            cfg_scale=args.cfg_scale, seed=args.seed, batch_size=args.batch_size,
            rank=rank, world_size=world_size
        )
    
    if rank == 0:
        print(f"\n🎉 分布式生成完成!")
        print(f"总样本数: {args.num_users * args.samples_per_user}")
        print(f"输出目录: {args.output_dir}")
    
    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
