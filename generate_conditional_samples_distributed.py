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

def load_weights_with_shape_check(model, checkpoint, rank=0):
    """使用形状检查加载权重（完全按照官方实现）"""
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

def load_model_and_config(checkpoint_path, config_path, local_rank):
    """加载模型和配置（按照官方方式）"""
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
        class_dropout_prob=config['model'].get('class_dropout_prob', 0.1),
        use_qknorm=config['model']['use_qknorm'],
        use_swiglu=config['model'].get('use_swiglu', False),
        use_rope=config['model'].get('use_rope', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False),
        wo_shift=config['model'].get('wo_shift', False),
        in_channels=config['model'].get('in_chans', 4),
        use_checkpoint=config['model'].get('use_checkpoint', False),
    ).to(device)
    
    # 按照官方方式加载权重
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"📦 从checkpoint加载权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # 处理权重键名（按照官方方式）
        if 'ema' in checkpoint:
            # 优先使用EMA权重（推理时更稳定）
            checkpoint_weights = {'model': checkpoint['ema']}
            if local_rank == 0:
                print("📦 使用EMA权重进行推理")
        elif 'model' in checkpoint:
            checkpoint_weights = checkpoint
            if local_rank == 0:
                print("📦 使用模型权重进行推理")
        else:
            checkpoint_weights = {'model': checkpoint}
            if local_rank == 0:
                print("📦 使用直接权重进行推理")
        
        # 清理键名（remove the prefix 'module.'）
        checkpoint_weights['model'] = {k.replace('module.', ''): v for k, v in checkpoint_weights['model'].items()}
        
        # 使用官方权重加载函数
        model = load_weights_with_shape_check(model, checkpoint_weights, rank=local_rank)
        
        if local_rank == 0:
            print("✅ 权重加载完成")
    else:
        if local_rank == 0:
            print(f"⚠️ 警告: 找不到checkpoint文件 {checkpoint_path}")
            print("⚠️ 使用未训练的随机权重，生成结果将是噪声！")
    
    model.eval()
    
    # 创建VAE（完全按照官方train_dit_s_official.py方式）
    vae = None
    try:
        # 添加LightningDiT路径到系统路径
        import sys
        lightningdit_path = os.path.join(os.getcwd(), 'LightningDiT')
        if lightningdit_path not in sys.path:
            sys.path.insert(0, lightningdit_path)
        
        from tokenizer.vavae import VA_VAE
        import tempfile
        
        # 使用训练好的VAE模型路径
        custom_vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
        
        # 创建与train_dit_s_official.py完全一致的配置
        vae_config = {
            'ckpt_path': custom_vae_checkpoint,
            'model': {
                'base_learning_rate': 2.0e-05,
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'monitor': 'val/rec_loss',
                    'embed_dim': 32,
                    'use_vf': 'dinov2',
                    'reverse_proj': True,
                    'ddconfig': {
                        'double_z': True, 'z_channels': 32, 'resolution': 256,
                        'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2,
                        'attn_resolutions': [16], 'dropout': 0.0
                    },
                    'lossconfig': {
                        'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                        'params': {
                            'disc_start': 1, 'disc_num_layers': 3, 'disc_weight': 0.5,
                            'disc_factor': 1.0, 'disc_in_channels': 3, 'disc_conditional': False,
                            'disc_loss': 'hinge', 'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,
                            'kl_weight': 1e-6, 'logvar_init': 0.0, 'use_actnorm': False,
                            'pp_style': False, 'vf_weight': 0.1, 'adaptive_vf': False,
                            'distmat_weight': 1.0, 'cos_weight': 1.0,
                            'distmat_margin': 0.25, 'cos_margin': 0.5
                        }
                    }
                }
            }
        }
        
        # 写入临时配置文件
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(vae_config, f, default_flow_style=False)
        os.close(temp_config_fd)
        
        try:
            # 使用官方VA_VAE类加载
            vae = VA_VAE(temp_config_path)
            vae = vae.to(device)
            vae.eval()
            if local_rank == 0:
                print(f"✅ VAE加载完成: {custom_vae_checkpoint}")
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
            
    except Exception as e:
        if local_rank == 0:
            print(f"⚠️ VAE加载失败: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ 尝试使用简化VAE作为备用")
        # 备用方案
        try:
            vae = SimplifiedVAVAE(config['vae']['model_name']).to(device)
            vae.eval()
            if local_rank == 0:
                print(f"✅ 备用VAE加载完成: {config['vae']['model_name']}")
        except Exception as e2:
            if local_rank == 0:
                print(f"⚠️ 备用VAE也加载失败: {e2}")
            vae = None
    
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
    # 创建采样器和采样函数 - 完全按照配置文件
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="dopri5",      # 高精度ODE求解器
        num_steps=300,                 # 采样步数（与配置文件一致）
        atol=1e-6,                     # 绝对误差容限
        rtol=1e-3,                     # 相对误差容限
        reverse=False,                 # 不反向采样
        timestep_shift=0.1             # 时间步偏移（官方设置）
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
            
            # CFG采样 - 完全按照官方train_dit_s_official.py实现
            if cfg_scale > 1.0:
                # 构建CFG batch
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31] * current_batch_size, device=device)  # null class
                y_cfg = torch.cat([y, y_null], 0)
                
                # 使用官方CFG配置
                cfg_interval_start = 0.11  # 与官方保持一致
                model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                
                # 使用CFG前向传播（与官方完全一致）
                if hasattr(model, 'forward_with_cfg'):
                    samples = sample_fn(z_cfg, model.forward_with_cfg, **model_kwargs)
                else:
                    # 如果模型没有forward_with_cfg方法，使用手动CFG
                    def model_fn_cfg(x, t, **kwargs):
                        pred = model(x, t, **kwargs)
                        pred_cond, pred_uncond = pred.chunk(2, dim=0)
                        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    samples = sample_fn(z_cfg, model_fn_cfg, **model_kwargs)
                
                samples = samples[-1]  # 获取最终时间步的样本
                samples, _ = samples.chunk(2, dim=0)  # 去掉null class样本
            else:
                # 标准采样
                samples = sample_fn(z, model, **dict(y=y))
                samples = samples[-1]
            
            # 检查latent范围 (仅rank 0输出)
            if rank == 0 and batch_idx == 0:
                print(f"🔍 生成的Latent范围: [{samples.min():.3f}, {samples.max():.3f}], 标准差: {samples.std():.3f}")
            
            # 🔴 关键步骤：反归一化！（完全按照官方train_dit_s_official.py实现）
            # 官方公式：samples_denorm = (samples * std) / latent_multiplier + mean
            # 因为训练时做了：feature = (feature - mean) / std * latent_multiplier
            latent_stats_path = '/kaggle/working/VA-VAE/latents_safetensors/train/latent_stats.pt'
            if os.path.exists(latent_stats_path):
                stats = torch.load(latent_stats_path, map_location=device)
                mean = stats['mean'].to(device)  # [32, 1, 1]
                std = stats['std'].to(device)     # [32, 1, 1]
                latent_multiplier = 1.0  # VA-VAE使用1.0，不是0.18215
                
                # 官方反归一化公式（与train_dit_s_official.py第549行完全一致）
                samples_denorm = (samples * std) / latent_multiplier + mean
                
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
                        torch.save({'mean': mean, 'std': std}, './latents_safetensors/train/latents_stats.pt')
                        print(f"✅ 从数据集计算统计完成，已保存到 ./latents_safetensors/train/latents_stats.pt")
                        
                        # 反归一化（与官方公式保持一致）
                        latent_multiplier = 1.0  # VA-VAE使用1.0
                        samples_denorm = (samples * std) / latent_multiplier + mean
                        
                        if rank == 0 and batch_idx == 0:
                            print(f"🔍 反归一化后范围: [{samples_denorm.min():.3f}, {samples_denorm.max():.3f}], 标准差: {samples_denorm.std():.3f}")
                            print(f"📊 使用latent统计信息: mean shape={mean.shape}, std shape={std.shape}")
                            print(f"✅ 反归一化公式与官方train_dit_s_official.py完全一致")
                    except Exception as e:
                        if rank == 0:
                            print(f"❌ 无法计算统计信息: {e}")
                            print(f"💡 请先运行: python prepare_latent_stats.py --data_dir ./latents_safetensors/train")
                        samples_denorm = samples
                else:
                    samples_denorm = samples
            
            # VAE解码（使用反归一化后的latent）
            if vae is not None:
                try:
                    # 确保VAE在正确设备上
                    if hasattr(vae, 'to'):
                        vae = vae.to(device)
                    
                    # 使用VA-VAE解码latent为图像（与train_dit_s_official.py第568行一致）
                    decoded_images = vae.decode_to_images(samples_denorm)
                    
                    # 按照官方方式保存单个图像文件
                    from PIL import Image
                    for i, image in enumerate(decoded_images):
                        save_path = user_dir / f"sample_{sample_idx + i:06d}.png"
                        Image.fromarray(image).save(save_path)
                except Exception as e:
                    if rank == 0:
                        print(f"❌ VAE解码失败: {e}")
                sample_idx += 1
    
    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"✅ 完成用户 {user_id}: {num_samples} 个样本已保存到 {user_dir}")

def main():
    parser = argparse.ArgumentParser(description='Distributed conditional sample generation')
    parser.add_argument('--checkpoint', type=str, 
                       default='/kaggle/input/50000-pt/0050000.pt', 
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, 
                       default='configs/dit_s_microdoppler.yaml', 
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help='Output directory')
    parser.add_argument('--num_samples', '--samples_per_user', type=int, default=200, help='Samples per user')
    parser.add_argument('--cfg_scale', type=float, default=10.0, help='CFG scale（与配置文件一致）')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"🚀 分布式推理: {world_size} GPUs")
        print(f"📂 输出目录: {args.output_dir}")
    
    # 加载模型和配置（使用训练好的模型）
    if rank == 0:
        print(f"🚀 使用训练好的DiT模型: {args.checkpoint}")
        print(f"📋 使用配置文件: {args.config}")
    
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
    for user_id in range(31):
        generate_samples_for_user_distributed(
            model, vae, transport, sampler, 
            user_id, args.num_samples, args.output_dir,
            cfg_scale=args.cfg_scale, seed=args.seed, batch_size=args.batch_size,
            rank=rank, world_size=world_size
        )
    
    if rank == 0:
        print(f"🎯 分布式条件样本生成完成！")
        print(f"✅ 所有用户的样本已生成到: {args.output_dir}")
        print(f"📈 每用户生成了 {args.num_samples} 个样本")
    
    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
