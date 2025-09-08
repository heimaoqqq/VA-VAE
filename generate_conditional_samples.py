"""
条件扩散样本生成脚本
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

def load_model_and_config(checkpoint_path, config_path):
    """加载模型和配置"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        model.load_state_dict(checkpoint['ema'])
        print(f"Loaded EMA model from {checkpoint_path}")
    elif 'model' in checkpoint:
        # 移除DDP前缀
        state_dict = {}
        for k, v in checkpoint['model'].items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict)
        print(f"Loaded regular model from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded direct state dict from {checkpoint_path}")
    
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

def generate_samples_for_user(model, vae, transport, config, device, user_id, num_samples, output_dir):
    """为指定用户生成样本"""
    user_dir = Path(output_dir) / f"user_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # 采样配置
    sample_config = config.get('sample', {})
    num_steps = sample_config.get('num_sampling_steps', 300)
    cfg_scale = sample_config.get('cfg_scale', 10.0)
    cfg_interval_start = sample_config.get('cfg_interval_start', 0.11)
    timestep_shift = sample_config.get('timestep_shift', 0.1)
    sampling_method = sample_config.get('sampling_method', 'dopri5')
    atol = sample_config.get('atol', 1e-6)
    rtol = sample_config.get('rtol', 1e-3)
    using_cfg = cfg_scale > 1.0
    
    # 创建采样器
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=num_steps,
        atol=atol,
        rtol=rtol,
        reverse=sample_config.get('reverse', False),
        timestep_shift=timestep_shift,
    )
    
    print(f"Generating {num_samples} samples for user {user_id}...")
    print(f"Using CFG={cfg_scale}, steps={num_steps}, method={sampling_method}")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"User {user_id}"):
            # 设置固定随机种子确保可重复性
            torch.manual_seed(user_id * 10000 + i)
            np.random.seed(user_id * 10000 + i)
            
            # 生成latent噪声
            z = torch.randn(1, 32, 16, 16, device=device)
            y = torch.tensor([user_id], dtype=torch.long, device=device)
            
            # CFG采样
            if using_cfg:
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31], device=device)  # null class
                y_cfg = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                
                samples = sample_fn(z_cfg, model.forward_with_cfg, **model_kwargs)
                samples = samples[-1]
                samples, _ = samples.chunk(2, dim=0)
            else:
                model_kwargs = dict(y=y)
                samples = sample_fn(z, model, **model_kwargs)
                samples = samples[-1]
            
            # 解码为图像
            images = vae.decode_latents(samples)
            
            # 后处理：[0,1] -> [0,255]
            images = torch.clamp(images, 0, 1)
            images = (images * 255).round().byte()
            
            # 转换为PIL图像并保存
            image = images[0].permute(1, 2, 0).cpu().numpy()
            if image.shape[2] == 1:  # 灰度图
                image = image.squeeze(2)
                pil_image = Image.fromarray(image, mode='L')
            else:  # RGB图
                pil_image = Image.fromarray(image, mode='RGB')
            
            # 保存图像
            filename = user_dir / f"sample_{i:03d}.png"
            pil_image.save(filename)
    
    print(f"Completed user {user_id}: {num_samples} samples saved to {user_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate conditional samples for evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', help='Output directory')
    parser.add_argument('--samples_per_user', type=int, default=200, help='Number of samples per user')
    parser.add_argument('--start_user', type=int, default=0, help='Start user ID')
    parser.add_argument('--end_user', type=int, default=31, help='End user ID (exclusive)')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model, vae, transport, config, device = load_model_and_config(args.checkpoint, args.config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个用户生成样本
    for user_id in range(args.start_user, args.end_user):
        generate_samples_for_user(
            model, vae, transport, config, device,
            user_id, args.samples_per_user, args.output_dir
        )
    
    print(f"\nGeneration completed!")
    print(f"Total samples: {(args.end_user - args.start_user) * args.samples_per_user}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
