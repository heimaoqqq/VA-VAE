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
    print(f"🚀 可用GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"   将使用多GPU: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
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
    
    # 多GPU并行化
    if torch.cuda.device_count() > 1:
        print(f"🔧 启用多GPU并行化 ({torch.cuda.device_count()} GPUs)")
        model = torch.nn.DataParallel(model)
    
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
        # 移除'module.'和'_orig_mod.'前缀
        clean_key = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned_state_dict[clean_key] = v
    
    # 如果模型被DataParallel包装，需要添加module.前缀
    if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in cleaned_state_dict.keys()):
        wrapped_state_dict = {f'module.{k}': v for k, v in cleaned_state_dict.items()}
        model.load_state_dict(wrapped_state_dict, strict=False)
    else:
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

def generate_samples_for_user(model, vae, transport, sampler, user_id, num_samples, output_dir, cfg_scale=10.0, seed=42):
    """为指定用户生成条件样本"""
    # 创建采样函数
    sample_fn = sampler.sample_ode(
        sampling_method="dopri5",
        num_steps=300,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
    )
    torch.manual_seed(seed + user_id)  # 每个用户使用不同种子
    np.random.seed(seed + user_id)
    
    # 获取设备信息
    if torch.cuda.device_count() > 1:
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device
        
    user_dir = Path(output_dir) / f"user_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 生成用户 {user_id} 的样本...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"User {user_id}"):
            # 准备条件
            y = torch.tensor([user_id], device=device)
            
            # 创建随机噪声 - 修复：使用正确的latent维度
            z = torch.randn(1, 32, 16, 16, device=device)  # [B, C, H, W]
            
            # CFG采样
            if cfg_scale > 1.0:
                # 构建CFG batch
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31], device=device)  # null class
                y_cfg = torch.cat([y, y_null], 0)
                
                # 创建模型包装函数
                if torch.cuda.device_count() > 1:
                    def model_fn(x, t):
                        return model.module.forward_with_cfg(x, t, y=y_cfg, cfg_scale=cfg_scale)
                else:
                    def model_fn(x, t):
                        return model.forward_with_cfg(x, t, y=y_cfg, cfg_scale=cfg_scale)
                
                samples = sample_fn(z_cfg, model_fn)
                samples = samples[-1]
                samples, _ = samples.chunk(2, dim=0)
            else:
                # 创建标准模型包装函数
                if torch.cuda.device_count() > 1:
                    def model_fn(x, t):
                        return model.module(x, t, y=y)
                else:
                    def model_fn(x, t):
                        return model(x, t, y=y)
                
                samples = sample_fn(z, model_fn)
                samples = samples[-1]
            
            # 解码为图像
            images = vae.decode(samples)
            
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
    
    print(f"✅ 完成用户 {user_id}: {num_samples} 个样本已保存到 {user_dir}")

def main():
    parser = argparse.ArgumentParser(description='生成条件扩散样本')
    parser.add_argument('--checkpoint', required=True, help='模型checkpoint路径')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--vae_checkpoint', help='VAE checkpoint路径')
    parser.add_argument('--output_dir', default='./generated_samples', help='输出目录')
    parser.add_argument('--samples_per_user', type=int, default=200, help='每用户生成样本数')
    parser.add_argument('--cfg_scale', type=float, default=10.0, help='CFG引导强度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_users', type=int, default=31, help='用户数量')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='使用的GPU ID，逗号分隔')
    
    args = parser.parse_args()
    
    # 设置可见GPU
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"🔧 设置可见GPU: {args.gpu_ids}")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model, vae, transport, config, device = load_model_and_config(args.checkpoint, args.config)
    
    # 创建采样器
    sampler = Sampler(transport)
    
    # 创建采样函数
    sample_fn = sampler.sample_ode
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个用户生成样本
    for user_id in range(args.num_users):
        generate_samples_for_user(
            model, vae, transport, sampler, 
            user_id, args.samples_per_user, args.output_dir,
            cfg_scale=args.cfg_scale, seed=args.seed
        )
    
    print(f"\n🎉 生成完成!")
    print(f"总样本数: {args.num_users * args.samples_per_user}")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()
