#!/usr/bin/env python3
"""
Step 7: 优化的DiT采样配置和生成
专门针对微多普勒时频图像特征优化
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 环境设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))

# 内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

try:
    import torch._dynamo
    torch._dynamo.disable()
    torch._dynamo.config.suppress_errors = True
except:
    pass

# 导入必要模块
from transport import create_transport, Sampler
from models.lightningdit import LightningDiT_L_2, LightningDiT_B_2
from tokenizer.autoencoder import AutoencoderKL

class OptimizedSamplingConfig:
    """优化的采样配置 - 专门针对微多普勒数据"""
    
    # 模型配置
    model_type = 'LightningDiT-L'
    num_classes = 31
    
    # 🔧 优化的采样参数（基于微多普勒特征）
    sampling_method = 'dopri5'     # 高精度自适应ODE求解器，适合时频图像
    num_sampling_steps = 150        # 平衡质量和速度
    cfg_scale = 7.0                 # 适度CFG，避免过度引导丢失微动细节
    cfg_interval_start = 0.11       # 官方设置
    timestep_shift = 0.1            # 较小偏移，保留更多早期去噪过程
    
    # 高级采样参数
    atol = 1e-6                     # 绝对误差容限
    rtol = 1e-3                     # 相对误差容限
    
    # 批量生成配置
    batch_size = 4                  # 每批生成4个样本
    num_samples_per_user = 5        # 每个用户生成5个样本
    
    # 路径配置
    model_checkpoint = '/kaggle/working/dit_outputs/checkpoints/best_model.pt'
    vae_checkpoint = '/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt'
    output_dir = '/kaggle/working/optimized_samples'
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models(config):
    """加载DiT和VAE模型"""
    
    print("📦 加载模型...")
    
    # 加载DiT模型
    if config.model_type == 'LightningDiT-L':
        model = LightningDiT_L_2(
            input_size=16,
            num_classes=config.num_classes,
            in_channels=32,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
        )
    else:
        model = LightningDiT_B_2(
            input_size=16,
            num_classes=config.num_classes,
            in_channels=32,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
        )
    
    # 加载训练好的权重
    if Path(config.model_checkpoint).exists():
        print(f"✅ 加载DiT权重: {config.model_checkpoint}")
        checkpoint = torch.load(config.model_checkpoint, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 处理DataParallel的module前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ 未找到训练权重，使用随机初始化")
    
    model = model.to(config.device).eval()
    
    # 加载VAE模型
    print(f"✅ 加载VAE模型: {config.vae_checkpoint}")
    vae = AutoencoderKL(
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=config.vae_checkpoint
    ).to(config.device).eval()
    
    return model, vae

def generate_samples_optimized(model, vae, config):
    """使用优化配置生成样本"""
    
    print("\n" + "="*60)
    print("🎨 开始优化采样生成")
    print(f"   采样方法: {config.sampling_method}")
    print(f"   采样步数: {config.num_sampling_steps}")
    print(f"   CFG强度: {config.cfg_scale}")
    print(f"   时间偏移: {config.timestep_shift}")
    print("="*60)
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建Transport和Sampler
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    )
    
    sampler = Sampler(transport)
    
    # 生成样本
    all_samples = []
    
    with torch.no_grad():
        # 为每个用户生成样本
        for user_id in tqdm(range(config.num_classes), desc="生成用户样本"):
            user_samples = []
            
            for sample_idx in range(config.num_samples_per_user):
                # 准备噪声
                z = torch.randn(1, 32, 16, 16, device=config.device)
                
                # 准备标签
                y = torch.tensor([user_id], device=config.device)
                
                # CFG采样
                if config.cfg_scale > 1.0:
                    # 双倍batch用于CFG
                    z_double = torch.cat([z, z], dim=0)
                    y_cond = y
                    y_uncond = torch.full_like(y, config.num_classes)
                    y_double = torch.cat([y_cond, y_uncond], dim=0)
                    
                    model_kwargs = dict(y=y_double, cfg_scale=config.cfg_scale)
                    
                    # 使用优化的采样参数
                    sample_fn = sampler.sample_ode(
                        sampling_method=config.sampling_method,
                        num_steps=config.num_sampling_steps,
                        atol=config.atol,
                        rtol=config.rtol,
                        time_shifting_factor=config.timestep_shift,  # 添加时间偏移
                    )
                    
                    # 获取模型函数
                    model_fn = model.forward_with_cfg
                    
                    # 生成
                    samples = sample_fn(z_double, model_fn, **model_kwargs)[-1]
                    sample = samples[:1]  # 取条件生成结果
                else:
                    # 无CFG采样
                    model_kwargs = dict(y=y)
                    
                    sample_fn = sampler.sample_ode(
                        sampling_method=config.sampling_method,
                        num_steps=config.num_sampling_steps,
                        atol=config.atol,
                        rtol=config.rtol,
                        time_shifting_factor=config.timestep_shift,
                    )
                    
                    sample = sample_fn(z, model, **model_kwargs)[-1]
                
                # VAE解码
                sample = sample / 0.13025
                decoded = vae.decode(sample)
                
                # 转换为图像
                decoded = (decoded + 1) / 2
                decoded = decoded.clamp(0, 1)
                
                user_samples.append(decoded.cpu())
            
            # 合并用户样本
            user_samples = torch.cat(user_samples, dim=0)
            all_samples.append(user_samples)
            
            # 保存单个用户的样本
            save_user_samples(user_samples, user_id, output_dir)
    
    # 创建总览图
    create_overview_grid(all_samples, output_dir)
    
    print(f"\n✅ 生成完成！样本保存在: {output_dir}")

def save_user_samples(samples, user_id, output_dir):
    """保存单个用户的样本"""
    user_dir = output_dir / f"user_{user_id:02d}"
    user_dir.mkdir(exist_ok=True)
    
    for idx, sample in enumerate(samples):
        # 转换为PIL图像
        sample_np = sample.permute(1, 2, 0).numpy()
        sample_np = (sample_np * 255).astype(np.uint8)
        img = Image.fromarray(sample_np)
        
        # 保存
        img.save(user_dir / f"sample_{idx:02d}.png")

def create_overview_grid(all_samples, output_dir):
    """创建所有用户的样本总览图"""
    from torchvision.utils import make_grid
    
    # 每个用户取第一个样本
    overview_samples = []
    for user_samples in all_samples[:16]:  # 最多显示16个用户
        if len(user_samples) > 0:
            overview_samples.append(user_samples[0:1])
    
    if overview_samples:
        overview_samples = torch.cat(overview_samples, dim=0)
        grid = make_grid(overview_samples, nrow=4, padding=2)
        
        # 转换为PIL图像
        grid_np = grid.permute(1, 2, 0).numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
        img = Image.fromarray(grid_np)
        
        # 保存
        img.save(output_dir / 'overview.png')
        print(f"📊 总览图已保存: {output_dir / 'overview.png'}")

def compare_sampling_methods():
    """比较不同采样方法的效果"""
    
    config = OptimizedSamplingConfig()
    model, vae = load_models(config)
    
    # 测试不同的采样配置
    test_configs = [
        # 原始配置（问题配置）
        {
            'name': 'original_euler',
            'method': 'euler',
            'steps': 250,
            'cfg': 10.0,
            'shift': 0.1
        },
        # 优化配置1：dopri5高精度
        {
            'name': 'optimized_dopri5',
            'method': 'dopri5',
            'steps': 150,
            'cfg': 7.0,
            'shift': 0.1
        },
        # 优化配置2：heun二阶方法
        {
            'name': 'heun_balanced',
            'method': 'heun',
            'steps': 100,
            'cfg': 5.0,
            'shift': 0.05
        },
        # 优化配置3：midpoint方法
        {
            'name': 'midpoint_fast',
            'method': 'midpoint',
            'steps': 50,
            'cfg': 4.0,
            'shift': 0.0
        }
    ]
    
    comparison_dir = Path(config.output_dir) / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 固定种子以便公平比较
    torch.manual_seed(42)
    
    for test_cfg in test_configs:
        print(f"\n🧪 测试配置: {test_cfg['name']}")
        
        # 更新配置
        config.sampling_method = test_cfg['method']
        config.num_sampling_steps = test_cfg['steps']
        config.cfg_scale = test_cfg['cfg']
        config.timestep_shift = test_cfg['shift']
        
        # 生成样本
        with torch.no_grad():
            # 固定用户ID和噪声
            user_id = 5  # 选择一个中间的用户
            z = torch.randn(1, 32, 16, 16, device=config.device)
            y = torch.tensor([user_id], device=config.device)
            
            # 创建Transport和Sampler
            transport = create_transport(
                path_type='Linear',
                prediction='velocity',
            )
            sampler = Sampler(transport)
            
            # CFG采样
            z_double = torch.cat([z, z], dim=0)
            y_cond = y
            y_uncond = torch.full_like(y, config.num_classes)
            y_double = torch.cat([y_cond, y_uncond], dim=0)
            
            model_kwargs = dict(y=y_double, cfg_scale=config.cfg_scale)
            
            sample_fn = sampler.sample_ode(
                sampling_method=config.sampling_method,
                num_steps=config.num_sampling_steps,
                atol=1e-6,
                rtol=1e-3,
            )
            
            model_fn = model.forward_with_cfg
            samples = sample_fn(z_double, model_fn, **model_kwargs)[-1]
            sample = samples[:1]
            
            # VAE解码
            sample = sample / 0.13025
            decoded = vae.decode(sample)
            decoded = (decoded + 1) / 2
            decoded = decoded.clamp(0, 1)
            
            # 保存
            img_np = decoded[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            img.save(comparison_dir / f"{test_cfg['name']}.png")
            print(f"   ✅ 保存: {comparison_dir / f'{test_cfg['name']}.png'}")

def main():
    """主函数"""
    
    print("="*60)
    print("🚀 优化的DiT采样生成")
    print("="*60)
    
    config = OptimizedSamplingConfig()
    
    # 选择模式
    mode = 'generate'  # 或 'compare'
    
    if mode == 'generate':
        # 生成优化样本
        model, vae = load_models(config)
        generate_samples_optimized(model, vae, config)
    elif mode == 'compare':
        # 比较不同采样方法
        compare_sampling_methods()
    
    print("\n✨ 完成！")

if __name__ == "__main__":
    main()
