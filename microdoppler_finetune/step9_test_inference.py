"""
独立测试推理脚本 - 诊断生成质量问题
"""
import torch
import numpy as np
from PIL import Image
import os
import yaml
from pathlib import Path

# 添加项目路径
import sys
sys.path.append('g:/VA-VAE')
sys.path.append('g:/VA-VAE/LightningDiT')

def test_inference():
    """测试推理，诊断问题"""
    
    # 1. 加载配置
    config_path = "g:/VA-VAE/microdoppler_finetune/config_dit_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. 检查checkpoint
    checkpoint_dir = Path(config['train']['output_dir']) / config['train']['exp_name'] / "checkpoints"
    
    # 查找最新的checkpoint
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("❌ 没有找到任何checkpoint!")
        return
    
    # 按修改时间排序，获取最新的
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"📦 Loading checkpoint: {latest_checkpoint}")
    
    # 3. 加载checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # 分析checkpoint内容
    print("\n📊 Checkpoint内容分析:")
    print(f"   Keys: {checkpoint.keys()}")
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"   Model参数数量: {len(model_state)}")
        # 显示几个关键参数的统计
        for key in list(model_state.keys())[:5]:
            param = model_state[key]
            print(f"     {key}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
    
    if 'ema' in checkpoint:
        ema_state = checkpoint['ema']
        print(f"   EMA参数数量: {len(ema_state)}")
        # 显示几个关键参数的统计
        for key in list(ema_state.keys())[:5]:
            param = ema_state[key]
            print(f"     {key}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
    
    # 4. 比较model和ema的差异
    if 'model' in checkpoint and 'ema' in checkpoint:
        print("\n🔍 Model vs EMA参数对比:")
        model_state = checkpoint['model']
        ema_state = checkpoint['ema']
        
        # 计算参数差异
        total_diff = 0
        param_count = 0
        for key in model_state.keys():
            if key in ema_state:
                diff = (model_state[key] - ema_state[key]).abs().mean().item()
                total_diff += diff
                param_count += 1
                if param_count <= 5:  # 只显示前5个
                    print(f"   {key}: diff={diff:.6f}")
        
        avg_diff = total_diff / param_count if param_count > 0 else 0
        print(f"   平均参数差异: {avg_diff:.6f}")
        
        if avg_diff < 1e-6:
            print("   ⚠️ 警告: Model和EMA参数几乎相同，EMA可能没有正确更新!")
    
    # 5. 检查训练状态
    if 'epoch' in checkpoint:
        print(f"\n📈 训练状态:")
        print(f"   Epoch: {checkpoint['epoch']}")
    
    if 'best_val_loss' in checkpoint:
        print(f"   Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    # 6. 测试实际生成
    print("\n🎨 开始测试生成...")
    
    # 导入必要的模块
    from transport import create_transport, Sampler
    from models import DiT_models
    from tokenizer.vavae import VA_VAE
    
    # 创建模型
    model = DiT_models[config['model']['model_type']](
        input_size=config['data']['image_size'] // config['vae']['downsample_ratio'],
        num_classes=config['data']['num_classes'],
        in_channels=config['model']['in_channels'],
        depth=config['model']['depth'],
        hidden_size=config['model']['hidden_size'],
        patch_size=config['model']['patch_size'],
        num_heads=config['model']['num_heads'],
        learn_sigma=config['model']['learn_sigma'],
        class_dropout_prob=config['model']['class_dropout_prob'],
    ).to(device)
    
    # 加载EMA权重
    if 'ema' in checkpoint:
        print("   使用EMA模型权重")
        model.load_state_dict(checkpoint['ema'])
    else:
        print("   ⚠️ 没有找到EMA权重，使用普通模型权重")
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    
    # 创建transport
    transport = create_transport(
        config['transport']['path_type'],
        config['transport']['prediction'],
        config['transport']['loss_weight'],
        config['transport']['train_eps'],
        config['transport']['sample_eps'],
    )
    
    # 创建sampler
    sampler = Sampler(transport)
    
    # 测试不同的采样配置
    test_configs = [
        {'method': 'euler', 'steps': 50, 'cfg': 1.0},
        {'method': 'euler', 'steps': 100, 'cfg': 4.0},
        {'method': 'dopri5', 'steps': 150, 'cfg': 5.0},
    ]
    
    for test_cfg in test_configs:
        print(f"\n   测试配置: {test_cfg}")
        
        sample_fn = sampler.sample_ode(
            sampling_method=test_cfg['method'],
            num_steps=test_cfg['steps'],
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            timestep_shift=0.0,  # 关键：确保为0
        )
        
        # 生成一个样本
        latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        y = torch.tensor([0], device=device)
        
        # CFG设置
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([config['data']['num_classes']], device=device)
        y = torch.cat([y, y_null], 0)
        
        model_kwargs = dict(y=y, cfg_scale=test_cfg['cfg'], cfg_interval=True, cfg_interval_start=0.11)
        
        # 采样
        with torch.no_grad():
            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)
        
        print(f"     生成的latent统计: mean={samples.mean():.4f}, std={samples.std():.4f}, min={samples.min():.4f}, max={samples.max():.4f}")
        
        # 检查是否全是噪声
        if samples.std() > 10 or samples.std() < 0.01:
            print(f"     ⚠️ 异常: latent标准差异常 (std={samples.std():.4f})")
    
    print("\n✅ 诊断完成!")

if __name__ == "__main__":
    test_inference()
