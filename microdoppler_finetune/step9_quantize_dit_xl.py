#!/usr/bin/env python3
"""
步骤9: 量化已训练的LightningDiT XL模型
对已完成训练的DiT XL模型进行推理优化，减少内存占用和加速生成
"""

import torch
import torch.quantization as quantization
import time
import os
import sys
from pathlib import Path
import psutil
import numpy as np
from tqdm import tqdm

# 添加LightningDiT路径
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')
sys.path.append('/kaggle/working/LightningDiT')

from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler

def load_trained_dit_xl(checkpoint_path, device):
    """加载已训练的DiT XL模型"""
    print(f"📂 加载训练完成的DiT XL模型: {checkpoint_path}")
    
    # 首先加载checkpoint来检查模型架构
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    
    # 从权重推断模型配置
    pos_embed_shape = None
    y_embed_shape = None
    final_layer_shape = None
    has_swiglu = False
    
    for key, tensor in state_dict.items():
        if key == 'pos_embed':
            pos_embed_shape = tensor.shape  # [1, seq_len, dim]
        elif key == 'y_embedder.embedding_table.weight':
            y_embed_shape = tensor.shape    # [num_classes, dim]
        elif key == 'final_layer.linear.weight':
            final_layer_shape = tensor.shape  # [out_channels, dim]
        elif 'mlp.w12' in key:
            has_swiglu = True
    
    # 推断参数
    input_size = int((pos_embed_shape[1])**0.5) if pos_embed_shape else 16
    num_classes = y_embed_shape[0] if y_embed_shape else 1000
    out_channels = final_layer_shape[0] if final_layer_shape else 4
    patch_size = 2  # 官方XL模型使用patch_size=2
    
    print(f"📋 检测到的模型配置:")
    print(f"   输入尺寸: {input_size}x{input_size}")
    print(f"   类别数量: {num_classes}")
    print(f"   输出通道: {out_channels}")
    print(f"   MLP类型: {'SwiGLU' if has_swiglu else 'Standard'}")
    print(f"   补丁大小: {patch_size}")
    
    # 创建模型架构 - 使用检测到的官方配置
    model = LightningDiT_models['LightningDiT-XL/2'](
        input_size=input_size,      # 从权重推断
        num_classes=num_classes,    # ImageNet 1000类 
        class_dropout_prob=0.0,     # 推理时不dropout
        use_qknorm=True,           # XL模型标配
        use_swiglu=has_swiglu,     # 从权重检测
        use_rope=True,             # XL模型使用RoPE
        use_rmsnorm=True,          # XL模型使用RMSNorm
        wo_shift=False,
        in_channels=out_channels,   # 与final_layer输出匹配
        use_checkpoint=False,       # 推理不需要checkpoint
    )
    
    # 移除可能的'module.'前缀
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 尝试加载权重，使用strict=False以忽略不匹配的层
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️ 缺失的权重键 ({len(missing_keys)}个):")
        for key in missing_keys[:5]:  # 只显示前5个
            print(f"   - {key}")
        if len(missing_keys) > 5:
            print(f"   ... 和其他 {len(missing_keys)-5} 个")
    
    if unexpected_keys:
        print(f"⚠️ 意外的权重键 ({len(unexpected_keys)}个):")
        for key in unexpected_keys[:5]:  # 只显示前5个
            print(f"   - {key}")
        if len(unexpected_keys) > 5:
            print(f"   ... 和其他 {len(unexpected_keys)-5} 个")
    
    model = model.to(device)
    model.eval()
    
    # 计算成功加载的权重比例
    total_params = len(state_dict)
    loaded_params = total_params - len(missing_keys)
    load_ratio = loaded_params / total_params * 100 if total_params > 0 else 0
    
    print(f"✅ DiT XL模型加载完成 ({load_ratio:.1f}%权重匹配)")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"   模型大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2):.1f}MB")
    
    # 如果权重匹配率太低，给出警告
    if load_ratio < 80:
        print(f"⚠️ 权重匹配率较低 ({load_ratio:.1f}%)，量化后的性能可能受影响")
        print(f"   建议使用与checkpoint完全匹配的模型架构")
    
    return model

def apply_dynamic_quantization(model):
    """应用PyTorch动态量化"""
    print("\n🔧 开始应用动态量化...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 应用动态量化 - 只量化Linear层（最稳定）
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 只量化Linear层
        dtype=torch.qint8,   # 使用8位整数
        inplace=False        # 不修改原模型
    )
    
    print("✅ 动态量化完成")
    print("   量化目标: Linear layers only")
    print("   量化精度: INT8")
    print("   量化方式: 动态量化（推理时实时量化activation）")
    
    return quantized_model

def measure_model_size(model, model_name):
    """测量模型大小"""
    # 保存到临时文件测量大小
    temp_path = f'temp_{model_name}.pt'
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024**2)
    os.remove(temp_path)
    return size_mb

def benchmark_inference_speed(model, model_name, device, num_runs=50):
    """基准测试推理速度"""
    print(f"\n⏱️ 测试 {model_name} 推理速度...")
    
    # 准备测试数据
    batch_size = 4
    test_latents = torch.randn(batch_size, 32, 16, 16).to(device)  # VA-VAE latent format
    test_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    test_labels = torch.randint(0, 31, (batch_size,)).to(device)
    
    model.eval()
    times = []
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_latents, test_timesteps, y=test_labels)
    
    # 正式测试
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in tqdm(range(num_runs), desc=f"Testing {model_name}"):
            start_time = time.time()
            _ = model(test_latents, test_timesteps, y=test_labels)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   平均推理时间: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"   吞吐量: {batch_size/avg_time:.2f} samples/sec")
    
    return avg_time, std_time

def benchmark_memory_usage(model, model_name, device):
    """基准测试内存使用"""
    print(f"\n💾 测试 {model_name} 内存使用...")
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 测试数据
    batch_size = 4
    test_latents = torch.randn(batch_size, 32, 16, 16).to(device)
    test_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    test_labels = torch.randint(0, 31, (batch_size,)).to(device)
    
    # 执行推理
    model.eval()
    with torch.no_grad():
        _ = model(test_latents, test_timesteps, y=test_labels)
    
    # 获取内存使用
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    model_size_mb = measure_model_size(model, model_name)
    
    print(f"   峰值显存使用: {memory_mb:.1f}MB")
    print(f"   模型文件大小: {model_size_mb:.1f}MB")
    
    return memory_mb, model_size_mb

def comprehensive_benchmark(original_model, quantized_model, device):
    """全面的性能对比基准测试"""
    print("\n" + "="*60)
    print("🔍 开始全面性能基准测试")
    print("="*60)
    
    results = {}
    
    # 1. 推理速度测试
    print("\n📊 1. 推理速度对比")
    original_time, _ = benchmark_inference_speed(original_model, "Original DiT XL", device)
    quantized_time, _ = benchmark_inference_speed(quantized_model, "Quantized DiT XL", device)
    
    speedup = original_time / quantized_time
    print(f"\n   加速比: {speedup:.2f}x")
    
    # 2. 内存使用测试
    print("\n📊 2. 内存使用对比")
    orig_memory, orig_size = benchmark_memory_usage(original_model, "original", device)
    quant_memory, quant_size = benchmark_memory_usage(quantized_model, "quantized", device)
    
    memory_reduction = (orig_memory - quant_memory) / orig_memory * 100
    size_reduction = (orig_size - quant_size) / orig_size * 100
    
    print(f"\n   显存节省: {memory_reduction:.1f}%")
    print(f"   文件大小减少: {size_reduction:.1f}%")
    
    # 3. 汇总结果
    results = {
        'speedup': speedup,
        'memory_reduction': memory_reduction,
        'size_reduction': size_reduction,
        'original_time': original_time,
        'quantized_time': quantized_time,
        'original_memory': orig_memory,
        'quantized_memory': quant_memory,
        'original_size': orig_size,
        'quantized_size': quant_size
    }
    
    print("\n" + "="*60)
    print("📈 量化效果汇总:")
    print(f"   🚀 推理加速:     {speedup:.2f}x")
    print(f"   💾 显存节省:     {memory_reduction:.1f}%")
    print(f"   📦 模型压缩:     {size_reduction:.1f}%")
    print(f"   ⏱️  延迟改善:     {original_time*1000:.1f}ms → {quantized_time*1000:.1f}ms")
    print("="*60)
    
    return results

def test_generation_quality(model, vae_model, device, num_samples=16):
    """测试生成质量（简单验证）"""
    print(f"\n🎨 测试生成质量 ({num_samples} samples)...")
    
    model.eval()
    
    # 创建transport用于采样
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        train_eps=1e-5,
        sample_eps=1e-5,
    )
    
    # 生成样本
    with torch.no_grad():
        # 随机噪声
        latents = torch.randn(num_samples, 32, 16, 16).to(device)
        
        # 随机用户标签
        labels = torch.randint(0, 31, (num_samples,)).to(device)
        
        # 使用dopri5采样器（高质量）
        sampler = Sampler(transport)
        samples = sampler.sample_ode(
            latents, 
            model, 
            steps=150,
            method='dopri5',
            cfg_scale=7.0,
            model_kwargs={'y': labels}
        )
        
        # 检查输出范围和分布
        sample_mean = samples.mean().item()
        sample_std = samples.std().item()
        sample_min = samples.min().item()
        sample_max = samples.max().item()
        
        print(f"   生成样本统计:")
        print(f"     Mean: {sample_mean:.4f}")
        print(f"     Std:  {sample_std:.4f}")
        print(f"     Range: [{sample_min:.4f}, {sample_max:.4f}]")
        
        # 检查是否有异常值
        if sample_std > 0 and not (torch.isnan(samples).any() or torch.isinf(samples).any()):
            print("   ✅ 生成质量正常")
            quality_ok = True
        else:
            print("   ⚠️ 生成质量异常")
            quality_ok = False
    
    return quality_ok, {
        'mean': sample_mean,
        'std': sample_std,
        'min': sample_min,
        'max': sample_max
    }

def main_quantization_pipeline(checkpoint_path, output_path):
    """主量化流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始DiT XL量化流程")
    print(f"   设备: {device}")
    print(f"   输入模型: {checkpoint_path}")
    print(f"   输出路径: {output_path}")
    
    # 1. 加载原始模型
    print("\n" + "="*50)
    print("步骤1: 加载原始DiT XL模型")
    print("="*50)
    original_model = load_trained_dit_xl(checkpoint_path, device)
    
    # 2. 应用量化
    print("\n" + "="*50)
    print("步骤2: 应用动态量化")
    print("="*50)
    quantized_model = apply_dynamic_quantization(original_model)
    
    # 3. 性能基准测试
    print("\n" + "="*50)
    print("步骤3: 性能基准测试")
    print("="*50)
    benchmark_results = comprehensive_benchmark(original_model, quantized_model, device)
    
    # 4. 生成质量测试
    print("\n" + "="*50)
    print("步骤4: 生成质量验证")
    print("="*50)
    
    # 原始模型质量测试
    orig_quality_ok, orig_stats = test_generation_quality(original_model, None, device)
    
    # 量化模型质量测试
    quant_quality_ok, quant_stats = test_generation_quality(quantized_model, None, device)
    
    print(f"\n   原始模型生成: {'✅ 正常' if orig_quality_ok else '❌ 异常'}")
    print(f"   量化模型生成: {'✅ 正常' if quant_quality_ok else '❌ 异常'}")
    
    # 5. 保存量化模型
    print("\n" + "="*50)
    print("步骤5: 保存量化模型")
    print("="*50)
    
    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存量化模型和结果
    save_data = {
        'quantized_model': quantized_model.state_dict(),
        'quantization_config': {
            'method': 'dynamic',
            'dtype': 'int8',
            'target_layers': 'Linear',
            'framework': 'PyTorch'
        },
        'benchmark_results': benchmark_results,
        'quality_test': {
            'original_quality_ok': orig_quality_ok,
            'original_stats': orig_stats,
            'quantized_quality_ok': quant_quality_ok,
            'quantized_stats': quant_stats
        },
        'model_config': {
            'model_type': 'LightningDiT-XL/2',
            'input_size': 16,
            'num_classes': 31,
            'in_channels': 32
        }
    }
    
    torch.save(save_data, output_path)
    print(f"✅ 量化模型已保存: {output_path}")
    
    # 6. 最终报告
    print("\n" + "="*60)
    print("🎯 DiT XL量化完成！")
    print("="*60)
    print(f"📈 性能提升:")
    print(f"   推理加速: {benchmark_results['speedup']:.2f}x")
    print(f"   显存节省: {benchmark_results['memory_reduction']:.1f}%")
    print(f"   模型压缩: {benchmark_results['size_reduction']:.1f}%")
    print(f"🎨 生成质量:")
    print(f"   原始模型: {'✅ 正常' if orig_quality_ok else '❌ 异常'}")
    print(f"   量化模型: {'✅ 正常' if quant_quality_ok else '❌ 异常'}")
    print(f"📦 输出文件: {output_path}")
    print("="*60)
    
    return quantized_model, benchmark_results

def load_quantized_model(quantized_path, device):
    """加载已量化的模型（用于后续推理）"""
    print(f"📂 加载量化模型: {quantized_path}")
    
    # 加载保存的数据
    save_data = torch.load(quantized_path, map_location=device)
    
    # 重建模型架构
    model = LightningDiT_models['LightningDiT-XL/2'](
        input_size=save_data['model_config']['input_size'],
        num_classes=save_data['model_config']['num_classes'],
        class_dropout_prob=0.0,
        use_qknorm=True,
        in_channels=save_data['model_config']['in_channels'],
        use_checkpoint=False,
    )
    
    # 应用量化
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )
    
    # 加载权重
    quantized_model.load_state_dict(save_data['quantized_model'])
    quantized_model.eval()
    
    print("✅ 量化模型加载完成")
    return quantized_model, save_data

if __name__ == "__main__":
    # 配置路径 - 与step2_download_models.py下载的文件名保持一致
    base_path = Path('/kaggle/working/VA-VAE') if os.path.exists('/kaggle/working') else Path.cwd()
    models_dir = base_path / "models"
    
    # 可能的模型路径（按优先级排序）
    possible_checkpoints = [
        # 1. 训练后的最佳模型（如果存在）
        "/kaggle/working/dit_xl_best_model.pt",
        "/kaggle/working/microdoppler_finetune/checkpoints/best_model.pt",
        
        # 2. step2下载的预训练模型
        str(models_dir / "lightningdit-xl-imagenet256-64ep.pt"),
        
        # 3. 其他可能的checkpoint位置
        "/kaggle/input/dit-xl-checkpoint/dit_xl_checkpoint.pt",
        "/kaggle/input/lightningdit-models/lightningdit-xl-imagenet256-64ep.pt"
    ]
    
    # 自动检测可用的checkpoint
    checkpoint_path = None
    for path in possible_checkpoints:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✅ 找到模型文件: {checkpoint_path}")
            break
    
    if checkpoint_path is None:
        print("❌ 未找到任何可用的DiT XL模型文件")
        print("\n可能的解决方案:")
        print("1. 运行 python step2_download_models.py 下载预训练模型")
        print("2. 确保你已完成DiT XL训练并保存checkpoint")
        print("3. 检查Kaggle输入数据集中是否包含模型文件")
        print(f"\n搜索路径:")
        for path in possible_checkpoints:
            print(f"   - {path}")
        exit(1)
    
    output_path = "/kaggle/working/dit_xl_quantized.pt"       # 量化后的输出
    
    # 执行量化流程
    quantized_model, results = main_quantization_pipeline(checkpoint_path, output_path)
    
    print(f"\n🎉 量化完成！可以使用量化后的模型进行更快的推理了。")
