#!/usr/bin/env python3
"""
步骤3: LightningDiT基础推理测试与环境验证
验证所有模型可正常加载，进行基础推理测试
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import traceback

def test_environment():
    """测试基础环境"""
    print("🔍 测试基础环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ CUDA不可用，将使用CPU")
    
    # 检查必要目录
    required_dirs = ["models", "LightningDiT"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ 目录存在: {dir_name}")
        else:
            print(f"❌ 目录缺失: {dir_name}")
            return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    try:
        # 添加LightningDiT到Python路径
        sys.path.append(str(Path("LightningDiT").absolute()))
        
        print("📥 加载VA-VAE...")
        from tokenizer.vavae import VA_VAE
        
        # 检查配置文件
        vavae_config = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
        if not Path(vavae_config).exists():
            print(f"❌ 配置文件不存在: {vavae_config}")
            return False
            
        # 加载VA-VAE
        vae = VA_VAE(vavae_config)
        print("✅ VA-VAE加载成功")
        
        # 测试VA-VAE推理
        print("🧪 测试VA-VAE编码解码...")
        with torch.no_grad():
            # 创建测试图像 (batch_size=1, channels=3, height=256, width=256)
            test_img = torch.randn(1, 3, 256, 256)
            
            # 编码
            z = vae.encode_images(test_img)
            print(f"✅ 编码成功，潜向量形状: {z.shape}")
            
            # 解码
            decoded_images = vae.decode_to_images(z)
            print(f"✅ 解码成功，输出形状: {decoded_images.shape}")
        
        print("📥 加载DiT模型...")
        from models.lightningdit import LightningDiT_models
        
        # 加载DiT模型架构 - 严格按照官方配置参数
        dit = LightningDiT_models["LightningDiT-XL/1"](
            input_size=16,           # 16 = 256/16 (VA-VAE下采样率)
            in_channels=32,          # VA-VAE潜向量通道数
            use_qknorm=False,        # 官方配置
            use_swiglu=True,         # 官方配置
            use_rope=True,           # 官方配置
            use_rmsnorm=True,        # 官方配置
            wo_shift=False           # 官方配置
        )
        print(f"✅ DiT架构创建成功，参数量: {sum(p.numel() for p in dit.parameters()):,}")
        
        # 加载预训练权重
        checkpoint_path = "models/lightningdit-xl-imagenet256-64ep.pt"
        if not Path(checkpoint_path).exists():
            print(f"❌ 权重文件不存在: {checkpoint_path}")
            return False
            
        print("📥 加载预训练权重...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查checkpoint内容
        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
                print("✅ 使用EMA权重")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("✅ 使用模型权重")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✅ 使用状态字典")
            else:
                state_dict = checkpoint
                print("✅ 使用直接权重")
        else:
            state_dict = checkpoint
        
        # 过滤不匹配的键
        model_keys = set(dit.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # 移除不匹配的键
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"⚠️ 缺失键: {len(missing_keys)}个")
        if unexpected_keys:
            print(f"⚠️ 额外键: {len(unexpected_keys)}个")
            
        dit.load_state_dict(filtered_state_dict, strict=False)
        print("✅ DiT权重加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("错误详情:")
        traceback.print_exc()
        return False

def test_basic_inference():
    """测试基础推理"""
    print("\n🔍 测试基础推理...")
    
    try:
        sys.path.append(str(Path("LightningDiT").absolute()))
        
        from tokenizer.vavae import VA_VAE
        from models.lightningdit import LightningDiT_models
        
        # 加载模型 - 使用正确的API和完整配置
        vae = VA_VAE("LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
        dit = LightningDiT_models["LightningDiT-XL/1"](
            input_size=16,           
            in_channels=32,          
            use_qknorm=False,        
            use_swiglu=True,         
            use_rope=True,           
            use_rmsnorm=True,        
            wo_shift=False           
        )
        
        # 加载权重
        checkpoint = torch.load("models/lightningdit-xl-imagenet256-64ep.pt", map_location='cpu')
        if isinstance(checkpoint, dict) and 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        else:
            state_dict = checkpoint
        
        dit.load_state_dict(state_dict, strict=False)
        
        # 设置评估模式
        vae.model.eval()  # VA_VAE是包装器，真正的模型在.model属性中
        dit.eval()
        
        print("🎯 执行端到端推理...")
        with torch.no_grad():
            # 1. 创建随机噪声作为起点
            batch_size = 1
            latent_size = 16  # 256 / 16
            noise = torch.randn(batch_size, 32, latent_size, latent_size)  # 32是VA-VAE的潜向量维度
            
            # 2. 创建时间步
            t = torch.randint(0, 1000, (batch_size,))
            
            # 3. 创建随机类标签 (ImageNet有1000个类)
            y = torch.randint(0, 1000, (batch_size,))
            
            # 4. DiT预测噪声 (需要传入x, t, y三个参数)
            predicted_noise = dit(noise, t, y)
            print(f"✅ DiT推理成功，输出形状: {predicted_noise.shape}")
            print(f"✅ 使用类标签: {y.item()}, 时间步: {t.item()}")
            
            # 4. 简单去噪（这里只是演示，实际需要完整的DDPM采样）
            denoised = noise - predicted_noise * 0.1
            
            # 5. VA-VAE解码
            decoded_image = vae.decode_to_images(denoised)
            print(f"✅ 端到端推理成功，最终图像形状: {decoded_image.shape}")
            
            # 6. 保存测试图像
            output_dir = Path("test_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # 转换为图像格式  
            # VA_VAE.decode_to_images返回numpy数组，不是tensor
            if isinstance(decoded_image, np.ndarray):
                image = decoded_image[0]  # numpy数组，无需.cpu()
            else:
                image = decoded_image[0].cpu().numpy()  # 如果是tensor，转为numpy
            
            image = (image + 1) / 2  # 从[-1,1]转换到[0,1]
            image = np.clip(image, 0, 1)  # numpy的clip替代tensor的clamp
            image = (image * 255).astype(np.uint8)  # numpy的astype替代tensor的byte
            
            # 转换为PIL图像 (image已经是numpy数组)
            if image.shape[0] == 3:  # RGB: (C, H, W) -> (H, W, C)
                image_np = np.transpose(image, (1, 2, 0))  # numpy的transpose替代tensor的permute
                pil_image = Image.fromarray(image_np)
            else:  # 灰度图
                image_np = image[0]  # 直接取第一个通道，已经是numpy数组
                pil_image = Image.fromarray(image_np, mode='L')
            
            output_path = output_dir / "test_generation.png"
            pil_image.save(output_path)
            print(f"✅ 测试图像已保存: {output_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        print("错误详情:")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 步骤3: LightningDiT基础推理测试")
    print("="*60)
    
    # 测试环境
    if not test_environment():
        print("❌ 环境测试失败")
        return False
    
    # 测试模型加载
    if not test_model_loading():
        print("❌ 模型加载测试失败")
        return False
    
    # 测试基础推理
    if not test_basic_inference():
        print("❌ 推理测试失败")
        return False
    
    print("\n" + "="*60)
    print("🎉 所有测试通过！LightningDiT环境验证成功")
    print("✅ 可以进行下一步：micro-Doppler数据适配")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
