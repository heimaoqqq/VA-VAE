#!/usr/bin/env python3
"""
简单生成测试脚本 - 无筛选，直接检查扩散模型生成质量
"""

import os
import torch
import argparse
from pathlib import Path
from PIL import Image
import yaml
import sys

# 添加LightningDiT路径
sys.path.insert(0, './LightningDiT')
from lightningdit.inference import create_model_and_transport
from simplified_vavae import SimplifiedVAVAE

def load_models(args, device, rank=0):
    """加载DiT和VAE模型"""
    print(f"🔄 [GPU{rank}] 加载模型...")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建DiT模型和transport
    model, transport = create_model_and_transport(
        config_path=args.config,
        checkpoint_path=args.dit_checkpoint,
        device=device,
        use_compile=False
    )
    
    # 加载VAE
    print(f"🔄 [GPU{rank}] 加载VA-VAE...")
    vae = SimplifiedVAVAE.from_pretrained('./VA-VAE-checkpoint')
    vae = vae.to(device)
    vae.eval()
    
    return model, transport, vae

def generate_samples(model, vae, transport, user_id, num_samples, batch_size, device, rank=0):
    """直接生成样本，无任何筛选"""
    print(f"🔄 [GPU{rank}] 为用户{user_id}生成{num_samples}个样本...")
    
    generated_images = []
    total_generated = 0
    
    while len(generated_images) < num_samples:
        current_batch_size = min(batch_size, num_samples - len(generated_images))
        
        # 生成条件向量
        y = torch.full((current_batch_size,), user_id, dtype=torch.long, device=device)
        
        # 生成latent
        latents = torch.randn(current_batch_size, 32, 16, 16, device=device)
        
        # DiT生成
        with torch.no_grad():
            samples = transport.sample_ode(
                model.forward_with_cfg,
                latents,
                model_kwargs={"y": y},
                cfg_scale=12.0,
                sample_steps=50
            )
        
        # VAE解码
        if vae is not None:
            try:
                # 直接解码，不做反归一化（基于之前的诊断结果）
                decoded_images = vae.decode_to_images(samples)
                images_pil = [Image.fromarray(img) for img in decoded_images]
                generated_images.extend(images_pil)
                
                total_generated += current_batch_size
                print(f"✅ [GPU{rank}] 用户{user_id}: 已生成 {len(generated_images)}/{num_samples}")
                
            except Exception as e:
                print(f"❌ [GPU{rank}] VAE解码错误: {e}")
                break
        else:
            print(f"❌ [GPU{rank}] VAE未加载")
            break
    
    return generated_images[:num_samples], total_generated

def save_samples(images, user_id, output_dir, rank=0):
    """保存生成的样本"""
    user_dir = Path(output_dir) / f"User_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for i, img in enumerate(images):
        try:
            img_path = user_dir / f"sample_{rank}_{i:04d}.png"
            img.save(img_path)
            saved_count += 1
        except Exception as e:
            print(f"❌ [GPU{rank}] 保存图像失败: {e}")
    
    print(f"✅ [GPU{rank}] 用户{user_id}: 保存了 {saved_count} 张图像到 {user_dir}")
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='简单生成测试 - 无筛选')
    parser.add_argument('--dit_checkpoint', type=str, required=True,
                       help='DiT checkpoint path')
    parser.add_argument('--config', type=str, 
                       default='configs/dit_s_microdoppler.yaml',
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./raw_samples_test',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to generate per user')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for generation')
    parser.add_argument('--start_user', type=int, default=0,
                       help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=5,
                       help='Ending user ID (exclusive)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"🚀 开始简单生成测试")
    print(f"📊 目标: 每用户 {args.num_samples} 张样本")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"👥 用户范围: {args.start_user} - {args.end_user-1}")
    
    # 加载模型
    model, transport, vae = load_models(args, device, rank)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 为每个用户生成样本
    total_stats = {'generated': 0, 'saved': 0}
    
    for user_id in range(args.start_user, args.end_user):
        print(f"\n🎯 [GPU{rank}] 处理用户 {user_id}...")
        
        # 生成样本
        images, generated_count = generate_samples(
            model, vae, transport, user_id, 
            args.num_samples, args.batch_size, device, rank
        )
        
        # 保存样本
        saved_count = save_samples(images, user_id, args.output_dir, rank)
        
        total_stats['generated'] += generated_count
        total_stats['saved'] += saved_count
        
        print(f"✅ [GPU{rank}] 用户{user_id}完成: 生成{generated_count}, 保存{saved_count}")
    
    # 最终统计
    print(f"\n🎉 生成测试完成!")
    print(f"📊 总计生成: {total_stats['generated']} 张")
    print(f"💾 总计保存: {total_stats['saved']} 张")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"\n💡 请手工检查生成样本的质量:")
    print(f"   1. 图像是否清晰可辨认?")
    print(f"   2. 是否存在明显的生成伪影?")
    print(f"   3. 不同用户的样本是否有区别?")
    print(f"   4. 是否符合微多普勒数据的预期特征?")

if __name__ == "__main__":
    main()
