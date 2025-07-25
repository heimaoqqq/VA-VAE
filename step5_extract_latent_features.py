#!/usr/bin/env python3
"""
步骤5: 提取潜在特征
- 使用微调后的VA-VAE编码器
- 提取所有训练数据的潜在特征
- 保存为扩散模型训练格式
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

def load_finetuned_vavae(checkpoint_path):
    """加载微调后的VA-VAE模型"""
    print(f"📥 加载微调后的VA-VAE: {checkpoint_path}")
    
    # 这里需要根据实际的checkpoint格式来加载
    # 通常Lightning保存的checkpoint包含model state_dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 根据实际保存格式调整
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载模型（这里需要根据实际的模型类来调整）
        from LightningDiT.tokenizer.vavae import VAVAE
        model = VAVAE()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print("✅ VA-VAE模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 加载VA-VAE失败: {e}")
        return None

def extract_features_from_dataset(model, dataset_dir, output_dir, split_name):
    """从数据集提取潜在特征"""
    print(f"\n🔄 提取 {split_name} 集特征...")
    
    split_dir = Path(dataset_dir) / split_name
    output_split_dir = Path(output_dir) / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 统计信息
    total_samples = 0
    user_counts = {}
    
    # 遍历每个用户目录
    for user_id in range(1, 32):  # 用户1到31
        user_dir = split_dir / f"user{user_id}"
        if not user_dir.exists():
            continue
        
        print(f"  处理用户{user_id}...")
        
        # 创建输出用户目录
        output_user_dir = output_split_dir / f"user{user_id}"
        output_user_dir.mkdir(exist_ok=True)
        
        # 处理该用户的所有图像
        image_files = list(user_dir.glob("*.png")) + \
                     list(user_dir.glob("*.jpg")) + \
                     list(user_dir.glob("*.jpeg"))
        
        user_samples = 0
        for img_file in tqdm(image_files, desc=f"用户{user_id}"):
            try:
                # 加载图像
                from PIL import Image
                image = Image.open(img_file).convert('RGB')
                image = image.resize((256, 256), Image.LANCZOS)
                
                # 转换为tensor
                image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
                image_tensor = image_tensor.to(device)
                
                # 提取潜在特征
                with torch.no_grad():
                    latent = model.encode(image_tensor)
                    if hasattr(latent, 'sample'):
                        latent = latent.sample()
                
                # 保存潜在特征
                latent_file = output_user_dir / f"{img_file.stem}.pt"
                torch.save({
                    'latent': latent.cpu(),
                    'user_id': user_id - 1,  # 转换为0-30索引
                    'original_file': str(img_file)
                }, latent_file)
                
                user_samples += 1
                total_samples += 1
                
            except Exception as e:
                print(f"    ❌ 处理失败 {img_file}: {e}")
        
        user_counts[user_id] = user_samples
        print(f"    ✅ 用户{user_id}: {user_samples} 个特征")
    
    # 保存统计信息
    stats = {
        'split': split_name,
        'total_samples': total_samples,
        'user_counts': user_counts,
        'latent_shape': latent.shape[1:] if 'latent' in locals() else None
    }
    
    stats_file = output_split_dir / "extraction_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"  ✅ {split_name} 集完成: {total_samples} 个特征")
    return stats

def create_latent_dataset_config(output_dir, train_stats, val_stats):
    """创建潜在特征数据集配置"""
    print("\n📝 创建潜在特征数据集配置...")
    
    config_content = f"""# 微多普勒潜在特征数据集配置
# 用于LightningDiT扩散模型训练

dataset:
  name: "micro_doppler_latents"
  num_users: 31
  latent_shape: {train_stats['latent_shape']}
  
  # 数据路径
  train_dir: "{output_dir}/train"
  val_dir: "{output_dir}/val"
  
  # 数据统计
  train_samples: {train_stats['total_samples']}
  val_samples: {val_stats['total_samples']}
  
  # 用户分布
  train_user_counts: {train_stats['user_counts']}
  val_user_counts: {val_stats['user_counts']}

# LightningDiT训练配置
lightningdit:
  model_type: "LightningDiT-XL"
  in_chans: {train_stats['latent_shape'][0] if train_stats['latent_shape'] else 32}
  num_classes: 31  # 31个用户
  
  # 训练参数
  batch_size: 2  # T4×2 GPU
  learning_rate: 1.0e-04
  max_epochs: 800
  
  # 采样参数
  num_sampling_steps: 50
  cfg_scale: 4.0
"""
    
    config_file = Path(output_dir) / "latent_dataset_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 潜在特征配置已保存: {config_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='步骤5: 提取潜在特征')
    parser.add_argument('--dataset_dir', type=str, default='micro_doppler_dataset',
                       help='原始数据集目录')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='微调后的VA-VAE checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='micro_doppler_latents',
                       help='潜在特征输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 步骤5: 提取潜在特征")
    print("=" * 60)
    print(f"数据集目录: {args.dataset_dir}")
    print(f"VA-VAE checkpoint: {args.checkpoint_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 检查输入
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"❌ 数据集目录不存在: {args.dataset_dir}")
        return False
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ VA-VAE checkpoint不存在: {args.checkpoint_path}")
        print("💡 请先运行: python step4_finetune_vavae.py")
        return False
    
    # 2. 加载微调后的VA-VAE
    model = load_finetuned_vavae(args.checkpoint_path)
    if model is None:
        return False
    
    # 3. 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 4. 提取训练集特征
    train_stats = extract_features_from_dataset(
        model, args.dataset_dir, args.output_dir, 'train'
    )
    
    # 5. 提取验证集特征
    val_stats = extract_features_from_dataset(
        model, args.dataset_dir, args.output_dir, 'val'
    )
    
    # 6. 创建配置文件
    create_latent_dataset_config(args.output_dir, train_stats, val_stats)
    
    print("\n✅ 步骤5完成！潜在特征提取完成")
    print(f"📁 特征位置: {args.output_dir}")
    print(f"📊 训练集: {train_stats['total_samples']} 个特征")
    print(f"📊 验证集: {val_stats['total_samples']} 个特征")
    print("📋 下一步: python step6_train_diffusion_model.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
