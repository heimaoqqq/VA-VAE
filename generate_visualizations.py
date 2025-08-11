#!/usr/bin/env python3
"""
手动生成条件DiT模型的可视化图像
用于从训练好的检查点生成用户条件样本
"""

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

# 导入必要的模型类
import sys
sys.path.append('LightningDiT')

from models.lightningdit import LightningDiT_models
from tokenizer.vavae import VA_VAE


class UserConditionEncoder(torch.nn.Module):
    """用户条件编码器"""
    def __init__(self, num_users, embed_dim=1152):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim)
        self.projection = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, user_ids):
        embeddings = self.user_embedding(user_ids)
        return self.projection(embeddings)


class ConditionalDiT(torch.nn.Module):
    """条件DiT模型"""
    def __init__(self, 
                 model_name="LightningDiT-XL/1",
                 num_users=31,
                 condition_dim=1152,
                 pretrained_path=None):
        super().__init__()
        
        # 获取DiT模型
        self.dit = LightningDiT_models[model_name](
            input_size=16,
            patch_size=1,
            in_channels=32,
            num_classes=num_users,
            use_qknorm=False,
            use_swiglu=False,
            use_rope=False,
            use_rmsnorm=False,
            wo_shift=False
        )
        
        # 用户条件编码器
        self.user_encoder = UserConditionEncoder(num_users, condition_dim)
        
        # 加载预训练权重(如果提供)
        if pretrained_path and Path(pretrained_path).exists():
            print(f"加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                # 只加载DiT相关的权重，忽略新增的条件编码器
                dit_state_dict = {}
                for k, v in checkpoint['model'].items():
                    if k.startswith('dit.'):
                        dit_state_dict[k[4:]] = v  # 去掉'dit.'前缀
                self.dit.load_state_dict(dit_state_dict, strict=False)
                print("✅ 预训练权重加载完成")
        
        self.num_users = num_users
    
    def forward(self, x, t, user_ids):
        """前向传播"""
        # 编码用户条件
        user_condition = self.user_encoder(user_ids)
        
        # DiT前向传播
        return self.dit(x, t, y=user_condition)


def load_trained_model(checkpoint_path, config_path, device='cuda'):
    """加载训练好的模型"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ 检查点加载成功: epoch {checkpoint['epoch']}")
    
    # 创建模型
    model_config = config.get('model', {}).get('params', {})
    model = ConditionalDiT(
        model_name=model_config.get('model', "LightningDiT-XL/1"),
        num_users=model_config.get('num_users', 31),
        condition_dim=model_config.get('condition_dim', 1152)
    )
    
    # 加载训练的权重
    if 'model_state_dict' in checkpoint:
        # 处理DataParallel保存的权重
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            # 移除DataParallel的'module.'前缀
            state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
        
        model.load_state_dict(state_dict, strict=False)
        print("✅ 训练权重加载成功")
    
    model.to(device)
    model.eval()
    return model, config


def load_vae_model(device='cuda'):
    """加载VA-VAE模型"""
    vae_config_path = "LightningDiT/tokenizer/configs/vae_config.yaml"
    
    if not Path(vae_config_path).exists():
        raise FileNotFoundError(f"VA-VAE配置文件未找到: {vae_config_path}")
    
    vae = VA_VAE(vae_config_path)
    vae.to(device)
    vae.eval()
    print("✅ VA-VAE模型加载成功")
    return vae


def generate_samples(model, vae, device='cuda', num_samples_per_user=4, selected_users=None):
    """生成用户条件样本"""
    model.eval()
    vae.eval()
    
    # 如果未指定用户，则选择代表性用户
    if selected_users is None:
        # 选择8个代表性用户 (0-based索引，对应ID_1到ID_31)
        selected_users = [0, 4, 9, 14, 19, 24, 29, 30]
    
    num_users_to_sample = len(selected_users)
    total_samples = num_users_to_sample * num_samples_per_user
    
    # 准备用户标签
    user_ids = []
    for user_idx in selected_users:
        user_ids.extend([user_idx] * num_samples_per_user)
    user_ids = torch.tensor(user_ids, device=device)
    
    print(f"🎨 开始生成样本...")
    print(f"   - 用户数: {num_users_to_sample}")
    print(f"   - 每用户样本数: {num_samples_per_user}")
    print(f"   - 总样本数: {total_samples}")
    
    with torch.no_grad():
        # 生成随机噪声 (VA-VAE潜向量形状: B x 32 x 16 x 16)
        z_shape = (total_samples, 32, 16, 16)
        z = torch.randn(z_shape, device=device)
        
        # 创建时间步 (使用随机时间步进行多样性)
        t = torch.randint(0, 1000, (total_samples,), device=device)
        
        # 条件生成
        print("🔄 DiT生成中...")
        generated_z = model(z, t, user_ids)
        
        # 解码为图像
        print("🔄 VA-VAE解码中...")
        generated_images = vae.decode_to_images(generated_z)
        
        print("✅ 生成完成!")
        
        return generated_images, selected_users, num_samples_per_user


def visualize_and_save(images, selected_users, num_samples_per_user, output_path=None):
    """可视化并保存结果"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"manual_visualization_{timestamp}.png"
    
    num_users = len(selected_users)
    
    # 创建图像网格
    fig, axes = plt.subplots(num_users, num_samples_per_user, 
                            figsize=(num_samples_per_user * 3, num_users * 3))
    
    if num_users == 1:
        axes = axes.reshape(1, -1)
    if num_samples_per_user == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('User-Conditional Micro-Doppler Generation', fontsize=16, fontweight='bold')
    
    for user_idx in range(num_users):
        for sample_idx in range(num_samples_per_user):
            img_idx = user_idx * num_samples_per_user + sample_idx
            img = images[img_idx]
            
            # 处理图像格式
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                img = img.transpose(1, 2, 0)
            
            if img.shape[-1] == 1:  # 灰度图
                img = img.squeeze(-1)
            
            # 显示图像
            axes[user_idx, sample_idx].imshow(img, cmap='viridis')
            
            # 设置标题 (显示实际用户ID)
            actual_user_id = selected_users[user_idx] + 1  # 0-based -> 1-based
            if sample_idx == 0:  # 只在第一列显示用户ID
                axes[user_idx, sample_idx].set_ylabel(f'User ID_{actual_user_id}', 
                                                    fontweight='bold', fontsize=12)
            
            axes[user_idx, sample_idx].set_xticks([])
            axes[user_idx, sample_idx].set_yticks([])
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📸 可视化结果已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='手动生成条件DiT可视化')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='训练好的模型检查点路径 (best.ckpt 或 last.ckpt)')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (config.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径 (默认: manual_visualization_timestamp.png)')
    parser.add_argument('--users', type=int, nargs='+', default=None,
                        help='要生成的用户ID列表 (0-based, 例如: --users 0 5 10 15)')
    parser.add_argument('--samples-per-user', type=int, default=4,
                        help='每个用户生成的样本数 (默认: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    print("🚀 开始手动生成可视化...")
    print(f"   - 检查点: {args.checkpoint}")
    print(f"   - 配置: {args.config}")
    print(f"   - 设备: {args.device}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")
    
    try:
        # 加载模型
        print("\n📥 加载模型...")
        model, config = load_trained_model(args.checkpoint, args.config, device)
        vae = load_vae_model(device)
        
        # 生成样本
        print("\n🎨 生成样本...")
        images, selected_users, num_samples_per_user = generate_samples(
            model, vae, device, 
            num_samples_per_user=args.samples_per_user,
            selected_users=args.users
        )
        
        # 可视化并保存
        print("\n📸 保存可视化...")
        output_path = visualize_and_save(
            images, selected_users, num_samples_per_user, args.output
        )
        
        print(f"\n🎉 完成! 可视化图像已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        raise


if __name__ == "__main__":
    main()
