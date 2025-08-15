#!/usr/bin/env python
"""VA-VAE模型验证与导出脚本"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
from omegaconf import OmegaConf

def setup_taming_path():
    """设置taming-transformers路径"""
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),
        Path('/kaggle/working/.taming_path'),
        Path.cwd().parent / 'taming-transformers',
        Path.cwd() / '.taming_path'
    ]
    
    for location in taming_locations:
        if location.name == '.taming_path' and location.exists():
            try:
                with open(location, 'r') as f:
                    taming_path = f.read().strip()
                if Path(taming_path).exists() and taming_path not in sys.path:
                    sys.path.insert(0, taming_path)
                    print(f"✅ Taming路径已添加: {taming_path}")
                    return True
            except Exception:
                continue
        elif location.name == 'taming-transformers' and location.exists():
            taming_path = str(location.absolute())
            if taming_path not in sys.path:
                sys.path.insert(0, taming_path)
                print(f"✅ Taming路径已添加: {taming_path}")
                return True
    
    print("⚠️ 未找到taming-transformers路径，尝试直接导入...")
    return False

# 设置taming路径
setup_taming_path()

# 导入必要的模块
try:
    from ldm.util import instantiate_from_config
    from ldm.models.autoencoder import AutoencoderKL
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保taming-transformers和latent-diffusion已正确安装")
    sys.exit(1)

def load_model(checkpoint_path, device='cuda'):
    """加载VA-VAE模型"""
    print(f"📂 加载模型: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 创建默认配置
        config = OmegaConf.create({
            'model': {
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'embed_dim': 32,
                    'use_vf': 'dinov2',
                    'reverse_proj': True,
                    'ddconfig': {
                        'double_z': True,
                        'z_channels': 32,
                        'resolution': 256,
                        'in_channels': 3,
                        'out_ch': 3,
                        'ch': 128,
                        'ch_mult': [1, 1, 2, 2, 4],
                        'num_res_blocks': 2,
                        'attn_resolutions': [16],
                        'dropout': 0.0
                    },
                    'lossconfig': {
                        'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                        'params': {
                            'disc_start': 1,
                            'kl_weight': 1e-6,
                            'disc_weight': 0.5
                        }
                    }
                }
            }
        })
    
    # 实例化模型
    model = instantiate_from_config(config.model)
    
    # 加载权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✅ 模型加载完成")
    return model


def validate_reconstruction(model, data_root, split_file, num_samples=16, device='cuda'):
    """验证重建质量"""
    
    print("\n🔍 验证重建质量...")
    
    # 加载数据
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 处理不同的数据结构格式
    if isinstance(split_data['val'], list):
        val_data = split_data['val'][:num_samples]
    else:
        # 如果val是字典格式，转换为列表
        val_data = list(split_data['val'].values())[:num_samples]
    
    # 检查数据项格式并调试
    if val_data and len(val_data) > 0:
        print(f"📊 数据项示例: {val_data[0]}")
        print(f"📊 数据项类型: {type(val_data[0])}")
    
    # 准备图像
    images = []
    reconstructions = []
    
    model = model.to(device)
    
    with torch.no_grad():
        processed_count = 0
        for idx, user_paths in enumerate(tqdm(val_data, desc="处理用户")):
            # 每个user_paths是一个用户的图片路径列表
            if isinstance(user_paths, list):
                # 从每个用户选择几张图片
                num_to_select = min(2, len(user_paths), num_samples - processed_count)
                selected_paths = user_paths[:num_to_select]
            else:
                selected_paths = [user_paths]  # 如果是单个路径
                
            for img_path_str in selected_paths:
                if processed_count >= num_samples:
                    break
                    
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                    
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256), Image.LANCZOS)
                
                # 转换为tensor
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 重建
                reconstructed, _, _, _ = model(img_tensor)
                
                images.append(img_tensor.cpu())
                reconstructions.append(reconstructed.cpu())
                processed_count += 1
                
            if processed_count >= num_samples:
                break
    
    # 计算指标
    images_cat = torch.cat(images, dim=0)
    recons_cat = torch.cat(reconstructions, dim=0)
    
    # MSE
    mse = torch.mean((images_cat - recons_cat) ** 2).item()
    
    # PSNR
    psnr = 20 * np.log10(2.0) - 10 * np.log10(mse)
    
    print(f"✅ 重建指标:")
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    # 保存可视化
    save_reconstruction_grid(images_cat, recons_cat, 'reconstruction_results.png')
    
    return mse, psnr


def save_reconstruction_grid(images, reconstructions, save_path, num_show=8):
    """保存重建对比图"""
    
    num_show = min(num_show, len(images))
    
    fig, axes = plt.subplots(2, num_show, figsize=(num_show * 2, 4))
    
    for i in range(num_show):
        # 原图
        img = images[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # 重建
        rec = reconstructions[i].permute(1, 2, 0).numpy()
        rec = (rec + 1) / 2
        axes[1, i].imshow(np.clip(rec, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 保存重建对比图: {save_path}")


def test_vf_alignment(model, data_root, split_file, device='cuda'):
    """测试Vision Foundation对齐能力 - VA-VAE核心创新"""
    
    print("\n🎨 Vision Foundation对齐能力验证...")
    
    model = model.to(device)
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 处理不同的数据结构格式
    if isinstance(split_data['val'], list):
        test_samples = split_data['val'][:20]
    else:
        test_samples = list(split_data['val'].values())[:20]
    vf_similarities, reconstruction_errors = [], []
    
    with torch.no_grad():
        processed_count = 0
        for idx, user_paths in enumerate(tqdm(test_samples, desc="VF对齐测试")):
            if isinstance(user_paths, list):
                num_to_select = min(2, len(user_paths), 20 - processed_count)
                selected_paths = user_paths[:num_to_select]
            else:
                selected_paths = [user_paths]
                
            for img_path_str in selected_paths:
                if processed_count >= 20:
                    break
                    
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                    
                img = Image.open(img_path).convert('RGB').resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                reconstructed, posterior, aux_feature, z = model(img_tensor)
                
                if hasattr(model, 'foundation_model'):
                    with torch.no_grad():
                        # 获取原始图像和重建图像的VF特征
                        orig_vf = model.foundation_model(img_tensor)
                        recon_vf = model.foundation_model(reconstructed)
                        
                    # 比较原始图像和重建图像的VF特征相似度
                    similarity = torch.cosine_similarity(
                        orig_vf.flatten(), recon_vf.flatten(), dim=0).item()
                    
                    vf_similarities.append(similarity)
                    reconstruction_errors.append(torch.mean((img_tensor - reconstructed) ** 2).item())
                    processed_count += 1
                    
            if processed_count >= 20:
                break
    
    avg_vf_similarity = np.mean(vf_similarities)
    
    print(f"✅ Vision Foundation对齐结果:")
    print(f"   平均VF语义相似度: {avg_vf_similarity:.4f}")
    print(f"   VF相似度标准差: {np.std(vf_similarities):.4f}")
    
    if avg_vf_similarity > 0.95:
        print(f"   🏆 VF对齐质量: 优秀 (>0.95)")
    elif avg_vf_similarity > 0.85:
        print(f"   ✅ VF对齐质量: 良好 (>0.85)")
    else:
        print(f"   ⚠️ VF对齐质量: 需要改进 (<0.85)")
    
    return avg_vf_similarity


def test_user_discrimination(model, data_root, split_file, device='cuda'):
    """测试用户区分能力 - VA-VAE Stage3创新"""
    
    print("\n👥 用户区分能力验证...")
    
    model = model.to(device)
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 使用验证集数据
    if isinstance(split_data['val'], dict):
        # 字典格式：键是用户ID
        val_data = split_data['val']
        user_items = [(uid, paths) for uid, paths in val_data.items()]
    else:
        # 列表格式：索引作为用户ID
        val_data = split_data['val']
        user_items = [(f"user_{idx}", paths) for idx, paths in enumerate(val_data)]
    
    user_features = {}
    
    with torch.no_grad():
        # 为每个用户提取特征
        for user_id, user_paths in tqdm(user_items[:31], desc="提取用户特征"):
            features = []
            
            if isinstance(user_paths, list):
                selected_paths = user_paths[:min(10, len(user_paths))]  # 每用户最多10张
            else:
                selected_paths = [user_paths]
            
            for img_path_str in selected_paths[:10]:  # 限制每用户样本数
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                    
                img = Image.open(img_path).convert('RGB').resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 编码获取特征
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                
                # 使用平均池化获取全局特征
                z_pooled = torch.mean(z, dim=[2, 3])  # [B, C]
                features.append(z_pooled.flatten().cpu().numpy())
            
            if features:
                user_features[user_id] = features
    
    # 计算分离度指标
    if len(user_features) > 1:
        # 准备数据用于聚类分析
        all_features = []
        all_labels = []
        
        for idx, (user_id, feats) in enumerate(user_features.items()):
            all_features.extend(feats)
            all_labels.extend([idx] * len(feats))
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # 计算Silhouette Score
        if len(np.unique(all_labels)) > 1:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(all_features, all_labels)
            print(f"✅ 用户分离度 (Silhouette): {silhouette:.4f}")
        else:
            silhouette = 0
            print("⚠️ 用户数不足，无法计算Silhouette分数")
        
        # 计算类间/类内距离比
        inter_distances = []
        intra_distances = []
        
        user_keys = list(user_features.keys())
        for i, user_i in enumerate(user_keys):
            user_i_feats = np.array(user_features[user_i])
            
            # 类内距离
            if len(user_i_feats) > 1:
                for j in range(len(user_i_feats)):
                    for k in range(j+1, len(user_i_feats)):
                        dist = np.linalg.norm(user_i_feats[j] - user_i_feats[k])
                        intra_distances.append(dist)
            
            # 类间距离（只计算部分避免计算量过大）
            for j in range(i+1, min(i+5, len(user_keys))):
                user_j = user_keys[j]
                user_j_feats = np.array(user_features[user_j])
                
                # 采样计算
                for feat_i in user_i_feats[:3]:
                    for feat_j in user_j_feats[:3]:
                        dist = np.linalg.norm(feat_i - feat_j)
                        inter_distances.append(dist)
        
        # 计算平均距离
        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0
        separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
        
        print(f"✅ 类间/类内距离比: {separation_ratio:.4f}")
        
        # 可视化
        visualize_user_distribution(user_features)
        
        return silhouette, separation_ratio
    
    return None, None


def visualize_user_distribution(user_features, save_path='user_distribution.png'):
    """可视化用户特征分布"""
    from sklearn.manifold import TSNE
    
    all_features = []
    all_labels = []
    
    for idx, (user_id, feats) in enumerate(user_features.items()):
        all_features.extend(feats)
        all_labels.extend([idx] * len(feats))
    
    all_features = np.array(all_features)
    
    if len(all_features) > 50:
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(all_features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=all_labels, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('用户特征分布 (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 用户分布图保存至: {save_path}")


def extract_latent_statistics(model, dataset_root, split_file, device='cuda'):
    """提取潜在空间统计信息"""
    print("\n📈 提取潜在空间统计...")
    
    model = model.to(device)
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 使用训练集数据
    if isinstance(split_data['train'], list):
        train_data = split_data['train']
    else:
        train_data = list(split_data['train'].values())
    
    all_latents = []
    
    with torch.no_grad():
        processed_count = 0
        
        for user_paths in tqdm(train_data, desc="编码图像"):
            if processed_count >= 500:
                break
                
            if isinstance(user_paths, list):
                selected_paths = user_paths[:min(5, len(user_paths))]  # 每用户最多5张
            else:
                selected_paths = [user_paths]
            
            for img_path_str in selected_paths:
                if processed_count >= 500:
                    break
                    
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                
                img = Image.open(img_path).convert('RGB').resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                
                all_latents.append(z.cpu())
                processed_count += 1
    
    # 计算统计
    all_latents = torch.cat(all_latents, dim=0)
    mean = all_latents.mean(dim=[0, 2, 3])
    std = all_latents.std(dim=[0, 2, 3])
    
    stats = {
        'global_mean': mean.numpy().tolist(),
        'global_std': std.numpy().tolist(),
        'num_samples': len(all_latents),
        'latent_dim': all_latents.shape[1],
        'spatial_size': [all_latents.shape[2], all_latents.shape[3]]
    }
    
    stats_path = 'latent_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 潜在空间统计已保存至: {stats_path}")
    print(f"   潜在维度: {stats['latent_dim']}")
    print(f"   空间尺寸: {stats['spatial_size']}")
    print(f"   样本数量: {stats['num_samples']}")
    
    return stats


def export_encoder_for_dit(model, checkpoint_path, output_path=None):
    """导出编码器供DiT训练使用"""
    
    print("\n🎯 导出编码器供DiT训练...")
    
    if output_path is None:
        checkpoint_name = Path(checkpoint_path).stem
        output_path = f"encoder_decoder_{checkpoint_name}.pt"
    
    # 提取编码器和解码器状态
    state_dict = model.state_dict()
    encoder_decoder_state = {k: v for k, v in state_dict.items() 
                             if 'encoder' in k or 'decoder' in k or 'quant' in k}
    
    # 保存配置信息
    config_info = {
        'embed_dim': getattr(model, 'embed_dim', 32),
        'z_channels': getattr(model, 'z_channels', 32),
        'use_vf': getattr(model, 'use_vf', 'dinov2'),
        'reverse_proj': getattr(model, 'reverse_proj', True),
        'resolution': 256,
        'type': 'vavae_encoder_decoder'
    }
    
    checkpoint = {
        'state_dict': encoder_decoder_state,
        'config': config_info
    }
    
    torch.save(checkpoint, output_path)
    print(f"✅ 编码器已导出至: {output_path}")
    
    # 检查文件大小
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   文件大小: {file_size:.2f} MB")
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VA-VAE模型验证和导出工具')
    
    # 路径参数
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/stage3/last.ckpt',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str,
                       default='checkpoints/stage3/config.yaml',
                       help='模型配置文件')
    parser.add_argument('--data_root', type=str,
                       default='/kaggle/input/dataset',
                       help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                       default='/kaggle/working/data_split/dataset_split.json',
                       help='数据划分文件')
    
    # 功能选择
    parser.add_argument('--validate', action='store_true',
                       help='验证重建质量')
    parser.add_argument('--vf_test', action='store_true',
                       help='测试VF对齐能力')
    parser.add_argument('--user_test', action='store_true',
                       help='测试用户区分能力')
    parser.add_argument('--extract_stats', action='store_true',
                       help='提取潜在空间统计')
    parser.add_argument('--export_dit', action='store_true',
                       help='导出DiT编码器')
    parser.add_argument('--comprehensive', action='store_true',
                       help='执行综合VA-VAE验证 (推荐)')
    parser.add_argument('--all', action='store_true',
                       help='执行所有功能')
    
    # Kaggle标志
    parser.add_argument('--kaggle', action='store_true',
                       help='Kaggle环境标志')
    
    args = parser.parse_args()
    
    # Kaggle环境检测
    if args.kaggle:
        kaggle_input = Path('/kaggle/input')
        kaggle_working = Path('/kaggle/working')
        if kaggle_input.exists():
            print("✅ 检测到Kaggle环境")
            # 查找checkpoint
            if (kaggle_working / 'checkpoints').exists():
                ckpt_dir = kaggle_working / 'checkpoints'
                # 查找最新阶段
                for stage in [3, 2, 1]:
                    stage_dir = ckpt_dir / f'stage{stage}'
                    if stage_dir.exists() and (stage_dir / 'last.ckpt').exists():
                        args.checkpoint = str(stage_dir / 'last.ckpt')
                        args.config = str(stage_dir / 'config.yaml')
                        print(f"使用第{stage}阶段checkpoint")
                        break
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 执行功能
    if args.comprehensive or (not any([args.validate, args.vf_test, args.user_test, 
                                      args.extract_stats, args.export_dit, args.all])):
        # 默认执行综合验证
        print("🚀 VA-VAE综合验证测试")
        print("="*60)
        
        results = {}
        
        # 基础重建验证
        mse, psnr = validate_reconstruction(model, args.data_root, args.split_file, device=device)
        results['mse'] = mse
        results['psnr'] = psnr
        
        # VA-VAE特有功能验证
        vf_score = test_vf_alignment(model, args.data_root, args.split_file, device)
        results['vf_alignment'] = vf_score
        
        silhouette, separation_ratio = test_user_discrimination(
            model, args.data_root, args.split_file, device)
        results['user_discrimination'] = silhouette
        results['feature_separation'] = separation_ratio
        
        # 潜在空间统计
        stats = extract_latent_statistics(model, args.data_root, args.split_file, device)
        results['latent_stats'] = stats
        
        # 导出DiT编码器
        export_encoder_for_dit(model, args.checkpoint, 'vavae_encoder_for_dit.pt')
        
        # 综合评估
        print(f"\n🏆 VA-VAE综合评估报告:")
        print(f"="*60)
        
        # 评分系统
        mse_grade = "A+" if mse < 0.005 else "A" if mse < 0.01 else "B+" if mse < 0.02 else "B"
        vf_grade = "A+" if vf_score > 0.95 else "A" if vf_score > 0.90 else "B+" if vf_score > 0.85 else "B"
        user_grade = "A+" if silhouette > 0.3 else "A" if silhouette > 0.2 else "B+" if silhouette > 0.1 else "B"
        sep_grade = "A+" if separation_ratio > 2.0 else "A" if separation_ratio > 1.5 else "B+" if separation_ratio > 1.2 else "B"
        
        print(f"📊 重建质量 (MSE): {mse:.6f} (等级: {mse_grade})")
        print(f"📊 重建质量 (PSNR): {psnr:.2f} dB")
        print(f"🎨 Vision Foundation对齐: {vf_score:.4f} (等级: {vf_grade})")
        print(f"👥 用户区分能力: {silhouette:.4f} (等级: {user_grade})")
        print(f"🎯 特征分离度: {separation_ratio:.4f} (等级: {sep_grade})")
        
        # 整体评价
        grades = [mse_grade, vf_grade, user_grade, sep_grade]
        grade_scores = {"A+": 4, "A": 3, "B+": 2, "B": 1, "C": 0}
        avg_score = np.mean([grade_scores[g] for g in grades])
        
        if avg_score >= 3.5:
            overall = "优秀 - 完全胜任微多普勒用户区分任务"
        elif avg_score >= 2.5:
            overall = "良好 - 基本胜任，有改进空间"
        else:
            overall = "一般 - 需要进一步优化"
        
        print(f"\n🎖️ 整体评价: {overall}")
        print(f"="*60)
        
    else:
        # 分别执行指定功能
        if args.all:
            args.validate = args.vf_test = args.user_test = True
            args.extract_stats = args.export_dit = True
        
        if args.validate:
            validate_reconstruction(model, args.data_root, args.split_file, device=device)
        
        if args.vf_test:
            test_vf_alignment(model, args.data_root, args.split_file, device)
        
        if args.user_test:
            test_user_discrimination(model, args.data_root, args.split_file, device)
        
        if args.extract_stats:
            extract_latent_statistics(model, args.data_root, args.split_file, device=device)
        
        if args.export_dit:
            output_path = 'vavae_encoder_for_dit.pt'
            export_encoder_for_dit(model, args.checkpoint, output_path)
    
    print("\n✅ 所有验证任务完成!")


if __name__ == '__main__':
    main()
