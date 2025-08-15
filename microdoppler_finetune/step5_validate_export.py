#!/usr/bin/env python
"""VA-VAE模型验证与导出脚本 - 修复版"""

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
                    'monitor': 'val/rec_loss',
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


def validate_reconstruction(model, dataset_root, split_file, num_samples=16, device='cuda'):
    """验证重建质量"""
    print("\n🔍 重建质量验证...")
    
    # 加载数据划分
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 处理不同的数据结构格式
    if isinstance(split_data['val'], list):
        val_data = split_data['val']
    else:
        val_data = list(split_data['val'].values())
    
    # 检查数据格式
    if val_data and len(val_data) > 0:
        print(f"📊 验证数据用户数: {len(val_data)}")
        print(f"📊 第一个用户的图片数: {len(val_data[0]) if isinstance(val_data[0], list) else 1}")
    
    images = []
    reconstructions = []
    
    model = model.to(device)
    
    with torch.no_grad():
        processed_count = 0
        
        for user_idx, user_paths in enumerate(tqdm(val_data, desc="处理用户")):
            if processed_count >= num_samples:
                break
                
            # 每个user_paths是一个用户的图片路径列表
            if isinstance(user_paths, list):
                # 计算该用户要选择的图片数
                remaining = num_samples - processed_count
                num_to_select = min(2, len(user_paths), remaining)
                selected_paths = user_paths[:num_to_select]
            else:
                selected_paths = [user_paths]
                
            for img_path_str in selected_paths:
                if processed_count >= num_samples:
                    break
                    
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                    
                # 加载和预处理图像
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256), Image.LANCZOS)
                
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 编码和解码
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                reconstructed = model.decode(z)
                
                images.append(img_tensor.cpu())
                reconstructions.append(reconstructed.cpu())
                processed_count += 1
    
    # 计算指标
    if images:
        images_tensor = torch.cat(images, dim=0)
        recons_tensor = torch.cat(reconstructions, dim=0)
        
        mse = torch.mean((images_tensor - recons_tensor) ** 2).item()
        psnr = 20 * np.log10(2.0) - 10 * np.log10(mse)
        
        print(f"✅ 重建指标:")
        print(f"   MSE: {mse:.6f}")
        print(f"   PSNR: {psnr:.2f} dB")
        
        # 保存对比图
        save_reconstruction_comparison(images_tensor, recons_tensor)
        
        return mse, psnr
    
    return None, None


def test_vf_alignment(model, dataset_root, split_file, device='cuda'):
    """测试Vision Foundation对齐"""
    print("\n🎨 Vision Foundation对齐能力验证...")
    
    model = model.to(device)
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 处理数据格式
    if isinstance(split_data['val'], list):
        test_samples = split_data['val'][:20]
    else:
        test_samples = list(split_data['val'].values())[:20]
    
    vf_similarities = []
    
    with torch.no_grad():
        processed_count = 0
        
        for user_idx, user_paths in enumerate(tqdm(test_samples, desc="VF对齐测试")):
            if processed_count >= 20:
                break
                
            if isinstance(user_paths, list):
                remaining = 20 - processed_count
                num_to_select = min(2, len(user_paths), remaining)
                selected_paths = user_paths[:num_to_select]
            else:
                selected_paths = [user_paths]
                
            for img_path_str in selected_paths:
                if processed_count >= 20:
                    break
                    
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                    
                # 加载图像
                img = Image.open(img_path).convert('RGB').resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 通过模型
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                reconstructed = model.decode(z)
                
                # 如果模型有VF功能，计算相似度
                if hasattr(model, 'foundation_model') and model.foundation_model is not None:
                    # 比较原始图像和重建图像的VF特征
                    orig_vf = model.foundation_model(img_tensor)
                    recon_vf = model.foundation_model(reconstructed)
                    
                    similarity = torch.cosine_similarity(
                        orig_vf.flatten(), 
                        recon_vf.flatten(), 
                        dim=0
                    ).item()
                    
                    vf_similarities.append(similarity)
                    processed_count += 1
    
    # 输出结果
    if vf_similarities:
        avg_similarity = np.mean(vf_similarities)
        std_similarity = np.std(vf_similarities)
        
        print(f"✅ Vision Foundation对齐结果:")
        print(f"   平均VF语义相似度: {avg_similarity:.4f}")
        print(f"   VF相似度标准差: {std_similarity:.4f}")
        
        if avg_similarity > 0.95:
            print(f"   🏆 VF对齐质量: 优秀 (>0.95)")
        elif avg_similarity > 0.85:
            print(f"   ✅ VF对齐质量: 良好 (>0.85)")
        else:
            print(f"   ⚠️ VF对齐质量: 需要改进 (<0.85)")
        
        return avg_similarity
    else:
        print("⚠️ 模型不支持VF功能或未找到VF模块")
        return None


def test_user_discrimination(model, dataset_root, split_file, device='cuda'):
    """测试用户区分能力"""
    print("\n👥 用户区分能力验证...")
    
    model = model.to(device)
    model.eval()
    
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
                
                # 加载图像
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


def save_reconstruction_comparison(originals, reconstructions, save_path='reconstruction_results.png'):
    """保存重建对比图"""
    n_samples = min(8, len(originals))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    for i in range(n_samples):
        # 原图
        img = originals[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # 重建图
        rec = reconstructions[i].permute(1, 2, 0).numpy()
        rec = (rec + 1) / 2
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 保存重建对比图: {save_path}")


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
                selected_paths = user_paths[:min(5, len(user_paths))]
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


def export_for_dit(model, save_path='vae_encoder_for_dit.pt'):
    """导出编码器用于DiT训练"""
    print("\n💾 导出编码器用于DiT训练...")
    
    # 只保存编码器和解码器的权重
    encoder_decoder_state = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'quant_conv': model.quant_conv.state_dict(),
        'post_quant_conv': model.post_quant_conv.state_dict(),
    }
    
    # 保存配置信息
    config_info = {
        'embed_dim': model.embed_dim if hasattr(model, 'embed_dim') else 32,
        'z_channels': model.encoder.z_channels if hasattr(model.encoder, 'z_channels') else 32,
    }
    
    checkpoint = {
        'state_dict': encoder_decoder_state,
        'config': config_info
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ 编码器已导出至: {save_path}")
    
    return save_path


def comprehensive_evaluation(model, dataset_root, split_file, device='cuda'):
    """综合评估并生成报告"""
    print("\n" + "="*50)
    print("📊 VA-VAE Stage 3 综合评估报告")
    print("="*50)
    
    scores = {}
    
    # 1. 重建质量
    mse, psnr = validate_reconstruction(model, dataset_root, split_file, device=device)
    if psnr:
        scores['reconstruction'] = min(100, psnr / 30 * 100)  # PSNR 30dB = 100分
    
    # 2. VF对齐
    vf_sim = test_vf_alignment(model, dataset_root, split_file, device=device)
    if vf_sim:
        scores['vf_alignment'] = vf_sim * 100
    
    # 3. 用户区分
    silhouette, sep_ratio = test_user_discrimination(model, dataset_root, split_file, device=device)
    if silhouette is not None:
        scores['user_discrimination'] = max(0, (silhouette + 1) * 50)  # [-1,1] -> [0,100]
    
    # 4. 潜在空间统计
    stats = extract_latent_statistics(model, dataset_root, split_file, device=device)
    
    # 5. 导出模型
    export_path = export_for_dit(model)
    
    # 计算总分
    if scores:
        total_score = np.mean(list(scores.values()))
        
        print("\n" + "="*50)
        print("📊 评估结果汇总")
        print("="*50)
        
        for key, score in scores.items():
            print(f"   {key}: {score:.2f}/100")
        
        print(f"\n📊 综合得分: {total_score:.2f}/100")
        
        # 评级
        if total_score >= 90:
            grade = "A+ (卓越)"
        elif total_score >= 85:
            grade = "A (优秀)"
        elif total_score >= 80:
            grade = "A- (良好)"
        elif total_score >= 75:
            grade = "B+ (合格)"
        else:
            grade = "B (需改进)"
        
        print(f"🏆 最终评级: {grade}")
        print("="*50)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='VA-VAE验证与导出')
    parser.add_argument('--checkpoint', type=str, 
                       default='/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt',
                       help='模型checkpoint路径')
    parser.add_argument('--dataset_root', type=str, 
                       default='/kaggle/input/dataset',
                       help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                       default='/kaggle/working/data_split/dataset_split.json',
                       help='数据划分文件')
    parser.add_argument('--device', type=str, default='cuda')
    
    # 功能选择
    parser.add_argument('--validate', action='store_true', help='验证重建质量')
    parser.add_argument('--vf_test', action='store_true', help='测试VF对齐')
    parser.add_argument('--user_test', action='store_true', help='测试用户区分')
    parser.add_argument('--extract_stats', action='store_true', help='提取潜在统计')
    parser.add_argument('--export_dit', action='store_true', help='导出DiT编码器')
    parser.add_argument('--comprehensive', action='store_true', help='综合评估(推荐)')
    parser.add_argument('--all', action='store_true', help='运行所有功能')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.checkpoint, device=args.device)
    
    # 执行选定的功能
    if args.comprehensive or args.all:
        comprehensive_evaluation(model, args.dataset_root, args.split_file, args.device)
    else:
        if args.validate:
            validate_reconstruction(model, args.dataset_root, args.split_file, device=args.device)
        
        if args.vf_test:
            test_vf_alignment(model, args.dataset_root, args.split_file, device=args.device)
        
        if args.user_test:
            test_user_discrimination(model, args.dataset_root, args.split_file, device=args.device)
        
        if args.extract_stats:
            extract_latent_statistics(model, args.dataset_root, args.split_file, device=args.device)
        
        if args.export_dit:
            export_for_dit(model)
    
    print("\n✅ 验证完成!")


if __name__ == '__main__':
    main()
