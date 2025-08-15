#!/usr/bin/env python3
"""
Step 5: VA-VAE完整验证脚本 (集成版)
包含基础验证、VA-VAE特有功能验证和模型导出
"""

import os
import sys
import argparse
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# 添加LightningDiT路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'LightningDiT' / 'vavae'))
sys.path.insert(0, str(project_root / 'LightningDiT'))
sys.path.insert(0, str(project_root))  # 添加根目录

# 关键：在导入ldm之前设置taming路径！
def setup_taming_path():
    """设置taming路径，必须在导入ldm之前调用"""
    # 按优先级检查taming位置
    taming_locations = [
        Path('/kaggle/working/taming-transformers'),  # Kaggle标准位置
        Path('/kaggle/working/.taming_path'),  # 路径文件
        Path.cwd().parent / 'taming-transformers',  # 项目根目录
        Path.cwd() / '.taming_path'  # 当前目录路径文件
    ]
    
    for location in taming_locations:
        if location.name == '.taming_path' and location.exists():
            # 读取路径文件
            try:
                with open(location, 'r') as f:
                    taming_path = f.read().strip()
                if Path(taming_path).exists() and taming_path not in sys.path:
                    sys.path.insert(0, taming_path)
                    print(f"📂 已加载taming路径: {taming_path}")
                    return True
            except Exception as e:
                continue
        elif location.name == 'taming-transformers' and location.exists():
            # 直接路径
            taming_path = str(location.absolute())
            if taming_path not in sys.path:
                sys.path.insert(0, taming_path)
                print(f"📂 发现并加载taming: {taming_path}")
                return True
    
    # 静默失败，因为可能已经通过其他方式加载
    return False

# 设置taming路径（必须在导入ldm之前）
setup_taming_path()

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model(checkpoint_path, config_path=None):
    """加载训练好的模型"""
    
    print(f"📦 加载模型: {checkpoint_path}")
    
    # 加载配置
    if config_path and Path(config_path).exists():
        config = OmegaConf.load(config_path)
    else:
        # 使用默认配置
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
                            'vf_weight': 0.1,
                            'distmat_weight': 1.0,
                            'cos_weight': 1.0
                        }
                    }
                }
            }
        })
    
    # 实例化模型
    model = instantiate_from_config(config if isinstance(config, dict) else config.model)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
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
        for item in tqdm(val_data, desc="处理图像"):
            # 加载图像
            img_path = Path(data_root) / item['path']
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
        for item in tqdm(test_samples, desc="VF对齐测试"):
            img_path = Path(data_root) / item['path']
            if not img_path.exists():
                continue
                
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            reconstructed, posterior, aux_feature, z = model(img_tensor)
            
            if aux_feature is not None and hasattr(model, 'foundation_model'):
                with torch.no_grad():
                    recon_vf = model.foundation_model(reconstructed)
                    
                similarity = torch.cosine_similarity(
                    aux_feature.flatten(), recon_vf.flatten(), dim=0).item()
                
                vf_similarities.append(similarity)
                reconstruction_errors.append(torch.mean((img_tensor - reconstructed) ** 2).item())
    
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
    
    # 处理不同的数据结构格式
    if isinstance(split_data['train'], list):
        train_samples = split_data['train'][:300]
    else:
        train_samples = list(split_data['train'].values())[:300]
    
    user_features, user_labels, all_features = {}, [], []
    
    with torch.no_grad():
        for item in tqdm(train_samples, desc="提取用户特征"):
            img_path = Path(data_root) / item['path']
            if not img_path.exists():
                continue
                
            user_id = item['user_id']
            
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            posterior = model.encode(img_tensor)
            z = posterior.sample().cpu().numpy().flatten()
            
            if user_id not in user_features:
                user_features[user_id] = []
            user_features[user_id].append(z)
            all_features.append(z)
            user_labels.append(user_id)
    
    all_features = np.array(all_features)
    
    # 计算分离度
    silhouette = 0
    if len(set(user_labels)) > 1:
        silhouette = silhouette_score(all_features, user_labels)
        print(f"✅ 用户分离度 (Silhouette): {silhouette:.4f}")
    
    # 类内外距离比
    intra_distances, inter_distances = [], []
    user_means = {}
    
    for user_id, features in user_features.items():
        if len(features) > 1:
            features_array = np.array(features)
            user_mean = np.mean(features_array, axis=0)
            user_means[user_id] = user_mean
            
            for feat in features_array:
                intra_distances.append(np.linalg.norm(feat - user_mean))
    
    user_ids = list(user_means.keys())
    for i in range(len(user_ids)):
        for j in range(i+1, len(user_ids)):
            distance = np.linalg.norm(user_means[user_ids[i]] - user_means[user_ids[j]])
            inter_distances.append(distance)
    
    avg_intra = np.mean(intra_distances) if intra_distances else 0
    avg_inter = np.mean(inter_distances) if inter_distances else 0
    separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    print(f"✅ 类间/类内距离比: {separation_ratio:.4f}")
    
    # 生成t-SNE可视化
    if len(all_features) > 50:
        visualize_user_distribution(user_features)
    
    return silhouette, separation_ratio, user_features


def visualize_user_distribution(user_features, save_path='user_distribution.png'):
    """可视化用户特征分布"""
    
    print("\n📊 生成用户分布可视化...")
    
    all_features, all_labels = [], []
    for user_id, features in user_features.items():
        for feat in features:
            all_features.append(feat)
            all_labels.append(user_id)
    
    all_features = np.array(all_features)
    
    if len(all_features) > 50:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
        features_2d = tsne.fit_transform(all_features)
        
        plt.figure(figsize=(12, 8))
        unique_users = list(set(all_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_users)))
        
        for i, user_id in enumerate(unique_users):
            mask = np.array(all_labels) == user_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'User {user_id}', alpha=0.7)
        
        plt.title('VA-VAE潜在空间中的用户分布 (t-SNE)')
        plt.xlabel('t-SNE维度 1')
        plt.ylabel('t-SNE维度 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 用户分布图保存至: {save_path}")


def extract_latent_statistics(model, data_root, split_file, device='cuda'):
    """提取潜在空间统计信息"""
    
    print("\n📈 提取潜在空间统计...")
    
    # 加载数据
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # 处理不同的数据结构格式
    if isinstance(split_data['train'], list):
        train_data = split_data['train']
    else:
        train_data = list(split_data['train'].values())
    
    model = model.to(device)
    
    all_latents = []
    
    with torch.no_grad():
        for item in tqdm(train_data, desc="编码图像"):
            img_path = Path(data_root) / item['path']
            if not img_path.exists():
                continue
            
            img = Image.open(img_path).convert('RGB').resize((256, 256), Image.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 编码
            posterior = model.encode(img_tensor)
            z = posterior.sample()
            
            all_latents.append(z.cpu())
            
            # 按用户分组
            user_id = item['user_id']
            if user_id not in user_latents:
                user_latents[user_id] = []
            user_latents[user_id].append(z.cpu())
    
    # 计算全局统计
    all_latents = torch.cat(all_latents, dim=0)
    mean = all_latents.mean(dim=[0, 2, 3])  # [C]
    std = all_latents.std(dim=[0, 2, 3])    # [C]
    
    # 计算用户间差异
    user_means = {}
    for user_id, latents in user_latents.items():
        user_latents_cat = torch.cat(latents, dim=0)
        user_means[user_id] = user_latents_cat.mean(dim=[0, 2, 3])
    
    # 保存统计信息
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
    
    print(f"✅ 潜在空间统计:")
    print(f"   维度: {stats['latent_dim']} x {stats['spatial_size'][0]} x {stats['spatial_size'][1]}")
    print(f"   样本数: {stats['num_samples']}")
    print(f"   均值范围: [{min(mean):.3f}, {max(mean):.3f}]")
    print(f"   标准差范围: [{min(std):.3f}, {max(std):.3f}]")
    print(f"📊 统计信息保存至: {stats_path}")
    
    return stats


def export_for_dit(checkpoint_path, output_path):
    """导出模型用于DiT训练"""
    
    print(f"\n📦 导出模型用于DiT训练...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取必要的组件
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # 只保留编码器和量化层（DiT只需要编码器）
    encoder_keys = [k for k in state_dict.keys() if 
                   k.startswith('encoder.') or 
                   k.startswith('quant_conv.') or
                   k.startswith('linear_proj.')]
    
    encoder_state = {k: state_dict[k] for k in encoder_keys}
    
    # 保存DiT版本
    dit_checkpoint = {
        'encoder_state_dict': encoder_state,
        'full_state_dict': state_dict,  # 也保留完整版本
        'config': {
            'embed_dim': 32,
            'z_channels': 32,
            'use_vf': 'dinov2',
            'reverse_proj': True,
            'resolution': 256
        },
        'type': 'vavae_encoder_for_dit'
    }
    
    torch.save(dit_checkpoint, output_path)
    print(f"✅ DiT编码器导出至: {output_path}")
    
    # 检查文件大小
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   文件大小: {file_size:.2f} MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    
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
    model = load_model(args.checkpoint, args.config)
    
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
        
        silhouette, separation_ratio, user_features = test_user_discrimination(
            model, args.data_root, args.split_file, device)
        results['user_discrimination'] = silhouette
        results['feature_separation'] = separation_ratio
        
        # 潜在空间统计
        stats = extract_latent_statistics(model, args.data_root, args.split_file, device)
        results['latent_stats'] = stats
        
        # 导出DiT编码器
        export_for_dit(args.checkpoint, 'vavae_encoder_for_dit.pt')
        
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
            export_for_dit(args.checkpoint, output_path)
    
    print("\n✅ 所有验证任务完成!")


if __name__ == '__main__':
    main()
