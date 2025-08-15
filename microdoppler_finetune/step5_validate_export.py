#!/usr/bin/env python
"""
VA-VAE模型综合验证脚本
集成所有验证功能于一个文件
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import json
import argparse
from omegaconf import OmegaConf
from datetime import datetime

# 添加LightningDiT路径
if os.path.exists('/kaggle/working'):
    sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT/vavae')
    sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT')
else:
    vavae_path = Path(__file__).parent / 'LightningDiT' / 'vavae'
    if vavae_path.exists():
        sys.path.insert(0, str(vavae_path))
        sys.path.insert(0, str(vavae_path.parent))

# 导入模型模块
try:
    from ldm.models.autoencoder import AutoencoderKL
    from ldm.util import instantiate_from_config
    print("✅ 成功导入VA-VAE模型模块")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保LightningDiT/vavae项目在正确位置")
    sys.exit(1)


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的VA-VAE模型"""
    print(f"\n📂 Loading model from: {checkpoint_path}")
    
    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if 'config' in ckpt:
        config = ckpt['config']
        if isinstance(config, dict):
            config = OmegaConf.create(config)
    else:
        # 默认配置
        config = OmegaConf.create({
            'target': 'ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 32,
                'use_vf': 'dinov2',
                'ddconfig': {
                    'double_z': False,
                    'z_channels': 32,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                }
            }
        })
    
    # 实例化模型
    if hasattr(config, 'target'):
        model = instantiate_from_config(config)
    else:
        model = instantiate_from_config(config.model if hasattr(config, 'model') else config)
    
    # 加载权重
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


def test_reconstruction(model, data_root, num_samples=30, device='cuda'):
    """测试重建质量"""
    print("\n" + "="*50)
    print("🔍 Testing Reconstruction Quality")
    print("="*50)
    
    data_path = Path(data_root)
    mse_scores = []
    psnr_scores = []
    
    sample_count = 0
    pbar = tqdm(total=num_samples, desc="Processing images")
    
    for user_id in range(1, 32):
        if sample_count >= num_samples:
            break
            
        user_folder = data_path / f"ID_{user_id}"
        if not user_folder.exists():
            continue
            
        images = list(user_folder.glob("*.jpg"))
        if images:
            img_path = images[0]  # 取第一张
            
            # 加载和预处理
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 编码-解码
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                rec = model.decode(z)
                
                # 计算MSE和PSNR
                mse = F.mse_loss(rec, img_tensor).item()
                mse_scores.append(mse)
                psnr = 10 * np.log10(4.0 / mse)  # 范围[-1, 1]
                psnr_scores.append(psnr)
            
            sample_count += 1
            pbar.update(1)
    
    pbar.close()
    
    # 统计结果
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    avg_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    
    print(f"📊 Results:")
    print(f"  MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    
    # 评级
    if avg_mse < 0.01:
        grade = "Excellent ⭐⭐⭐"
    elif avg_mse < 0.02:
        grade = "Good ⭐⭐"
    else:
        grade = "Fair ⭐"
    print(f"  Grade: {grade}")
    
    return {
        'mse': avg_mse,
        'mse_std': std_mse,
        'psnr': avg_psnr,
        'psnr_std': std_psnr,
        'grade': grade
    }


def test_vf_alignment(model, data_root, num_samples=20, device='cuda'):
    """测试VF对齐"""
    print("\n" + "="*50)
    print("🔍 Testing Vision Foundation Alignment")
    print("="*50)
    
    # 检查VF模型
    if not hasattr(model, 'vf_model') or model.vf_model is None:
        print("⚠️ No VF model found, skipping test")
        return None
    
    data_path = Path(data_root)
    similarities = []
    
    sample_count = 0
    pbar = tqdm(total=num_samples, desc="Processing VF alignment")
    
    for user_id in range(1, 32):
        if sample_count >= num_samples:
            break
            
        user_folder = data_path / f"ID_{user_id}"
        if not user_folder.exists():
            continue
            
        images = list(user_folder.glob("*.jpg"))
        if images:
            img_path = images[0]
            
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 原始图像的VF特征
                vf_input = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                vf_input = (vf_input + 1.0) / 2.0
                orig_features = model.vf_model.forward_features(vf_input)['x_norm_clstoken']
                
                # 重建图像
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                rec = model.decode(z)
                
                # 重建图像的VF特征
                rec_vf_input = F.interpolate(rec, size=(224, 224), mode='bilinear', align_corners=False)
                rec_vf_input = (rec_vf_input + 1.0) / 2.0
                rec_features = model.vf_model.forward_features(rec_vf_input)['x_norm_clstoken']
                
                # 计算余弦相似度
                cos_sim = F.cosine_similarity(orig_features, rec_features, dim=1).mean().item()
                similarities.append(cos_sim)
            
            sample_count += 1
            pbar.update(1)
    
    pbar.close()
    
    if similarities:
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        print(f"📊 Results:")
        print(f"  VF Similarity: {avg_sim:.4f} ± {std_sim:.4f}")
        
        # 评级
        if avg_sim > 0.95:
            grade = "Excellent ⭐⭐⭐"
        elif avg_sim > 0.90:
            grade = "Good ⭐⭐"
        else:
            grade = "Fair ⭐"
        print(f"  Grade: {grade}")
        
        return {
            'similarity': avg_sim,
            'similarity_std': std_sim,
            'grade': grade
        }
    
    return None


def test_user_discrimination(model, data_root, samples_per_user=10, device='cuda'):
    """测试用户区分能力"""
    print("\n" + "="*50)
    print("🔍 Testing User Discrimination")
    print("="*50)
    
    data_path = Path(data_root)
    user_features = {}
    all_features = []
    all_labels = []
    
    # 提取每个用户的特征
    for user_id in tqdm(range(1, 32), desc="Extracting user features"):
        user_folder = data_path / f"ID_{user_id}"
        if not user_folder.exists():
            continue
        
        features = []
        images = list(user_folder.glob("*.jpg"))[:samples_per_user]
        
        for img_path in images:
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 获取潜在特征
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                
                # 全局池化
                z_pooled = z.mean(dim=[2, 3]).cpu().numpy().flatten()
                features.append(z_pooled)
                all_features.append(z_pooled)
                all_labels.append(user_id)
        
        if features:
            user_features[user_id] = np.mean(features, axis=0)
    
    if len(user_features) < 2:
        print("⚠️ Not enough users for discrimination analysis")
        return None
    
    # 计算Silhouette分数
    all_features_array = np.array(all_features)
    all_labels_array = np.array(all_labels)
    
    silhouette = 0
    if len(np.unique(all_labels_array)) > 1:
        silhouette = silhouette_score(all_features_array, all_labels_array)
    
    # 计算类间/类内距离比
    intra_distances = []
    inter_distances = []
    
    # 类间距离
    users = list(user_features.keys())
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            feat_i = user_features[users[i]]
            feat_j = user_features[users[j]]
            dist = np.linalg.norm(feat_i - feat_j)
            inter_distances.append(dist)
    
    # 类内距离
    for user_id in np.unique(all_labels_array):
        user_mask = all_labels_array == user_id
        user_feats = all_features_array[user_mask]
        if len(user_feats) > 1:
            for i in range(len(user_feats)):
                for j in range(i+1, len(user_feats)):
                    dist = np.linalg.norm(user_feats[i] - user_feats[j])
                    intra_distances.append(dist)
    
    avg_intra = np.mean(intra_distances) if intra_distances else 0
    avg_inter = np.mean(inter_distances) if inter_distances else 0
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    print(f"📊 Results:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Intra-class distance: {avg_intra:.4f}")
    print(f"  Inter-class distance: {avg_inter:.4f}")
    print(f"  Ratio (inter/intra): {ratio:.4f}")
    print(f"  Number of users: {len(user_features)}")
    
    # 评级
    if ratio > 2.0:
        grade = "Excellent ⭐⭐⭐"
    elif ratio > 1.5:
        grade = "Good ⭐⭐"
    else:
        grade = "Fair ⭐"
    print(f"  Grade: {grade}")
    
    return {
        'silhouette_score': silhouette,
        'intra_distance': avg_intra,
        'inter_distance': avg_inter,
        'ratio': ratio,
        'num_users': len(user_features),
        'grade': grade
    }


def extract_latent_statistics(model, data_root, num_samples=100, device='cuda'):
    """提取潜在空间统计"""
    print("\n" + "="*50)
    print("📊 Extracting Latent Space Statistics")
    print("="*50)
    
    data_path = Path(data_root)
    all_latents = []
    
    sample_count = 0
    for user_id in tqdm(range(1, 32), desc="Processing latents"):
        if sample_count >= num_samples:
            break
            
        user_folder = data_path / f"ID_{user_id}"
        if not user_folder.exists():
            continue
            
        images = list(user_folder.glob("*.jpg"))[:5]
        
        for img_path in images:
            if sample_count >= num_samples:
                break
                
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                posterior = model.encode(img_tensor)
                z = posterior.sample()
                all_latents.append(z.cpu().numpy())
            
            sample_count += 1
    
    # 计算统计
    all_latents = np.concatenate(all_latents, axis=0)
    mean = np.mean(all_latents)
    std = np.std(all_latents)
    
    print(f"  Latent mean: {mean:.6f}")
    print(f"  Latent std: {std:.6f}")
    print(f"  Latent shape: {all_latents.shape}")
    
    # 评估
    if abs(mean) < 0.1 and 0.8 < std < 1.2:
        status = "✅ Well-normalized"
    else:
        status = "⚠️ Needs regularization"
    print(f"  Status: {status}")
    
    return {
        'mean': float(mean),
        'std': float(std),
        'shape': list(all_latents.shape),
        'status': status
    }


def export_encoder_decoder(model, checkpoint_path):
    """导出编码器和解码器"""
    print("\n" + "="*50)
    print("💾 Exporting Encoder and Decoder")
    print("="*50)
    
    # 导出编码器
    encoder_path = checkpoint_path.replace('.pt', '_encoder.pt')
    encoder_state = {
        'encoder': model.encoder.state_dict(),
        'quant_conv': model.quant_conv.state_dict() if hasattr(model, 'quant_conv') else None,
        'embed_dim': model.embed_dim if hasattr(model, 'embed_dim') else 32,
        'config': {
            'z_channels': 32,
            'resolution': 256,
            'ch_mult': [1, 1, 2, 2, 4]
        }
    }
    torch.save(encoder_state, encoder_path)
    print(f"✅ Encoder exported to: {encoder_path}")
    
    # 导出解码器
    decoder_path = checkpoint_path.replace('.pt', '_decoder.pt')
    decoder_state = {
        'decoder': model.decoder.state_dict(),
        'post_quant_conv': model.post_quant_conv.state_dict() if hasattr(model, 'post_quant_conv') else None,
        'embed_dim': model.embed_dim if hasattr(model, 'embed_dim') else 32
    }
    torch.save(decoder_state, decoder_path)
    print(f"✅ Decoder exported to: {decoder_path}")
    
    return encoder_path, decoder_path


def generate_report(results, checkpoint_path):
    """生成综合报告"""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE VALIDATION REPORT")
    print("="*60)
    
    # 重建质量
    if 'reconstruction' in results:
        rec = results['reconstruction']
        print(f"\n1️⃣ Reconstruction Quality:")
        print(f"   MSE: {rec['mse']:.6f} ± {rec['mse_std']:.6f}")
        print(f"   PSNR: {rec['psnr']:.2f} ± {rec['psnr_std']:.2f} dB")
        print(f"   {rec['grade']}")
    
    # VF对齐
    if 'vf_alignment' in results and results['vf_alignment']:
        vf = results['vf_alignment']
        print(f"\n2️⃣ Vision Foundation Alignment:")
        print(f"   Similarity: {vf['similarity']:.4f} ± {vf['similarity_std']:.4f}")
        print(f"   {vf['grade']}")
    
    # 用户区分
    if 'user_discrimination' in results and results['user_discrimination']:
        disc = results['user_discrimination']
        print(f"\n3️⃣ User Discrimination:")
        print(f"   Silhouette Score: {disc['silhouette_score']:.4f}")
        print(f"   Inter/Intra Ratio: {disc['ratio']:.4f}")
        print(f"   {disc['grade']}")
    
    # 潜在统计
    if 'latent_statistics' in results:
        lat = results['latent_statistics']
        print(f"\n4️⃣ Latent Space Statistics:")
        print(f"   Mean: {lat['mean']:.6f}, Std: {lat['std']:.6f}")
        print(f"   {lat['status']}")
    
    # 总体评估
    print("\n" + "="*60)
    print("🎯 OVERALL ASSESSMENT")
    print("="*60)
    
    rec_ok = results.get('reconstruction', {}).get('mse', 1.0) < 0.02
    disc_ok = results.get('user_discrimination', {}).get('ratio', 0) > 1.5
    
    if rec_ok and disc_ok:
        print("✅ Model is ready for DiT training!")
        print("   Both reconstruction and user discrimination meet requirements")
    elif rec_ok:
        print("⚠️ Model has good reconstruction but weak user discrimination")
        print("   Consider training with stronger user contrastive loss")
    elif disc_ok:
        print("⚠️ Model has good user discrimination but poor reconstruction")
        print("   Consider adjusting reconstruction loss weights")
    else:
        print("❌ Model needs more training")
        print("   Both metrics need improvement")
    
    # 保存JSON报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"validation_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Report saved to: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='VA-VAE Model Validation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/kaggle/input/dataset',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--test_reconstruction', action='store_true', default=True,
                       help='Test reconstruction quality')
    parser.add_argument('--test_vf', action='store_true',
                       help='Test VF alignment')
    parser.add_argument('--test_discrimination', action='store_true',
                       help='Test user discrimination')
    parser.add_argument('--test_latents', action='store_true',
                       help='Extract latent statistics')
    parser.add_argument('--export_models', action='store_true',
                       help='Export encoder/decoder for DiT')
    parser.add_argument('--full_test', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    # 如果指定full_test，启用所有测试
    if args.full_test:
        args.test_vf = True
        args.test_discrimination = True
        args.test_latents = True
        args.export_models = True
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 运行测试
    results = {}
    
    # 1. 重建质量测试（默认开启）
    if args.test_reconstruction:
        results['reconstruction'] = test_reconstruction(
            model, args.data_root, device=args.device
        )
    
    # 2. VF对齐测试
    if args.test_vf:
        results['vf_alignment'] = test_vf_alignment(
            model, args.data_root, device=args.device
        )
    
    # 3. 用户区分测试
    if args.test_discrimination:
        results['user_discrimination'] = test_user_discrimination(
            model, args.data_root, device=args.device
        )
    
    # 4. 潜在空间统计
    if args.test_latents:
        results['latent_statistics'] = extract_latent_statistics(
            model, args.data_root, device=args.device
        )
    
    # 5. 导出模型
    if args.export_models:
        encoder_path, decoder_path = export_encoder_decoder(model, args.checkpoint)
        results['exported_models'] = {
            'encoder': encoder_path,
            'decoder': decoder_path
        }
    
    # 生成报告
    generate_report(results, args.checkpoint)
    
    # 使用提示
    if not args.full_test:
        print("\n💡 Tips:")
        print("  • Use --full_test to run all validation tests")
        print("  • Use --test_vf to test VF alignment")
        print("  • Use --test_discrimination to test user discrimination")
        print("  • Use --export_models to export for DiT training")


if __name__ == '__main__':
    main()
