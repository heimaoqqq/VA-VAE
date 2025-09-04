"""
分析是否需要重头训练VA-VAE和VFM的诊断脚本
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torchvision.transforms as T
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2

sys.path.append('/kaggle/working/VA-VAE/LightningDiT/vavae')
sys.path.append('/kaggle/working/taming-transformers')  # 添加taming路径
sys.path.append('/kaggle/working/VA-VAE')  # 添加VA-VAE根路径

# 确保taming模块可以正确导入
import os
os.environ['PYTHONPATH'] = '/kaggle/working/taming-transformers:' + os.environ.get('PYTHONPATH', '')

def analyze_vf_semantic_relevance():
    """分析VF特征对微多普勒数据的语义相关性"""
    
    print("="*60)
    print("VF语义相关性分析")
    print("="*60)
    
    # 1. 加载不同用户的微多普勒图像
    # 尝试多个可能的数据路径
    possible_paths = [
        Path('/kaggle/input/dataset'),
        Path('/kaggle/input/microdoppler-dataset'),
        Path('/kaggle/input/microdoppler-dataset/train'),
        Path('/kaggle/input/dataset/train')
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists():
            print(f"🔍 检查路径: {path}")
            # 列出所有子目录进行调试
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            print(f"   子目录: {[d.name for d in subdirs[:10]]}")  # 只显示前10个
            
            user_dirs = list(path.glob('ID_*')) or list(path.glob('user*'))
            if user_dirs:
                data_dir = path
                print(f"📂 找到数据目录: {data_dir}")
                print(f"   用户目录: {[d.name for d in user_dirs[:5]]}")
                break
    
    if not data_dir:
        print(f"❌ 未找到数据目录，尝试的路径:")
        for path in possible_paths:
            if path.exists():
                subdirs = [d.name for d in path.iterdir() if d.is_dir()][:5]
                print(f"   {path} - 存在，子目录: {subdirs}")
            else:
                print(f"   {path} - 不存在")
        return {}
    
    user_samples = {}
    # 每个用户取5张图像
    user_dirs = list(data_dir.glob('ID_*')) or list(data_dir.glob('user*'))
    for user_dir in sorted(user_dirs)[:5]:  # 只分析前5个用户
        user_id = user_dir.name
        print(f"🔍 检查用户目录: {user_dir}")
        
        # 尝试多种图像格式，jpg优先
        image_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
        images = []
        for pattern in image_patterns:
            found_images = list(user_dir.glob(pattern))
            if found_images:
                images.extend(found_images)
                print(f"   找到 {len(found_images)} 个 {pattern} 文件")
                break
        
        if not images:
            # 调试：列出目录中的所有文件
            all_files = list(user_dir.iterdir())
            print(f"   目录中的所有文件: {[f.name for f in all_files[:10]]}")
        
        if images:
            user_samples[user_id] = images[:5]  # 只取前5张
            print(f"用户 {user_id}: {len(user_samples[user_id])} 张图像")
    
    if not user_samples:
        print(f"❌ 在 {data_dir} 中未找到任何用户图像")
        return {}
    
    return user_samples

def load_dinov2_features(image_path, model):
    """提取DINOv2特征"""
    from PIL import Image
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0)
    
    # 移动到正确设备
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        features = model(x)
    
    return features.squeeze(0).cpu()  # 返回CPU上的特征

def calculate_fid_score(real_images, fake_images, device='cuda'):
    """计算FID分数 - 针对重建任务优化"""
    
    # 检查样本数量
    n_samples = real_images.shape[0]
    if n_samples < 50:
        print(f"⚠️ FID计算：样本数量太少({n_samples})，FID需要大量样本(>50)才准确")
        print("   建议：增加测试图像数量或使用其他重建质量指标")
        return None
    
    try:
        from torchvision.models import inception_v3
        
        # 加载预训练的Inception网络
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        
        def get_inception_features(images):
            with torch.no_grad():
                # 调整图像尺寸到299x299 (Inception输入要求)
                if images.shape[-1] != 299:
                    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
                
                # 获取特征
                features = inception(images)
                return features.cpu().numpy()
        
        # 计算真实图像和重建图像的特征
        real_features = get_inception_features(real_images)
        fake_features = get_inception_features(fake_images)
        
        # 检查特征维度
        if real_features.shape[0] != fake_features.shape[0]:
            print("⚠️ FID计算：特征数量不匹配")
            return None
        
        # 计算均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # 添加正则化防止奇异矩阵
        eps = 1e-6
        sigma1 += eps * np.eye(sigma1.shape[0])
        sigma2 += eps * np.eye(sigma2.shape[0])
        
        # 计算FID - 修复负数问题
        diff = mu1 - mu2
        
        # 使用scipy的矩阵平方根来避免数值不稳定
        try:
            from scipy.linalg import sqrtm
            covmean = sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except ImportError:
            # 退回到简化计算
            print("⚠️ scipy不可用，使用简化FID计算")
            eigvals = np.linalg.eigvals(sigma1.dot(sigma2))
            covmean_trace = np.sqrt(np.abs(eigvals)).sum()
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * covmean_trace
            return max(0, fid)  # 确保非负
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return max(0, fid)  # 确保FID非负
        
    except Exception as e:
        print(f"FID计算失败: {e}")
        return None


def calculate_feature_similarity(real_images, fake_images, device='cuda'):
    """计算特征层面的相似度 - 更适合重建任务"""
    try:
        from torchvision.models import inception_v3
        
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        
        def get_inception_features(images):
            with torch.no_grad():
                if images.shape[-1] != 299:
                    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
                features = inception(images)
                return features
        
        # 获取特征
        real_features = get_inception_features(real_images)
        fake_features = get_inception_features(fake_images)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(real_features, fake_features, dim=1).mean().item()
        
        # 计算特征距离
        feature_dist = F.mse_loss(real_features, fake_features).item()
        
        return {
            'cosine_similarity': cos_sim,
            'feature_mse': feature_dist
        }
        
    except Exception as e:
        print(f"特征相似度计算失败: {e}")
        return None

def calculate_lpips_score(real_images, fake_images):
    """计算LPIPS感知距离"""
    try:
        import lpips
        
        # 初始化LPIPS网络
        lpips_net = lpips.LPIPS(net='alex')  # 使用AlexNet
        if torch.cuda.is_available():
            lpips_net = lpips_net.cuda()
        
        with torch.no_grad():
            # 确保图像在[-1,1]范围内
            real_norm = (real_images - real_images.min()) / (real_images.max() - real_images.min()) * 2 - 1
            fake_norm = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min()) * 2 - 1
            
            # 计算LPIPS距离
            lpips_scores = []
            for i in range(real_norm.shape[0]):
                score = lpips_net(real_norm[i:i+1], fake_norm[i:i+1])
                lpips_scores.append(score.item())
            
            return np.mean(lpips_scores)
            
    except Exception as e:
        print(f"LPIPS计算失败: {e}")
        return None

def calculate_traditional_metrics(real_images, fake_images):
    """计算传统图像质量指标：PSNR和SSIM"""
    psnr_scores = []
    ssim_scores = []
    
    for i in range(real_images.shape[0]):
        # 转换为numpy数组 [0,1]
        real_np = real_images[i].cpu().numpy().transpose(1, 2, 0)
        fake_np = fake_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # 确保值在[0,1]范围
        real_np = np.clip((real_np + 1) / 2, 0, 1)
        fake_np = np.clip((fake_np + 1) / 2, 0, 1)
        
        # 计算PSNR
        psnr = compare_psnr(real_np, fake_np, data_range=1.0)
        psnr_scores.append(psnr)
        
        # 计算SSIM
        if real_np.shape[-1] == 3:  # RGB图像
            ssim = compare_ssim(real_np, fake_np, multichannel=True, data_range=1.0, channel_axis=2)
        else:  # 灰度图像
            ssim = compare_ssim(real_np, fake_np, data_range=1.0)
        ssim_scores.append(ssim)
    
    return np.mean(psnr_scores), np.mean(ssim_scores)

def analyze_inter_vs_intra_user_similarity():
    """分析用户内vs用户间的VF特征相似性"""
    
    print("\n" + "="*60)
    print("用户间/用户内VF特征相似性分析")
    print("="*60)
    
    try:
        # 加载DINOv2模型
        print("正在下载DINOv2模型...")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dinov2_model.eval()
        if torch.cuda.is_available():
            dinov2_model = dinov2_model.cuda()
        print("✓ DINOv2模型加载成功")
        
        user_samples = analyze_vf_semantic_relevance()
        
        if not user_samples:
            print("❌ 未找到用户样本，跳过VF特征分析")
            return None
        
        # 提取所有特征
        all_features = {}
        for user_id, image_paths in user_samples.items():
            user_features = []
            for img_path in image_paths:
                try:
                    features = load_dinov2_features(img_path, dinov2_model)
                    user_features.append(features)
                except Exception as e:
                    print(f"跳过图像 {img_path}: {e}")
                    
            if user_features:
                all_features[user_id] = torch.stack(user_features)
                print(f"{user_id}: {len(user_features)} 个特征向量")
        
        # 计算相似性矩阵
        intra_user_similarities = []
        inter_user_similarities = []
        
        users = list(all_features.keys())
        
        # 用户内相似性
        for user_id in users:
            features = all_features[user_id]
            n_samples = features.shape[0]
            
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    sim = F.cosine_similarity(features[i], features[j], dim=0)
                    intra_user_similarities.append(sim.item())
        
        # 用户间相似性
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users):
                if i < j:  # 避免重复
                    features1 = all_features[user1]
                    features2 = all_features[user2]
                    
                    # 计算所有组合的相似性
                    for f1 in features1:
                        for f2 in features2:
                            sim = F.cosine_similarity(f1, f2, dim=0)
                            inter_user_similarities.append(sim.item())
        
        # 统计分析
        intra_mean = np.mean(intra_user_similarities)
        intra_std = np.std(intra_user_similarities)
        inter_mean = np.mean(inter_user_similarities)
        inter_std = np.std(inter_user_similarities)
        
        print(f"\n📊 VF特征相似性统计:")
        print(f"用户内相似性: {intra_mean:.4f} ± {intra_std:.4f}")
        print(f"用户间相似性: {inter_mean:.4f} ± {inter_std:.4f}")
        print(f"判别能力 (差值): {intra_mean - inter_mean:.4f}")
        
        # 判断VF特征的判别能力
        if intra_mean - inter_mean > 0.05:
            print("\n✅ VF特征对用户有较好判别能力，当前VA-VAE可用")
        elif intra_mean - inter_mean > 0.02:
            print("\n⚠️ VF特征判别能力一般，可考虑优化")
        else:
            print("\n❌ VF特征判别能力差，建议重头训练VFM")
            
        return {
            'intra_similarities': intra_user_similarities,
            'inter_similarities': inter_user_similarities,
            'discrimination_power': intra_mean - inter_mean
        }
        
    except Exception as e:
        print(f"分析失败: {e}")
        return None

def analyze_latent_space_quality():
    """分析当前VA-VAE潜在空间的质量"""
    
    print("\n" + "="*60) 
    print("VA-VAE潜在空间质量分析")
    print("="*60)
    
    try:
        # 加载VA-VAE - 处理taming导入问题
        sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT/vavae')
        sys.path.insert(0, '/kaggle/working/taming-transformers')
        
        # 手动导入taming模块以避免导入错误
        try:
            import taming
        except ImportError:
            print("警告: taming模块导入失败，尝试跳过VA-VAE分析")
            return None
            
        from ldm.models.autoencoder import AutoencoderKL
        
        # 先加载checkpoint检查实际架构
        checkpoint = torch.load('/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt', 
                               map_location='cpu')
        
        # 使用step4_train_vavae.py中的正确配置
        vae = AutoencoderKL(
            embed_dim=32,
            use_vf='dinov2',
            reverse_proj=True,
            ddconfig=dict(
                double_z=True,
                z_channels=32,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 1, 2, 2, 4],  # 使用训练时的正确配置
                num_res_blocks=2,
                attn_resolutions=[16],
                dropout=0.0
            ),
            lossconfig=dict(
                target="ldm.modules.losses.contperceptual.LPIPSWithDiscriminator",
                params=dict(
                    disc_start=50001,
                    disc_num_layers=3,
                    disc_weight=0.5,
                    disc_factor=1.0,
                    disc_in_channels=3,
                    disc_conditional=False,
                    disc_loss='hinge',
                    pixelloss_weight=1.0,
                    perceptual_weight=1.0,
                    kl_weight=1e-6,
                    logvar_init=0.0,
                    use_actnorm=False,
                    pp_style=False,
                    vf_weight=1.0,
                    adaptive_vf=False,
                    distmat_weight=1.0,
                    cos_weight=1.0,
                    distmat_margin=0.0,
                    cos_margin=0.0
                )
            )
        )
        
        # 加载checkpoint  
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        vae.load_state_dict(state_dict, strict=False)
        vae.eval()
        
        # 将VAE移到GPU
        if torch.cuda.is_available():
            vae = vae.cuda()
            print("✓ VA-VAE加载成功并移到GPU")
        else:
            print("✓ VA-VAE加载成功 (CPU模式)")
        
        # 分析潜在空间的用户判别性 - 直接在这里查找数据
        possible_paths = [
            Path('/kaggle/input/dataset'),
            Path('/kaggle/input/microdoppler-dataset'),
            Path('/kaggle/input/microdoppler-dataset/train'),
            Path('/kaggle/input/dataset/train')
        ]
        
        data_dir = None
        for path in possible_paths:
            if path.exists():
                user_dirs = list(path.glob('ID_*')) or list(path.glob('user*'))
                if user_dirs:
                    data_dir = path
                    print(f"📂 潜在空间分析找到数据: {data_dir}")
                    break
        
        if not data_dir:
            print("❌ 未找到数据目录，跳过潜在空间分析")
            return None
            
        user_latents = {}
        
        # 获取用户目录和图像
        user_dirs = list(data_dir.glob('ID_*')) or list(data_dir.glob('user*'))
        for user_dir in sorted(user_dirs)[:3]:  # 分析前3个用户
            user_id = user_dir.name
            # 查找jpg图像文件
            image_paths = list(user_dir.glob('*.jpg')) or list(user_dir.glob('*.JPG'))
            image_paths = image_paths[:3]  # 每用户3张图
            
            if not image_paths:
                continue
            latents = []
            print(f"处理用户 {user_id} 的 {len(image_paths)} 张图像...")
            for i, img_path in enumerate(image_paths):
                try:
                    print(f"  处理图像 {i+1}/{len(image_paths)}: {img_path.name}")
                    # 加载图像
                    img = Image.open(img_path).convert('RGB').resize((256, 256))
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2.0 - 1.0  # 归一化到[-1,1]
                    img_tensor = img_tensor.unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        img_tensor = img_tensor.cuda()
                    
                    # 编码到潜在空间
                    with torch.no_grad():
                        posterior = vae.encode(img_tensor)
                        latent = posterior.sample()
                        latents.append(latent.squeeze(0).cpu())
                        print(f"    ✓ 编码完成，潜在向量形状: {latent.squeeze(0).shape}")
                        
                except Exception as e:
                    print(f"    ❌ 处理图像失败 {img_path}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if latents:
                user_latents[user_id] = torch.stack(latents)
                print(f"{user_id}: {len(latents)} 个潜在向量")
        
        # 计算潜在空间的用户判别性
        if len(user_latents) >= 2:
            users = list(user_latents.keys())
            
            # 用户内距离
            intra_distances = []
            for user_id in users:
                latents = user_latents[user_id]
                for i in range(len(latents)):
                    for j in range(i+1, len(latents)):
                        dist = torch.norm(latents[i] - latents[j]).item()
                        intra_distances.append(dist)
            
            # 用户间距离  
            inter_distances = []
            for i, user1 in enumerate(users):
                for j, user2 in enumerate(users):
                    if i < j:
                        latents1 = user_latents[user1]
                        latents2 = user_latents[user2]
                        for l1 in latents1:
                            for l2 in latents2:
                                dist = torch.norm(l1 - l2).item()
                                inter_distances.append(dist)
            
            intra_mean = np.mean(intra_distances)
            inter_mean = np.mean(inter_distances)
            
            print(f"\n📊 潜在空间距离统计:")
            print(f"用户内距离: {intra_mean:.4f}")
            print(f"用户间距离: {inter_mean:.4f}") 
            print(f"分离度: {inter_mean / intra_mean:.4f}")
            
            if inter_mean / intra_mean > 1.2:
                print("\n✅ 潜在空间用户分离度良好")
                return True
            else:
                print("\n❌ 潜在空间用户分离度不足")
                return False
        
    except Exception as e:
        print(f"潜在空间分析失败: {e}")
        return None

def analyze_reconstruction_quality():
    """分析VA-VAE重建质量"""
    
    print("\n" + "="*60)
    print("VA-VAE重建质量分析")
    print("="*60)
    
    try:
        # 加载VA-VAE (复用之前的代码)
        sys.path.insert(0, '/kaggle/working/VA-VAE/LightningDiT/vavae')
        sys.path.insert(0, '/kaggle/working/taming-transformers')
        
        try:
            import taming
        except ImportError:
            print("警告: taming模块导入失败，跳过重建质量分析")
            return None
            
        from ldm.models.autoencoder import AutoencoderKL
        
        # 加载checkpoint
        checkpoint = torch.load('/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt', 
                               map_location='cpu')
        
        # 创建VAE模型
        vae = AutoencoderKL(
            embed_dim=32,
            use_vf='dinov2',
            reverse_proj=True,
            ddconfig=dict(
                double_z=True, z_channels=32, resolution=256, in_channels=3, out_ch=3,
                ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, 
                attn_resolutions=[16], dropout=0.0
            ),
            lossconfig=dict(
                target="ldm.modules.losses.contperceptual.LPIPSWithDiscriminator",
                params=dict(
                    disc_start=50001, disc_num_layers=3, disc_weight=0.5, disc_factor=1.0,
                    disc_in_channels=3, disc_conditional=False, disc_loss='hinge',
                    pixelloss_weight=1.0, perceptual_weight=1.0, kl_weight=1e-6, logvar_init=0.0,
                    use_actnorm=False, pp_style=False, vf_weight=1.0, adaptive_vf=False,
                    distmat_weight=1.0, cos_weight=1.0, distmat_margin=0.0, cos_margin=0.0
                )
            )
        )
        
        # 加载权重
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        vae.load_state_dict(state_dict, strict=False)
        vae.eval()
        
        if torch.cuda.is_available():
            vae = vae.cuda()
        
        print("✓ VA-VAE加载成功")
        
        # 收集测试图像
        data_dir = Path('/kaggle/input/dataset')
        if not data_dir.exists():
            print("❌ 数据目录不存在")
            return None
        
        test_images = []
        user_dirs = list(data_dir.glob('ID_*'))[:3]  # 前3个用户
        
        for user_dir in user_dirs:
            images = list(user_dir.glob('*.jpg'))[:5]  # 每用户5张图
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('RGB').resize((256, 256))
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2.0 - 1.0  # 归一化到[-1,1]
                    test_images.append(img_tensor)
                except Exception as e:
                    continue
        
        if len(test_images) < 5:
            print(f"❌ 测试图像不足: 只找到{len(test_images)}张")
            return None
        
        # 批处理重建
        batch_size = min(8, len(test_images))
        test_batch = torch.stack(test_images[:batch_size])
        
        if torch.cuda.is_available():
            test_batch = test_batch.cuda()
        
        print(f"🖼️ 处理 {batch_size} 张测试图像...")
        
        with torch.no_grad():
            # 编码-解码重建
            posterior = vae.encode(test_batch)
            latents = posterior.sample()
            reconstructed = vae.decode(latents)
        
        print("✓ 重建完成，开始质量评估...")
        
        # 计算各种质量指标
        results = {}
        
        # 1. 传统指标 (PSNR, SSIM)
        try:
            psnr, ssim = calculate_traditional_metrics(test_batch, reconstructed)
            results['PSNR'] = psnr
            results['SSIM'] = ssim
            print(f"📊 PSNR: {psnr:.2f} dB")
            print(f"📊 SSIM: {ssim:.4f}")
        except Exception as e:
            print(f"⚠️ 传统指标计算失败: {e}")
        
        # 2. LPIPS感知距离
        try:
            lpips_score = calculate_lpips_score(test_batch, reconstructed)
            if lpips_score is not None:
                results['LPIPS'] = lpips_score
                print(f"📊 LPIPS: {lpips_score:.4f}")
        except Exception as e:
            print(f"⚠️ LPIPS计算失败: {e}")
        
        # 3. FID分数 (适用于大样本分布比较)
        try:
            fid_score = calculate_fid_score(test_batch, reconstructed)
            if fid_score is not None:
                results['FID'] = fid_score
                print(f"📊 FID: {fid_score:.2f}")
            else:
                print("📊 FID: 样本数量不足，跳过FID计算")
        except Exception as e:
            print(f"⚠️ FID计算失败: {e}")
        
        # 4. 特征相似度 (更适合重建任务)
        try:
            feat_sim = calculate_feature_similarity(test_batch, reconstructed)
            if feat_sim is not None:
                results['Feature_Cosine_Sim'] = feat_sim['cosine_similarity']
                results['Feature_MSE'] = feat_sim['feature_mse']
                print(f"📊 特征余弦相似度: {feat_sim['cosine_similarity']:.4f}")
                print(f"📊 特征MSE: {feat_sim['feature_mse']:.6f}")
        except Exception as e:
            print(f"⚠️ 特征相似度计算失败: {e}")
        
        # 5. 像素级误差
        mse = F.mse_loss(test_batch, reconstructed).item()
        mae = F.l1_loss(test_batch, reconstructed).item()
        results['MSE'] = mse
        results['MAE'] = mae
        print(f"📊 MSE: {mse:.6f}")
        print(f"📊 MAE: {mae:.6f}")
        
        # 评估结果
        print(f"\n🎯 重建质量评估:")
        
        # PSNR评估 (越高越好)
        if 'PSNR' in results:
            if results['PSNR'] > 25:
                print(f"   ✅ PSNR {results['PSNR']:.1f}dB - 重建质量优秀")
            elif results['PSNR'] > 20:
                print(f"   ⚠️ PSNR {results['PSNR']:.1f}dB - 重建质量良好")
            else:
                print(f"   ❌ PSNR {results['PSNR']:.1f}dB - 重建质量较差")
        
        # SSIM评估 (越高越好)
        if 'SSIM' in results:
            if results['SSIM'] > 0.9:
                print(f"   ✅ SSIM {results['SSIM']:.3f} - 结构相似性优秀")
            elif results['SSIM'] > 0.8:
                print(f"   ⚠️ SSIM {results['SSIM']:.3f} - 结构相似性良好")
            else:
                print(f"   ❌ SSIM {results['SSIM']:.3f} - 结构相似性较差")
        
        # LPIPS评估 (越低越好)
        if 'LPIPS' in results:
            if results['LPIPS'] < 0.1:
                print(f"   ✅ LPIPS {results['LPIPS']:.3f} - 感知质量优秀")
            elif results['LPIPS'] < 0.2:
                print(f"   ⚠️ LPIPS {results['LPIPS']:.3f} - 感知质量良好")
            else:
                print(f"   ❌ LPIPS {results['LPIPS']:.3f} - 感知质量较差")
        
        return results
        
    except Exception as e:
        print(f"❌ 重建质量分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主分析函数"""
    print("🔍 VA-VAE和VFM重训练必要性分析")
    print("="*70)
    
    # 分析1: VF特征判别能力
    vf_analysis = analyze_inter_vs_intra_user_similarity()
    
    # 分析2: 潜在空间质量
    latent_quality = analyze_latent_space_quality()
    
    # 分析3: 重建质量评估
    reconstruction_quality = analyze_reconstruction_quality()
    
    # 综合建议
    print("\n" + "="*70)
    print("🎯 综合建议")
    print("="*70)
    
    if vf_analysis and latent_quality:
        discrimination = vf_analysis['discrimination_power']
        
        if discrimination > 0.05 and latent_quality:
            print("✅ 建议：继续使用当前VA-VAE")
            print("   理由：VF特征和潜在空间都有足够的判别能力")
            
        elif discrimination > 0.02:
            print("⚠️ 建议：可尝试改进当前VA-VAE")
            print("   理由：性能尚可，但有改进空间")
            print("   方案：调整VF loss权重或添加对比学习")
            
        else:
            print("❌ 建议：考虑重头训练专门的VFM")
            print("   理由：当前VF特征对微多普勒数据判别能力不足")
            print("   方案：设计微多普勒专用的语义特征提取器")
    
    print("\n💡 如果选择重头训练，建议的方向:")
    print("1. 自监督学习：对比学习训练微多普勒特征提取器")
    print("2. 运动模式分类：基于物理运动类别训练判别器")
    print("3. 时频模式学习：专门的时频域特征学习")

if __name__ == '__main__':
    main()
