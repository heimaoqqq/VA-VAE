"""
生成并筛选高质量样本脚本
基于 generate_and_filter_samples.py 改进，支持多指标筛选
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# 抑制torch._dynamo错误，回退到eager模式
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import torchvision.transforms as transforms
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from simplified_vavae import SimplifiedVAVAE


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def load_weights_with_shape_check(model, checkpoint, rank=0):
    """使用形状检查加载权重（完全按照官方实现）"""
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model


def load_model_and_config(checkpoint_path, config_path, local_rank):
    """加载模型和配置（按照官方方式）"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    device = torch.device(f'cuda:{local_rank}')
    
    # 创建DiT模型
    latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']
    model = LightningDiT_models[config['model']['model_type']](
        input_size=latent_size,
        num_classes=config['data']['num_classes'],
        class_dropout_prob=config['model'].get('class_dropout_prob', 0.1),
        use_qknorm=config['model']['use_qknorm'],
        use_swiglu=config['model'].get('use_swiglu', False),
        use_rope=config['model'].get('use_rope', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False),
        wo_shift=config['model'].get('wo_shift', False),
        in_channels=config['model'].get('in_chans', 4),
        use_checkpoint=config['model'].get('use_checkpoint', False),
    ).to(device)
    
    # 按照官方方式加载权重
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"📦 从checkpoint加载权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # 处理权重键名（按照官方方式）
        if 'ema' in checkpoint:
            checkpoint_weights = {'model': checkpoint['ema']}
            if local_rank == 0:
                print("📦 使用EMA权重进行推理")
        elif 'model' in checkpoint:
            checkpoint_weights = checkpoint
            if local_rank == 0:
                print("📦 使用模型权重进行推理")
        else:
            checkpoint_weights = {'model': checkpoint}
            if local_rank == 0:
                print("📦 使用直接权重进行推理")
        
        # 清理键名
        checkpoint_weights['model'] = {k.replace('module.', ''): v for k, v in checkpoint_weights['model'].items()}
        
        # 使用官方权重加载函数
        model = load_weights_with_shape_check(model, checkpoint_weights, rank=local_rank)
        
        if local_rank == 0:
            print("✅ 权重加载完成")
    else:
        if local_rank == 0:
            print(f"⚠️ 警告: 找不到checkpoint文件 {checkpoint_path}")
            print("⚠️ 使用未训练的随机权重，生成结果将是噪声！")
    
    model.eval()
    
    # 创建VAE（完全按照官方train_dit_s_official.py方式）
    vae = None
    try:
        # 添加LightningDiT路径到系统路径
        import sys
        lightningdit_path = os.path.join(os.getcwd(), 'LightningDiT')
        if lightningdit_path not in sys.path:
            sys.path.insert(0, lightningdit_path)
        
        from tokenizer.vavae import VA_VAE
        import tempfile
        
        # 使用训练好的VAE模型路径
        custom_vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
        
        # 创建与train_dit_s_official.py完全一致的配置
        vae_config = {
            'ckpt_path': custom_vae_checkpoint,
            'model': {
                'base_learning_rate': 2.0e-05,
                'target': 'ldm.models.autoencoder.AutoencoderKL',
                'params': {
                    'monitor': 'val/rec_loss',
                    'embed_dim': 32,
                    'use_vf': 'dinov2',
                    'reverse_proj': True,
                    'ddconfig': {
                        'double_z': True, 'z_channels': 32, 'resolution': 256,
                        'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2,
                        'attn_resolutions': [16], 'dropout': 0.0
                    },
                    'lossconfig': {
                        'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                        'params': {
                            'disc_start': 1, 'disc_num_layers': 3, 'disc_weight': 0.5,
                            'disc_factor': 1.0, 'disc_in_channels': 3, 'disc_conditional': False,
                            'disc_loss': 'hinge', 'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,
                            'kl_weight': 1e-6, 'logvar_init': 0.0, 'use_actnorm': False,
                            'pp_style': False, 'vf_weight': 0.1, 'adaptive_vf': False,
                            'distmat_weight': 1.0, 'cos_weight': 1.0,
                            'distmat_margin': 0.25, 'cos_margin': 0.5
                        }
                    }
                }
            }
        }
        
        # 写入临时配置文件
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(vae_config, f, default_flow_style=False)
        os.close(temp_config_fd)
        
        try:
            # 使用官方VA_VAE类加载
            vae = VA_VAE(temp_config_path)
            # 检查是否有.to()方法
            if hasattr(vae, 'to'):
                vae = vae.to(device)
            if hasattr(vae, 'eval'):
                vae.eval()
            if local_rank == 0:
                print(f"✅ VAE加载完成: {custom_vae_checkpoint}")
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
            
    except Exception as e:
        if local_rank == 0:
            print(f"⚠️ VAE加载失败: {e}")
            print("⚠️ 尝试使用简化VAE作为备用")
        # 备用方案
        try:
            vae = SimplifiedVAVAE(config['vae']['model_name']).to(device)
            vae.eval()
            if local_rank == 0:
                print(f"✅ 备用VAE加载完成: {config['vae']['model_name']}")
        except Exception as e2:
            if local_rank == 0:
                print(f"⚠️ 备用VAE也加载失败: {e2}")
            vae = None
    
    # 创建transport
    transport = create_transport(
        config['transport']['path_type'],
        config['transport']['prediction'],
        config['transport']['loss_weight'],
        config['transport']['train_eps'],
        config['transport']['sample_eps'],
        use_cosine_loss=config['transport'].get('use_cosine_loss', False),
        use_lognorm=config['transport'].get('use_lognorm', False),
        partitial_train=config['transport'].get('partitial_train', None),
        partial_ratio=config['transport'].get('partial_ratio', 1.0),
        shift_lg=config['transport'].get('shift_lg', False),
    )
    
    return model, vae, transport, config, device


def load_classifier(checkpoint_path, device):
    """加载预训练的分类器"""
    import torchvision.models as models
    
    # 创建与train_calibrated_classifier.py完全一致的DomainAdaptiveClassifier结构
    class DomainAdaptiveClassifier(nn.Module):
        def __init__(self, num_classes=31, dropout_rate=0.3, feature_dim=512):
            super().__init__()
            
            # 使用ResNet18作为backbone
            self.backbone = models.resnet18(pretrained=False)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # 特征投影层（与训练时完全一致）
            self.feature_projector = nn.Sequential(
                nn.Linear(backbone_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            
            # 分类头（与训练时完全一致）
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
            
            # 身份特征记忆库
            self.register_buffer('feature_bank', torch.zeros(num_classes, feature_dim))
            self.register_buffer('feature_count', torch.zeros(num_classes))
        
        def forward(self, x):
            backbone_features = self.backbone(x)
            features = self.feature_projector(backbone_features)
            logits = self.classifier(features)
            return logits
    
    # 创建模型
    model = DomainAdaptiveClassifier(num_classes=31)
    
    # 加载权重 - 现在结构完全匹配
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 分类器加载完成: {checkpoint_path}")
    if 'epoch' in checkpoint:
        val_acc = checkpoint.get('val_acc', 0) * 100  # 转换为百分比
        print(f"📊 Epoch: {checkpoint['epoch']}, Val Acc: {val_acc:.2f}%, ECE: {checkpoint.get('val_ece', 0):.4f}")
    
    return model


def extract_features(images, classifier, device):
    """提取图像特征用于多样性评估"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_list = []
    
    with torch.no_grad():
        for img in images:
            img_tensor = transform(img).unsqueeze(0).to(device)
            # 提取backbone特征（在分类头之前）
            features = classifier.backbone(img_tensor)
            features_list.append(features.cpu().numpy().flatten())
    
    return np.array(features_list)


def compute_diversity_metrics(features):
    """计算特征多样性指标"""
    if len(features) < 2:
        return {'diversity_score': 0.0, 'avg_pairwise_dist': 0.0}
    
    cosine_sim_matrix = cosine_similarity(features)
    upper_triangle = np.triu(cosine_sim_matrix, k=1)
    avg_similarity = np.sum(upper_triangle) / (len(features) * (len(features) - 1) / 2)
    diversity_score = 1.0 - avg_similarity
    
    from scipy.spatial.distance import pdist
    pairwise_distances = pdist(features, metric='euclidean')
    avg_pairwise_dist = np.mean(pairwise_distances)
    
    return {
        'diversity_score': diversity_score,
        'avg_pairwise_dist': avg_pairwise_dist,
        'avg_similarity': avg_similarity
    }


def simple_quality_check(images):
    """简化的图像质量检查（仅检测基本异常）"""
    quality_scores = []
    
    for img in images:
        img_array = np.array(img)
        
        # 只检测基本像素值异常
        pixel_mean = np.mean(img_array)
        pixel_std = np.std(img_array)
        
        # 简单的质量分数：只检查是否全黑/全白或无变化
        is_valid = (
            10 < pixel_mean < 245 and  # 不是全黑或全白
            pixel_std > 5              # 有一定变化
        )
        
        quality_score = {
            'pixel_mean': pixel_mean,
            'pixel_std': pixel_std,
            'is_valid': is_valid,
            'overall': 1.0 if is_valid else 0.0
        }
        
        quality_scores.append(quality_score)
    
    return quality_scores


def compute_user_specific_metrics(images, classifier, user_id, device, user_prototypes=None):
    """计算用户特定指标"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    user_metrics_list = []
    
    for img in images:
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 获取分类器输出和特征
            outputs = classifier(img_tensor)
            probs = F.softmax(outputs, dim=1)
            features = classifier.backbone(img_tensor)
            
            # 1. 基本指标
            confidence, pred = torch.max(probs, dim=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
            
            # 2. 用户特异性分数（与其他用户的区分度）
            user_prob = probs[0, user_id].item()
            other_probs = torch.cat([probs[0, :user_id], probs[0, user_id+1:]])
            max_other_prob = torch.max(other_probs).item()
            user_specificity = user_prob - max_other_prob
            
            # 3. 增强稳定性
            aug_preds = []
            for _ in range(3):
                aug_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                aug_tensor = aug_transform(img).unsqueeze(0).to(device)
                aug_outputs = classifier(aug_tensor)
                aug_pred = torch.argmax(aug_outputs, dim=1)
                aug_preds.append(aug_pred.item())
            
            stability = sum([p == pred.item() for p in aug_preds]) / len(aug_preds)
            
            # 4. 与用户原型的相似度（如果提供）
            prototype_similarity = 0.0
            if user_prototypes is not None and user_id in user_prototypes:
                prototype_features = user_prototypes[user_id]
                current_features = features.cpu().numpy().flatten()
                prototype_similarity = cosine_similarity(
                    [current_features], [prototype_features]
                )[0, 0]
            
            metrics = {
                'predicted': pred.item(),
                'confidence': confidence.item(),
                'margin': margin,
                'stability': stability,
                'user_specificity': user_specificity,
                'prototype_similarity': prototype_similarity,
                'correct': pred.item() == user_id,
                'features': features.cpu().numpy().flatten()
            }
            
            user_metrics_list.append(metrics)
    
    return user_metrics_list


def generate_and_filter_advanced(model, vae, transport, classifier, user_id, 
                                 target_samples=800, batch_size=100, 
                                 confidence_threshold=0.95, margin_threshold=0.2,
                                 stability_threshold=0.8, diversity_threshold=0.3,
                                 user_specificity_threshold=0.1, 
                                 conservative_mode=False, cfg_scale=12.0, 
                                 domain_coverage=True,
                                 output_dir='./filtered_samples', device=None, rank=0,
                                 user_prototypes=None):
    """为单个用户生成并使用多指标筛选样本"""
    
    # 域覆盖策略：多样化生成条件
    if domain_coverage:
        # 定义不同的生成条件组合（覆盖不同域）
        domain_conditions = [
            {"cfg": 10.0, "steps": 250, "name": "low_guidance"},     # 低引导，更多样
            {"cfg": 12.0, "steps": 300, "name": "standard"},         # 标准设置
            {"cfg": 15.0, "steps": 400, "name": "high_guidance"},    # 高引导，更精确
        ]
        samples_per_condition = target_samples // len(domain_conditions)
        print(f"🌐 域覆盖模式: {len(domain_conditions)}种生成条件，每种{samples_per_condition}张")
    else:
        # 单一条件生成
        domain_conditions = [{"cfg": cfg_scale, "steps": 300, "name": "single"}]
        samples_per_condition = target_samples
    
    # 创建采样器（将在循环中动态调整参数）
    sampler = Sampler(transport)
    
    # 创建输出目录
    user_dir = Path(output_dir) / f"User_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    collected_samples = []
    total_generated = 0
    
    print(f"🎯 开始为User_{user_id:02d}生成样本，目标: {target_samples}张")
    print(f"📊 统一筛选标准（按文献重要性排序）:")
    print(f"   1. 身份一致性(置信度): >{confidence_threshold:.2f}")
    print(f"   2. 用户特异性: >{user_specificity_threshold:.2f}")
    print(f"   3. 预测稳定性: >{stability_threshold:.2f}")
    print(f"   4. 决策边界: >{margin_threshold:.2f}")
    print(f"   5. 特征多样性: >{diversity_threshold:.2f}")
    
    # 存储已收集的特征用于多样性评估
    collected_features = []
    condition_stats = {cond["name"]: 0 for cond in domain_conditions}
    
    # 创建进度条
    pbar = tqdm(total=target_samples, desc=f"User_{user_id:02d}", unit="样本", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        # 按域条件循环生成
        for condition in domain_conditions:
            condition_target = samples_per_condition
            condition_collected = 0
            
            # 为当前条件创建特定的采样函数
            current_sample_fn = sampler.sample_ode(
                sampling_method="dopri5",
                num_steps=condition["steps"],
                atol=1e-6,
                rtol=1e-3,
                reverse=False
            )
            
            print(f"🌍 生成域: {condition['name']} (CFG={condition['cfg']:.1f}, Steps={condition['steps']})")
            
            while condition_collected < condition_target and len(collected_samples) < target_samples:
                # 生成一批样本
                remaining_total = target_samples - len(collected_samples)
                remaining_condition = condition_target - condition_collected
                current_batch_size = min(batch_size, remaining_condition, remaining_total)
                
                # 准备条件
                y = torch.tensor([user_id] * current_batch_size, device=device)
                
                # 创建随机噪声
                z = torch.randn(current_batch_size, 32, 16, 16, device=device)
                
                # 使用当前域条件生成样本
                current_cfg = condition["cfg"]
                if current_cfg > 1.0:
                    # CFG采样（按照generate_and_filter_samples.py的实现）
                    z_cfg = torch.cat([z, z], 0)  # 复制噪声tensor
                    y_null = torch.tensor([31] * current_batch_size, device=device)
                    y_cfg = torch.cat([y, y_null], 0)
                    
                    cfg_interval_start = 0.11
                    model_kwargs = dict(y=y_cfg, cfg_scale=current_cfg, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                    
                    if hasattr(model, 'forward_with_cfg'):
                        samples = current_sample_fn(z_cfg, model.forward_with_cfg, **model_kwargs)
                    else:
                        def model_fn_cfg(x, t, **kwargs):
                            pred = model(x, t, **kwargs)
                            pred_cond, pred_uncond = pred.chunk(2, dim=0)
                            return pred_uncond + current_cfg * (pred_cond - pred_uncond)
                        samples = current_sample_fn(z_cfg, model_fn_cfg, **model_kwargs)
                    
                    samples = samples[-1]
                    samples, _ = samples.chunk(2, dim=0)  # 只保留conditional样本
                else:
                    # 不使用CFG
                    samples = current_sample_fn(z, model, **dict(y=y))
                    samples = samples[-1]
                
                # 反归一化（按照generate_and_filter_samples.py的实现）
                latent_stats_path = '/kaggle/working/VA-VAE/latents_safetensors/train/latent_stats.pt'
                if os.path.exists(latent_stats_path):
                    stats = torch.load(latent_stats_path, map_location=device)
                    mean = stats['mean'].to(device)
                    std = stats['std'].to(device)
                    latent_multiplier = 1.0
                    samples_denorm = (samples * std) / latent_multiplier + mean
                else:
                    # 备用：无反归一化
                    print("⚠️ 无法加载latent统计，使用原始样本")
                    samples_denorm = samples
                
                # VAE解码
                if vae is not None:
                    try:
                        decoded_images = vae.decode_to_images(samples_denorm)
                        images_pil = [Image.fromarray(img) for img in decoded_images]
                        
                        # 计算用户特定指标
                        metrics_list = compute_user_specific_metrics(
                            images_pil, classifier, user_id, device, user_prototypes
                        )
                        
                        # 简化的质量检查
                        visual_quality_scores = simple_quality_check(images_pil)
                        
                        # 提取当前批次的特征
                        current_features = [m['features'] for m in metrics_list]
                        
                        # 筛选高质量样本
                        batch_accepted = 0
                        batch_candidates = []  # 候选样本
                        
                        # 第一步：基本质量筛选
                        for i, metrics in enumerate(metrics_list):
                            # 应用保守模式调整阈值
                            actual_conf_thresh = confidence_threshold * (1.05 if conservative_mode else 1.0)
                            actual_margin_thresh = margin_threshold * (1.1 if conservative_mode else 1.0)
                            actual_stability_thresh = stability_threshold * (1.05 if conservative_mode else 1.0)
                            
                            # 核心4指标筛选（按文献重要性排序）
                            if (metrics['correct'] and 
                                metrics['confidence'] > actual_conf_thresh and      # 1. 身份一致性(最重要)
                                metrics['user_specificity'] > user_specificity_threshold and  # 2. 用户特异性
                                metrics['stability'] > actual_stability_thresh and  # 3. 预测稳定性
                                metrics['margin'] > actual_margin_thresh and        # 4. 决策边界
                                visual_quality_scores[i]['is_valid']):              # 5. 基本有效性
                                
                                # 简化的统计异常检测（仅在保守模式下）
                                if conservative_mode and len(collected_features) > 15:
                                    try:
                                        # 使用简单的Z-score检测
                                        feature_array = np.array(collected_features)
                                        current_feature = metrics['features']
                                        
                                        # 计算与现有样本的欧氏距离
                                        distances = [np.linalg.norm(current_feature - f) for f in feature_array]
                                        mean_dist = np.mean(distances)
                                        std_dist = np.std(distances)
                                        
                                        # Z-score > 2.5 被认为异常
                                        if std_dist > 0 and (distances[-1] - mean_dist) / std_dist > 2.5:
                                            continue
                                    except:
                                        pass
                                
                                batch_candidates.append({
                                    'image': images_pil[i],
                                    'features': metrics['features'],
                                    'metrics': metrics,
                                    'index': i
                                })
                        
                        # 第二步：多样性筛选
                        for candidate in batch_candidates:
                            # 检查与已收集样本的多样性
                            if len(collected_features) > 0:
                                candidate_features = candidate['features'].reshape(1, -1)
                                collected_array = np.array(collected_features)
                                
                                # 计算与现有样本的最大相似度
                                similarities = cosine_similarity(candidate_features, collected_array)[0]
                                max_similarity = np.max(similarities)
                                diversity_score = 1.0 - max_similarity
                                
                                # 如果多样性不足，跳过
                                if diversity_score < diversity_threshold:
                                    continue
                            
                            # 通过所有筛选，保存样本
                            save_path = user_dir / f"sample_{len(collected_samples):06d}.png"
                            candidate['image'].save(save_path)
                            collected_samples.append(save_path)
                            collected_features.append(candidate['features'])
                            batch_accepted += 1
                            pbar.update(1)  # 更新进度条
                            
                            if len(collected_samples) >= target_samples:
                                break
                        
                        # 更新条件统计
                        condition_collected += batch_accepted
                        condition_stats[condition["name"]] += batch_accepted
                        
                        if len(collected_samples) >= target_samples:
                            break
                        
                        total_generated += current_batch_size
                        
                        # 更新进度条的后缀信息
                        success_rate = len(collected_samples) / total_generated * 100 if total_generated > 0 else 0
                        pbar.set_postfix({
                            '生成': total_generated,
                            '成功率': f'{success_rate:.1f}%',
                            '当前域': condition["name"]
                        })
                    
                    except Exception as e:
                        print(f"❌ 处理批次时出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ VAE未加载，无法解码")
                    break
    
    # 关闭进度条
    pbar.close()
    
    # 最终统计
    final_success_rate = len(collected_samples) / total_generated * 100 if total_generated > 0 else 0
    print(f"✅ User_{user_id:02d} 完成: 收集了 {len(collected_samples)}/{target_samples} 个高质量样本 | "
          f"总生成: {total_generated} 张 | 最终成功率: {final_success_rate:.1f}%")
    
    return len(collected_samples)


def main():
    parser = argparse.ArgumentParser(description='Generate and filter high-quality samples with multi-metrics')
    parser.add_argument('--dit_checkpoint', type=str, 
                       default='/kaggle/input/50000-pt/0050000.pt', 
                       help='DiT model checkpoint path')
    parser.add_argument('--classifier_checkpoint', type=str,
                       default='./calibrated_classifier/best_calibrated_model.pth',
                       help='Classifier checkpoint path')
    parser.add_argument('--config', type=str, 
                       default='configs/dit_s_microdoppler.yaml', 
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./filtered_samples_multi', 
                       help='Output directory')
    parser.add_argument('--target_samples', type=int, default=800, 
                       help='Target number of samples per user')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='Batch size for generation')
    parser.add_argument('--confidence_threshold', type=float, default=0.95, 
                       help='Confidence threshold for filtering')
    parser.add_argument('--margin_threshold', type=float, default=0.2,
                       help='Decision margin threshold')
    parser.add_argument('--stability_threshold', type=float, default=0.8,
                       help='Augmentation stability threshold')
    parser.add_argument('--diversity_threshold', type=float, default=0.3,
                       help='Feature diversity threshold')
    parser.add_argument('--user_specificity_threshold', type=float, default=0.1,
                       help='User specificity threshold')
    # 移除visual_quality_threshold参数
    parser.add_argument('--conservative_mode', action='store_true',
                       help='Enable conservative filtering (stricter thresholds)')
    # 移除max_outlier_ratio参数（简化版本不需要）
    parser.add_argument('--cfg_scale', type=float, default=12.0, 
                       help='Base CFG scale for generation')
    parser.add_argument('--domain_coverage', action='store_true', default=True,
                       help='Enable domain coverage with diverse generation conditions')
    parser.add_argument('--single_condition', action='store_true',
                       help='Use single generation condition (disable domain coverage)')
    parser.add_argument('--start_user', type=int, default=0,
                       help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=30,
                       help='Ending user ID (inclusive)')
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"🚀 生成并筛选高质量样本（多指标版本）")
        print(f"📂 输出目录: {args.output_dir}")
        print(f"🎯 目标: 每用户 {args.target_samples} 张样本")
        print(f"📊 筛选阈值:")
        print(f"   - 置信度: {args.confidence_threshold}")
        print(f"   - 决策边界: {args.margin_threshold}")
        print(f"   - 稳定性: {args.stability_threshold}")
        print(f"   - 多样性: {args.diversity_threshold}")
        print(f"   - 用户特异性: {args.user_specificity_threshold}")
        print(f"   - 简化质量检查: 开启")
        if args.conservative_mode:
            print(f"   - 保守模式: 开启（更严格的统计检测）")
        domain_coverage_enabled = not args.single_condition
        print(f"🌐 域覆盖: {'开启' if domain_coverage_enabled else '关闭'}")
        print(f"⚙️ 基础CFG: {args.cfg_scale}")
    
    # 加载DiT模型
    model, vae, transport, config, device = load_model_and_config(
        args.dit_checkpoint, args.config, local_rank
    )
    
    # 加载分类器
    classifier = load_classifier(args.classifier_checkpoint, device)
    
    # 创建输出目录
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # 分布式处理：每个GPU处理不同的用户
    total_collected = 0
    user_list = list(range(args.start_user, args.end_user + 1))
    
    # 将用户分配给不同的GPU
    users_per_gpu = len(user_list) // world_size
    extra_users = len(user_list) % world_size
    
    # 计算当前GPU负责的用户范围
    start_idx = rank * users_per_gpu + min(rank, extra_users)
    end_idx = start_idx + users_per_gpu + (1 if rank < extra_users else 0)
    
    my_users = user_list[start_idx:end_idx]
    
    if rank == 0:
        print(f"🔄 分布式处理: {world_size} GPUs")
        print(f"📊 GPU分配: 每个GPU处理约{users_per_gpu}个用户")
    
    print(f"GPU {rank}: 处理用户 {my_users}")
    
    # 每个GPU处理自己分配的用户
    for user_id in my_users:
        collected = generate_and_filter_advanced(
            model, vae, transport, classifier, user_id,
            target_samples=args.target_samples,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            stability_threshold=args.stability_threshold,
            diversity_threshold=args.diversity_threshold,
            user_specificity_threshold=args.user_specificity_threshold,
            # visual_quality_threshold 参数已移除
            conservative_mode=args.conservative_mode,
            cfg_scale=args.cfg_scale,
            domain_coverage=not args.single_condition,
            output_dir=args.output_dir,
            device=device,
            rank=rank
        )
        total_collected += collected
    
    # 打印当前GPU完成情况
    print(f"GPU {rank} 完成: 收集了 {total_collected} 个样本 (处理了 {len(my_users)} 个用户)")
    
    # 同步所有GPU的结果
    if world_size > 1:
        try:
            dist.barrier()
            total_tensor = torch.tensor([total_collected], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            total_collected = total_tensor.item()
        except Exception as e:
            print(f"⚠️ GPU {rank} 同步失败: {e}")
    
    if rank == 0:
        print(f"🎯 生成完成！")
        print(f"✅ 总共收集了 {total_collected} 个高质量样本")
        print(f"📁 样本保存在: {args.output_dir}")
        if world_size > 1:
            expected_total = len(user_list) * args.target_samples
            print(f"📊 预期总数: {expected_total} (31用户 × {args.target_samples}样本/用户)")
    
    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
