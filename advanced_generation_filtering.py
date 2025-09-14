"""
高级生成和筛选策略
实现多指标筛选、渐进式生成、桥接样本选择
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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

# 导入已训练的分类器
sys.path.append('./')
from train_calibrated_classifier import DomainAdaptiveClassifier


def setup_ddp():
    """DDP初始化"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed generation")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """DDP清理"""
    if dist.is_initialized():
        dist.destroy_process_group()

class AdvancedSampleFilter:
    """多指标样本筛选器"""
    
    def __init__(self, classifiers, device, feature_bank=None):
        """
        Args:
            classifiers: 分类器列表（多个不同设置的分类器）
            device: 计算设备
            feature_bank: 真实数据特征库（用于计算特征距离）
        """
        self.classifiers = classifiers if isinstance(classifiers, list) else [classifiers]
        self.device = device
        self.feature_bank = feature_bank
        
    def extract_features(self, model, images):
        """提取倒数第二层特征"""
        features = []
        model.eval()
        with torch.no_grad():
            # 获取backbone特征
            if hasattr(model, 'backbone'):
                features = model.backbone(images)
            else:
                # 使用hook提取特征
                def hook_fn(module, input, output):
                    features.append(output)
                handle = model.classifier[-1].register_forward_hook(hook_fn)
                _ = model(images)
                handle.remove()
        return features[0] if features else None
    
    def compute_consistency_score(self, image, user_id):
        """多模型一致性评分"""
        predictions = []
        confidences = []
        
        for classifier in self.classifiers:
            classifier.eval()
            with torch.no_grad():
                output = classifier(image.unsqueeze(0))
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, dim=1)
                predictions.append(pred.item())
                confidences.append(conf.item())
        
        # 检查预测一致性
        is_consistent = all(p == user_id for p in predictions)
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)  # 置信度方差，越小越好
        
        # 综合评分
        consistency_score = avg_confidence * (1 - confidence_std) if is_consistent else 0
        
        return consistency_score, is_consistent, avg_confidence
    
    def compute_margin_score(self, classifier, image):
        """计算top-1和top-2之间的margin"""
        classifier.eval()
        with torch.no_grad():
            output = classifier(image.unsqueeze(0))
            logits = output[0]
            top2_values, _ = torch.topk(logits, 2)
            margin = (top2_values[0] - top2_values[1]).item()
        return margin
    
    def compute_feature_distance(self, image, user_id):
        """计算与真实数据原型的特征距离"""
        if self.feature_bank is None:
            return 0.5  # 默认中等距离
        
        # 提取特征
        features = self.extract_features(self.classifiers[0], image.unsqueeze(0))
        if features is None:
            return 0.5
        
        features = features.cpu().numpy().flatten()
        
        # 获取该用户的真实数据原型
        if user_id in self.feature_bank:
            user_prototype = self.feature_bank[user_id]['mean']
            user_cov = self.feature_bank[user_id]['cov']
            
            # 马氏距离（考虑协方差）
            try:
                dist = mahalanobis(features, user_prototype, np.linalg.inv(user_cov))
            except:
                # 如果协方差矩阵奇异，使用欧氏距离
                dist = np.linalg.norm(features - user_prototype)
            
            # 归一化到[0, 1]
            normalized_dist = 1 / (1 + np.exp(-0.1 * (dist - 10)))
            return normalized_dist
        
        return 0.5
    
    def compute_augmentation_consistency(self, classifier, image, num_augmentations=5):
        """测试轻微增强下的预测稳定性"""
        classifier.eval()
        
        # 定义轻微增强
        light_augment = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(num_augmentations):
                # 应用增强
                aug_image = image.clone()
                if aug_image.dim() == 3:  # CHW格式
                    # 转为PIL再转回
                    pil_img = transforms.ToPILImage()(aug_image)
                    aug_pil = light_augment(pil_img)
                    aug_tensor = transforms.ToTensor()(aug_pil)
                    aug_tensor = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    )(aug_tensor)
                    aug_image = aug_tensor.to(self.device)
                
                output = classifier(aug_image.unsqueeze(0))
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, dim=1)
                predictions.append(pred.item())
                confidences.append(conf.item())
        
        # 计算稳定性
        unique_preds = len(set(predictions))
        stability = 1.0 / unique_preds  # 预测越一致，稳定性越高
        avg_conf = np.mean(confidences)
        
        return stability * avg_conf, predictions[0]  # 返回稳定性分数和主要预测
    
    def filter_samples(self, images, user_ids, config):
        """
        综合多指标筛选样本
        
        Args:
            images: 批量图像 tensor [B, C, H, W]
            user_ids: 对应的用户ID列表
            config: 筛选配置
                - consistency_threshold: 一致性阈值
                - margin_threshold: margin阈值
                - feature_dist_range: 特征距离范围 [min, max]
                - augment_stability_threshold: 增强稳定性阈值
                - weights: 各指标权重
        
        Returns:
            selected_indices: 通过筛选的样本索引
            scores: 各样本的综合评分
        """
        batch_size = images.shape[0]
        scores = []
        details = []
        
        for i in range(batch_size):
            image = images[i]
            user_id = user_ids[i]
            
            # 1. 多模型一致性
            consistency_score, is_consistent, avg_conf = self.compute_consistency_score(image, user_id)
            
            # 2. Margin评分
            margin = self.compute_margin_score(self.classifiers[0], image)
            margin_score = min(margin / 5.0, 1.0)  # 归一化
            
            # 3. 特征距离
            feature_dist = self.compute_feature_distance(image, user_id)
            # 理想距离在[0.3, 0.7]范围（不太近也不太远）
            if config['feature_dist_range'][0] <= feature_dist <= config['feature_dist_range'][1]:
                feature_score = 1.0
            else:
                feature_score = 0.5
            
            # 4. 增强稳定性
            aug_score, _ = self.compute_augmentation_consistency(self.classifiers[0], image)
            
            # 综合评分
            weights = config.get('weights', {
                'consistency': 0.35,
                'margin': 0.25,
                'feature': 0.20,
                'augmentation': 0.20
            })
            
            total_score = (
                weights['consistency'] * consistency_score +
                weights['margin'] * margin_score +
                weights['feature'] * feature_score +
                weights['augmentation'] * aug_score
            )
            
            scores.append(total_score)
            details.append({
                'user_id': user_id,
                'consistency': consistency_score,
                'margin': margin,
                'feature_dist': feature_dist,
                'aug_stability': aug_score,
                'total': total_score
            })
        
        # 筛选高分样本
        threshold = config.get('total_threshold', 0.7)
        selected_indices = [i for i, s in enumerate(scores) if s >= threshold]
        
        return selected_indices, scores, details


class ProgressiveGenerator:
    """渐进式生成策略 - 支持分布式生成"""
    
    def __init__(self, model, vae, transport, device, rank=0, world_size=1):
        self.model = model
        self.vae = vae
        self.transport = transport
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
    def generate_progressive(self, user_id, num_rounds=3, samples_per_user=300):
        """
        渐进式生成：逐步调整CFG和采样步数
        支持分布式生成，每个GPU生成一部分样本
        """
        all_samples = []
        
        # 分布式生成：每个GPU生成一部分
        per_gpu_samples = samples_per_user // self.world_size
        if self.rank < (samples_per_user % self.world_size):
            per_gpu_samples += 1
        
        configs = [
            {'cfg': 6.0, 'steps': 100, 'ratio': 0.5, 'name': 'diverse'},
            {'cfg': 8.0, 'steps': 150, 'ratio': 0.3, 'name': 'balanced'}, 
            {'cfg': 10.0, 'steps': 200, 'ratio': 0.2, 'name': 'quality'}
        ]
        
        # 按比例分配每一轮的样本数
        for config in configs:
            config['batch_size'] = max(1, int(per_gpu_samples * config['ratio']))
        
        for round_idx, config in enumerate(configs):
            if self.rank == 0:
                print(f"📍 第{round_idx+1}轮生成 ({config['name']}): CFG={config['cfg']}, Steps={config['steps']}, Samples={config['batch_size']}")
            
            if config['batch_size'] == 0:
                continue
            
            # 创建采样器
            sampler = Sampler(self.transport)
            sample_fn = sampler.sample_ode(
                sampling_method="dopri5",
                num_steps=config['steps'],
                atol=1e-6,
                rtol=1e-3,
                reverse=False,
                timestep_shift=0.1
            )
            
            # 生成样本
            with torch.no_grad():
                # 为了确保不同 GPU 生成不同样本，设置不同随机种子
                torch.manual_seed(42 + self.rank * 1000 + round_idx * 100)
                
                y = torch.tensor([user_id] * config['batch_size'], device=self.device)
                z = torch.randn(config['batch_size'], 32, 16, 16, device=self.device)
                
                # CFG采样
                if config['cfg'] > 1.0:
                    z_cfg = torch.cat([z, z], 0)
                    y_null = torch.tensor([31] * config['batch_size'], device=self.device)
                    y_cfg = torch.cat([y, y_null], 0)
                    
                    model_kwargs = dict(
                        y=y_cfg, 
                        cfg_scale=config['cfg'], 
                        cfg_interval=True, 
                        cfg_interval_start=0.11
                    )
                    
                    samples = sample_fn(z_cfg, self.model, **model_kwargs)
                    samples = samples[-1]
                    samples, _ = samples.chunk(2, dim=0)
                else:
                    samples = sample_fn(z, self.model, **dict(y=y))
                    samples = samples[-1]
                
                # 解码并收集所有GPU的结果
                if self.vae is not None:
                    images = self.vae.decode_to_images(samples)
                    all_samples.extend(images)
        
        return all_samples
    
    def save_samples_distributed(self, samples, user_id, output_dir):
        """分布式保存样本"""
        output_dir = Path(output_dir)
        user_dir = output_dir / f"ID_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # 每个GPU保存自己的样本
        for i, img in enumerate(samples):
            filename = f"generated_rank{self.rank}_idx{i:04d}.jpg"
            img.save(user_dir / filename, quality=95)
        
        if self.rank == 0:
            print(f"💾 Rank {self.rank} saved {len(samples)} samples for ID_{user_id}")


def build_feature_bank(real_data_dir, classifier, device, num_classes=31):
    """构建真实数据特征库"""
    print("📊 构建真实数据特征库...")
    feature_bank = {}
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for user_id in range(num_classes):
        user_dir = Path(real_data_dir) / f"User_{user_id:02d}"
        if not user_dir.exists():
            continue
        
        features_list = []
        image_files = list(user_dir.glob("*.png"))[:50]  # 每个用户取50个样本
        
        for img_file in image_files:
            img = Image.open(img_file).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 提取特征
            classifier.eval()
            with torch.no_grad():
                if hasattr(classifier, 'backbone'):
                    feat = classifier.backbone(img_tensor)
                    features_list.append(feat.cpu().numpy().flatten())
        
        if features_list:
            features_array = np.array(features_list)
            feature_bank[user_id] = {
                'mean': np.mean(features_array, axis=0),
                'cov': np.cov(features_array.T),
                'samples': features_array
            }
    
    print(f"✅ 特征库构建完成，包含 {len(feature_bank)} 个用户")
    return feature_bank


def load_trained_classifier(checkpoint_path, num_classes, device):
    """加载已训练的分类器"""
    print(f"Loading classifier from: {checkpoint_path}")
    
    # 创建模型
    model = DomainAdaptiveClassifier(num_classes=num_classes, dropout_rate=0.3)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Advanced generation and filtering with DDP')
    
    # 模型路径
    parser.add_argument('--dit_checkpoint', type=str, 
                       default='./LightningDiT/checkpoints/lightningdit-xl-imagenet256-64ep.pt')
    parser.add_argument('--classifier_checkpoint', type=str, 
                       default='./calibrated_classifier/best_calibrated_model.pth',
                       help='已训练的分类器检查点')
    parser.add_argument('--vavae_checkpoint', type=str,
                       default='./LightningDiT/checkpoints/vavae_checkpoint.pth')
    parser.add_argument('--config', type=str, 
                       default='configs/dit_s_microdoppler.yaml')
    
    # 生成配置  
    parser.add_argument('--output_dir', type=str, 
                       default='/kaggle/working/advanced_filtered_samples')
    parser.add_argument('--samples_per_user', type=int, default=300,
                       help='每个用户生成的样本数')
    parser.add_argument('--target_users', type=str, nargs='+', 
                       default=[f'{i}' for i in range(31)],
                       help='目标用户ID列表')
    
    # 筛选配置
    parser.add_argument('--confidence_threshold', type=float, default=0.95)
    parser.add_argument('--feature_similarity_threshold', type=float, default=0.8)
    parser.add_argument('--total_threshold', type=float, default=0.8)
    parser.add_argument('--max_samples_per_user', type=int, default=150,
                       help='筛选后每个用户保留的最大样本数')
    
    args = parser.parse_args()
    
    # 初始化DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print("🚀 高级分布式生成和筛选系统启动")
        print(f"📊 使用 {world_size} 个 GPU 并行生成")
        print(f"🎯 筛选阈值: 置信度>{args.confidence_threshold}")
        print(f"💾 输出目录: {args.output_dir}")
    
    try:
        # 加载配置
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 加载 VA-VAE
        if rank == 0:
            print("📋 加载 VA-VAE...")
        vae = SimplifiedVAVAE.from_pretrained(args.vavae_checkpoint)
        vae.to(device)
        vae.eval()
        
        # 加载 DiT 模型
        if rank == 0:
            print("📋 加载 DiT 模型...")
        dit_model = LightningDiT_models['LightningDiT-XL/2'](
            input_size=16,
            num_classes=32  # 31个用户 + 1个空类
        )
        dit_checkpoint = torch.load(args.dit_checkpoint, map_location='cpu')
        dit_model.load_state_dict(dit_checkpoint['model'])
        dit_model.to(device)
        
        # 包装为DDP
        if world_size > 1:
            dit_model = DDP(dit_model, device_ids=[local_rank])
        
        # 加载分类器
        if rank == 0:
            print("📋 加载已训练的分类器...")
        classifier = load_trained_classifier(args.classifier_checkpoint, 31, device)
        
        # 创建 transport
        transport = create_transport(
            path_type="Linear",
            prediction="velocity", 
            loss_weight=None,
            train_eps=None
        )
        
        # 创建生成器和筛选器
        generator = ProgressiveGenerator(dit_model, vae, transport, device, rank, world_size)
        sample_filter = AdvancedSampleFilter([classifier], device)
        
        # 主生成循环
        target_users = [int(uid) for uid in args.target_users]
        
        for user_id in target_users:
            if rank == 0:
                print(f"\n📍 开始生成用户 ID_{user_id} 的数据...")
            
            # 生成样本
            generated_samples = generator.generate_progressive(
                user_id=user_id, 
                samples_per_user=args.samples_per_user
            )
            
            if rank == 0:
                print(f"🎨 Rank {rank} 生成了 {len(generated_samples)} 个样本")
            
            # 筛选高质量样本
            if len(generated_samples) > 0:
                # 转换为 tensor
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                sample_tensors = []
                for img in generated_samples:
                    if isinstance(img, Image.Image):
                        tensor = transform(img)
                        sample_tensors.append(tensor)
                
                if sample_tensors:
                    batch_tensors = torch.stack(sample_tensors).to(device)
                    user_ids = [user_id] * len(sample_tensors)
                    
                    # 简化的筛选：只用置信度
                    selected_samples = []
                    classifier.eval()
                    
                    with torch.no_grad():
                        for i, img_tensor in enumerate(batch_tensors):
                            logits, _ = classifier(img_tensor.unsqueeze(0))
                            probs = F.softmax(logits, dim=1)
                            confidence, pred = torch.max(probs, dim=1)
                            
                            if pred.item() == user_id and confidence.item() >= args.confidence_threshold:
                                selected_samples.append(generated_samples[i])
                    
                    # 限制数量
                    if len(selected_samples) > args.max_samples_per_user:
                        selected_samples = selected_samples[:args.max_samples_per_user]
                    
                    # 保存筛选后的样本
                    generator.save_samples_distributed(selected_samples, user_id, args.output_dir)
                    
                    if rank == 0:
                        print(f"✅ 筛选完成：{len(generated_samples)} -> {len(selected_samples)} 个高质量样本")
        
        if rank == 0:
            print("\n🎉 所有用户的数据生成完成！")
            print(f"💾 结果保存在: {args.output_dir}")
    
    except Exception as e:
        if rank == 0:
            print(f"❌ 错误: {e}")
        raise
    
    finally:
        cleanup_ddp()
    
if __name__ == "__main__":
    main()
