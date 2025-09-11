#!/usr/bin/env python3
"""
微多普勒生成样本的综合评估框架
融合身份保持度、类内多样性、特征覆盖度和频谱一致性
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import lpips
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


class ComprehensiveGenerationEvaluator:
    """
    综合生成质量评估器
    评估三个维度：身份保持度、类内多样性、特征覆盖度
    """
    
    def __init__(self, classifier_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载分类器用于身份评估
        self.classifier = self.load_classifier(classifier_path)
        
        # 初始化LPIPS用于感知距离
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_classifier(self, model_path):
        """加载训练好的分类器"""
        from improved_classifier_training import ImprovedClassifier
        
        model = ImprovedClassifier(num_classes=31)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_images(self, image_paths):
        """批量加载图像"""
        images = []
        valid_paths = []
        
        if not image_paths:
            print("Warning: No image paths provided")
            return torch.empty(0, 3, 224, 224).to(self.device), []
        
        for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not images:
            print("Warning: No images successfully loaded")
            return torch.empty(0, 3, 224, 224).to(self.device), []
        
        return torch.stack(images).to(self.device), valid_paths
    
    def compute_identity_preservation(self, generated_samples, target_user_id):
        """
        1. 身份保持度评估
        衡量生成样本是否保持目标用户的身份特征
        """
        with torch.no_grad():
            logits = self.classifier(generated_samples)
            predictions = torch.softmax(logits, dim=1)
            
            # 目标用户的预测概率
            target_confidence = predictions[:, target_user_id]
            
            # Top-1准确率
            top1_accuracy = (torch.argmax(predictions, dim=1) == target_user_id).float().mean()
            
            # 平均置信度
            avg_confidence = target_confidence.mean()
            
            # 置信度分布统计
            confidence_std = target_confidence.std()
            
        return {
            'identity_accuracy': top1_accuracy.item(),
            'identity_confidence': avg_confidence.item(),
            'confidence_std': confidence_std.item(),
            'identity_score': (top1_accuracy * avg_confidence).item()
        }
    
    def compute_intra_class_diversity(self, generated_samples, sample_size=50):
        """
        2. 类内多样性评估
        基于LPIPS的感知多样性 + 特征空间多样性
        """
        # 为了计算效率，随机采样部分样本
        if len(generated_samples) > sample_size:
            indices = torch.randperm(len(generated_samples))[:sample_size]
            samples = generated_samples[indices]
        else:
            samples = generated_samples
        
        n_samples = len(samples)
        
        # LPIPS感知距离
        lpips_distances = []
        with torch.no_grad():
            for i in range(min(n_samples, 20)):  # 限制计算量
                for j in range(i+1, min(n_samples, 20)):
                    dist = self.lpips_fn(samples[i:i+1], samples[j:j+1])
                    lpips_distances.append(dist.item())
        
        # 特征空间多样性
        with torch.no_grad():
            features, _ = self.classifier(samples, return_features=True)
            features_np = features.cpu().numpy()
            
            # 计算特征间的余弦距离
            cosine_sim_matrix = cosine_similarity(features_np)
            # 提取上三角矩阵（排除对角线）
            upper_tri_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
            cosine_distances = 1 - cosine_sim_matrix[upper_tri_indices]
        
        return {
            'mean_lpips_distance': np.mean(lpips_distances) if lpips_distances else 0,
            'std_lpips_distance': np.std(lpips_distances) if lpips_distances else 0,
            'mean_feature_distance': np.mean(cosine_distances),
            'std_feature_distance': np.std(cosine_distances),
            'diversity_score': np.mean(lpips_distances) if lpips_distances else 0
        }
    
    def compute_feature_coverage(self, generated_samples, real_user_samples, k=5, threshold=0.5):
        """
        3. 特征空间覆盖度
        基于改进的Precision & Recall思想
        """
        with torch.no_grad():
            # 提取特征
            gen_features, _ = self.classifier(generated_samples, return_features=True)
            real_features, _ = self.classifier(real_user_samples, return_features=True)
            
            gen_features_np = gen_features.cpu().numpy()
            real_features_np = real_features.cpu().numpy()
        
        # 构建k-NN
        nbrs_real = NearestNeighbors(n_neighbors=min(k, len(real_features_np))).fit(real_features_np)
        nbrs_gen = NearestNeighbors(n_neighbors=min(k, len(gen_features_np))).fit(gen_features_np)
        
        # Precision: 生成样本有多少在真实数据流形附近
        if len(gen_features_np) > 0:
            distances_gen_to_real, _ = nbrs_real.kneighbors(gen_features_np)
            precision = (distances_gen_to_real[:, -1] < threshold).mean()
        else:
            precision = 0
        
        # Recall: 真实样本有多少被生成样本覆盖
        if len(real_features_np) > 0 and len(gen_features_np) > 0:
            distances_real_to_gen, _ = nbrs_gen.kneighbors(real_features_np)
            recall = (distances_real_to_gen[:, -1] < threshold).mean()
        else:
            recall = 0
        
        # F1 Score
        f1_score = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'coverage_score': f1_score
        }
    
    
    def comprehensive_evaluate(self, generated_dir, real_user_dir, target_user_id, 
                              max_samples=200):
        """
        综合评估主函数
        """
        print(f"Starting comprehensive evaluation for User {target_user_id}...")
        
        # 加载生成样本
        gen_paths = list(Path(generated_dir).glob("*.jpg")) + list(Path(generated_dir).glob("*.png"))
        if len(gen_paths) > max_samples:
            gen_paths = gen_paths[:max_samples]
        
        generated_samples, _ = self.load_images(gen_paths)
        
        # 加载真实用户样本
        real_paths = list(Path(real_user_dir).glob("*.jpg")) + list(Path(real_user_dir).glob("*.png"))
        if len(real_paths) > max_samples:
            real_paths = real_paths[:max_samples]
        
        real_user_samples, _ = self.load_images(real_paths)
        
        print(f"Loaded {len(generated_samples)} generated samples and {len(real_user_samples)} real samples")
        
        # 检查是否有足够的样本进行评估
        if len(generated_samples) == 0:
            print("❌ No generated samples found or loaded successfully")
            print("Please check the generated_dir path and image files")
            return None
        
        if len(real_user_samples) == 0:
            print("❌ No real user samples found or loaded successfully")
            print("Please check the real_user_dir path and image files")
            return None
        
        # 三个维度的评估
        results = {}
        
        # 1. 身份保持度
        print("Evaluating identity preservation...")
        results['identity'] = self.compute_identity_preservation(generated_samples, target_user_id)
        
        # 2. 类内多样性  
        print("Evaluating intra-class diversity...")
        results['diversity'] = self.compute_intra_class_diversity(generated_samples)
        
        # 3. 特征覆盖度
        print("Evaluating feature coverage...")
        results['coverage'] = self.compute_feature_coverage(generated_samples, real_user_samples)
        
        # 计算综合得分
        results['overall'] = self.compute_overall_score(results)
        
        return results
    
    def compute_overall_score(self, results):
        """
        计算综合得分
        权衡三个维度：身份保持、多样性、覆盖度
        """
        # 提取各维度得分
        identity_score = results['identity']['identity_score']
        diversity_score = min(1.0, results['diversity']['diversity_score'] * 10)  # LPIPS通常很小
        coverage_score = results['coverage']['coverage_score'] 
        
        # 权重设计：身份保持最重要，其次是多样性
        weights = {
            'identity': 0.5,    # 50% - 必须保持用户身份
            'diversity': 0.3,   # 30% - 样本间多样性
            'coverage': 0.2,    # 20% - 特征空间覆盖
        }
        
        overall_score = (
            weights['identity'] * identity_score +
            weights['diversity'] * diversity_score + 
            weights['coverage'] * coverage_score
        )
        
        return {
            'overall_score': overall_score,
            'identity_component': weights['identity'] * identity_score,
            'diversity_component': weights['diversity'] * diversity_score,
            'coverage_component': weights['coverage'] * coverage_score,
            'weights': weights
        }
    
    def print_report(self, results, user_id):
        """打印评估报告"""
        print("\n" + "="*60)
        print(f"COMPREHENSIVE GENERATION EVALUATION - USER {user_id}")
        print("="*60)
        
        # 身份保持度
        identity = results['identity']
        print(f"\n🔍 IDENTITY PRESERVATION:")
        print(f"   • Accuracy: {identity['identity_accuracy']:.1%}")
        print(f"   • Confidence: {identity['identity_confidence']:.3f}")
        print(f"   • Identity Score: {identity['identity_score']:.3f}")
        
        # 多样性
        diversity = results['diversity'] 
        print(f"\n🎨 INTRA-CLASS DIVERSITY:")
        print(f"   • LPIPS Distance: {diversity['mean_lpips_distance']:.3f} ± {diversity['std_lpips_distance']:.3f}")
        print(f"   • Feature Distance: {diversity['mean_feature_distance']:.3f} ± {diversity['std_feature_distance']:.3f}")
        print(f"   • Diversity Score: {diversity['diversity_score']:.3f}")
        
        # 覆盖度
        coverage = results['coverage']
        print(f"\n📊 FEATURE COVERAGE:")
        print(f"   • Precision: {coverage['precision']:.3f}")
        print(f"   • Recall: {coverage['recall']:.3f}")
        print(f"   • F1-Score: {coverage['f1_score']:.3f}")
        
        # 综合得分
        overall = results['overall']
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   • Overall Score: {overall['overall_score']:.3f}")
        print(f"   • Identity Component: {overall['identity_component']:.3f} (50%)")
        print(f"   • Diversity Component: {overall['diversity_component']:.3f} (30%)")
        print(f"   • Coverage Component: {overall['coverage_component']:.3f} (20%)")
        
        # 质量评估
        score = overall['overall_score']
        if score >= 0.7:
            grade = "🟢 EXCELLENT"
            recommendation = "High-quality generation with good identity-diversity balance"
        elif score >= 0.5:
            grade = "🟡 GOOD" 
            recommendation = "Acceptable quality, consider improving diversity or coverage"
        elif score >= 0.3:
            grade = "🟠 MODERATE"
            recommendation = "Needs improvement in multiple aspects"
        else:
            grade = "🔴 POOR"
            recommendation = "Significant issues with generation quality"
        
        print(f"\n📋 QUALITY GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive generation evaluation')
    parser.add_argument('--classifier_path', required=True, help='Path to trained classifier')
    parser.add_argument('--generated_dir', required=True, help='Generated samples directory')
    parser.add_argument('--real_user_dir', required=True, help='Real user samples directory')  
    parser.add_argument('--user_id', type=int, required=True, help='Target user ID (0-30)')
    parser.add_argument('--max_samples', type=int, default=200, help='Maximum samples to evaluate')
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveGenerationEvaluator(args.classifier_path)
    results = evaluator.comprehensive_evaluate(
        args.generated_dir, 
        args.real_user_dir,
        args.user_id,
        args.max_samples
    )
    
    evaluator.print_report(results, args.user_id)


if __name__ == "__main__":
    main()
