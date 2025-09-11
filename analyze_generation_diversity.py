#!/usr/bin/env python3
"""
分析生成样本的多样性：验证是否为简单重构vs真正的新样本
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

class DiversityAnalyzer:
    def __init__(self, feature_extractor_path, real_data_dir, generated_data_dir):
        """
        多样性分析器
        
        Args:
            feature_extractor_path: 训练好的分类器路径
            real_data_dir: 真实数据目录
            generated_data_dir: 生成数据目录
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载特征提取器（使用训练好的分类器）
        self.feature_extractor = self.load_feature_extractor(feature_extractor_path)
        
        self.real_data_dir = Path(real_data_dir)
        self.generated_data_dir = Path(generated_data_dir)
        
    def load_feature_extractor(self, model_path):
        """加载训练好的分类器作为特征提取器"""
        from improved_classifier_training import ImprovedClassifier
        
        model = ImprovedClassifier(num_classes=31)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def extract_features(self, image_paths):
        """提取图像特征向量"""
        features = []
        
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="Extracting features"):
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256))
                
                # 转换为tensor
                img_tensor = torch.from_numpy(np.array(img)).float()
                img_tensor = img_tensor.permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.feature_extractor.extract_features(img_tensor)
                features.append(feature.cpu().numpy().flatten())
        
        return np.array(features)
    
    def compute_nearest_neighbor_distances(self, generated_features, real_features):
        """
        计算每个生成样本到最近真实样本的距离
        距离越小 = 越像重构，距离越大 = 越有新颖性
        """
        distances = []
        
        for gen_feat in generated_features:
            # 计算与所有真实样本的余弦相似度
            similarities = cosine_similarity([gen_feat], real_features)[0]
            # 转换为距离（1 - 相似度）
            min_distance = 1 - np.max(similarities)
            distances.append(min_distance)
            
        return np.array(distances)
    
    def analyze_intra_class_diversity(self, features, user_ids):
        """分析每个用户内部生成样本的多样性"""
        user_diversities = {}
        
        for user_id in np.unique(user_ids):
            user_mask = user_ids == user_id
            user_features = features[user_mask]
            
            if len(user_features) < 2:
                user_diversities[user_id] = 0.0
                continue
                
            # 计算用户内部样本的平均距离
            similarities = cosine_similarity(user_features)
            # 排除对角线（自己与自己）
            np.fill_diagonal(similarities, 0)
            
            # 平均距离 = 1 - 平均相似度
            avg_distance = 1 - np.mean(similarities[similarities > 0])
            user_diversities[user_id] = avg_distance
            
        return user_diversities
    
    def plot_diversity_analysis(self, real_features, gen_features, gen_user_ids, 
                               real_user_ids, output_dir):
        """可视化多样性分析结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. t-SNE可视化特征分布
        print("Computing t-SNE visualization...")
        all_features = np.vstack([real_features, gen_features])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(all_features)
        
        real_embedded = embedded[:len(real_features)]
        gen_embedded = embedded[len(real_features):]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制真实数据点
        scatter_real = plt.scatter(real_embedded[:, 0], real_embedded[:, 1], 
                                  c=real_user_ids, cmap='tab20', alpha=0.6, 
                                  s=20, label='Real Data')
        
        # 绘制生成数据点  
        scatter_gen = plt.scatter(gen_embedded[:, 0], gen_embedded[:, 1],
                                 c=gen_user_ids, cmap='tab20', alpha=0.8,
                                 s=40, marker='^', label='Generated Data')
        
        plt.colorbar(scatter_real, label='User ID')
        plt.legend()
        plt.title('Feature Distribution: Real vs Generated Data')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 最近邻距离分布
        nn_distances = self.compute_nearest_neighbor_distances(gen_features, real_features)
        
        plt.figure(figsize=(10, 6))
        plt.hist(nn_distances, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(nn_distances), color='red', linestyle='--', 
                   label=f'Mean Distance: {np.mean(nn_distances):.3f}')
        plt.axvline(np.median(nn_distances), color='green', linestyle='--',
                   label=f'Median Distance: {np.median(nn_distances):.3f}')
        plt.xlabel('Distance to Nearest Real Sample')
        plt.ylabel('Number of Generated Samples')
        plt.title('Generated Samples: Distance to Nearest Real Sample')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'nearest_neighbor_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 用户内多样性分析
        gen_diversities = self.analyze_intra_class_diversity(gen_features, gen_user_ids)
        real_diversities = self.analyze_intra_class_diversity(real_features, real_user_ids)
        
        user_ids = sorted(gen_diversities.keys())
        gen_div_values = [gen_diversities[uid] for uid in user_ids]
        real_div_values = [real_diversities.get(uid, 0) for uid in user_ids]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(user_ids))
        width = 0.35
        
        plt.bar(x - width/2, real_div_values, width, label='Real Data', alpha=0.7)
        plt.bar(x + width/2, gen_div_values, width, label='Generated Data', alpha=0.7)
        
        plt.xlabel('User ID')
        plt.ylabel('Intra-class Diversity Score')
        plt.title('Intra-class Diversity: Real vs Generated Data')
        plt.xticks(x, user_ids)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'intra_class_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'nearest_neighbor_distances': nn_distances,
            'generated_diversities': gen_diversities,
            'real_diversities': real_diversities,
            'tsne_embedding': embedded
        }
    
    def compute_diversity_metrics(self, real_data_dir, generated_data_dir, confidence_threshold=0.95):
        """计算全面的多样性指标"""
        print(f"Analyzing diversity for pre-selected samples...")
        
        # 收集图像路径 - 支持jpg和png格式
        real_paths = list(self.real_data_dir.glob("**/*.jpg")) + list(self.real_data_dir.glob("**/*.png"))
        
        # 对于已经筛选好的样本，直接获取所有png文件
        gen_paths = list(self.generated_data_dir.glob("**/*.png"))
        
        if len(gen_paths) == 0:
            print(f"No generated samples found in {generated_data_dir}")
            return None
            
        print(f"Found {len(real_paths)} real samples and {len(gen_paths)} pre-selected generated samples")
        
        # 提取特征
        print("Extracting features from real data...")
        real_features = self.extract_features(real_paths)
        
        print("Extracting features from generated data...")  
        gen_features = self.extract_features(gen_paths)
        
        # 提取用户ID
        real_user_ids = np.array([self.extract_user_id_from_path(p) for p in real_paths])
        gen_user_ids = np.array([self.extract_user_id_from_path(p) for p in gen_paths])
        
        # 分析结果
        results = self.plot_diversity_analysis(
            real_features, gen_features, gen_user_ids, real_user_ids,
            f"diversity_analysis_conf_{confidence_threshold}"
        )
        
        # 计算总结指标
        nn_distances = results['nearest_neighbor_distances']
        
        diversity_summary = {
            'confidence_threshold': confidence_threshold,
            'num_generated_samples': len(gen_paths),
            'mean_distance_to_real': np.mean(nn_distances),
            'median_distance_to_real': np.median(nn_distances),
            'std_distance_to_real': np.std(nn_distances),
            'reconstruction_ratio': np.sum(nn_distances < 0.1) / len(nn_distances),  # 很相似的比例
            'novel_ratio': np.sum(nn_distances > 0.3) / len(nn_distances),  # 新颖样本比例
        }
        
        return diversity_summary
    
    def extract_user_id_from_path(self, path):
        """从文件路径提取用户ID"""
        # 对于已保存的样本结构: /user_XX/sample_XXXXXX_confX.XXX.png
        path_parts = Path(path).parts
        
        # 查找包含user_的部分
        for part in path_parts:
            if 'user_' in part:
                try:
                    return int(part.split('user_')[1])
                except:
                    continue
        
        # 对于原始数据格式: ID1_case1_1_Doppler1.jpg
        stem = Path(path).stem
        if stem.startswith('ID'):
            try:
                # 提取ID后面的数字，映射到0-30范围
                id_num = int(stem.split('_')[0][2:])  # ID1 -> 1, ID10 -> 10
                return id_num - 1  # 转换为0-based索引
            except:
                pass
        
        # 如果路径中有user_，尝试提取
        if 'user_' in stem:
            try:
                return int(stem.split('user_')[1].split('_')[0])
            except:
                pass
        
        return 0
    
    def analyze_single_threshold(self, threshold=0.95):
        """分析单个置信度阈值的多样性"""
        print(f"Analyzing diversity for confidence threshold: {threshold}")
        
        result = self.compute_diversity_metrics(
            self.real_data_dir, self.generated_data_dir, threshold
        )
        
        return result
    
    def compare_multiple_thresholds(self, thresholds=[0.99, 0.95, 0.9, 0.8, 0.7]):
        """比较不同置信度阈值下的多样性"""
        results = []
        
        for threshold in thresholds:
            result = self.compute_diversity_metrics(
                self.real_data_dir, self.generated_data_dir, threshold
            )
            if result:
                results.append(result)
        
        # 绘制比较图
        if results:
            self.plot_threshold_comparison(results)
        
        return results
    
    def plot_threshold_comparison(self, results):
        """绘制不同阈值的比较结果"""
        thresholds = [r['confidence_threshold'] for r in results]
        mean_distances = [r['mean_distance_to_real'] for r in results]
        novel_ratios = [r['novel_ratio'] for r in results]
        reconstruction_ratios = [r['reconstruction_ratio'] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 平均距离
        axes[0].plot(thresholds, mean_distances, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('Mean Distance to Real Data')
        axes[0].set_title('Novelty vs Confidence Threshold')
        axes[0].grid(True, alpha=0.3)
        
        # 新颖样本比例
        axes[1].plot(thresholds, novel_ratios, 'o-', color='green', linewidth=2, markersize=8)
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('Novel Samples Ratio')
        axes[1].set_title('Novel Samples vs Confidence Threshold')
        axes[1].grid(True, alpha=0.3)
        
        # 重构样本比例
        axes[2].plot(thresholds, reconstruction_ratios, 'o-', color='red', linewidth=2, markersize=8)
        axes[2].set_xlabel('Confidence Threshold')
        axes[2].set_ylabel('Reconstruction Samples Ratio')
        axes[2].set_title('Reconstruction vs Confidence Threshold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze generation diversity')
    parser.add_argument('--model_path', required=True, 
                       help='Path to trained classifier (e.g., best_classifier_model.pth)')
    parser.add_argument('--real_data_dir', required=True, 
                       help='Real data directory (original dataset)')
    parser.add_argument('--generated_data_dir', required=True, 
                       help='Generated data directory (where high-confidence samples are saved)')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Single confidence threshold to analyze (default: 0.95)')
    parser.add_argument('--compare_multiple', action='store_true',
                       help='Compare multiple thresholds instead of single')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.99, 0.95, 0.9, 0.8, 0.7],
                       help='Multiple confidence thresholds to compare')
    
    args = parser.parse_args()
    
    analyzer = DiversityAnalyzer(args.model_path, args.real_data_dir, args.generated_data_dir)
    
    if args.compare_multiple:
        print("Starting multi-threshold diversity analysis...")
        results = analyzer.compare_multiple_thresholds(args.thresholds)
        
        # 输出总结
        print("\n" + "="*50)
        print("MULTI-THRESHOLD DIVERSITY SUMMARY")
        print("="*50)
        
        for result in results:
            conf = result['confidence_threshold']
            mean_dist = result['mean_distance_to_real']
            novel_ratio = result['novel_ratio']
            recon_ratio = result['reconstruction_ratio']
            
            print(f"\nConfidence {conf}:")
            print(f"  Samples: {result['num_generated_samples']}")
            print(f"  Mean distance to real: {mean_dist:.3f}")
            print(f"  Novel samples (>0.3 dist): {novel_ratio:.1%}")
            print(f"  Reconstruction samples (<0.1 dist): {recon_ratio:.1%}")
            
            # 判断质量
            if recon_ratio > 0.5:
                print("  ❌ HIGH RECONSTRUCTION - mostly memorizing training data")
            elif novel_ratio > 0.3:
                print("  ✅ GOOD DIVERSITY - generating novel samples")
            else:
                print("  🟡 MODERATE DIVERSITY - balanced but could be better")
    else:
        print(f"Starting single-threshold diversity analysis for confidence {args.threshold}...")
        result = analyzer.analyze_single_threshold(args.threshold)
        
        if result:
            print("\n" + "="*50)
            print(f"DIVERSITY ANALYSIS FOR CONFIDENCE {args.threshold}")
            print("="*50)
            
            mean_dist = result['mean_distance_to_real']
            novel_ratio = result['novel_ratio']
            recon_ratio = result['reconstruction_ratio']
            
            print(f"Total samples analyzed: {result['num_generated_samples']}")
            print(f"Mean distance to real data: {mean_dist:.3f}")
            print(f"Novel samples (distance >0.3): {novel_ratio:.1%}")
            print(f"Reconstruction samples (distance <0.1): {recon_ratio:.1%}")
            print(f"Standard deviation: {result['std_distance_to_real']:.3f}")
            
            # 详细判断
            print(f"\n📊 QUALITY ASSESSMENT:")
            if recon_ratio > 0.6:
                print("❌ POOR DIVERSITY: Mostly reconstructing training data")
                print("   Recommendation: Lower confidence threshold or use balanced selection")
            elif recon_ratio > 0.3:
                print("🟡 MODERATE DIVERSITY: Some reconstruction, some novelty")
                print("   Recommendation: Consider balanced selection to improve novelty")
            else:
                print("✅ GOOD DIVERSITY: Low reconstruction, good novelty")
                
            if novel_ratio > 0.4:
                print("✅ EXCELLENT NOVELTY: Many truly new samples")
            elif novel_ratio > 0.2:
                print("🟡 MODERATE NOVELTY: Some new samples")
            else:
                print("❌ LOW NOVELTY: Few truly new samples")
        else:
            print("❌ No samples found for analysis. Check your paths and confidence threshold.")

if __name__ == "__main__":
    main()
