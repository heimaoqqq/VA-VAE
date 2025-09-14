"""
测试分类器在真实数据上的性能
验证分类器是否真正学习了微多普勒特征而非过拟合
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


class RealMicroDopplerDataset(Dataset):
    """真实微多普勒数据集，适配ID_*目录格式"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 扫描所有ID_*或User_*目录
        print(f"Scanning directory: {self.data_dir}")
        id_dirs = list(self.data_dir.glob("ID_*"))
        user_dirs = list(self.data_dir.glob("User_*"))
        all_dirs = id_dirs + user_dirs
        print(f"Found directories: {[d.name for d in all_dirs]}")
        
        for user_dir in sorted(all_dirs):
            if user_dir.is_dir():
                if user_dir.name.startswith("ID_"):
                    # ID_1 -> user_id 0, ID_2 -> user_id 1, etc.
                    user_id = int(user_dir.name.split('_')[1]) - 1
                elif user_dir.name.startswith("User_"):
                    # User_00 -> user_id 0, User_01 -> user_id 1, etc.
                    user_id = int(user_dir.name.split('_')[1])
                else:
                    continue
                print(f"Processing {user_dir.name} -> user_id {user_id}")
                
                # 扫描该用户的所有图像（支持多种格式）
                found_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    files = list(user_dir.glob(ext))
                    found_files.extend(files)
                    if files:
                        print(f"  Found {len(files)} {ext} files")
                
                for img_path in sorted(found_files):
                    self.samples.append((str(img_path), user_id))
                    
                if not found_files:
                    print(f"  WARNING: No image files found in {user_dir}")
                    # List all files to debug
                    all_files = list(user_dir.iterdir())
                    print(f"  Available files: {[f.name for f in all_files[:5]]}...")  # Show first 5
        
        print(f"Found {len(self.samples)} real samples from {len(set([s[1] for s in self.samples]))} users")
        
        # 统计每个用户的样本数
        user_counts = defaultdict(int)
        for _, user_id in self.samples:
            user_counts[user_id] += 1
        
        print("\nSamples per user:")
        for user_id in sorted(user_counts.keys()):
            print(f"  User {user_id}: {user_counts[user_id]} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, true_label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, true_label, img_path


def load_classifier(model_path, device):
    """加载训练好的分类器"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 导入ImprovedClassifier类
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from improved_classifier_training import ImprovedClassifier
    
    # 创建ImprovedClassifier模型 - 匹配训练时的架构
    model = ImprovedClassifier(
        num_classes=checkpoint['num_classes'],
        backbone='resnet18',
        dropout_rate=0.5,
        freeze_layers='minimal'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded classifier from {model_path}")
    print(f"Training validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model, checkpoint


def test_on_real_data(model, dataloader, device):
    """在真实数据上测试分类器"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_paths = []
    
    user_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'confidences': []})
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing on real data")
        for images, true_labels, img_paths in pbar:
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predictions = torch.max(probabilities, dim=1)
            
            # 转换为numpy
            predictions_np = predictions.cpu().numpy()
            true_labels_np = true_labels.numpy()
            max_probs_np = max_probs.cpu().numpy()
            
            # 记录结果
            for i in range(len(predictions_np)):
                pred = predictions_np[i]
                true_label = true_labels_np[i]
                confidence = max_probs_np[i]
                
                all_predictions.append(pred)
                all_labels.append(true_label)
                all_confidences.append(confidence)
                all_paths.append(img_paths[i])
                
                # 更新用户统计
                user_stats[true_label]['total'] += 1
                user_stats[true_label]['confidences'].append(confidence)
                if pred == true_label:
                    user_stats[true_label]['correct'] += 1
            
            # 更新进度条
            correct_so_far = sum([1 for p, l in zip(all_predictions, all_labels) if p == l])
            accuracy = correct_so_far / len(all_predictions) if all_predictions else 0
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            
            pbar.set_postfix({
                'Acc': f'{accuracy*100:.1f}%',
                'AvgConf': f'{avg_confidence:.3f}'
            })
    
    return all_predictions, all_labels, all_confidences, all_paths, user_stats


def analyze_results(predictions, labels, confidences, user_stats):
    """分析测试结果"""
    
    # 总体准确率
    correct = sum([1 for p, l in zip(predictions, labels) if p == l])
    total = len(predictions)
    
    if total == 0:
        print("❌ ERROR: No samples found for testing!")
        print("Please check the dataset path and file formats.")
        return None
        
    overall_accuracy = correct / total
    
    # 置信度分析
    avg_confidence = np.mean(confidences)
    confidence_std = np.std(confidences)
    
    # 高置信度样本（>0.9）
    high_conf_mask = np.array(confidences) > 0.9
    high_conf_count = np.sum(high_conf_mask)
    high_conf_ratio = high_conf_count / total
    
    # 高置信度样本的准确率
    if high_conf_count > 0:
        high_conf_correct = sum([1 for i, (p, l) in enumerate(zip(predictions, labels)) 
                                if high_conf_mask[i] and p == l])
        high_conf_accuracy = high_conf_correct / high_conf_count
    else:
        high_conf_accuracy = 0
    
    print("\n" + "="*60)
    print("REAL DATA TEST RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.3f} ± {confidence_std:.3f}")
    
    print(f"\n🎯 High-confidence samples (>0.9):")
    print(f"  Count: {high_conf_count} ({high_conf_ratio*100:.1f}% of total)")
    print(f"  Accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)")
    
    # 每用户准确率
    print(f"\n👥 Per-user accuracy:")
    user_accuracies = []
    for user_id in sorted(user_stats.keys()):
        stats = user_stats[user_id]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            avg_conf = np.mean(stats['confidences'])
            user_accuracies.append(acc)
            print(f"  User {user_id:2d}: {stats['correct']:3d}/{stats['total']:3d} "
                  f"({acc*100:5.1f}%) | Avg conf: {avg_conf:.3f}")
    
    # 用户准确率分布
    if user_accuracies:
        print(f"\n📈 User accuracy distribution:")
        print(f"  Min: {min(user_accuracies)*100:.1f}%")
        print(f"  Max: {max(user_accuracies)*100:.1f}%")
        print(f"  Mean: {np.mean(user_accuracies)*100:.1f}%")
        print(f"  Std: {np.std(user_accuracies)*100:.1f}%")
    
    return {
        'overall_accuracy': overall_accuracy,
        'avg_confidence': avg_confidence,
        'high_conf_ratio': high_conf_ratio,
        'high_conf_accuracy': high_conf_accuracy,
        'user_accuracies': user_accuracies
    }


def evaluate_classifier_reliability(results):
    """根据测试结果评估分类器可靠性"""
    
    print("\n" + "="*60)
    print("CLASSIFIER RELIABILITY ASSESSMENT")
    print("="*60)
    
    overall_acc = results['overall_accuracy']
    high_conf_acc = results['high_conf_accuracy']
    user_acc_std = np.std(results['user_accuracies']) if results['user_accuracies'] else 0
    
    # 判断标准
    if overall_acc >= 0.95:
        print("✅ EXCELLENT: Overall accuracy ≥ 95%")
        print("   → Classifier learned correct features")
        print("   → Generated sample evaluation is reliable")
        reliability = "HIGHLY RELIABLE"
        
    elif overall_acc >= 0.85:
        print("✅ GOOD: Overall accuracy 85-95%")
        print("   → Classifier is generally reliable")
        print("   → Minor overfitting possible but acceptable")
        reliability = "RELIABLE"
        
    elif overall_acc >= 0.70:
        print("⚠️ MODERATE: Overall accuracy 70-85%")
        print("   → Some overfitting likely")
        print("   → Generated sample evaluation should be cautious")
        reliability = "MODERATELY RELIABLE"
        
    else:
        print("❌ POOR: Overall accuracy < 70%")
        print("   → Significant overfitting detected")
        print("   → Generated sample evaluation NOT reliable")
        reliability = "UNRELIABLE"
    
    # 额外检查
    if user_acc_std > 0.2:
        print("\n⚠️ WARNING: High variance in per-user accuracy")
        print("   → Classifier may be biased toward certain users")
    
    if high_conf_acc < overall_acc - 0.1:
        print("\n⚠️ WARNING: High-confidence samples less accurate")
        print("   → Confidence calibration issues")
    
    print(f"\n🏁 FINAL VERDICT: Classifier is {reliability}")
    
    return reliability


def plot_results(predictions, labels, confidences, user_stats, output_dir):
    """可视化测试结果"""
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 置信度分布
    axes[0, 0].hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.9, color='red', linestyle='--', label='High conf threshold')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Confidence Distribution on Real Data')
    axes[0, 0].legend()
    
    # 2. 每用户准确率
    user_ids = sorted(user_stats.keys())
    user_accs = [user_stats[uid]['correct']/user_stats[uid]['total'] 
                 if user_stats[uid]['total'] > 0 else 0 
                 for uid in user_ids]
    
    axes[0, 1].bar(user_ids, user_accs)
    axes[0, 1].axhline(y=np.mean(user_accs), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(user_accs):.3f}')
    axes[0, 1].set_xlabel('User ID')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Per-User Accuracy on Real Data')
    axes[0, 1].legend()
    
    # 3. 混淆矩阵（简化版，只显示前10个用户）
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels[:1000], predictions[:1000])  # 前1000个样本
    im = axes[1, 0].imshow(cm[:10, :10], cmap='Blues')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix (First 10 Users)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 准确率vs置信度
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_counts = []
    
    for i in range(len(conf_bins)-1):
        mask = (np.array(confidences) >= conf_bins[i]) & (np.array(confidences) < conf_bins[i+1])
        if np.sum(mask) > 0:
            bin_preds = [predictions[j] for j in range(len(predictions)) if mask[j]]
            bin_labels = [labels[j] for j in range(len(labels)) if mask[j]]
            bin_acc = sum([1 for p, l in zip(bin_preds, bin_labels) if p == l]) / len(bin_preds)
            bin_accs.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0)
            bin_counts.append(0)
    
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    axes[1, 1].bar(bin_centers, bin_accs, width=0.08, alpha=0.7)
    axes[1, 1].set_xlabel('Confidence Bin')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Confidence')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_data_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir / 'real_data_test_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='Test classifier on real data')
    parser.add_argument('--data_dir', type=str, required=True, help='Real data directory (with ID_* folders)')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to trained classifier')
    parser.add_argument('--output_dir', type=str, default='./real_data_test', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载分类器
    print("\nLoading classifier...")
    model, checkpoint = load_classifier(args.classifier_path, device)
    
    # 创建数据变换（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载真实数据
    print("\nLoading real data...")
    dataset = RealMicroDopplerDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 测试
    print("\nTesting classifier on real data...")
    predictions, labels, confidences, paths, user_stats = test_on_real_data(model, dataloader, device)
    
    # 分析结果
    results = analyze_results(predictions, labels, confidences, user_stats)
    
    # 评估可靠性
    reliability = evaluate_classifier_reliability(results)
    
    # 可视化
    plot_results(predictions, labels, confidences, user_stats, output_dir)
    
    # 保存结果
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': float(results['overall_accuracy']),
            'avg_confidence': float(results['avg_confidence']),
            'high_conf_ratio': float(results['high_conf_ratio']),
            'high_conf_accuracy': float(results['high_conf_accuracy']),
            'reliability': reliability,
            'total_samples': len(predictions)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results, reliability


if __name__ == "__main__":
    main()
