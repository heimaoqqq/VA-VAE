"""
条件生成质量评估脚本
使用训练好的分类器评估生成样本，筛选置信度>90%的样本
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

class GeneratedSampleDataset(Dataset):
    """生成样本数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 扫描所有用户目录
        for user_dir in sorted(self.data_dir.glob("user_*")):
            if user_dir.is_dir():
                user_id = int(user_dir.name.split('_')[1])
                
                # 扫描该用户的所有生成图像
                for img_path in sorted(user_dir.glob("*.png")):
                    self.samples.append((str(img_path), user_id))
        
        print(f"Found {len(self.samples)} generated samples from {len(set([s[1] for s in self.samples]))} users")
    
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
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = timm.create_model('resnet18', pretrained=False, num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded classifier from {model_path}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model

def create_transforms():
    """创建数据变换（与分类器训练时保持一致）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def evaluate_samples(model, dataloader, device, confidence_threshold=0.9):
    """评估生成样本"""
    model.eval()
    
    results = {
        'predictions': [],
        'true_labels': [],
        'confidences': [],
        'max_probs': [],
        'image_paths': [],
        'correct': [],
        'high_confidence': []
    }
    
    user_stats = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'high_conf': 0, 'high_conf_correct': 0
    })
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
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
                max_prob = max_probs_np[i]
                img_path = img_paths[i]
                
                is_correct = (pred == true_label)
                is_high_conf = (max_prob >= confidence_threshold)
                
                results['predictions'].append(pred)
                results['true_labels'].append(true_label)
                results['confidences'].append(probabilities[i].cpu().numpy())
                results['max_probs'].append(max_prob)
                results['image_paths'].append(img_path)
                results['correct'].append(is_correct)
                results['high_confidence'].append(is_high_conf)
                
                # 更新用户统计
                user_stats[true_label]['total'] += 1
                if is_correct:
                    user_stats[true_label]['correct'] += 1
                if is_high_conf:
                    user_stats[true_label]['high_conf'] += 1
                    if is_correct:
                        user_stats[true_label]['high_conf_correct'] += 1
            
            # 更新进度条
            total_correct = sum(results['correct'])
            total_high_conf = sum(results['high_confidence'])
            accuracy = total_correct / len(results['correct']) if results['correct'] else 0
            high_conf_ratio = total_high_conf / len(results['high_confidence']) if results['high_confidence'] else 0
            
            pbar.set_postfix({
                'Acc': f'{accuracy*100:.1f}%',
                'HighConf': f'{high_conf_ratio*100:.1f}%'
            })
    
    return results, user_stats

def save_high_confidence_samples(results, output_dir, confidence_threshold=0.9):
    """保存高置信度样本"""
    output_dir = Path(output_dir)
    
    # 创建总目录
    high_conf_dir = output_dir / f"high_confidence_samples_threshold_{confidence_threshold}"
    high_conf_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个用户创建目录
    for user_id in range(31):  # 假设有31个用户
        user_dir = high_conf_dir / f"user_{user_id:02d}"
        user_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    saved_correct = 0
    
    for i, (is_high_conf, is_correct, img_path, true_label, pred) in enumerate(
        zip(results['high_confidence'], results['correct'], results['image_paths'], 
            results['true_labels'], results['predictions'])
    ):
        if is_high_conf:
            # 确定保存路径
            user_dir = high_conf_dir / f"user_{true_label:02d}"
            original_filename = Path(img_path).name
            
            # 添加预测信息到文件名
            name_parts = original_filename.split('.')
            if is_correct:
                new_filename = f"{name_parts[0]}_CORRECT_pred{pred}.{name_parts[1]}"
                saved_correct += 1
            else:
                new_filename = f"{name_parts[0]}_WRONG_pred{pred}.{name_parts[1]}"
            
            new_path = user_dir / new_filename
            
            # 复制文件
            shutil.copy2(img_path, new_path)
            saved_count += 1
    
    print(f"\nSaved {saved_count} high-confidence samples")
    print(f"  - Correct predictions: {saved_correct}")
    print(f"  - Wrong predictions: {saved_count - saved_correct}")
    print(f"  - Accuracy among high-confidence samples: {saved_correct/saved_count*100:.1f}%")
    
    return saved_count, saved_correct

def generate_evaluation_report(results, user_stats, output_dir, confidence_threshold=0.9):
    """生成评估报告"""
    output_dir = Path(output_dir)
    
    # 计算总体统计
    total_samples = len(results['predictions'])
    total_correct = sum(results['correct'])
    total_high_conf = sum(results['high_confidence'])
    high_conf_correct = sum(c and h for c, h in zip(results['correct'], results['high_confidence']))
    
    overall_accuracy = total_correct / total_samples
    high_conf_ratio = total_high_conf / total_samples
    high_conf_accuracy = high_conf_correct / total_high_conf if total_high_conf > 0 else 0
    
    # 生成报告
    report = {
        'evaluation_summary': {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'high_confidence_samples': total_high_conf,
            'high_confidence_ratio': high_conf_ratio,
            'high_confidence_correct': high_conf_correct,
            'high_confidence_accuracy': high_conf_accuracy,
            'confidence_threshold': confidence_threshold
        },
        'per_user_stats': {}
    }
    
    # 每用户统计
    for user_id in sorted(user_stats.keys()):
        stats = user_stats[user_id]
        user_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        user_high_conf_ratio = stats['high_conf'] / stats['total'] if stats['total'] > 0 else 0
        user_high_conf_accuracy = stats['high_conf_correct'] / stats['high_conf'] if stats['high_conf'] > 0 else 0
        
        report['per_user_stats'][f'user_{user_id:02d}'] = {
            'total_samples': stats['total'],
            'correct_predictions': stats['correct'],
            'accuracy': user_accuracy,
            'high_confidence_samples': stats['high_conf'],
            'high_confidence_ratio': user_high_conf_ratio,
            'high_confidence_correct': stats['high_conf_correct'],
            'high_confidence_accuracy': user_high_conf_accuracy
        }
    
    # 保存JSON报告
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成文本报告
    text_report_path = output_dir / 'evaluation_report.txt'
    with open(text_report_path, 'w') as f:
        f.write("=== CONDITIONAL GENERATION EVALUATION REPORT ===\n\n")
        f.write(f"Confidence Threshold: {confidence_threshold}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total Samples: {total_samples}\n")
        f.write(f"  Correct Predictions: {total_correct}\n")
        f.write(f"  Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)\n")
        f.write(f"  High-Confidence Samples: {total_high_conf}\n")
        f.write(f"  High-Confidence Ratio: {high_conf_ratio:.3f} ({high_conf_ratio*100:.1f}%)\n")
        f.write(f"  High-Confidence Accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)\n\n")
        
        f.write("PER-USER STATISTICS:\n")
        f.write("User | Total | Correct | Acc(%) | HighConf | HCRatio(%) | HCAcc(%)\n")
        f.write("-" * 70 + "\n")
        
        for user_id in sorted(user_stats.keys()):
            stats = user_stats[user_id]
            user_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            user_high_conf_ratio = stats['high_conf'] / stats['total'] if stats['total'] > 0 else 0
            user_high_conf_accuracy = stats['high_conf_correct'] / stats['high_conf'] if stats['high_conf'] > 0 else 0
            
            f.write(f"{user_id:4d} | {stats['total']:5d} | {stats['correct']:7d} | "
                   f"{user_accuracy*100:6.1f} | {stats['high_conf']:8d} | "
                   f"{user_high_conf_ratio*100:10.1f} | {user_high_conf_accuracy*100:7.1f}\n")
    
    # 绘制统计图表
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    user_ids = sorted(user_stats.keys())
    accuracies = [user_stats[uid]['correct'] / user_stats[uid]['total'] for uid in user_ids]
    high_conf_ratios = [user_stats[uid]['high_conf'] / user_stats[uid]['total'] for uid in user_ids]
    high_conf_accuracies = [user_stats[uid]['high_conf_correct'] / user_stats[uid]['high_conf'] 
                           if user_stats[uid]['high_conf'] > 0 else 0 for uid in user_ids]
    
    # 子图1: 每用户准确率
    plt.subplot(2, 2, 1)
    plt.bar(user_ids, accuracies)
    plt.axhline(y=overall_accuracy, color='r', linestyle='--', label=f'Overall: {overall_accuracy:.3f}')
    plt.title('Accuracy per User')
    plt.xlabel('User ID')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 子图2: 每用户高置信度比例
    plt.subplot(2, 2, 2)
    plt.bar(user_ids, high_conf_ratios)
    plt.axhline(y=high_conf_ratio, color='r', linestyle='--', label=f'Overall: {high_conf_ratio:.3f}')
    plt.title('High-Confidence Ratio per User')
    plt.xlabel('User ID')
    plt.ylabel('High-Confidence Ratio')
    plt.legend()
    
    # 子图3: 每用户高置信度样本准确率
    plt.subplot(2, 2, 3)
    plt.bar(user_ids, high_conf_accuracies)
    plt.axhline(y=high_conf_accuracy, color='r', linestyle='--', label=f'Overall: {high_conf_accuracy:.3f}')
    plt.title('High-Confidence Sample Accuracy per User')
    plt.xlabel('User ID')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 子图4: 置信度分布
    plt.subplot(2, 2, 4)
    plt.hist(results['max_probs'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=confidence_threshold, color='r', linestyle='--', 
               label=f'Threshold: {confidence_threshold}')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation report saved to {report_path}")
    print(f"Text report saved to {text_report_path}")
    print(f"Charts saved to {output_dir / 'evaluation_charts.png'}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated samples')
    parser.add_argument('--generated_dir', type=str, required=True, help='Directory with generated samples')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to trained classifier')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载分类器
    print("Loading classifier...")
    classifier = load_classifier(args.classifier_path, device)
    
    # 创建数据变换
    transform = create_transforms()
    
    # 加载生成样本数据集
    print("Loading generated samples...")
    dataset = GeneratedSampleDataset(args.generated_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 评估样本
    print(f"Evaluating samples with confidence threshold {args.confidence_threshold}...")
    results, user_stats = evaluate_samples(classifier, dataloader, device, args.confidence_threshold)
    
    # 保存高置信度样本
    print("Saving high-confidence samples...")
    saved_count, saved_correct = save_high_confidence_samples(results, args.output_dir, args.confidence_threshold)
    
    # 生成评估报告
    print("Generating evaluation report...")
    report = generate_evaluation_report(results, user_stats, args.output_dir, args.confidence_threshold)
    
    # 打印总结
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Total samples evaluated: {len(results['predictions'])}")
    print(f"Overall accuracy: {report['evaluation_summary']['overall_accuracy']:.3f} ({report['evaluation_summary']['overall_accuracy']*100:.1f}%)")
    print(f"High-confidence samples: {report['evaluation_summary']['high_confidence_samples']} ({report['evaluation_summary']['high_confidence_ratio']*100:.1f}%)")
    print(f"High-confidence accuracy: {report['evaluation_summary']['high_confidence_accuracy']:.3f} ({report['evaluation_summary']['high_confidence_accuracy']*100:.1f}%)")
    print(f"High-confidence samples saved: {saved_count}")

if __name__ == "__main__":
    main()
