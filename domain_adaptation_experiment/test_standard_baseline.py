#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 ResNet18 Baseline 评估
评估标准ResNet18（无对比学习）在多个目标域上的表现
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd


class StandardResNet18Classifier(nn.Module):
    """
    标准 ResNet18 分类器（用于加载训练好的模型）
    结构需要与训练时保持一致
    """
    def __init__(self, num_classes=31):
        super(StandardResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.resnet(x)
    
    @property
    def backbone(self):
        """提供backbone接口，用于特征提取（兼容NCC）"""
        # 返回一个去掉最后fc层的模型
        class BackboneWrapper(nn.Module):
            def __init__(self, resnet):
                super().__init__()
                self.resnet = resnet
            
            def forward(self, x):
                # 执行ResNet除了fc层之外的所有操作
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)
                
                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
                
                x = self.resnet.avgpool(x)
                x = torch.flatten(x, 1)
                return x
        
        return BackboneWrapper(self.resnet)


def load_standard_classifier(checkpoint_path, num_classes=31, device='cuda'):
    """加载标准ResNet18分类器"""
    model = StandardResNet18Classifier(num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载模型，验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ 加载模型成功")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_baseline(model, test_loader, device):
    """评估baseline准确率（直接用分类器预测）"""
    model.eval()
    
    correct = 0
    total = 0
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集预测概率的最大值（置信度）
            max_probs = probs.max(dim=1)[0]
            all_probs.extend(max_probs.cpu().numpy())
    
    accuracy = correct / total
    avg_confidence = sum(all_probs) / len(all_probs)
    
    return accuracy, avg_confidence


def main():
    parser = argparse.ArgumentParser(description='标准ResNet18 Baseline评估')
    
    # 模型和数据
    parser.add_argument('--model_path', type=str,
                       default='/kaggle/working/VA-VAE/improved_classifier/best_improved_classifier.pth',
                       help='标准ResNet18模型路径')
    parser.add_argument('--num_classes', type=int, default=31)
    
    # 其他
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*80)
    print("📊 标准 ResNet18 Baseline 评估")
    print("="*80)
    
    # 加载模型
    print(f"\n📦 加载标准ResNet18: {args.model_path}")
    model = load_standard_classifier(args.model_path, args.num_classes, device)
    
    # 定义所有测试数据集
    datasets = {
        'Backpack_free': '/kaggle/input/organized-gait-dataset/Backpack_free',
        'Backpack_line': '/kaggle/input/organized-gait-dataset/Backpack_line',
        'Bag_free': '/kaggle/input/organized-gait-dataset/Bag_free',
        'Bag_line': '/kaggle/input/organized-gait-dataset/Bag_line',
        'Bag_Phone_free': '/kaggle/input/organized-gait-dataset/Bag_Phone_free',
        'Bag_Phone_line': '/kaggle/input/organized-gait-dataset/Bag_Phone_line',
        'Normal_free': '/kaggle/input/organized-gait-dataset/Normal_free'
    }
    
    all_results = []
    
    # 循环评估每个数据集
    for dataset_name, data_dir in datasets.items():
        print("\n" + "="*80)
        print(f"🔍 评估数据集: {dataset_name}")
        print("="*80)
        print(f"📂 路径: {data_dir}")
        
        # 创建数据集（加载所有数据）
        class SimpleGaitDataset(Dataset):
            def __init__(self, data_dir, transform):
                self.data_dir = Path(data_dir)
                self.transform = transform
                self.samples = []
                
                user_dirs = sorted([d for d in self.data_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith('ID_')])
                
                for user_dir in user_dirs:
                    user_id = int(user_dir.name.split('_')[1]) - 1
                    image_files = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
                    
                    for img_path in image_files:
                        self.samples.append({
                            'path': img_path,
                            'label': user_id
                        })
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                image = Image.open(sample['path']).convert('RGB')
                label = sample['label']
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        # 检查数据集是否存在
        if not Path(data_dir).exists():
            print(f"⚠️  数据集不存在，跳过")
            continue
        
        test_dataset = SimpleGaitDataset(data_dir, transform)
        print(f"✅ 测试集: {len(test_dataset)} 样本")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # 评估 Baseline
        baseline_acc, baseline_conf = evaluate_baseline(model, test_loader, device)
        
        print(f"\n📊 结果:")
        print(f"   准确率: {baseline_acc*100:.2f}%")
        print(f"   置信度: {baseline_conf*100:.2f}%")
        
        all_results.append({
            'dataset': dataset_name,
            'accuracy': baseline_acc * 100,
            'confidence': baseline_conf * 100,
            'num_samples': len(test_dataset)
        })
    
    # ========== 汇总所有结果 ==========
    print("\n" + "="*80)
    print("📈 所有数据集结果汇总")
    print("="*80)
    
    print(f"\n{'数据集':<20} {'样本数':<10} {'准确率':<12} {'置信度':<10}")
    print("-"*80)
    
    for res in all_results:
        print(f"{res['dataset']:<20} "
              f"{res['num_samples']:<10} "
              f"{res['accuracy']:>6.2f}%     "
              f"{res['confidence']:>6.2f}%")
    
    # 计算平均
    avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
    avg_conf = sum(r['confidence'] for r in all_results) / len(all_results)
    
    print("-"*80)
    print(f"{'平均':<20} {'':<10} {avg_acc:>6.2f}%     {avg_conf:>6.2f}%")
    
    # 保存结果
    output_dir = Path(args.model_path).parent
    results_path = output_dir / 'baseline_results.csv'
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存: {results_path}")


if __name__ == '__main__':
    main() 
