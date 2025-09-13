"""
验证合成数据标签映射正确性
检查生成的User_XX目录中的图像是否被分类器正确识别为对应的用户ID
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class SyntheticDataset(Dataset):
    """合成数据集加载器"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # 加载User_XX目录中的图像
        user_dirs = list(self.data_dir.glob("User_*"))
        
        if not user_dirs:
            raise ValueError(f"在 {self.data_dir} 中未找到User_*目录")
        
        for user_dir in sorted(user_dirs):
            if user_dir.is_dir():
                # 解析用户ID：User_00 → 0, User_01 → 1
                user_id = int(user_dir.name.split('_')[1])
                
                # 加载该用户目录下的所有图像
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in user_dir.glob(ext):
                        self.samples.append((str(img_path), user_id))
        
        print(f"加载了 {len(self.samples)} 个合成样本，来自 {len(set(s[1] for s in self.samples))} 个用户")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, expected_label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, expected_label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), expected_label, img_path


def load_classifier(checkpoint_path, device):
    """加载分类器模型"""
    import torchvision.models as models
    
    class MicroDopplerModel(nn.Module):
        def __init__(self, num_classes=31, dropout_rate=0.3):
            super().__init__()
            
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
            
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 64)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # 创建并加载模型
    model = MicroDopplerModel(num_classes=31)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 分类器加载完成: {checkpoint_path}")
    return model


def verify_synthetic_data(synthetic_dir, classifier_path, device='cuda', batch_size=32):
    """验证合成数据的标签映射正确性"""
    
    print(f"🔍 验证合成数据映射正确性")
    print(f"📂 合成数据目录: {synthetic_dir}")
    print(f"🤖 分类器路径: {classifier_path}")
    
    # 数据变换（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集和分类器
    dataset = SyntheticDataset(synthetic_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    classifier = load_classifier(classifier_path, device)
    
    # 统计结果
    user_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
    total_correct = 0
    total_samples = 0
    
    print(f"\n📊 开始验证...")
    
    with torch.no_grad():
        for images, expected_labels, img_paths in tqdm(dataloader, desc="验证中"):
            images = images.to(device)
            expected_labels = expected_labels.to(device)
            
            # 分类器预测
            outputs = classifier(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            # 统计每个样本
            for i in range(len(expected_labels)):
                expected = expected_labels[i].item()
                predicted = predictions[i].item()
                confidence = confidences[i].item()
                
                user_stats[expected]['total'] += 1
                user_stats[expected]['predictions'].append((predicted, confidence))
                
                if expected == predicted:
                    user_stats[expected]['correct'] += 1
                    total_correct += 1
                
                total_samples += 1
    
    # 生成详细报告
    print(f"\n" + "="*80)
    print(f"📈 合成数据标签映射验证报告")
    print(f"="*80)
    
    print(f"\n🎯 总体统计:")
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"   • 总样本数量: {total_samples}")
    print(f"   • 正确标签数量: {total_correct}")
    print(f"   • 整体准确率: {overall_accuracy:.1f}%")
    
    print(f"\n👤 各用户详细统计:")
    problem_users = []
    
    for user_id in sorted(user_stats.keys()):
        stats = user_stats[user_id]
        accuracy = stats['correct'] / stats['total'] * 100
        
        # 分析预测分布
        pred_counts = defaultdict(int)
        avg_confidence = 0
        for pred, conf in stats['predictions']:
            pred_counts[pred] += 1
            avg_confidence += conf
        avg_confidence /= len(stats['predictions'])
        
        # 找出最常被错误预测为的类别
        most_common_wrong = None
        if pred_counts[user_id] < stats['total']:
            wrong_preds = {k: v for k, v in pred_counts.items() if k != user_id}
            if wrong_preds:
                most_common_wrong = max(wrong_preds, key=wrong_preds.get)
        
        status = "✅" if accuracy > 90 else "⚠️" if accuracy > 70 else "❌"
        print(f"   {status} User_{user_id:02d}: {accuracy:5.1f}% ({stats['correct']}/{stats['total']}) "
              f"置信度: {avg_confidence:.3f}")
        
        if most_common_wrong is not None:
            print(f"      └── 最常错误预测为: User_{most_common_wrong:02d} "
                  f"({pred_counts[most_common_wrong]}次)")
        
        if accuracy < 90:
            problem_users.append(user_id)
    
    # 问题分析
    if problem_users:
        print(f"\n🚨 发现问题用户: {problem_users}")
        print(f"   这些用户的合成数据可能存在标签映射错误")
    else:
        print(f"\n✅ 所有用户的合成数据标签映射都正确！")
    
    # 映射正确性判断
    print(f"\n🔍 映射正确性分析:")
    if overall_accuracy > 95:
        print(f"   ✅ 映射完全正确 - 可以直接使用现有合成数据")
    elif overall_accuracy > 85:
        print(f"   ⚠️ 映射基本正确 - 少数用户可能需要重新生成")
    else:
        print(f"   ❌ 映射存在严重错误 - 建议重新生成所有合成数据")
    
    return overall_accuracy, problem_users


def main():
    parser = argparse.ArgumentParser(description='验证合成数据标签映射')
    parser.add_argument('--synthetic_dir', type=str, required=True,
                       help='合成数据目录路径')
    parser.add_argument('--classifier_path', type=str, required=True,
                       help='分类器checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 执行验证
    accuracy, problem_users = verify_synthetic_data(
        args.synthetic_dir,
        args.classifier_path,
        device,
        args.batch_size
    )
    
    print(f"\n🎯 验证完成！整体准确率: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
