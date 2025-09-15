"""
基于真实生成样本分析user_specificity分布
确定合理的阈值设置
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def load_classifier(model_path, device):
    """加载分类器"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    # 根据checkpoint判断模型类型
    if 'feature_projector.0.weight' in checkpoint['model_state_dict']:
        from train_calibrated_classifier import DomainAdaptiveClassifier
        model = DomainAdaptiveClassifier(
            num_classes=checkpoint['num_classes'],
            dropout_rate=0.3,
            feature_dim=512
        )
    else:
        from improved_classifier_training import ImprovedClassifier
        model = ImprovedClassifier(
            num_classes=checkpoint['num_classes'],
            backbone='resnet18',
            dropout_rate=0.5,
            freeze_layers='minimal'
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def compute_user_specificity_from_samples(samples_dir, classifier, device):
    """从真实样本计算user_specificity分布"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_specificities_diff = []
    all_specificities_ratio = []
    all_user_probs = []
    all_max_other_probs = []
    all_correct = []
    
    samples_path = Path(samples_dir)
    
    # 调试：检查目录是否存在
    if not samples_path.exists():
        print(f"❌ 样本目录不存在: {samples_path}")
        return {'diff_mode': np.array([]), 'ratio_mode': np.array([]), 'user_probs': np.array([]), 'max_other_probs': np.array([]), 'correct': np.array([])}
    
    # 调试：列出目录内容
    all_items = list(samples_path.iterdir())
    print(f"📂 目录内容: {[item.name for item in all_items[:10]]}...")  # 只显示前10个
    
    # 遍历所有用户文件夹
    user_folders = [f for f in samples_path.iterdir() if f.is_dir() and f.name.startswith('user_')]
    
    print(f"🔍 分析 {len(user_folders)} 个用户的样本...")
    if len(user_folders) == 0:
        print("❌ 未找到user_XX格式的文件夹！")
        print("   请确认文件夹命名格式为: user_00, user_01, user_02...")
    
    total_images_processed = 0
    total_images_found = 0
    
    for user_folder in tqdm(user_folders, desc="处理用户"):
        try:
            user_id = int(user_folder.name.split('_')[1])
        except:
            print(f"⚠️  无法解析用户ID: {user_folder.name}")
            continue
            
        # 获取该用户的所有图像
        image_files = list(user_folder.glob('*.png')) + list(user_folder.glob('*.jpg'))
        total_images_found += len(image_files)
        
        if len(image_files) == 0:
            print(f"⚠️  用户{user_id}没有找到图像文件")
            continue
            
        print(f"📷 用户{user_id}: 找到{len(image_files)}张图像")
            
        # 随机选择最多50张图像进行分析（避免内存问题）
        if len(image_files) > 50:
            image_files = np.random.choice(image_files, 50, replace=False).tolist()
            print(f"   → 随机选择{len(image_files)}张进行分析")
        
        user_processed = 0
        user_errors = 0
        
        for img_path in image_files:
            try:
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 获取分类器输出
                    outputs = classifier(img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    
                    # 计算指标
                    confidence, pred = torch.max(probs, dim=1)
                    
                    user_prob = probs[0, user_id].item()
                    other_probs = torch.cat([probs[0, :user_id], probs[0, user_id+1:]])
                    max_other_prob = torch.max(other_probs).item()
                    
                    # 两种计算方法
                    specificity_diff = user_prob - max_other_prob
                    specificity_ratio = user_prob / (user_prob + max_other_prob) if (user_prob + max_other_prob) > 0 else 0.0
                    
                    all_specificities_diff.append(specificity_diff)
                    all_specificities_ratio.append(specificity_ratio)
                    all_user_probs.append(user_prob)
                    all_max_other_probs.append(max_other_prob)
                    all_correct.append(pred.item() == user_id)
                    
                    user_processed += 1
                    total_images_processed += 1
                    
            except Exception as e:
                user_errors += 1
                if user_errors <= 3:  # 只显示前3个错误
                    print(f"   ❌ 处理图像出错 {img_path.name}: {str(e)[:50]}...")
        
        if user_processed > 0:
            print(f"   ✅ 成功处理{user_processed}张图像")
        if user_errors > 0:
            print(f"   ⚠️  {user_errors}张图像处理失败")
    
    print(f"\n📊 总体统计:")
    print(f"   找到图像总数: {total_images_found}")
    print(f"   成功处理: {total_images_processed}")
    print(f"   最终数据点: {len(all_specificities_ratio)}")
    
    result = {
        'diff_mode': np.array(all_specificities_diff),
        'ratio_mode': np.array(all_specificities_ratio),
        'user_probs': np.array(all_user_probs),
        'max_other_probs': np.array(all_max_other_probs),
        'correct': np.array(all_correct)
    }
    
    return result

def analyze_threshold_performance(data):
    """分析不同阈值的性能"""
    
    # 检查是否有数据
    if len(data['correct']) == 0:
        print("❌ 没有找到任何样本数据！")
        print("   请检查样本目录路径和文件夹命名格式")
        return np.array([])
    
    # 只分析正确预测的样本
    correct_mask = data['correct'].astype(bool)
    ratio_correct = data['ratio_mode'][correct_mask]
    diff_correct = data['diff_mode'][correct_mask]
    
    if len(ratio_correct) == 0:
        print("❌ 没有找到正确预测的样本！")
        return np.array([])
    
    print(f"📊 基于 {len(ratio_correct)} 个正确预测样本的分析:")
    
    # 基础统计
    print(f"\n🔢 比例模式统计:")
    print(f"   均值: {np.mean(ratio_correct):.3f}")
    print(f"   标准差: {np.std(ratio_correct):.3f}")
    print(f"   分位数:")
    for p in [25, 50, 75, 90, 95]:
        value = np.percentile(ratio_correct, p)
        print(f"     {p}%: {value:.3f}")
    
    print(f"\n🔢 差值模式统计 (参考):")
    print(f"   均值: {np.mean(diff_correct):.3f}")
    print(f"   标准差: {np.std(diff_correct):.3f}")
    
    # 通过率分析
    print(f"\n📈 比例模式不同阈值的通过率:")
    ratio_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    for thresh in ratio_thresholds:
        pass_rate = np.mean(ratio_correct >= thresh) * 100
        print(f"   阈值 {thresh:.2f}: {pass_rate:.1f}%")
    
    # 推荐阈值（目标15-25%通过率）
    target_rates = [15, 20, 25, 30]
    print(f"\n🎯 推荐阈值（目标通过率）:")
    
    for target_rate in target_rates:
        # 找到最接近目标通过率的阈值
        best_thresh = None
        best_diff = float('inf')
        
        for thresh in np.arange(0.50, 0.90, 0.01):
            actual_rate = np.mean(ratio_correct >= thresh) * 100
            diff = abs(actual_rate - target_rate)
            if diff < best_diff:
                best_diff = diff
                best_thresh = thresh
        
        actual_rate = np.mean(ratio_correct >= best_thresh) * 100
        print(f"   目标{target_rate}%: 阈值{best_thresh:.3f} (实际{actual_rate:.1f}%)")
    
    # 对比当前0.65阈值
    current_rate = np.mean(ratio_correct >= 0.65) * 100
    print(f"\n📊 当前0.65阈值表现:")
    print(f"   通过率: {current_rate:.1f}%")
    
    if current_rate > 30:
        print(f"   💡 建议提高阈值到0.70-0.75以获得更严格筛选")
    elif current_rate < 15:
        print(f"   💡 建议降低阈值到0.60-0.65以获得合理通过率")
    else:
        print(f"   ✅ 当前阈值合适")
    
    return ratio_correct

def main():
    parser = argparse.ArgumentParser(description='分析真实样本的user_specificity分布')
    parser.add_argument('--samples_dir', type=str, 
                       default='/kaggle/working/VA-VAE/generated_samples2',
                       help='生成样本目录')
    parser.add_argument('--classifier_path', type=str,
                       default='/kaggle/working/VA-VAE/domain_adaptive_classifier/best_calibrated_model.pth',
                       help='分类器路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("🔍 基于真实样本的User Specificity阈值分析")
    print(f"📂 样本目录: {args.samples_dir}")
    print(f"🤖 分类器: {args.classifier_path}")
    
    # 加载分类器
    print("\n📥 加载分类器...")
    classifier = load_classifier(args.classifier_path, device)
    
    # 分析样本
    print("\n🧮 计算user_specificity分布...")
    data = compute_user_specificity_from_samples(args.samples_dir, classifier, device)
    
    # 分析结果
    print("\n" + "="*60)
    ratio_data = analyze_threshold_performance(data)
    
    # 可视化（可选）
    try:
        plt.figure(figsize=(12, 8))
        plt.hist(ratio_data, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0.65, color='red', linestyle='--', label='当前阈值: 0.65')
        
        # 添加推荐阈值线
        recommended_20 = np.percentile(ratio_data, 80)  # 20%通过率
        plt.axvline(recommended_20, color='green', linestyle='--', 
                   label=f'推荐阈值(20%): {recommended_20:.3f}')
        
        plt.xlabel('User Specificity (比例模式)')
        plt.ylabel('频次')
        plt.title('真实样本User Specificity分布')
        plt.legend()
        plt.savefig('real_user_specificity_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📊 分布图已保存: real_user_specificity_analysis.png")
        plt.show()
    except:
        print("\n⚠️  无法生成可视化图表")

if __name__ == "__main__":
    main()
