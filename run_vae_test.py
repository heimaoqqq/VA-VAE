#!/usr/bin/env python3
"""
运行VA-VAE重建测试 - 适配Kaggle数据结构
专门为 /kaggle/input/dataset/ 下的31个用户目录设计
"""

import os
import sys
from pathlib import Path
from vae_reconstruction_test import VAEReconstructionTester

def main():
    """主函数 - 运行VA-VAE重建测试"""
    print("🚀 开始VA-VAE重建测试")
    print("="*60)
    
    # 数据路径配置
    data_dir = "/kaggle/input/dataset"
    output_dir = "vae_test_results"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    
    # 检查数据目录
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("💡 请确保数据已上传到Kaggle")
        return False
    
    # 检查模型文件
    if not Path(vae_model_path).exists():
        print(f"❌ VA-VAE模型文件不存在: {vae_model_path}")
        print("💡 请先运行 step2_download_models.py 下载模型")
        return False
    
    # 检查用户目录结构
    user_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('ID_')]
    print(f"🔍 发现 {len(user_dirs)} 个用户目录:")
    for user_dir in sorted(user_dirs):
        image_count = len(list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg')) + list(user_dir.glob('*.jpeg')))
        print(f"   {user_dir.name}: {image_count} 张图像")
    
    if len(user_dirs) == 0:
        print("❌ 未找到用户目录（ID_1, ID_2, ...）")
        return False
    
    # 创建VA-VAE测试器
    print(f"\n🔧 初始化VA-VAE测试器...")
    device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    print(f"🔥 使用设备: {device}")
    
    tester = VAEReconstructionTester(vae_model_path, device)
    if tester.vae is None:
        print("❌ VA-VAE模型加载失败")
        return False
    
    # 运行重建测试
    print(f"\n🚀 开始重建测试...")
    results = tester.test_batch_reconstruction(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=4,           # 适中的批次大小
        max_images=100          # 限制总测试图像数量
    )
    
    if results is None:
        print("❌ 重建测试失败")
        return False
    
    # 分析结果
    print(f"\n📊 测试完成！结果分析:")
    mse_values = [r['mse_loss'] for r in results]
    
    print(f"   总图像数: {len(results)}")
    print(f"   平均MSE: {sum(mse_values)/len(mse_values):.6f}")
    print(f"   MSE范围: {min(mse_values):.6f} - {max(mse_values):.6f}")
    
    # 按用户分析
    user_results = {}
    for result in results:
        user_id = result.get('user_id', 'Unknown')
        if user_id not in user_results:
            user_results[user_id] = []
        user_results[user_id].append(result['mse_loss'])
    
    print(f"\n👥 各用户重建质量:")
    for user_id in sorted(user_results.keys()):
        user_mse = user_results[user_id]
        avg_mse = sum(user_mse) / len(user_mse)
        print(f"   {user_id}: {len(user_mse)}张, 平均MSE={avg_mse:.6f}")
    
    # 给出建议
    overall_mse = sum(mse_values) / len(mse_values)
    print(f"\n💡 建议:")
    if overall_mse < 0.01:
        print("   ✅ 重建质量很好！可以直接使用预训练VA-VAE")
        print("   📋 下一步: 进入阶段2，设计UNet扩散模型")
    elif overall_mse < 0.05:
        print("   ⚠️ 重建质量一般，建议考虑微调VA-VAE")
        print("   📋 下一步: 设计VA-VAE微调策略")
    else:
        print("   ❌ 重建质量较差，可能需要重新训练或更换方案")
        print("   📋 下一步: 考虑重新训练VA-VAE或使用其他编码器")
    
    print(f"\n📁 详细结果已保存到: {output_dir}/")
    print("   - reconstruction_stats.txt: 详细统计")
    print("   - batch_*.png: 对比图像")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 VA-VAE重建测试完成！")
    else:
        print("\n❌ VA-VAE重建测试失败！")
    
    sys.exit(0 if success else 1)
