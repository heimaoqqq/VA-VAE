#!/usr/bin/env python3
"""
Kaggle快速修复脚本
直接修复数据加载和模型加载问题
"""

import os
import sys

def fix_data_loading():
    """修复数据加载问题"""
    print("🔧 修复数据加载问题...")
    
    # 读取原始文件
    dataset_file = "minimal_micro_doppler_dataset.py"
    if not os.path.exists(dataset_file):
        print(f"❌ 文件不存在: {dataset_file}")
        return False
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复__getitem__方法中的维度转换
    old_code = '''        # 转换为tensor
        image_tensor = torch.from_numpy(spectrogram).float()
        
        return {
            'image': image_tensor,      # LightningDiT期望的键名
            'user_id': item['user_id']  # 新增的用户条件
        }'''
    
    new_code = '''        # 转换为tensor
        image_tensor = torch.from_numpy(spectrogram).float()
        
        # 确保维度为 (C, H, W)
        if image_tensor.dim() == 3:
            if image_tensor.shape[2] == 3:  # (H, W, C) -> (C, H, W)
                image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.shape[0] != 3:  # 其他情况
                # 如果不是3通道，转换为3通道
                if image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.repeat(3, 1, 1)
                else:
                    # 取第一个通道并复制
                    image_tensor = image_tensor[0:1].repeat(3, 1, 1)
        elif image_tensor.dim() == 2:  # (H, W) -> (3, H, W)
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # 验证最终维度
        assert image_tensor.shape == (3, 256, 256), f"维度错误: {image_tensor.shape}"
        
        return {
            'image': image_tensor,      # LightningDiT期望的键名
            'user_id': item['user_id']  # 新增的用户条件
        }'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(dataset_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 数据加载修复完成")
        return True
    else:
        print("⚠️ 未找到需要修复的代码段")
        return False

def fix_model_loading():
    """修复模型加载问题"""
    print("🔧 修复模型加载问题...")
    
    # 读取原始文件
    training_file = "minimal_training_modification.py"
    if not os.path.exists(training_file):
        print(f"❌ 文件不存在: {training_file}")
        return False
    
    with open(training_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复导入问题
    old_import = "from tokenizer.vavae import VAVAE"
    new_import = "from tokenizer.autoencoder import AutoencoderKL"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
    
    # 修复模型创建
    old_model_code = '''            # 创建VA-VAE模型实例
            original_vavae = VAVAE()
            
            # 加载权重
            if 'state_dict' in checkpoint:
                original_vavae.load_state_dict(checkpoint['state_dict'])
            else:
                original_vavae.load_state_dict(checkpoint)'''
    
    new_model_code = '''            # 创建VA-VAE模型实例并直接加载权重
            original_vavae = AutoencoderKL(
                embed_dim=32,  # f16d32配置
                ch_mult=(1, 1, 2, 2, 4),
                ckpt_path=args.original_vavae,  # 直接使用ckpt_path参数
                model_type='vavae'
            )'''
    
    if old_model_code in content:
        content = content.replace(old_model_code, new_model_code)
    
    with open(training_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 模型加载修复完成")
    return True

def create_simple_test():
    """创建简单测试脚本"""
    print("🔧 创建简单测试脚本...")
    
    test_code = '''#!/usr/bin/env python3
"""简单的数据和模型测试"""

import torch
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('LightningDiT')

def test_data():
    """测试数据加载"""
    print("测试数据加载...")
    from minimal_micro_doppler_dataset import MicroDopplerDataset
    
    dataset = MicroDopplerDataset("/kaggle/working/data_split/train", split='train')
    sample = dataset[0]
    image = sample['image']
    
    print(f"图像维度: {image.shape}")
    print(f"图像类型: {image.dtype}")
    print(f"图像范围: [{image.min():.3f}, {image.max():.3f}]")
    
    assert image.shape == (3, 256, 256), f"维度错误: {image.shape}"
    print("✅ 数据测试通过")

def test_model():
    """测试模型加载"""
    print("测试模型加载...")
    from tokenizer.autoencoder import AutoencoderKL
    
    model = AutoencoderKL(
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path="/kaggle/working/pretrained/vavae-imagenet256-f16d32-dinov2.pt",
        model_type='vavae'
    )
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model.encode(dummy_input)
    
    print("✅ 模型测试通过")

if __name__ == "__main__":
    test_data()
    test_model()
    print("🎉 所有测试通过！")
'''
    
    with open("simple_test.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("✅ 测试脚本创建完成")

def main():
    """主函数"""
    print("🚀 开始Kaggle快速修复...")
    print("=" * 50)
    
    # 检查当前目录
    print(f"当前目录: {os.getcwd()}")
    files = [f for f in os.listdir('.') if f.endswith('.py')]
    print(f"Python文件: {files}")
    
    # 执行修复
    fix_data_loading()
    fix_model_loading()
    create_simple_test()
    
    print("=" * 50)
    print("🎯 修复完成！")
    print("下一步:")
    print("1. 运行测试: python simple_test.py")
    print("2. 如果测试通过，运行训练: python minimal_training_modification.py ...")

if __name__ == "__main__":
    main()
