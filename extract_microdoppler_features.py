"""
基于官方extract_features.py，适配微多普勒数据集
完全按照官方格式生成safetensors文件
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
from datetime import datetime
from PIL import Image
import sys

# 添加LightningDiT路径
lightningdit_path = '/kaggle/working/VA-VAE/LightningDiT'
if not os.path.exists(lightningdit_path):
    lightningdit_path = './LightningDiT'  # 备用路径
sys.path.append(lightningdit_path)

class MicrodopplerDataset(torch.utils.data.Dataset):
    """微多普勒数据集，模仿官方ImageFolder结构"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 无条件生成，所有标签为0
        label = 0
        
        return image, label

def load_image_paths(dataset_root, split_file, split_name):
    """加载指定split的图像路径"""
    print(f"📂 加载{split_name}数据集路径...")
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    if split_name not in splits:
        raise ValueError(f"Split '{split_name}' not found in {split_file}")
    
    image_paths = []
    user_data = splits[split_name]
    
    for user_id, paths in user_data.items():
        for rel_path in paths:
            full_path = os.path.join(dataset_root, rel_path)
            if os.path.exists(full_path):
                image_paths.append(full_path)
            else:
                print(f"⚠️ 文件不存在: {full_path}")
    
    print(f"✅ 加载了{len(image_paths)}张图像")
    return image_paths

def create_transform(vae):
    """使用官方VA-VAE的图像变换"""
    # 使用官方VA-VAE的预处理管道
    return vae.img_transform(p_hflip=0.0)  # 无水平翻转

def main(args):
    """
    提取微多普勒latent特征并保存为safetensors格式
    完全遵循官方extract_features.py的逻辑
    """
    print("🚀 开始提取微多普勒latent特征...")
    
    # 设置设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device) if torch.cuda.is_available() else None
    
    # 设置输出目录
    output_dir = os.path.join(args.output_path, args.split)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 创建VA-VAE模型（使用官方VA-VAE接口）
    print("🔧 加载VA-VAE模型...")
    
    # 检查LightningDiT目录是否存在
    lightningdit_check_path = '/kaggle/working/VA-VAE/LightningDiT'
    if not os.path.exists(lightningdit_check_path):
        lightningdit_check_path = './LightningDiT'
    
    if not os.path.exists(lightningdit_check_path):
        print(f"❌ LightningDiT目录不存在: {lightningdit_check_path}")
        print("   请先运行: git clone https://github.com/Alpha-VLLM/LightningDiT.git")
        return
    
    # 调试：显示LightningDiT目录内容
    print(f"📂 LightningDiT路径: {lightningdit_check_path}")
    datasets_dir = os.path.join(lightningdit_check_path, 'datasets')
    tokenizer_dir = os.path.join(lightningdit_check_path, 'tokenizer')
    
    if os.path.exists(datasets_dir):
        print("✅ datasets目录存在")
        datasets_files = os.listdir(datasets_dir)
        print(f"   datasets文件: {datasets_files}")
        if 'img_latent_dataset.py' in datasets_files:
            print("✅ img_latent_dataset.py存在")
        else:
            print("❌ img_latent_dataset.py不存在")
        if '__init__.py' in datasets_files:
            print("✅ datasets/__init__.py存在")
        else:
            print("❌ datasets/__init__.py不存在 - 这是导入失败的原因!")
    else:
        print("❌ datasets目录不存在")
    
    if os.path.exists(tokenizer_dir):
        print("✅ tokenizer目录存在")
        tokenizer_files = os.listdir(tokenizer_dir)
        print(f"   tokenizer文件: {tokenizer_files}")
        if 'vavae.py' in tokenizer_files:
            print("✅ vavae.py存在")
        else:
            print("❌ vavae.py不存在")
        if '__init__.py' in tokenizer_files:
            print("✅ tokenizer/__init__.py存在")
        else:
            print("❌ tokenizer/__init__.py不存在 - 这是导入失败的原因!")
    else:
        print("❌ tokenizer目录不存在")
    
    # 检查根目录的__init__.py
    root_init = os.path.join(lightningdit_check_path, '__init__.py')
    if os.path.exists(root_init):
        print("✅ LightningDiT/__init__.py存在")
    else:
        print("❌ LightningDiT/__init__.py不存在")
    
    # 自动创建缺失的__init__.py文件
    init_files_to_create = [
        os.path.join(lightningdit_check_path, '__init__.py'),
        os.path.join(datasets_dir, '__init__.py'),
        os.path.join(tokenizer_dir, '__init__.py')
    ]
    
    for init_file in init_files_to_create:
        if not os.path.exists(init_file):
            print(f"🔧 创建缺失的__init__.py: {init_file}")
            with open(init_file, 'w') as f:
                f.write("# Auto-generated __init__.py for package imports\n")
    
    # 导入官方模块 - 使用直接导入方式
    try:
        # 方法1: 尝试标准包导入
        from tokenizer.vavae import VA_VAE
        from datasets.img_latent_dataset import ImgLatentDataset
        print("✅ 成功导入官方模块 (标准包导入)")
    except ImportError as e1:
        print(f"❌ 标准导入失败: {e1}")
        try:
            # 方法2: 使用importlib直接导入
            import importlib.util
            
            # 导入vavae模块
            vavae_path = os.path.join(tokenizer_dir, 'vavae.py')
            spec_vavae = importlib.util.spec_from_file_location("vavae", vavae_path)
            vavae_module = importlib.util.module_from_spec(spec_vavae)
            spec_vavae.loader.exec_module(vavae_module)
            VA_VAE = vavae_module.VA_VAE
            
            # 导入img_latent_dataset模块
            dataset_path = os.path.join(datasets_dir, 'img_latent_dataset.py')
            spec_dataset = importlib.util.spec_from_file_location("img_latent_dataset", dataset_path)
            dataset_module = importlib.util.module_from_spec(spec_dataset)
            spec_dataset.loader.exec_module(dataset_module)
            ImgLatentDataset = dataset_module.ImgLatentDataset
            
            print("✅ 成功导入官方模块 (直接文件导入)")
        except Exception as e2:
            print(f"❌ 直接导入也失败: {e2}")
            print(f"   当前Python路径: {sys.path[-3:]}")
            print("   请确保已正确克隆LightningDiT仓库到正确位置")
            return
    
    # 创建与官方一致的VA-VAE配置
    vae_config = {
        'model_name': 'vavae_f16d32',
        'downsample_ratio': 16,
        'checkpoint_path': args.vae_checkpoint
    }
    
    # 使用官方VA-VAE类 - 需要先修改配置文件中的checkpoint路径
    vae_config_path = os.path.join(lightningdit_check_path, 'tokenizer/configs/vavae_f16d32.yaml')
    
    # 读取并修改配置
    with open(vae_config_path, 'r') as f:
        vae_config_content = f.read()
    
    # 替换checkpoint路径
    if args.vae_checkpoint:
        print(f"   设置检查点路径: {args.vae_checkpoint}")
        vae_config_content = vae_config_content.replace('/path/to/checkpoint.pt', args.vae_checkpoint)
    
    # 创建临时配置文件
    temp_config_path = './temp_vavae_config.yaml'
    with open(temp_config_path, 'w') as f:
        f.write(vae_config_content)
    
    # 初始化VA-VAE
    vae = VA_VAE(temp_config_path)
    
    # VA_VAE的model已经在load()中设置为.cuda().eval()，无需再次设置
    print("✅ VA-VAE模型加载完成")
    
    # 加载数据
    image_paths = load_image_paths(args.data_path, args.split_file, args.split)
    
    # 创建数据集和加载器（使用官方VA-VAE变换）
    transform = create_transform(vae)
    dataset = MicrodopplerDataset(image_paths, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    total_data_in_loop = len(loader.dataset)
    print(f"📊 总计图像数: {total_data_in_loop}")
    
    # 提取特征（无数据增强）
    run_images = 0
    saved_files = 0
    latents = []
    labels = []
    
    print("🔄 开始提取latent特征...")
    
    for batch_idx, data in enumerate(loader):
        x = data[0].to(device)  # (N, C, H, W)
        y = data[1]             # (N,) - 标签
        
        run_images += x.shape[0]
        if run_images % 100 == 0:
            print(f'{datetime.now()} 处理 {run_images}/{total_data_in_loop} 图像')
        
        # 编码为latent（使用官方VA-VAE接口）
        with torch.no_grad():
            z = vae.encode_images(x).detach().cpu()  # (N, 32, 16, 16)
        
        if batch_idx == 0:
            print(f'Latent shape: {z.shape}, dtype: {z.dtype}')
        
        latents.append(z)
        labels.append(y)
        
        # 每10000张图像保存一次（官方设置）
        if len(latents) >= 10000 // args.batch_size:
            # 拼接tensor
            latents = torch.cat(latents, dim=0)
            labels = torch.cat(labels, dim=0)
            
            # 保存为safetensors（无数据增强格式）
            save_dict = {
                'latents': latents,
                'latents_flip': latents,  # 使用相同数据，保持格式兼容
                'labels': labels
            }
            
            print(f"保存批次 {saved_files}:")
            for key in save_dict:
                print(f"  {key}: {save_dict[key].shape}")
            
            save_filename = os.path.join(output_dir, f'latents_rank00_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}'}
            )
            print(f'✅ 保存: {save_filename}')
            
            # 重置
            latents = []
            labels = []
            saved_files += 1
    
    # 保存剩余的latents
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        labels = torch.cat(labels, dim=0)
        
        save_dict = {
            'latents': latents,
            'latents_flip': latents,  # 使用相同数据，保持格式兼容
            'labels': labels
        }
        
        print(f"保存最终批次 {saved_files}:")
        for key in save_dict:
            print(f"  {key}: {save_dict[key].shape}")
        
        save_filename = os.path.join(output_dir, f'latents_rank00_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}'}
        )
        print(f'✅ 保存: {save_filename}')
    
    # 计算latent统计（官方方式）
    print("📊 计算latent统计...")
    dataset = ImgLatentDataset(output_dir, latent_norm=True)
    print(f"✅ 统计计算完成，数据集包含 {len(dataset)} 个样本")
    
    print("🎉 特征提取完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微多普勒特征提取")
    parser.add_argument("--data_path", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split_file", type=str, required=True, help="数据集划分文件")
    parser.add_argument("--split", type=str, choices=['train', 'val'], required=True, help="要处理的split")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VA-VAE检查点路径")
    parser.add_argument("--output_path", type=str, default="./latents_safetensors", help="输出路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    args = parser.parse_args()
    main(args)
