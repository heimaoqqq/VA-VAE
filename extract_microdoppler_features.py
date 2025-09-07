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
        
        # 条件生成：从文件路径提取用户ID
        # 路径格式: /path/to/ID_X/IDX_caseX_X_DopplerX.jpg
        import re
        path_parts = Path(image_path).parts
        user_folder = None
        for part in path_parts:
            if part.startswith('ID_'):
                user_folder = part
                break
                
        if user_folder:
            # 提取用户ID: ID_1->0, ID_2->1, ..., ID_31->30
            match = re.match(r'ID_(\d+)', user_folder)
            if match:
                user_id = int(match.group(1))
                label = user_id - 1  # ID_1->0, ID_2->1, etc.
                if label < 0 or label >= 31:
                    print(f"⚠️ 用户ID超出范围: {user_folder} -> {label}, 使用0")
                    label = 0
            else:
                print(f"⚠️ 无法解析用户ID: {user_folder}, 使用0")
                label = 0
        else:
            print(f"⚠️ 未找到用户文件夹: {image_path}, 使用0")
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
    
    # 设置设备 - 支持多GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔧 检测到 {num_gpus} 个GPU")
        device = 0
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
        print("⚠️ 未检测到GPU，使用CPU")
    
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
    
    # 设置路径
    datasets_dir = os.path.join(lightningdit_check_path, 'datasets')
    tokenizer_dir = os.path.join(lightningdit_check_path, 'tokenizer')
    
    # 自动创建缺失的__init__.py文件
    init_files_to_create = [
        os.path.join(lightningdit_check_path, '__init__.py'),
        os.path.join(datasets_dir, '__init__.py'),
        os.path.join(tokenizer_dir, '__init__.py')
    ]
    
    for init_file in init_files_to_create:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated __init__.py for package imports\n")
    
    # 直接文件导入官方模块
    import importlib.util
    
    try:
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
        
        print("✅ 官方模块导入完成")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
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
        vae_config_content = vae_config_content.replace('/path/to/checkpoint.pt', args.vae_checkpoint)
    
    # 创建临时配置文件
    temp_config_path = './temp_vavae_config.yaml'
    with open(temp_config_path, 'w') as f:
        f.write(vae_config_content)
    
    # 初始化VA-VAE
    vae = VA_VAE(temp_config_path)
    
    # 启用多GPU支持
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        vae.model = torch.nn.DataParallel(vae.model)
        print(f"📊 VA-VAE使用 {torch.cuda.device_count()} 个GPU进行并行处理")
    
    # VA_VAE的model已经在load()中设置为.cuda().eval()，无需再次设置
    print("✅ VA-VAE模型加载完成")
    
    # 加载数据
    image_paths = load_image_paths(args.data_path, args.split_file, args.split)
    
    # 创建数据集和加载器（使用官方VA-VAE变换）
    transform = create_transform(vae)
    dataset = MicrodopplerDataset(image_paths, transform=transform)
    
    # 根据GPU数量调整batch_size
    effective_batch_size = args.batch_size
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # 多GPU时可以使用更大的batch_size
        print(f"📊 多GPU环境，batch_size保持为 {effective_batch_size}")
    
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False  # 提高数据加载效率
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
        if run_images % 1000 == 0:
            print(f'📊 处理进度: {run_images}/{total_data_in_loop} ({run_images/total_data_in_loop*100:.1f}%)')
        
        # 编码为latent（使用官方VA-VAE接口）
        with torch.no_grad():
            z = vae.encode_images(x).detach().cpu()  # (N, 32, 16, 16)
        
        # 第一次显示latent形状
        if batch_idx == 0:
            print(f'✅ Latent形状: {z.shape}')
        
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
                'latents_flip': latents.clone(),  # 使用独立内存拷贝，避免共享内存错误
                'labels': labels
            }
            
            save_filename = os.path.join(output_dir, f'latents_rank00_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}'}
            )
            print(f'💾 保存批次 {saved_files}: {latents.shape[0]} 样本')
            
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
            'latents_flip': latents.clone(),  # 使用相同数据，保持格式兼容
            'labels': labels
        }
        
        save_filename = os.path.join(output_dir, f'latents_rank00_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}'}
        )
        print(f'💾 保存最终批次: {latents.shape[0]} 样本')
    
    # 计算latent统计（官方方式）
    print("📊 计算统计信息...")
    dataset = ImgLatentDataset(output_dir, latent_norm=True)
    mean_tensor, std_tensor = dataset.get_latent_stats()  # 正确：返回tuple (mean, std)
    
    mean_range = f"[{mean_tensor.min():.3f}, {mean_tensor.max():.3f}]"
    std_range = f"[{std_tensor.min():.3f}, {std_tensor.max():.3f}]"
    print(f"   均值范围: {mean_range}, 标准差范围: {std_range}")
    print(f'✅ 数据集包含 {len(dataset)} 个样本')
    print('🎉 特征提取完成！可以开始训练了')

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
