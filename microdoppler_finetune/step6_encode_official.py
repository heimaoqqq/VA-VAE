"""
基于官方LightningDiT的extract_features.py修改
用于编码微多普勒数据集
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
import sys
from safetensors.torch import save_file
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# 添加LightningDiT路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT'))
from tokenizer.vavae import VA_VAE
import shutil

# 微多普勒数据集类（替代ImageFolder）
class MicroDopplerDataset(torch.utils.data.Dataset):
    """微多普勒数据集 - 支持训练/验证划分"""
    def __init__(self, data_root, split='all', split_file=None, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        
        # 收集图像
        self.images = []
        self.labels = []
        
        if split == 'all' or split_file is None:
            # 使用所有数据（原始行为）
            for user_id in range(1, 32):
                user_folder = self.data_root / f"ID_{user_id}"
                if user_folder.exists():
                    user_images = sorted(user_folder.glob("*.jpg"))
                    for img_path in user_images:
                        self.images.append(str(img_path))
                        self.labels.append(user_id - 1)  # 0-30
        else:
            # 使用划分文件
            with open(split_file, 'r') as f:
                dataset_split = json.load(f)
            
            # 收集指定划分的所有图像
            for user_id in range(1, 32):
                user_key = f"ID_{user_id}"
                if user_key in dataset_split[split]:
                    user_images = dataset_split[split][user_key]
                    for img_path in user_images:
                        self.images.append(img_path)
                        self.labels.append(user_id - 1)  # 0-30       
                    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path)
        
        # 微多普勒时频图处理 - 保持彩色信息
        if img.mode == 'RGBA':
            img = img.convert('RGB')  # 移除alpha通道
        elif img.mode != 'RGB':
            img = img.convert('RGB')  # 确保是RGB格式
        
        img_tensor = self.transform(img)
        return img_tensor, label

def encode_split(args, split='all'):
    """
    编码指定的数据划分
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP (保持官方结构，即使单GPU也用DDP)
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders (保持官方结构)
    data_split_name = f'{args.data_split}_{split}' if split != 'all' else args.data_split
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{data_split_name}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📁 输出目录: {output_dir}")

    # Create model (完全按官方方式) - 直接使用YAML配置文件
    tokenizer = VA_VAE(args.config)

    # Setup data (修改为微多普勒数据集)
    datasets = [
        MicroDopplerDataset(args.data_path, split=split, split_file=args.split_file, 
                          transform=tokenizer.img_transform(p_hflip=0.0)),
        MicroDopplerDataset(args.data_path, split=split, split_file=args.split_file,
                          transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    # 官方编码循环
    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            
            # 官方编码方式（第92行）
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
            else:
                latents_flip.append(z)

        # 官方保存逻辑（每10000个样本保存一次）
        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # 保存剩余的latents（少于10000个）
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # 完成编码
    if rank == 0:
        print(f"✅ {split}集编码完成")
    
    # 清理分布式
    if world_size > 1:
        dist.barrier()
    
    return output_dir


def main(args):
    """
    主函数：根据参数决定编码模式
    """
    if args.split_file and os.path.exists(args.split_file):
        # 分别编码训练集和验证集
        print("📊 使用数据划分文件，分别编码训练集和验证集")
        print(f"   划分文件: {args.split_file}")
        
        for split in ['train', 'val']:
            print(f"\n{'='*60}")
            print(f"🎯 编码 {split} 集")
            print(f"{'='*60}")
            output_dir = encode_split(args, split=split)
            print(f"✅ {split}集完成: {output_dir}")
    else:
        # 编码所有数据（原始行为）
        print("📊 编码所有数据（未使用数据划分）")
        encode_split(args, split='all')
    
    # 最终清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改为正确的Kaggle数据路径
    parser.add_argument("--data_path", type=str, default='/kaggle/input/dataset')  # 原始数据路径
    parser.add_argument("--data_split", type=str, default='microdoppler')
    parser.add_argument("--output_path", type=str, default="/kaggle/working/latents_official")
    parser.add_argument("--config", type=str, default="vavae_config_for_dit.yaml")  # YAML配置文件（包含ckpt_path）
    parser.add_argument("--split_file", type=str, default="/kaggle/working/data_split/dataset_split.json",
                       help="数据划分文件，如果提供则分别编码训练/验证集")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)  # 降低到4以避免警告
    args = parser.parse_args()
    main(args)
