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
from datetime import datetime
from PIL import Image

# 添加LightningDiT路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LightningDiT'))
from tokenizer.vavae import VA_VAE

# 微多普勒数据集类（替代ImageFolder）
class MicroDopplerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 收集所有图像和标签 - 数据格式是 ID_1, ID_2, ..., ID_31
        for user_id in range(1, 32):  # 31个用户
            user_folder = f"ID_{user_id}"
            user_path = os.path.join(root_dir, user_folder)
            
            if not os.path.isdir(user_path):
                print(f"警告: 缺少用户文件夹 {user_path}")
                continue
            
            # 标签从0开始，所以ID_1对应label=0
            label = user_id - 1
            
            for img_file in os.listdir(user_path):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(user_path, img_file)
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        
        # 微多普勒时频图处理 - 保持彩色信息
        if img.mode == 'RGBA':
            img = img.convert('RGB')  # 移除alpha通道
        elif img.mode != 'RGB':
            img = img.convert('RGB')  # 确保是RGB格式
        
        img_tensor = self.transform(img)
        return img_tensor, label

def main(args):
    """
    官方extract_features.py主函数，修改为微多普勒数据
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
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model (完全按官方方式)
    tokenizer = VA_VAE(
        args.config
    )

    # Setup data (修改为微多普勒数据集)
    datasets = [
        MicroDopplerDataset(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        MicroDopplerDataset(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
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
        print("✅ 编码完成")
    
    # 清理分布式
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改为正确的Kaggle数据路径
    parser.add_argument("--data_path", type=str, default='/kaggle/input/dataset')  # 原始数据路径
    parser.add_argument("--data_split", type=str, default='microdoppler_train')
    parser.add_argument("--output_path", type=str, default="/kaggle/working/latents_official")
    parser.add_argument("--config", type=str, default="/kaggle/input/vavae-stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
