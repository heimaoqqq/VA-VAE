"""
将预编码的latent数据转换为LightningDiT官方的safetensors格式
保持与官方数据格式完全一致
"""

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
from tqdm import tqdm
import argparse

def convert_to_safetensors(input_path, output_dir, split='train'):
    """
    转换latent数据到safetensors格式
    
    官方格式:
    - latents: [N, C, H, W] 的latent向量
    - latents_flip: 水平翻转的latent（数据增强）
    - labels: 类别标签（无条件生成时为0）
    """
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载预编码的latents
    print(f"📂 加载 {split} latents...")
    latent_file = Path(input_path) / f"{split}_latents.pt"
    
    if not latent_file.exists():
        print(f"❌ 文件不存在: {latent_file}")
        return
    
    data = torch.load(latent_file, map_location='cpu', weights_only=False)
    
    # 调试：检查数据格式
    print(f"📊 数据类型: {type(data)}")
    if isinstance(data, dict):
        print(f"   字典键: {list(data.keys())}")
        latents = data['latents']
        user_ids = data.get('user_ids', None)
    elif isinstance(data, (list, tuple)):
        print(f"   列表长度: {len(data)}")
        # 检查列表中第一个元素的类型
        if len(data) > 0:
            first_item = data[0]
            print(f"   列表元素类型: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"   字典键: {list(first_item.keys())}")
                # 提取每个字典中的latent tensor，尝试多种可能的键名
                latents = []
                user_ids = []
                for item in data:
                    # 尝试多种可能的键名
                    if 'latent' in item:
                        latents.append(item['latent'])
                    elif 'tensor' in item:
                        latents.append(item['tensor'])
                    elif 'latents' in item:
                        latents.append(item['latents'])
                    else:
                        # 如果找不到预期的键，打印所有键并使用第一个张量
                        tensor_keys = [k for k, v in item.items() if isinstance(v, torch.Tensor)]
                        if tensor_keys:
                            print(f"   使用键 '{tensor_keys[0]}' 作为latent张量")
                            latents.append(item[tensor_keys[0]])
                        else:
                            print(f"   警告：在字典中找不到张量，跳过此项")
                            continue
                    
                    # 提取用户ID（如果有）
                    user_ids.append(item.get('user_id', 0))
                
                user_ids = user_ids if len(user_ids) > 0 else None
            else:
                latents = data  # 直接使用列表
                user_ids = None
        else:
            latents = []
            user_ids = None
    else:
        print(f"   单个tensor形状: {data.shape}")
        latents = [data] if data.dim() == 3 else data
        user_ids = None
    
    # 转换为tensor列表
    if not isinstance(latents, torch.Tensor):
        latents = torch.stack(latents) if isinstance(latents[0], torch.Tensor) else torch.tensor(latents)
    
    print(f"✅ 加载了 {len(latents)} 个latents")
    print(f"   Shape: {latents.shape}")
    
    if user_ids is not None:
        print(f"   用户标签: {len(user_ids)} 个，范围 {user_ids.min()}-{user_ids.max()}")
    else:
        print("   ⚠️ 未找到用户标签，将使用0作为默认标签")
    
    # 计算channel-wise统计（与官方一致）
    print("📊 计算channel-wise统计...")
    mean = latents.mean(dim=[0, 2, 3], keepdim=True)  # [1, 32, 1, 1]
    std = latents.std(dim=[0, 2, 3], keepdim=True)
    
    # 保存统计信息（官方格式）
    stats = {
        'mean': mean.squeeze(0),  # [32, 1, 1]
        'std': std.squeeze(0)      # [32, 1, 1]
    }
    stats_file = output_dir / 'latents_stats.pt'
    torch.save(stats, stats_file)
    print(f"✅ 保存统计信息到 {stats_file}")
    
    # 分批保存为safetensors（官方每个文件1000个样本）
    batch_size = 1000
    num_batches = (len(latents) + batch_size - 1) // batch_size
    
    print(f"📦 转换为safetensors格式...")
    for batch_idx in tqdm(range(num_batches), desc="批次"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(latents))
        
        batch_latents = latents[start_idx:end_idx]
        
        # 无数据增强 - 使用相同的latents
        batch_latents_flip = batch_latents  # 不做翻转
        
        # 无条件生成，labels全为0
        batch_labels = torch.zeros(len(batch_latents), dtype=torch.long)
        
        # 保存为safetensors
        safetensors_data = {
            'latents': batch_latents,
            'latents_flip': batch_latents_flip,
            'labels': batch_labels
        }
        
        output_file = output_dir / f'latents_rank00_shard{batch_idx:03d}.safetensors'
        save_file(safetensors_data, output_file)
    
    print(f"✅ 完成！保存了 {num_batches} 个safetensors文件到 {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./latents',
                        help='包含train_latents.pt和val_latents.pt的目录')
    parser.add_argument('--output_dir', type=str, default='./latents_safetensors',
                        help='输出safetensors格式的目录')
    args = parser.parse_args()
    
    # 转换训练集
    print("\n=== 转换训练集 ===")
    train_output = Path(args.output_dir) / 'train'
    convert_to_safetensors(args.input_dir, train_output, 'train')
    
    # 转换验证集
    print("\n=== 转换验证集 ===")
    val_output = Path(args.output_dir) / 'val'
    convert_to_safetensors(args.input_dir, val_output, 'val')
    
    print("\n✅ 全部完成！")
    print(f"请更新配置文件中的data_path为: {train_output}")
    print(f"请更新配置文件中的valid_path为: {val_output}")

if __name__ == "__main__":
    main()
