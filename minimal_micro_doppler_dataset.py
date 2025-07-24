"""
最小化微多普勒数据集 - 直接适配LightningDiT
仅做必要的数据加载，不做任何数据增强
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class MicroDopplerDataset(Dataset):
    """
    最简单的微多普勒数据集类
    直接适配LightningDiT的数据格式要求
    支持数据集划分
    """

    def __init__(self, data_dir, img_size=256, split='train', original_structure=False):
        """
        初始化数据集

        Args:
            data_dir: 数据目录，可以是原始目录或划分后的目录
            img_size: 图像尺寸 (LightningDiT默认256)
            split: 数据集划分 ('train', 'val', 'test', 'all')
            original_structure: 是否是原始的ID_1, ID_2...结构
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.split = split
        self.original_structure = original_structure

        # 根据split确定数据目录
        if split in ['train', 'val', 'test'] and not original_structure:
            # 检查是否存在划分后的目录
            split_dir = self.data_dir / split
            if split_dir.exists():
                scan_dir = split_dir
                print(f"使用划分后的数据: {scan_dir}")
            else:
                scan_dir = self.data_dir
                print(f"未找到划分目录，使用原始数据: {scan_dir}")
        else:
            scan_dir = self.data_dir

        # 扫描数据文件
        self.data_files = []

        if original_structure:
            # 原始结构: ID_1/, ID_2/, ..., ID_31/
            self._scan_original_structure(scan_dir)
        else:
            # 划分后结构: train/, val/, test/ 包含重命名的文件
            self._scan_split_structure(scan_dir)

        print(f"加载了 {len(self.data_files)} 个微多普勒样本 (split: {split})")
        
        # 统计用户信息
        self.user_ids = sorted(list(set(item['user_id'] for item in self.data_files)))
        self.num_users = len(self.user_ids)
        print(f"用户数量: {self.num_users}, 用户ID: {self.user_ids}")

    def _scan_original_structure(self, scan_dir):
        """扫描原始ID_1, ID_2...结构"""
        image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

        for user_dir in scan_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    # 解析用户ID: ID_1 -> 1, ID_2 -> 2, ...
                    user_id = int(user_dir.name.split('_')[1])

                    # 扫描该用户目录下的所有图像文件
                    sample_id = 1
                    for ext in image_extensions:
                        for file_path in sorted(user_dir.glob(f"*.{ext}")):
                            self.data_files.append({
                                'file_path': str(file_path),
                                'user_id': user_id,
                                'sample_id': sample_id
                            })
                            sample_id += 1

                except (ValueError, IndexError):
                    print(f"警告: 无法解析目录名 {user_dir.name}")

    def _scan_split_structure(self, scan_dir):
        """扫描划分后的结构"""
        # 扫描所有图像文件（假设已经重命名为user_XX_sample_XXX格式）
        image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'npy']

        for ext in image_extensions:
            for file_path in scan_dir.glob(f"*.{ext}"):
                # 解析文件名: user_XX_sample_XXX.ext
                parts = file_path.stem.split('_')
                if len(parts) >= 4 and parts[0] == 'user':
                    try:
                        user_id = int(parts[1])
                        sample_id = int(parts[3])
                        self.data_files.append({
                            'file_path': str(file_path),
                            'user_id': user_id,
                            'sample_id': sample_id
                        })
                    except ValueError:
                        print(f"警告: 无法解析文件名 {file_path.name}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """
        获取数据项 - 保持LightningDiT的数据格式
        
        Returns:
            dict: {
                'image': torch.Tensor,  # (3, H, W) - LightningDiT期望的格式
                'user_id': int         # 用户ID - 新增的条件信息
            }
        """
        item = self.data_files[idx]

        # 加载时频图数据
        file_path = Path(item['file_path'])

        if file_path.suffix.lower() == '.npy':
            # NumPy文件
            spectrogram = np.load(file_path)
        else:
            # 图像文件
            from PIL import Image
            pil_img = Image.open(file_path)
            spectrogram = np.array(pil_img)
        
        # 预处理: 调整尺寸到目标大小
        if spectrogram.shape != (self.img_size, self.img_size):
            # 使用PIL进行尺寸调整
            if spectrogram.ndim == 2:
                # 转换为PIL图像
                # 归一化到0-255范围
                spec_norm = ((spectrogram - spectrogram.min()) / 
                           (spectrogram.max() - spectrogram.min()) * 255).astype(np.uint8)
                pil_img = Image.fromarray(spec_norm, mode='L')
                pil_img = pil_img.resize((self.img_size, self.img_size), Image.LANCZOS)
                spectrogram = np.array(pil_img).astype(np.float32) / 255.0
        
        # 转换为3通道 (LightningDiT期望RGB格式)
        if spectrogram.ndim == 2:
            # 灰度 -> RGB: 复制3次，格式为 (3, H, W)
            spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
        elif spectrogram.ndim == 3:
            if spectrogram.shape[2] == 3:
                # (H, W, 3) -> (3, H, W) - RGB图像
                spectrogram = np.transpose(spectrogram, (2, 0, 1))
            elif spectrogram.shape[2] == 1:
                # (H, W, 1) -> (3, H, W) - 单通道图像
                spectrogram = spectrogram.squeeze(2)
                spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
            elif spectrogram.shape[0] == 1:
                # (1, H, W) -> (3, H, W)
                spectrogram = np.repeat(spectrogram, 3, axis=0)
            elif spectrogram.shape[0] == 3:
                # 已经是 (3, H, W) 格式，无需转换
                pass
            else:
                # 其他情况，强制转换为 (3, H, W)
                if spectrogram.shape[0] != 3:
                    # 取第一个通道并复制3次
                    if len(spectrogram.shape) == 3:
                        spectrogram = spectrogram[0]  # 取第一个通道
                    spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=0)

        # 确保数据范围在[0, 1]
        if spectrogram.max() > 1.0:
            spectrogram = spectrogram / 255.0

        # 转换为tensor，确保维度为 (C, H, W)
        image_tensor = torch.from_numpy(spectrogram).float()

        # 最终检查：确保张量维度正确
        if image_tensor.dim() == 3 and image_tensor.shape[0] != 3:
            # 如果第一个维度不是3，进行转换
            if image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # 验证最终维度
        assert image_tensor.dim() == 3, f"期望3维张量，得到{image_tensor.dim()}维"
        assert image_tensor.shape[0] == 3, f"期望3个通道，得到{image_tensor.shape[0]}个通道"
        assert image_tensor.shape[1] == self.img_size, f"期望高度{self.img_size}，得到{image_tensor.shape[1]}"
        assert image_tensor.shape[2] == self.img_size, f"期望宽度{self.img_size}，得到{image_tensor.shape[2]}"
        
        return {
            'image': image_tensor,      # LightningDiT期望的键名
            'user_id': item['user_id']  # 新增的用户条件
        }


def create_micro_doppler_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4, split='train', original_structure=False):
    """
    创建微多普勒数据加载器 - 直接替换LightningDiT的数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        split: 数据集划分
        original_structure: 是否是原始的ID_1, ID_2...结构

    Returns:
        DataLoader: 可直接用于LightningDiT训练的数据加载器
        num_users: 用户数量
    """
    dataset = MicroDopplerDataset(data_dir, split=split, original_structure=original_structure)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # LightningDiT通常需要固定批次大小
    )

    return dataloader, dataset.num_users


def create_split_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    创建训练/验证/测试数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数

    Returns:
        train_loader, val_loader, test_loader, num_users
    """
    train_loader, num_users = create_micro_doppler_dataloader(
        data_dir, batch_size, shuffle=True, num_workers=num_workers, split='train'
    )
    val_loader, _ = create_micro_doppler_dataloader(
        data_dir, batch_size, shuffle=False, num_workers=num_workers, split='val'
    )
    test_loader, _ = create_micro_doppler_dataloader(
        data_dir, batch_size, shuffle=False, num_workers=num_workers, split='test'
    )

    return train_loader, val_loader, test_loader, num_users


# 使用示例
if __name__ == "__main__":
    # 测试数据集
    data_dir = "data/processed"  # 您的数据目录
    
    if os.path.exists(data_dir):
        dataset = MicroDopplerDataset(data_dir)
        
        print(f"数据集大小: {len(dataset)}")
        print(f"用户数量: {dataset.num_users}")
        
        # 测试加载一个样本
        sample = dataset[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"用户ID: {sample['user_id']}")
        print(f"数值范围: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        
        # 测试数据加载器
        dataloader, num_users = create_micro_doppler_dataloader(data_dir, batch_size=4)
        batch = next(iter(dataloader))
        print(f"批次图像形状: {batch['image'].shape}")
        print(f"批次用户ID: {batch['user_id']}")
    else:
        print(f"数据目录不存在: {data_dir}")
        print("请将您的微多普勒数据放在该目录下，文件名格式: user_XX_sample_XXX.npy")
