"""
最小化训练脚本修改
基于LightningDiT的训练脚本，仅添加用户条件功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse
from pathlib import Path

# 导入我们的最小修改模块
from minimal_micro_doppler_dataset import create_micro_doppler_dataloader, MicroDopplerDataset
from minimal_vavae_modification import UserConditionedVAVAE


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='微多普勒VA-VAE训练 - 最小修改版本')

    parser.add_argument('--data_dir', type=str, required=True, help='微多普勒数据目录')
    parser.add_argument('--original_vavae', type=str, required=True, help='原始VA-VAE模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小（每个GPU）')
    parser.add_argument('--condition_dim', type=int, default=128, help='条件向量维度')
    parser.add_argument('--kl_weight', type=float, default=1e-6, help='KL散度权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # PyTorch Lightning 训练参数（遵循原项目方式）
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--devices', type=int, default=1, help='GPU数量')
    parser.add_argument('--num_nodes', type=int, default=1, help='节点数量')
    parser.add_argument('--strategy', type=str, default='auto', help='训练策略 (auto, ddp, ddp_find_unused_parameters_true)')
    parser.add_argument('--accelerator', type=str, default='gpu', help='加速器类型')
    parser.add_argument('--precision', type=str, default='32', help='精度 (16, 32, bf16)')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')

    return parser.parse_args()


class MicroDopplerVAVAEModule(pl.LightningModule):
    """
    PyTorch Lightning模块包装用户条件化VA-VAE
    遵循原项目的训练方式
    """

    def __init__(self, original_vavae, num_users, condition_dim=128, lr=1e-4, kl_weight=1e-6):
        super().__init__()
        self.save_hyperparameters(ignore=['original_vavae'])

        # 创建用户条件化模型
        self.model = UserConditionedVAVAE(
            original_vavae=original_vavae,
            num_users=num_users,
            condition_dim=condition_dim
        )

        self.lr = lr
        self.kl_weight = kl_weight

    def forward(self, x, user_ids=None):
        return self.model(x, user_ids)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        user_ids = batch.get('user_id', None)

        # 前向传播
        reconstructed, posterior = self.model(images, user_ids)

        # 计算损失
        recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='mean')
        kl_loss = posterior.kl().mean()
        total_loss = recon_loss + self.kl_weight * kl_loss

        # 记录损失
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/kl_loss', kl_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        user_ids = batch.get('user_id', None)

        # 前向传播
        reconstructed, posterior = self.model(images, user_ids)

        # 计算损失
        recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='mean')
        kl_loss = posterior.kl().mean()
        total_loss = recon_loss + self.kl_weight * kl_loss

        # 记录损失
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/kl_loss', kl_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MicroDopplerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning数据模块
    遵循原项目的数据加载方式
    """

    def __init__(self, data_dir, batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_users = None

    def setup(self, stage=None):
        # 创建训练和验证数据集
        self.train_dataset = MicroDopplerDataset(self.data_dir, split='train')
        self.val_dataset = MicroDopplerDataset(self.data_dir, split='val')

        # 获取用户数量
        self.num_users = self.train_dataset.num_users

        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
        print(f"用户数量: {self.num_users}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def setup_seed(seed):
    """设置随机种子"""
    pl.seed_everything(seed)
    print(f"设置随机种子: {seed}")


def main():
    """主训练函数 - 使用PyTorch Lightning"""
    args = parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("微多普勒VA-VAE训练 - PyTorch Lightning版本")
    print("=" * 60)

    # 1. 创建数据模块
    print("1. 创建数据模块...")
    data_module = MicroDopplerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    # 设置数据模块以获取用户数量
    data_module.setup()
    num_users = data_module.num_users

    # 2. 创建模型
    print("2. 创建用户条件化VA-VAE模型...")
    print(f"注意: 需要实际的VA-VAE模型文件: {args.original_vavae}")
    print("当前为演示版本，请替换为实际的模型加载代码")

    # 创建一个简单的测试模型 (实际使用时需要替换)
    class DummyVAVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(256, 512, 4, 2, 1),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Sigmoid()
            )
            self.quant_conv = nn.Conv2d(512, 1024, 1)  # 输出2倍通道用于均值和方差
            self.post_quant_conv = nn.Conv2d(512, 512, 1)

    dummy_vavae = DummyVAVAE()

    # 创建Lightning模块
    model = MicroDopplerVAVAEModule(
        original_vavae=dummy_vavae,
        num_users=num_users,
        condition_dim=args.condition_dim,
        lr=args.lr,
        kl_weight=args.kl_weight
    )

    # 3. 设置回调函数
    print("3. 设置训练回调...")
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='best-{epoch:02d}-{val/total_loss:.4f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # 4. 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='lightning_logs'
    )

    # 5. 创建训练器
    print("4. 创建PyTorch Lightning训练器...")
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True
    )

    print(f"训练配置:")
    print(f"  - 最大轮数: {args.max_epochs}")
    print(f"  - GPU数量: {args.devices}")
    print(f"  - 训练策略: {args.strategy}")
    print(f"  - 批次大小: {args.batch_size} (每个GPU)")
    print(f"  - 学习率: {args.lr}")

    # 6. 开始训练
    print("5. 开始训练...")
    trainer.fit(model, data_module)

    print("\n训练完成！")
    print(f"最佳模型保存在: {output_dir / 'checkpoints'}")
    print(f"训练日志保存在: {output_dir / 'lightning_logs'}")


if __name__ == '__main__':
    main()



