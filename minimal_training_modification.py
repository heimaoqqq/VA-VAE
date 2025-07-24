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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import os
import argparse
import time
from pathlib import Path

# 导入我们的最小修改模块
from minimal_micro_doppler_dataset import create_micro_doppler_dataloader, MicroDopplerDataset
from minimal_vavae_modification import UserConditionedVAVAE

class TrainingSummaryCallback(Callback):
    """自定义回调：提供清晰的训练总结"""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        """记录epoch开始时间"""
        # 只在主进程输出
        if trainer.is_global_zero:
            self.epoch_start_time = time.time()
            print(f"\n🚀 Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} 开始训练...")

    def on_train_epoch_end(self, trainer, pl_module):
        """训练epoch结束总结"""
        # 只在主进程输出
        if trainer.is_global_zero:
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            metrics = trainer.callback_metrics
            train_loss = metrics.get('train/loss', 0.0)

            print(f"⏱️  训练完成 - 用时: {epoch_time:.1f}s, 损失: {train_loss:.6f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束总结"""
        # 只在主进程输出
        if trainer.is_global_zero:
            metrics = trainer.callback_metrics
            val_loss = metrics.get('val/loss', 0.0)
            train_loss = metrics.get('train/loss', 0.0)
            val_recon = metrics.get('val/recon', 0.0)
            val_kl = metrics.get('val/kl', 0.0)

            print(f"📊 Epoch {trainer.current_epoch + 1} 总结:")
            print(f"   训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
            print(f"   验证重建: {val_recon:.6f} | 验证KL: {val_kl:.2f}")

            # 显示改进情况
            if hasattr(self, 'best_val_loss'):
                if val_loss < self.best_val_loss:
                    print(f"   🎉 验证损失改进! ({self.best_val_loss:.6f} → {val_loss:.6f})")
                    self.best_val_loss = val_loss
                else:
                    print(f"   📈 当前最佳: {self.best_val_loss:.6f}")
            else:
                self.best_val_loss = val_loss
                print(f"   🎯 初始验证损失: {val_loss:.6f}")

            print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='微多普勒VA-VAE训练 - 最小修改版本')

    parser.add_argument('--data_dir', type=str, required=True, help='微多普勒数据目录')
    parser.add_argument('--original_vavae', type=str, required=True, help='原始VA-VAE模型路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小（每个GPU）')
    parser.add_argument('--condition_dim', type=int, default=128, help='条件向量维度')
    parser.add_argument('--kl_weight', type=float, default=1e-4, help='KL散度权重')
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
        self.kl_weight_max = 1e-3  # 最大KL权重
        self.kl_weight_min = 1e-6  # 最小KL权重

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

        # KL散度裁剪，防止爆炸
        kl_loss = torch.clamp(kl_loss, max=1000.0)

        # 动态KL权重：根据KL值大小调整权重
        if kl_loss > 10000:
            current_kl_weight = self.kl_weight_min  # 使用最小权重
        elif kl_loss > 1000:
            current_kl_weight = self.kl_weight_min * 10  # 稍微增加
        else:
            current_kl_weight = self.kl_weight  # 使用原始权重

        total_loss = recon_loss + current_kl_weight * kl_loss

        # 记录损失（简化显示）
        self.log('train/loss', total_loss, prog_bar=True, logger=True)
        self.log('train/recon', recon_loss, prog_bar=False, logger=True)
        self.log('train/kl', kl_loss, prog_bar=False, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        user_ids = batch.get('user_id', None)

        # 前向传播
        reconstructed, posterior = self.model(images, user_ids)

        # 计算损失
        recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='mean')
        kl_loss = posterior.kl().mean()

        # KL散度裁剪，防止爆炸
        kl_loss = torch.clamp(kl_loss, max=1000.0)

        # 动态KL权重：根据KL值大小调整权重
        if kl_loss > 10000:
            current_kl_weight = self.kl_weight_min  # 使用最小权重
        elif kl_loss > 1000:
            current_kl_weight = self.kl_weight_min * 10  # 稍微增加
        else:
            current_kl_weight = self.kl_weight  # 使用原始权重

        total_loss = recon_loss + current_kl_weight * kl_loss

        # 记录损失（添加分布式同步）
        self.log('val/loss', total_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/recon', recon_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/kl', kl_loss, prog_bar=False, logger=True, sync_dist=True)

        return total_loss

    def on_train_epoch_end(self):
        """训练epoch结束时的总结 - 移除，避免重复"""
        pass

    def on_validation_epoch_end(self):
        """验证epoch结束时的总结 - 移除，避免重复"""
        pass

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


def create_dummy_vavae():
    """创建测试用的DummyVAVAE模型"""
    class DummyVAVAE(nn.Module):
        def __init__(self):
            super().__init__()
            # 编码器: 256x256 -> 16x16 (16倍下采样)
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),    # 256->128
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),  # 128->64
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1), # 64->32
                nn.ReLU(),
                nn.Conv2d(256, 512, 4, 2, 1), # 32->16
                nn.ReLU()
            )
            # 解码器: 16x16 -> 256x256
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16->32
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32->64
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64->128
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 128->256
                nn.Sigmoid()
            )
            # 量化层
            self.quant_conv = nn.Conv2d(512, 64, 1)  # 输出32通道的均值和方差
            self.post_quant_conv = nn.Conv2d(32, 512, 1)

        def encode(self, x):
            """编码"""
            h = self.encoder(x)
            moments = self.quant_conv(h)
            # 分离均值和方差
            mean, logvar = torch.chunk(moments, 2, dim=1)
            return mean, logvar

        def decode(self, z):
            """解码"""
            z = self.post_quant_conv(z)
            return self.decoder(z)

        def forward(self, x):
            """前向传播"""
            mean, logvar = self.encode(x)
            # 重参数化技巧
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            recon = self.decode(z)
            return recon, mean, logvar

    return DummyVAVAE()


def main():
    """主训练函数 - 使用PyTorch Lightning"""
    args = parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 微多普勒VA-VAE用户条件化训练")
    print("=" * 50)

    # 1. 创建数据模块
    print("📊 创建数据模块...")
    data_module = MicroDopplerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    # 设置数据模块以获取用户数量
    data_module.setup()
    num_users = data_module.num_users

    # 2. 创建模型
    print("🤖 创建用户条件化VA-VAE模型...")

    # 加载预训练的VA-VAE模型
    if os.path.exists(args.original_vavae):
        print(f"📥 加载预训练模型: {Path(args.original_vavae).name}")
        try:
            # 从LightningDiT导入VA-VAE模型
            import sys
            sys.path.append('LightningDiT')
            from tokenizer.autoencoder import AutoencoderKL

            # 创建VA-VAE模型实例并直接加载权重
            original_vavae = AutoencoderKL(
                embed_dim=32,  # f16d32配置
                ch_mult=(1, 1, 2, 2, 4),
                ckpt_path=args.original_vavae,  # 直接使用ckpt_path参数
                model_type='vavae'
            )

            print("✅ 预训练VA-VAE模型加载成功")

        except Exception as e:
            print(f"❌ 加载预训练模型失败: {e}")
            print("使用DummyVAVAE进行测试...")
            original_vavae = create_dummy_vavae()
    else:
        print(f"⚠️ 预训练模型文件不存在: {args.original_vavae}")
        print("使用DummyVAVAE进行测试...")
        original_vavae = create_dummy_vavae()

    # 创建Lightning模块
    model = MicroDopplerVAVAEModule(
        original_vavae=original_vavae,
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
            filename='best-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TrainingSummaryCallback()  # 添加自定义训练总结回调
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
        log_every_n_steps=200,  # 减少日志频率
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=False,  # 简化输出
        enable_checkpointing=True,
        num_sanity_val_steps=0  # 跳过sanity check，减少输出
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



