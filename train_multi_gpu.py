#!/usr/bin/env python3
"""
多GPU训练脚本
专门用于确保正确的多GPU分布式训练
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

def setup_multi_gpu_training():
    """设置多GPU训练环境"""
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行多GPU训练")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"❌ 只有{gpu_count}个GPU，无法进行多GPU训练")
        return False
    
    print(f"✅ 检测到{gpu_count}个GPU，可以进行多GPU训练")
    
    # 显示GPU信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True

def create_ddp_trainer(args, callbacks):
    """创建DDP训练器"""
    
    # 设置DDP策略
    ddp_strategy = DDPStrategy(
        process_group_backend="nccl",  # 使用NCCL后端
        find_unused_parameters=False,   # 提高性能
        static_graph=True,             # 静态图优化
        ddp_comm_hook=None,            # 可以添加通信钩子
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=2,  # 明确指定使用2个GPU
        accelerator='gpu',
        strategy=ddp_strategy,
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=args.output_dir,
        sync_batchnorm=True,  # 同步BatchNorm
        # 添加DDP相关配置
        replace_sampler_ddp=True,  # 自动替换采样器
    )
    
    return trainer

def monitor_gpu_usage():
    """监控GPU使用情况"""
    if not torch.cuda.is_available():
        return
    
    print("\n📊 GPU使用情况:")
    for i in range(torch.cuda.device_count()):
        memory_used = torch.cuda.memory_allocated(i) / 1e9
        memory_cached = torch.cuda.memory_reserved(i) / 1e9
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        utilization = (memory_used / memory_total) * 100
        
        print(f"  GPU {i}: {memory_used:.2f}GB / {memory_total:.1f}GB ({utilization:.1f}%)")
        print(f"    缓存: {memory_cached:.2f}GB")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多GPU DiT训练')
    parser.add_argument('--latent_dir', type=str, required=True, help='潜在特征目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小 (每个GPU)')
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--precision', default='16-mixed', help='精度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print("🎯 多GPU DiT训练")
    print("=" * 50)
    
    # 检查多GPU环境
    if not setup_multi_gpu_training():
        print("❌ 多GPU环境检查失败，退出")
        return
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 导入必要的模块
    from stage2_train_dit import MicroDopplerDataModule, UserConditionedDiT
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    
    # 创建数据模块
    data_module = MicroDopplerDataModule(
        train_latent_file=os.path.join(args.latent_dir, 'train.safetensors'),
        val_latent_file=os.path.join(args.latent_dir, 'val.safetensors'),
        batch_size=args.batch_size,  # 每个GPU的批次大小
        num_workers=4
    )
    
    # 设置数据模块
    data_module.setup()
    
    # 创建模型
    model = UserConditionedDiT(
        num_users=data_module.num_users,
        lr=args.lr
    )
    
    # 设置回调
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='dit-multi-gpu-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 创建DDP训练器
    trainer = create_ddp_trainer(args, callbacks)
    
    print("🚀 开始多GPU训练...")
    print(f"  总批次大小: {args.batch_size * 2} (每GPU: {args.batch_size})")
    
    # 开始训练
    trainer.fit(model, data_module)
    
    # 训练完成后监控GPU使用
    monitor_gpu_usage()
    
    print("✅ 多GPU训练完成!")

if __name__ == "__main__":
    main()
