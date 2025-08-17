#!/usr/bin/env python3
"""
继续训练DiT模型 - 从checkpoint恢复训练
目标：将损失从0.21降到<0.15
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.append('/kaggle/working/VA-VAE')
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')

# 设置环境变量
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datetime import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.img_latent_dataset import build_dit_dataset

# DiT模型
from models import DiT_XL_2

def continue_training():
    """继续训练DiT模型"""
    
    print("="*60)
    print("🔄 继续训练DiT模型")
    print("="*60)
    
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/kaggle/working/dit_outputs/checkpoints/best_model_epoch20_val0.1888.pt'
    output_dir = Path('/kaggle/working/dit_outputs_continued')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练配置 - 调整以加速收敛
    config = {
        'batch_size': 16,  # 增大批量
        'learning_rate': 5e-5,  # 降低学习率（微调阶段）
        'num_epochs': 30,  # 继续训练30轮
        'gradient_accumulation': 2,
        'validate_every': 5,
        'save_every': 5,
        'target_loss': 0.15,  # 目标损失
    }
    
    print("📊 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = DiT_XL_2(
        input_size=16,
        num_classes=31,
        in_channels=32,
        learn_sigma=True,
    ).to(device)
    
    # 加载checkpoint
    if Path(checkpoint_path).exists():
        print(f"✅ 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 20) + 1
            best_val_loss = checkpoint.get('best_val_loss', 0.1888)
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 21
            best_val_loss = 0.1888
            
        print(f"   开始轮次: {start_epoch}")
        print(f"   当前最佳验证损失: {best_val_loss:.4f}")
    else:
        print("❌ 未找到checkpoint，退出")
        return
    
    # 数据集
    print("\n📊 加载数据集...")
    train_dataset = build_dit_dataset(
        name='custom',
        data_dir='/kaggle/working/latent_data',
        phase='train',
        seed=42
    )
    
    val_dataset = build_dit_dataset(
        name='custom',
        data_dir='/kaggle/working/latent_data',
        phase='val',
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   训练样本: {len(train_dataset)}")
    print(f"   验证样本: {len(val_dataset)}")
    
    # 优化器 - 使用较小的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器 - 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    print("\n🎯 开始训练...")
    print(f"   目标: 损失 < {config['target_loss']}")
    
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            
            # 时间步采样
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            
            # 添加噪声
            noise = torch.randn_like(x)
            noisy_x = x + noise * (t.float() / 1000).view(-1, 1, 1, 1)
            
            # 前向传播
            pred = model(noisy_x, t, y)
            
            # 计算损失
            if pred.shape[1] == 64:  # learn_sigma
                pred_noise, pred_var = pred.chunk(2, dim=1)
                loss = criterion(pred_noise, noise)
            else:
                loss = criterion(pred, noise)
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
            # 早停检查
            if loss.item() < config['target_loss']:
                print(f"\n🎉 达到目标损失 {loss.item():.4f} < {config['target_loss']}")
                break
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        
        # 验证阶段
        if epoch % config['validate_every'] == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    
                    t = torch.randint(0, 1000, (x.shape[0],), device=device)
                    noise = torch.randn_like(x)
                    noisy_x = x + noise * (t.float() / 1000).view(-1, 1, 1, 1)
                    
                    pred = model(noisy_x, t, y)
                    
                    if pred.shape[1] == 64:
                        pred_noise, _ = pred.chunk(2, dim=1)
                        loss = criterion(pred_noise, noise)
                    else:
                        loss = criterion(pred, noise)
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            print(f"\nEpoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = output_dir / f'best_model_epoch{epoch}_val{avg_val_loss:.4f}.pt'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                }, save_path)
                
                print(f"✅ 保存最佳模型: {save_path}")
                
                # 检查是否达到目标
                if avg_val_loss < config['target_loss']:
                    print(f"\n🎉 达到目标验证损失 {avg_val_loss:.4f} < {config['target_loss']}")
                    print("训练完成！")
                    return save_path
        
        # 定期保存
        if epoch % config['save_every'] == 0:
            save_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"   学习率: {current_lr:.6f}")
    
    print("\n✅ 训练完成")
    print(f"   最终验证损失: {best_val_loss:.4f}")
    
    if best_val_loss > config['target_loss']:
        print(f"⚠️ 未达到目标损失，建议继续训练或使用增强版训练脚本")
    
    return output_dir / 'best_model.pt'

if __name__ == "__main__":
    # 运行继续训练
    best_model_path = continue_training()
    
    if best_model_path:
        print(f"\n📁 最佳模型保存在: {best_model_path}")
        print("请使用此模型重新运行 step7_optimized_sampling.py")
