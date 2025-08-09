#!/usr/bin/env python3
"""
🚀 ConditionalDiT快速验证脚本
验证简化版本是否可行，无需完整epoch训练
总预计时间：5-10分钟
"""
import os
import sys
import torch
import torch._dynamo
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 清除dynamo缓存
torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_smoke_test(trainer, train_loader, device):
    """烟雾测试：验证基本前向/反向传播"""
    print("🔥 开始烟雾测试...")
    trainer.model.train()
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5: break
            
            # 使用trainer的compute_loss方法
            loss = trainer.compute_loss(batch)
            
            # 🔍 显示损失组件分解
            if hasattr(trainer, 'log_losses'):
                logs = trainer.log_losses
                print(f"  Batch {batch_idx}: Total = {loss:.4f}")
                print(f"    - Diffusion: {logs['diffusion_loss']:.4f}")  
                print(f"    - Contrastive: {logs['contrastive_loss']:.4f} (weight: {logs['contrastive_weight']:.3f})")
                print(f"    - Inter-user: {logs['inter_user_loss']:.4f}")
                print(f"    - Regularization: {logs['user_regularization']:.4f}")
            else:
                print(f"  Batch {batch_idx}: Loss = {loss:.4f}")
            
            # 反向传播
            trainer.optimizer.zero_grad()
            loss.backward()
            
        print("✅ 烟雾测试通过：基本训练流程工作正常")
        return True
        
    except Exception as e:
        print(f"❌ 烟雾测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def loss_trend_test(trainer, train_loader, device, num_steps=50):
    """损失趋势测试：观察损失是否有改善趋势"""
    print(f"📊 开始损失趋势测试 ({num_steps}步)...")
    trainer.model.train()
    
    losses = []
    start_time = time.time()
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_steps: break
            
            # 使用trainer的compute_loss方法
            loss = trainer.compute_loss(batch)
            
            # 优化步骤
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()  # 使用真实的优化器步骤
            
            losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                recent_avg = np.mean(losses[-5:])
                print(f"  Step {batch_idx}: Loss = {loss:.4f}, Recent avg = {recent_avg:.4f}")
        
        # 分析趋势
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        elapsed = time.time() - start_time
        print(f"📈 损失趋势分析:")
        print(f"  初始损失: {initial_loss:.4f}")
        print(f"  最终损失: {final_loss:.4f}")
        print(f"  改善百分比: {improvement:.1f}%")
        print(f"  用时: {elapsed:.1f}秒")
        
        # 判断标准
        if improvement > 5:
            print("✅ 损失趋势测试通过：模型显示学习能力")
            return True
        else:
            print("⚠️ 损失趋势测试警告：改善幅度较小")
            return False
            
    except Exception as e:
        print(f"❌ 损失趋势测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def gradient_health_check(trainer, train_loader, device):
    """梯度健康检查"""
    print("🔬 开始梯度健康检查...")
    trainer.model.train()
    
    try:
        batch = next(iter(train_loader))
        
        # 使用trainer的compute_loss方法
        loss = trainer.compute_loss(batch)
        trainer.optimizer.zero_grad()
        loss.backward()
        
        # 梯度统计
        grad_stats = {}
        total_params = 0
        params_with_grad = 0
        grad_norms = []
        
        for name, param in trainer.model.named_parameters():
            total_params += 1
            
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                grad_stats[name] = grad_norm
            else:
                grad_stats[name] = 0.0
        
        # 分析结果
        print(f"📊 梯度统计:")
        print(f"  总参数: {total_params}")
        print(f"  有梯度参数: {params_with_grad}")
        print(f"  梯度覆盖率: {params_with_grad/total_params*100:.1f}%")
        
        if grad_norms:
            print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
            print(f"  梯度范数范围: [{np.min(grad_norms):.6f}, {np.max(grad_norms):.6f}]")
        
        # 显示关键层的梯度
        key_layers = ['condition_encoder', 'dit.y_embedder', 'dit.blocks.0']
        print(f"🔍 关键层梯度:")
        for name, grad_norm in grad_stats.items():
            if any(key in name for key in key_layers):
                status = "✅" if grad_norm > 1e-8 else "❌"
                print(f"  {status} {name}: {grad_norm:.6f}")
        
        # 判断梯度健康
        healthy_grads = sum(1 for g in grad_norms if 1e-8 < g < 10)
        health_ratio = healthy_grads / len(grad_norms) if grad_norms else 0
        
        if health_ratio > 0.8:
            print("✅ 梯度健康检查通过：梯度流正常")
            return True
        else:
            print("⚠️ 梯度健康检查警告：可能存在梯度问题")
            return False
            
    except Exception as e:
        print(f"❌ 梯度健康检查失败: {str(e)}")
        return False

def main():
    """主验证流程"""
    print("=" * 60)
    print("🚀 ConditionalDiT快速验证开始")
    print("=" * 60)
    
    try:
        # 导入模块
        from step5_conditional_dit_training import ConditionalDiTTrainer
        from step4_microdoppler_adapter import MicroDopplerDataModule
        
        # 设备设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {device}")
        
        # 创建数据模块（小批次）
        print("📦 创建数据模块...")
        data_module = MicroDopplerDataModule(
            data_dir="/kaggle/input/dataset",  # 根据实际路径调整
            batch_size=4,  # 小批次快速测试
            num_workers=0  # 避免多进程问题
        )
        data_module.setup()
        train_loader = data_module.train_dataloader()
        
        # 创建模型
        print("🤖 创建ConditionalDiT模型...")
        
        # 嵌套配置字典结构
        config = {
            'model': {
                'params': {
                    'model': "LightningDiT-XL/1",
                    'num_users': 31,
                    'condition_dim': 1152,
                    'frozen_backbone': False,
                    'dropout': 0.15
                    # 不包含pretrained_path，避免与_setup_model中的硬编码冲突
                }
            },
            'data': {
                'params': {
                    'data_dir': "/kaggle/input/dataset",
                    'batch_size': 16,  # 🚀 增加到16，提高用户重复概率
                    'num_workers': 4   # 双GPU环境增加workers
                }
            },
            'optimizer': {
                'params': {
                    'lr': 1e-5,              # ✅ PyTorch AdamW使用'lr'不是'learning_rate'
                    'weight_decay': 1e-4,
                    'betas': [0.9, 0.999]
                }
            },
            'training': {
                'max_epochs': 1,
                'contrastive_weight': 0.1,
                'regularization_weight': 0.01,
                'warmup_steps': 100,
                'gradient_clip_val': 1.0
            }
        }
        
        trainer = ConditionalDiTTrainer(config)
        
        # 验证测试 - 使用trainer而不是model
        results = {}
        
        # 1. 烟雾测试
        results['smoke_test'] = quick_smoke_test(trainer, train_loader, device)
        
        # 2. 损失趋势测试  
        results['loss_trend'] = loss_trend_test(trainer, train_loader, device, num_steps=30)
        
        # 3. 梯度健康检查
        results['gradient_health'] = gradient_health_check(trainer, train_loader, device)
        
        # 最终评估
        print("\n" + "=" * 60)
        print("📋 快速验证结果汇总")
        print("=" * 60)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name:20s}: {status}")
        
        print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests >= 2:
            print("🎉 验证成功！简化版ConditionalDiT基本可行")
            print("💡 建议：可以进行短时训练或添加更复杂的条件机制")
        else:
            print("⚠️ 验证失败！需要修复基础问题")
            print("💡 建议：检查模型实现和数据管道")
        
        return passed_tests >= 2
        
    except Exception as e:
        print(f"❌ 验证过程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
