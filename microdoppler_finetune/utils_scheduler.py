"""
Learning rate scheduler utilities for DiT training
"""
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing学习率调度器
    适合从头训练B模型
    """
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    简单的线性warmup调度器
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def create_scheduler(optimizer, config, total_steps):
    """
    根据配置创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 训练配置
        total_steps: 总训练步数
    
    Returns:
        scheduler: 学习率调度器
    """
    scheduler_type = config.get('scheduler', {}).get('type', 'cosine')
    warmup_steps = config.get('scheduler', {}).get('warmup_steps', int(0.05 * total_steps))
    min_lr = float(config.get('scheduler', {}).get('min_lr', 1e-6))  # 确保转换为float
    
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_steps=warmup_steps,
            max_steps=total_steps,
            min_lr=min_lr
        )
    elif scheduler_type == 'linear_warmup':
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
    elif scheduler_type == 'constant':
        # 保持恒定学习率
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def plot_lr_schedule(config, total_steps, save_path=None):
    """
    可视化学习率调度曲线
    """
    import matplotlib.pyplot as plt
    
    # 创建虚拟优化器
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=config['optimizer']['lr'])
    
    # 创建调度器
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    if scheduler is None:
        print("Using constant learning rate")
        return
    
    # 收集学习率
    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    # 标记关键点
    warmup_steps = config.get('scheduler', {}).get('warmup_steps', int(0.05 * total_steps))
    plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5, label=f'Warmup ends ({warmup_steps} steps)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning rate schedule saved to {save_path}")
    else:
        plt.show()
