#!/usr/bin/env python3
"""
VA-VAE微调脚本 - 完整版本 (集成DINOv2对齐)
基于原项目的核心创新：DINOv2视觉基础模型对齐
支持完整的VA-VAE训练流程
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from tokenizer.vavae import VA_VAE

# 添加DINOv2支持
try:
    import timm
    DINOV2_AVAILABLE = True
    print("✅ DINOv2支持已启用")
except ImportError:
    DINOV2_AVAILABLE = False
    print("⚠️ timm未安装，DINOv2对齐将被禁用")

class MicroDopplerDataset(Dataset):
    """微多普勒数据集 - 用于VA-VAE微调"""
    
    def __init__(self, data_dir, transform=None, max_images_per_user=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = []
        self.user_labels = []
        
        # 收集所有用户目录下的图像
        user_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
        user_dirs.sort()
        
        for user_dir in user_dirs:
            user_id = user_dir.name
            images = list(user_dir.glob('*.png')) + list(user_dir.glob('*.jpg'))
            
            # 限制每个用户的图像数量
            if max_images_per_user and len(images) > max_images_per_user:
                images = images[:max_images_per_user]
            
            self.image_files.extend(images)
            self.user_labels.extend([user_id] * len(images))
        
        print(f"📁 微调数据集: {len(self.image_files)} 张图像，来自 {len(set(self.user_labels))} 个用户")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        user_id = self.user_labels[idx]
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, user_id
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            # 返回黑色图像作为fallback
            black_image = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, user_id

class VAEFineTuner:
    """VA-VAE微调器"""
    
    def __init__(self, vae_model_path, device='cuda', enable_dinov2=True):
        self.device = device
        self.vae_model_path = vae_model_path
        self.enable_dinov2 = enable_dinov2 and DINOV2_AVAILABLE

        # 加载预训练VA-VAE
        print("🔧 加载预训练VA-VAE模型...")
        self.vae = self.load_vae_model()

        # 加载DINOv2模型 (VA-VAE的核心创新)
        if self.enable_dinov2:
            print("🔧 加载DINOv2模型 (VA-VAE核心创新)...")
            self.dinov2_model = self.load_dinov2_model()
        else:
            self.dinov2_model = None
            print("⚠️ DINOv2对齐已禁用")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_vae_model(self):
        """加载VA-VAE模型"""
        try:
            config_path = "LightningDiT/tokenizer/configs/vavae_f16d32.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['ckpt_path'] = self.vae_model_path
            
            temp_config = "temp_finetune_vavae_config.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            vae = VA_VAE(config=temp_config)
            print("✅ VA-VAE模型加载成功")
            return vae
        except Exception as e:
            print(f"❌ VA-VAE模型加载失败: {e}")
            return None

    def load_dinov2_model(self):
        """加载DINOv2模型 - VA-VAE的核心创新"""
        try:
            model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
            model.requires_grad_(False)
            model.to(self.device)
            model.eval()
            print("✅ DINOv2模型加载成功")
            return model
        except Exception as e:
            print(f"❌ DINOv2模型加载失败: {e}")
            return None
    
    def create_optimizer(self, learning_rate):
        """创建优化器 - 同时训练编码器和解码器"""
        # 基于研究证据：同时优化全模型
        optimizer = torch.optim.AdamW(
            self.vae.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        return optimizer

    def create_scheduler(self, optimizer, total_steps):
        """创建学习率调度器"""
        # Cosine annealing with warmup
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    def compute_dinov2_features(self, images):
        """计算DINOv2特征 - VA-VAE的核心创新"""
        if self.dinov2_model is None:
            return None

        with torch.no_grad():
            # 调整图像尺寸到224x224 (DINOv2标准输入)
            resized_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            # 获取DINOv2特征 (去掉CLS token)
            features = self.dinov2_model.forward_features(resized_images)[:, 1:]
            # 重塑为空间特征图
            b, n, d = features.shape
            h = w = int(n ** 0.5)  # 16x16 for 224x224 input
            features = features.reshape(b, h, w, d).permute(0, 3, 1, 2)
            return features

    def compute_loss(self, images, vf_weight=0.1):
        """计算完整损失 (重建 + DINOv2对齐)"""
        with torch.cuda.amp.autocast():
            # 编码
            latents = self.vae.model.encode(images).latent_dist.sample()

            # 解码
            reconstructed = self.vae.model.decode(latents).sample

            # 重建损失 (L1 + L2)
            l1_loss = F.l1_loss(reconstructed, images)
            l2_loss = F.mse_loss(reconstructed, images)
            recon_loss = l1_loss + 0.1 * l2_loss

            # KL散度损失
            kl_loss = torch.mean(torch.sum(latents ** 2, dim=[1, 2, 3]))

            # DINOv2对齐损失 (VA-VAE核心创新)
            dinov2_loss = 0.0
            if self.enable_dinov2 and self.dinov2_model is not None:
                # 获取原图和重建图的DINOv2特征
                orig_features = self.compute_dinov2_features(images)
                recon_features = self.compute_dinov2_features(reconstructed)

                if orig_features is not None and recon_features is not None:
                    # 特征对齐损失 (余弦相似度)
                    orig_flat = orig_features.flatten(2)  # [B, D, H*W]
                    recon_flat = recon_features.flatten(2)

                    # 归一化
                    orig_norm = F.normalize(orig_flat, dim=1)
                    recon_norm = F.normalize(recon_flat, dim=1)

                    # 余弦相似度损失
                    cos_sim = F.cosine_similarity(orig_norm, recon_norm, dim=1)
                    dinov2_loss = 1.0 - cos_sim.mean()

            # 总损失
            total_loss = recon_loss + 1e-6 * kl_loss + vf_weight * dinov2_loss

            return total_loss, recon_loss, kl_loss, dinov2_loss, reconstructed
    
    def train_epoch(self, dataloader, optimizer, scheduler, epoch, vf_weight=0.1):
        """训练一个epoch"""
        self.vae.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_dinov2_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            optimizer.zero_grad()

            # 前向传播 (包含DINOv2对齐)
            loss, recon_loss, kl_loss, dinov2_loss, _ = self.compute_loss(images, vf_weight)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.model.parameters(), 1.0)
            optimizer.step()

            # 更新学习率
            if scheduler:
                scheduler.step()

            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            if isinstance(dinov2_loss, torch.Tensor):
                total_dinov2_loss += dinov2_loss.item()

            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            postfix = {
                'Loss': f'{loss.item():.6f}',
                'Recon': f'{recon_loss.item():.6f}',
                'KL': f'{kl_loss.item():.8f}',
                'LR': f'{current_lr:.2e}'
            }
            if self.enable_dinov2:
                postfix['DINOv2'] = f'{dinov2_loss.item():.6f}' if isinstance(dinov2_loss, torch.Tensor) else '0.000'

            pbar.set_postfix(postfix)

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_dinov2_loss = total_dinov2_loss / len(dataloader) if self.enable_dinov2 else 0

        return avg_loss, avg_recon_loss, avg_kl_loss, avg_dinov2_loss
    
    def validate(self, dataloader, vf_weight=0.1):
        """验证"""
        self.vae.model.eval()

        total_loss = 0
        total_recon_loss = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                loss, recon_loss, _, _, _ = self.compute_loss(images, vf_weight)

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)

        return avg_loss, avg_recon_loss
    
    def save_checkpoint(self, epoch, optimizer, loss, save_path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.vae.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        print(f"💾 检查点已保存: {save_path}")
    
    def finetune(self, data_dir, output_dir, config):
        """执行微调 - 完整版本"""
        print("🚀 开始VA-VAE微调 (同时训练编码器和解码器)")
        print("="*60)
        print(f"📊 配置: {config['epochs']} epochs, lr={config['learning_rate']:.2e}, 早停patience={config['patience']}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建数据集
        dataset = MicroDopplerDataset(data_dir, transform=self.transform)

        # 划分训练和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

        print(f"📊 训练集: {len(train_dataset)} 张图像")
        print(f"📊 验证集: {len(val_dataset)} 张图像")

        # 创建优化器和调度器
        optimizer = self.create_optimizer(config['learning_rate'])
        total_steps = len(train_loader) * config['epochs']
        scheduler = self.create_scheduler(optimizer, total_steps)

        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        # 训练循环
        start_time = time.time()

        for epoch in range(1, config['epochs'] + 1):
            print(f"\n🔥 Epoch {epoch}/{config['epochs']}")

            # 训练 (包含DINOv2对齐)
            train_loss, train_recon, train_kl, train_dinov2 = self.train_epoch(
                train_loader, optimizer, scheduler, epoch, config['vf_weight']
            )

            # 验证
            val_loss, val_recon = self.validate(val_loader, config['vf_weight'])

            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 打印结果
            current_lr = optimizer.param_groups[0]['lr']
            result_str = f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            if self.enable_dinov2:
                result_str += f", DINOv2: {train_dinov2:.6f}"
            print(result_str)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 保存最佳模型
                best_model_path = output_path / "best_model.pt"
                torch.save(self.vae.model.state_dict(), best_model_path)
                print(f"✅ 新的最佳模型已保存 (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"⏳ 早停计数: {patience_counter}/{config['patience']}")

            # 早停
            if patience_counter >= config['patience']:
                print(f"🛑 早停触发！最佳验证损失: {best_val_loss:.6f}")
                break

            # 定期保存检查点
            if epoch % 10 == 0:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(epoch, optimizer, train_loss, checkpoint_path)

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！总时间: {total_time/3600:.2f} 小时")
        print(f"📊 最佳验证损失: {best_val_loss:.6f}")

        # 保存最终模型
        final_model_path = output_path / "finetuned_vavae.pt"
        torch.save(self.vae.model.state_dict(), final_model_path)

        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, output_path / "training_curves.png")

        # 保存训练日志
        self.save_training_log(train_losses, val_losses, best_val_loss, total_time, output_path / "training_log.txt")

        return best_model_path

    def plot_training_curves(self, train_losses, val_losses, save_path):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 8))

        # 主图：损失曲线
        plt.subplot(2, 1, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(epochs, val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VA-VAE Fine-tuning Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图：最后50%的损失曲线（放大显示）
        plt.subplot(2, 1, 2)
        start_idx = len(train_losses) // 2
        epochs_zoom = range(start_idx + 1, len(train_losses) + 1)
        plt.plot(epochs_zoom, train_losses[start_idx:], label='Training Loss (Zoomed)', color='blue', alpha=0.7)
        plt.plot(epochs_zoom, val_losses[start_idx:], label='Validation Loss (Zoomed)', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves (Last 50% of Training)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 训练曲线已保存: {save_path}")

    def save_training_log(self, train_losses, val_losses, best_val_loss, total_time, save_path):
        """保存训练日志"""
        with open(save_path, 'w') as f:
            f.write("VA-VAE微调训练日志\n")
            f.write("=" * 40 + "\n")
            f.write(f"总训练时间: {total_time/3600:.2f} 小时\n")
            f.write(f"训练轮数: {len(train_losses)} epochs\n")
            f.write(f"最佳验证损失: {best_val_loss:.6f}\n")
            f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
            f.write(f"最终验证损失: {val_losses[-1]:.6f}\n")
            f.write(f"损失改善: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%\n")
            f.write("\n详细训练历史:\n")
            f.write("Epoch\tTrain Loss\tVal Loss\n")
            for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                f.write(f"{i+1}\t{train_loss:.6f}\t{val_loss:.6f}\n")

        print(f"📝 训练日志已保存: {save_path}")

def run_complete_finetune():
    """完整的微调流程 - 一键运行"""
    print("🚀 VA-VAE完整微调流程")
    print("="*60)

    # 检查环境
    data_dir = "/kaggle/input/dataset"
    vae_model_path = "models/vavae-imagenet256-f16d32-dinov2.pt"
    output_dir = "vavae_finetuned"

    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    if not Path(vae_model_path).exists():
        print(f"❌ 模型文件不存在: {vae_model_path}")
        print("💡 请先运行 step2_download_models.py")
        return False

    # 基于原项目参数的完整微调配置 (集成DINOv2对齐)
    config = {
        'batch_size': 4,           # 适合Kaggle GPU内存 (原项目用8)
        'epochs': 80,              # 原项目总计130，我们适当减少
        'learning_rate': 1e-4,     # 原项目的学习率
        'patience': 15,            # 适当的早停patience
        'vf_weight': 0.1,          # DINOv2对齐权重 (原项目关键参数)
        'enable_dinov2': True,     # 启用VA-VAE核心创新
    }

    print("⚙️ 微调配置 (基于原项目的完整VA-VAE方案):")
    print(f"   同时训练编码器和解码器: ✅")
    print(f"   DINOv2对齐 (核心创新): {'✅' if config['enable_dinov2'] else '❌'}")
    print(f"   最大训练轮数: {config['epochs']} epochs")
    print(f"   学习率: {config['learning_rate']:.2e} (原项目标准)")
    print(f"   DINOv2权重: {config['vf_weight']} (原项目参数)")
    print(f"   早停patience: {config['patience']}")
    print(f"   批次大小: {config['batch_size']}")
    print(f"   预计时间: 4-8小时")
    print(f"   注意: 完整VA-VAE方案，包含语义对齐")

    # 创建微调器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 使用设备: {device}")

    tuner = VAEFineTuner(vae_model_path, device, enable_dinov2=config['enable_dinov2'])
    if tuner.vae is None:
        print("❌ VA-VAE模型加载失败")
        return False

    # 开始微调
    try:
        print(f"\n🚀 开始微调...")
        start_time = time.time()

        best_model_path = tuner.finetune(data_dir, output_dir, config)

        total_time = time.time() - start_time
        print(f"\n🎉 微调完成！总时间: {total_time/3600:.2f} 小时")
        print(f"📁 最佳模型: {best_model_path}")
        print(f"📊 训练日志: {output_dir}/training_log.txt")
        print(f"📈 训练曲线: {output_dir}/training_curves.png")

        # 建议下一步
        print(f"\n💡 下一步建议:")
        print(f"1. 运行评估脚本验证微调效果:")
        print(f"   !python evaluate_finetuned_vae.py")
        print(f"2. 如果效果满意，进入阶段2 UNet扩散模型训练")

        return True

    except Exception as e:
        print(f"❌ 微调失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🎯 VA-VAE微调工具 - 完整版本")
    print("="*50)

    print("📋 使用方法:")
    print("1. 直接运行完整微调:")
    print("   python finetune_vavae.py")
    print("2. 或者在代码中调用:")
    print("   from finetune_vavae import VAEFineTuner")
    print("   tuner = VAEFineTuner('models/vavae-imagenet256-f16d32-dinov2.pt')")
    print("   config = {'batch_size': 4, 'epochs': 100, 'learning_rate': 2e-5, 'patience': 10}")
    print("   tuner.finetune('/kaggle/input/dataset', 'vavae_finetuned', config)")

    # 运行完整微调
    success = run_complete_finetune()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
