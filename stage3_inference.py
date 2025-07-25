#!/usr/bin/env python3
"""
阶段3: 图像生成推理
基于LightningDiT原项目的inference.py
使用训练好的DiT模型生成用户条件化的微多普勒图像
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from accelerate import Accelerator

# 导入LightningDiT组件
import sys
import os

# 确保正确的路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
lightningdit_path = os.path.join(current_dir, 'LightningDiT')
if lightningdit_path not in sys.path:
    sys.path.append(lightningdit_path)

from models.lightningdit import LightningDiT_models
from transport import create_transport
from tokenizer.vavae import VA_VAE

def test_imports():
    """测试导入是否正常"""
    print("🧪 测试导入...")

    try:
        # 测试RMSNorm
        try:
            from simple_rmsnorm import RMSNorm
            print("✅ simple_rmsnorm导入成功")
        except ImportError as e:
            print(f"⚠️  simple_rmsnorm导入失败: {e}")

        # 测试LightningDiT模型
        from models import LightningDiT_models
        print("✅ LightningDiT_models导入成功")

        # 测试Transport
        from transport import create_transport
        print("✅ Transport导入成功")

        # 测试VA-VAE
        from tokenizer.vavae import VA_VAE
        print("✅ VA_VAE导入成功")

        # 测试Safetensors
        from safetensors.torch import load_file
        print("✅ Safetensors导入成功")

        # 测试模型创建
        model = LightningDiT_models['LightningDiT-B/1'](
            input_size=16,
            num_classes=31,
            in_channels=32,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        print("✅ 模型创建成功")

        return True

    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

class MicroDopplerGenerator:
    """微多普勒图像生成器 (基于原项目inference.py)"""

    def __init__(self, dit_checkpoint, vavae_config, model_name='LightningDiT-B/1', accelerator=None):
        self.model_name = model_name
        # 支持双GPU推理
        if accelerator is not None:
            self.accelerator = accelerator
            self.device = accelerator.device
            self.is_distributed = True
        else:
            self.accelerator = None
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.is_distributed = False
        
        # 加载VA-VAE (VA-VAE在初始化时已经设置为eval模式并移到GPU)
        if not self.is_distributed or self.accelerator.is_main_process:
            print("📥 加载VA-VAE...")
        self.vavae = VA_VAE(vavae_config)

        # 设置潜在特征的归一化参数 (参考原项目配置)
        self.latent_multiplier = 1.0  # VA-VAE使用1.0而不是0.18215
        self.use_latent_norm = True

        # 加载潜在特征统计信息 (如果可用)
        self.latent_mean = None
        self.latent_std = None
        # 注意：在实际使用中应该从latents_stats.pt加载这些统计信息

        # 加载DiT模型 (参考原项目)
        if not self.is_distributed or self.accelerator.is_main_process:
            print("📥 加载DiT模型...")
        self._load_dit_model(dit_checkpoint)
        
        # 创建transport (参考原项目)
        self.transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=1e-5,
            sample_eps=1e-3
        )

        # 创建采样器 (使用原项目的正确参数)
        from transport import Sampler
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method="euler",  # 原项目使用euler
            num_steps=250,
            atol=0.000001,           # 原项目配置
            rtol=0.001,              # 原项目配置
            timestep_shift=0.3,      # 原项目配置
        )
        
        print("✅ 模型加载完成!")

    def _get_num_classes_from_checkpoint(self, checkpoint_path):
        """从检查点或数据中获取正确的类别数"""
        try:
            # 方法1: 尝试从训练数据中获取用户数
            latent_dir = os.path.dirname(checkpoint_path).replace('trained_models', 'latent_features')
            train_file = os.path.join(latent_dir, 'train.safetensors')

            if os.path.exists(train_file):
                from safetensors import safe_open
                with safe_open(train_file, framework="pt", device="cpu") as f:
                    num_users = f.get_tensor('num_users').item()
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print(f"📊 从训练数据获取用户数: {num_users}")
                    return num_users

            # 方法2: 如果找不到训练数据，使用默认值
            if not self.is_distributed or self.accelerator.is_main_process:
                print("⚠️  无法从训练数据获取用户数，使用默认值31")
            return 31

        except Exception as e:
            if not self.is_distributed or self.accelerator.is_main_process:
                print(f"⚠️  获取类别数失败: {e}，使用默认值31")
            return 31

    def _load_dit_model(self, checkpoint_path):
        """加载DiT模型检查点"""
        # 这里需要根据实际的检查点格式来调整
        # 参考原项目的模型加载方式
        
        # 首先尝试从检查点中获取正确的类别数
        num_classes = self._get_num_classes_from_checkpoint(checkpoint_path)

        # 使用与训练时一致的模型配置
        self.dit_model = LightningDiT_models[self.model_name](  # 使用指定的模型
            input_size=16,  # 256/16=16 (downsample_ratio=16)
            num_classes=num_classes,  # 从检查点或数据中获取的正确类别数
            in_channels=32,  # VA-VAE使用32通道
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            wo_shift=False
        )
        
        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            if not self.is_distributed or self.accelerator.is_main_process:
                print(f"📥 从 {checkpoint_path} 加载模型权重")

            # 检查是否是Accelerate保存的检查点目录
            # 优先检查safetensors格式
            safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")

            if os.path.exists(safetensors_path):
                # Accelerate保存的safetensors格式
                if not self.is_distributed or self.accelerator.is_main_process:
                    print(f"🔍 发现Accelerate检查点 (safetensors): {safetensors_path}")
                try:
                    from safetensors.torch import load_file
                    checkpoint = load_file(safetensors_path)
                    self.dit_model.load_state_dict(checkpoint)
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print("✅ 成功加载Accelerate检查点 (safetensors)")
                except Exception as e:
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print(f"❌ 加载safetensors检查点失败: {e}")
                        print("⚠️  使用随机初始化的模型")
            elif os.path.exists(pytorch_model_path):
                # Accelerate保存的pytorch_model.bin格式
                if not self.is_distributed or self.accelerator.is_main_process:
                    print(f"🔍 发现Accelerate检查点 (pytorch): {pytorch_model_path}")
                try:
                    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
                    self.dit_model.load_state_dict(checkpoint)
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print("✅ 成功加载Accelerate检查点 (pytorch)")
                except Exception as e:
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print(f"❌ 加载pytorch检查点失败: {e}")
                        print("⚠️  使用随机初始化的模型")
            else:
                # 尝试直接加载文件
                try:
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print(f"🔍 尝试直接加载: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')

                    # 处理不同的检查点格式
                    if 'model' in checkpoint:
                        self.dit_model.load_state_dict(checkpoint['model'])
                    elif 'state_dict' in checkpoint:
                        self.dit_model.load_state_dict(checkpoint['state_dict'])
                    elif 'ema' in checkpoint:
                        self.dit_model.load_state_dict(checkpoint['ema'])
                    else:
                        self.dit_model.load_state_dict(checkpoint)

                    if not self.is_distributed or self.accelerator.is_main_process:
                        print("✅ 成功加载检查点")
                except Exception as e:
                    if not self.is_distributed or self.accelerator.is_main_process:
                        print(f"❌ 加载检查点失败: {e}")
                        print("⚠️  使用随机初始化的模型")
        else:
            if not self.is_distributed or self.accelerator.is_main_process:
                if checkpoint_path:
                    print(f"⚠️  检查点不存在: {checkpoint_path}")
                else:
                    print("⚠️  未指定检查点路径")
                print("⚠️  使用随机初始化的模型")
        
        self.dit_model.eval()
        self.dit_model.to(self.device)
    
    def generate_samples(self, user_ids, num_samples_per_user=4, guidance_scale=4.0, num_steps=250):
        """
        生成用户条件化的微多普勒图像
        参考原项目的采样方法
        """
        if not self.is_distributed or self.accelerator.is_main_process:
            print(f"🎨 开始生成图像...")
            print(f"  用户ID: {user_ids}")
            print(f"  每用户样本数: {num_samples_per_user}")
            print(f"  引导尺度: {guidance_scale}")
            print(f"  采样步数: {num_steps}")
            if self.is_distributed:
                print(f"  分布式推理: {self.accelerator.num_processes} GPU")

        all_images = []
        all_user_labels = []

        # 计算分布式任务分配
        total_samples = len(user_ids) * num_samples_per_user
        if self.is_distributed:
            samples_per_gpu = total_samples // self.accelerator.num_processes
            start_idx = self.accelerator.process_index * samples_per_gpu
            end_idx = start_idx + samples_per_gpu
            if self.accelerator.process_index == self.accelerator.num_processes - 1:
                end_idx = total_samples
        else:
            start_idx = 0
            end_idx = total_samples

        with torch.no_grad():
            sample_idx = 0
            for user_id in user_ids:
                for sample_num in range(num_samples_per_user):
                    # 检查是否是当前GPU负责的样本
                    if self.is_distributed and (sample_idx < start_idx or sample_idx >= end_idx):
                        sample_idx += 1
                        continue

                    # 准备条件 (参考原项目)
                    batch_size = 1  # 每次生成一个样本
                    y = torch.full((batch_size,), user_id - 1, dtype=torch.long, device=self.device)  # 0-based

                    sample_idx += 1
                
                # 生成随机噪声 (参考原项目)
                z = torch.randn(batch_size, 32, 16, 16, device=self.device)

                # 设置分类器自由引导 (参考原项目第206-214行)
                if guidance_scale > 1.0:
                    z = torch.cat([z, z], 0)  # 复制噪声
                    # null class应该是num_classes (超出有效类别范围)
                    null_class = self.dit_model.y_embedder.num_classes
                    y_null = torch.tensor([null_class] * batch_size, device=self.device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(
                        y=y,
                        cfg_scale=guidance_scale,
                        cfg_interval=True,
                        cfg_interval_start=0.125  # 原项目配置
                    )
                    model_fn = self.dit_model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = self.dit_model.forward

                # 使用正确的采样方法
                samples = self._sample_with_transport(z, model_fn, model_kwargs, num_steps)

                # 如果使用CFG，移除null class样本 (参考原项目第217-218行)
                if guidance_scale > 1.0:
                    samples, _ = samples.chunk(2, dim=0)

                # 应用潜在特征反归一化 (参考原项目第220行)
                # samples = (samples * latent_std) / latent_multiplier + latent_mean
                if self.use_latent_norm:
                    # 由于我们没有latent_stats.pt，使用简化版本
                    # 原项目: samples = (samples * latent_std) / latent_multiplier + latent_mean
                    samples = samples / self.latent_multiplier  # 简化版本

                if not self.is_distributed or self.accelerator.is_main_process:
                    print(f"🔍 反归一化后样本范围: [{samples.min():.3f}, {samples.max():.3f}]")

                # 使用VA-VAE解码为图像
                images = self.vavae.decode_to_images(samples)

                # 后处理图像
                images = self._postprocess_images(images)
                
                all_images.extend(images)
                all_user_labels.extend([user_id] * num_samples_per_user)
        
        return all_images, all_user_labels
    
    def _sample_with_transport(self, z, model_fn, model_kwargs, num_steps):
        """
        使用transport进行采样 - 正确版本
        """
        # 使用正确的ODE采样器
        try:
            if not self.is_distributed or self.accelerator.is_main_process:
                print(f"🎯 使用ODE采样器，步数: {num_steps}")

            # 使用预配置的采样函数 (参考原项目第216行)
            samples = self.sample_fn(z, model_fn, **model_kwargs)

            # 返回最后一个时间步的结果并修复维度
            if isinstance(samples, list):
                result = samples[-1]
            else:
                result = samples

            # 修复维度问题：从[num_steps, batch, channels, h, w]到[batch, channels, h, w]
            if result.dim() == 5:
                result = result[-1]  # 取最后一个时间步
            elif result.dim() > 4:
                # 如果还有其他维度问题，强制reshape
                while result.dim() > 4:
                    result = result[-1]

            if not self.is_distributed or self.accelerator.is_main_process:
                print(f"🔍 采样结果形状: {result.shape}")

            return result

        except Exception as e:
            if not self.is_distributed or self.accelerator.is_main_process:
                print(f"⚠️  ODE采样失败: {e}")
                print("⚠️  使用简化采样方法")

            # 改进的简化采样过程
            dt = 1.0 / num_steps
            x = z.clone()

            for i in range(num_steps):
                # 使用正确的时间步 (从0到1)
                t = torch.full((x.shape[0],), i / num_steps, device=self.device)

                # 模型预测 (velocity prediction)
                with torch.no_grad():
                    velocity = model_fn(x, t, **model_kwargs)

                # 使用velocity进行更新 (Euler方法)
                x = x + velocity * dt

            return x
    
    def _postprocess_images(self, images):
        """
        后处理图像 - 改进版本
        """
        if not self.is_distributed or self.accelerator.is_main_process:
            print(f"🔍 图像后处理调试:")
            print(f"  输入形状: {images.shape}")
            print(f"  数据类型: {images.dtype}")
            print(f"  值范围: [{images.min()}, {images.max()}]")

        # decode_to_images已经返回了uint8格式的numpy数组 (B, H, W, C)
        pil_images = []
        for i, img_np in enumerate(images):
            # 确保数据类型正确
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # 确保形状正确 (H, W, C)
            if img_np.ndim == 3 and img_np.shape[-1] in [1, 3]:
                # 如果是单通道，转换为RGB
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)

                # 创建PIL图像
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
            else:
                print(f"⚠️  跳过异常形状的图像 {i}: {img_np.shape}")

        return pil_images
    
    def save_images(self, images, user_labels, output_dir, prefix="micro_doppler"):
        """保存生成的图像"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存图像到: {output_dir}")
        
        for i, (image, user_id) in enumerate(zip(images, user_labels)):
            filename = f"{prefix}_user{user_id:02d}_{i+1:03d}.png"
            filepath = output_dir / filename
            image.save(filepath)
        
        # 创建网格图像 (参考原项目)
        self._create_grid_image(images, user_labels, output_dir, prefix)
        
        print(f"✅ 保存了 {len(images)} 张图像")
    
    def _create_grid_image(self, images, user_labels, output_dir, prefix):
        """创建网格展示图像"""
        if not images:
            return
        
        # 计算网格尺寸
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # 获取单张图像尺寸
        img_width, img_height = images[0].size
        
        # 创建网格图像
        grid_width = grid_size * img_width
        grid_height = grid_size * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # 填充网格
        for i, image in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            x = col * img_width
            y = row * img_height
            grid_image.paste(image, (x, y))
        
        # 保存网格图像
        grid_path = output_dir / f"{prefix}_grid.png"
        grid_image.save(grid_path)
        print(f"📊 网格图像保存到: {grid_path}")

def main(accelerator=None):
    parser = argparse.ArgumentParser(description='微多普勒图像生成')
    parser.add_argument('--dit_checkpoint', type=str, help='DiT模型检查点 (如果不存在将使用随机模型)')
    parser.add_argument('--vavae_config', type=str, help='VA-VAE配置文件')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--user_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='用户ID列表')
    parser.add_argument('--num_samples_per_user', type=int, default=4, help='每用户生成样本数')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='引导尺度')
    parser.add_argument('--num_steps', type=int, default=250, help='采样步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dual_gpu', action='store_true', help='使用双GPU推理')
    parser.add_argument('--test_imports', action='store_true', help='仅测试导入，不进行推理')
    parser.add_argument('--model_name', type=str, default='LightningDiT-B/1', help='DiT模型名称 (应与训练时一致)')

    args = parser.parse_args()

    # 如果只是测试导入
    if args.test_imports:
        success = test_imports()
        if success:
            print("✅ 所有导入测试通过!")
        else:
            print("❌ 导入测试失败")
        return

    # 对于推理，检查必需参数
    if not args.vavae_config:
        parser.error("推理模式需要 --vavae_config 参数")
    if not args.output_dir:
        parser.error("推理模式需要 --output_dir 参数")

    # 设置随机种子
    if accelerator:
        seed = args.seed * accelerator.num_processes + accelerator.process_index
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not accelerator or accelerator.is_main_process:
        print("🎯 微多普勒图像生成")
        print("=" * 50)
        if accelerator:
            print(f"🔧 分布式推理: {accelerator.num_processes} GPU")
    
    # 创建生成器
    generator = MicroDopplerGenerator(
        dit_checkpoint=args.dit_checkpoint,
        vavae_config=args.vavae_config,
        model_name=args.model_name,
        accelerator=accelerator
    )
    
    # 生成图像
    images, user_labels = generator.generate_samples(
        user_ids=args.user_ids,
        num_samples_per_user=args.num_samples_per_user,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps
    )
    
    # 保存图像
    generator.save_images(images, user_labels, args.output_dir)
    
    print("✅ 图像生成完成!")

def dual_gpu_inference():
    """双GPU推理包装函数"""
    def inference_worker():
        from accelerate import Accelerator
        accelerator = Accelerator()
        main(accelerator)

    from accelerate import notebook_launcher
    print("🚀 启动双GPU推理...")
    notebook_launcher(inference_worker, num_processes=2)
    print("✅ 双GPU推理完成")

if __name__ == "__main__":
    import sys

    # 检查是否使用双GPU
    if '--dual_gpu' in sys.argv:
        sys.argv.remove('--dual_gpu')  # 移除这个参数，避免argparse报错
        dual_gpu_inference()
    else:
        main()
