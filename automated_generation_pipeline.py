#!/usr/bin/env python3
"""
自动化条件生成-筛选管道
结合条件扩散生成和质量筛选，自动保存到目标数量
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from collections import defaultdict
import logging

# 添加路径以导入必要模块
sys.path.append(str(Path(__file__).parent / "LightningDiT"))
sys.path.append(str(Path(__file__).parent))

# 导入生成相关模块 - 使用现有的生成脚本中的函数
import yaml
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from improved_classifier_training import ImprovedClassifier

# 导入LightningDiT相关模块
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from simplified_vavae import SimplifiedVAVAE

# 注意：VA_VAE会在load_models中动态导入，这里不需要提前导入

import tempfile

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

class AutomatedGenerationPipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.setup_logging()
        self.setup_directories()
        self.load_models()
        self.setup_transforms()
        self.user_progress = defaultdict(int)  # 跟踪每个用户已保存的样本数
        
    def setup_logging(self):
        """设置日志"""
        # 确保输出目录存在
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(self.args.output_dir) / 'generation_log.txt'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """创建必要的目录结构"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 为每个用户创建输出目录
        self.user_dirs = {}
        for user_id in range(self.args.num_users):
            user_dir = self.output_dir / f"ID_{user_id:02d}"
            user_dir.mkdir(exist_ok=True)
            self.user_dirs[user_id] = user_dir
            
        self.logger.info(f"设置输出目录: {self.output_dir}")
        
    def load_models(self):
        """加载扩散模型和分类器"""
        # 加载配置
        with open(self.args.config, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 导入必要的模块
        sys.path.append('LightningDiT')
        from models.lightningdit import LightningDiT_models
        from transport import create_transport, Sampler
        from simplified_vavae import SimplifiedVAVAE
        
        # 加载扩散模型
        self.logger.info("加载扩散模型...")
        latent_size = self.config['data']['image_size'] // self.config['vae']['downsample_ratio']
        self.model = LightningDiT_models[self.config['model']['model_type']](
            input_size=latent_size,
            num_classes=self.config['data']['num_classes'],
            class_dropout_prob=self.config['model'].get('class_dropout_prob', 0.1),
            use_qknorm=self.config['model']['use_qknorm'],
            use_swiglu=self.config['model'].get('use_swiglu', False),
            use_rope=self.config['model'].get('use_rope', False),
            use_rmsnorm=self.config['model'].get('use_rmsnorm', False),
            wo_shift=self.config['model'].get('wo_shift', False),
            in_channels=self.config['model'].get('in_chans', 4),
            use_checkpoint=self.config['model'].get('use_checkpoint', False),
        ).to(self.device)
        
        # 加载checkpoint
        if self.args.checkpoint and os.path.exists(self.args.checkpoint):
            self.logger.info(f"加载checkpoint: {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
            
            # 处理权重键名
            if 'ema' in checkpoint:
                checkpoint_weights = {'model': checkpoint['ema']}
                self.logger.info("使用EMA权重进行推理")
            elif 'model' in checkpoint:
                checkpoint_weights = checkpoint
                self.logger.info("使用模型权重进行推理")
            else:
                checkpoint_weights = {'model': checkpoint}
                
            # 清理键名
            checkpoint_weights['model'] = {k.replace('module.', ''): v for k, v in checkpoint_weights['model'].items()}
            
            # 加载权重
            self.load_weights_with_shape_check(self.model, checkpoint_weights)
            
        self.model.eval()
        
        # 初始化transport和sampler
        self.transport = create_transport(
            self.config['transport']['path_type'],
            self.config['transport']['prediction'],
            self.config['transport']['loss_weight'],
            self.config['transport']['train_eps'],
            self.config['transport']['sample_eps'],
            use_cosine_loss=self.config['transport'].get('use_cosine_loss', False),
            use_lognorm=self.config['transport'].get('use_lognorm', False),
            partitial_train=self.config['transport'].get('partitial_train', None),
            partial_ratio=self.config['transport'].get('partial_ratio', 1.0),
            shift_lg=self.config['transport'].get('shift_lg', False),
        )
        self.sampler = Sampler(self.transport)
        
        # 加载VAE（完全按照generate_conditional_samples_distributed.py方式）
        self.vae = None
        try:
            # 添加LightningDiT路径到系统路径
            lightningdit_path = os.path.join(os.getcwd(), 'LightningDiT')
            if lightningdit_path not in sys.path:
                sys.path.insert(0, lightningdit_path)
            
            from tokenizer.vavae import VA_VAE
            
            # 使用训练好的VAE模型路径
            custom_vae_checkpoint = "/kaggle/input/stage3/vavae-stage3-epoch26-val_rec_loss0.0000.ckpt"
            
            # 创建与train_dit_s_official.py完全一致的配置
            vae_config = {
                'ckpt_path': custom_vae_checkpoint,
                'model': {
                    'base_learning_rate': 2.0e-05,
                    'target': 'ldm.models.autoencoder.AutoencoderKL',
                    'params': {
                        'monitor': 'val/rec_loss',
                        'embed_dim': 32,
                        'use_vf': 'dinov2',
                        'reverse_proj': True,
                        'ddconfig': {
                            'double_z': True, 'z_channels': 32, 'resolution': 256,
                            'in_channels': 3, 'out_ch': 3, 'ch': 128,
                            'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2,
                            'attn_resolutions': [16], 'dropout': 0.0
                        },
                        'lossconfig': {
                            'target': 'ldm.modules.losses.contperceptual.LPIPSWithDiscriminator',
                            'params': {
                                'disc_start': 1, 'disc_num_layers': 3, 'disc_weight': 0.5,
                                'disc_factor': 1.0, 'disc_in_channels': 3, 'disc_conditional': False,
                                'disc_loss': 'hinge', 'pixelloss_weight': 1.0, 'perceptual_weight': 1.0,
                                'kl_weight': 1e-6, 'logvar_init': 0.0, 'use_actnorm': False,
                                'pp_style': False, 'vf_weight': 0.1, 'adaptive_vf': False,
                                'distmat_weight': 1.0, 'cos_weight': 1.0,
                                'distmat_margin': 0.25, 'cos_margin': 0.5
                            }
                        }
                    }
                }
            }
            
            # 写入临时配置文件
            temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
            with open(temp_config_path, 'w') as f:
                yaml.dump(vae_config, f, default_flow_style=False)
            os.close(temp_config_fd)
            
            try:
                # 使用官方VA_VAE类加载
                self.vae = VA_VAE(temp_config_path)
                # 检查是否有.to()方法（与官方train_dit_s_official.py一致）
                if hasattr(self.vae, 'to'):
                    self.vae = self.vae.to(self.device)
                if hasattr(self.vae, 'eval'):
                    self.vae.eval()
                self.logger.info(f"✅ VAE加载完成: {custom_vae_checkpoint}")
                print(f"✅ VAE加载成功: 使用VA-VAE {custom_vae_checkpoint}")
            finally:
                # 清理临时文件
                os.unlink(temp_config_path)
                
        except Exception as e:
            self.logger.warning(f"⚠️ VAE加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.logger.warning("⚠️ 尝试使用简化VAE作为备用")
            # 备用方案
            try:
                self.vae = SimplifiedVAVAE(self.config['vae']['model_name']).to(self.device)
                self.vae.eval()
                self.logger.info(f"✅ 备用VAE加载完成: {self.config['vae']['model_name']}")
            except Exception as e2:
                self.logger.error(f"⚠️ 备用VAE也加载失败: {e2}")
                self.vae = None
        
        # 加载latent统计信息
        self.latent_stats = None
        latent_stats_path = 'latents_safetensors/train/latent_stats.pt'
        if os.path.exists(latent_stats_path):
            self.latent_stats = torch.load(latent_stats_path, map_location='cpu')
            print(f"✅ 已加载latent统计信息: {latent_stats_path}")
        else:
            print(f"⚠️ 未找到latent统计文件: {latent_stats_path}")
        
        # 加载分类器
        self.logger.info("加载分类器...")
        self.classifier = ImprovedClassifier(num_classes=self.args.num_users)
        classifier_checkpoint = torch.load(self.args.classifier_path, map_location='cpu')
        
        # 处理checkpoint格式
        if 'model_state_dict' in classifier_checkpoint:
            state_dict = classifier_checkpoint['model_state_dict']
        else:
            state_dict = classifier_checkpoint
            
        self.classifier.load_state_dict(state_dict)
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
    def load_weights_with_shape_check(self, model, checkpoint):
        """使用形状检查加载权重"""
        model_state_dict = model.state_dict()
        for name, param in checkpoint['model'].items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)
                elif name == 'x_embedder.proj.weight':
                    weight = torch.zeros_like(model_state_dict[name])
                    weight[:, :16] = param[:, :16]
                    model_state_dict[name] = weight
                else:
                    self.logger.warning(f"跳过参数 '{name}' 形状不匹配: "
                                      f"checkpoint {param.shape}, model {model_state_dict[name].shape}")
            else:
                self.logger.warning(f"参数 '{name}' 在模型中未找到，跳过")
        model.load_state_dict(model_state_dict, strict=False)
        
    def setup_transforms(self):
        """设置图像预处理"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def generate_batch(self, user_ids, batch_size):
        """生成一个批次的样本"""
        with torch.no_grad():
            current_batch_size = len(user_ids)
            
            # 创建条件标签
            y = torch.tensor(user_ids, device=self.device, dtype=torch.long)
            
            # 创建随机噪声 (VA-VAE使用32通道，16x16空间分辨率)
            z = torch.randn(current_batch_size, 32, 16, 16, device=self.device)
            
            # 创建采样函数 (完全按照generate_conditional_samples_distributed.py的实现)
            sample_fn = self.sampler.sample_ode(
                sampling_method="dopri5",
                num_steps=300,
                atol=1e-6,
                rtol=1e-3,
                reverse=False,
                timestep_shift=0.1
            )
            
            # CFG采样 - 完全按照generate_conditional_samples_distributed.py实现
            if self.args.cfg_scale > 1.0:
                # 构建CFG batch
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31] * current_batch_size, device=self.device)  # null class
                y_cfg = torch.cat([y, y_null], 0)
                
                # 使用官方CFG配置
                cfg_interval_start = 0.11
                model_kwargs = dict(y=y_cfg, cfg_scale=self.args.cfg_scale, 
                                  cfg_interval=True, cfg_interval_start=cfg_interval_start)
                
                # 使用CFG前向传播（与官方完全一致）
                if hasattr(self.model, 'forward_with_cfg'):
                    samples = sample_fn(z_cfg, self.model.forward_with_cfg, **model_kwargs)
                else:
                    # 如果模型没有forward_with_cfg方法，使用手动CFG
                    def model_fn_cfg(x, t, **kwargs):
                        pred = self.model(x, t, **kwargs)
                        pred_cond, pred_uncond = pred.chunk(2, dim=0)
                        return pred_uncond + self.args.cfg_scale * (pred_cond - pred_uncond)
                    samples = sample_fn(z_cfg, model_fn_cfg, **model_kwargs)
                
                samples = samples[-1]  # 获取最终时间步的样本
                samples, _ = samples.chunk(2, dim=0)  # 去掉null class样本
            else:
                # 标准采样
                samples = sample_fn(z, self.model, **dict(y=y))
                samples = samples[-1]
            
            # 反归一化处理 (完全按照generate_conditional_samples_distributed.py实现)
            if self.latent_stats is not None:
                mean = self.latent_stats['mean'].to(self.device)
                std = self.latent_stats['std'].to(self.device)
                latent_multiplier = 1.0  # VA-VAE使用1.0，不是0.18215
                
                # 官方反归一化公式（与train_dit_s_official.py第549行完全一致）
                samples_denorm = (samples * std) / latent_multiplier + mean
            else:
                print("⚠️ 无latent统计信息，跳过反归一化")
                samples_denorm = samples
            
            # VAE解码 (使用VA-VAE解码latent为图像)
            decoded_images = self.vae.decode_to_images(samples_denorm)
            
            return decoded_images
            
    def evaluate_samples(self, samples, expected_user_ids):
        """评估生成样本的质量"""
        batch_results = []
        
        with torch.no_grad():
            # 转换为PIL图像并预处理
            processed_samples = []
            pil_images = []
            
            for sample in samples:
                # VAE输出的图像已经在0-1范围内，转换为0-255
                if isinstance(sample, torch.Tensor):
                    # 转换为numpy数组
                    if sample.dim() == 3:  # CHW格式
                        sample_np = sample.permute(1, 2, 0).cpu().numpy()
                    else:  # HWC格式
                        sample_np = sample.cpu().numpy()
                else:
                    sample_np = sample
                
                # 确保在0-1范围内，然后转换为0-255
                sample_np = np.clip(sample_np, 0, 1)
                sample_uint8 = (sample_np * 255).astype(np.uint8)
                
                # 处理灰度或单通道图像
                if len(sample_uint8.shape) == 2:
                    sample_uint8 = np.stack([sample_uint8] * 3, axis=2)
                elif sample_uint8.shape[2] == 1:
                    sample_uint8 = np.repeat(sample_uint8, 3, axis=2)
                
                # 转换为PIL图像
                pil_image = Image.fromarray(sample_uint8)
                pil_images.append(pil_image)
                
                # 应用分类器预处理
                tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
                processed_samples.append(tensor_image)
            
            # 批量处理
            if processed_samples:
                batch_tensor = torch.cat(processed_samples, dim=0)
                logits = self.classifier(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(logits, dim=1)
                max_probabilities = torch.max(probabilities, dim=1)[0]
                
                # 评估每个样本
                for i, (pred_class, confidence, expected_id) in enumerate(zip(
                    predicted_classes.cpu().numpy(),
                    max_probabilities.cpu().numpy(), 
                    expected_user_ids
                )):
                    is_correct = (pred_class == expected_id)
                    is_high_confidence = (confidence >= self.args.confidence_threshold)
                    
                    result = {
                        'sample_idx': i,
                        'expected_id': expected_id,
                        'predicted_id': pred_class,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'is_high_confidence': is_high_confidence,
                        'accept': is_correct and is_high_confidence,
                        'pil_image': pil_images[i]
                    }
                    batch_results.append(result)
                    
        return batch_results
        
    def save_accepted_samples(self, batch_results):
        """保存通过筛选的样本"""
        saved_count = 0
        
        for result in batch_results:
            if result['accept']:
                user_id = result['expected_id']
                
                # 检查是否已达到目标数量
                if self.user_progress[user_id] >= self.args.target_per_user:
                    continue
                    
                # 保存样本
                filename = f"generated_{self.user_progress[user_id]:04d}_conf_{result['confidence']:.3f}.png"
                save_path = self.user_dirs[user_id] / filename
                
                # 确保是RGB格式
                pil_image = result['pil_image']
                if pil_image.mode != 'RGB':
                    if pil_image.mode == 'L':
                        pil_image = pil_image.convert('RGB')
                        
                pil_image.save(save_path)
                self.user_progress[user_id] += 1
                saved_count += 1
                
                # 记录保存信息
                self.logger.info(
                    f"保存样本: User_{user_id:02d} -> {filename} "
                    f"(置信度: {result['confidence']:.3f}, "
                    f"进度: {self.user_progress[user_id]}/{self.args.target_per_user})"
                )
                
        return saved_count
        
    def check_completion(self):
        """检查是否所有用户都达到目标数量"""
        completed_users = sum(1 for count in self.user_progress.values() 
                            if count >= self.args.target_per_user)
        
        return completed_users >= self.args.num_users
        
    def print_progress(self):
        """打印当前进度"""
        print(f"\n{'='*60}")
        print(f"📊 生成进度统计")
        print(f"{'='*60}")
        
        for user_id in range(self.args.num_users):
            progress = self.user_progress[user_id]
            percentage = (progress / self.args.target_per_user) * 100
            bar_length = 20
            filled_length = int(bar_length * progress / self.args.target_per_user)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"User_{user_id:02d}: [{bar}] {progress:3d}/{self.args.target_per_user} ({percentage:5.1f}%)")
            
        total_saved = sum(self.user_progress.values())
        total_target = self.args.num_users * self.args.target_per_user
        print(f"\n🎯 总进度: {total_saved}/{total_target} ({total_saved/total_target*100:.1f}%)")
        
    def run(self):
        """运行自动化管道"""
        self.logger.info("🚀 启动自动化生成-筛选管道")
        
        total_generated = 0
        total_accepted = 0
        batch_count = 0
        
        try:
            while not self.check_completion():
                batch_count += 1
                
                # 创建当前批次的用户ID列表
                current_batch_ids = []
                for user_id in range(self.args.num_users):
                    if self.user_progress[user_id] < self.args.target_per_user:
                        # 根据当前进度决定该用户在批次中的样本数
                        remaining = self.args.target_per_user - self.user_progress[user_id]
                        batch_size_for_user = min(self.args.batch_size // self.args.num_users + 1, 
                                                remaining * 2)  # 生成2倍数量以提高筛选效率
                        current_batch_ids.extend([user_id] * batch_size_for_user)
                
                if not current_batch_ids:
                    break
                    
                # 限制批次大小
                if len(current_batch_ids) > self.args.batch_size:
                    current_batch_ids = current_batch_ids[:self.args.batch_size]
                
                self.logger.info(f"批次 {batch_count}: 生成 {len(current_batch_ids)} 个样本...")
                
                # 生成样本
                samples = self.generate_batch(current_batch_ids, len(current_batch_ids))
                total_generated += len(samples)
                
                # 评估样本
                results = self.evaluate_samples(samples, current_batch_ids)
                
                # 保存通过筛选的样本
                saved_count = self.save_accepted_samples(results)
                total_accepted += saved_count
                
                # 计算批次统计
                batch_accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                batch_acceptance = saved_count / len(results)
                
                self.logger.info(
                    f"批次 {batch_count} 完成: "
                    f"准确率 {batch_accuracy:.1%}, "
                    f"接受率 {batch_acceptance:.1%}, "
                    f"保存 {saved_count} 个样本"
                )
                
                # 每10个批次打印一次进度
                if batch_count % 10 == 0:
                    self.print_progress()
                    
        except KeyboardInterrupt:
            self.logger.info("用户中断生成过程")
            
        # 最终统计
        self.print_progress()
        self.logger.info(f"🎉 生成完成!")
        self.logger.info(f"📊 总统计: 生成 {total_generated} 个样本, 接受 {total_accepted} 个样本")
        self.logger.info(f"📊 总体接受率: {total_accepted/total_generated:.1%}")
        
        # 保存统计信息
        stats = {
            'total_generated': total_generated,
            'total_accepted': total_accepted,
            'acceptance_rate': total_accepted / total_generated,
            'user_progress': dict(self.user_progress),
            'batch_count': batch_count
        }
        
        stats_file = self.output_dir / 'generation_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info(f"📄 统计信息保存到: {stats_file}")


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def main():
    # 初始化分布式环境
    rank, local_rank, world_size = setup_distributed()
    
    parser = argparse.ArgumentParser(description='自动化条件生成-筛选管道')
    
    # 扩散模型参数
    parser.add_argument('--checkpoint', required=True, help='扩散模型checkpoint路径')
    parser.add_argument('--config', required=True, help='模型配置文件路径')
    parser.add_argument('--cfg_scale', type=float, default=10.0, help='CFG scale')
    
    # 分类器参数  
    parser.add_argument('--classifier_path', required=True, help='分类器模型路径')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, 
                       help='置信度阈值')
    
    # 生成参数
    parser.add_argument('--num_users', type=int, default=31, help='用户数量')
    parser.add_argument('--target_per_user', type=int, default=300, 
                       help='每个用户目标样本数量')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    
    # 输出参数
    parser.add_argument('--output_dir', default='./automated_samples', 
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子（每个进程使用不同的种子）
    set_seed(args.seed + rank * 1000)
    
    # 在所有进程上创建和运行管道
    if rank == 0 and world_size > 1:
        print(f"🚀 使用 {world_size} 个GPU进行分布式生成")
    
    # 创建并运行管道（所有进程都运行）
    pipeline = AutomatedGenerationPipeline(args)
    pipeline.rank = rank
    pipeline.local_rank = local_rank 
    pipeline.world_size = world_size
    
    # 设置正确的设备
    if world_size > 1:
        pipeline.device = torch.device(f'cuda:{local_rank}')
    
    pipeline.run()
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
