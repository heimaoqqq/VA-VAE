"""
生成并筛选高质量样本脚本
使用DiT生成样本，通过预训练分类器筛选置信度>0.95的正确样本
每个用户ID持续生成直到收集到800个高质量样本
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.models import resnet50

# 添加LightningDiT路径
sys.path.append('LightningDiT')
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from simplified_vavae import SimplifiedVAVAE

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

def load_weights_with_shape_check(model, checkpoint, rank=0):
    """使用形状检查加载权重（完全按照官方实现）"""
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

def load_model_and_config(checkpoint_path, config_path, local_rank):
    """加载模型和配置（按照官方方式）"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    device = torch.device(f'cuda:{local_rank}')
    
    # 创建DiT模型
    latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']
    model = LightningDiT_models[config['model']['model_type']](
        input_size=latent_size,
        num_classes=config['data']['num_classes'],
        class_dropout_prob=config['model'].get('class_dropout_prob', 0.1),
        use_qknorm=config['model']['use_qknorm'],
        use_swiglu=config['model'].get('use_swiglu', False),
        use_rope=config['model'].get('use_rope', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False),
        wo_shift=config['model'].get('wo_shift', False),
        in_channels=config['model'].get('in_chans', 4),
        use_checkpoint=config['model'].get('use_checkpoint', False),
    ).to(device)
    
    # 按照官方方式加载权重
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"📦 从checkpoint加载权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # 处理权重键名（按照官方方式）
        if 'ema' in checkpoint:
            checkpoint_weights = {'model': checkpoint['ema']}
            if local_rank == 0:
                print("📦 使用EMA权重进行推理")
        elif 'model' in checkpoint:
            checkpoint_weights = checkpoint
            if local_rank == 0:
                print("📦 使用模型权重进行推理")
        else:
            checkpoint_weights = {'model': checkpoint}
            if local_rank == 0:
                print("📦 使用直接权重进行推理")
        
        # 清理键名
        checkpoint_weights['model'] = {k.replace('module.', ''): v for k, v in checkpoint_weights['model'].items()}
        
        # 使用官方权重加载函数
        model = load_weights_with_shape_check(model, checkpoint_weights, rank=local_rank)
        
        if local_rank == 0:
            print("✅ 权重加载完成")
    else:
        if local_rank == 0:
            print(f"⚠️ 警告: 找不到checkpoint文件 {checkpoint_path}")
            print("⚠️ 使用未训练的随机权重，生成结果将是噪声！")
    
    model.eval()
    
    # 创建VAE（完全按照官方train_dit_s_official.py方式）
    vae = None
    try:
        # 添加LightningDiT路径到系统路径
        import sys
        lightningdit_path = os.path.join(os.getcwd(), 'LightningDiT')
        if lightningdit_path not in sys.path:
            sys.path.insert(0, lightningdit_path)
        
        from tokenizer.vavae import VA_VAE
        import tempfile
        
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
            vae = VA_VAE(temp_config_path)
            # 检查是否有.to()方法
            if hasattr(vae, 'to'):
                vae = vae.to(device)
            if hasattr(vae, 'eval'):
                vae.eval()
            if local_rank == 0:
                print(f"✅ VAE加载完成: {custom_vae_checkpoint}")
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
            
    except Exception as e:
        if local_rank == 0:
            print(f"⚠️ VAE加载失败: {e}")
            print("⚠️ 尝试使用简化VAE作为备用")
        # 备用方案
        try:
            vae = SimplifiedVAVAE(config['vae']['model_name']).to(device)
            vae.eval()
            if local_rank == 0:
                print(f"✅ 备用VAE加载完成: {config['vae']['model_name']}")
        except Exception as e2:
            if local_rank == 0:
                print(f"⚠️ 备用VAE也加载失败: {e2}")
            vae = None
    
    # 创建transport
    transport = create_transport(
        config['transport']['path_type'],
        config['transport']['prediction'],
        config['transport']['loss_weight'],
        config['transport']['train_eps'],
        config['transport']['sample_eps'],
        use_cosine_loss=config['transport'].get('use_cosine_loss', False),
        use_lognorm=config['transport'].get('use_lognorm', False),
        partitial_train=config['transport'].get('partitial_train', None),
        partial_ratio=config['transport'].get('partial_ratio', 1.0),
        shift_lg=config['transport'].get('shift_lg', False),
    )
    
    return model, vae, transport, config, device

def load_classifier(checkpoint_path, device):
    """加载预训练的分类器"""
    import torchvision.models as models
    
    # 创建与improved_classifier_training.py完全一致的模型结构
    class MicroDopplerModel(nn.Module):
        def __init__(self, num_classes=31, dropout_rate=0.3):
            super().__init__()
            
            # 使用ResNet18作为backbone
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            # 对比学习投影头
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 64)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    # 创建模型
    model = MicroDopplerModel(num_classes=31)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 分类器加载完成: {checkpoint_path}")
    print(f"📊 Epoch: {checkpoint['epoch']}, Best Acc: {checkpoint['best_acc']:.2f}%")
    
    return model

def generate_and_filter_for_user(model, vae, transport, classifier, user_id, 
                                 target_samples=800, batch_size=100, 
                                 confidence_threshold=0.95, cfg_scale=12.0, 
                                 output_dir='./filtered_samples', device=None, rank=0):
    """为单个用户生成并筛选样本直到收集到目标数量"""
    
    # 创建采样器和采样函数
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="dopri5",
        num_steps=300,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.1
    )
    
    # 创建输出目录
    user_dir = Path(output_dir) / f"User_{user_id:02d}"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # 图像预处理（与分类器训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    collected_samples = []
    total_generated = 0
    
    print(f"🎯 开始为User_{user_id:02d}生成样本，目标: {target_samples}张")
    
    with torch.no_grad():
        while len(collected_samples) < target_samples:
            # 生成一批样本
            current_batch_size = min(batch_size, target_samples - len(collected_samples))
            
            # 准备条件
            y = torch.tensor([user_id] * current_batch_size, device=device)
            
            # 创建随机噪声
            z = torch.randn(current_batch_size, 32, 16, 16, device=device)
            
            # CFG采样
            if cfg_scale > 1.0:
                z_cfg = torch.cat([z, z], 0)
                y_null = torch.tensor([31] * current_batch_size, device=device)
                y_cfg = torch.cat([y, y_null], 0)
                
                cfg_interval_start = 0.11
                model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                
                if hasattr(model, 'forward_with_cfg'):
                    samples = sample_fn(z_cfg, model.forward_with_cfg, **model_kwargs)
                else:
                    def model_fn_cfg(x, t, **kwargs):
                        pred = model(x, t, **kwargs)
                        pred_cond, pred_uncond = pred.chunk(2, dim=0)
                        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    samples = sample_fn(z_cfg, model_fn_cfg, **model_kwargs)
                
                samples = samples[-1]
                samples, _ = samples.chunk(2, dim=0)
            else:
                samples = sample_fn(z, model, **dict(y=y))
                samples = samples[-1]
            
            # 反归一化
            latent_stats_path = '/kaggle/working/VA-VAE/latents_safetensors/train/latent_stats.pt'
            if os.path.exists(latent_stats_path):
                stats = torch.load(latent_stats_path, map_location=device)
                mean = stats['mean'].to(device)
                std = stats['std'].to(device)
                latent_multiplier = 1.0
                samples_denorm = (samples * std) / latent_multiplier + mean
            else:
                # 备用：从数据集计算
                try:
                    from LightningDiT.datasets.img_latent_dataset import ImgLatentDataset
                    train_dataset = ImgLatentDataset('./latents_safetensors/train', latent_norm=True)
                    stats = train_dataset.compute_latent_stats()
                    mean = stats['mean'].to(device).squeeze(0)
                    std = stats['std'].to(device).squeeze(0)
                    
                    # 保存统计文件
                    os.makedirs('./latents_safetensors/train', exist_ok=True)
                    torch.save({'mean': mean, 'std': std}, './latents_safetensors/train/latent_stats.pt')
                    
                    samples_denorm = (samples * std) / 1.0 + mean
                except:
                    print("⚠️ 无法加载latent统计，使用原始样本")
                    samples_denorm = samples
            
            # VAE解码
            if vae is not None:
                try:
                    decoded_images = vae.decode_to_images(samples_denorm)
                    
                    # 准备批量分类
                    batch_tensors = []
                    for image in decoded_images:
                        pil_image = Image.fromarray(image)
                        image_tensor = transform(pil_image).unsqueeze(0)
                        batch_tensors.append(image_tensor)
                    
                    batch_input = torch.cat(batch_tensors, dim=0).to(device)
                    
                    # 分类器预测
                    with torch.no_grad():
                        outputs = classifier(batch_input)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidences, predictions = torch.max(probabilities, dim=1)
                    
                    # 筛选高质量样本
                    for i in range(len(decoded_images)):
                        if predictions[i].item() == user_id and confidences[i].item() > confidence_threshold:
                            # 保存图像
                            save_path = user_dir / f"sample_{len(collected_samples):06d}.png"
                            Image.fromarray(decoded_images[i]).save(save_path)
                            collected_samples.append(save_path)
                            
                            if len(collected_samples) >= target_samples:
                                break
                    
                    total_generated += current_batch_size
                    
                    # 打印进度
                    success_rate = len(collected_samples) / total_generated * 100 if total_generated > 0 else 0
                    print(f"User_{user_id:02d}: 已收集 {len(collected_samples)}/{target_samples} | "
                          f"生成了 {total_generated} 张 | 成功率: {success_rate:.1f}%")
                    
                except Exception as e:
                    print(f"❌ 处理批次时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ VAE未加载，无法解码")
                break
    
    print(f"✅ User_{user_id:02d} 完成: 收集了 {len(collected_samples)} 个高质量样本")
    return len(collected_samples)

def main():
    parser = argparse.ArgumentParser(description='Generate and filter high-quality samples')
    parser.add_argument('--dit_checkpoint', type=str, 
                       default='/kaggle/input/50000-pt/0050000.pt', 
                       help='DiT model checkpoint path')
    parser.add_argument('--classifier_checkpoint', type=str,
                       default='./best_classifier.pth',
                       help='Classifier checkpoint path')
    parser.add_argument('--config', type=str, 
                       default='configs/dit_s_microdoppler.yaml', 
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./filtered_samples_cfg12_095', 
                       help='Output directory')
    parser.add_argument('--target_samples', type=int, default=800, 
                       help='Target number of samples per user')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='Batch size for generation')
    parser.add_argument('--confidence_threshold', type=float, default=0.95, 
                       help='Confidence threshold for filtering')
    parser.add_argument('--cfg_scale', type=float, default=12.0, 
                       help='CFG scale for generation')
    parser.add_argument('--start_user', type=int, default=0,
                       help='Starting user ID')
    parser.add_argument('--end_user', type=int, default=30,
                       help='Ending user ID (inclusive)')
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"🚀 生成并筛选高质量样本")
        print(f"📂 输出目录: {args.output_dir}")
        print(f"🎯 目标: 每用户 {args.target_samples} 张样本")
        print(f"📊 置信度阈值: {args.confidence_threshold}")
        print(f"⚙️ CFG Scale: {args.cfg_scale}")
    
    # 加载DiT模型
    model, vae, transport, config, device = load_model_and_config(
        args.dit_checkpoint, args.config, local_rank
    )
    
    # 加载分类器
    classifier = load_classifier(args.classifier_checkpoint, device)
    
    # 创建输出目录
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # 为指定范围的用户生成样本
    total_collected = 0
    for user_id in range(args.start_user, args.end_user + 1):
        if rank == 0:  # 单卡处理
            collected = generate_and_filter_for_user(
                model, vae, transport, classifier, user_id,
                target_samples=args.target_samples,
                batch_size=args.batch_size,
                confidence_threshold=args.confidence_threshold,
                cfg_scale=args.cfg_scale,
                output_dir=args.output_dir,
                device=device,
                rank=rank
            )
            total_collected += collected
    
    if rank == 0:
        print(f"🎯 生成完成！")
        print(f"✅ 总共收集了 {total_collected} 个高质量样本")
        print(f"📁 样本保存在: {args.output_dir}")
    
    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
