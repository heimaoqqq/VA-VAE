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

from LightningDiT.sample import create_model_and_diffusion, sample_from_model
from LightningDiT.utils import set_logger, set_seed
from improved_classifier_training import ImprovedClassifier
from prepare_safetensors_dataset import MicroDopplerDataset
import torchvision.transforms as transforms


class AutomatedGenerationPipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_directories()
        self.load_models()
        self.setup_transforms()
        self.user_progress = defaultdict(int)  # 跟踪每个用户已保存的样本数
        
    def setup_logging(self):
        """设置日志"""
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
        # 加载扩散模型
        self.logger.info("加载扩散模型...")
        self.model, self.diffusion = create_model_and_diffusion(self.args.config)
        
        # 加载checkpoint
        if self.args.checkpoint:
            checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            self.logger.info(f"加载checkpoint: {self.args.checkpoint}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载分类器
        self.logger.info("加载分类器...")
        self.classifier = ImprovedClassifier(num_classes=self.args.num_users)
        classifier_checkpoint = torch.load(self.args.classifier_path, map_location='cpu')
        self.classifier.load_state_dict(classifier_checkpoint)
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
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
            # 创建条件标签
            y = torch.tensor(user_ids, device=self.device)
            
            # 生成样本
            samples = sample_from_model(
                self.model, 
                self.diffusion,
                batch_size=len(user_ids),
                class_labels=y,
                cfg_scale=self.args.cfg_scale,
                device=self.device
            )
            
            return samples
            
    def evaluate_samples(self, samples, expected_user_ids):
        """评估生成样本的质量"""
        batch_results = []
        
        with torch.no_grad():
            # 转换为PIL图像并预处理
            processed_samples = []
            for sample in samples:
                # 假设sample是CHW格式，需要转换为HWC
                if len(sample.shape) == 3:
                    sample = sample.permute(1, 2, 0)
                
                # 归一化到0-255范围
                sample = ((sample + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
                
                # 转换为PIL图像
                if sample.shape[2] == 1:
                    sample = np.repeat(sample, 3, axis=2)  # 灰度转RGB
                pil_image = Image.fromarray(sample)
                
                # 应用变换
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
                        'pil_image': Image.fromarray(
                            ((samples[i].permute(1, 2, 0) + 1) * 127.5)
                            .clamp(0, 255).cpu().numpy().astype(np.uint8)
                        )
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


def main():
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
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建并运行管道
    pipeline = AutomatedGenerationPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
