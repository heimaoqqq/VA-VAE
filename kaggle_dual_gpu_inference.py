#!/usr/bin/env python3
"""
Kaggle双GPU推理包装器
基于原项目inference.py的分布式推理方法
"""

import os
import sys
import subprocess
from accelerate import notebook_launcher

def setup_paths():
    """设置Python路径"""
    print("🔧 设置推理路径...")
    
    # 获取当前工作目录
    if '/kaggle/working' in os.getcwd():
        base_dir = '/kaggle/working/VA-VAE'
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加必要的路径
    paths_to_add = [
        os.path.join(base_dir, 'LightningDiT'),
        base_dir
    ]
    
    for path in reversed(paths_to_add):
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ 已添加路径: {path}")

def kaggle_dual_gpu_inference():
    """双GPU推理函数"""
    
    def inference_worker():
        # 设置路径
        setup_paths()
        
        # 导入Accelerator
        from accelerate import Accelerator
        import torch
        
        # 初始化Accelerator
        accelerator = Accelerator()
        
        print(f"🔧 推理进程 {accelerator.process_index}/{accelerator.num_processes}")
        print(f"🔧 设备: {accelerator.device}")
        
        # 设置命令行参数
        sys.argv = [
            'stage3_inference.py',
            '--dit_checkpoint', '/kaggle/working/trained_models/best_model',
            '--vavae_config', 'vavae_config.yaml',
            '--output_dir', '/kaggle/working/generated_images',
            '--user_ids', '1', '2', '3', '4', '5',
            '--num_samples_per_user', '4',
            '--seed', '42'
        ]
        
        # 导入并运行推理
        from stage3_inference_distributed import main
        main(accelerator)
    
    # 使用notebook_launcher启动双GPU推理
    print("🚀 启动双GPU推理...")
    notebook_launcher(inference_worker, num_processes=2)
    print("✅ 双GPU推理完成")

if __name__ == "__main__":
    kaggle_dual_gpu_inference()
