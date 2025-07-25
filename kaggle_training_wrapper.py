#!/usr/bin/env python3
"""
Kaggle双GPU训练包装器
使用notebook_launcher而不是accelerate launch
修复路径和依赖问题
"""

import os
import sys
import subprocess
from accelerate import notebook_launcher

def install_dependencies():
    """安装缺失的依赖"""
    print("🔧 检查并安装依赖...")
    
    required_packages = [
        "torchdiffeq>=0.2.3",
        "scipy>=1.9.0", 
        "einops>=0.6.0",
        "omegaconf>=2.3.0"
    ]
    
    for package in required_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                         check=True, capture_output=True)
            print(f"✅ 已安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  安装失败: {package} - {e}")

def setup_paths():
    """设置Python路径"""
    print("🔧 设置Python路径...")

    # 获取当前工作目录
    if '/kaggle/working' in os.getcwd():
        base_dir = '/kaggle/working/VA-VAE'
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 清理旧的路径
    paths_to_remove = []
    for path in sys.path:
        if 'VA-VAE' in path or 'LightningDiT' in path:
            paths_to_remove.append(path)

    for path in paths_to_remove:
        sys.path.remove(path)
        print(f"🗑️ 已移除旧路径: {path}")

    # 添加必要的路径到开头
    paths_to_add = [
        os.path.join(base_dir, 'LightningDiT'),
        base_dir
    ]

    for path in reversed(paths_to_add):  # 反向添加，确保正确的优先级
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ 已添加路径: {path}")

    # 清理模块缓存
    modules_to_clear = []
    for module_name in sys.modules.keys():
        if any(x in module_name for x in ['stage2_train_dit', 'models.lightningdit', 'transport', 'datasets']):
            modules_to_clear.append(module_name)

    for module_name in modules_to_clear:
        del sys.modules[module_name]
        print(f"🗑️ 已清理模块缓存: {module_name}")

    print(f"📍 当前工作目录: {os.getcwd()}")
    print(f"📍 Python路径前3项: {sys.path[:3]}")

def kaggle_stage1_extract_features():
    """阶段1: 特征提取 (Kaggle双GPU版本)"""
    
    def extract_features_worker():
        # 安装依赖和设置路径
        install_dependencies()
        setup_paths()
        
        # 导入并设置参数
        from stage1_extract_features import main

        # 模拟命令行参数
        class Args:
            data_dir = '/kaggle/working/data_split'
            vavae_config = 'vavae_config.yaml'
            output_path = '/kaggle/working/latent_features'
            batch_size = 16
            seed = 42

        args = Args()

        # 运行特征提取
        main(args)
    
    # 使用notebook_launcher启动双GPU训练
    print("🚀 启动双GPU特征提取...")
    notebook_launcher(extract_features_worker, num_processes=2)
    print("✅ 特征提取完成")

def kaggle_stage2_train_dit():
    """阶段2: DiT训练 (Kaggle双GPU版本)"""
    
    def train_dit_worker():
        # 安装依赖和设置路径
        install_dependencies()
        setup_paths()

        # 强制更新代码
        os.system("cd /kaggle/working/VA-VAE && git reset --hard origin/master")

        # 设置GPU内存管理
        import torch
        if torch.cuda.is_available():
            # 清理所有GPU缓存
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)

            # 设置内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试

            print(f"🔧 GPU内存管理已设置")
            print(f"🔧 可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"🔧 GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"🔧 GPU {i} 内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")

        # 设置命令行参数 - 保守配置避免内存问题
        sys.argv = [
            'stage2_train_dit.py',
            '--latent_dir', '/kaggle/working/latent_features',
            '--output_dir', '/kaggle/working/trained_models',
            '--model_name', 'LightningDiT-B/1',  # 使用较小的B模型
            '--batch_size', '8',  # 减少batch_size避免内存不足
            '--max_epochs', '50',
            '--lr', '1e-4',
            '--seed', '42',
            '--save_every', '10'
        ]

        # 导入并运行DiT训练
        from stage2_train_dit import main
        main()
    
    # 使用notebook_launcher启动双GPU训练
    print("🚀 启动双GPU DiT训练...")
    notebook_launcher(train_dit_worker, num_processes=2)
    print("✅ DiT训练完成")

def kaggle_complete_pipeline():
    """完整的Kaggle双GPU训练流程"""
    print("🎯 Kaggle双GPU完整训练流程")
    print("=" * 50)
    
    # 阶段1: 特征提取
    print("\n📊 阶段1: 特征提取")
    kaggle_stage1_extract_features()
    
    # 阶段2: DiT训练
    print("\n🤖 阶段2: DiT训练")
    kaggle_stage2_train_dit()
    
    # 阶段3: 图像生成 (单GPU即可)
    print("\n🎨 阶段3: 图像生成")
    os.system("""python stage3_inference.py \
        --dit_checkpoint /kaggle/working/trained_models/best_model \
        --vavae_config vavae_config.yaml \
        --output_dir /kaggle/working/generated_images \
        --user_ids 1 2 3 4 5 \
        --num_samples_per_user 4 \
        --seed 42""")
    
    print("🎉 完整流程完成!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "stage1":
            kaggle_stage1_extract_features()
        elif sys.argv[1] == "stage2":
            kaggle_stage2_train_dit()
        elif sys.argv[1] == "complete":
            kaggle_complete_pipeline()
        else:
            print("用法: python kaggle_training_wrapper.py [stage1|stage2|complete]")
    else:
        kaggle_complete_pipeline()
