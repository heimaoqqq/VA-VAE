#!/usr/bin/env python3
"""
VA-VAE 官方3阶段微调脚本
按照官方LightningDiT推荐的策略依次执行3个阶段的训练
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import inspect

# 修复 academictorrents 在 Python 3.11 中的兼容性问题
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# Kaggle环境自动设置
def setup_kaggle_paths():
    """自动设置Kaggle环境路径"""
    taming_path = Path("taming-transformers").absolute()
    if taming_path.exists() and str(taming_path) not in sys.path:
        sys.path.insert(0, str(taming_path))
        print(f"🔧 自动添加taming路径: {taming_path}")

# 在导入检查前先设置路径
setup_kaggle_paths()

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    # 检查taming-transformers - 使用正确的导入方式
    try:
        import taming.data.utils as tdu
        import taming.modules.losses.vqperceptual
        from taming.modules.vqvae.quantize import VectorQuantizer2
        print("✅ taming-transformers 已安装并可正常导入")
    except ImportError as e:
        print("❌ taming-transformers 未正确安装")
        print(f"   导入错误: {e}")
        print("💡 请按照官方方式安装:")
        print("   git clone https://github.com/CompVis/taming-transformers.git")
        print("   cd taming-transformers")
        print("   pip install -e .")
        print("   # 修复torch 2.x兼容性:")
        print("   sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' taming/data/utils.py")
        return False
    
    # 检查pytorch-lightning
    try:
        import pytorch_lightning as pl
        print(f"✅ pytorch-lightning {pl.__version__} 已安装")
    except ImportError:
        print("❌ pytorch-lightning 未安装")
        print("💡 请先安装: pip install pytorch-lightning")
        return False
    
    # 检查其他必要依赖
    try:
        import omegaconf
        import einops
        print("✅ 其他依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("💡 请安装: pip install omegaconf einops")
        return False
    
    return True

def check_model_and_data():
    """检查预训练模型和数据"""
    print("🔍 检查模型和数据...")
    
    # 检查预训练模型
    model_path = Path("models/vavae-imagenet256-f16d32-dinov2.pt")
    if not model_path.exists():
        print(f"❌ 预训练模型不存在: {model_path}")
        print("💡 请先下载模型到 models/ 目录")
        return False
    print(f"✅ 预训练模型存在: {model_path}")
    
    # 检查数据目录（假设在Kaggle环境）
    data_dir = Path("/kaggle/input/dataset")
    if data_dir.exists():
        print(f"✅ 数据目录存在: {data_dir}")
    else:
        print(f"⚠️ 数据目录不存在: {data_dir}")
        print("💡 请确保数据已正确挂载")
    
    return True

def run_training_stage(stage_name, config_path, stage_num):
    """运行单个训练阶段"""
    print(f"\n🚀 开始{stage_name} (阶段{stage_num})")
    print(f"📋 配置文件: {config_path}")
    
    # 切换到LightningDiT/vavae目录
    vavae_dir = Path("LightningDiT/vavae")
    if not vavae_dir.exists():
        print(f"❌ 目录不存在: {vavae_dir}")
        return False
    
    # 构建训练命令
    cmd = [
        sys.executable, "main.py",
        "--base", f"../../{config_path}",
        "--train"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"📁 工作目录: {vavae_dir.absolute()}")
    
    # 设置环境变量，包含taming路径
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"  # 双GPU
    
    # 添加taming-transformers到PYTHONPATH
    taming_path = str(Path("taming-transformers").absolute())
    if taming_path not in sys.path:
        sys.path.insert(0, taming_path)
    
    # 设置PYTHONPATH环境变量供子进程使用
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{taming_path}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = taming_path
    
    print(f"🔧 设置PYTHONPATH: {env['PYTHONPATH']}")
    
    try:
        start_time = time.time()
        
        # 运行训练
        result = subprocess.run(
            cmd,
            cwd=vavae_dir,
            env=env,
            check=True,
            capture_output=False  # 实时显示输出
        )
        
        end_time = time.time()
        duration = (end_time - start_time) / 60  # 转换为分钟
        
        print(f"✅ {stage_name}完成 (用时: {duration:.1f}分钟)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {stage_name}失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {stage_name}被用户中断")
        return False
    except Exception as e:
        print(f"❌ {stage_name}出错: {e}")
        return False

def update_config_paths():
    """更新配置文件中的checkpoint路径"""
    print("🔧 更新配置文件中的checkpoint路径...")
    
    # 查找最新的checkpoint
    stage1_ckpt_dir = Path("LightningDiT/vavae/logs")
    if stage1_ckpt_dir.exists():
        # 这里可以添加自动查找最新checkpoint的逻辑
        print("💡 请手动检查并更新配置文件中的weight_init路径")
    
    return True

def main():
    """主函数 - 执行3阶段微调"""
    print("=" * 60)
    print("🎯 VA-VAE官方3阶段微调")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请先安装必要的依赖")
        return False
    
    # 检查模型和数据
    if not check_model_and_data():
        print("❌ 模型或数据检查失败")
        return False
    
    # 定义3个阶段
    stages = [
        ("DINOv2对齐训练", "configs/stage1_alignment.yaml", 1),
        ("重建优化训练", "configs/stage2_reconstruction.yaml", 2),
        ("Margin优化训练", "configs/stage3_margin.yaml", 3)
    ]
    
    # 依次执行3个阶段
    for stage_name, config_path, stage_num in stages:
        success = run_training_stage(stage_name, config_path, stage_num)
        
        if not success:
            print(f"❌ {stage_name}失败，停止后续训练")
            return False
        
        # 在阶段1和2之后，提示用户更新配置文件
        if stage_num < 3:
            print(f"\n⚠️ 请检查并更新阶段{stage_num+1}配置文件中的weight_init路径")
            print("💡 路径通常在: LightningDiT/vavae/logs/*/checkpoints/last.ckpt")
            
            # 可选：等待用户确认
            input("按Enter键继续下一阶段...")
    
    print("\n🎉 3阶段微调全部完成！")
    print("📊 建议运行评估脚本检查FID改善情况")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
