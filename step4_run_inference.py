#!/usr/bin/env python3
"""
步骤4: 运行LightningDiT推理
Kaggle环境优化版本
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

def check_environment():
    """检查推理环境"""
    print("🔍 检查推理环境...")
    
    # 检查CUDA
    print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 检查GPU内存
        for i in range(torch.cuda.device_count()):
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i} 内存: {memory_total:.1f} GB")
    
    # 检查关键模块
    required_modules = ['accelerate', 'torchdiffeq', 'timm', 'diffusers']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 可用")
        except ImportError:
            print(f"❌ {module}: 不可用")
            return False
    
    return True

def check_files():
    """检查必需文件"""
    print("\n🔍 检查必需文件...")
    
    # 检查配置文件
    config_file = Path("kaggle_inference_config.yaml")
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    print(f"✅ 配置文件: {config_file}")
    
    # 检查LightningDiT目录
    lightningdit_dir = Path("LightningDiT")
    if not lightningdit_dir.exists():
        print(f"❌ LightningDiT目录不存在: {lightningdit_dir}")
        return False
    print(f"✅ LightningDiT目录: {lightningdit_dir}")
    
    # 检查推理脚本
    inference_script = lightningdit_dir / "inference.py"
    if not inference_script.exists():
        print(f"❌ 推理脚本不存在: {inference_script}")
        return False
    print(f"✅ 推理脚本: {inference_script}")
    
    # 检查模型文件
    models_dir = Path("models")
    required_models = [
        "vavae-imagenet256-f16d32-dinov2.pt",
        "lightningdit-xl-imagenet256-800ep.pt",
        "latents_stats.pt"
    ]
    
    for model in required_models:
        model_path = models_dir / model
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"✅ {model}: {size_mb:.1f} MB")
    
    return True

def run_inference():
    """运行推理 - 使用官方accelerate launch方式"""
    print("\n🚀 开始LightningDiT推理...")

    # 切换到LightningDiT目录
    original_cwd = os.getcwd()
    lightningdit_dir = Path("LightningDiT")

    try:
        os.chdir(lightningdit_dir)
        current_dir = Path.cwd()
        print(f"📁 切换到目录: {current_dir}")

        # 使用官方配置文件
        config_path = "configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml"
        config_abs_path = Path(config_path).absolute()

        print(f"📋 使用官方配置: {config_abs_path}")

        # 验证配置文件存在
        if not Path(config_path).exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False

        # 构建官方推理命令 - 使用accelerate launch + --demo参数
        cmd = f"accelerate launch --mixed_precision bf16 inference.py --config {config_path} --demo"
        print(f"💻 执行官方命令: {cmd}")

        # 运行推理
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )

        # 输出结果
        if result.stdout:
            print("📤 标准输出:")
            print(result.stdout)

        if result.stderr:
            print("📤 错误输出:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ 推理成功完成！")
            return True
        else:
            print(f"❌ 推理失败，返回码: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 推理超时（30分钟）")
        return False
    except Exception as e:
        print(f"❌ 推理异常: {e}")
        return False
    finally:
        # 恢复原始目录
        os.chdir(original_cwd)

def verify_results():
    """验证推理结果"""
    print("\n🔍 验证推理结果...")
    
    # 检查输出目录
    output_dirs = [
        Path("LightningDiT/demo_images"),
        Path("LightningDiT/output"),
        Path("LightningDiT/samples")
    ]
    
    found_images = []
    
    for output_dir in output_dirs:
        if output_dir.exists():
            print(f"📁 发现输出目录: {output_dir}")
            
            # 查找图像文件
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images = list(output_dir.glob(ext))
                found_images.extend(images)
                
                for img in images:
                    size_mb = img.stat().st_size / 1024 / 1024
                    print(f"   📸 {img.name}: {size_mb:.2f} MB")
    
    if found_images:
        print(f"✅ 找到 {len(found_images)} 个生成图像")
        
        # 显示主要输出文件
        demo_samples = Path("LightningDiT/demo_images/demo_samples.png")
        if demo_samples.exists():
            size_mb = demo_samples.stat().st_size / 1024 / 1024
            print(f"🎯 主要输出: {demo_samples} ({size_mb:.2f} MB)")
        
        return True
    else:
        print("❌ 未找到生成的图像")
        return False

def display_summary():
    """显示总结"""
    print("\n" + "="*60)
    print("🎉 LightningDiT推理完成！")
    print("="*60)
    
    # 显示输出文件
    demo_samples = Path("LightningDiT/demo_images/demo_samples.png")
    if demo_samples.exists():
        print(f"📸 生成图像: {demo_samples}")
        print("💡 这是ImageNet-256类别的高质量生成图像")
        print("💡 FID=1.35，达到SOTA水平")
    
    print("\n📋 复现成功标志:")
    print("✅ 环境安装正确")
    print("✅ 模型下载完整")
    print("✅ 配置设置正确")
    print("✅ 推理运行成功")
    print("✅ 图像生成完成")
    
    print("\n🎯 下一步建议:")
    print("1. 查看生成的图像质量")
    print("2. 理解VA-VAE + LightningDiT pipeline")
    print("3. 考虑如何适配您的31用户微多普勒数据")

def main():
    """主函数"""
    print("🚀 步骤4: 运行LightningDiT推理")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 检查文件
    if not check_files():
        print("❌ 文件检查失败")
        return False
    
    # 运行推理
    if not run_inference():
        print("❌ 推理失败")
        return False
    
    # 验证结果
    if not verify_results():
        print("❌ 结果验证失败")
        return False
    
    # 显示总结
    display_summary()
    
    print("\n✅ 步骤4完成！LightningDiT官方复现成功")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
