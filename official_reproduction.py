#!/usr/bin/env python3
"""
严格按照LightningDiT官方README进行复现
不做任何修改，完全按照官方流程
"""

import os
import subprocess
import sys
import requests
from pathlib import Path
import yaml

def run_command(cmd, description="", cwd=None):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"💻 执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True, cwd=cwd)
        print("✅ 成功")
        if result.stdout.strip():
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print(f"错误: {e.stderr.strip()}")
        if e.stdout:
            print(f"输出: {e.stdout.strip()}")
        return False

def download_file(url, local_path):
    """下载文件"""
    print(f"📥 下载: {url}")
    print(f"📁 保存到: {local_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def step1_install_dependencies():
    """步骤1: 安装依赖"""
    print("\n" + "="*60)
    print("📦 步骤1: 安装官方依赖")
    print("="*60)
    
    # 进入LightningDiT目录
    lightningdit_dir = Path("LightningDiT")
    if not lightningdit_dir.exists():
        print("❌ LightningDiT目录不存在")
        return False
    
    # 检查requirements.txt
    requirements_file = lightningdit_dir / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt不存在")
        return False
    
    print("📋 官方requirements.txt内容:")
    with open(requirements_file, 'r') as f:
        content = f.read()
        print(content)
    
    # 安装依赖
    print("\n🔧 安装依赖...")
    return run_command(
        f"pip install -r requirements.txt",
        "安装官方requirements.txt",
        cwd=str(lightningdit_dir)
    )

def step2_download_models():
    """步骤2: 下载预训练模型"""
    print("\n" + "="*60)
    print("📥 步骤2: 下载预训练模型")
    print("="*60)
    
    # 创建模型目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 官方README中的模型链接
    models = {
        "VA-VAE": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt",
        "LightningDiT-XL-800ep": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
        "Latent Statistics": "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt"
    }
    
    success_count = 0
    for name, url in models.items():
        filename = url.split('/')[-1]
        filepath = models_dir / filename
        
        if filepath.exists():
            print(f"✅ {name}: 已存在 ({filepath})")
            success_count += 1
        else:
            print(f"\n📥 下载 {name}...")
            if download_file(url, str(filepath)):
                success_count += 1
            else:
                print(f"❌ {name} 下载失败")
    
    print(f"\n📊 下载结果: {success_count}/{len(models)} 个模型成功")
    return success_count == len(models)

def step3_setup_config():
    """步骤3: 设置配置文件"""
    print("\n" + "="*60)
    print("⚙️ 步骤3: 设置配置文件")
    print("="*60)
    
    # 使用官方的reproduction配置
    config_file = Path("LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml")
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"✅ 使用官方配置: {config_file}")
    
    # 读取配置文件
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📋 配置文件内容:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    # 更新模型路径
    models_dir = Path("models").absolute()
    
    # 更新VA-VAE配置
    vavae_config = Path("LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
    if vavae_config.exists():
        print(f"\n🔧 更新VA-VAE配置: {vavae_config}")
        
        with open(vavae_config, 'r') as f:
            vavae_cfg = yaml.safe_load(f)
        
        # 更新检查点路径
        vavae_cfg['ckpt_path'] = str(models_dir / "vavae-imagenet256-f16d32-dinov2.pt")
        
        with open(vavae_config, 'w') as f:
            yaml.dump(vavae_cfg, f, default_flow_style=False, indent=2)
        
        print("✅ VA-VAE配置已更新")
    
    return True

def step4_run_inference():
    """步骤4: 运行推理"""
    print("\n" + "="*60)
    print("🚀 步骤4: 运行官方推理")
    print("="*60)
    
    # 使用官方推荐的配置
    config_path = "configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml"
    
    # 运行官方快速推理脚本
    print("🎯 运行官方快速推理脚本...")
    
    lightningdit_dir = Path("LightningDiT")
    
    # 检查脚本是否存在
    inference_script = lightningdit_dir / "run_fast_inference.sh"
    if not inference_script.exists():
        print(f"❌ 推理脚本不存在: {inference_script}")
        return False
    
    # 在Windows上，我们需要直接运行Python脚本
    print("🔧 在Windows环境下运行推理...")
    
    # 直接运行inference.py
    cmd = f"python inference.py --config {config_path}"
    
    return run_command(
        cmd,
        "运行LightningDiT推理",
        cwd=str(lightningdit_dir)
    )

def verify_results():
    """验证结果"""
    print("\n" + "="*60)
    print("🔍 验证结果")
    print("="*60)
    
    # 检查输出图像
    demo_images = Path("LightningDiT/demo_images/demo_samples.png")
    
    if demo_images.exists():
        print(f"✅ 生成图像成功: {demo_images}")
        print(f"📊 文件大小: {demo_images.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"❌ 未找到生成图像: {demo_images}")
        
        # 检查可能的其他输出位置
        possible_paths = [
            Path("LightningDiT/demo_images"),
            Path("LightningDiT/output"),
            Path("LightningDiT/samples")
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"📁 发现目录: {path}")
                for file in path.iterdir():
                    print(f"   - {file.name}")
        
        return False

def main():
    """主函数"""
    print("🚀 LightningDiT官方复现")
    print("严格按照官方README执行，不做任何修改")
    print("="*60)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    if not Path("LightningDiT").exists():
        print("❌ LightningDiT目录不存在，请确保已正确克隆项目")
        return False
    
    # 执行步骤
    steps = [
        ("安装依赖", step1_install_dependencies),
        ("下载模型", step2_download_models),
        ("设置配置", step3_setup_config),
        ("运行推理", step4_run_inference),
        ("验证结果", verify_results)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🎯 开始: {step_name}")
        if not step_func():
            print(f"❌ {step_name} 失败")
            return False
        print(f"✅ {step_name} 完成")
    
    print("\n" + "="*60)
    print("🎉 官方复现完成！")
    print("📁 生成图像位置: LightningDiT/demo_images/demo_samples.png")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
