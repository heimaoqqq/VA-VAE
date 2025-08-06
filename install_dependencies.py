#!/usr/bin/env python3
"""
VA-VAE Kaggle环境一键安装脚本
解决所有依赖问题，确保taming-transformers正确工作
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Kaggle环境一键设置"""
    print("🎯 VA-VAE Kaggle环境设置")
    print("=" * 40)
    
    # 1. 安装基础依赖
    print("📦 安装基础依赖...")
    deps = ["pytorch-lightning", "omegaconf", "einops", "transformers", "six"]
    for dep in deps:
        print(f"   安装 {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"], 
                      capture_output=True)
    
    # 修复academictorrents的Python 3.11兼容性问题
    print("   修复academictorrents兼容性...")
    # 先安装pypubsub的兼容版本
    subprocess.run([sys.executable, "-m", "pip", "install", "pypubsub==4.0.3", "-q"], 
                  capture_output=True)
    # 然后安装academictorrents
    subprocess.run([sys.executable, "-m", "pip", "install", "academictorrents", "-q"], 
                  capture_output=True)
    
    # 2. 设置taming-transformers
    taming_dir = Path("taming-transformers")
    if not taming_dir.exists():
        print("📥 克隆taming-transformers...")
        subprocess.run(["git", "clone", 
                       "https://github.com/CompVis/taming-transformers.git"],
                      capture_output=True)
    
    # 3. 修复兼容性
    utils_file = taming_dir / "taming" / "data" / "utils.py"
    if utils_file.exists():
        print("🔧 修复torch兼容性...")
        content = utils_file.read_text()
        if "from torch._six import string_classes" in content:
            content = content.replace(
                "from torch._six import string_classes",
                "from six import string_types as string_classes"
            )
            utils_file.write_text(content)
    
    # 4. 添加到Python路径
    taming_path = str(taming_dir.absolute())
    if taming_path not in sys.path:
        sys.path.insert(0, taming_path)
    
    # 5. 验证
    try:
        import taming.data.utils
        import pytorch_lightning as pl
        print("✅ 环境设置完成！")
        print(f"   - taming-transformers: 已添加到路径")
        print(f"   - pytorch-lightning: {pl.__version__}")
        print("\n💡 现在可以运行: python finetune_vavae.py")
        
        # 保存路径信息供后续使用
        with open(".taming_path", "w") as f:
            f.write(taming_path)
        
        return True
    except ImportError as e:
        print(f"❌ 验证失败: {e}")
        print("💡 请重启内核后重试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
