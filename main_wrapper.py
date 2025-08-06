#!/usr/bin/env python3
"""
main.py的包装器，在启动前应用所有兼容性修复
"""

import sys
import os
import inspect
from pathlib import Path

# 修复 academictorrents 在 Python 3.11 中的兼容性问题
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
    print("✅ 已应用 getargspec 兼容性修复")

# 确保 taming-transformers 在路径中
taming_path = str(Path("../../taming-transformers").absolute())
if taming_path not in sys.path:
    sys.path.insert(0, taming_path)
    print(f"✅ 已添加 taming 路径: {taming_path}")

# 导入并运行原始 main.py
if __name__ == "__main__":
    # 切换到正确的工作目录
    os.chdir(Path(__file__).parent)
    
    # 导入原始main模块并执行
    import importlib.util
    main_path = Path(__file__).parent / "main.py"
    
    spec = importlib.util.spec_from_file_location("main", main_path)
    main_module = importlib.util.module_from_spec(spec)
    
    # 将命令行参数传递给main模块
    sys.argv[0] = str(main_path)  # 修正argv[0]为main.py
    
    spec.loader.exec_module(main_module)
