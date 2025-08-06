#!/usr/bin/env python3
"""
修复 academictorrents 在 Python 3.11 中的兼容性问题
通过猴子补丁修复 getargspec 导入错误
"""

import sys
import inspect

def fix_getargspec():
    """修复 getargspec 在 Python 3.11 中被移除的问题"""
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

def apply_academictorrents_fix():
    """应用 academictorrents 兼容性修复"""
    try:
        # 修复 getargspec 问题
        fix_getargspec()
        
        # 尝试导入 academictorrents
        import academictorrents
        print("✅ academictorrents 兼容性修复成功")
        return True
    except Exception as e:
        print(f"❌ academictorrents 修复失败: {e}")
        return False

if __name__ == "__main__":
    apply_academictorrents_fix()
