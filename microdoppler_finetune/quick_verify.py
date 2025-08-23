#!/usr/bin/env python3
"""快速验证量化模型是否真实量化"""

import torch
from pathlib import Path

def quick_check():
    quantized_path = "/kaggle/working/dit_xl_quantized.pt"
    
    if not Path(quantized_path).exists():
        print("❌ 量化模型文件不存在")
        return
    
    # 加载模型（PyTorch 2.6+ 需要 weights_only=False）
    try:
        model = torch.load(quantized_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 检查是否有量化层
    quantized_layers = 0
    total_modules = 0
    
    for name, module in model.named_modules():
        total_modules += 1
        if hasattr(module, '_packed_params'):
            quantized_layers += 1
            print(f"✅ 找到量化层: {name}")
    
    print(f"\n📊 量化统计:")
    print(f"   总模块数: {total_modules}")
    print(f"   量化层数: {quantized_layers}")
    
    if quantized_layers == 0:
        print("❌ 没有找到任何量化层！量化失败")
    else:
        print(f"✅ 找到 {quantized_layers} 个量化层")
    
    # 检查权重精度
    for name, param in list(model.named_parameters())[:3]:
        unique_vals = torch.unique(param.data.flatten()).numel()
        total_vals = param.data.numel()
        ratio = unique_vals / total_vals
        print(f"   {name}: 唯一值比例 {ratio:.4f}")

if __name__ == "__main__":
    quick_check()
