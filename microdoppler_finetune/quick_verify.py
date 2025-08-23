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
        data = torch.load(quantized_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 检查加载的数据类型
    print(f"📋 加载数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print("⚠️ 加载的是字典（state_dict），不是完整模型")
        print("   这表明保存的只是权重，不是量化模型结构")
        
        # 检查字典键
        print(f"   权重键数量: {len(data)}")
        print(f"   示例键: {list(data.keys())[:5]}")
        
        # 检查权重精度
        print("\n📊 权重精度分析:")
        for i, (name, param) in enumerate(list(data.items())[:5]):
            if isinstance(param, torch.Tensor) and param.numel() > 100:
                unique_vals = torch.unique(param.flatten()).numel()
                total_vals = param.numel()
                ratio = unique_vals / total_vals
                print(f"   {name[:50]:50} | 唯一值比例: {ratio:.4f} | dtype: {param.dtype}")
        
        print("\n❌ 结论: 量化失败！")
        print("   - 保存的是state_dict而非完整量化模型")
        print("   - 权重仍为float32，未量化")
        print("   - 文件小是因为只保存了权重，没有模型结构")
        
    elif hasattr(data, 'named_modules'):
        print("✅ 加载的是完整模型")
        
        # 检查是否有量化层
        quantized_layers = 0
        total_modules = 0
        
        for name, module in data.named_modules():
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
            
    else:
        print(f"❓ 未知数据类型: {type(data)}")

if __name__ == "__main__":
    quick_check()
