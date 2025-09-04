#!/usr/bin/env python3
"""快速验证量化模型是否真实量化"""

import torch
from pathlib import Path
import sys
import os

# 添加必要的路径
sys.path.append('/kaggle/working/VA-VAE')
sys.path.append('/kaggle/working/VA-VAE/LightningDiT')

try:
    from models.lightningdit import LightningDiT_models
except ImportError:
    print("⚠️ 无法导入LightningDiT模型，跳过模型结构验证")

def quick_check():
    quantized_path = "/kaggle/working/dit_xl_quantized.pt"
    
    print(f"🔍 检查量化模型: {quantized_path}")
    
    # 检查文件大小
    if Path(quantized_path).exists():
        file_size_mb = Path(quantized_path).stat().st_size / (1024**2)
        print(f"📁 文件大小: {file_size_mb:.1f}MB")
        
        if file_size_mb < 1000:  # 小于1GB
            print("⚠️ 文件太小，可能量化失败")
        else:
            print("✅ 文件大小正常，可能是真实量化")
    else:
        print("❌ 文件不存在")
    
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
        print("📦 加载的是结构化字典")
        
        # 检查字典键
        print(f"   包含键: {list(data.keys())}")
        
        # 检查是否有量化模型
        if 'quantized_model' in data:
            print("✅ 找到 'quantized_model' 键")
            quantized_model = data['quantized_model']
            print(f"   量化模型类型: {type(quantized_model)}")
            
            if hasattr(quantized_model, 'named_modules'):
                print("✅ 量化模型是完整模型对象")
                
                # 检查量化层
                quantized_layers = 0
                total_modules = 0
                
                for name, module in quantized_model.named_modules():
                    total_modules += 1
                    # 检查多种量化标识
                    if (hasattr(module, '_packed_params') or 
                        'quantized' in str(type(module)).lower() or
                        hasattr(module, 'scale') and hasattr(module, 'zero_point')):
                        quantized_layers += 1
                        print(f"   ✅ 量化层: {name} | 类型: {type(module).__name__}")
                
                print(f"\n📊 量化统计:")
                print(f"   总模块数: {total_modules}")
                print(f"   量化层数: {quantized_layers}")
                
                if quantized_layers == 0:
                    print("❌ 未找到量化层！检查权重类型...")
                    # 检查权重精度
                    param_count = 0
                    for name, param in quantized_model.named_parameters():
                        if param_count >= 3:
                            break
                        print(f"   {name[:40]:40} | dtype: {param.dtype} | shape: {param.shape}")
                        param_count += 1
                else:
                    print(f"✅ 量化成功！找到 {quantized_layers} 个量化层")
                    
            elif isinstance(quantized_model, dict):
                print("⚠️ 量化模型是state_dict，不是完整模型")
                print(f"   权重数量: {len(quantized_model)}")
                print("❌ 量化失败：保存的是权重而非量化模型结构")
            else:
                print(f"❓ 未知量化模型类型: {type(quantized_model)}")
        else:
            print("❌ 未找到 'quantized_model' 键")
            print("   可能的键:", list(data.keys()))
            print("❌ 量化失败：文件结构不正确")
        
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
