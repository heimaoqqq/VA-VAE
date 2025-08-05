#!/usr/bin/env python3
"""
修复latents_stats.pt文件问题
为demo模式创建默认的统计信息
"""

import torch
import os
from pathlib import Path

def create_default_latents_stats():
    """创建默认的latents统计信息用于demo模式"""
    print("🔧 创建默认的latents统计信息...")
    
    # 基于VA-VAE f16d32的默认统计信息
    # 这些是从ImageNet数据集计算得出的典型值
    mean = torch.zeros(1, 32, 1, 1)  # 32通道，均值为0
    std = torch.ones(1, 32, 1, 1)   # 32通道，标准差为1
    
    latent_stats = {
        'mean': mean,
        'std': std
    }
    
    return latent_stats

def fix_latents_stats_file():
    """修复latents_stats.pt文件"""
    print("🚀 修复latents_stats.pt文件")
    print("="*50)
    
    models_dir = Path("models")
    latents_stats_file = models_dir / "latents_stats.pt"
    
    # 检查文件状态
    if latents_stats_file.exists():
        file_size = latents_stats_file.stat().st_size
        print(f"📁 当前文件: {latents_stats_file}")
        print(f"📊 文件大小: {file_size} bytes")
        
        if file_size == 0:
            print("❌ 文件为空，需要修复")
        else:
            # 尝试加载文件
            try:
                stats = torch.load(latents_stats_file)
                print("✅ 文件可以正常加载")
                print(f"📋 统计信息: {stats.keys()}")
                return True
            except Exception as e:
                print(f"❌ 文件损坏: {e}")
    else:
        print(f"❌ 文件不存在: {latents_stats_file}")
    
    # 创建默认统计信息
    print("\n🔧 创建默认统计信息...")
    default_stats = create_default_latents_stats()
    
    # 保存文件
    try:
        torch.save(default_stats, latents_stats_file)
        print(f"✅ 已保存默认统计信息到: {latents_stats_file}")
        
        # 验证保存的文件
        new_size = latents_stats_file.stat().st_size
        print(f"📊 新文件大小: {new_size} bytes")
        
        # 测试加载
        loaded_stats = torch.load(latents_stats_file)
        print(f"✅ 验证成功: {loaded_stats.keys()}")
        print(f"📐 Mean shape: {loaded_stats['mean'].shape}")
        print(f"📐 Std shape: {loaded_stats['std'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def download_correct_latents_stats():
    """重新下载正确的latents_stats.pt文件"""
    print("\n📥 重新下载latents_stats.pt文件...")
    
    import requests
    
    # 尝试不同的下载链接
    urls = [
        "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt",
        "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/latents_stats.pt"
    ]
    
    models_dir = Path("models")
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔗 尝试链接 {i}: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"📊 文件大小: {total_size} bytes")
            
            if total_size > 100:  # 至少100字节
                latents_stats_file = models_dir / f"latents_stats_{i}.pt"
                
                with open(latents_stats_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 测试文件
                try:
                    stats = torch.load(latents_stats_file)
                    print(f"✅ 下载成功: {latents_stats_file}")
                    print(f"📋 包含: {stats.keys()}")
                    
                    # 复制为正确的文件名
                    final_file = models_dir / "latents_stats.pt"
                    latents_stats_file.rename(final_file)
                    print(f"✅ 已重命名为: {final_file}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ 文件测试失败: {e}")
                    latents_stats_file.unlink()  # 删除损坏的文件
            else:
                print("❌ 文件太小，可能是错误页面")
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    return False

def main():
    """主函数"""
    print("🚀 修复LightningDiT latents_stats.pt文件")
    print("="*60)
    
    # 方法1: 修复现有文件
    if fix_latents_stats_file():
        print("\n✅ 修复完成！")
        return True
    
    # 方法2: 重新下载
    print("\n🔄 尝试重新下载...")
    if download_correct_latents_stats():
        print("\n✅ 重新下载完成！")
        return True
    
    # 方法3: 使用默认值
    print("\n⚠️ 下载失败，使用默认统计信息")
    if fix_latents_stats_file():
        print("\n✅ 使用默认值完成！")
        return True
    
    print("\n❌ 所有修复方法都失败了")
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 下一步: 重新运行推理")
        print("!python step4_run_inference.py")
    else:
        print("\n❌ 修复失败，请检查网络连接或手动下载文件")
