#!/usr/bin/env python3
"""
Kaggle双GPU测试脚本
在干净的环境中测试notebook_launcher
必须在新的Python进程中运行，避免CUDA初始化冲突
"""

def test_dual_gpu():
    """测试双GPU功能"""
    import torch
    from accelerate import Accelerator
    
    accelerator = Accelerator()
    
    print(f"🔧 进程 {accelerator.process_index}/{accelerator.num_processes}")
    print(f"🔧 设备: {accelerator.device}")
    print(f"🔧 分布式类型: {accelerator.distributed_type}")
    print(f"🔧 混合精度: {accelerator.mixed_precision}")
    
    # 测试GPU通信
    if accelerator.is_main_process:
        print("✅ 主进程启动成功")
    
    # 简单的张量操作测试
    x = torch.randn(100, 100).to(accelerator.device)
    y = x @ x.T
    
    print(f"✅ 张量计算成功，设备: {y.device}")
    print(f"✅ 张量形状: {y.shape}")
    
    # 测试内存使用
    memory_used = torch.cuda.memory_allocated(accelerator.device) / 1e6
    print(f"✅ GPU内存使用: {memory_used:.1f}MB")
    
    return True

def main():
    """主函数"""
    print("🎯 Kaggle双GPU测试")
    print("=" * 50)
    
    # 使用notebook_launcher启动双GPU测试
    from accelerate import notebook_launcher
    
    try:
        print("🚀 启动双GPU测试...")
        notebook_launcher(test_dual_gpu, num_processes=2)
        print("✅ 双GPU测试成功!")
        return True
    except Exception as e:
        print(f"❌ 双GPU测试失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Kaggle双GPU环境工作正常!")
    else:
        print("\n❌ 双GPU环境有问题，请检查配置")
