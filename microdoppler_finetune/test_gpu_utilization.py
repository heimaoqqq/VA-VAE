#!/usr/bin/env python3
"""
测试GPU利用率和内存分配 - 验证双GPU分布式训练是否正确工作
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import psutil
import subprocess

def get_gpu_memory_info():
    """获取GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        info = []
        for line in lines:
            parts = line.split(', ')
            info.append({
                'index': int(parts[0]),
                'name': parts[1],
                'memory_used': float(parts[2]),
                'memory_total': float(parts[3]),
                'utilization': float(parts[4])
            })
        return info
    except:
        return None

def test_worker(rank, world_size):
    """测试单个worker的GPU分配"""
    
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    # 设置当前进程的GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"\n{'='*60}")
    print(f"Process {rank}/{world_size-1} started")
    print(f"  PID: {os.getpid()}")
    print(f"  Device: {device}")
    print(f"  GPU Name: {torch.cuda.get_device_name(rank)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(rank).total_memory / 1024**3:.1f} GB")
    
    # 分配一些内存来验证GPU使用
    print(f"\nAllocating test tensors on GPU {rank}...")
    
    # 创建不同大小的张量来测试内存分配
    test_sizes = [
        (1024, 1024),      # ~4MB
        (4096, 4096),      # ~64MB
        (8192, 8192),      # ~256MB
    ]
    
    tensors = []
    for size in test_sizes:
        try:
            tensor = torch.randn(size, device=device)
            allocated = tensor.element_size() * tensor.nelement() / 1024**2
            tensors.append(tensor)
            print(f"  ✓ Allocated {size[0]}x{size[1]} tensor ({allocated:.1f} MB) on GPU {rank}")
            
            # 检查当前内存使用
            allocated_mb = torch.cuda.memory_allocated(rank) / 1024**2
            reserved_mb = torch.cuda.memory_reserved(rank) / 1024**2
            print(f"    Current GPU {rank} - Allocated: {allocated_mb:.1f} MB, Reserved: {reserved_mb:.1f} MB")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ✗ Failed to allocate {size[0]}x{size[1]} on GPU {rank}: {e}")
            break
    
    # 执行一些计算来验证GPU活动
    print(f"\nPerforming test computation on GPU {rank}...")
    if tensors:
        result = torch.zeros_like(tensors[0])
        for _ in range(100):
            result = result + tensors[0] * 0.01
        print(f"  ✓ Computation completed on GPU {rank}")
    
    # 测试分布式通信
    print(f"\nTesting distributed communication from GPU {rank}...")
    test_tensor = torch.ones(10, device=device) * (rank + 1)
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1)) * 10
    actual = test_tensor.sum().item()
    if abs(actual - expected) < 0.01:
        print(f"  ✓ All-reduce successful on GPU {rank} (sum={actual:.0f})")
    else:
        print(f"  ✗ All-reduce failed on GPU {rank} (expected={expected}, got={actual})")
    
    # 最终内存状态
    print(f"\nFinal memory status for GPU {rank}:")
    print(f"  Allocated: {torch.cuda.memory_allocated(rank) / 1024**2:.1f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(rank) / 1024**2:.1f} MB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated(rank) / 1024**2:.1f} MB")
    
    # 清理
    dist.destroy_process_group()
    print(f"\nProcess {rank} completed successfully")
    print(f"{'='*60}\n")

def test_batch_distribution(rank, world_size):
    """测试batch size在多GPU上的分配"""
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 模拟批次分配
    total_batch_size = 16
    batch_size_per_gpu = total_batch_size // world_size
    
    print(f"\n{'='*60}")
    print(f"Batch Distribution Test - Process {rank}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Batch size per GPU: {batch_size_per_gpu}")
    print(f"  Current GPU: {rank}")
    
    # 创建模拟数据
    print(f"\nCreating simulated batch on GPU {rank}...")
    batch = torch.randn(batch_size_per_gpu, 3, 256, 256, device=device)
    memory_mb = batch.element_size() * batch.nelement() / 1024**2
    print(f"  Batch shape: {list(batch.shape)}")
    print(f"  Memory usage: {memory_mb:.1f} MB")
    print(f"  Device: {batch.device}")
    
    # 模拟前向传播
    print(f"\nSimulating forward pass on GPU {rank}...")
    with torch.no_grad():
        # 简单的卷积操作
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        output = conv(batch)
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Output device: {output.device}")
    
    # 验证内存分配
    allocated_mb = torch.cuda.memory_allocated(rank) / 1024**2
    print(f"\nMemory allocated on GPU {rank}: {allocated_mb:.1f} MB")
    
    dist.destroy_process_group()
    print(f"{'='*60}\n")

def main():
    """主测试函数"""
    
    print("=" * 80)
    print("GPU UTILIZATION AND MEMORY ALLOCATION TEST")
    print("=" * 80)
    
    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"\n🔍 Detected {num_gpus} GPU(s)")
    
    if num_gpus == 0:
        print("❌ No GPUs available. Exiting.")
        return
    
    # 显示GPU信息
    print("\n📊 GPU Configuration:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    CUDA Capability: {props.major}.{props.minor}")
        print(f"    Multiprocessors: {props.multi_processor_count}")
    
    # 显示初始内存状态
    print("\n💾 Initial GPU Memory Status:")
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        for gpu in gpu_info:
            usage_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"  GPU {gpu['index']}: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({usage_pct:.1f}% used)")
    
    if num_gpus >= 2:
        print("\n" + "=" * 80)
        print("TEST 1: MULTI-GPU PROCESS SPAWNING")
        print("=" * 80)
        
        world_size = min(num_gpus, 2)  # 使用最多2个GPU
        print(f"\n🚀 Spawning {world_size} processes for distributed testing...")
        
        try:
            mp.spawn(test_worker, args=(world_size,), nprocs=world_size, join=True)
            print("\n✅ Multi-GPU process spawning test completed successfully")
        except Exception as e:
            print(f"\n❌ Multi-GPU test failed: {e}")
        
        # 测试batch分配
        print("\n" + "=" * 80)
        print("TEST 2: BATCH SIZE DISTRIBUTION")
        print("=" * 80)
        
        try:
            mp.spawn(test_batch_distribution, args=(world_size,), nprocs=world_size, join=True)
            print("\n✅ Batch distribution test completed successfully")
        except Exception as e:
            print(f"\n❌ Batch distribution test failed: {e}")
        
        # 最终内存状态
        print("\n💾 Final GPU Memory Status:")
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu in gpu_info:
                usage_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"  GPU {gpu['index']}: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({usage_pct:.1f}% used)")
        
    else:
        print("\n⚠️ Only 1 GPU available. Skipping distributed tests.")
        print("Running single GPU test...")
        
        device = torch.device('cuda:0')
        print(f"\nTesting on {torch.cuda.get_device_name(0)}")
        
        # 测试内存分配
        test_tensor = torch.randn(8192, 8192, device=device)
        memory_mb = test_tensor.element_size() * test_tensor.nelement() / 1024**2
        print(f"✓ Allocated {memory_mb:.1f} MB tensor on GPU 0")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
