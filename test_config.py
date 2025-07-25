#!/usr/bin/env python3
"""
测试配置文件完整性
确保包含所有官方要求的字段
"""

import yaml
import os

def test_config():
    """测试配置文件是否包含所有必要字段"""
    
    print("🔍 测试配置文件完整性")
    print("=" * 40)
    
    config_file = "inference_config.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("💡 请先运行: python step2_setup_configs.py")
        return False
    
    # 读取配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查必要字段
    required_fields = [
        ('ckpt_path', str),
        ('data.data_path', str),
        ('data.image_size', int),
        ('data.num_classes', int),
        ('data.latent_norm', bool),
        ('vae.model_name', str),
        ('vae.downsample_ratio', int),
        ('model.model_type', str),
        ('model.in_chans', int),
        ('train.global_seed', int),  # 之前缺失
        ('train.max_steps', int),
        ('train.global_batch_size', int),
        ('optimizer.lr', float),
        ('optimizer.beta2', float),
        ('transport.path_type', str),
        ('transport.prediction', str),
        ('sample.mode', str),
        ('sample.sampling_method', str),
        ('sample.num_sampling_steps', int),
        ('sample.cfg_scale', float),
        ('sample.fid_num', int),  # 之前缺失
        ('sample.cfg_interval_start', float),
        ('sample.timestep_shift', float)
    ]
    
    missing_fields = []
    
    for field_path, expected_type in required_fields:
        keys = field_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            
            if not isinstance(current, expected_type):
                print(f"⚠️  {field_path}: 类型错误 (期望{expected_type.__name__}, 实际{type(current).__name__})")
            else:
                print(f"✅ {field_path}: {current}")
                
        except KeyError:
            missing_fields.append(field_path)
            print(f"❌ {field_path}: 缺失")
    
    if missing_fields:
        print(f"\n❌ 发现 {len(missing_fields)} 个缺失字段:")
        for field in missing_fields:
            print(f"   - {field}")
        return False
    else:
        print(f"\n✅ 配置文件完整！包含所有 {len(required_fields)} 个必要字段")
        return True

def main():
    """主函数"""
    if test_config():
        print("\n🎉 配置测试通过！")
        print("🚀 可以运行推理: python step3_run_inference.py")
    else:
        print("\n❌ 配置测试失败！")
        print("🔧 请重新运行: python step2_setup_configs.py")

if __name__ == "__main__":
    main()
