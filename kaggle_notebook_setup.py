"""
Kaggle Notebook环境安装脚本
在Kaggle Notebook中运行此脚本来配置微多普勒VA-VAE项目环境
"""

# Cell 1: 环境检查和基础安装
def setup_environment():
    import subprocess
    import sys
    import os
    
    print("🔧 开始配置Kaggle环境...")
    print("="*50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU可用")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if any(gpu in line for gpu in ['Tesla', 'RTX', 'GTX', 'P100', 'T4', 'V100']):
                    print(f"GPU: {line.strip()}")
        else:
            print("⚠️ GPU不可用")
    except:
        print("⚠️ 无法检查GPU状态")
    
    # 检查当前工作目录
    print(f"当前目录: {os.getcwd()}")
    print(f"可用磁盘空间: {subprocess.getoutput('df -h /kaggle/working | tail -1')}")

# 运行环境检查
setup_environment()

# Cell 2: 安装PyTorch和相关包
def install_packages():
    import subprocess
    import sys
    
    print("🔧 安装必要的包...")
    print("="*50)
    
    # 包列表
    packages = [
        "pytorch-lightning>=2.0.0",
        "pillow>=9.0.0", 
        "packaging>=21.0",
        "tensorboard>=2.8.0"
    ]
    
    # 检查并安装包
    for package in packages:
        print(f"\n安装 {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {package} 安装成功")
            else:
                print(f"❌ {package} 安装失败")
                print(f"错误: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {package} 安装超时")
        except Exception as e:
            print(f"❌ {package} 安装异常: {e}")

# 运行包安装
install_packages()

# Cell 3: 验证安装
def verify_installation():
    print("🔍 验证安装...")
    print("="*50)
    
    # 测试导入
    test_packages = {
        'torch': 'PyTorch',
        'pytorch_lightning': 'PyTorch Lightning', 
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard'
    }
    
    success_count = 0
    for module_name, display_name in test_packages.items():
        try:
            if module_name == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'Unknown')
            
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError:
            print(f"❌ {display_name}: 导入失败")
        except Exception as e:
            print(f"⚠️ {display_name}: {str(e)}")
    
    print(f"\n导入测试: {success_count}/{len(test_packages)} 成功")
    
    # 测试PyTorch功能
    try:
        import torch
        print(f"\n🔥 PyTorch功能测试:")
        print(f"  版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 简单张量测试
        x = torch.randn(2, 3)
        y = torch.randn(3, 4) 
        z = torch.mm(x, y)
        print(f"  张量运算: ✅ {z.shape}")
        
        # CUDA测试
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            print(f"  CUDA运算: ✅ 设备 {x_cuda.device}")
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
    
    # 测试PyTorch Lightning
    try:
        import pytorch_lightning as pl
        print(f"\n⚡ PyTorch Lightning功能测试:")
        print(f"  版本: {pl.__version__}")
        
        # 创建简单的trainer
        trainer = pl.Trainer(
            max_epochs=1, 
            enable_progress_bar=False, 
            logger=False,
            enable_checkpointing=False
        )
        print(f"  Trainer创建: ✅")
        
    except Exception as e:
        print(f"❌ PyTorch Lightning测试失败: {e}")
    
    return success_count == len(test_packages)

# 运行验证
verification_success = verify_installation()

# Cell 4: 克隆项目
def clone_project():
    import subprocess
    import os
    
    print("📥 克隆VA-VAE项目...")
    print("="*50)
    
    # 切换到工作目录
    os.chdir('/kaggle/working')
    
    # 检查项目是否已存在
    if os.path.exists('VA-VAE'):
        print("⚠️ 项目已存在，删除旧版本...")
        subprocess.run(['rm', '-rf', 'VA-VAE'])
    
    # 克隆项目
    try:
        result = subprocess.run([
            'git', 'clone', 'https://github.com/heimaoqqq/VA-VAE.git'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ 项目克隆成功")
            
            # 切换到项目目录
            os.chdir('VA-VAE')
            print(f"当前目录: {os.getcwd()}")
            
            # 列出项目文件
            print("\n项目文件:")
            files = os.listdir('.')
            for file in sorted(files):
                if os.path.isfile(file):
                    print(f"  📄 {file}")
                else:
                    print(f"  📁 {file}/")
            
            return True
        else:
            print(f"❌ 项目克隆失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 克隆超时")
        return False
    except Exception as e:
        print(f"❌ 克隆异常: {e}")
        return False

# 运行项目克隆
if verification_success:
    clone_success = clone_project()
else:
    print("⚠️ 环境验证失败，跳过项目克隆")
    clone_success = False

# Cell 5: 生成最终报告
def generate_final_report():
    import os
    import sys
    import subprocess
    from datetime import datetime
    
    print("📋 生成环境报告...")
    print("="*50)
    
    report_content = f"""
Kaggle环境配置报告
{'='*50}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Python环境:
  版本: {sys.version}
  路径: {sys.executable}

工作目录: {os.getcwd()}

包版本信息:
"""
    
    # 获取包版本
    packages = ['torch', 'pytorch_lightning', 'PIL', 'numpy', 'tqdm', 'tensorboard']
    for pkg in packages:
        try:
            if pkg == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'Unknown')
            report_content += f"  {pkg}: {version}\n"
        except:
            report_content += f"  {pkg}: 未安装\n"
    
    # GPU信息
    try:
        import torch
        report_content += f"\nGPU信息:\n"
        report_content += f"  CUDA可用: {torch.cuda.is_available()}\n"
        if torch.cuda.is_available():
            report_content += f"  CUDA版本: {torch.version.cuda}\n"
            report_content += f"  GPU数量: {torch.cuda.device_count()}\n"
            for i in range(torch.cuda.device_count()):
                report_content += f"  GPU {i}: {torch.cuda.get_device_name(i)}\n"
    except:
        report_content += "\nGPU信息: 获取失败\n"
    
    # 磁盘空间
    try:
        disk_info = subprocess.getoutput('df -h /kaggle/working')
        report_content += f"\n磁盘空间:\n{disk_info}\n"
    except:
        report_content += "\n磁盘空间: 获取失败\n"
    
    # 保存报告
    report_path = '/kaggle/working/environment_report.txt'
    try:
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"✅ 报告已保存到: {report_path}")
    except Exception as e:
        print(f"❌ 报告保存失败: {e}")
    
    # 显示报告
    print("\n" + report_content)
    
    return report_content

# 生成最终报告
final_report = generate_final_report()

# Cell 6: 最终总结和下一步指导
def final_summary():
    print("🎉 环境配置完成总结")
    print("="*50)
    
    if verification_success and clone_success:
        print("✅ 环境配置成功！")
        print("\n📋 下一步操作:")
        print("1. 数据划分:")
        print("   python data_split.py \\")
        print("     --input_dir /kaggle/input/dataset \\")
        print("     --output_dir /kaggle/working/data_split \\")
        print("     --train_ratio 0.8 --val_ratio 0.2")
        print()
        print("2. 快速验证训练:")
        print("   python minimal_training_modification.py \\")
        print("     --data_dir /kaggle/working/data_split \\")
        print("     --original_vavae /path/to/vavae.pth \\")
        print("     --batch_size 8 --max_epochs 2 --devices 1")
        print()
        print("3. 正式训练:")
        print("   python minimal_training_modification.py \\")
        print("     --data_dir /kaggle/working/data_split \\")
        print("     --original_vavae /path/to/vavae.pth \\")
        print("     --batch_size 16 --max_epochs 100 \\")
        print("     --devices 1 --precision 16")
        print()
        print("📄 详细报告: /kaggle/working/environment_report.txt")
        
    else:
        print("❌ 环境配置失败")
        if not verification_success:
            print("  - 包安装验证失败")
        if not clone_success:
            print("  - 项目克隆失败")
        print("\n请检查错误信息并手动解决问题")

# 运行最终总结
final_summary()
