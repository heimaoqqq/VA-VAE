#!/usr/bin/env python3
"""
Kaggle环境安装和检测脚本
专门为微多普勒VA-VAE项目设计，适配Kaggle环境
"""

import subprocess
import sys
import os
import importlib
import pkg_resources
from packaging import version
import warnings
warnings.filterwarnings('ignore')

def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_step(step, text):
    """打印步骤"""
    print(f"\n🔧 Step {step}: {text}")
    print("-" * 40)

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ 成功: {description}")
            return True, result.stdout
        else:
            print(f"❌ 失败: {description}")
            print(f"错误信息: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时: {description}")
        return False, "Command timeout"
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False, str(e)

def check_python_version():
    """检查Python版本"""
    print_step(1, "检查Python版本")
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 8:
        print("✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("❌ Python版本过低，需要Python 3.8+")
        return False

def check_gpu():
    """检查GPU环境"""
    print_step(2, "检查GPU环境")
    
    # 检查nvidia-smi
    success, output = run_command("nvidia-smi", "检查GPU")
    if success:
        print("✅ GPU可用")
        print("GPU信息:")
        lines = output.split('\n')
        for line in lines:
            if 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'P100' in line or 'T4' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print("⚠️  GPU不可用，将使用CPU训练")
        return False

def install_pytorch():
    """安装PyTorch - 修复torch._C模块问题"""
    print_step(3, "安装/检查PyTorch")

    # 检查当前PyTorch版本
    try:
        import torch
        # 测试关键模块
        import torch._C
        current_version = torch.__version__
        print(f"当前PyTorch版本: {current_version}")

        # 检查CUDA支持
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")

        # 检查版本是否满足要求
        if version.parse(current_version) >= version.parse("2.0.0"):
            print("✅ PyTorch版本符合要求且模块完整")
            return True
        else:
            print("⚠️  PyTorch版本过低，尝试重新安装...")
    except ImportError as e:
        print(f"PyTorch模块缺失: {e}")
        print("🔄 重新安装PyTorch...")
    except Exception as e:
        print(f"PyTorch检查失败: {e}")
        print("🔄 重新安装PyTorch...")

    # 先卸载可能损坏的PyTorch
    print("🗑️  清理现有PyTorch安装...")
    run_command("pip uninstall torch torchvision torchaudio -y", "卸载PyTorch")

    # 安装PyTorch - 使用Kaggle推荐的版本
    print("📦 安装新的PyTorch...")
    cuda_available, _ = run_command("nvidia-smi", "检查CUDA")
    if cuda_available:
        # 使用与Kaggle兼容的CUDA版本
        cmd = "pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121"
    else:
        cmd = "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu"

    success, _ = run_command(cmd, "安装PyTorch")

    # 验证安装
    if success:
        try:
            import torch
            import torch._C  # 验证C++扩展
            print(f"✅ PyTorch {torch.__version__} 安装成功，模块完整")
            return True
        except ImportError as e:
            print(f"❌ PyTorch安装后仍有问题: {e}")
            return False

    return success

def install_pytorch_lightning():
    """安装PyTorch Lightning"""
    print_step(4, "安装/检查PyTorch Lightning")
    
    try:
        import pytorch_lightning as pl
        current_version = pl.__version__
        print(f"当前PyTorch Lightning版本: {current_version}")
        
        if version.parse(current_version) >= version.parse("2.0.0"):
            print("✅ PyTorch Lightning版本符合要求")
            return True
        else:
            print("⚠️  PyTorch Lightning版本过低，尝试升级...")
    except ImportError:
        print("PyTorch Lightning未安装，开始安装...")
    
    success, _ = run_command("pip install pytorch-lightning>=2.0.0", "安装PyTorch Lightning")
    return success

def install_core_packages():
    """安装核心依赖包"""
    print_step(5, "安装核心依赖包")

    # 基础依赖
    basic_packages = [
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.8.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.1.0",
        "safetensors>=0.3.0",  # 用于安全的张量存储
        "accelerate>=0.20.0",  # 用于模型加速
        "transformers>=4.30.0"  # 可能需要的transformer组件
    ]

    success_count = 0
    for package in basic_packages:
        success, _ = run_command(f"pip install {package}", f"安装 {package}")
        if success:
            success_count += 1

    print(f"✅ 成功安装 {success_count}/{len(basic_packages)} 个基础包")
    return success_count == len(basic_packages)

def install_lightningdit_packages():
    """安装LightningDiT特定依赖"""
    print_step(6, "安装LightningDiT特定依赖")

    # LightningDiT特定依赖
    lightningdit_packages = [
        "fairscale>=0.4.13",     # 分布式训练支持
        "einops>=0.6.0",         # 张量操作
        "timm>=0.9.0",           # 图像模型库
        "torchdiffeq>=0.2.3",    # ODE求解器 (transport需要)
        "omegaconf>=2.3.0",      # 配置管理
        "diffusers>=0.20.0",     # 扩散模型库
        "pytorch-fid>=0.3.0",    # FID评估
        "scipy>=1.9.0",          # 科学计算 (transport需要)
        "flash-attn>=2.0.0",     # 高效注意力机制 (可选)
        "triton>=2.0.0",         # GPU内核优化 (可选)
        "xformers>=0.0.20"       # 内存高效的transformer (可选)
    ]

    # 分离必需和可选包
    required_packages = [
        "fairscale>=0.4.13",
        "einops>=0.6.0",
        "timm>=0.9.0",
        "torchdiffeq>=0.2.3",
        "omegaconf>=2.3.0",
        "diffusers>=0.20.0",
        "pytorch-fid>=0.3.0",
        "scipy>=1.9.0"
    ]

    optional_packages = [
        "flash-attn>=2.0.0",
        "triton>=2.0.0",
        "xformers>=0.0.20"
    ]

    # 安装必需包
    print("安装必需的LightningDiT依赖:")
    required_success = 0
    for package in required_packages:
        print(f"\n📦 安装: {package}")
        success, _ = run_command(f"pip install {package}", f"安装 {package}")
        if success:
            required_success += 1
        else:
            print(f"❌ 必需包 {package} 安装失败")

    # 安装可选包
    print("\n安装可选的性能优化包:")
    optional_success = 0
    for package in optional_packages:
        print(f"\n🔧 尝试安装: {package}")

        if "flash-attn" in package:
            success, _ = run_command(f"pip install {package} --no-build-isolation", f"安装 {package}")
        else:
            success, _ = run_command(f"pip install {package}", f"安装 {package}")

        if success:
            optional_success += 1
            print(f"✅ 可选包 {package} 安装成功")
        else:
            print(f"⚠️  可选包 {package} 安装失败，跳过")

    print(f"\n📊 安装结果:")
    print(f"  必需包: {required_success}/{len(required_packages)}")
    print(f"  可选包: {optional_success}/{len(optional_packages)}")

    # 必需包必须全部安装成功
    return required_success == len(required_packages)

def test_lightningdit_imports():
    """测试LightningDiT相关的导入"""
    print_step(8, "测试LightningDiT导入")

    # 测试关键导入
    test_imports = [
        ("safetensors", "from safetensors import safe_open"),
        ("fairscale", "import fairscale"),
        ("einops", "import einops"),
        ("timm", "import timm"),
        ("accelerate", "import accelerate"),
        ("transformers", "import transformers"),
        ("torchdiffeq", "from torchdiffeq import odeint"),
        ("omegaconf", "from omegaconf import OmegaConf"),
        ("diffusers", "import diffusers"),
        ("pytorch_fid", "import pytorch_fid"),
        ("scipy", "import scipy")
    ]

    success_count = 0
    for name, import_statement in test_imports:
        try:
            exec(import_statement)
            print(f"✅ {name}: 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"⚠️  {name}: 导入失败 - {str(e)}")
        except Exception as e:
            print(f"❌ {name}: 测试失败 - {str(e)}")

    # 测试可选的高性能包
    optional_imports = [
        ("flash_attn", "import flash_attn"),
        ("triton", "import triton"),
        ("xformers", "import xformers")
    ]

    optional_success = 0
    for name, import_statement in optional_imports:
        try:
            exec(import_statement)
            print(f"✅ {name}: 导入成功 (性能优化)")
            optional_success += 1
        except ImportError:
            print(f"⚠️  {name}: 未安装 (可选性能优化)")
        except Exception as e:
            print(f"⚠️  {name}: 测试失败 - {str(e)}")

    print(f"\n核心导入: {success_count}/{len(test_imports)}")
    print(f"性能优化: {optional_success}/{len(optional_imports)}")

    return success_count >= 4  # 至少4个核心包导入成功

def check_package_versions():
    """检查包版本兼容性"""
    print_step(7, "检查包版本兼容性")

    required_packages = {
        'torch': '2.0.0',
        'pytorch_lightning': '2.0.0',
        'PIL': '9.0.0',  # Pillow
        'numpy': '1.21.0',
        'tqdm': '4.64.0',
        'tensorboard': '2.8.0',
        'safetensors': '0.3.0',
        'accelerate': '0.20.0',
        'transformers': '4.30.0'
    }

    optional_packages = {
        'fairscale': '0.4.13',
        'einops': '0.6.0',
        'timm': '0.9.0',
        'flash_attn': '2.0.0',
        'triton': '2.0.0',
        'xformers': '0.0.20'
    }
    
    compatible_count = 0
    total_count = len(required_packages)

    print("检查必需包:")
    for package_name, min_version in required_packages.items():
        try:
            if package_name == 'PIL':
                import PIL
                current_version = PIL.__version__
                package_display = 'Pillow'
            else:
                module = importlib.import_module(package_name)
                current_version = module.__version__
                package_display = package_name

            if version.parse(current_version) >= version.parse(min_version):
                print(f"✅ {package_display}: {current_version} (>= {min_version})")
                compatible_count += 1
            else:
                print(f"❌ {package_display}: {current_version} (< {min_version})")
        except ImportError:
            print(f"❌ {package_display}: 未安装")
        except Exception as e:
            print(f"⚠️  {package_display}: 检查失败 ({str(e)})")

    print(f"\n必需包兼容性: {compatible_count}/{total_count}")

    # 检查可选包
    print("\n检查可选包 (LightningDiT优化):")
    optional_count = 0
    for package_name, min_version in optional_packages.items():
        try:
            module = importlib.import_module(package_name)
            current_version = module.__version__

            if version.parse(current_version) >= version.parse(min_version):
                print(f"✅ {package_name}: {current_version} (>= {min_version})")
                optional_count += 1
            else:
                print(f"⚠️  {package_name}: {current_version} (< {min_version})")
        except ImportError:
            print(f"⚠️  {package_name}: 未安装 (可选)")
        except Exception as e:
            print(f"⚠️  {package_name}: 检查失败 ({str(e)})")

    print(f"可选包可用: {optional_count}/{len(optional_packages)}")

    return compatible_count == total_count

def test_imports():
    """测试关键模块导入"""
    print_step(7, "测试关键模块导入")
    
    test_modules = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('tensorboard', 'TensorBoard')
    ]
    
    success_count = 0
    for module_name, display_name in test_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}: 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {str(e)}")
        except Exception as e:
            print(f"⚠️  {display_name}: 导入异常 - {str(e)}")
    
    print(f"\n导入测试: {success_count}/{len(test_modules)} 个模块成功")
    return success_count == len(test_modules)

def test_pytorch_functionality():
    """测试PyTorch功能和完整性"""
    print_step(10, "测试PyTorch功能和完整性")

    try:
        import torch
        import torch.nn as nn
        import torch.distributed as dist

        # 测试关键C++扩展模块
        try:
            import torch._C
            print("✅ torch._C模块: 可用")
        except ImportError:
            print("❌ torch._C模块: 缺失 - PyTorch安装不完整")
            return False

        # 测试基本张量操作
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"✅ 张量运算: {z.shape}")

        # 测试神经网络
        model = nn.Linear(10, 1)
        input_tensor = torch.randn(5, 10)
        output = model(input_tensor)
        print(f"✅ 神经网络: {output.shape}")

        # 测试自动求导
        x = torch.randn(2, 2, requires_grad=True)
        y = x.sum()
        y.backward()
        print(f"✅ 自动求导: 梯度形状 {x.grad.shape}")

        # 测试CUDA（如果可用）
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x_cuda = x.to(device)
            print(f"✅ CUDA运算: 设备 {x_cuda.device}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA不可用，使用CPU")

        # 测试分布式功能（不初始化，只检查模块）
        try:
            from torch.nn.parallel import DistributedDataParallel
            print("✅ 分布式模块: 可用")
        except ImportError:
            print("⚠️  分布式模块: 不可用")

        return True
    except Exception as e:
        print(f"❌ PyTorch功能测试失败: {str(e)}")
        return False

def test_lightning_functionality():
    """测试PyTorch Lightning功能"""
    print_step(9, "测试PyTorch Lightning功能")
    
    try:
        import pytorch_lightning as pl
        import torch
        import torch.nn as nn
        
        # 创建简单的Lightning模块
        class TestModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.layer(x)
            
            def training_step(self, batch, batch_idx):
                return torch.tensor(0.0)
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters())
        
        model = TestModule()
        print("✅ Lightning模块创建成功")
        
        # 测试Trainer创建
        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
        print("✅ Lightning Trainer创建成功")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch Lightning功能测试失败: {str(e)}")
        return False

def create_environment_summary():
    """创建环境总结报告"""
    print_step(10, "生成环境报告")
    
    report_path = "/kaggle/working/environment_report.txt"
    
    try:
        with open(report_path, 'w') as f:
            f.write("Kaggle环境配置报告\n")
            f.write("="*50 + "\n\n")
            
            # Python信息
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"Python路径: {sys.executable}\n\n")
            
            # 包版本信息
            f.write("关键包版本:\n")
            packages = ['torch', 'pytorch_lightning', 'PIL', 'numpy', 'tqdm']
            for pkg in packages:
                try:
                    if pkg == 'PIL':
                        import PIL
                        version_str = PIL.__version__
                    else:
                        module = importlib.import_module(pkg)
                        version_str = module.__version__
                    f.write(f"  {pkg}: {version_str}\n")
                except:
                    f.write(f"  {pkg}: 未安装\n")
            
            # GPU信息
            f.write("\nGPU信息:\n")
            try:
                import torch
                if torch.cuda.is_available():
                    f.write(f"  CUDA可用: True\n")
                    f.write(f"  CUDA版本: {torch.version.cuda}\n")
                    f.write(f"  GPU数量: {torch.cuda.device_count()}\n")
                    for i in range(torch.cuda.device_count()):
                        f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
                else:
                    f.write("  CUDA可用: False\n")
            except:
                f.write("  GPU信息获取失败\n")
        
        print(f"✅ 环境报告已保存到: {report_path}")
        return True
    except Exception as e:
        print(f"❌ 环境报告生成失败: {str(e)}")
        return False

def main():
    """主函数"""
    print_header("Kaggle环境安装和检测脚本")
    print("专为微多普勒VA-VAE项目设计")
    
    # 执行所有检查和安装步骤
    steps = [
        check_python_version,
        check_gpu,
        install_pytorch,
        install_pytorch_lightning,
        install_core_packages,
        install_lightningdit_packages,
        check_package_versions,
        test_imports,
        test_lightningdit_imports,
        test_pytorch_functionality,
        test_lightning_functionality,
        create_environment_summary
    ]
    
    success_count = 0
    for step_func in steps:
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            print(f"❌ 步骤执行失败: {str(e)}")
    
    # 最终总结
    print_header("安装总结")
    print(f"✅ 成功完成: {success_count}/{len(steps)} 个步骤")
    
    if success_count == len(steps):
        print("🎉 环境配置完成！可以开始训练微多普勒VA-VAE模型")
        print("\n下一步:")
        print("1. 运行数据划分: python data_split.py")
        print("2. 开始训练: python minimal_training_modification.py")
    else:
        print("⚠️  部分步骤失败，请检查错误信息")
        print("建议手动安装失败的包或联系技术支持")
    
    print(f"\n详细报告已保存到: /kaggle/working/environment_report.txt")

if __name__ == "__main__":
    main()
