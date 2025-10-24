import sys
import os
import subprocess
import time
import re

base_path = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(base_path, "requirements.txt")
python_exe_path = f"{base_path}\\toolkit\\python.exe"


def get_cuda_version():
    """通过nvidia-smi获取CUDA版本。"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            encoding='gbk',
            errors='ignore',
            timeout=10
        )

        if result.returncode == 0:
            # 从nvidia-smi输出中提取CUDA版本
            # 查找类似 "CUDA Version: 12.1" 的文本
            match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
            if match:
                major, minor = match.groups()
                cuda_version = f"{major}.{minor}"
                print(f"检测到CUDA版本: {cuda_version}")
                return cuda_version

        print("未检测到NVIDIA GPU或CUDA")
        return None

    except FileNotFoundError:
        print("未检测到NVIDIA GPU，将使用CPU运行模型")
        return None
    except Exception as e:
        print(f"检测CUDA版本时出错: {e}")
        return None


def get_pytorch_install_command(cuda_version):
    """根据CUDA版本返回对应的PyTorch安装命令。"""
    if cuda_version is None:
        # CPU版本
        print("将安装CPU版本的PyTorch")
        return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    # 解析CUDA版本
    try:
        major, minor = cuda_version.split('.')
        cuda_major = int(major)
        cuda_minor = int(minor)

        # 根据CUDA版本选择对应的PyTorch版本（从新到旧）
        if cuda_major >= 13:
            # CUDA 13.0+
            print("安装CUDA 13.0版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu130"]
        elif cuda_major >= 12 and cuda_minor >= 8:
            # CUDA 12.8
            print("安装CUDA 12.8版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu128"]
        elif cuda_major >= 12 and cuda_minor >= 6:
            # CUDA 12.6
            print("安装CUDA 12.6版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu126"]
        elif cuda_major >= 12 and cuda_minor >= 4:
            # CUDA 12.4
            print("安装CUDA 12.4版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu124"]
        elif cuda_major >= 12 and cuda_minor >= 1:
            # CUDA 12.1
            print("安装CUDA 12.1版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu121"]
        elif cuda_major >= 11 and cuda_minor >= 8:
            # CUDA 11.8
            print("安装CUDA 11.8版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                    "https://download.pytorch.org/whl/cu118"]
        else:
            # 旧版本CUDA，使用CPU版本
            print(f"CUDA版本 {cuda_version} 过旧，将安装CPU版本的PyTorch")
            return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    except Exception as e:
        print(f"解析CUDA版本时出错: {e}，将使用CPU版本")
        return [python_exe_path, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]


def is_pytorch_installed():
    """检查PyTorch是否已安装。"""
    try:
        import torch
        print(f"检测到已安装的PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        return False


def install_pytorch():
    """检测CUDA版本并安装对应的PyTorch。"""
    print("\n==============================================")
    print("正在检测GPU和CUDA版本。。。")
    print("==============================================")

    # 检查是否已安装PyTorch
    if is_pytorch_installed():
        print("PyTorch已安装，跳过安装步骤")
        return True

    # 获取CUDA版本
    cuda_version = get_cuda_version()

    # 获取对应的安装命令
    install_command = get_pytorch_install_command(cuda_version)

    print("\n==============================================")
    print("正在安装最新版本的PyTorch。。。")
    print("这可能需要几分钟。。。")
    print("==============================================")

    try:
        process = subprocess.Popen(
            install_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='gbk',
            errors='ignore'
        )

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc == 0:
            print("\nPyTorch安装成功")
            # 验证安装
            if is_pytorch_installed():
                return True
            else:
                print("警告：PyTorch安装完成但无法导入")
                return False
        else:
            print(f"\nPyTorch安装失败，退出代码： {rc}")
            return False

    except Exception as e:
        print(f"\nPyTorch安装过程中出现错误：{e}")
        return False


def check_dependencies():
    """检查所有依赖是否已安装。"""
    print("==============================================")
    print("正在检查依赖。。。")
    print("==============================================")

    if not os.path.exists(requirements_path):
        print(f"错误：在{requirements_path}未找到 requirements.txt")
        return False

    try:
        import pkg_resources
        with open(requirements_path, "r", encoding="utf-8") as f:
            # 过滤掉注释和空行
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # 检查每个包
        pkg_resources.require(requirements)

        print("\n所有依赖已安装，程序即将启动。。。")
        return True
    except (ImportError, pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
        print(f"\n检测到缺失或冲突的依赖项: {e}")
        return False


def install_dependencies():
    """安装所有依赖。"""
    print("\n==============================================")
    print("正在安装/更新依赖项。。。")
    print("可能需要几分钟。。。")
    print("==============================================")

    # 使用模块方式调用 pip，确保可移植性
    command = [python_exe_path, "-m", "pip", "install", "-r", requirements_path, "--upgrade"]

    try:
        # 实时显示 pip 的输出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='gbk',  # 使用GBK编码来适应中文Windows命令行
            errors='ignore'  # 忽略无法解码的字符
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc == 0:
            print("\n依赖已成功安装。")
            return True
        else:
            print(f"\n依赖项安装失败，退出代码： {rc}")
            return False

    except Exception as e:
        print(f"\n依赖安装过程中出现错误：{e}")
        return False


if __name__ == "__main__":
    # 首先安装PyTorch（根据CUDA版本）
    if not is_pytorch_installed():
        if not install_pytorch():
            print("\nPyTorch安装失败。请检查以上错误。")
            print("程序将在 15 秒后退出。")
            time.sleep(15)
            sys.exit(1)

    # 然后检查并安装其他依赖
    if not check_dependencies():
        if not install_dependencies():
            print("\n环境设置失败。请检查以上错误。")
            print("程序将在 15 秒后退出。")
            time.sleep(15)
            sys.exit(1)

    print("\n环境检查完成。正在启动应用程序。。。")
    time.sleep(2)  # 短暂停顿，让用户看到消息
    sys.exit(0)  # 以成功码退出