import sys
import os
import subprocess
import time


base_path = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(base_path, "requirements.txt")
python_exe_path = f"{base_path}\\toolkit\\python.exe"


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
    if not check_dependencies():
        if not install_dependencies():
            print("\n环境设置失败。请检查以上错误。")
            print("程序将在 15 秒后退出。")
            time.sleep(15)
            sys.exit(1)  # 以错误码退出

    print("\n环境检查完成。正在启动应用程序。。。")
    time.sleep(2)  # 短暂停顿，让用户看到消息
    sys.exit(0)  # 以成功码退出