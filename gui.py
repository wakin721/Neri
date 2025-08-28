"""
物种信息检测应用程序
支持图像物种识别、探测图片保存、Excel输出和图像分类功能
现代化桌面应用程序界面 - 优化版本
"""
import sys
import os
import json
import logging
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# --- existing setup_virtual_environment and install_requirements functions ---

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保system文件夹在路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))
sys.path.append(os.path.join(os.path.dirname(__file__)))


# V V V V V V V V V V V V V V V V
# MODIFICATION: Added import for check_for_updates
# V V V V V V V V V V V V V V V V
from system.gui.main_window import ObjectDetectionGUI
from system.config import APP_TITLE
from system.settings_manager import SettingsManager
from system.update_checker import check_for_updates


def main():
    """程序入口点"""
    root = tk.Tk()
    root.withdraw()

    def check_cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    cuda_available = check_cuda_available()
    if not cuda_available:
        messagebox.showwarning("CUDA检测", "未检测到CUDA/Rocm，请检查是否正确安装对应PyTorch版本。", parent=root)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    settings_manager = SettingsManager(base_dir)
    settings = settings_manager.load_settings()

    # Resume logic
    cache_file = os.path.join(settings_manager.settings_dir, "cache.json")
    resume_processing = False
    cache_data = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            if cache_data:
                 resume_processing = messagebox.askyesno(
                    "发现未完成任务",
                    "检测到上次有未完成的处理任务，是否从上次进度继续处理？",
                    parent=root
                 )
                 if not resume_processing:
                     os.remove(cache_file)
                     cache_data = None
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")
            cache_data = None


    # 创建主窗口
    app = ObjectDetectionGUI(
        master=root,
        settings_manager=settings_manager,
        settings=settings,
        resume_processing=resume_processing,
        cache_data=cache_data
        )

    # 显示窗口并启动主循环
    root.deiconify()
    root.mainloop()

if __name__ == "__main__":
    main()
    # --- existing GUI launch logic ---
    '''if '--gui-only' in sys.argv:
        main()
    else:
        python_executable = sys.executable
        if os.name == 'nt':
            gui_executable = os.path.join(os.path.dirname(python_executable), 'pythonw.exe')
            if not os.path.exists(gui_executable):
                gui_executable = python_executable
        else:
            gui_executable = python_executable
        args = [gui_executable, __file__] + sys.argv[1:] + ['--gui-only']
        if os.name == 'nt':
            subprocess.Popen(args, creationflags=0x00000008, close_fds=True)
        else:
            subprocess.Popen(args, close_fds=True)
        sys.exit(0)'''