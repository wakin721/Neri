"""
物种信息检测应用程序
支持图像物种识别、探测图片保存、Excel输出和图像分类功能
现代化桌面应用程序界面 - PySide6版本
"""
import sys
import os
import json
import logging

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

# PySide6 导入
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# 应用程序导入
from system.gui.main_window import ObjectDetectionGUI
from system.config import APP_TITLE
from system.settings_manager import SettingsManager
from system.utils import resource_path
# 导入自定义的 Material You 风格弹窗 (请根据你的实际目录结构调整导入路径)
from system.gui.ui_components import MaterialMessageBox


def check_cuda_available():
    """检查CUDA可用性"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def show_cuda_warning():
    """显示CUDA警告"""
    # 使用 MaterialMessageBox 替换 QMessageBox
    MaterialMessageBox.warning(
        None,
        "CUDA检测",
        "未检测到CUDA/ROCm，请检查是否正确安装对应PyTorch版本。\n\n程序将使用CPU模式运行，处理速度可能较慢。"
    )


def ask_resume_processing():
    """询问是否恢复处理"""
    # 使用 MaterialMessageBox 替换 QMessageBox
    result = MaterialMessageBox.question(
        None,
        "发现未完成任务",
        "检测到上次有未完成的处理任务\n\n是否从上次进度继续处理？"
    )
    return result == QMessageBox.StandardButton.Yes


def main():
    """程序入口点"""
    # 创建QApplication实例
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName(APP_TITLE)
    app.setApplicationDisplayName(APP_TITLE)
    app.setOrganizationName("Neri")
    app.setOrganizationDomain("neri.app")

    # 设置应用程序图标
    try:
        icon_path = resource_path(os.path.join("res", "ico.ico"))
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception as e:
        logger.warning(f"无法加载应用程序图标: {e}")

    # 检查CUDA可用性
    cuda_available = check_cuda_available()
    if not cuda_available:
        show_cuda_warning()

    # 初始化设置管理器
    base_dir = os.path.dirname(os.path.abspath(__file__))
    settings_manager = SettingsManager(base_dir)
    settings = settings_manager.load_settings()

    # 检查缓存和恢复逻辑
    cache_file = os.path.join(settings_manager.settings_dir, "cache.json")
    resume_processing = False
    cache_data = None

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            if cache_data:
                resume_processing = ask_resume_processing()
                if not resume_processing:
                    # 用户选择不恢复，删除缓存文件
                    try:
                        os.remove(cache_file)
                        cache_data = None
                    except OSError as e:
                        logger.error(f"删除缓存文件失败: {e}")
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")
            cache_data = None

    # 创建主窗口
    try:
        main_window = ObjectDetectionGUI(
            settings_manager=settings_manager,
            settings=settings,
            resume_processing=resume_processing,
            cache_data=cache_data
        )

        # 显示主窗口
        main_window.show()

        # 启动应用程序事件循环
        return app.exec()

    except Exception as e:
        logger.error(f"启动应用程序失败: {e}")
        # 显示错误消息：使用 MaterialMessageBox 替换 QMessageBox
        MaterialMessageBox.critical(
            None,
            "启动错误",
            f"应用程序启动失败\n\n错误详情：{str(e)}"
        )
        return 1


if __name__ == "__main__":
    # 确保在主线程中运行
    sys.exit(main())