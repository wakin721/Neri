"""
工具模块 - 提供通用工具函数和辅助类
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径，支持PyInstaller打包"""
    try:
        if getattr(sys, 'frozen', False):  # 是否使用PyInstaller打包
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    except Exception as e:
        logger.error(f"获取资源路径失败: {e}")
        return os.path.join(os.path.abspath("."), relative_path)