"""
工具模块 - 提供通用工具函数和辅助类
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)


def _canonical_resource_alias(relative_path: str) -> str:
    normalized = os.path.normpath(relative_path)
    legacy_tracker = os.path.normpath(os.path.join('res', 'model_cls', 'tracker.yaml'))
    legacy_detect = os.path.normpath(os.path.join('res', 'model'))
    legacy_cls = os.path.normpath(os.path.join('res', 'model_cls'))

    if normalized == legacy_tracker:
        return os.path.join('res', 'Model', 'tracker.yaml')
    if normalized == legacy_detect or normalized.startswith(legacy_detect + os.sep):
        suffix = normalized[len(legacy_detect):].lstrip(os.sep)
        return os.path.join('res', 'Model', 'detect', suffix) if suffix else os.path.join('res', 'Model', 'detect')
    if normalized == legacy_cls or normalized.startswith(legacy_cls + os.sep):
        suffix = normalized[len(legacy_cls):].lstrip(os.sep)
        return os.path.join('res', 'Model', 'cls', suffix) if suffix else os.path.join('res', 'Model', 'cls')
    return relative_path


def resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径，支持PyInstaller打包。"""
    relative_path = _canonical_resource_alias(relative_path)
    try:
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath('.')
        return os.path.join(base_path, relative_path)
    except Exception as e:
        logger.error(f"获取资源路径失败: {e}")
        return os.path.join(os.path.abspath('.'), relative_path)
