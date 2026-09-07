"""
工具模块 - 提供通用工具函数和辅助类
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)


def _canonical_resource_alias(relative_path: str) -> str:
    normalized = os.path.normpath(relative_path)
    canonical_root = os.path.normpath(os.path.join('res', 'model'))
    alpha2_root = os.path.normpath(os.path.join('res', 'Model'))
    legacy_cls = os.path.normpath(os.path.join('res', 'model_cls'))
    legacy_tracker = os.path.normpath(os.path.join('res', 'model_cls', 'tracker.yaml'))

    if normalized == legacy_tracker:
        return os.path.join('res', 'model', 'tracker.yaml')

    # Alpha2 used the same canonical tree with an uppercase M. Normalize those
    # references without changing their detect/cls/user/sync structure.
    if normalized == alpha2_root:
        return canonical_root
    if normalized.startswith(alpha2_root + os.sep):
        suffix = normalized[len(alpha2_root):].lstrip(os.sep)
        return os.path.join(canonical_root, suffix)

    if normalized == legacy_cls:
        return os.path.join('res', 'model', 'cls', 'user')
    if normalized.startswith(legacy_cls + os.sep):
        suffix = normalized[len(legacy_cls):].lstrip(os.sep)
        return os.path.join('res', 'model', 'cls', 'user', suffix)

    if normalized == canonical_root:
        return canonical_root
    if normalized.startswith(canonical_root + os.sep):
        suffix = normalized[len(canonical_root):].lstrip(os.sep)
        first_component = suffix.split(os.sep, 1)[0] if suffix else ''
        if (
            first_component in {'detect', 'cls'}
            or suffix in {'tracker.yaml', '.sync-state.json'}
        ):
            return normalized
        # Before model synchronization, detection models lived directly under
        # res/model. Preserve old callers by resolving those files into user/.
        return os.path.join('res', 'model', 'detect', 'user', suffix)

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
