"""
元数据提取模块 - 负责处理图像元数据
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image

from system.config import DATE_FORMATS

logger = logging.getLogger(__name__)

class ImageMetadataExtractor:
    """图像元数据提取器，用于获取图像的EXIF信息"""

    @staticmethod
    def extract_metadata(img_path: str, filename: str) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """提取图像元数据

        Args:
            img_path: 图像文件路径
            filename: 图像文件名

        Returns:
            包含元数据的字典和PIL图像对象
        """
        try:
            img = Image.open(img_path)
            file_type = filename.split('.')[-1].lower()

            image_info = {
                '文件名': filename,
                '格式': file_type,
                '拍摄日期': None,
                '拍摄时间': None,
                '拍摄日期对象': None,
                '工作天数': None,
                '物种名称': '',
                '物种数量': '',
                'detect_results': None,
                '最低置信度': None,
                '独立探测首只': '',
            }

            # 提取EXIF数据
            exif = img._getexif()
            if exif:
                date_taken = ImageMetadataExtractor._get_date_from_exif(exif, filename)
                if date_taken:
                    image_info['拍摄日期'] = date_taken.strftime('%Y-%m-%d')
                    image_info['拍摄时间'] = date_taken.strftime('%H:%M')
                    image_info['拍摄日期对象'] = date_taken

            return image_info, img
        except Exception as e:
            logger.error(f"提取图像元数据失败 ({filename}): {e}")
            return {
                '文件名': filename,
                '格式': filename.split('.')[-1].lower(),
            }, None

    @staticmethod
    def _get_date_from_exif(exif: Dict, filename: str) -> Optional[datetime]:
        """从EXIF数据中提取拍摄日期

        Args:
            exif: EXIF数据字典
            filename: 文件名（用于日志记录）

        Returns:
            日期时间对象或None
        """
        # 尝试从两个可能的EXIF标签中获取日期
        date_str = exif.get(36867) or exif.get(306)
        if not date_str:
            return None

        # 尝试不同的日期格式
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"无法解析图片 '{filename}' 的日期格式: '{date_str}'")
        return None