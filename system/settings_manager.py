"""
配置管理模块 - 提供JSON配置的保存和加载功能
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SettingsManager:
    """配置管理器类，用于处理应用程序设置的保存和加载"""

    def __init__(self, base_dir: str = ""):
        """初始化配置管理器

        Args:
            base_dir: 基本目录路径，默认为当前目录
        """
        self.base_dir = base_dir
        self.settings_dir = os.path.join(base_dir, "temp")
        self.settings_file = os.path.join(self.settings_dir, "settings.json")
        self.cache_file = os.path.join(self.settings_dir, "cache.json")

        # 确保设置目录存在
        self._ensure_settings_dir()

    def _ensure_settings_dir(self) -> None:
        """确保设置目录存在"""
        if not os.path.exists(self.settings_dir):
            try:
                os.makedirs(self.settings_dir)
                logger.info(f"创建设置目录: {self.settings_dir}")
            except Exception as e:
                logger.error(f"创建设置目录失败: {e}")

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """保存设置到JSON文件

        Args:
            settings: 设置字典

        Returns:
            保存是否成功
        """
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            logger.info(f"设置已保存到: {self.settings_file}")
            return True
        except Exception as e:
            logger.error(f"保存设置失败: {e}")
            return False

    def load_settings(self) -> Optional[Dict[str, Any]]:
        """从JSON文件加载设置

        Returns:
            设置字典，如果加载失败则返回None
        """
        if not os.path.exists(self.settings_file):
            logger.info(f"设置文件不存在: {self.settings_file}")
            return None

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            logger.info(f"设置已从 {self.settings_file} 加载")
            return settings
        except Exception as e:
            logger.error(f"加载设置失败: {e}")
            return None

    def save_cache(self, cache_data: Dict[str, Any]) -> bool:
        """保存处理缓存到JSON文件

        Args:
            cache_data: 缓存数据字典

        Returns:
            保存是否成功
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)
            logger.info(f"处理缓存已保存到: {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"保存处理缓存失败: {e}")
            return False

    def load_cache(self) -> Optional[Dict[str, Any]]:
        """从JSON文件加载处理缓存

        Returns:
            缓存数据字典，如果加载失败则返回None
        """
        if not os.path.exists(self.cache_file):
            logger.info(f"缓存文件不存在: {self.cache_file}")
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            logger.info(f"处理缓存已从 {self.cache_file} 加载")
            return cache_data
        except Exception as e:
            logger.error(f"加载处理缓存失败: {e}")
            return None

    def delete_cache(self) -> bool:
        """删除处理缓存文件

        Returns:
            删除是否成功
        """
        if not os.path.exists(self.cache_file):
            logger.info(f"缓存文件不存在，无需删除: {self.cache_file}")
            return True

        try:
            os.remove(self.cache_file)
            logger.info(f"处理缓存文件已删除: {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"删除处理缓存文件失败: {e}")
            return False

    def has_cache(self) -> bool:
        """检查是否存在处理缓存文件

        Returns:
            是否存在缓存文件
        """
        return os.path.exists(self.cache_file)

    def get_setting(self, key: str, default=None):
        """获取单个设置项的值

        Args:
            key: 设置项键名
            default: 如果设置项不存在时返回的默认值

        Returns:
            设置项值，如果不存在则返回默认值
        """
        # 加载设置
        settings = self.load_settings()

        # 如果设置存在，返回对应值，否则返回默认值
        if settings and key in settings:
            return settings[key]
        return default

    def load_confidence_settings(self) -> Dict[str, float]:
        """从conf.json加载物种置信度设置"""
        conf_file = os.path.join(self.settings_dir, "conf.json")
        if not os.path.exists(conf_file):
            return {}
        try:
            with open(conf_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载置信度配置文件失败: {e}")
            return {}

    def save_confidence_settings(self, conf_data: Dict[str, float]) -> bool:
        """保存物种置信度设置到conf.json"""
        conf_file = os.path.join(self.settings_dir, "conf.json")
        try:
            with open(conf_file, 'w', encoding='utf-8') as f:
                json.dump(conf_data, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            logger.error(f"保存置信度配置文件失败: {e}")
            return False