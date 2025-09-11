import os
import sys
import platform
import logging
import threading
import json
import time
from datetime import datetime
import gc
import hashlib
import shutil
from PIL import Image
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMessageBox, QFileDialog, QApplication, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QIcon, QPixmap, QPalette

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS
from system.utils import resource_path
from system.image_processor import ImageProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.data_processor import DataProcessor
from system.settings_manager import SettingsManager
from system.update_checker import check_for_updates, get_latest_version_info, compare_versions, start_download_thread, \
    _show_messagebox

# Import GUI components
from system.gui.sidebar import Sidebar
from system.gui.start_page import StartPage
from system.gui.preview_page import PreviewPage
from system.gui.species_validation_page import SpeciesValidationPage
from system.gui.advanced_page import AdvancedPage
from system.gui.about_page import AboutPage
from system.gui.ui_components import InfoBar, Win11Colors, ThemeManager

if platform.system() == "Windows":
    import ctypes

logger = logging.getLogger(__name__)


class UpdateCheckThread(QThread):
    """更新检查线程"""
    update_found = Signal(str)  # 发现更新信号
    check_complete = Signal(bool)  # 检查完成信号

    def __init__(self, channel='stable', silent=True):
        super().__init__()
        self.channel = channel
        self.silent = silent

    def run(self):
        try:
            latest_info = get_latest_version_info(self.channel)
            if latest_info:
                remote_version = latest_info['version']
                if compare_versions(APP_VERSION, remote_version):
                    self.update_found.emit(remote_version)
            self.check_complete.emit(True)
        except Exception as e:
            logger.error(f"检查更新失败: {e}")
            self.check_complete.emit(False)


class ProcessingThread(QThread):
    """图像处理线程"""
    progress_updated = Signal(int, int, float, object)
    file_processed = Signal(str, object, str)
    processing_complete = Signal(bool)
    status_message = Signal(str)
    file_processing_result = Signal(str, dict)
    current_file_changed = Signal(str, int, int)
    current_file_preview = Signal(str, dict)

    def __init__(self, controller, file_path, save_path, use_fp16, resume_from=0):
        super().__init__()
        self.controller = controller
        self.file_path = file_path
        self.save_path = save_path
        self.use_fp16 = use_fp16
        self.resume_from = resume_from
        self.stop_flag = False

    def stop(self):
        """停止处理线程"""
        self.stop_flag = True

    def run(self):
        start_time = time.time()
        excel_data = [] if self.resume_from == 0 else self.controller.excel_data
        processed_files = self.resume_from
        stopped_manually = False
        earliest_date = None
        temp_photo_dir = self.controller.get_temp_photo_dir()

        try:
            iou = self.controller.advanced_page.iou_var
            conf = self.controller.advanced_page.conf_var
            augment = self.controller.advanced_page.use_augment_var
            agnostic_nms = self.controller.advanced_page.use_agnostic_nms_var

            image_files = sorted([f for f in os.listdir(self.file_path)
                                  if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)])
            total_files = len(image_files)

            if self.resume_from > 0:
                image_files = image_files[self.resume_from:]
                if excel_data:
                    valid_dates = [item['拍摄日期对象'] for item in excel_data if item.get('拍摄日期对象')]
                    if valid_dates:
                        earliest_date = min(valid_dates)

            for idx, filename in enumerate(image_files):
                if self.stop_flag:
                    stopped_manually = True
                    break

                current_index = processed_files + 1
                img_path = os.path.join(self.file_path, filename)

                # 发射当前处理文件变化信号
                self.current_file_changed.emit(img_path, current_index, total_files)

                self.status_message.emit(f"正在处理: {filename}")

                elapsed_time = time.time() - start_time
                speed = (processed_files - self.resume_from + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_time = (total_files - (processed_files + 1)) / speed if speed > 0 else float('inf')

                self.progress_updated.emit(processed_files + 1, total_files, speed, remaining_time)

                try:
                    image_info, img = ImageMetadataExtractor.extract_metadata(img_path, filename)
                    species_info = self.controller.image_processor.detect_species(
                        img_path, bool(self.use_fp16), iou, conf, augment, agnostic_nms
                    )
                    species_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detect_results = species_info.get('detect_results')

                    if detect_results:
                        self.controller.image_processor.save_detection_info_json(
                            detect_results, filename, species_info, temp_photo_dir
                        )
                        # 发射文件处理完成信号
                        self.file_processed.emit(img_path, detect_results, filename)

                        # 发射当前文件预览信号，包含完整检测结果
                        complete_detection_info = {
                            **species_info,
                            'detect_results': detect_results,
                            'filename': filename
                        }
                        self.current_file_preview.emit(img_path, complete_detection_info)

                    if 'detect_results' in species_info:
                        del species_info['detect_results']
                    image_info.update(species_info)
                    excel_data.append(image_info)

                except Exception as e:
                    logger.error(f"处理文件 {filename} 失败: {e}")

                processed_files += 1

                if processed_files % 10 == 0:
                    self._save_processing_cache(excel_data, processed_files, total_files)

                # 清理内存
                try:
                    del img_path, image_info, img, species_info, detect_results
                except NameError:
                    pass
                gc.collect()

            if not stopped_manually:
                self.progress_updated.emit(total_files, total_files, 0, "已完成")
                self.controller.excel_data = excel_data
                excel_data = DataProcessor.process_independent_detection(excel_data,
                                                                         self.controller.confidence_settings)
                if earliest_date:
                    excel_data = DataProcessor.calculate_working_days(excel_data, earliest_date)
                self._delete_processing_cache()
                self.status_message.emit("处理完成！")
                QTimer.singleShot(0, lambda: QMessageBox.information(None, "成功", "图像处理完成！"))

            self.processing_complete.emit(not stopped_manually)

        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            QTimer.singleShot(0, lambda: QMessageBox.critical(None, "错误", f"处理过程中发生错误: {e}"))
            self.processing_complete.emit(False)
        finally:
            gc.collect()

    def _save_processing_cache(self, excel_data, processed_files, total_files):
        def make_serializable(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if k != 'detect_results'}
            if isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            if hasattr(obj, '__dict__'):
                return None
            return obj

        serializable_excel_data = make_serializable(excel_data)
        cache_data = {
            'file_path': self.file_path,
            'save_path': self.save_path,
            'use_fp16': self.use_fp16,
            'processed_files': processed_files,
            'total_files': total_files,
            'excel_data': serializable_excel_data,
        }
        cache_file = os.path.join(self.controller.settings_manager.settings_dir, "cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _delete_processing_cache(self):
        cache_file = os.path.join(self.controller.settings_manager.settings_dir, "cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)


class ObjectDetectionGUI(QMainWindow):
    """主应用程序窗口 - PySide6版本"""

    def __init__(self, settings_manager: SettingsManager, settings: dict, resume_processing: bool, cache_data: dict):
        super().__init__()
        self.settings_manager = settings_manager
        self.settings = settings
        self.resume_processing = resume_processing
        self.cache_data = cache_data
        self.current_temp_photo_dir = None
        self.is_dark_mode = False
        self.accent_color = Win11Colors.LIGHT_ACCENT  # 默认使用浅色模式的颜色

        # 检查CUDA可用性
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            self.cuda_available = False

        self.is_processing = False
        self.processing_thread = None
        self.excel_data = []
        self.current_page = "settings"
        self.update_channel_var = "稳定版 (Release)"
        self.model_var = ""
        self.confidence_settings = self.settings_manager.load_confidence_settings()

        # 新增：用于存储最新进度信息
        self.last_progress_value = 0
        self.last_progress_total = 0
        self.last_progress_speed = 0.0
        self.last_progress_remaining_time = float('inf')
        self.current_processing_file = None

        # 主题检测和应用
        self._apply_system_theme()
        self._setup_window()
        self._initialize_model(settings)
        self._create_ui_elements()
        self._setup_connections()

	# 在窗口初始化后设置标题栏颜色
        self._set_title_bar_color()

        # 加载设置
        if self.settings:
            self._load_settings_to_ui(self.settings)
        else:
            self._create_default_settings()

        # 设置定时器进行启动后的初始化
        QTimer.singleShot(100, self._post_init)

    def _apply_system_theme(self):
        """应用系统主题"""
        try:
            import darkdetect
            system_theme = darkdetect.theme().lower()
            if system_theme == 'dark':
                self.is_dark_mode = True
                self.accent_color = Win11Colors.DARK_ACCENT
            else:
                self.is_dark_mode = False
                self.accent_color = Win11Colors.LIGHT_ACCENT
        except Exception as e:
            self.is_dark_mode = False
            self.accent_color = Win11Colors.LIGHT_ACCENT
            logger.warning(f"无法检测系统主题: {e}")

        # 应用Win11样式
        ThemeManager.apply_win11_style(QApplication.instance())

    def _setup_window(self):
        """设置窗口"""
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1100, 750)
        self.resize(1100, 750)

        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        window_rect = self.geometry()
        x = (screen.width() - window_rect.width()) // 2
        y = (screen.height() - window_rect.height()) // 2
        self.move(x, y)

        # 设置图标
        try:
            ico_path = resource_path("res/ico.ico")
            self.setWindowIcon(QIcon(ico_path))
            # --- 为Windows设置任务栏图标 ---
            if platform.system() == "Windows":
                # 使用APP_TITLE和APP_VERSION创建一个更独特的ID
                myappid = f'mycompany.{APP_TITLE}.{APP_VERSION}'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            # ---------------------------------
        except Exception as e:
            logger.warning(f"无法加载窗口图标: {e}")

    def _initialize_model(self, settings: dict):
        """初始化模型"""
        saved_model_name = settings.get("selected_model") if settings else None
        model_path = None
        res_dir = resource_path("res")

        # 尝试从设置中加载模型
        if saved_model_name:
            potential_path = os.path.join(res_dir, saved_model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                logger.info(f"从设置加载模型: {saved_model_name}")
            else:
                logger.warning(f"设置中保存的模型文件不存在: {saved_model_name}。将尝试加载默认模型。")

        # 如果设置中没有模型或文件不存在，则查找第一个可用的模型
        if not model_path:
            model_path = self._find_model_file()
            if model_path:
                logger.info(f"加载找到的第一个模型: {os.path.basename(model_path)}")

        # 初始化 ImageProcessor
        self.image_processor = ImageProcessor(model_path)
        if model_path:
            self.image_processor.model_path = model_path
            self.model_var = os.path.basename(model_path)
        else:
            self.image_processor.model = None
            self.image_processor.model_path = None
            self.model_var = ""
            logger.error("在 res 目录中未找到任何有效的模型文件 (.pt)。")

    def _find_model_file(self) -> str:
        """查找模型文件"""
        try:
            res_dir = resource_path("res")
            if not os.path.exists(res_dir) or not os.path.isdir(res_dir):
                return None
            model_files = [f for f in os.listdir(res_dir) if f.lower().endswith('.pt')]
            if not model_files:
                return None
            return os.path.join(res_dir, model_files[0])
        except Exception as e:
            logger.error(f"查找模型文件时出错: {e}")
            return None

    def _create_ui_elements(self):
        """创建UI元素"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 创建侧边栏
        self.sidebar = Sidebar(self)
        splitter.addWidget(self.sidebar)

        # 创建内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 创建页面堆栈
        self.content_stack = QStackedWidget()
        content_layout.addWidget(self.content_stack)

        # 创建状态栏
        self.status_bar = InfoBar()
        content_layout.addWidget(self.status_bar)

        splitter.addWidget(content_widget)

        # 设置分割器比例
        splitter.setStretchFactor(0, 0)  # 侧边栏不拉伸
        splitter.setStretchFactor(1, 1)  # 内容区域拉伸

        # 创建页面
        self.start_page = StartPage(self)
        self.preview_page = PreviewPage(self, self)
        self.species_validation_page = SpeciesValidationPage(self)
        self.advanced_page = AdvancedPage(self, self)
        self.about_page = AboutPage(self)

        # 添加页面到堆栈
        self.content_stack.addWidget(self.start_page)
        self.content_stack.addWidget(self.preview_page)
        self.content_stack.addWidget(self.species_validation_page)
        self.content_stack.addWidget(self.advanced_page)
        self.content_stack.addWidget(self.about_page)

        # 显示默认页面
        self._show_page("settings")

    def _setup_connections(self):
        """设置信号连接"""
        # 侧边栏导航
        self.sidebar.page_requested.connect(self._show_page)

        # 开始页面
        self.start_page.browse_file_path_requested.connect(self.browse_file_path)
        self.start_page.browse_save_path_requested.connect(self.browse_save_path)
        self.start_page.file_path_changed.connect(self._validate_and_update_file_path)
        self.start_page.save_path_changed.connect(self._validate_and_update_save_path)
        self.start_page.toggle_processing_requested.connect(self.toggle_processing_state)
        self.start_page.settings_changed.connect(self._save_current_settings)

        # 高级页面
        if hasattr(self.advanced_page, 'settings_changed'):
            self.advanced_page.settings_changed.connect(self._save_current_settings)
        self.advanced_page.update_check_requested.connect(self.check_for_updates_from_ui)
        self.advanced_page.theme_changed.connect(self.change_theme)
        self.advanced_page.params_help_requested.connect(self.show_params_help)
        self.advanced_page.cache_clear_requested.connect(self.clear_image_cache)
        self.advanced_page.settings_changed.connect(self._save_current_settings)

        # 预览页面
        self.preview_page.settings_changed.connect(self._save_current_settings)

        # 物种校验页面
        self.species_validation_page.settings_changed.connect(self._save_current_settings)
        self.species_validation_page.quick_marks_updated.connect(
            self.advanced_page.load_quick_mark_settings)

    def _post_init(self):
        """后期初始化"""
        # 检查模型
        if not self.image_processor.model:
            QMessageBox.critical(
                self, "错误",
                "未找到有效的模型文件(.pt)。请在res目录中放入至少一个模型文件。"
            )
            if hasattr(self.start_page, 'set_processing_enabled'):
                self.start_page.set_processing_enabled(False)

        # 设置主题监控
        self.setup_theme_monitoring()

        # 加载验证数据
        if hasattr(self.preview_page, '_load_validation_data'):
            self.preview_page._load_validation_data()

        # 检查更新
        self._check_for_updates(silent=True)

        # 恢复处理
        if self.resume_processing and self.cache_data:
            QTimer.singleShot(1000, self._resume_processing)

    def _create_default_settings(self):
        """创建默认设置"""
        logger.info("未找到配置文件，正在使用默认值创建 'setting.json'。")
        default_settings = self._get_current_settings()
        self.settings_manager.save_settings(default_settings)
        self.settings = default_settings
        self.change_theme()

    def _check_for_updates(self, silent=False):
        """检查更新"""

        def on_update_found(version):
            if hasattr(self.sidebar, 'show_update_notification'):
                self.sidebar.show_update_notification()

        def on_check_complete(success):
            if not silent and not success:
                QMessageBox.warning(self, "更新错误", "检查更新失败，请稍后重试。")

        channel_selection = self.update_channel_var
        channel = 'preview' if '预览版' in channel_selection else 'stable'

        self.update_thread = UpdateCheckThread(channel, silent)
        self.update_thread.update_found.connect(on_update_found)
        self.update_thread.check_complete.connect(on_check_complete)
        self.update_thread.start()

    def check_for_updates_from_ui(self):
        """从UI手动检查更新"""
        channel_selection = self.update_channel_var
        channel = 'preview' if '预览版' in channel_selection else 'stable'

        def on_update_found(version):
            if hasattr(self.sidebar, 'show_update_notification'):
                self.sidebar.show_update_notification()
            update_message = f"发现新版本: {version}\n\n是否立即下载并安装？"
            reply = QMessageBox.question(
                self, "发现新版本", update_message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # 这里需要实现下载逻辑
                pass

        def on_check_complete(success):
            if success:
                # 如果没有发现更新
                QMessageBox.information(self, "无更新", "您目前使用的是最新版本。")
            else:
                QMessageBox.critical(self, "更新错误", "检查更新时发生错误。")

        self.update_thread = UpdateCheckThread(channel, False)
        self.update_thread.update_found.connect(on_update_found)
        self.update_thread.check_complete.connect(on_check_complete)
        self.update_thread.start()

    def change_theme(self):
        """更改主题"""
        if hasattr(self.advanced_page, 'get_theme_selection'):
            selected_theme = self.advanced_page.get_theme_selection()
        else:
            selected_theme = "自动"

        if selected_theme == "自动":
            self._apply_system_theme()
        elif selected_theme == "深色":
            self.is_dark_mode = True
            self.accent_color = Win11Colors.DARK_ACCENT
        else:  # "浅色"
            self.is_dark_mode = False
            self.accent_color = Win11Colors.LIGHT_ACCENT

        # 应用主题
        ThemeManager.apply_win11_style(QApplication.instance())
        # 在主题更改后设置标题栏颜色
        self._set_title_bar_color()
        self._save_current_settings()

    def setup_theme_monitoring(self):
        """设置主题监控"""
        if platform.system() in ["Windows", "Darwin"]:
            self.theme_timer = QTimer()
            self.theme_timer.timeout.connect(self._check_theme_change)
            self.theme_timer.start(10000)  # 每10秒检查一次

    def _check_theme_change(self):
        """检查主题变化"""
        try:
            if hasattr(self.advanced_page, 'get_theme_selection') and \
                    self.advanced_page.get_theme_selection() == "自动":
                import darkdetect
                current_theme = darkdetect.theme().lower()
                if (current_theme == 'dark' and not self.is_dark_mode) or \
                        (current_theme == 'light' and self.is_dark_mode):
                    self._apply_system_theme()
                    ThemeManager.apply_win11_style(QApplication.instance())
        except Exception as e:
            logger.warning(f"检查主题变化失败: {e}")

    def _show_page(self, page_id: str):
        """显示页面"""
        # 在处理时限制页面访问
        if self.is_processing and page_id not in ["preview", "settings"]:
            return

        self.sidebar.set_active_button(page_id)

        # 设置当前页面索引
        page_index = {
            "settings": 0,
            "preview": 1,
            "species_validation": 2,
            "advanced": 3,
            "about": 4
        }.get(page_id, 0)

        self.content_stack.setCurrentIndex(page_index)
        self.current_page = page_id

        # 更新状态栏
        if page_id == "settings":
            if self.is_processing:
                # 如果正在处理，则恢复状态栏的进度显示
                self._on_progress_updated(
                    self.last_progress_value,
                    self.last_progress_total,
                    self.last_progress_speed,
                    self.last_progress_remaining_time
                )
            else:
                self.status_bar.status_label.setText("就绪")
        elif page_id == "preview":
            if hasattr(self.start_page, 'get_file_path'):
                file_path = self.start_page.get_file_path()
                if file_path and os.path.isdir(file_path):
                    if hasattr(self.preview_page, 'update_file_list'):
                        self.preview_page.update_file_list(file_path)
                    file_count = self.preview_page.get_file_count() if hasattr(self.preview_page,
                                                                               'get_file_count') else 0
                    self.status_bar.status_label.setText(f"当前文件夹下有 {file_count} 个图像文件")
                else:
                    self.status_bar.status_label.setText('请在"开始"页面中设置有效的图像文件路径')
        elif page_id == "species_validation":
            if not self.is_processing:
                self.status_bar.status_label.setText("就绪")
            # 确保物种校验页面的数据被加载
            if hasattr(self.species_validation_page, '_load_species_data'):
                # 使用QTimer延迟执行，确保页面完全显示后再加载数据
                QTimer.singleShot(100, self.species_validation_page._load_species_data)
            if hasattr(self.species_validation_page, '_load_species_buttons'):
                QTimer.singleShot(150, self.species_validation_page._load_species_buttons)
        elif page_id in ["advanced", "about"] and not self.is_processing:
            self.status_bar.status_label.setText("就绪")

    def browse_file_path(self):
        """浏览文件路径"""
        folder_selected = QFileDialog.getExistingDirectory(
            self, "选择图像文件所在文件夹"
        )
        if folder_selected:
            if hasattr(self.start_page, 'set_file_path'):
                self.start_page.set_file_path(folder_selected)
            self._validate_and_update_file_path(folder_selected)

    def browse_save_path(self):
        """浏览保存路径"""
        folder_selected = QFileDialog.getExistingDirectory(
            self, "选择结果保存文件夹"
        )
        if folder_selected:
            if hasattr(self.start_page, 'set_save_path'):
                self.start_page.set_save_path(folder_selected)
            self._validate_and_update_save_path(folder_selected)

    def _validate_and_update_file_path(self, folder_selected):
        """验证和更新文件路径"""
        if hasattr(self.preview_page, 'clear_preview'):
            self.preview_page.clear_preview()

        if not folder_selected:
            self.status_bar.status_label.setText("文件路径已清除")
            self._save_current_settings()
            return

        if os.path.isdir(folder_selected):
            self.get_temp_photo_dir(update=True)
            if hasattr(self.preview_page, 'update_file_list'):
                self.preview_page.update_file_list(folder_selected)
            file_count = self.preview_page.get_file_count() if hasattr(self.preview_page, 'get_file_count') else 0
            self.status_bar.status_label.setText(f"文件路径已设置，找到 {file_count} 个图像文件。")
            self._save_current_settings()
            if self.current_page == "preview":
                self._show_page("preview")
        else:
            QMessageBox.critical(
                self, "路径错误",
                f"提供的图像文件路径不存在或不是一个文件夹:\n'{folder_selected}'"
            )
            self.status_bar.status_label.setText("无效的文件路径")

    def _validate_and_update_save_path(self, save_path):
        """验证和更新保存路径"""
        if not save_path:
            return

        if not os.path.isdir(save_path):
            reply = QMessageBox.question(
                self, "确认创建路径",
                f"结果保存路径不存在，是否要创建它？\n\n{save_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.makedirs(save_path, exist_ok=True)
                    if hasattr(self.start_page, 'set_save_path'):
                        self.start_page.set_save_path(save_path)
                    self.status_bar.status_label.setText(f"结果保存路径已创建: {save_path}")
                    self._save_current_settings()
                except Exception as e:
                    QMessageBox.critical(self, "路径错误", f"无法创建结果保存路径:\n{e}")
                    self.status_bar.status_label.setText("结果保存路径创建失败")
            else:
                self.status_bar.status_label.setText("操作已取消，请输入有效的结果保存路径。")
        else:
            if hasattr(self.start_page, 'set_save_path'):
                self.start_page.set_save_path(save_path)
            self.status_bar.status_label.setText(f"结果保存路径已设置: {save_path}")
            self._save_current_settings()

    def show_params_help(self):
        """显示参数帮助"""
        help_text = """
**检测阈值设置**
- **IOU阈值:** 控制对象检测中非极大值抑制（NMS）的重叠阈值。较高的值会减少重叠框，但可能导致部分目标漏检。
- **置信度阈值:** 检测对象的最小置信度分数。较高的值只显示高置信度的检测结果，减少误检。

**模型加速选项**
- **使用FP16加速:** 使用半精度浮点数进行推理，可以加快速度但可能会略微降低精度。需要兼容的NVIDIA GPU。

**高级检测选项**
- **使用数据增强:** 在测试时使用数据增强（TTA），通过对输入图像进行多种变换并综合结果，可能会提高准确性，但会显著降低处理速度。
- **使用类别无关NMS:** 在所有类别上一起执行NMS，对于检测多种相互重叠的物种可能有用。
    """
        QMessageBox.information(self, "参数说明", help_text.strip())

    def get_temp_photo_dir(self, update=False):
        """获取临时图片目录"""
        if hasattr(self.start_page, 'get_file_path'):
            source_path = self.start_page.get_file_path()
        else:
            return None

        if not source_path:
            return None

        path_hash = hashlib.md5(source_path.encode()).hexdigest()
        base_dir = self.settings_manager.base_dir
        temp_dir = os.path.join(base_dir, "temp", "photo", path_hash)

        if update:
            self.current_temp_photo_dir = temp_dir

        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def clear_image_cache(self):
        """清除图像缓存"""
        cache_dir = os.path.join(self.settings_manager.base_dir, "temp", "photo")
        reply = QMessageBox.question(
            self, "确认清除缓存",
            f"是否清空图片缓存？\n\n此操作将删除以下文件夹及其所有内容：\n{cache_dir}\n\n注意：这不会影响您的原始图片或已保存的结果。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    QMessageBox.information(self, "成功", "图片缓存已成功清除。")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"清除缓存时发生错误：\n{e}")
            else:
                QMessageBox.information(self, "提示", "缓存目录不存在，无需清除。")

    def toggle_processing_state(self):
        """切换处理状态"""
        if not self.is_processing:
            self.check_for_cache_and_process()
        else:
            self.stop_processing()

    def check_for_cache_and_process(self):
        """检查缓存并处理"""
        cache_file = os.path.join(self.settings_manager.settings_dir, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if 'processed_files' in cache_data and 'total_files' in cache_data:
                    processed = cache_data.get('processed_files', 0)
                    total = cache_data.get('total_files', 0)
                    file_path = cache_data.get('file_path', '')
                    reply = QMessageBox.question(
                        self, "发现未完成任务",
                        f"检测到上次有未完成的任务，是否继续？\n已处理：{processed}/{total} at {file_path}",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._load_cache_data_from_file(cache_data)
                        self.start_processing(resume_from=processed)
                        return
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}")
        self.start_processing()

    def start_processing(self, resume_from=0):
        """开始处理"""
        # 获取处理参数
        file_path = self.start_page.get_file_path() if hasattr(self.start_page, 'get_file_path') else ""
        save_path = self.start_page.get_save_path() if hasattr(self.start_page, 'get_save_path') else ""
        use_fp16 = self.advanced_page.get_use_fp16() if hasattr(self.advanced_page, 'get_use_fp16') else False

        if not self._validate_inputs(file_path, save_path):
            return

        if self.is_processing:
            return

        # 切换到预览页面
        self._show_page("preview")

        if resume_from == 0:
            self.excel_data = []
            self._clear_current_validation_file()

        # 设置处理状态
        self._set_processing_state(True)

        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self, file_path, save_path, use_fp16, resume_from
        )

        # 连接信号
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.file_processed.connect(self._on_file_processed)
        self.processing_thread.processing_complete.connect(self._on_processing_complete)
        self.processing_thread.status_message.connect(self._on_status_message)
        self.processing_thread.file_processing_result.connect(self._on_file_processing_result)
        self.processing_thread.current_file_changed.connect(self._on_current_file_changed)
        self.processing_thread.current_file_preview.connect(self._on_current_file_preview)

        self.processing_thread.start()

    def _on_current_file_changed(self, img_path, current_index, total_files):
        """当前处理文件变化处理"""
        filename = os.path.basename(img_path)
        self.current_processing_file = filename  # 保存当前文件名

        # 如果当前在预览页面，则同步显示
        if self.current_page == "preview":
            if hasattr(self.preview_page, 'sync_current_processing_file'):
                self.preview_page.sync_current_processing_file(img_path, current_index, total_files)

    def _on_current_file_preview(self, img_path, detection_info):
        """当前文件预览处理"""
        # 如果当前在预览页面，则显示检测结果
        if self.current_page == "preview":
            if hasattr(self.preview_page, 'sync_current_processing_result'):
                self.preview_page.sync_current_processing_result(img_path, detection_info)

    def _on_file_processing_result(self, img_path, detection_info):
        """处理文件处理结果，同步到预览页面"""
        if hasattr(self.preview_page, 'sync_processing_result'):
            self.preview_page.sync_processing_result(img_path, detection_info)

    def stop_processing(self):
        """停止处理"""
        reply = QMessageBox.question(
            self, "停止确认",
            "确定要停止图像处理吗？\n处理进度将被保存，下次可以继续。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if hasattr(self, 'processing_thread') and self.processing_thread is not None:
                if hasattr(self.processing_thread, 'stop'):
                    self.processing_thread.stop()
                elif hasattr(self.processing_thread, 'stop_flag'):
                    self.processing_thread.stop_flag = True
            self.status_bar.status_label.setText("正在停止处理...")
        else:
            QMessageBox.information(self, "信息", "处理继续进行。")

    def _validate_inputs(self, file_path: str, save_path: str) -> bool:
        if not file_path or not os.path.isdir(file_path):
            QMessageBox.critical(self, "错误", "请提供有效的源文件夹路径。")
            return False
        if not save_path or not os.path.isdir(save_path):
            QMessageBox.critical(self, "错误", "请提供有效的保存文件夹路径。")
            return False
        return True

    def _set_processing_state(self, is_processing: bool):
        """设置处理状态"""
        self.is_processing = is_processing

        if hasattr(self.start_page, 'set_processing_state'):
            self.start_page.set_processing_state(is_processing)
        if hasattr(self.sidebar, 'set_processing_state'):
            self.sidebar.set_processing_state(is_processing)

        if is_processing:
            if hasattr(self.preview_page, 'set_show_detection'):
                self.preview_page.set_show_detection(True)
        else:
            if not is_processing and self.status_bar.status_label.text() != "处理完成！":
                self.status_bar.status_label.setText("就绪")

    def _on_progress_updated(self, value, total, speed, remaining_time):
        """处理进度更新，直接更新状态栏"""
        # 存储最新的进度信息
        self.last_progress_value = value
        self.last_progress_total = total
        self.last_progress_speed = speed
        self.last_progress_remaining_time = remaining_time

        # 格式化剩余时间
        if isinstance(remaining_time, (int, float)) and remaining_time > 0:
            if remaining_time == float('inf') or remaining_time > 3600 * 24:
                time_text = "计算中..."
            elif remaining_time > 3600:
                hours = int(remaining_time // 3600)
                minutes = int((remaining_time % 3600) // 60)
                time_text = f"{hours}h{minutes}m"
            elif remaining_time > 60:
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                time_text = f"{minutes}m{seconds}s"
            else:
                time_text = f"{int(remaining_time)}s"
        else:
            time_text = "已完成" if value == total and total > 0 else "计算中..."

        # 获取当前处理的文件名
        current_file = getattr(self, 'current_processing_file', '...')

        # 创建状态栏消息
        status_message = (
            f"处理中: {value}/{total} ({speed:.2f} 张/秒) | "
            f"剩余: {time_text} | "
            f"文件: {current_file}"
        )
        self.status_bar.status_label.setText(status_message)

    def _on_file_processed(self, img_path, detection_results, filename):
        """处理文件完成"""
        if hasattr(self.preview_page, 'on_file_processed'):
            self.preview_page.on_file_processed(img_path, detection_results, filename)

    def _on_processing_complete(self, success):
        """处理完成"""
        self._set_processing_state(False)

        # 如果当前在物种校验页面，刷新数据
        if self.current_page == "species_validation" and success:
            if hasattr(self.species_validation_page, '_load_species_data'):
                QTimer.singleShot(500, self.species_validation_page._load_species_data)

    def _on_status_message(self, message):
        """状态消息 - 简化状态栏显示，仅处理非进度消息"""
        # 进度信息由 _on_progress_updated 处理，这里只显示其他状态
        if "正在处理:" not in message:
            self.status_bar.status_label.setText(message)

    def _save_current_settings(self):
        """保存当前设置"""
        if not self.settings_manager:
            return
        settings = self._get_current_settings()
        if self.settings_manager.save_settings(settings):
            logger.info("设置已保存")

    def _get_current_settings(self):
        """获取当前设置"""
        settings = {}

        # 从开始页面获取设置
        if hasattr(self.start_page, 'get_settings'):
            settings.update(self.start_page.get_settings())

        # 从高级页面获取设置
        if hasattr(self.advanced_page, 'get_settings'):
            settings.update(self.advanced_page.get_settings())

        # 从预览页面获取设置
        if hasattr(self.preview_page, 'get_settings'):
            settings.update(self.preview_page.get_settings())

            # 从物种校验页面获取设置
            if hasattr(self.species_validation_page, 'get_settings'):
                settings.update(self.species_validation_page.get_settings())

        # 添加其他设置，确保包含模型信息
        settings.update({
            "selected_model": getattr(self, 'model_var', ''),
        })

        # 如果model_var为空，尝试从image_processor获取
        if not settings.get("selected_model") and hasattr(self, 'image_processor'):
            if hasattr(self.image_processor, 'model_path') and self.image_processor.model_path:
                settings["selected_model"] = os.path.basename(self.image_processor.model_path)

        return settings

    def _load_settings_to_ui(self, settings: dict):
        """加载设置到UI"""
        if not settings:
            return
        try:
            # 加载到各个页面
            if hasattr(self.start_page, 'load_settings'):
                self.start_page.load_settings(settings)
            if hasattr(self.advanced_page, 'load_settings'):
                self.advanced_page.load_settings(settings)
            if hasattr(self.preview_page, 'load_settings'):
                self.preview_page.load_settings(settings)
            if hasattr(self.species_validation_page, 'load_settings'):
                self.species_validation_page.load_settings(settings)

            # 加载主题
            theme = settings.get("theme", "自动")
            if hasattr(self.advanced_page, 'set_theme_selection'):
                self.advanced_page.set_theme_selection(theme)
            self.change_theme()

        except Exception as e:
            logger.error(f"加载设置到UI失败: {e}")

    def _load_cache_data_from_file(self, cache_data):
        """从文件加载缓存数据"""
        self._load_settings_to_ui(cache_data)
        self.excel_data = cache_data.get('excel_data', [])
        for item in self.excel_data:
            if '拍摄日期对象' in item and isinstance(item['拍摄日期对象'], str):
                try:
                    item['拍摄日期对象'] = datetime.fromisoformat(item['拍摄日期对象'])
                except ValueError:
                    item['拍摄日期对象'] = None

    def _resume_processing(self):
        """恢复处理"""
        self._load_cache_data_from_file(self.cache_data)
        self.start_processing(resume_from=self.cache_data.get('processed_files', 0))

    def _clear_current_validation_file(self):
        """清除当前验证文件"""
        temp_photo_dir = self.get_temp_photo_dir()
        if temp_photo_dir:
            validation_file_path = os.path.join(temp_photo_dir, "validation.json")
            if os.path.exists(validation_file_path):
                try:
                    os.remove(validation_file_path)
                    logger.info(f"已清除旧的校验文件: {validation_file_path}")
                except Exception as e:
                    logger.error(f"清除旧的校验文件失败: {e}")

        # 清除内存中的数据
        if hasattr(self.preview_page, 'clear_validation_data'):
            self.preview_page.clear_validation_data()

    def closeEvent(self, event):
        """关闭事件"""
        if self.is_processing:
            reply = QMessageBox.question(
                self, "确认退出",
                "图像处理正在进行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            # 检查处理线程是否存在且有stop方法
            if hasattr(self, 'processing_thread') and self.processing_thread is not None:
                if hasattr(self.processing_thread, 'stop'):
                    self.processing_thread.stop()
                else:
                    # 如果没有stop方法，设置停止标志
                    if hasattr(self.processing_thread, 'stop_flag'):
                        self.processing_thread.stop_flag = True

        # 保存验证数据
        if hasattr(self.preview_page, '_save_validation_data'):
            self.preview_page._save_validation_data()

        # 保存设置
        self._save_current_settings()

        event.accept()

    def _set_title_bar_color(self):
        """设置窗口标题栏颜色 (仅限Windows)"""
        if platform.system() != "Windows":
            return

        try:
            from PySide6.QtGui import QColor

            # Constant for DWMWA_CAPTION_COLOR from Windows API
            DWMWA_CAPTION_COLOR = 35

            if self.is_dark_mode:
                # 深色模式标题栏颜色
                color_str = "#5d3a4f"
            else:
                # 浅色模式标题栏颜色
                color_str = "#f6dce0"
            
            color = QColor(color_str)
            # Windows API COLORREF is in 0x00BBGGRR format
            color_ref = color.blue() << 16 | color.green() << 8 | color.red()
            
            hwnd = self.winId()
            if hwnd:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(hwnd),
                    DWMWA_CAPTION_COLOR,
                    ctypes.byref(ctypes.c_int(color_ref)),
                    ctypes.sizeof(ctypes.c_int)
                )
        except Exception as e:
            logger.warning(f"无法设置标题栏颜色: {e}")

