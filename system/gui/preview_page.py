from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QLabel, QTextEdit, QPushButton,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QMessageBox, QFileDialog, QInputDialog, QScrollArea,
    QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QFont, QPainter, QPen, QColor, QIcon, QImage, QPalette
import sys
import os
import json
import logging
import cv2
import threading
import re
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

# 原有的导入保持不变
from system.data_processor import DataProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.config import NORMAL_FONT, SUPPORTED_IMAGE_EXTENSIONS
from system.utils import resource_path
from system.gui.ui_components import Win11Colors, ModernSlider, ModernGroupBox, SwitchRow

logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    """检测工作线程"""
    detection_completed = Signal(dict, str)  # 检测完成信号：(loaded_detection_info, filename)
    detection_failed = Signal(str)  # 检测失败信号：error_message

    def __init__(self, controller, img_path, filename):
        super().__init__()
        self.controller = controller
        self.img_path = img_path
        self.filename = filename

    def run(self):
        try:
            # 获取设置参数
            use_fp16 = self.controller.advanced_page.get_use_fp16()
            iou = self.controller.advanced_page.iou_var
            conf = self.controller.advanced_page.conf_var
            use_augment = self.controller.advanced_page.use_augment_var
            use_agnostic_nms = self.controller.advanced_page.use_agnostic_nms_var

            from datetime import datetime
            results = self.controller.image_processor.detect_species(
                self.img_path,
                use_fp16,
                iou,
                conf,
                use_augment,
                use_agnostic_nms
            )

            current_detection_results = results['detect_results']
            species_info = {k: v for k, v in results.items() if k != 'detect_results'}
            species_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if current_detection_results:
                temp_photo_dir = self.controller.get_temp_photo_dir()
                # 保存JSON文件
                json_path = self.controller.image_processor.save_detection_info_json(
                    current_detection_results, self.filename, species_info, temp_photo_dir
                )

                # 从刚保存的JSON中读回数据
                with open(json_path, 'r', encoding='utf-8') as f:
                    loaded_detection_info = json.load(f)

                # 发射完成信号
                self.detection_completed.emit(loaded_detection_info, self.filename)
            else:
                # 没有检测结果
                self.detection_completed.emit({}, self.filename)

        except Exception as e:
            logger.error(f"检测图像失败: {e}")
            self.detection_failed.emit(str(e))


class ImageLoaderThread(QThread):
    """用于在后台加载图像和元数据的工作线程（安全取消版）"""
    image_loaded = Signal(object, str, dict)  # (pixmap, file_path, image_info)
    loading_failed = Signal(str, str)         # (file_path, error_message)

    def __init__(self, file_path, display_size, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.display_size = display_size
        self._is_cancelled = False

    def cancel(self):
        """请求线程停止"""
        self._is_cancelled = True

    def run(self):
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                self.loading_failed.emit(self.file_path, "文件路径无效")
                return

            # --- 线程取消点 1 ---
            if self._is_cancelled: return

            from PIL import Image
            import numpy as np
            from PySide6.QtGui import QImage, QPixmap
            from system.metadata_extractor import ImageMetadataExtractor

            # 1. 加载图像 (这是一个潜在的耗时I/O操作)
            img = Image.open(self.file_path)
            img.load() # 确保图像数据已完全加载到内存

            # --- 线程取消点 2 ---
            if self._is_cancelled: return

            # 2. 提取元数据
            file_name = os.path.basename(self.file_path)
            image_info, _ = ImageMetadataExtractor.extract_metadata(self.file_path, file_name)

            # --- 线程取消点 3 ---
            if self._is_cancelled: return

            # 3. 调整图像大小并转换为QPixmap (这是耗时的CPU操作)
            max_width = max(self.display_size.width(), 400)
            max_height = max(self.display_size.height(), 300)

            w, h = img.size
            scale = min(max_width / w, max_height / h) if w > 0 and h > 0 else 1
            if scale < 1:
                new_width = max(1, int(w * scale))
                new_height = max(1, int(h * scale))
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized_img = img

            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')

            img_array = np.array(resized_img)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # --- 线程取消点 4 ---
            if self._is_cancelled: return

            # 4. 发送完成信号
            self.image_loaded.emit(pixmap, self.file_path, image_info)

        except Exception as e:
            if not self._is_cancelled:
                self.loading_failed.emit(self.file_path, str(e))


class PreviewPage(QWidget):
    """图像预览页面"""
    settings_changed = Signal()

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent)
        self.controller = controller
        self.original_image = None
        self.loaded_image_path = None
        self.requested_image_path = None
        self.current_detection_results = None
        self.image_loader_thread = None

        self.current_preview_info = {}
        self.active_keybinds = []
        self._is_navigating = False

        # Win11 配色方案
        self.color_palette = [
            '#0078d4', '#00bcf2', '#40e0d0', '#f7630c', '#ffb900', '#107c10',
            '#bad80a', '#00b4ff', '#0078d4', '#5c2d91', '#e74856', '#ff8c00',
            '#ffd700', '#32cd32', '#00ced1', '#da70d6', '#ff6347', '#4682b4'
        ]
        self.species_color_map = {}

        global_conf = self.controller.confidence_settings.get("global", 0.25)
        self.preview_conf_var = global_conf

        self._create_widgets()
        self._apply_theme()

        # 用于处理窗口大小调整的计时器
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._redraw_image_on_resize)


    def _apply_theme(self):
        """应用当前的主题样式"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            # Dark theme colors
            bg_color = Win11Colors.DARK_BACKGROUND.name()
            text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            pane_border_color = Win11Colors.DARK_BORDER.name()
            pane_bg_color = Win11Colors.DARK_CARD.name()
            list_widget_bg_color = Win11Colors.DARK_CARD.name()
            list_widget_border_color = Win11Colors.DARK_BORDER.name()
            list_widget_selection_bg_color = Win11Colors.DARK_ACCENT.name()
            list_widget_selection_text_color = "#ffffff"
            list_widget_item_hover_bg_color = Win11Colors.DARK_HOVER.name()
            group_box_text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            group_box_border_color = Win11Colors.DARK_BORDER.name()
            group_box_bg_color = Win11Colors.DARK_BACKGROUND.name()
            button_bg_color = Win11Colors.DARK_ACCENT.name()
            button_text_color = "#ffffff"
            button_hover_bg_color = Win11Colors.DARK_ACCENT.lighter(120).name()
            button_pressed_bg_color = Win11Colors.DARK_ACCENT.darker(110).name()
            button_disabled_bg_color = Win11Colors.DARK_BORDER.name()
            button_disabled_text_color = Win11Colors.DARK_TEXT_SECONDARY.name()
            slider_groove_border_color = Win11Colors.DARK_BORDER.name()
            slider_groove_bg_color = Win11Colors.DARK_SURFACE.name()
            slider_handle_bg_color = Win11Colors.DARK_ACCENT.name()
            slider_handle_hover_bg_color = Win11Colors.DARK_ACCENT.lighter(120).name()
            checkbox_text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            checkbox_indicator_border_color = Win11Colors.DARK_BORDER.name()
            checkbox_indicator_bg_color = Win11Colors.DARK_SURFACE.name()
            checkbox_indicator_checked_bg_color = Win11Colors.DARK_ACCENT.name()
            combo_box_border_color = Win11Colors.DARK_BORDER.name()
            combo_box_bg_color = Win11Colors.DARK_SURFACE.name()
            combo_box_focus_border_color = Win11Colors.DARK_ACCENT.name()
            text_edit_bg_color = Win11Colors.DARK_CARD.name()
            text_edit_border_color = Win11Colors.DARK_BORDER.name()
            label_text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
        else:
            # Light theme colors
            bg_color = Win11Colors.LIGHT_BACKGROUND.name()
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            pane_border_color = Win11Colors.LIGHT_BORDER.name()
            pane_bg_color = Win11Colors.LIGHT_CARD.name()
            list_widget_bg_color = Win11Colors.LIGHT_CARD.name()
            list_widget_border_color = Win11Colors.LIGHT_BORDER.name()
            list_widget_selection_bg_color = Win11Colors.LIGHT_ACCENT.name()
            list_widget_selection_text_color = "#ffffff"
            list_widget_item_hover_bg_color = Win11Colors.LIGHT_HOVER.name()
            group_box_text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            group_box_border_color = Win11Colors.LIGHT_BORDER.name()
            group_box_bg_color = Win11Colors.LIGHT_BACKGROUND.name()
            button_bg_color = Win11Colors.LIGHT_ACCENT.name()
            button_text_color = "#ffffff"
            button_hover_bg_color = Win11Colors.LIGHT_ACCENT.darker(110).name()
            button_pressed_bg_color = Win11Colors.LIGHT_ACCENT.darker(120).name()
            button_disabled_bg_color = "#cccccc"
            button_disabled_text_color = "#666666"
            slider_groove_border_color = Win11Colors.LIGHT_BORDER.name()
            slider_groove_bg_color = Win11Colors.LIGHT_SURFACE.name()
            slider_handle_bg_color = Win11Colors.LIGHT_ACCENT.name()
            slider_handle_hover_bg_color = Win11Colors.LIGHT_ACCENT.darker(110).name()
            checkbox_text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            checkbox_indicator_border_color = Win11Colors.LIGHT_BORDER.name()
            checkbox_indicator_bg_color = Win11Colors.LIGHT_CARD.name()
            checkbox_indicator_checked_bg_color = Win11Colors.LIGHT_ACCENT.name()
            combo_box_border_color = Win11Colors.LIGHT_BORDER.name()
            combo_box_bg_color = Win11Colors.LIGHT_CARD.name()
            combo_box_focus_border_color = Win11Colors.LIGHT_ACCENT.name()
            text_edit_bg_color = Win11Colors.LIGHT_CARD.name()
            text_edit_border_color = Win11Colors.LIGHT_BORDER.name()
            label_text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()

        # 设置 Win11 风格
        self.setStyleSheet(f"""
                        QWidget {{
                            background-color: {bg_color};
                            color: {text_color};
                            font-family: 'Segoe UI', Arial, sans-serif;
                        }}
                        QListWidget {{
                            background-color: {list_widget_bg_color};
                            border: 1px solid {list_widget_border_color};
                            border-radius: 6px;
                            selection-background-color: {list_widget_selection_bg_color};
                            selection-color: {list_widget_selection_text_color};
                            font-size: 14px;
                            padding: 4px;
                        }}
                        QListWidget::item {{
                            padding: 6px;
                            border-radius: 4px;
                            margin: 1px;
                        }}
                        QListWidget::item:hover {{
                            background-color: {list_widget_item_hover_bg_color};
                        }}
                        QListWidget::item:selected {{
                            background-color: {list_widget_selection_bg_color};
                            color: {list_widget_selection_text_color};
                        }}
                        ModernGroupBox {{
                            font-weight: 600;
                            font-size: 14px;
                            color: {group_box_text_color};
                            border: 2px solid {group_box_border_color};
                            border-radius: 8px;
                            margin-top: 10px;
                            padding-top: 10px;
                        }}
                        ModernGroupBox::title {{
                            subcontrol-origin: margin;
                            left: 10px;
                            padding: 0 8px 0 8px;
                            background-color: {group_box_bg_color};
                        }}
                        QPushButton {{
                            background-color: {button_bg_color};
                            color: {button_text_color};
                            border: none;
                            padding: 8px 16px;
                            border-radius: 6px;
                            font-size: 14px;
                            font-weight: 500;
                            min-width: 80px;
                        }}
                        QPushButton:hover {{
                            background-color: {button_hover_bg_color};
                        }}
                        QPushButton:pressed {{
                            background-color: {button_pressed_bg_color};
                        }}
                        QPushButton:disabled {{
                            background-color: {button_disabled_bg_color};
                            color: {button_disabled_text_color};
                        }}
                        QSlider::groove:horizontal {{
                            border: 1px solid {slider_groove_border_color};
                            height: 6px;
                            background: {slider_groove_bg_color};
                            border-radius: 3px;
                        }}
                        QSlider::handle:horizontal {{
                            background: {slider_handle_bg_color};
                            border: 1px solid {slider_handle_bg_color};
                            width: 16px;
                            height: 16px;
                            border-radius: 8px;
                            margin: -6px 0;
                        }}
                        QSlider::handle:horizontal:hover {{
                            background: {slider_handle_hover_bg_color};
                        }}
                        SwitchRow {{
                            font-size: 14px;
                            color: {checkbox_text_color};
                        }}
                        SwitchRow::indicator {{
                            width: 18px;
                            height: 18px;
                            border: 2px solid {checkbox_indicator_border_color};
                            border-radius: 4px;
                            background-color: {checkbox_indicator_bg_color};
                        }}
                        SwitchRow::indicator:checked {{
                            background-color: {checkbox_indicator_checked_bg_color};
                            border-color: {checkbox_indicator_checked_bg_color};
                            image: url(checkmark.png);
                        }}
                        QComboBox {{
                            border: 2px solid {combo_box_border_color};
                            border-radius: 6px;
                            padding: 6px 12px;
                            background-color: {combo_box_bg_color};
                            min-width: 100px;
                            font-size: 14px;
                        }}
                        QComboBox:focus {{
                            border-color: {combo_box_focus_border_color};
                        }}
                        QComboBox::drop-down {{
                            border: none;
                            width: 20px;
                        }}
                        QComboBox::down-arrow {{
                            image: url(down_arrow.png);
                            width: 12px;
                            height: 12px;
                        }}
                        QTextEdit {{
                            background-color: {text_edit_bg_color};
                            border: 1px solid {text_edit_border_color};
                            border-radius: 6px;
                            padding: 8px;
                            font-size: 14px;
                            line-height: 1.4;
                        }}
                        QLabel {{
                            color: {label_text_color};
                            font-size: 14px;
                        }}
                    """)

    def _create_widgets(self):
        """创建预览页面的所有控件"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 直接创建图像预览内容，不使用标签页
        self._create_image_preview_content(layout)

    def _create_image_preview_content(self, parent_layout):
        """创建图像预览内容"""
        # 主内容区域
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # 左侧文件列表
        list_group = ModernGroupBox("图像文件")
        list_group.setFixedWidth(200)
        list_layout = QVBoxLayout(list_group)

        self.file_listbox = QListWidget()
        self.file_listbox.setMinimumWidth(180)
        self.file_listbox.itemSelectionChanged.connect(self.on_file_selected)
        list_layout.addWidget(self.file_listbox)

        content_layout.addWidget(list_group)

        # 右侧预览区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 图像预览区域
        image_group = ModernGroupBox("图像预览")
        image_layout = QVBoxLayout(image_group)

        self.image_label = QLabel("请从左侧列表选择图像")
        self.image_label.pixmap = None
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(self._get_placeholder_style())
        image_layout.addWidget(self.image_label)

        right_layout.addWidget(image_group, 3)

        # 图像信息区域
        info_group = ModernGroupBox("图像信息")
        info_group.setFixedHeight(160)  # 设置固定高度
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setFixedHeight(100)  # 增加文本框高度
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)

        right_layout.addWidget(info_group)  # 添加分组框，不设置拉伸因子

        # 控制面板
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.show_detection_checkbox = SwitchRow("显示检测结果")
        self.show_detection_checkbox.toggled.connect(self.toggle_detection_preview)
        control_layout.addWidget(self.show_detection_checkbox)

        # 置信度滑块
        control_layout.addWidget(QLabel("置信度:"))
        self.preview_conf_slider = ModernSlider(Qt.Horizontal)
        self.preview_conf_slider.setRange(5, 95)
        self.preview_conf_slider.setValue(int(self.preview_conf_var * 100))
        self.preview_conf_slider.valueChanged.connect(self._on_preview_confidence_slider_changed)
        control_layout.addWidget(self.preview_conf_slider)

        self.preview_conf_label = QLabel(f"{self.preview_conf_var:.2f}")
        control_layout.addWidget(self.preview_conf_label)

        control_layout.addStretch()

        self.detect_button = QPushButton("检测当前图像")
        self.detect_button.clicked.connect(self.detect_current_image)
        control_layout.addWidget(self.detect_button)

        right_layout.addWidget(control_widget)
        content_layout.addWidget(right_widget, 1)

        parent_layout.addLayout(content_layout)

    def clear_preview(self):
        """清除预览"""
        try:
            self.file_listbox.clear()
            self.current_image_path = None
            self.current_detection_results = None
            self._safe_clear_image()
            self._clear_details_panel()
            logger.info("预览已清除")
        except Exception as e:
            logger.error(f"清除预览时出错: {e}")

    def _safe_clear_image(self):
        """安全地清除图像显示"""
        try:
            self.image_label.clear()
            self.image_label.setText("无图像")
            if hasattr(self, '_current_pixmap'):
                self._current_pixmap = None
        except Exception as e:
            logger.warning(f"清除图像显示时出现警告: {e}")

    def _clear_details_panel(self):
        """清除详情面板内容"""
        try:
            if hasattr(self, 'file_name_label'):
                self.file_name_label.setText("文件名: -")
            if hasattr(self, 'file_size_label'):
                self.file_size_label.setText("文件大小: -")
            if hasattr(self, 'dimensions_label'):
                self.dimensions_label.setText("尺寸: -")
            if hasattr(self, 'modified_label'):
                self.modified_label.setText("修改时间: -")
        except Exception as e:
            logger.warning(f"清除详情面板时出现警告: {e}")

    def get_file_count(self):
        """获取文件列表中的文件数量"""
        return self.file_listbox.count()

    def update_file_list(self, directory: str):
        """更新文件列表，加载指定目录下的图像文件"""
        self.file_listbox.clear()

        if not os.path.isdir(directory):
            return

        try:
            image_files = [f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            image_files.sort()

            for file in image_files:
                from PySide6.QtWidgets import QListWidgetItem
                item = QListWidgetItem(file)
                full_path = os.path.join(directory, file)
                # 将完整路径存储在列表项的用户数据中
                item.setData(Qt.ItemDataRole.UserRole, full_path)
                self.file_listbox.addItem(item)
        except Exception as e:
            logger.error(f"更新文件列表失败: {e}")

    def on_file_selected(self):
        """文件选择事件处理（修复版本 - 防止切换页面时崩溃）"""
        selected_items = self.file_listbox.selectedItems()
        if not selected_items:
            return

        file_path = selected_items[0].data(Qt.ItemDataRole.UserRole)

        if not file_path or file_path == self.requested_image_path:
            return

        self.requested_image_path = file_path

        # === 关键修复1: 强制停止并清理之前的线程 ===
        if self.image_loader_thread and self.image_loader_thread.isRunning():
            self.image_loader_thread.cancel()
            # 断开所有信号连接,防止已删除对象被访问
            try:
                self.image_loader_thread.image_loaded.disconnect()
                self.image_loader_thread.loading_failed.disconnect()
                self.image_loader_thread.finished.disconnect()
            except:
                pass
            # 等待线程结束(最多等待100ms)
            self.image_loader_thread.wait(100)
            self.image_loader_thread = None
        # ============================================

        self.image_label.setText("正在加载图像...")
        self.image_label.pixmap = None
        self.info_text.setPlainText(f"正在加载: {os.path.basename(file_path)}")
        self.current_preview_info = {}

        # 启动新的加载线程
        self.image_loader_thread = ImageLoaderThread(file_path, self.image_label.size())

        # === 关键修复2: 使用 Lambda 包装信号槽,避免直接传递 self ===
        self.image_loader_thread.image_loaded.connect(
            lambda pixmap, fp, info: self._on_image_loaded_safe(pixmap, fp, info)
        )
        self.image_loader_thread.loading_failed.connect(
            lambda fp, err: self._on_loading_failed_safe(fp, err)
        )
        # =======================================================

        self.image_loader_thread.finished.connect(self.image_loader_thread.deleteLater)
        self.image_loader_thread.finished.connect(self._on_thread_finished)

        self.image_loader_thread.start()

    def _on_image_loaded_safe(self, pixmap, file_path, image_info):
        """安全的图像加载完成回调 - 添加对象有效性检查"""
        # === 关键修复3: 检查对象是否仍然有效 ===
        try:
            if not self or not hasattr(self, 'image_label'):
                return
            if self.image_label is None or not self.image_label.isVisible():
                return
            # ==========================================

            self._on_image_loaded(pixmap, file_path, image_info)
        except RuntimeError as e:
            logger.warning(f"图像加载回调时对象已删除: {e}")

    def _on_loading_failed_safe(self, file_path, error_message):
        """安全的加载失败回调 - 添加对象有效性检查"""
        # === 关键修复4: 同样的保护 ===
        try:
            if not self or not hasattr(self, 'image_label'):
                return
            if self.image_label is None:
                return
            # ==================================

            self._on_loading_failed(file_path, error_message)
        except RuntimeError as e:
            logger.warning(f"加载失败回调时对象已删除: {e}")

    def update_image_preview(self, file_path: str, show_detection: bool = False, detection_results=None,
                             is_temp_result: bool = False):
        """更新图像预览，支持显示检测结果"""
        try:
            # 始终从文件路径加载图像，以确保 self.original_image 是最新的
            self.original_image = Image.open(file_path)
            self.current_image_path = file_path # 确保当前路径被更新

            image_to_show = self.original_image

            if is_temp_result:
                # 临时结果直接显示
                image_to_show = Image.open(file_path)
            elif show_detection and detection_results:
                # 如果需要显示检测结果，则在原图上绘制
                result_img_array = detection_results[0].plot()
                image_to_show = Image.fromarray(cv2.cvtColor(result_img_array, cv2.COLOR_BGR2RGB))

            # 使用统一的辅助函数来设置和显示图片
            self._update_pixmap_for_label(image_to_show)

        except Exception as e:
            logger.error(f"更新图像预览失败: {e}")
            self.image_label.clear()
            self.image_label.setText("无法加载图像")
            self.image_label.setStyleSheet(self._get_placeholder_style())
            # 清理状态，防止后续操作出错
            self.original_image = None
            self.current_image_path = None
            self.image_label.pixmap = None

    def update_image_info(self, file_path: str, file_name: str, is_processing: bool = False):
        """更新图像信息显示"""
        from system.metadata_extractor import ImageMetadataExtractor

        try:
            image_info, _ = ImageMetadataExtractor.extract_metadata(file_path, file_name)

            # 清空并重新设置文本内容
            self.info_text.clear()

            # 构建信息文本
            info1 = f"文件名: {image_info.get('文件名', '')}    格式: {image_info.get('格式', '')}"
            info2 = f"拍摄日期: {image_info.get('拍摄日期', '未知')} {image_info.get('拍摄时间', '')}    "

            try:
                with Image.open(file_path) as img:
                    file_size_kb = os.path.getsize(file_path) / 1024
                    info2 += f"尺寸: {img.width}x{img.height}px    文件大小: {file_size_kb:.1f} KB"
            except Exception as e:
                logger.warning(f"获取图像尺寸信息失败: {e}")
                info2 += "尺寸信息获取失败"

            # 如果正在处理，添加处理状态
            processing_status = ""
            if is_processing:
                processing_status = "\n🔄 正在检测中..."

            # 设置完整的信息文本
            full_info = info1 + "\n" + info2 + processing_status
            self.info_text.setPlainText(full_info)

        except Exception as e:
            logger.error(f"更新图像信息失败: {e}")
            status_text = "无法获取图像信息"
            if is_processing:
                status_text += "\n🔄 正在检测中..."
            self.info_text.setPlainText(status_text)

    def toggle_detection_preview(self, checked):
        """切换检测结果预览显示"""
        if self.controller.is_processing:
            self.show_detection_checkbox.setChecked(True)
            return

        # 获取当前选中的文件
        selected_items = self.file_listbox.selectedItems()
        if not selected_items:
            self.show_detection_checkbox.setChecked(False)
            return

        if checked:
            # ===== 修改:添加原始图像检查 =====
            if not self.original_image:
                QMessageBox.warning(self, "提示", "图像尚未加载完成，请稍后再试。")
                self.show_detection_checkbox.setChecked(False)
                return
            # =================================

            # 显示带检测框的图像
            if self.current_preview_info and self.current_preview_info.get("检测框"):
                # 使用现有的检测信息重新绘制图像
                self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
            else:
                # 没有检测信息时，显示提示并取消选中状态
                QMessageBox.information(self, "提示", "当前图像还没有检测结果，请先点击'检测当前图像'按钮。")
                self.show_detection_checkbox.setChecked(False)
        else:
            # ===== 修改:只显示不带检测框的原始图片 =====
            if self.original_image:
                self._update_pixmap_for_label(self.original_image)
            # ==========================================

    def detect_current_image(self):
        """检测当前选中的图像"""
        selected_items = self.file_listbox.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择一张图像。")
            return

        file_name = selected_items[0].text()
        directory = self.controller.start_page.get_file_path()
        file_path = os.path.join(directory, file_name)

        # 更新状态并禁用按钮
        self.detect_button.setEnabled(False)
        self.detect_button.setText("检测中...")

        # 创建检测线程
        self.detection_worker = DetectionWorker(self.controller, file_path, file_name)

        # 连接信号
        self.detection_worker.detection_completed.connect(self._on_detection_completed)
        self.detection_worker.detection_failed.connect(self._on_detection_failed)
        self.detection_worker.finished.connect(self._on_detection_finished)

        # 启动线程
        self.detection_worker.start()

    def _update_detection_info(self, species_info):
        """更新检测信息显示"""
        try:
            # 获取当前文本内容的前两行（基本信息）
            current_text = self.info_text.toPlainText().strip()
            current_lines = current_text.split('\n') if current_text else []
            basic_info = "\n".join(current_lines[:2]) if len(current_lines) >= 2 else current_text

            # 构建检测结果信息
            detection_parts = ["检测结果:"]
            if species_info and species_info.get('物种名称') and species_info['物种名称'] != '空':
                names = species_info['物种名称'].split(',')
                counts = species_info.get('物种数量', '').split(',')
                info_parts = [f"{n}: {c}只" for n, c in zip(names, counts)]
                detection_parts.append(", ".join(info_parts))
                if species_info.get('最低置信度'):
                    detection_parts.append(f"最低置信度: {species_info['最低置信度']}")
                if species_info.get('检测时间'):
                    detection_parts.append(f"检测于: {species_info['检测时间']}")
            else:
                detection_parts.append("未检测到已知物种")

            # 合并基本信息和检测信息
            full_info = basic_info + "\n" + " | ".join(detection_parts)

            # 设置完整文本内容
            self.info_text.setPlainText(full_info)

        except Exception as e:
            logger.error(f"更新检测信息失败: {e}")

    def _resize_image_to_fit(self, img, max_width, max_height):
        if not all([max_width > 0, max_height > 0]):
            max_width, max_height = 400, 300
        w, h = img.size
        if w == 0 or h == 0: return img
        scale = min(max_width / w, max_height / h)
        if scale >= 1: return img
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))
        return img.resize((new_width, new_height), Image.LANCZOS)

    def _draw_detection_boxes(self, image_label, original_image, detection_info, conf_threshold_str):
        """
        根据给定的置信度阈值，在指定的原始图像上绘制检测框，并更新对应的UI标签。
        """
        # 1. 检查是否有原始图像
        if not original_image:
            placeholder_text = "请从左侧列表选择图像"
            image_label.clear()
            image_label.setText(placeholder_text)
            if hasattr(image_label, 'pixmap'):
                image_label.pixmap = None
            return

        # 如果有图像但没有检测信息，则只显示原始图片
        if not detection_info or not detection_info.get("检测框"):
            resized_img = self._resize_image_to_fit(
                original_image,
                image_label.width() or 400,
                image_label.height() or 300
            )
            # 转换PIL图像为QPixmap
            img_array = np.array(resized_img)
            if len(img_array.shape) == 3:  # RGB图像
                height, width, channel = img_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:  # 灰度图像
                height, width = img_array.shape
                bytes_per_line = width
                q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            image_label.setPixmap(pixmap)
            image_label.pixmap = pixmap  # 保持引用
            return

        # 处理置信度阈值
        try:
            if isinstance(conf_threshold_str, (int, float)):
                conf_threshold = float(conf_threshold_str)
                if conf_threshold > 1.0:  # 如果是百分比形式
                    conf_threshold = conf_threshold / 100.0
            else:
                conf_threshold = float(conf_threshold_str)
        except (ValueError, TypeError):
            conf_threshold = 0.25

        # 字体加载
        try:
            font_path = resource_path("assets/simhei.ttf")
            font_size = max(16, int(0.02 * min(original_image.width, original_image.height)))
            font = ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            logger.warning("中文字体文件未找到，使用默认字体")
            font = ImageFont.load_default()

        # 绘制逻辑
        img_to_draw = original_image.copy()
        draw = ImageDraw.Draw(img_to_draw)
        boxes_info = detection_info.get("检测框", [])

        for box in boxes_info:
            try:
                # 获取置信度，支持多种字段名
                confidence = box.get("置信度", box.get("confidence", 0))
                if confidence < conf_threshold:
                    continue

                # 获取物种名称，支持多种字段名
                species_name = box.get("物种", box.get("species", box.get("class_name", "未知")))

                # 获取边界框坐标，支持多种格式
                if "边界框" in box:
                    bbox = box["边界框"]
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                    else:
                        continue
                elif all(key in box for key in ["x1", "y1", "x2", "y2"]):
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                elif "bbox" in box:
                    bbox = box["bbox"]
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                    else:
                        continue
                else:
                    logger.warning(f"无法解析检测框坐标: {box}")
                    continue

                # 确保坐标为整数并且合理
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 确保坐标在图像范围内
                img_width, img_height = original_image.size
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                # 确保边界框有效
                if x2 <= x1 or y2 <= y1:
                    continue

                # 获取物种颜色
                color = self._get_color_for_species(species_name)

                # 绘制检测框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 准备标签文本
                label_text = f"{species_name} ({confidence:.2f})"

                # 计算文本框大小
                try:
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except (AttributeError, OSError):
                    # 兼容旧版PIL
                    text_width, text_height = draw.textsize(label_text, font=font)

                # 确保标签在图像范围内
                label_y = max(text_height + 5, y1)

                # 绘制标签背景
                draw.rectangle(
                    [x1, label_y - text_height - 5, x1 + text_width + 10, label_y],
                    fill=color
                )

                # 绘制标签文本（白色）
                draw.text((x1 + 5, label_y - text_height - 2), label_text, fill='white', font=font)

            except Exception as e:
                logger.error(f"绘制检测框时出错: {e}, 检测框数据: {box}")
                continue

        # 更新UI
        try:
            label_width = image_label.width() or 400
            label_height = image_label.height() or 300
            resized_img = self._resize_image_to_fit(img_to_draw, label_width, label_height)

            # 将PIL图像转换为QPixmap
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')

            img_array = np.array(resized_img)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            image_label.setPixmap(pixmap)
            image_label.pixmap = pixmap  # 保持引用避免垃圾回收

        except Exception as e:
            logger.error(f"更新图像显示时出错: {e}")
            image_label.clear()
            image_label.setText("图像显示出错")

    def _redraw_preview_boxes_with_new_confidence(self, conf_threshold_str):
        """根据新的置信度阈值，在预览图像上重新绘制检测框"""
        try:
            # ===== 新增:检查原始图像是否存在 =====
            if not self.original_image:
                logger.warning("原始图像未加载，无法绘制检测框")
                return
            # ====================================

            # 确保有有效的置信度值
            if isinstance(conf_threshold_str, int):
                conf_threshold = conf_threshold_str / 100.0
            else:
                conf_threshold = float(conf_threshold_str) / 100.0

            self._draw_detection_boxes(
                image_label=self.image_label,
                original_image=self.original_image,
                detection_info=self.current_preview_info,
                conf_threshold_str=conf_threshold
            )
        except Exception as e:
            logger.error(f"重绘预览检测框失败: {e}")

    def _get_color_for_species(self, species_name):
        """为物种分配一个固定的颜色"""
        if species_name not in self.species_color_map:
            # 使用hash确保同一物种总能得到相同的颜色索引
            color_index = hash(species_name) % len(self.color_palette)
            self.species_color_map[species_name] = self.color_palette[color_index]
        return self.species_color_map[species_name]

    def _on_preview_confidence_slider_changed(self, value):
        """处理预览页置信度滑块值的变化"""
        if self.preview_conf_label:
            self.preview_conf_label.setText(f"{value / 100.0:.2f}")

        # 更新预览置信度变量
        self.preview_conf_var = value / 100.0

        # 如果显示检测结果且有检测信息，则重绘
        if self.show_detection_checkbox.isChecked() and self.current_preview_info:
            self._redraw_preview_boxes_with_new_confidence(value)

    def _on_detection_completed(self, loaded_detection_info, filename):
        """检测完成处理"""
        try:
            if loaded_detection_info:  # 有检测结果
                self.current_preview_info = loaded_detection_info
                # 更新检测信息显示
                self._update_detection_info(loaded_detection_info)

                # 自动显示检测框
                if not self.show_detection_checkbox.isChecked():
                    self.show_detection_checkbox.setChecked(True)
                else:
                    # 如果已经选中，手动触发重绘
                    self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
            else:
                QMessageBox.information(self, "提示", "未检测到任何对象。")

        except Exception as e:
            logger.error(f"处理检测结果失败: {e}")
            QMessageBox.critical(self, "错误", f"处理检测结果失败: {e}")

    def _on_detection_failed(self, error_message):
        """检测失败处理"""
        QMessageBox.critical(self, "错误", f"检测图像失败: {error_message}")

    def _on_detection_finished(self):
        """检测线程结束处理"""
        # 恢复按钮状态
        self.detect_button.setEnabled(True)
        self.detect_button.setText("检测当前图像")

        # 清理线程对象
        if hasattr(self, 'detection_worker'):
            self.detection_worker.deleteLater()
            self.detection_worker = None

    def sync_processing_result(self, img_path, detection_info):
        """同步批量处理的结果到预览页面"""
        try:
            # 获取文件名
            filename = os.path.basename(img_path)

            # 检查当前选中的文件是否是正在处理的文件
            selected_items = self.file_listbox.selectedItems()
            if selected_items:
                current_selected = selected_items[0].text()
                if current_selected == filename:
                    # 如果当前选中的文件正是被处理的文件，更新显示
                    self._update_current_preview_with_processing_result(img_path, detection_info)

        except Exception as e:
            logger.error(f"同步处理结果失败: {e}")

    def _update_current_preview_with_processing_result(self, img_path, detection_info):
        """使用处理结果更新当前预览"""
        try:
            # 更新当前预览信息
            self.current_preview_info = {
                '物种名称': detection_info.get('物种名称', ''),
                '物种数量': detection_info.get('物种数量', ''),
                '最低置信度': detection_info.get('最低置信度', ''),
                '检测时间': detection_info.get('检测时间', ''),
                '检测框': []
            }

            # 转换检测结果为JSON格式
            detect_results = detection_info.get('detect_results')
            if detect_results:
                # 保存检测信息到临时目录
                temp_photo_dir = self.controller.get_temp_photo_dir()
                if temp_photo_dir:
                    base_name, _ = os.path.splitext(os.path.basename(img_path))
                    json_path = os.path.join(temp_photo_dir, f"{base_name}.json")

                    # 如果JSON文件存在，读取完整信息
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            self.current_preview_info = json.load(f)

            # 更新检测信息显示
            self._update_detection_info(self.current_preview_info)

            # 如果显示检测框选项已选中，重新绘制图像
            if self.show_detection_checkbox.isChecked():
                self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
            else:
                # 自动选中显示检测框
                self.show_detection_checkbox.setChecked(True)

        except Exception as e:
            logger.error(f"更新预览处理结果失败: {e}")

    def sync_current_processing_file(self, img_path, current_index, total_files):
        """同步当前处理的文件到预览界面"""
        try:
            filename = os.path.basename(img_path)

            # 在文件列表中找到并选中当前处理的文件
            for i in range(self.file_listbox.count()):
                item = self.file_listbox.item(i)
                if item and item.text() == filename:
                    # 选中当前文件
                    self.file_listbox.setCurrentRow(i)
                    # 确保该项目可见
                    self.file_listbox.scrollToItem(item)
                    break

            # 更新当前图像路径
            self.current_image_path = img_path

            # 显示原始图像（在检测结果出来之前）
            if os.path.exists(img_path):
                self.update_image_preview(img_path)
                self.update_image_info(img_path, filename, is_processing=True)  # 标记为正在处理

            # 更新状态信息
            status_text = f"正在处理第 {current_index}/{total_files} 张图像: {filename}"
            if hasattr(self.controller, 'status_bar'):
                self.controller.status_bar.status_label.setText(status_text)

        except Exception as e:
            logger.error(f"同步当前处理文件失败: {e}")

    def sync_current_processing_result(self, img_path, detection_info):
        """同步当前处理的结果到预览界面"""
        try:
            filename = os.path.basename(img_path)

            # 只有当前选中的文件与处理的文件一致时才更新
            selected_items = self.file_listbox.selectedItems()
            if selected_items and selected_items[0].text() == filename:

                # 更新检测信息
                self.current_preview_info = {
                    '物种名称': detection_info.get('物种名称', ''),
                    '物种数量': detection_info.get('物种数量', ''),
                    '最低置信度': detection_info.get('最低置信度', ''),
                    '检测时间': detection_info.get('检测时间', ''),
                    '检测框': []
                }

                # 如果有检测结果，转换为JSON格式
                detect_results = detection_info.get('detect_results')
                if detect_results:
                    # 从临时目录加载完整的JSON信息
                    temp_photo_dir = self.controller.get_temp_photo_dir()
                    if temp_photo_dir:
                        base_name, _ = os.path.splitext(filename)
                        json_path = os.path.join(temp_photo_dir, f"{base_name}.json")

                        # 等待JSON文件生成（最多等待2秒）
                        wait_count = 0
                        while not os.path.exists(json_path) and wait_count < 20:
                            QTimer.singleShot(100, lambda: None)  # 等待100ms
                            QApplication.processEvents()  # 处理事件循环
                            wait_count += 1

                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    self.current_preview_info = json.load(f)
                            except Exception as e:
                                logger.error(f"读取JSON文件失败: {e}")

                # 更新检测信息显示
                self._update_detection_info(self.current_preview_info)

                # 如果显示检测框选项已选中，显示带检测框的图像
                if self.show_detection_checkbox.isChecked():
                    self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
                else:
                    # 自动选中显示检测框以显示结果
                    self.show_detection_checkbox.setChecked(True)

        except Exception as e:
            logger.error(f"同步当前处理结果失败: {e}")

    def _get_placeholder_style(self):
        """根据主题获取占位符样式"""
        is_dark = self.controller.is_dark_mode if hasattr(self.controller, 'is_dark_mode') else False
        if is_dark:
            return """
                QLabel {
                    border: 2px dashed #444;
                    border-radius: 8px;
                    background-color: #2a2a2a;
                    color: #888;
                    font-size: 16px;
                }
            """
        else:
            return """
                QLabel {
                    border: 2px dashed #e0e0e0;
                    border-radius: 8px;
                    background-color: #fafafa;
                    color: #999999;
                    font-size: 16px;
                }
            """

    def get_settings(self):
        """获取设置"""
        return {
            "preview_conf": self.preview_conf_var,
        }

    def load_settings(self, settings):
        """加载设置"""
        if "preview_conf" in settings:
            self.preview_conf_var = settings["preview_conf"]
            self.preview_conf_slider.setValue(int(self.preview_conf_var * 100))

    def update_theme(self):
        """更新主题（已修复）"""
        # 重新应用样式
        self._apply_theme()
        # 更新图片标签的占位符样式
        if not self.image_label.pixmap:
            self.image_label.setStyleSheet(self._get_placeholder_style())

    def set_show_detection(self, show):
        """设置是否显示检测结果"""
        self.show_detection_checkbox.setChecked(show)

    def on_file_processed(self, img_path, detection_results, filename):
        """处理文件完成回调"""
        # 可以在这里添加文件处理完成后的逻辑
        pass

    def clear_validation_data(self):
        """清除验证数据 - 兼容性方法"""
        pass

    def resizeEvent(self, event):
        """处理窗口大小变化事件，在调整大小时自动缩放图片。"""
        super().resizeEvent(event)
        if hasattr(self, 'original_image') and self.original_image:
            # 延迟100毫秒后执行重绘，避免在快速拖动窗口时过于频繁地刷新，优化性能
            self._resize_timer.start(100)

    def _redraw_image_on_resize(self):
        """根据新的窗口大小重绘当前显示的图片。"""
        if not hasattr(self, 'original_image') or not self.original_image:
            return

        if self.show_detection_checkbox.isChecked() and self.current_preview_info:
            # 如果当前显示的是检测结果，则调用已有的绘制函数，它会自动适应新的标签大小
            self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
        else:
            # 否则，只更新原始图片的预览
            self._update_pixmap_for_label(self.original_image)

    def _update_pixmap_for_label(self, img_to_display):
        """
        一个辅助函数，用于将给定的PIL图像调整大小以适应image_label，并设置其Pixmap。
        """
        if not img_to_display:
            return

        label_size = self.image_label.size()
        max_width = max(label_size.width(), 1)
        max_height = max(label_size.height(), 1)

        resized_img = self._resize_image_to_fit(img_to_display, max_width, max_height)

        try:
            # 将PIL图像转换为QPixmap
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')

            img_array = np.array(resized_img)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.image_label.setPixmap(pixmap)
            # 保持引用以避免pixmap被垃圾回收
            self.image_label.pixmap = pixmap
        except Exception as e:
            logger.error(f"将图像设置为标签时出错: {e}")

    def select_file_by_name(self, filename: str):
        """根据文件名在列表中以编程方式选中文件"""
        for i in range(self.file_listbox.count()):
            item = self.file_listbox.item(i)
            if item and item.text() == filename:
                self.file_listbox.setCurrentItem(item)
                # 滚动到选中项确保其可见
                self.file_listbox.scrollToItem(item)
                return

    def _on_image_loaded(self, pixmap, file_path, image_info):
        """当图片成功加载后，在主线程中更新UI"""
        # 关键修复:仅当加载完成的图片是用户最新请求的那一张时,才更新界面
        if file_path != self.requested_image_path:
            return

        # 更新已加载的路径状态
        self.loaded_image_path = file_path

        # 加载并保存原始图像
        try:
            self.original_image = Image.open(file_path)
            self.current_image_path = file_path
        except Exception as e:
            logger.error(f"加载原始图像失败: {e}")
            self.original_image = None
            self.current_image_path = None

        # 更新图像
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.pixmap = pixmap

        # ===== 修改:更新信息显示，直接从原始图像获取尺寸 =====
        self.info_text.clear()
        info1 = f"文件名: {image_info.get('文件名', '')}    格式: {image_info.get('格式', '')}"
        info2 = f"拍摄日期: {image_info.get('拍摄日期', '未知')} {image_info.get('拍摄时间', '')}    "

        try:
            # 直接从原始图像获取实际尺寸
            if self.original_image:
                img_width, img_height = self.original_image.size
                file_size_kb = os.path.getsize(file_path) / 1024
                info2 += f"尺寸: {img_width}x{img_height}px    文件大小: {file_size_kb:.1f} KB"
            else:
                # 如果原始图像加载失败，尝试从image_info获取
                width = image_info.get('宽度', image_info.get('width', '未知'))
                height = image_info.get('高度', image_info.get('height', '未知'))
                file_size_kb = os.path.getsize(file_path) / 1024
                info2 += f"尺寸: {width}x{height}px    文件大小: {file_size_kb:.1f} KB"
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            info2 += "尺寸: 未知"

        self.info_text.setPlainText(info1 + "\n" + info2)
        # =====================================================

        # 检查并加载已有的检测结果
        self.current_preview_info = {}  # 重置检测信息
        temp_photo_dir = self.controller.get_temp_photo_dir()
        if temp_photo_dir:
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            json_path = os.path.join(temp_photo_dir, f"{base_name}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_preview_info = json.load(f)
                    self._update_detection_info(self.current_preview_info)
                    # 如果已勾选显示检测结果,则绘制检测框
                    if self.show_detection_checkbox.isChecked():
                        self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
                except Exception as e:
                    logger.error(f"加载 {json_path} 文件失败: {e}")

    def _on_loading_failed(self, file_path, error_message):
        """当图片加载失败时显示错误信息"""
        # 关键修复：仅当加载失败的图片是用户最新请求的那一张时，才显示错误
        if file_path != self.requested_image_path:
            return

        self.image_label.setText(f"无法加载图像:\n{error_message}")
        self.image_label.setStyleSheet(self._get_placeholder_style())
        self.loaded_image_path = None  # 加载失败，重置状态
        logger.error(f"图像加载失败: {error_message}")

    def _on_thread_finished(self):
        """当加载线程结束后,清理其在主页面中的引用（增强版）"""
        # === 关键修复5: 延迟清理引用,确保所有信号处理完毕 ===
        QTimer.singleShot(50, self._clear_thread_reference)

    def _clear_thread_reference(self):
        """延迟清理线程引用"""
        self.image_loader_thread = None