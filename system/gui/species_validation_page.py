from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QLabel, QPushButton, QFrame, QGroupBox,
    QMessageBox, QFileDialog, QInputDialog, QComboBox,
    QSizePolicy, QApplication, QDialog, QLineEdit, QFormLayout,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QEvent, QRectF, QPoint
from PySide6.QtGui import (
    QFont, QPalette, QPixmap, QImage, QPainter, QColor,
    QKeySequence, QShortcut, QPainterPath
)
from PySide6.QtSvg import QSvgRenderer
import os
import json
import logging
import cv2
import time
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
import shutil
from collections import defaultdict, Counter

from system.config import SUPPORTED_IMAGE_EXTENSIONS, get_species_color
from system.gui.ui_components import Win11Colors, ModernSlider, ModernGroupBox, ModernComboBox
from system.data_processor import DataProcessor
from system.metadata_extractor import ImageMetadataExtractor

logger = logging.getLogger(__name__)

class CorrectionDialog(QDialog):
    """用于修正物种信息的弹窗"""

    def __init__(self, parent, title="修正信息", original_info=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.parent = parent
        self.result = None
        self.original_info = original_info

        # 设置新的颜色主题
        self.setStyleSheet("""
            QDialog {
                background-color: #dbbcc2;
                border: 1px solid #5d3a4f;
                border-radius: 8px;
            }
            QLabel {
                color: #5d3a4f;
                font-size: 14px;
                background-color: #dbbcc2;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #5d3a4f;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 14px;
                color: #5d3a4f;
            }
            QLineEdit:focus {
                border-color: #5d3a4f;
            }
            QPushButton {
                background-color: #5d3a4f;
                color: #dbbcc2;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #7a5f6f;
            }
            QPushButton:pressed {
                background-color: #4a2f3f;
            }
            QPushButton#cancelButton {
                background-color: #8c7f84;
            }
            QPushButton#cancelButton:hover {
                background-color: #a09398;
            }
        """)

        # 初始化输入框
        self.species_name_edit = QLineEdit()
        self.species_count_edit = QLineEdit()
        self.remark_edit = QLineEdit()

        # 如果有原始信息，则预先填充输入框
        if self.original_info:
            conf_threshold = parent.validation_conf_var if hasattr(parent, 'validation_conf_var') else 0.25

            recalculated_info = {}
            if self.original_info.get('最低置信度') != '人工校验' and '检测框' in self.original_info:
                boxes_info = self.original_info.get("检测框", [])
                filtered_species_counts = Counter()

                for box in boxes_info:
                    confidence = box.get("置信度", 0)
                    if confidence >= conf_threshold:
                        species_name = box.get("物种")
                        if species_name:
                            filtered_species_counts[species_name] += 1

                if not filtered_species_counts:
                    recalculated_info['物种名称'] = "空"
                    recalculated_info['物种数量'] = "空"
                else:
                    recalculated_info['物种名称'] = ",".join(filtered_species_counts.keys())
                    recalculated_info['物种数量'] = ",".join(map(str, filtered_species_counts.values()))
            else:
                recalculated_info['物种名称'] = self.original_info.get('物种名称', '')
                recalculated_info['物种数量'] = self.original_info.get('物种数量', '')

            self.species_name_edit.setText(recalculated_info.get('物种名称', ''))
            self.species_count_edit.setText(recalculated_info.get('物种数量', ''))
            self.remark_edit.setText(self.original_info.get('备注', ''))

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(12)

        form_layout.addRow("正确物种名称:", self.species_name_edit)
        form_layout.addRow("物种数量:", self.species_count_edit)
        form_layout.addRow("备注:", self.remark_edit)

        layout.addLayout(form_layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept_input)
        ok_button.setDefault(True)

        cancel_button = QPushButton("取消")
        cancel_button.setObjectName("cancelButton")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.resize(400, 200)

    def accept_input(self):
        # 将中文逗号替换为英文逗号
        species_name = self.species_name_edit.text().strip().replace('，', ',')
        species_count_str = self.species_count_edit.text().strip().replace('，', ',')
        remark = self.remark_edit.text().strip()

        # 校验物种名称
        if not species_name:
            QMessageBox.warning(self, "输入错误", "物种名称不能为空。")
            return

        if not species_count_str:
            if self.original_info and self.original_info.get('物种数量'):
                try:
                    original_counts = [int(c.strip()) for c in self.original_info['物种数量'].split(',')]
                    species_count_str = str(sum(original_counts))
                except (ValueError, TypeError):
                    species_count_str = '1'
            else:
                species_count_str = '1'

        # 检查物种数量格式
        if species_count_str.lower() != '空':
            try:
                counts = [int(c.strip()) for c in species_count_str.split(',')]
                if not all(c > 0 for c in counts):
                    raise ValueError("数量必须是正整数。")
            except ValueError:
                QMessageBox.warning(
                    self,
                    "输入格式错误",
                    "物种数量必须为以下格式之一：\n\n"
                    "1. 单个正整数 (例如: 3)\n"
                    "2. 以英文逗号隔开的多个正整数 (例如: 5,2)\n"
                    '3. 文字"空"'
                )
                return

        self.result = (species_name, species_count_str, remark)
        self.accept()


class NoArrowKeyListWidget(QListWidget):
    """一个将上下方向键事件传递给父控件的QListWidget子类"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Down:
            # 忽略事件，让它冒泡到父控件进行处理
            event.ignore()
        else:
            # 对于其他按键，保持默认行为
            super().keyPressEvent(event)


class VideoPlayerThread(QThread):
    """
    视频播放线程：读取视频流，绘制检测框（支持中文），并发送 QPixmap。
    包含帧过滤和物种投票逻辑。
    """
    frame_ready = Signal(QPixmap)
    playback_finished = Signal()
    pause_state_changed = Signal(bool)

    def __init__(self, video_path, json_path, conf_threshold, draw_boxes=True, min_frame_ratio=0.0, start_frame=0,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.json_path = json_path
        self.conf_threshold = conf_threshold
        self.draw_boxes = draw_boxes
        self.min_frame_ratio = min_frame_ratio
        self.start_frame = start_frame
        self.running = False
        self.paused = False
        self.current_frame_index = 0

        # 初始化字体
        try:
            from system.utils import resource_path
            self.font_path = resource_path(os.path.join("res", "AlibabaPuHuiTi-3-65-Medium.ttf"))
            self.font = ImageFont.truetype(self.font_path, 20)
            self.font_loaded = True
        except Exception as e:
            logger.warning(f"VideoThread: 字体加载失败 {e}")
            self.font = ImageFont.load_default()
            self.font_loaded = False

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_state_changed.emit(self.paused)

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps

        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # 调整字体大小
        v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self.font_loaded and v_h > 0:
            target_size = max(16, int(0.02 * min(v_w, v_h)))
            try:
                self.font = ImageFont.truetype(self.font_path, target_size)
            except:
                pass

        # 解析并处理 JSON 数据
        frames_data = self._parse_tracking_json()
        stride = frames_data.get('stride', 1)
        detections = frames_data.get('frames', {})

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.paused = True
                self.pause_state_changed.emit(True)
                continue

            self.current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # 绘制检测框
            if self.draw_boxes:
                # 查找最近的关键帧
                lookup_idx = self.current_frame_index - (self.current_frame_index % stride)
                if lookup_idx in detections:
                    self._draw_boxes_pil(pil_img, detections[lookup_idx])

            # 转换为 QImage/QPixmap
            img_data = pil_img.tobytes()
            w, h = pil_img.size
            bytes_per_line = 3 * w
            qt_image = QImage(img_data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            self.frame_ready.emit(QPixmap.fromImage(qt_image))

            process_time = time.time() - start_time
            wait_time = max(0, frame_delay - process_time)
            time.sleep(wait_time)

        cap.release()
        self.playback_finished.emit()

    def _parse_tracking_json(self):
        """解析 JSON，包含帧数过滤和物种投票逻辑"""
        parsed_frames = {'frames': {}, 'stride': 1}

        if not self.json_path or not os.path.exists(self.json_path):
            return parsed_frames

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            parsed_frames['stride'] = data.get('vid_stride', 1)
            total_frames = data.get('total_frames_processed', 0)
            tracks = data.get('tracks', {})
            min_frames_threshold = total_frames * self.min_frame_ratio

            for track_id, track_list in tracks.items():
                # 过滤
                if len(track_list) < min_frames_threshold:
                    continue

                # 投票确定最终物种
                species_list = [p.get('species') for p in track_list if p.get('species')]
                final_species = "Unknown"
                if species_list:
                    final_species = Counter(species_list).most_common(1)[0][0]

                for point in track_list:
                    f_idx = point.get('frame_index')
                    if f_idx is not None:
                        if f_idx not in parsed_frames['frames']:
                            parsed_frames['frames'][f_idx] = []

                        point['track_id'] = track_id
                        point['species'] = final_species  # 应用投票结果
                        parsed_frames['frames'][f_idx].append(point)

        except Exception as e:
            logger.error(f"JSON Parse Error: {e}")

        return parsed_frames

    def _draw_boxes_pil(self, pil_img, boxes):
        draw = ImageDraw.Draw(pil_img)
        for box in boxes:
            conf = box.get('confidence', 0)
            if conf < self.conf_threshold: continue

            bbox = box.get('bbox')
            if not bbox: continue
            x1, y1, x2, y2 = map(int, bbox)

            species = box.get('species', 'Unknown')
            track_id = box.get('track_id', '?')

            rgb_color = get_species_color(species, return_rgb=True)

            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)

            label = f"{species} #{track_id} ({conf:.2f})"
            try:
                text_bbox = draw.textbbox((0, 0), label, font=self.font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except:
                text_w, text_h = draw.textsize(label, font=self.font)

            label_y = max(text_h + 5, y1)
            draw.rectangle([x1, label_y - text_h - 5, x1 + text_w + 10, label_y], fill=rgb_color)
            draw.text((x1 + 5, label_y - text_h - 5), label, fill='white', font=self.font)

    def stop(self):
        self.running = False
        self.wait()


class SpeciesValidationPage(QWidget):
    """物种校验页面"""

    settings_changed = Signal()
    quick_marks_updated = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.species_image_map = defaultdict(list)
        self.current_selected_species = None
        self.current_species_info = {}
        self.species_conf_var = 0.25
        self.export_format_var = "CSV"
        # 从设置中加载导出格式
        if hasattr(self.controller, 'settings_manager'):
            try:
                # 尝试从通用设置中加载
                settings = self.controller.settings_manager.load_settings()
                self.export_format_var = settings.get('export_format', 'CSV')
            except Exception:
                # 如果加载失败，使用默认值
                self.export_format_var = 'CSV'

        self.last_selected_species_image = None
        self.format_combo = None

        # 从preview_page继承的validation_data
        self.validation_data = getattr(controller.preview_page, 'validation_data', {})

        # 标记相关变量
        self._species_marked = None
        self._count_marked = None
        self._selected_species_button = None
        self._selected_quantity_button = None

        self.species_validation_original_image = None

        self._setup_ui()
        self._apply_theme()

        # 监听自动排序开关状态变化
        if hasattr(self.controller, 'advanced_page') and hasattr(self.controller.advanced_page, 'auto_sort_switch_row'):
            self.controller.advanced_page.auto_sort_switch_row.toggled.connect(self._on_auto_sort_changed)

        # 用于处理窗口大小调整的计时器
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._redraw_image_on_resize)

        self.SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
        self.video_thread = None
        self._current_video_frame_pixmap = None
        self._is_video_paused = False

        # 启用事件过滤器以支持点击暂停
        self.species_image_label.installEventFilter(self)

    def _on_auto_sort_changed(self, checked):
        """当自动排序开关状态改变时，重新加载物种按钮"""
        self._load_species_buttons()

    def _apply_theme(self):
        """应用主题样式"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            # Dark theme colors
            bg_color = Win11Colors.DARK_BACKGROUND.name()
            text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            pane_border_color = Win11Colors.DARK_BORDER.name()
            pane_bg_color = Win11Colors.DARK_CARD.name()
            tab_bar_bg_color = Win11Colors.DARK_SURFACE.name()
            tab_text_color = Win11Colors.DARK_TEXT_SECONDARY.name()
            tab_selected_bg_color = Win11Colors.DARK_CARD.name()
            tab_selected_text_color = Win11Colors.DARK_ACCENT.name()
            tab_selected_border_color = Win11Colors.DARK_ACCENT.name()
            tab_hover_bg_color = Win11Colors.DARK_HOVER.name()
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

            # 设置 Win11 风格
            self.setStyleSheet(f"""
                        QWidget {{
                            background-color: {bg_color};
                            color: {text_color};
                            font-family: 'Segoe UI', Arial, sans-serif;
                        }}
                        QTabWidget::pane {{
                            border: 1px solid {pane_border_color};
                            border-radius: 8px;
                            background-color: {pane_bg_color};
                        }}
                        QTabWidget::tab-bar {{
                            alignment: left;
                        }}
                        QTabBar::tab {{
                            background-color: {tab_bar_bg_color};
                            border: 1px solid {pane_border_color};
                            border-bottom: none;
                            border-top-left-radius: 6px;
                            border-top-right-radius: 6px;
                            padding: 8px 16px;
                            margin-right: 2px;
                            color: {tab_text_color};
                        }}
                        QTabBar::tab:selected {{
                            background-color: {tab_selected_bg_color};
                            color: {tab_selected_text_color};
                            border-bottom: 2px solid {tab_selected_border_color};
                        }}
                        QTabBar::tab:hover {{
                            background-color: {tab_hover_bg_color};
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
                        QCheckBox {{
                            font-size: 14px;
                            color: {checkbox_text_color};
                        }}
                        QCheckBox::indicator {{
                            width: 18px;
                            height: 18px;
                            border: 2px solid {checkbox_indicator_border_color};
                            border-radius: 4px;
                            background-color: {checkbox_indicator_bg_color};
                        }}
                        QCheckBox::indicator:checked {{
                            background-color: {checkbox_indicator_checked_bg_color};
                            border-color: {checkbox_indicator_checked_bg_color};
                            image: url(checkmark.png);
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

        else:
            # Light theme colors
            bg_color = Win11Colors.LIGHT_BACKGROUND.name()
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            pane_border_color = Win11Colors.LIGHT_BORDER.name()
            pane_bg_color = Win11Colors.LIGHT_CARD.name()
            tab_bar_bg_color = Win11Colors.LIGHT_SURFACE.name()
            tab_text_color = Win11Colors.LIGHT_TEXT_SECONDARY.name()
            tab_selected_bg_color = Win11Colors.LIGHT_CARD.name()
            tab_selected_text_color = Win11Colors.LIGHT_ACCENT.name()
            tab_selected_border_color = Win11Colors.LIGHT_ACCENT.name()
            tab_hover_bg_color = Win11Colors.LIGHT_HOVER.name()
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
            # 设置 Win11 风格
            self.setStyleSheet(f"""
                        QWidget {{
                            background-color: {bg_color};
                            color: {text_color};
                            font-family: 'Segoe UI', Arial, sans-serif;
                        }}
                        QTabWidget::pane {{
                            border: 1px solid {pane_border_color};
                            border-radius: 8px;
                            background-color: {pane_bg_color};
                        }}
                        QTabWidget::tab-bar {{
                            alignment: left;
                        }}
                        QTabBar::tab {{
                            background-color: {tab_bar_bg_color};
                            border: 1px solid {pane_border_color};
                            border-bottom: none;
                            border-top-left-radius: 6px;
                            border-top-right-radius: 6px;
                            padding: 8px 16px;
                            margin-right: 2px;
                            color: {tab_text_color};
                        }}
                        QTabBar::tab:selected {{
                            background-color: {tab_selected_bg_color};
                            color: {tab_selected_text_color};
                            border-bottom: 2px solid {tab_selected_border_color};
                        }}
                        QTabBar::tab:hover {{
                            background-color: {tab_hover_bg_color};
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
                        QCheckBox {{
                            font-size: 14px;
                            color: {checkbox_text_color};
                        }}
                        QCheckBox::indicator {{
                            width: 18px;
                            height: 18px;
                            border: 2px solid {checkbox_indicator_border_color};
                            border-radius: 4px;
                            background-color: {checkbox_indicator_bg_color};
                        }}
                        QCheckBox::indicator:checked {{
                            background-color: {checkbox_indicator_checked_bg_color};
                            border-color: {checkbox_indicator_checked_bg_color};
                            image: url(checkmark.png);
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

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 创建主要内容区域
        main_frame = QFrame()
        main_layout = QHBoxLayout(main_frame)
        layout.addWidget(main_frame)

        # 左侧面板
        self._create_left_panel(main_layout)

        # 右侧面板
        self._create_right_panel(main_layout)

    def _create_left_panel(self, parent_layout):
        """创建左侧面板"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(220)

        # 物种列表
        species_list_group = ModernGroupBox("物种列表")
        species_list_layout = QVBoxLayout(species_list_group)
        self.species_listbox = NoArrowKeyListWidget()
        self.species_listbox.itemClicked.connect(self._on_species_selected)
        species_list_layout.addWidget(self.species_listbox)
        left_layout.addWidget(species_list_group)

        # 照片文件列表
        photo_list_group = ModernGroupBox("照片文件")
        photo_list_layout = QVBoxLayout(photo_list_group)
        self.species_photo_listbox = QListWidget()
        self.species_photo_listbox.itemSelectionChanged.connect(self._on_species_photo_selected)
        photo_list_layout.addWidget(self.species_photo_listbox)
        left_layout.addWidget(photo_list_group)

        parent_layout.addWidget(left_panel)

    def _create_right_panel(self, parent_layout):
        """创建右侧面板"""
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        # 顶部区域
        top_area_frame = QWidget()
        top_layout = QHBoxLayout(top_area_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 图片显示区域
        self.species_image_display_frame = ModernGroupBox("图片显示") # <--- 修改
        image_display_layout = QVBoxLayout(self.species_image_display_frame)

        self.species_image_label = QLabel("请从左侧列表选择物种和图像")
        self.species_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.species_image_label.setMinimumSize(400, 300)
        self.species_image_label.setStyleSheet(self._get_placeholder_style())
        image_display_layout.addWidget(self.species_image_label)
        top_layout.addWidget(self.species_image_display_frame, 1)

        # 快速标记按钮区域
        self._create_action_buttons(top_layout)

        # 数量按钮区域
        self._create_quantity_buttons(top_layout)

        right_layout.addWidget(top_area_frame, 1)

        # 底部区域
        self._create_bottom_area(right_layout)

        parent_layout.addWidget(right_panel, 1)

    def _create_action_buttons(self, parent_layout):
        """创建操作按钮区域"""
        action_buttons_group = ModernGroupBox("快速标记")
        action_buttons_layout = QVBoxLayout(action_buttons_group)
        action_buttons_group.setFixedWidth(110)  # 保持宽度

        # 设置布局的内边距，确保按钮不会太靠近边框
        action_buttons_layout.setContentsMargins(8, 16, 8, 8)
        action_buttons_layout.setSpacing(6)

        correct_button = QPushButton("正确")
        correct_button.setMaximumWidth(80)
        correct_button.setMinimumWidth(60)
        correct_button.setMinimumHeight(28)
        # 添加自定义样式确保按钮尺寸生效
        correct_button.setStyleSheet("""
            QPushButton {
                max-width: 80px;
                min-width: 60px;
                min-height: 28px;
                padding: 5px 8px;
                font-size: 12px;
            }
        """)
        correct_button.clicked.connect(lambda: self._mark_and_move_to_next(True))
        action_buttons_layout.addWidget(correct_button)

        empty_button = QPushButton("空")
        empty_button.setMaximumWidth(80)
        empty_button.setMinimumWidth(60)
        empty_button.setMinimumHeight(28)
        empty_button.setStyleSheet("""
            QPushButton {
                max-width: 80px;
                min-width: 60px;
                min-height: 28px;
                padding: 5px 8px;
                font-size: 12px;
            }
        """)
        empty_button.clicked.connect(lambda: self._mark_and_move_to_next(species_name="空", count="空"))
        action_buttons_layout.addWidget(empty_button)

        # 创建一个 QScrollArea 来容纳物种按钮
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 删除垂直滚动条

        # 物种按钮容器
        self.species_buttons_frame = QWidget()
        self.species_buttons_layout = QVBoxLayout(self.species_buttons_frame)
        self.species_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.species_buttons_layout.setSpacing(4)
        self.species_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area.setWidget(self.species_buttons_frame)

        # 将滚动区域添加到布局中，它会自动填充可用空间
        action_buttons_layout.addWidget(scroll_area)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        action_buttons_layout.addWidget(separator)

        other_button = QPushButton("其他")
        other_button.setMaximumWidth(80)
        other_button.setMinimumWidth(60)
        other_button.setMinimumHeight(28)
        other_button.setStyleSheet("""
            QPushButton {
                max-width: 80px;
                min-width: 60px;
                min-height: 28px;
                padding: 5px 8px;
                font-size: 12px;
            }
        """)
        other_button.clicked.connect(self._mark_other_species)
        action_buttons_layout.addWidget(other_button)

        parent_layout.addWidget(action_buttons_group)

    def _create_quantity_buttons(self, parent_layout):
        """创建数量按钮区域"""
        self.quantity_buttons_frame = ModernGroupBox("数量")
        quantity_buttons_layout = QVBoxLayout(self.quantity_buttons_frame)
        self.quantity_buttons_frame.setFixedWidth(80)  # 保持宽度

        # 设置布局的内边距和间距
        quantity_buttons_layout.setContentsMargins(6, 12, 6, 6)  # 保持边距
        quantity_buttons_layout.setSpacing(3)  # 保持按钮间距

        # 设置布局对齐方式，确保按钮从顶部开始排列
        quantity_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for i in range(1, 11):
            btn = QPushButton(str(i))
            btn.setMaximumWidth(58)  # 保持宽度
            btn.setMinimumWidth(45)  # 保持宽度
            btn.setFixedHeight(32)  # 从28增加到32

            # 为数量按钮设置自定义样式
            btn.setStyleSheet("""
                QPushButton {
                    max-width: 58px;
                    min-width: 45px;
                    height: 32px;
                    padding: 6px 6px;
                    font-size: 12px;
                    font-weight: 500;
                    text-align: center;
                }
            """)

            btn.clicked.connect(lambda checked, num=i: self._on_quantity_button_press(str(num), btn))
            quantity_buttons_layout.addWidget(btn)

        more_button = QPushButton("更多")
        more_button.setMaximumWidth(58)  # 保持宽度
        more_button.setMinimumWidth(45)  # 保持宽度
        more_button.setFixedHeight(32)  # 从28增加到32

        # 为"更多"按钮设置自定义样式
        more_button.setStyleSheet("""
            QPushButton {
                max-width: 58px;
                min-width: 45px;
                height: 32px;
                padding: 6px 6px;
                font-size: 11px;
                font-weight: 500;
                text-align: center;
            }
        """)

        more_button.clicked.connect(lambda: self._on_quantity_button_press("更多", more_button))
        quantity_buttons_layout.addWidget(more_button)

        # 添加弹性空间，将所有按钮推到顶部
        quantity_buttons_layout.addStretch()

        parent_layout.addWidget(self.quantity_buttons_frame)

    def _create_bottom_area(self, parent_layout):
        """创建底部区域"""
        bottom_area_frame = QWidget()
        bottom_layout = QHBoxLayout(bottom_area_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 检测信息与设置区域
        info_slider_group = ModernGroupBox("检测信息与设置")
        info_slider_layout = QVBoxLayout(info_slider_group)

        self.species_info_label = QLabel("物种:  | 数量:  | 置信度: ")
        info_slider_layout.addWidget(self.species_info_label)

        # 置信度控制
        conf_control_layout = QHBoxLayout()
        conf_control_layout.addWidget(QLabel("置信度阈值:"))

        self.species_conf_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.species_conf_slider.setRange(5, 95)
        self.species_conf_slider.setValue(int(self.species_conf_var * 100))
        self.species_conf_slider.valueChanged.connect(self._on_confidence_slider_changed)
        conf_control_layout.addWidget(self.species_conf_slider)

        self.species_conf_label = QLabel("0.25")
        conf_control_layout.addWidget(self.species_conf_label)

        info_slider_layout.addLayout(conf_control_layout)
        bottom_layout.addWidget(info_slider_group, 1)

        # 导出选项区域
        export_options_group = ModernGroupBox("导出选项")
        export_layout = QHBoxLayout(export_options_group)

        self.format_combo = ModernComboBox()
        self.format_combo.addItems(["CSV", "Excel", "错误照片"])
        self.format_combo.setCurrentText(self.export_format_var)
        self.format_combo.currentTextChanged.connect(self._on_export_format_changed)
        export_layout.addWidget(self.format_combo)

        export_button = QPushButton("导出")
        export_button.clicked.connect(self._dispatch_export)
        export_layout.addWidget(export_button)

        bottom_layout.addWidget(export_options_group)
        parent_layout.addWidget(bottom_area_frame)

    def _on_export_format_changed(self, text):
        """当导出格式改变时保存到设置"""
        self.export_format_var = text
        if hasattr(self.controller, 'settings_manager'):
            try:
                # 加载现有设置
                settings = self.controller.settings_manager.load_settings()
                settings['export_format'] = text
                # 保存设置
                self.controller.settings_manager.save_settings(settings)
            except Exception as e:
                logger.warning(f"保存导出格式设置失败: {e}")

    def _get_placeholder_style(self):
        """获取占位符样式"""
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

    def _load_species_data(self):
        """加载物种数据（支持图片和视频投票分类，优先显示人工校验结果）"""
        photo_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.get_file_path()

        if not photo_dir or not os.path.exists(photo_dir) or not source_dir:
            self.species_listbox.clear()
            self.species_image_map.clear()
            logger.warning(f"数据目录不存在 - photo_dir: {photo_dir}, source_dir: {source_dir}")
            return

        self.species_listbox.clear()
        self.species_image_map.clear()

        confidence_settings = self.controller.confidence_settings

        try:
            # 查找图片和视频文件
            source_files = [
                f for f in os.listdir(source_dir)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) or
                   f.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)
            ]
            image_basename_map = {os.path.splitext(f)[0]: f for f in source_files}
            logger.info(f"找到 {len(source_files)} 个媒体文件")
        except FileNotFoundError:
            logger.error(f"源目录未找到: {source_dir}")
            return

        try:
            json_files = [f for f in os.listdir(photo_dir) if f.lower().endswith('.json') and f != 'validation.json']
        except Exception as e:
            logger.error(f"读取临时目录失败: {e}")
            return

        all_species_keys = set()

        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            image_filename = image_basename_map.get(base_name)

            if not image_filename:
                continue

            json_path = os.path.join(photo_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    detection_info = json.load(f)

                valid_detections = []
                species_name = "标记为空"  # 默认值

                # ==================== 1. 优先处理：人工校验 (无论是视频还是图片) ====================
                # 如果用户手动修改过，直接采纳结果，不重新进行轨迹投票或置信度过滤
                if detection_info.get('最低置信度') == '人工校验':
                    species_name = detection_info.get('物种名称', '标记为空')
                    if species_name in ["", "未知", None]:
                        species_name = "标记为空"
                    # 创建一个虚拟的有效检测结果，确保文件被添加到列表
                    valid_detections = [{'置信度': 1.0}]

                # ==================== 2. 视频文件处理逻辑 (自动检测结果) ====================
                # 只有在非人工校验的情况下，才去根据 tracks 重新计算物种
                elif 'tracks' in detection_info:
                    total_frames = detection_info.get('total_frames_processed', 1)
                    tracks = detection_info.get('tracks', {})

                    # 获取高级设置中的最低帧数比例
                    min_frame_ratio = 0.0
                    if hasattr(self.controller, 'advanced_page'):
                        min_frame_ratio = self.controller.advanced_page.min_frame_ratio_var
                    threshold = total_frames * min_frame_ratio

                    detected_species_set = set()

                    # 遍历所有轨迹进行投票
                    for track_id, points in tracks.items():
                        # 过滤掉帧数不足的目标
                        if len(points) < threshold:
                            continue

                        # 收集该轨迹中所有帧的物种识别结果
                        s_list = [p.get('species') for p in points if p.get('species')]
                        if s_list:
                            # 使用 Counter 投票，选出该轨迹出现次数最多的物种
                            most_common_species = Counter(s_list).most_common(1)[0][0]
                            detected_species_set.add(most_common_species)

                    if detected_species_set:
                        # 将所有检测到的物种名称合并
                        species_name = ",".join(sorted(list(detected_species_set)))
                        valid_detections = [{'置信度': 1.0}]
                    else:
                        species_name = "标记为空"

                # ==================== 3. 图片文件处理逻辑 (自动检测结果) ====================
                else:
                    species_name = detection_info.get('物种名称', '未知')
                    if species_name in ["", "未知", None]:
                        species_name = "标记为空"

                    # 应用置信度筛选
                    global_conf = confidence_settings.get("global", 0.25)
                    species_conf = confidence_settings.get(species_name, global_conf)

                    # 检查检测框信息
                    detection_boxes = detection_info.get('检测框', [])

                    if detection_boxes:
                        valid_detections = [
                            det for det in detection_boxes
                            if det.get('置信度', det.get('confidence', 0)) >= species_conf
                        ]
                    elif species_name == "标记为空":
                        valid_detections = [{'置信度': 1.0}]

                # 将文件添加到对应的物种列表中
                if valid_detections or species_name in ["标记为空", "空"]:
                    all_species_keys.add(species_name)
                    self.species_image_map[species_name].append(image_filename)
                    logger.debug(f"添加物种 {species_name}: {image_filename}")

            except Exception as e:
                logger.error(f"处理文件 {json_file} 时出错: {e}")
                continue

        # 排序并刷新列表
        sorted_species = sorted(list(all_species_keys), key=lambda x: (x in ["标记为空", "空"], x))

        for species in sorted_species:
            count = len(self.species_image_map[species])
            display_text = f"{species} ({count})"
            self.species_listbox.addItem(display_text)
            logger.info(f"添加物种到列表: {display_text}")

        logger.info(f"物种校验页面加载完成，共 {len(sorted_species)} 个物种")
        self._load_species_buttons()

    def _on_species_selected(self):
        """物种选择事件处理"""
        selected_items = self.species_listbox.selectedItems()
        if not selected_items:
            return

        # 清空照片列表和信息显示
        self.species_photo_listbox.clear()
        self.species_image_label.clear()
        self.species_image_label.setText("请从左侧列表选择物种和图像")
        if hasattr(self.species_image_label, 'pixmap'):
            self.species_image_label.pixmap = None
        self.species_info_label.setText("物种:  | 数量:  | 置信度: ")

        # 解析选中的物种名称（移除数量部分）
        selected_text = selected_items[0].text()
        # 从 "物种名称 (数量)" 格式中提取物种名称
        species_name = selected_text.split(' (')[0] if ' (' in selected_text else selected_text

        self.current_selected_species = species_name
        image_files = self.species_image_map.get(species_name, [])

        photo_count = len(image_files)
        if hasattr(self.controller, 'status_bar'):
            self.controller.status_bar.status_label.setText(f"当前物种共有 {photo_count} 张照片")

        # 根据选择的物种来决定是否显示置信度滑块
        if species_name in ["标记为空", "空"]:
            self.species_conf_slider.setEnabled(False)
            self.species_conf_label.setText("N/A")
        else:
            self.species_conf_slider.setEnabled(True)
            self._update_confidence_label(self.species_conf_slider.value())

        # 添加图片到列表
        for image_file in image_files:
            self.species_photo_listbox.addItem(image_file)
            logger.debug(f"添加图片到列表: {image_file}")

        # 如果照片列表不为空，则自动选择第一个
        if self.species_photo_listbox.count() > 0:
            self.species_photo_listbox.setCurrentRow(0)
            logger.info(f"物种 {species_name} 已选择，共 {photo_count} 张照片")
        else:
            logger.warning(f"物种 {species_name} 没有找到对应的照片")

    def _on_species_photo_selected(self):
        """当选择物种照片/视频时的处理"""
        # 1. 清理工作
        self._species_marked = None
        self._count_marked = None
        self._stop_video_thread()  # 停止之前的视频线程
        self._reset_quantity_buttons()  # 封装了重置按钮样式的逻辑

        selection = self.species_photo_listbox.selectedItems()
        if not selection:
            self.species_info_label.setText("物种: - | 数量: - | 置信度: -")
            return

        file_name = selection[0].text()
        self.last_selected_species_image = file_name

        # 路径准备
        source_dir = self.controller.start_page.get_file_path()
        media_path = os.path.join(source_dir, file_name)

        # JSON 路径
        photo_dir = self.controller.get_temp_photo_dir()
        json_path = None
        if photo_dir:
            json_path = os.path.join(photo_dir, f"{os.path.splitext(file_name)[0]}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_species_info = json.load(f)
                    self._update_detection_info_display()
                except:
                    pass

        # === 2. 判断文件类型 ===
        if file_name.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS):
            # === 视频处理逻辑 ===
            if not os.path.exists(media_path):
                self.species_image_label.setText("视频文件不存在")
                return

            self.species_image_label.setText("正在加载视频...")
            self._start_video_thread(media_path, json_path)

        else:
            # === 图片处理逻辑 (原有代码) ===
            try:
                self.species_validation_original_image = Image.open(media_path)
                # 显示带检测框的图像
                self._display_image_with_detection_boxes(
                    self.species_validation_original_image,
                    self.current_species_info,
                    self.species_conf_var
                )
            except Exception as e:
                logger.error(f"加载图像失败: {e}")
                self.species_image_label.setText("无法加载图像")

    def _reset_quantity_buttons(self):
        """重置数量按钮样式的辅助函数"""
        if hasattr(self, '_selected_quantity_button') and self._selected_quantity_button:
            # ... (原有的重置样式代码) ...
            # 这里可以复用您现有的 styleSheet 字符串
            self._selected_quantity_button.setStyleSheet("""
                QPushButton {
                    max-width: 58px; min-width: 45px; height: 32px;
                    padding: 6px 6px; font-size: 12px; font-weight: 500;
                    text-align: center;
                }
            """)
            self._selected_quantity_button = None

    def _display_image(self, image_path):
        """显示图像到标签中"""
        try:
            from PySide6.QtGui import QPixmap
            import numpy as np

            # 加载并调整图像大小
            with Image.open(image_path) as img:
                # 获取显示区域大小
                label_size = self.species_image_label.size()
                max_width = max(label_size.width(), 400)
                max_height = max(label_size.height(), 300)

                # 调整图像大小
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

                # 转换为QPixmap
                img_array = np.array(img.convert('RGB'))
                height, width, channel = img_array.shape
                bytes_per_line = 3 * width

                from PySide6.QtGui import QImage
                q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                self.species_image_label.setPixmap(pixmap)
                self.species_image_label.setScaledContents(False)
                self.species_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            logger.error(f"显示图像失败: {e}")
            self.species_image_label.setText("无法显示图像")

    def _mark_and_move_to_next(self, is_correct=None, species_name=None, count=None):
        """
        处理标记逻辑并根据条件跳转到下一张图片。
        """
        selection = self.species_photo_listbox.selectedItems()
        if not selection:
            return

        file_name = selection[0].text()

        # "Correct" 和 "Empty" 按钮仍然会立即跳转
        if is_correct is True:
            self.validation_data[file_name] = True
            self._save_validation_data()
            self._move_to_next_image()
            # 将焦点设置回照片列表
            self.species_photo_listbox.setFocus()
            return
        if species_name == "空" and count == "空":
            self._update_json_file(file_name, new_species="空", new_count="空")
            self.validation_data[file_name] = False
            self._save_validation_data()
            self._move_to_next_image()
            # 将焦点设置回照片列表
            self.species_photo_listbox.setFocus()
            return

        # 处理物种按钮点击
        if species_name:
            self._species_marked = species_name
            self._increment_quick_mark_count(species_name)
            # 如果自动排序开启，则刷新按钮列表
            if hasattr(self.controller,
                       'advanced_page') and self.controller.advanced_page.auto_sort_switch_row.isChecked():
                self.controller.advanced_page.update_auto_sorted_list()
                self._load_species_buttons()
                self.quick_marks_updated.emit()

            new_count_str = None
            # 如果数量已经选择，则使用它
            if self._count_marked is not None:
                new_count_str = str(self._count_marked)
            # 否则，应用求和逻辑
            elif self.current_species_info and '物种数量' in self.current_species_info:
                count_str = str(self.current_species_info.get('物种数量', ''))
                if ',' in count_str:
                    try:
                        counts = [int(c.strip()) for c in count_str.split(',')]
                        new_count_str = str(sum(counts))
                    except (ValueError, TypeError):
                        new_count_str = None

            self._update_json_file(file_name, new_species=self._species_marked, new_count=new_count_str)
            self.validation_data[file_name] = False
            self._save_validation_data()

            # 如果数量也已选择，则移动到下一张图片
            if self._count_marked is not None:
                self._move_to_next_image()

            # 将焦点设置回照片列表
            self.species_photo_listbox.setFocus()

    def _mark_other_species(self):
        """处理"其他"按钮的逻辑，弹出对话框"""
        selected_items = self.species_photo_listbox.selectedItems()
        if not selected_items:
            return

        file_name = selected_items[0].text()

        dialog = CorrectionDialog(self, title="输入其他物种信息", original_info=self.current_species_info)
        # 执行对话框并检查用户是否点击了"确定"
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result:
            species_name, species_count, remark = dialog.result
            self._update_json_file(file_name, new_species=species_name, new_count=species_count, new_remark=remark)
            # 标记为错误并跳转
            self._mark_as_error_and_save(file_name)
            self._increment_quick_mark_count(species_name)

            # 如果自动排序开启，则刷新按钮列表
            if hasattr(self.controller,
                       'advanced_page') and self.controller.advanced_page.auto_sort_switch_row.isChecked():
                self.controller.advanced_page.update_auto_sorted_list()
                self._load_species_buttons()
                self.quick_marks_updated.emit()

            self._move_to_next_image()

        # 无论对话框结果如何，都将焦点设置回照片列表
        self.species_photo_listbox.setFocus()

    def _load_species_buttons(self):
        """根据自动排序设置，加载快速标记物种按钮"""
        # 1. 清空现有的物种按钮
        while self.species_buttons_layout.count():
            child = self.species_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # 2. 检查设置管理器是否存在
        if not hasattr(self.controller, 'settings_manager'):
            return

        # 3. 加载快速标记数据
        quick_marks_data = self.controller.settings_manager.load_quick_mark_species()

        # 4. 检查自动排序开关的状态
        use_auto_sort = False
        if hasattr(self.controller, 'advanced_page'):
            use_auto_sort = self.controller.advanced_page.auto_sort_switch_row.isChecked()

        # 5. 根据开关状态选择要显示的列表
        if use_auto_sort and "list_auto" in quick_marks_data:
            species_to_display = quick_marks_data["list_auto"]
        else:
            species_to_display = quick_marks_data.get("list", [])

        # 6. 为列表中的每个物种创建按钮
        for species in species_to_display:
            btn = QPushButton(species)
            btn.setMaximumWidth(80)  # 使用最大宽度
            btn.setMinimumWidth(60)  # 设置最小宽度

            # 根据文字长度动态调整高度（稍微减少）
            text_length = len(species)
            if text_length <= 3:
                # 短文字使用标准高度
                min_height = 24  # 从28减少到24
                padding = "3px 6px"
                font_size = "12px"
            elif text_length <= 6:
                # 中等长度文字增加高度
                min_height = 28  # 从32减少到28
                padding = "4px 6px"
                font_size = "11px"
            else:
                # 长文字使用更大高度，允许换行
                min_height = 34  # 从40减少到34
                padding = "5px 4px"
                font_size = "10px"

            btn.setMinimumHeight(min_height)

            # 为动态按钮设置自定义样式
            btn.setStyleSheet(f"""
                QPushButton {{
                    max-width: 80px;
                    min-width: 60px;
                    min-height: {min_height}px;
                    padding: {padding};
                    font-size: {font_size};
                    text-align: center;
                    word-wrap: break-word;
                }}
            """)

            # 对于长文字，设置允许换行
            if text_length > 6:
                btn.setWordWrap(True)

            # 连接点击信号，传递物种名称
            btn.clicked.connect(lambda checked=False, s=species: self._mark_and_move_to_next(species_name=s))
            self.species_buttons_layout.addWidget(btn)

    def _on_quantity_button_press(self, count, btn_widget):
        """处理数量按钮点击事件，并管理按钮状态"""
        selection = self.species_photo_listbox.selectedItems()
        if not selection:
            return

        file_name = selection[0].text()

        final_count = count
        if count == "更多":
            from PySide6.QtWidgets import QInputDialog
            result, ok = QInputDialog.getInt(self, "输入数量", "请输入物种的数量:", 1, 1, 999, 1)
            if ok:
                final_count = result
            else:
                # 取消时也要将焦点设置回照片列表
                self.species_photo_listbox.setFocus()
                return

        # 取消上一个数量按钮的选中状态
        if hasattr(self, '_selected_quantity_button') and self._selected_quantity_button:
            # 重置之前选中按钮的样式
            self._selected_quantity_button.setStyleSheet("""
                QPushButton {
                    max-width: 58px;
                    min-width: 45px;
                    height: 32px;
                    padding: 6px 6px;
                    font-size: 12px;
                    font-weight: 500;
                    text-align: center;
                }
            """)

        # 设置新按钮为选中状态
        btn_widget.setStyleSheet("""
            QPushButton {
                max-width: 58px;
                min-width: 45px;
                height: 32px;
                padding: 6px 6px;
                font-size: 12px;
                font-weight: 500;
                text-align: center;
                background-color: #5d3a4f;
                color: white;
            }
        """)
        self._selected_quantity_button = btn_widget
        self._count_marked = final_count

        self._mark_as_error_and_save(file_name)

        # 如果物种也已选择，则更新JSON并跳转
        new_species = self._species_marked if self._species_marked is not None else None
        self._update_json_file(file_name, new_species=new_species, new_count=str(self._count_marked))

        # 如果物种也已选择，则移动到下一张图片
        if self._species_marked is not None:
            self._move_to_next_image()

        # 将焦点设置回照片列表
        self.species_photo_listbox.setFocus()

    def _on_confidence_slider_changed(self, value):
        """处理置信度滑块值的变化"""
        self._update_confidence_label(value)

        # 重新计算并更新信息标签
        if hasattr(self, 'current_species_info') and self.current_species_info:
            self._update_detection_info_display()

        # 实时重新绘制检测框
        if (hasattr(self, 'species_validation_original_image') and
                self.species_validation_original_image and
                hasattr(self, 'current_species_info') and
                self.current_species_info):
            conf_threshold = value / 100.0 if isinstance(value, int) else value
            self._display_image_with_detection_boxes(
                self.species_validation_original_image,
                self.current_species_info,
                conf_threshold
            )

        # 保存置信度设置
        if not self.current_selected_species or self.current_selected_species == "标记为空":
            return

        species_name = self.current_selected_species
        new_conf = value / 100.0 if isinstance(value, int) else value

        if hasattr(self.controller, 'confidence_settings'):
            self.controller.confidence_settings[species_name] = new_conf
            if hasattr(self.controller, 'settings_manager'):
                self.controller.settings_manager.save_confidence_settings(self.controller.confidence_settings)

    def _export_validation_data(self):
        """从校验页面的数据导出为表格文件（Excel或CSV）"""
        temp_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.get_file_path()

        if not temp_dir or not os.path.exists(temp_dir) or not source_dir:
            QMessageBox.critical(self, "错误", "无法找到临时文件或源文件路径，请确保已进行批处理并且路径设置正确。")
            return

        json_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.json') and f != 'validation.json']
        if not json_files:
            QMessageBox.information(self, "提示", "没有找到任何处理后的数据，无法导出。")
            return

        # 根据下拉框选择确定文件类型
        file_format = self.export_format_var.lower()
        if file_format == 'excel':
            file_types = "Excel 文件 (*.xlsx);;所有文件 (*.*)"
            file_extension = ".xlsx"
        elif file_format == 'csv':
            file_types = "CSV 文件 (*.csv);;所有文件 (*.*)"
            file_extension = ".csv"
        else:
            return  # 如果格式未知则不执行操作

        default_filename = f"validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"

        # 弹出文件保存对话框
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择表格保存位置",
            default_filename,
            file_types
        )

        if not output_path:
            return

        # 加载置信度配置文件
        confidence_settings = self.controller.settings_manager.load_confidence_settings()
        if not confidence_settings:
            confidence_settings = {}

        all_image_data = []
        earliest_date = None

        # === 修复开始：同时支持图片和视频文件的查找与元数据提取 ===
        for json_file in json_files:
            json_path = os.path.join(temp_dir, json_file)
            image_filename_base = os.path.splitext(json_file)[0]

            found_path = None
            is_video = False

            # 1. 构建搜索列表：合并图片和视频扩展名
            search_extensions = list(SUPPORTED_IMAGE_EXTENSIONS)
            if hasattr(self, 'SUPPORTED_VIDEO_EXTENSIONS'):
                search_extensions.extend(list(self.SUPPORTED_VIDEO_EXTENSIONS))

            # 2. 查找对应的源文件
            for ext in search_extensions:
                temp_path = os.path.join(source_dir, image_filename_base + ext)
                if os.path.exists(temp_path):
                    found_path = temp_path
                    # 判断是否为视频
                    if hasattr(self, 'SUPPORTED_VIDEO_EXTENSIONS') and ext.lower() in self.SUPPORTED_VIDEO_EXTENSIONS:
                        is_video = True
                    break

            if not found_path:
                logger.warning(f"找不到原始文件: {image_filename_base}")
                continue

            try:
                metadata = {}

                # 3. 根据文件类型提取元数据
                if is_video:
                    # 视频：手动构建基础元数据
                    metadata['文件名'] = os.path.basename(found_path)
                    metadata['格式'] = os.path.splitext(found_path)[1].replace('.', '').upper()

                    # 使用文件修改时间作为拍摄时间
                    mtime = os.path.getmtime(found_path)
                    dt_obj = datetime.fromtimestamp(mtime)
                    metadata['拍摄日期'] = dt_obj.strftime("%Y-%m-%d")
                    metadata['拍摄时间'] = dt_obj.strftime("%H:%M:%S")
                    metadata['拍摄日期对象'] = dt_obj  # 用于后续排序和独立探测计算
                else:
                    # 图片：使用元数据提取器（支持EXIF）
                    metadata, _ = ImageMetadataExtractor.extract_metadata(found_path, os.path.basename(found_path))

                # 4. 加载JSON检测结果
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                metadata.update(json_data)
                all_image_data.append(metadata)

                date_taken = metadata.get('拍摄日期对象')
                if date_taken:
                    if earliest_date is None or date_taken < earliest_date:
                        earliest_date = date_taken
            except Exception as e:
                logger.error(f"处理文件 {json_file} 时出错: {e}")
        # === 修复结束 ===

        if not all_image_data:
            QMessageBox.critical(self, "错误", "未能成功处理任何数据，无法导出。")
            return

        # 获取检测过滤比例
        min_frame_ratio = 0.0
        if hasattr(self.controller, 'advanced_page'):
            min_frame_ratio = self.controller.advanced_page.min_frame_ratio_var

        # 处理数据 (传递 min_frame_ratio)
        processed_data = DataProcessor.process_independent_detection(
            all_image_data,
            confidence_settings,
            min_frame_ratio=min_frame_ratio
        )

        if earliest_date:
            processed_data = DataProcessor.calculate_working_days(processed_data, earliest_date)

        # 从高级设置页面获取用户选择的导出列
        columns_to_export = self.controller.advanced_page.get_selected_export_columns()

        # 导出文件 (传递 min_frame_ratio)
        success = DataProcessor.export_to_excel(
            processed_data,
            output_path,
            confidence_settings,
            file_format=file_format,
            columns_to_export=columns_to_export,
            min_frame_ratio=min_frame_ratio
        )

        if success:
            reply = QMessageBox.question(self, "成功", f"数据已成功导出到:\n{output_path}\n\n是否立即打开文件？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.startfile(output_path)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法打开文件: {e}")
        else:
            QMessageBox.critical(self, "导出失败", "导出文件时发生错误，请查看日志文件获取详情。")

    def _export_error_images(self):
        """导出被标记为错误的照片"""
        try:
            source_dir = self.controller.start_page.get_file_path()
            if not source_dir or not os.path.exists(source_dir):
                QMessageBox.warning(self, "错误", "源图像目录未设置或不存在。")
                return

            temp_photo_dir = self.controller.get_temp_photo_dir()
            if not temp_photo_dir or not os.path.exists(temp_photo_dir):
                QMessageBox.warning(self, "错误", "临时目录不存在，无法检查JSON文件。")
                return

            # Get destination directory
            dest_dir = self.controller.start_page.get_save_path()
            if not dest_dir or not os.path.isdir(dest_dir):
                dest_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
                if not dest_dir:
                    return  # User cancelled

            error_dir = os.path.join(dest_dir, "error")
            os.makedirs(error_dir, exist_ok=True)

            copied_count = 0
            json_files = [f for f in os.listdir(temp_photo_dir) if f.lower().endswith('.json')]
            for json_file in json_files:
                json_path = os.path.join(temp_photo_dir, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("最低置信度") == "人工校验":
                            species_name = data.get("物种名称", "unknown")
                            if species_name == "空":
                                species_name = "空"

                            species_dir = os.path.join(error_dir, species_name)
                            os.makedirs(species_dir, exist_ok=True)

                            image_filename_base = os.path.splitext(json_file)[0]

                            # Find the corresponding image file (support multiple extensions)
                            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                                image_filename = image_filename_base + ext
                                source_path = os.path.join(source_dir, image_filename)
                                if os.path.exists(source_path):
                                    dest_path = os.path.join(species_dir, image_filename)
                                    shutil.copy2(source_path, dest_path)
                                    copied_count += 1
                                    break
                except Exception as e:
                    logger.error(f"处理JSON文件 {json_file} 时出错: {e}")

            if copied_count > 0:
                QMessageBox.information(self, "成功",
                                        f"已成功导出 {copied_count} 张错误照片到:\n{error_dir}")
            else:
                QMessageBox.information(self, "提示", "没有错误照片可供导出。")

        except Exception as e:
            logger.error(f"导出错误照片失败: {e}")
            QMessageBox.critical(self, "导出失败", f"导出错误照片时发生错误: {e}")

    def _dispatch_export(self):
        """根据下拉框的选择来分派导出任务"""
        export_type = self.export_format_var
        if export_type == "错误照片":
            self._export_error_images()
        elif export_type in ["Excel", "CSV"]:
            self._export_validation_data()
        else:
            QMessageBox.warning(self, "错误", f"未知的导出格式: {export_type}")

    def get_settings(self):
        """获取设置"""
        return {
            "species_conf": self.species_conf_var,
            "export_format": self.export_format_var,
        }

    def load_settings(self, settings):
        """加载设置"""
        if "species_conf" in settings:
            self.species_conf_var = settings["species_conf"]
            self.species_conf_slider.setValue(int(self.species_conf_var * 100))

        if "export_format" in settings:
            self.export_format_var = settings["export_format"]

    def update_theme(self):
        """更新主题"""
        self._apply_theme()

    def _mark_as_error_and_save(self, file_name):
        """标记文件为错误并保存验证数据"""
        if not hasattr(self, 'validation_data'):
            self.validation_data = {}

        self.validation_data[file_name] = False  # 标记为错误
        self._save_validation_data()

    def _save_validation_data(self):
        """保存验证数据到文件"""
        try:
            temp_photo_dir = self.controller.get_temp_photo_dir()
            if temp_photo_dir:
                validation_file_path = os.path.join(temp_photo_dir, "validation.json")
                with open(validation_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.validation_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存验证数据失败: {e}")

    def _increment_quick_mark_count(self, species_name):
        """增加快速标记物种的使用次数"""
        if hasattr(self.controller, 'settings_manager'):
            quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
            if species_name in quick_marks_data:
                quick_marks_data[species_name] = quick_marks_data.get(species_name, 0) + 1
            else:
                quick_marks_data[species_name] = 1
            self.controller.settings_manager.save_quick_mark_species(quick_marks_data)

    def _update_json_file(self, file_name, new_species=None, new_count=None, new_remark=None):
        """更新JSON文件中的物种信息"""
        try:
            temp_photo_dir = self.controller.get_temp_photo_dir()
            if not temp_photo_dir:
                return

            base_name = os.path.splitext(file_name)[0]
            json_path = os.path.join(temp_photo_dir, f"{base_name}.json")

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    detection_info = json.load(f)
            else:
                detection_info = {}

            # 更新信息
            if new_species is not None:
                detection_info['物种名称'] = new_species
            if new_count is not None:
                detection_info['物种数量'] = new_count
            if new_remark is not None:
                detection_info['备注'] = new_remark

            # 标记为人工校验
            detection_info['最低置信度'] = '人工校验'
            detection_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 保存文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_info, f, ensure_ascii=False, indent=2)

            # 更新当前信息
            self.current_species_info = detection_info

            # 更新显示
            self._update_detection_info_display()

        except Exception as e:
            logger.error(f"更新JSON文件失败: {e}")

    def _update_detection_info_display(self):
        """更新检测信息显示（支持图片和视频）"""
        if not (hasattr(self, 'species_info_label') and self.current_species_info):
            return

        try:
            # 获取当前置信度阈值
            conf_threshold = self.species_conf_var

            # 默认值
            species_name = "未知"
            species_count = "0"
            confidence = "N/A"

            # === 情况A：人工校验 ===
            if self.current_species_info.get('最低置信度') == '人工校验':
                species_name = self.current_species_info.get('物种名称', '未知')
                species_count = self.current_species_info.get('物种数量', '未知')
                confidence = '人工校验'

            # === 情况B：视频数据 (tracks) ===
            elif 'tracks' in self.current_species_info:
                total_frames = self.current_species_info.get('total_frames_processed', 1)
                tracks = self.current_species_info.get('tracks', {})

                # 获取检测过滤比例
                min_frame_ratio = 0.0
                if hasattr(self.controller, 'advanced_page'):
                    min_frame_ratio = self.controller.advanced_page.min_frame_ratio_var
                threshold = total_frames * min_frame_ratio

                species_counts_map = Counter()
                min_conf_val = float('inf')
                valid_tracks_found = False

                for track_id, points in tracks.items():
                    # 1. 过滤掉帧数不足的目标
                    if len(points) < threshold:
                        continue

                    # 2. 过滤掉置信度不足的点
                    valid_points = [p for p in points if p.get('confidence', 0) >= conf_threshold]

                    # 只有当该轨迹包含满足置信度的点时，才算作有效检测
                    if valid_points:
                        valid_tracks_found = True

                        # 投票确定该轨迹的物种
                        s_list = [p.get('species') for p in valid_points if p.get('species')]
                        if s_list:
                            most_common = Counter(s_list).most_common(1)[0][0]
                            species_counts_map[most_common] += 1

                        # 更新最低置信度 (基于有效点)
                        for p in valid_points:
                            min_conf_val = min(min_conf_val, p.get('confidence', 1.0))

                if valid_tracks_found:
                    # 排序以保证显示顺序稳定
                    sorted_species = sorted(species_counts_map.keys())
                    species_name = ','.join(sorted_species)
                    species_count = ','.join([str(species_counts_map[s]) for s in sorted_species])
                    confidence = f"{min_conf_val:.2f}"
                else:
                    species_name = "空"
                    species_count = "0"
                    confidence = "N/A"

            # === 情况C：图片数据 (检测框) ===
            else:
                detection_boxes = self.current_species_info.get('检测框', [])
                if detection_boxes:
                    # 过滤满足置信度要求的检测框
                    valid_boxes = [
                        box for box in detection_boxes
                        if box.get('置信度', box.get('confidence', 0)) >= conf_threshold
                    ]

                    if valid_boxes:
                        species_counts_map = Counter()
                        min_conf_val = float('inf')

                        for box in valid_boxes:
                            box_species = box.get('物种', box.get('species', '未知'))
                            box_conf = box.get('置信度', box.get('confidence', 0))
                            species_counts_map[box_species] += 1
                            min_conf_val = min(min_conf_val, box_conf)

                        sorted_species = sorted(species_counts_map.keys())
                        species_name = ','.join(sorted_species)
                        species_count = ','.join([str(species_counts_map[s]) for s in sorted_species])
                        confidence = f"{min_conf_val:.2f}"
                    else:
                        species_name = "空"
                        species_count = "0"
                        confidence = "N/A"
                else:
                    # 如果原JSON就没有检测框，可能直接读取了统计字段
                    species_name = self.current_species_info.get('物种名称', '未知')
                    species_count = self.current_species_info.get('物种数量', '未知')
                    confidence = self.current_species_info.get('最低置信度', '未知')

            # 更新 UI 标签
            info_text = f"物种: {species_name} | 数量: {species_count} | 置信度: {confidence}"
            self.species_info_label.setText(info_text)

        except Exception as e:
            logger.error(f"更新检测信息显示失败: {e}")
            self.species_info_label.setText("信息显示错误")

    def _move_to_next_image(self):
        """移动到下一张图像"""
        current_row = self.species_photo_listbox.currentRow()
        next_row = current_row + 1

        if next_row < self.species_photo_listbox.count():
            self.species_photo_listbox.setCurrentRow(next_row)
        else:
            # 如果是最后一张，可以选择跳转到下一个物种或显示完成消息
            QMessageBox.information(self, "提示", "当前物种的所有图像已处理完成！")

    def _update_confidence_label(self, value):
        """更新置信度标签"""
        if hasattr(self, 'species_conf_label'):
            confidence_value = value / 100.0 if isinstance(value, int) else value
            self.species_conf_label.setText(f"{confidence_value:.2f}")
            self.species_conf_var = confidence_value

    def _on_confidence_slider_changed(self, value):
        """处理置信度滑块值的变化"""
        self._update_confidence_label(value)

        # 转换值为小数 (例如 25 -> 0.25)
        new_conf = value / 100.0 if isinstance(value, int) else value

        # 1. 重新计算并更新信息标签 (文本)
        if hasattr(self, 'current_species_info') and self.current_species_info:
            self._update_detection_info_display()

        # 2. 实时重绘图片检测框
        if (hasattr(self, 'species_validation_original_image') and
                self.species_validation_original_image and
                hasattr(self, 'current_species_info') and
                self.current_species_info):
            self._display_image_with_detection_boxes(
                self.species_validation_original_image,
                self.current_species_info,
                new_conf
            )

        # 3. === 新增：实时更新视频播放器的阈值 ===
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.conf_threshold = new_conf
        # ======================================

        # 4. 保存置信度设置到配置文件
        if not self.current_selected_species or self.current_selected_species == "标记为空":
            return

        species_name = self.current_selected_species
        if hasattr(self.controller, 'confidence_settings'):
            self.controller.confidence_settings[species_name] = new_conf
            if hasattr(self.controller, 'settings_manager'):
                self.controller.settings_manager.save_confidence_settings(self.controller.confidence_settings)

    def _display_image_with_detection_boxes(self, original_image, detection_info, conf_threshold):
        """显示带检测框的图像"""
        if not original_image:
            self.species_image_label.setText("无图像数据")
            return

        try:
            # 复制原始图像用于绘制
            img_to_draw = original_image.copy()

            # 如果有检测信息，绘制检测框
            if detection_info and detection_info.get("检测框"):
                img_to_draw = self._draw_detection_boxes_on_image(
                    img_to_draw,
                    detection_info,
                    conf_threshold
                )

            # 显示图像
            self._display_pil_image(img_to_draw)

        except Exception as e:
            logger.error(f"显示带检测框的图像失败: {e}")
            # 回退到显示原始图像
            self._display_original_image(original_image)

    def _display_original_image(self, pil_image):
        """显示原始图像（不带检测框）"""
        if not pil_image:
            self.species_image_label.setText("无图像数据")
            return

        try:
            self._display_pil_image(pil_image)
        except Exception as e:
            logger.error(f"显示原始图像失败: {e}")
            self.species_image_label.setText("图像显示失败")

    def _display_pil_image(self, pil_image):
        """将PIL图像显示到QLabel中"""
        try:
            # 获取显示区域大小
            label_size = self.species_image_label.size()
            max_width = max(label_size.width(), 400)
            max_height = max(label_size.height(), 300)

            # 调整图像大小保持比例
            resized_img = self._resize_image_to_fit(pil_image, max_width, max_height)

            # 转换PIL图像为QPixmap
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')

            # 转换为numpy数组
            import numpy as np
            img_array = np.array(resized_img)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width

            from PySide6.QtGui import QImage, QPixmap
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 设置到标签
            self.species_image_label.setPixmap(pixmap)
            self.species_image_label.setScaledContents(False)
            self.species_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # 保持引用避免垃圾回收
            self.species_image_label.pixmap = pixmap

        except Exception as e:
            logger.error(f"显示PIL图像失败: {e}")
            self.species_image_label.setText("图像转换失败")

    def _resize_image_to_fit(self, img, max_width, max_height):
        """调整图像大小以适应显示区域"""
        if not all([max_width > 0, max_height > 0]):
            max_width, max_height = 400, 300

        w, h = img.size
        if w == 0 or h == 0:
            return img

        # 计算缩放比例
        scale = min(max_width / w, max_height / h)
        if scale >= 1:
            return img

        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _draw_detection_boxes_on_image(self, img, detection_info, conf_threshold):
        """在图像上绘制检测框"""
        try:
            draw = ImageDraw.Draw(img)
            boxes_info = detection_info.get("检测框", [])

            if not boxes_info:
                logger.info("没有检测框信息")
                return img

            # 加载字体
            font = self._load_font_for_drawing(img.size)

            species_colors = {}
            color_index = 0

            for box in boxes_info:
                try:
                    # 获取置信度
                    confidence = box.get("置信度", box.get("confidence", 0))
                    if confidence < conf_threshold:
                        continue

                    # 获取物种名称
                    species_name = box.get("物种", box.get("species", box.get("class_name", "未知")))

                    # 获取边界框坐标
                    bbox = None
                    if "边界框" in box:
                        bbox = box["边界框"]
                    elif all(key in box for key in ["x1", "y1", "x2", "y2"]):
                        bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
                    elif "bbox" in box:
                        bbox = box["bbox"]

                    if not bbox or len(bbox) < 4:
                        logger.warning(f"无效的边界框数据: {box}")
                        continue

                    x1, y1, x2, y2 = map(int, bbox[:4])

                    # 确保坐标在图像范围内
                    img_width, img_height = img.size
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    # 确保边界框有效
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # 为物种分配颜色
                    color = get_species_color(species_name, return_rgb=False)

                    # 绘制检测框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # 准备标签文本
                    label_text = f"{species_name} ({confidence:.2f})"

                    # 计算文本尺寸
                    try:
                        bbox_text = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                    except AttributeError:
                        # 兼容旧版PIL
                        text_width, text_height = draw.textsize(label_text, font=font)

                    # 确保标签在图像范围内
                    label_y = max(text_height + 5, y1)

                    # 绘制标签背景
                    draw.rectangle(
                        [x1, label_y - text_height - 5, x1 + text_width + 10, label_y],
                        fill=color
                    )

                    # 绘制标签文本
                    draw.text((x1 + 5, label_y - text_height - 2), label_text, fill='white', font=font)

                except Exception as e:
                    logger.error(f"绘制检测框时出错: {e}, 检测框数据: {box}")
                    continue

            logger.info(
                f"成功绘制了 {len([b for b in boxes_info if b.get('置信度', b.get('confidence', 0)) >= conf_threshold])} 个检测框")
            return img

        except Exception as e:
            logger.error(f"绘制检测框失败: {e}")
            return img

    def _load_font_for_drawing(self, image_size):
        """为绘制加载合适的字体"""
        try:
            from system.utils import resource_path
            font_path = resource_path("assets/simhei.ttf")
            font_size = max(16, int(0.02 * min(image_size)))
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            logger.warning("中文字体文件未找到，使用默认字体")
            try:
                font_size = max(16, int(0.02 * min(image_size)))
                return ImageFont.truetype("arial.ttf", font_size)
            except:
                return ImageFont.load_default()

    def select_species_and_image(self, species_name: str, image_filename: str):
        """以编程方式选中指定的物种和图像"""
        # 1. 选中物种
        for i in range(self.species_listbox.count()):
            item = self.species_listbox.item(i)
            # 检查物种名称是否匹配 (忽略后面的数量)
            if item and item.text().startswith(species_name + " ("):
                self.species_listbox.setCurrentItem(item)
                # 滚动以确保可见
                self.species_listbox.scrollToItem(item)

                # 2. 定义一个内部函数来选中照片
                def select_image_item():
                    for j in range(self.species_photo_listbox.count()):
                        photo_item = self.species_photo_listbox.item(j)
                        if photo_item and photo_item.text() == image_filename:
                            self.species_photo_listbox.setCurrentItem(photo_item)
                            self.species_photo_listbox.scrollToItem(photo_item)
                            break

                # 3. 使用QTimer延迟执行照片选择，以确保照片列表已更新
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, select_image_item)
                return

    def keyPressEvent(self, event):
        """重写键盘事件，以实现全局上下键选择照片"""
        current_row = self.species_photo_listbox.currentRow()

        if event.key() == Qt.Key.Key_Up:
            if current_row > 0:
                self.species_photo_listbox.setCurrentRow(current_row - 1)
            event.accept()
        elif event.key() == Qt.Key.Key_Down:
            if current_row < self.species_photo_listbox.count() - 1:
                self.species_photo_listbox.setCurrentRow(current_row + 1)
            event.accept()
        else:
            # 对于其他按键，调用父类的默认实现
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 视频模式下重绘最后一帧
        if self._current_video_frame_pixmap and self.video_thread:
             self._on_video_frame_ready(self._current_video_frame_pixmap)
        # 图片模式下
        elif hasattr(self, 'species_validation_original_image') and self.species_validation_original_image:
            self._resize_timer.start(50)

    def _redraw_image_on_resize(self):
        """根据新的窗口大小重绘当前显示的图片。"""
        # 再次确认原始图像存在
        if not hasattr(self, 'species_validation_original_image') or self.species_validation_original_image is None:
            return

        # 直接调用现有的绘图函数，它会使用更新后的标签尺寸重新绘制
        self._display_image_with_detection_boxes(
            self.species_validation_original_image,
            self.current_species_info,
            self.species_conf_var
        )

    def select_species_and_image(self, species_name: str, image_filename: str):
        """以编程方式选中指定的物种和图像"""
        # 1. 选中物种
        for i in range(self.species_listbox.count()):
            item = self.species_listbox.item(i)
            # 检查物种名称是否匹配 (忽略后面的数量)
            if item and item.text().startswith(species_name + " ("):
                self.species_listbox.setCurrentItem(item)
                # 滚动以确保可见
                self.species_listbox.scrollToItem(item)

                # 2. 定义一个内部函数来选中照片
                def select_image_item():
                    for j in range(self.species_photo_listbox.count()):
                        photo_item = self.species_photo_listbox.item(j)
                        if photo_item and photo_item.text() == image_filename:
                            self.species_photo_listbox.setCurrentItem(photo_item)
                            self.species_photo_listbox.scrollToItem(photo_item)
                            break

                # 3. 使用QTimer延迟执行照片选择，以确保照片列表已更新
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, select_image_item)
                return

    def _start_video_thread(self, video_path, json_path):
        """启动视频播放线程"""
        self._stop_video_thread()
        self._is_video_paused = False

        # 获取高级设置中的过滤比例 (如果可用)
        min_ratio = 0.0
        if hasattr(self.controller, 'advanced_page'):
            min_ratio = self.controller.advanced_page.min_frame_ratio_var

        self.video_thread = VideoPlayerThread(
            video_path, json_path,
            self.species_conf_var,
            draw_boxes=True,
            min_frame_ratio=min_ratio
        )
        self.video_thread.frame_ready.connect(self._on_video_frame_ready)
        self.video_thread.pause_state_changed.connect(self._on_video_pause_state_changed)
        self.video_thread.start()

    def _stop_video_thread(self):
        """停止视频线程"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.deleteLater()
            self.video_thread = None

    def _on_video_frame_ready(self, pixmap):
        """接收视频帧并显示"""
        if not self.isVisible(): return

        self._current_video_frame_pixmap = pixmap

        # 如果暂停中，忽略新帧以保持暂停图标不被覆盖
        if self._is_video_paused:
            return

        if self.species_image_label.size().isValid():
            scaled_pixmap = pixmap.scaled(
                self.species_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.species_image_label.setPixmap(scaled_pixmap)

    def eventFilter(self, source, event):
        """处理点击事件以暂停/播放视频"""
        if source == self.species_image_label and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.video_thread and self.video_thread.isRunning():
                    self.video_thread.toggle_pause()
                    return True
        return super().eventFilter(source, event)

    def _generate_white_icon_pixmap(self, icon_path, size):
        """生成白色图标 (复用 PreviewPage 的逻辑)"""
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QPainter, QPixmap
        from system.utils import resource_path

        if not os.path.exists(icon_path): return None
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer = QSvgRenderer(icon_path)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), Qt.GlobalColor.white)
        painter.end()
        return pixmap

    def _on_video_pause_state_changed(self, is_paused):
        """处理暂停状态改变，绘制图标"""
        self._is_video_paused = is_paused
        if not is_paused or not self._current_video_frame_pixmap:
            return

        from system.utils import resource_path
        try:
            paused_pixmap = self._current_video_frame_pixmap.copy()
            w, h = paused_pixmap.width(), paused_pixmap.height()
            icon_size = max(64, int(min(w, h) * 0.2))
            x, y = (w - icon_size) // 2, (h - icon_size) // 2

            painter = QPainter(paused_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 背景
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(x + icon_size // 2, y + icon_size // 2), icon_size // 2 + 10,
                                icon_size // 2 + 10)

            # 图标
            icon_path = resource_path(os.path.join("res", "icon", "play.svg"))
            white_icon = self._generate_white_icon_pixmap(icon_path, icon_size)
            if white_icon:
                painter.drawPixmap(x, y, white_icon)

            painter.end()

            if self.species_image_label.size().isValid():
                self.species_image_label.setPixmap(paused_pixmap.scaled(
                    self.species_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
        except Exception as e:
            logger.error(f"绘制暂停图标失败: {e}")