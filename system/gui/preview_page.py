from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QLabel, QTextEdit, QPushButton,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QMessageBox, QFileDialog, QInputDialog, QScrollArea,
    QSizePolicy, QApplication, QStackedLayout, QComboBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QUrl, QSize, QEvent, QRectF, QPoint
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QFont, QAction, QKeySequence,
    QIcon, QDesktopServices, QShortcut, QPalette, QPainterPath
)
from PySide6.QtSvg import QSvgRenderer
import sys
import os
import json
import logging
import cv2
import threading
import time
import re
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

# 原有的导入保持不变
from system.data_processor import DataProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.config import NORMAL_FONT, SUPPORTED_IMAGE_EXTENSIONS, get_species_color
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
    image_loaded = Signal(object, str, dict)  # (q_image, file_path, image_info)
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
            q_image_copy = q_image.copy()

            # --- 线程取消点 4 ---
            if self._is_cancelled: return

            # 4. 发送完成信号
            self.image_loaded.emit(q_image_copy, self.file_path, image_info)

        except Exception as e:
            if not self._is_cancelled:
                self.loading_failed.emit(self.file_path, str(e))


class VideoPlayerThread(QThread):
    """
    Video player thread that reads video via OpenCV, converts to PIL to draw
    detection boxes (for TTF font support), and emits QImage for display.
    """
    frame_ready = Signal(QPixmap)
    playback_finished = Signal()
    pause_state_changed = Signal(bool)

    def __init__(self, video_path, json_path, conf_map, draw_boxes=True, min_frame_ratio=0.0, start_frame=0,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.json_path = json_path

        # 确保 conf_map 是字典，如果为空则给默认值
        self.conf_map = conf_map if conf_map else {"global": 0.25}

        self.draw_boxes = draw_boxes
        self.min_frame_ratio = min_frame_ratio
        self.start_frame = start_frame
        self.running = False
        self.paused = False
        self.current_frame_index = 0

        # === 初始化字体 ===
        try:
            self.font_path = resource_path(os.path.join("res", "AlibabaPuHuiTi-3-65-Medium.ttf"))
            self.font = ImageFont.truetype(self.font_path, 20)
            self.font_loaded = True
        except Exception as e:
            logger.warning(f"VideoThread: 字体加载失败 {e}")
            self.font = ImageFont.load_default()
            self.font_loaded = False

    def toggle_pause(self):
        """切换暂停/播放状态"""
        self.paused = not self.paused
        self.pause_state_changed.emit(self.paused)

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps

        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # 根据视频尺寸调整字体大小
        v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self.font_loaded and v_h > 0:
            target_size = max(16, int(0.02 * min(v_w, v_h)))
            try:
                self.font = ImageFont.truetype(self.font_path, target_size)
            except:
                pass

        # Parse JSON
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

            # 1. OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. 转换为 PIL Image
            pil_img = Image.fromarray(rgb_frame)

            # 3. 绘制检测框 (仅当 draw_boxes 为 True 时执行)
            if self.draw_boxes:
                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                lookup_idx = current_frame_idx - (current_frame_idx % stride)

                if lookup_idx in detections:
                    self._draw_boxes_pil(pil_img, detections[lookup_idx])

            # 4. PIL -> QImage
            # 这里的 pil_img 已经是绘制好的了
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
        """Converts Track-ID based JSON to Frame-Index based dictionary with Filtering and Species Unification"""
        parsed_frames = {'frames': {}, 'stride': 1}

        if not self.json_path or not os.path.exists(self.json_path):
            return parsed_frames

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            parsed_frames['stride'] = data.get('vid_stride', 1)
            total_frames = data.get('total_frames_processed', 0)

            # === 过滤逻辑 ===
            tracks = data.get('tracks', {})
            min_frames_threshold = total_frames * self.min_frame_ratio

            # Pivot data: Track -> Frame
            for track_id, track_list in tracks.items():
                # 1. 过滤掉帧数不足的目标
                if len(track_list) < min_frames_threshold:
                    continue

                # === 2. 新增：计算该 Track 的最终物种（投票法） ===
                # 统计该轨迹中出现次数最多的物种，消除单帧识别跳变
                species_list = [p.get('species') for p in track_list if p.get('species')]

                final_species = "Unknown"
                if species_list:
                    # Counter.most_common(1) 返回 [('物种名', 次数)]
                    final_species = Counter(species_list).most_common(1)[0][0]
                # ==========================================

                for point in track_list:
                    f_idx = point.get('frame_index')
                    if f_idx is not None:
                        if f_idx not in parsed_frames['frames']:
                            parsed_frames['frames'][f_idx] = []

                        point['track_id'] = track_id

                        # === 3. 覆盖每一帧的物种为最终投票结果 ===
                        point['species'] = final_species

                        parsed_frames['frames'][f_idx].append(point)

        except Exception as e:
            logger.error(f"JSON Parse Error: {e}")

        return parsed_frames

    def _draw_boxes_pil(self, pil_img, boxes):
        draw = ImageDraw.Draw(pil_img)
        img_w, img_h = pil_img.size

        for box in boxes:
            species = box.get('species', 'Unknown')
            track_id = box.get('track_id', '?')
            conf = box.get('confidence', 0)

            # 如果 conf_map 中有该物种，使用该物种的阈值；否则使用 global；如果没有 global，默认 0.25
            threshold = self.conf_map.get(species, self.conf_map.get("global", 0.25))

            if conf < threshold:
                continue

            bbox = box.get('bbox')
            if not bbox: continue

            try:
                x1_f, y1_f, x2_f, y2_f = map(float, bbox[:4])
                is_normalized = all(0.0 <= c <= 1.0 for c in [x1_f, y1_f, x2_f, y2_f]) and (x2_f > 0 or y2_f > 0)
                if is_normalized:
                    x1, y1, x2, y2 = int(x1_f * img_w), int(y1_f * img_h), int(x2_f * img_w), int(y2_f * img_h)
                else:
                    x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)
            except Exception:
                continue

            rgb_color = get_species_color(species, return_rgb=True)
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)
            label = f"{species} #{track_id} ({conf:.2f})"

            try:
                if hasattr(draw, 'textbbox'):
                    text_bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                else:
                    text_w, text_h = draw.textsize(label, font=self.font)
            except:
                text_w, text_h = 100, 20

            label_y = max(text_h + 5, y1)
            if label_y > img_h: label_y = y1
            draw.rectangle([x1, label_y - text_h - 5, x1 + text_w + 10, label_y], fill=rgb_color)
            draw.text((x1 + 5, label_y - text_h - 5), label, fill='white', font=self.font)

    def stop(self):
        self.running = False
        self.wait()


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

        self.current_image_path = None

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

        # 1. 初始化置信度字典
        self.species_conf_map = {"global": 0.25}
        self._load_species_conf()  # 加载 conf.json

        # 2. 设置默认的全局置信度变量 (用于初始化滑块)
        self.preview_conf_var = self.species_conf_map.get("global", 0.25)

        # === 新增：防抖动定时器 ===
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(200)  # 200毫秒延迟
        self.selection_timer.timeout.connect(self._load_image_deferred)

        # === 新增：防止线程被垃圾回收的列表 ===
        self._stopping_threads = []

        # === 新增：定义支持的视频格式 ===
        self.SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

        # === 新增：视频播放器初始化 ===
        self.video_thread = None
        self._current_video_frame_pixmap = None
        self._is_video_paused = False

        self.settings_connected = False  # 新增标记
        self._try_connect_settings_signal()  # 尝试连接

        # 默认禁用，只有播放视频时才启用，防止干扰列表选择
        self.play_pause_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.play_pause_shortcut.activated.connect(self.toggle_video_playback)
        self.play_pause_shortcut.setEnabled(False)

        self._create_widgets()
        self._apply_theme()

        # 用于处理窗口大小调整的计时器
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._redraw_image_on_resize)

        self.show_detection_checkbox.toggled.connect(lambda: self.settings_changed.emit())


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

    def _get_conf_path(self):
        """获取 conf.json 的路径"""
        # 1. 强制指定目标目录为 'temp'
        target_dir = "temp"

        # 2. 确保目录存在
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"创建 temp 目录失败: {e}")
                # 如果创建失败（例如权限问题），回退使用 controller 提供的目录
                fallback = self.controller.get_temp_photo_dir()
                if fallback:
                    return os.path.join(fallback, "conf.json")
                return "conf.json"

        # 3. 返回 temp/conf.json
        return os.path.join(target_dir, "conf.json")

    def _load_species_conf(self):
        """从 conf.json 加载置信度设置，如果不存在则自动创建"""
        try:
            json_path = self._get_conf_path()

            # 1. 尝试加载现有文件
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.species_conf_map.update(data)

            # 2. 确保内存中始终包含 'global' 键
            # (self.species_conf_map 在 __init__ 中已初始化为 {"global": 0.25}，
            # 但为了防止读取的 json 是空的或者被意外修改，这里做双重保险)
            if "global" not in self.species_conf_map:
                self.species_conf_map["global"] = 0.25

            # 3. === 新增：如果文件不存在，则保存当前默认配置到文件 ===
            if not os.path.exists(json_path):
                self._save_species_conf()
                logger.info(f"conf.json 不存在，已自动创建并初始化: {json_path}")

        except Exception as e:
            logger.error(f"加载 conf.json 失败: {e}")

    def _save_species_conf(self):
        """保存置信度设置到 conf.json"""
        try:
            json_path = self._get_conf_path()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.species_conf_map, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 conf.json 失败: {e}")

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
        self.image_label.installEventFilter(self)
        self.image_label.pixmap = None
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(self._get_placeholder_style())
        image_layout.addWidget(self.image_label)

        right_layout.addWidget(image_group, 3)

        # 图像信息区域
        info_group = ModernGroupBox("图像信息")
        info_group.setFixedHeight(170)
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setFixedHeight(110)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)

        right_layout.addWidget(info_group)

        # 控制面板
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.show_detection_checkbox = SwitchRow("显示检测结果")
        self.show_detection_checkbox.toggled.connect(self.toggle_detection_preview)
        control_layout.addWidget(self.show_detection_checkbox)

        control_layout.addWidget(QLabel("选择物种:"))
        self.species_selector = QComboBox()
        self.species_selector.addItem("全局设置 (Global)", "global")
        control_layout.addWidget(self.species_selector)

        # 置信度滑块
        control_layout.addWidget(QLabel("置信度:"))
        self.preview_conf_slider = ModernSlider(Qt.Horizontal)
        self.preview_conf_slider.setMinimumWidth(200)  # 之前增加宽度的修改
        self.preview_conf_slider.setRange(5, 95)
        self.preview_conf_slider.setValue(int(self.preview_conf_var * 100))
        self.preview_conf_slider.valueChanged.connect(self._on_preview_confidence_slider_changed)
        control_layout.addWidget(self.preview_conf_slider)

        self.preview_conf_label = QLabel(f"{self.preview_conf_var:.2f}")
        control_layout.addWidget(self.preview_conf_label)

        self.species_selector.currentIndexChanged.connect(self._on_species_selector_changed)

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
            self._stop_video_detection_thread()
            self.play_pause_shortcut.setEnabled(False)
            self.image_label.setVisible(True)

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
            all_files = os.listdir(directory)
            all_files.sort()

            for file in all_files:
                lower_file = file.lower()
                # 检查是否为支持的图片或视频
                is_image = lower_file.endswith(SUPPORTED_IMAGE_EXTENSIONS)
                is_video = lower_file.endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

                if is_image or is_video:
                    from PySide6.QtWidgets import QListWidgetItem
                    # 可以选择给视频文件加个不同的图标或标记
                    display_text = f"📹 {file}" if is_video else file
                    item = QListWidgetItem(display_text)

                    full_path = os.path.join(directory, file)
                    item.setData(Qt.ItemDataRole.UserRole, full_path)
                    # 存储类型标记，方便后续判断
                    item.setData(Qt.ItemDataRole.UserRole + 1, "video" if is_video else "image")

                    self.file_listbox.addItem(item)
        except Exception as e:
            logger.error(f"更新文件列表失败: {e}")

    def on_file_selected(self):
        """文件选择事件处理（防抖动版）"""
        # 停止之前的计时，重新开始
        self.selection_timer.stop()
        self.selection_timer.start()

    def _on_image_loaded_safe(self, q_image, file_path, image_info):
        """安全的图像加载完成回调"""
        try:
            if not self or not hasattr(self, 'image_label'):
                return
            if self.image_label is None or not self.image_label.isVisible():
                return

            self._on_image_loaded(q_image, file_path, image_info)
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
        """Toggle detection preview."""

        # 1. Determine if we are handling a video
        current_file = self.current_image_path
        is_video = current_file and current_file.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

        # 2. Setup JSON path logic
        json_path = None
        if current_file:
            temp_dir = self.controller.get_temp_photo_dir()
            base_name = os.path.splitext(os.path.basename(current_file))[0]
            json_path = os.path.join(temp_dir, f"{base_name}.json")

        if is_video:
            # === Video Mode: 统一使用 OpenCV 线程 ===
            # 无论 Checked 是 True 还是 False，都进入 OpenCV 模式
            # 获取设置
            min_ratio = 0.0
            if hasattr(self.controller, 'advanced_page'):
                min_ratio = self.controller.advanced_page.min_frame_ratio_var

            # 获取当前播放进度，以便无缝切换
            start_frame = 0
            if self.video_thread:
                # 获取当前播放到的帧索引
                start_frame = self.video_thread.current_frame_index

            # 刷新文本 (确保检测结果统计与当前过滤比例一致)
            self._update_video_info_text(current_file, json_path, min_ratio)

            # 更新下拉框
            self.current_preview_info = {}
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_preview_info = json.load(f)
                except:
                    pass
            self._update_species_selector_items()
            # 即使 JSON 不存在，VideoPlayerThread 内部也会安全处理（读取不到数据则不画框），
            self._start_video_detection_thread(current_file, json_path, draw_boxes=checked, start_frame=start_frame)

            return  # 视频逻辑处理完毕，直接返回

        # === Image Mode: Existing Logic (图片逻辑保持不变) ===
        if checked:
            if not current_file:
                self.show_detection_checkbox.setChecked(False)
                return

            if not self.original_image:
                QMessageBox.warning(self, "提示", "图像尚未加载完成。")
                self.show_detection_checkbox.setChecked(False)
                return

            # Load JSON if not already loaded
            if not self.current_preview_info and os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.current_preview_info = json.load(f)

            if self.current_preview_info:
                # 更新下拉框内容
                self._update_species_selector_items()
                # 绘制 (传递 None，内部使用 self.species_conf_map)
                self._redraw_preview_boxes_with_new_confidence(None)
            else:
                QMessageBox.information(self, "提示", "当前图像还没有检测结果。")
                self.show_detection_checkbox.setChecked(False)

        else:
            # Unchecked (Image)
            if self.original_image:
                self._update_pixmap_for_label(self.original_image)

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

    def _update_detection_info(self, species_info=None):
        """
        更新检测信息显示 (支持动态置信度过滤)
        :param species_info: 可选，如果传入则更新当前的 current_preview_info，否则使用现有的
        """
        try:
            # 1. 只有传入新数据时才更新缓存，否则使用已有的缓存进行重算
            if species_info:
                self.current_preview_info = species_info

            if not self.current_preview_info:
                return

            # 获取基本信息（文件名、尺寸等，保留原有的前两行）
            current_text = self.info_text.toPlainText().strip()
            current_lines = current_text.split('\n') if current_text else []
            # 尝试保留前两行基本信息，如果当前已经是检测结果文本，则重新生成可能比较麻烦，
            # 建议依靠 update_image_info 生成的基础信息，或者在这里简单保留 header
            # 这里为了稳健，如果行数不够，就不保留了，实际逻辑中通常 update_image_info 会先被调用
            basic_info = "\n".join(current_lines[:2]) if len(current_lines) >= 2 else ""
            if "检测结果" in basic_info: basic_info = ""  # 防止重复叠加

            # === 动态统计逻辑 ===
            counts = Counter()
            valid_confidences = []

            # 获取检测框列表
            boxes = self.current_preview_info.get("检测框",
                                                  self.current_preview_info.get("detect_results",
                                                                                self.current_preview_info.get("objects",
                                                                                                              [])))

            if boxes:
                for box in boxes:
                    # 1. 获取基础信息
                    species_name = box.get("物种", box.get("species", "未知"))
                    confidence = float(box.get("置信度", box.get("confidence", 0.0)))
                    final_name = species_name  # 默认使用原始识别结果
                    is_valid = False

                    # 2. 候选项逻辑 (与画框逻辑保持一致)
                    if "候选项" in box and box["候选项"]:
                        # 检查候选项是否满足其特定阈值
                        candidate_matched = False
                        for cand in box["候选项"]:
                            c_name = cand.get('name')
                            c_conf = float(cand.get('conf', 0))
                            # 获取该候选项的阈值
                            c_thresh = self.species_conf_map.get(c_name, self.species_conf_map.get("global", 0.25))

                            if c_conf >= c_thresh:
                                final_name = c_name
                                confidence = c_conf
                                candidate_matched = True
                                break  # 找到第一个满足的候选项即可

                        if candidate_matched:
                            is_valid = True

                    # 3. 如果没有匹配的候选项，检查主物种是否满足阈值
                    if not is_valid:
                        thresh = self.species_conf_map.get(species_name, self.species_conf_map.get("global", 0.25))
                        if confidence >= thresh:
                            is_valid = True
                            final_name = species_name

                    # 4. 统计
                    if is_valid:
                        counts[final_name] += 1
                        valid_confidences.append(confidence)

            # === 构建显示文本 ===
            detection_parts = ["检测结果:"]

            if counts:
                # 按数量降序排列
                info_parts = [f"{n}: {c}只" for n, c in counts.most_common()]
                detection_parts.append(", ".join(info_parts))

                if valid_confidences:
                    min_conf = min(valid_confidences)
                    detection_parts.append(f"最低置信度: {min_conf:.3f}")

                # 尝试获取原始的检测时间
                detect_time = self.current_preview_info.get('检测时间', '')
                if detect_time:
                    detection_parts.append(f"检测于: {detect_time}")
            else:
                detection_parts.append("当前置信度下未检测到目标")

            # 合并信息
            full_info = basic_info + "\n" + " | ".join(detection_parts)
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

    def _draw_detection_boxes(self, image_label, original_image, detection_info, conf_map):
        """
        根据给定的置信度阈值，在指定的原始图像上绘制检测框，并更新对应的UI标签。
        (修复了归一化坐标导致的无法绘制问题，并增加了候选项筛选逻辑)
        """
        # 1. 检查是否有原始图像
        if not original_image:
            placeholder_text = "请从左侧列表选择图像"
            image_label.clear()
            image_label.setText(placeholder_text)
            if hasattr(image_label, 'pixmap'):
                image_label.pixmap = None
            return

        # 2. 获取检测框列表（兼容多种 JSON 键名）
        boxes_info = []
        if detection_info:
            # 按优先级尝试获取列表
            for key in ["检测框", "detect_results", "objects", "frames"]:
                if key in detection_info and detection_info[key]:
                    boxes_info = detection_info[key]
                    break

        # 如果没有检测信息，显示原图
        if not detection_info or not boxes_info:
            self._update_pixmap_for_label(original_image)
            return

        # 4. 字体加载 (保持原逻辑)
        try:
            font_path = resource_path(os.path.join("res", "AlibabaPuHuiTi-3-65-Medium.ttf"))
            # 动态字体大小：图像短边的 2%
            font_size = max(12, int(0.02 * min(original_image.width, original_image.height)))
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            try:
                font = ImageFont.load_default()
            except:
                font = None  # 极端情况

        # 5. 绘制逻辑
        try:
            # 确保在副本上绘制，且为 RGB 模式
            if original_image.mode != 'RGB':
                img_to_draw = original_image.convert('RGB')
            else:
                img_to_draw = original_image.copy()

            draw = ImageDraw.Draw(img_to_draw)
            img_width, img_height = original_image.size

            for box in boxes_info:
                try:
                    # ==================== [修改开始] 多物种标签逻辑 ====================
                    valid_display_texts = []  # 存储最终显示的文本片段
                    seen_names = set()  # 用于去重
                    primary_species = None  # 用于决定框颜色的主要物种（最高置信度者）

                    # 1. 收集所有可能的物种来源 (候选项 + 主结果)
                    candidates = []

                    # A. 收集候选项
                    if "候选项" in box and box["候选项"]:
                        for cand in box["候选项"]:
                            candidates.append((cand.get('name'), float(cand.get('conf', 0))))

                    # B. 收集主结果 (作为补充，防止候选项缺失)
                    raw_name = box.get("物种", box.get("species", box.get("class_name", "未知")))
                    raw_conf = float(box.get("置信度", box.get("confidence", 0)))
                    candidates.append((raw_name, raw_conf))

                    # 2. 排序：按置信度降序，确保高置信度的排在前面
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    # 3. 遍历检查阈值并构建标签
                    for name, conf in candidates:
                        if name in seen_names:
                            continue  # 去重

                        # 获取该物种的特定阈值 (优先取特定设置，否则取全局)
                        thresh = conf_map.get(name, conf_map.get("global", 0.25))

                        if conf >= thresh:
                            valid_display_texts.append(f"{name} {conf:.2f}")
                            seen_names.add(name)

                            # 记录第一个通过阈值的物种（即置信度最高的），用于确定框的颜色
                            if primary_species is None:
                                primary_species = name

                    # 如果没有任何物种通过阈值，则不绘制此框
                    if not valid_display_texts:
                        continue

                    # 4. 组合最终显示的标签文本 (例如: "赤狐 0.95 | 狗 0.88")
                    label_text = " | ".join(valid_display_texts)
                    # ==================== [修改结束] ====================

                    # --- B. 获取并转换坐标 (归一化处理) ---
                    bbox = None
                    if "边界框" in box:
                        bbox = box["边界框"]
                    elif "bbox" in box:
                        bbox = box["bbox"]
                    elif all(k in box for k in ["x1", "y1", "x2", "y2"]):
                        bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]

                    if not bbox or len(bbox) < 4: continue

                    x1_f, y1_f, x2_f, y2_f = map(float, bbox[:4])

                    # 判断是否为归一化坐标 (0.0-1.0)
                    is_normalized = all(0.0 <= c <= 1.0 for c in [x1_f, y1_f, x2_f, y2_f]) and (x2_f > 0 or y2_f > 0)

                    if is_normalized:
                        x1 = int(x1_f * img_width)
                        y1 = int(y1_f * img_height)
                        x2 = int(x2_f * img_width)
                        y2 = int(y2_f * img_height)
                    else:
                        x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)

                    # 边界限制
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y2 = max(0, min(y2, img_height - 1))

                    if x2 <= x1 or y2 <= y1: continue

                    # --- C. 绘制样式 ---
                    # 使用置信度最高的有效物种来决定颜色
                    color = get_species_color(primary_species, return_rgb=True)

                    # 动态线宽
                    line_width = max(2, int(min(img_width, img_height) * 0.005))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                    # 绘制标签背景和文字 (使用上面生成的 label_text)
                    if font:
                        if hasattr(draw, 'textbbox'):  # PIL >= 9.2.0
                            left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
                            text_w, text_h = right - left, bottom - top
                        else:  # 旧版 PIL
                            text_w, text_h = draw.textsize(label_text, font=font)
                    else:
                        text_w, text_h = 100, 20

                    # 优先在框上方显示标签
                    label_y = y1 - text_h - 4
                    if label_y < 0: label_y = y1

                    draw.rectangle(
                        [x1, label_y, x1 + text_w + 8, label_y + text_h + 4],
                        fill=color
                    )

                    if font:
                        draw.text((x1 + 4, label_y), label_text, fill='white', font=font)

                except Exception as e:
                    logger.debug(f"绘制单框失败: {e}")
                    continue

            # 更新UI
            self._update_pixmap_for_label(img_to_draw)

        except Exception as e:
            logger.error(f"更新图像显示时出错: {e}")
            image_label.setText("图像显示出错")

    def _redraw_preview_boxes_with_new_confidence(self, unused_conf_str):
        """根据当前的置信度配置 Map，在预览图像上重新绘制检测框"""
        try:
            if not self.original_image:
                if not (self.current_image_path and self.current_image_path.lower().endswith(
                        self.SUPPORTED_VIDEO_EXTENSIONS)):
                    logger.warning("原始图像未加载，无法绘制检测框")
                    pass
                return

            self._draw_detection_boxes(
                image_label=self.image_label,
                original_image=self.original_image,
                detection_info=self.current_preview_info,
                conf_map=self.species_conf_map # 传递 map
            )
        except Exception as e:
            logger.error(f"重绘预览检测框失败: {e}")

    def _update_species_selector_items(self):
        """
        根据当前的检测结果更新下拉框内容。
        自动选择逻辑：
        1. 优先选择：当前置信度阈值下可见的、置信度最高的物种。
        2. 回退策略：如果所有物种都被当前阈值过滤掉（不可见），则选择绝对置信度最高的物种。
        """
        # 暂时阻断信号，防止清空时触发 change 事件
        self.species_selector.blockSignals(True)
        self.species_selector.clear()

        # 1. 恢复全局设置选项
        self.species_selector.addItem("全局设置 (Global)", "global")

        found_species = set()

        # === 变量定义 ===
        # A. 有效最高置信度 (满足阈值)
        best_valid_species_name = None
        max_valid_confidence = -1.0

        # B. 绝对最高置信度 (无视阈值，作为兜底)
        best_absolute_species_name = None
        max_absolute_confidence = -1.0

        # 获取全局默认阈值
        global_thresh = self.species_conf_map.get("global", 0.25)

        # 从当前 JSON 数据中提取所有物种
        if self.current_preview_info:
            # --- 情况 A: 处理图片 JSON 结构 ---
            boxes = self.current_preview_info.get("检测框",
                                                  self.current_preview_info.get("detect_results",
                                                                                self.current_preview_info.get("objects",
                                                                                                              [])))

            for box in boxes:
                # 1. 获取原始信息
                raw_name = box.get("物种", box.get("species", box.get("class_name")))
                raw_conf = float(box.get("置信度", box.get("confidence", 0.0)))

                if not raw_name: continue

                # 默认情况下，最终显示的物种和置信度就是原始的
                final_name = raw_name
                final_conf = raw_conf

                # 2. 处理候选项逻辑 (确定该框最终判定为什么物种)
                is_candidate_match = False
                if "候选项" in box and box["候选项"]:
                    candidates = box["候选项"]
                    for cand in candidates:
                        c_name = cand.get('name')
                        c_conf = float(cand.get('conf', 0.0))
                        c_thresh = self.species_conf_map.get(c_name, global_thresh)

                        if c_conf >= c_thresh:
                            final_name = c_name
                            final_conf = c_conf
                            is_candidate_match = True
                            break

                            # 3. 将所有出现过的名字加入下拉列表
                found_species.add(final_name)
                if "候选项" in box:
                    for c in box["候选项"]:
                        if c.get('name'): found_species.add(c['name'])

                # === 4. 更新绝对最大值 (兜底用) ===
                if final_conf > max_absolute_confidence:
                    max_absolute_confidence = final_conf
                    best_absolute_species_name = final_name

                # === 5. 更新有效最大值 (优先用) ===
                is_valid = False
                if is_candidate_match:
                    is_valid = True
                else:
                    thresh = self.species_conf_map.get(final_name, global_thresh)
                    if final_conf >= thresh:
                        is_valid = True

                if is_valid:
                    if final_conf > max_valid_confidence:
                        max_valid_confidence = final_conf
                        best_valid_species_name = final_name

            # --- 情况 B: 处理视频 JSON 结构 (tracks) ---
            tracks = self.current_preview_info.get("tracks", {})
            if tracks:
                min_ratio = 0.0
                if hasattr(self.controller, 'advanced_page'):
                    min_ratio = self.controller.advanced_page.min_frame_ratio_var

                total_frames = self.current_preview_info.get('total_frames_processed', 1)
                threshold_frames = total_frames * min_ratio

                for t_list in tracks.values():
                    if len(t_list) < threshold_frames:
                        continue

                    s_list = [p.get('species') for p in t_list if p.get('species')]
                    if not s_list: continue
                    dominant_species = Counter(s_list).most_common(1)[0][0]
                    found_species.add(dominant_species)

                    track_max_conf = max([float(p.get('confidence', 0.0)) for p in t_list])

                    # === 更新绝对最大值 ===
                    if track_max_conf > max_absolute_confidence:
                        max_absolute_confidence = track_max_conf
                        best_absolute_species_name = dominant_species

                    # === 更新有效最大值 ===
                    sp_thresh = self.species_conf_map.get(dominant_species, global_thresh)
                    if track_max_conf >= sp_thresh:
                        if track_max_conf > max_valid_confidence:
                            max_valid_confidence = track_max_conf
                            best_valid_species_name = dominant_species

        # 将发现的物种添加到下拉框
        for sp in sorted(list(found_species)):
            self.species_selector.addItem(sp, sp)

        self.species_selector.blockSignals(False)

        # === 核心逻辑：确定最终选中的目标 ===
        target_species_name = None

        # 策略1：优先选择“有效且置信度最高”的物种
        if best_valid_species_name:
            target_species_name = best_valid_species_name
        # 策略2：如果所有物种都被过滤了（没有有效的），则选择“绝对置信度最高”的物种
        elif best_absolute_species_name:
            target_species_name = best_absolute_species_name

        # 执行选中
        target_index = -1
        if target_species_name:
            target_index = self.species_selector.findData(target_species_name)

        if target_index != -1:
            self.species_selector.setCurrentIndex(target_index)
        else:
            # 降级策略：如果没有结果，默认保持 Global
            if self.species_selector.count() > 0:
                self.species_selector.setCurrentIndex(0)

                # 触发一次改变以更新滑块状态
        self._on_species_selector_changed()

    def _on_species_selector_changed(self):
        """当下拉框选择改变时，更新滑块到对应物种的保存值"""
        current_species = self.species_selector.currentData()  # 获取 user data

        # 如果没有选中任何物种（例如列表为空），默认读取 global 配置
        if not current_species:
            current_species = "global"

        # 从字典中获取该物种的保存值，如果没有，获取 global，如果还没有，默认 0.25
        saved_val = self.species_conf_map.get(current_species, self.species_conf_map.get("global", 0.25))

        # 阻断滑块信号，防止滑块移动反过来触发 _on_preview_confidence_slider_changed 重复保存
        self.preview_conf_slider.blockSignals(True)
        self.preview_conf_slider.setValue(int(saved_val * 100))
        self.preview_conf_slider.blockSignals(False)

        self.preview_conf_label.setText(f"{saved_val:.2f}")

    def _on_preview_confidence_slider_changed(self, value):
        """处理预览页置信度滑块值的变化"""
        new_conf = value / 100.0
        if self.preview_conf_label:
            self.preview_conf_label.setText(f"{new_conf:.2f}")

        # 1. 更新当前选中物种的阈值配置
        current_species = self.species_selector.currentData()
        if current_species:
            self.species_conf_map[current_species] = new_conf
            self._save_species_conf()

        is_video = self.current_image_path and self.current_image_path.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

        # 2. 视频模式处理
        if self.video_thread and self.video_thread.isRunning():
            # 更新线程内的配置
            self.video_thread.conf_map = self.species_conf_map

            # === 新增：实时更新视频信息面板 ===
            temp_dir = self.controller.get_temp_photo_dir()
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            json_path = os.path.join(temp_dir, f"{base_name}.json")

            min_ratio = 0.0
            if hasattr(self.controller, 'advanced_page'):
                min_ratio = self.controller.advanced_page.min_frame_ratio_var

            self._update_video_info_text(self.current_image_path, json_path, min_ratio)

        # 3. 图片模式处理
        elif self.show_detection_checkbox.isChecked() and self.current_preview_info:
            if not is_video:
                # 重绘框
                self._redraw_preview_boxes_with_new_confidence(None)
                # === 新增：实时更新图片信息面板 ===
                self._update_detection_info()  # 不传参，使用 self.current_preview_info

    def _on_detection_completed(self, loaded_detection_info, filename):
        """检测完成处理"""
        try:
            if loaded_detection_info:  # 有检测结果
                self.current_preview_info = loaded_detection_info
                # 更新检测信息显示
                self._update_detection_info(loaded_detection_info)

                self._update_species_selector_items()

                # 自动显示检测框
                if not self.show_detection_checkbox.isChecked():
                    self.show_detection_checkbox.setChecked(True)
                else:
                    # 如果已经选中，手动触发重绘
                    self._redraw_preview_boxes_with_new_confidence(self.preview_conf_slider.value())
            else:
                QMessageBox.information(self, "提示", "未检测到任何对象。")

                # 可选：如果未检测到对象，也可以刷新一下以重置为 Global
                self.current_preview_info = {}
                self._update_species_selector_items()

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
        """获取当前页面设置"""
        # 保留原有逻辑，添加 show_detection 字段
        settings = {
            "preview_conf": self.preview_conf_slider.value(),
            # 保存“显示检测结果”按钮的状态
            "show_detection": self.show_detection_checkbox.isChecked()
        }
        return settings

    def load_settings(self, settings):
        """加载设置到UI (修正版)"""
        if not settings:
            return

        # 加载“显示检测结果”按钮的状态
        if "show_detection" in settings:
            should_show = settings["show_detection"]

            # 关键：使用 blockSignals(True) 防止在软件刚启动且无图片时，
            # setChecked 触发 toggle_detection_preview 逻辑，导致按钮因无图片而被强制重置为 False。
            self.show_detection_checkbox.blockSignals(True)
            self.show_detection_checkbox.setChecked(should_show)
            self.show_detection_checkbox.blockSignals(False)

            # 如果当前恰好已有图片（例如热加载设置），则手动刷新一次显示状态
            if getattr(self, 'current_image_path', None):
                self.toggle_detection_preview(should_show)

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

    def _on_image_loaded(self, q_image, file_path, image_info):
        """当图片成功加载后，在主线程中更新UI"""
        if file_path != self.requested_image_path:
            return

        self.loaded_image_path = file_path

        # 加载并保存原始图像
        try:
            self.original_image = Image.open(file_path)
            self.current_image_path = file_path
        except Exception as e:
            logger.error(f"加载原始图像失败: {e}")
            self.original_image = None
            self.current_image_path = None

        pixmap = QPixmap.fromImage(q_image)
        # 更新图像
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.pixmap = pixmap

        # 更新信息显示
        self.info_text.clear()
        info1 = f"文件名: {image_info.get('文件名', '')}    格式: {image_info.get('格式', '')}"
        info2 = f"拍摄日期: {image_info.get('拍摄日期', '未知')} {image_info.get('拍摄时间', '')}    "

        try:
            if self.original_image:
                img_width, img_height = self.original_image.size
                file_size_kb = os.path.getsize(file_path) / 1024
                info2 += f"尺寸: {img_width}x{img_height}px    文件大小: {file_size_kb:.1f} KB"
            else:
                width = image_info.get('宽度', image_info.get('width', '未知'))
                height = image_info.get('高度', image_info.get('height', '未知'))
                file_size_kb = os.path.getsize(file_path) / 1024
                info2 += f"尺寸: {width}x{height}px    文件大小: {file_size_kb:.1f} KB"
        except Exception as e:
            logger.error(f"获取图像信息失败: {e}")
            info2 += "尺寸: 未知"

        self.info_text.setPlainText(info1 + "\n" + info2)

        # 检查并加载已有的检测结果
        self.current_preview_info = {}  # 重置检测信息
        temp_photo_dir = self.controller.get_temp_photo_dir()
        if temp_photo_dir:
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            json_path = os.path.join(temp_photo_dir, f"{base_name}.json")

            # 标记是否加载了有效的检测结果
            has_detections = False

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_preview_info = json.load(f)
                    self._update_detection_info(self.current_preview_info)
                    has_detections = True
                except Exception as e:
                    logger.error(f"加载 {json_path} 文件失败: {e}")

            # === 修复：无论是否有检测结果，都强制更新物种选择器 ===
            # 这样当切换到无结果的图片时，列表会被重置为仅包含 Global
            self._update_species_selector_items()
            # ====================================================

            # 如果已勾选显示检测结果且有检测数据,则绘制检测框
            if has_detections and self.show_detection_checkbox.isChecked():
                self._redraw_preview_boxes_with_new_confidence(None)

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
        """当前活跃线程结束后的清理"""
        # 避免把新启动的线程引用给清除了
        sender = self.sender()
        if sender == self.image_loader_thread:
            self.image_loader_thread = None

    def _clear_thread_reference(self):
        """延迟清理线程引用"""
        self.image_loader_thread = None

    def _load_image_deferred(self):
        """延迟执行的文件加载逻辑（支持图片和视频）"""
        selected_items = self.file_listbox.selectedItems()
        if not selected_items:
            return

        file_path = selected_items[0].data(Qt.ItemDataRole.UserRole)

        if not file_path:
            return
        if file_path == self.requested_image_path:
            return

        self.requested_image_path = file_path

        # 1. 切换文件时，如果正在进行 OpenCV 视频检测，必须强制停止线程
        self._stop_video_detection_thread()

        is_video = file_path.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

        # 清理旧图片加载线程逻辑
        if self.image_loader_thread and self.image_loader_thread.isRunning():
            self.image_loader_thread.cancel()
            try:
                self.image_loader_thread.image_loaded.disconnect()
                self.image_loader_thread.loading_failed.disconnect()
            except:
                pass
            self._stopping_threads.append(self.image_loader_thread)
            self.image_loader_thread.finished.connect(
                lambda t=self.image_loader_thread: self._cleanup_stopped_thread(t)
            )
            self.image_loader_thread.wait(50)
            self.image_loader_thread = None

        if is_video:
            # === 视频模式 (修改版) ===
            self.current_image_path = file_path

            self.play_pause_shortcut.setEnabled(True)

            # 统一 UI 状态：隐藏原生视频容器，显示 image_label (用于 OpenCV 绘制)
            self.original_image = None

            # 按钮状态
            self.detect_button.setEnabled(False)
            self.show_detection_checkbox.setEnabled(True)

            # 准备路径
            temp_dir = self.controller.get_temp_photo_dir()
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join(temp_dir, f"{base_name}.json")

            # 获取当前是否需要显示检测框
            show_boxes = self.show_detection_checkbox.isChecked()

            # 获取当前的过滤设置
            min_ratio = 0.0
            if hasattr(self.controller, 'advanced_page'):
                min_ratio = self.controller.advanced_page.min_frame_ratio_var

            self._update_video_info_text(file_path, json_path, min_ratio)

            # === 修复开始：在加载视频时，立即读取 JSON 并更新下拉框 ===
            self.current_preview_info = {}
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_preview_info = json.load(f)
                except Exception as e:
                    logger.error(f"加载视频JSON失败: {e}")

            # 立即更新物种选择器
            self._update_species_selector_items()
            # === 修复结束 ===

            # === 启动 OpenCV 线程 ===
            self._start_video_detection_thread(file_path, json_path, draw_boxes=show_boxes)
            return

        # === 图片模式 ===

        # 禁用空格快捷键（交还给列表用于选择）
        self.play_pause_shortcut.setEnabled(False)

        # 恢复图片相关设置
        self.image_label.setVisible(True)

        # 启用图片检测功能
        self.detect_button.setEnabled(True)
        self.show_detection_checkbox.setEnabled(True)

        self.image_label.setText("正在加载图像...")
        self.image_label.pixmap = None
        self.info_text.setPlainText(f"正在加载: {os.path.basename(file_path)}")
        self.current_preview_info = {}

        # 启动新的加载线程
        self.image_loader_thread = ImageLoaderThread(file_path, self.image_label.size())
        self.image_loader_thread.image_loaded.connect(
            lambda pixmap, fp, info: self._on_image_loaded_safe(pixmap, fp, info)
        )
        self.image_loader_thread.loading_failed.connect(
            lambda fp, err: self._on_loading_failed_safe(fp, err)
        )
        self.image_loader_thread.finished.connect(self._on_thread_finished)
        self.image_loader_thread.start()

    def _cleanup_stopped_thread(self, thread):
        """清理已停止的线程"""
        if thread in self._stopping_threads:
            self._stopping_threads.remove(thread)
        thread.deleteLater()

    def toggle_video_playback(self):
        """切换视频播放/暂停状态（支持原生播放器和 OpenCV 检测回放）"""

        # === 场景 1: OpenCV 检测结果回放模式 ===
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.toggle_pause()
            # 这里可以扩展逻辑：例如显示/隐藏暂停图标
            return

    def eventFilter(self, source, event):
        """事件过滤器：处理 image_label 上的点击事件"""
        if source == self.image_label and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                # 仅在视频检测线程运行时响应点击
                if self.video_thread and self.video_thread.isRunning():
                    self.toggle_video_playback()
                    return True
        return super().eventFilter(source, event)

    def _start_video_detection_thread(self, video_path, json_path, draw_boxes=True, start_frame=0):
        """Starts the OpenCV QThread for video"""
        self._stop_video_detection_thread()
        self._is_video_paused = False

        conf = self.preview_conf_slider.value() / 100.0

        # 获取过滤比例设置
        min_ratio = 0.0
        if hasattr(self.controller, 'advanced_page'):
            min_ratio = self.controller.advanced_page.min_frame_ratio_var

        # 传递 start_frame 参数
        self.video_thread = VideoPlayerThread(
            video_path, json_path,
            self.species_conf_map,
            draw_boxes=draw_boxes,
            min_frame_ratio=min_ratio,
            start_frame=start_frame  # <--- 传递起始帧
        )

        self.video_thread.frame_ready.connect(self._on_video_frame_ready)
        self.video_thread.playback_finished.connect(self._on_video_finished)
        self.video_thread.pause_state_changed.connect(self._on_video_pause_state_changed)

        self.video_thread.start()

        # Update Info Text (根据状态显示不同提示)
        if draw_boxes:
            self.info_text.append("▶ 正在回放检测结果 (OpenCV模式)")
        else:
            self.info_text.append("▶ 正在播放视频 (OpenCV模式)")

    def _stop_video_detection_thread(self):
        """安全停止 OpenCV 视频检测线程"""
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop() # 假设你的 VideoPlayerThread 有 stop 方法设置 flag
            self.video_thread.quit()
            self.video_thread.wait() # 等待线程完全退出
            self.video_thread.deleteLater()
            self.video_thread = None

    def _on_video_frame_ready(self, pixmap):
        """接收线程传来的图像帧并在 QLabel 上显示"""
        if not self.isVisible():
            return

        # 1. 始终更新缓存（保证后台数据是最新的）
        self._current_video_frame_pixmap = pixmap

        # === 新增：关键修复 ===
        # 如果当前 UI 处于暂停状态，忽略后台传来的这一帧“迟到”的原始画面
        # 防止它覆盖掉我们刚刚绘制了暂停图标的画面
        if self._is_video_paused:
            return

        # 2. 正常显示
        if self.image_label.size().isValid():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def _on_video_finished(self):
        """视频播放线程结束后的回调"""
        # 当视频线程停止时（例如切换文件或出错），重置部分状态
        # 如果需要，可以在这里让播放图标重新显示，或者做清理工作
        pass

    def _generate_white_icon_pixmap(self, icon_path, size):
        """
        生成一个白色的 SVG 图标 Pixmap。
        原理：先渲染 SVG，然后利用 CompositionMode 填充白色。
        """
        if not os.path.exists(icon_path):
            return None

        # 1. 创建一个透明的画布
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        # 2. 渲染 SVG 到画布上
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        renderer = QSvgRenderer(icon_path)
        renderer.render(painter, QRectF(0, 0, size, size))

        # 3. 关键步骤：设置混合模式为 SourceIn
        # 这会保留原图像的 Alpha 通道（形状），但用新的颜色替换 RGB
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), Qt.GlobalColor.white)

        painter.end()
        return pixmap

    def _on_video_pause_state_changed(self, is_paused):
        """处理视频回放的暂停/播放状态改变 (修复版)"""
        self._is_video_paused = is_paused

        # 如果是恢复播放，不需要做特殊处理，等待下一帧刷新即可
        if not is_paused:
            return

        # 确保我们有当前的视频帧缓存
        if not hasattr(self, '_current_video_frame_pixmap') or not self._current_video_frame_pixmap:
            return

        try:
            # 1. 复制当前帧，避免修改原始缓存
            paused_pixmap = self._current_video_frame_pixmap.copy()

            # 2. 计算图标大小和位置
            w = paused_pixmap.width()
            h = paused_pixmap.height()

            # 图标大小为短边的 20%，但不小于 64px
            icon_size = max(64, int(min(w, h) * 0.2))

            # 计算居中坐标
            x = (w - icon_size) // 2
            y = (h - icon_size) // 2

            # 3. 开始绘制
            painter = QPainter(paused_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # --- 绘制半透明黑色圆形背景 ---
            # 背景比图标稍大一点
            bg_radius = icon_size // 2 + 10
            center_x = x + icon_size // 2
            center_y = y + icon_size // 2

            painter.setBrush(QColor(0, 0, 0, 100))  # 黑色，透明度 100/255
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(center_x, center_y), bg_radius, bg_radius)

            # --- 绘制白色图标 ---
            icon_path = resource_path(os.path.join("res", "icon", "play.svg"))

            # 使用辅助函数生成纯白色的图标
            white_icon_pixmap = self._generate_white_icon_pixmap(icon_path, icon_size)

            if white_icon_pixmap:
                # 将处理好的白色图标贴上去
                painter.drawPixmap(x, y, white_icon_pixmap)
            else:
                # 备用方案：如果 SVG 加载失败，画一个白色的三角形
                painter.setBrush(QColor(255, 255, 255))
                path = QPainterPath()
                # 简单的播放三角形
                path.moveTo(x + icon_size * 0.3, y + icon_size * 0.2)
                path.lineTo(x + icon_size * 0.3, y + icon_size * 0.8)
                path.lineTo(x + icon_size * 0.8, y + icon_size * 0.5)
                path.closeSubpath()
                painter.drawPath(path)

            painter.end()

            # 4. 显示最终图像
            if self.image_label.size().isValid():
                scaled_pixmap = paused_pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            logger.error(f"绘制暂停图标失败: {e}")

    def _update_video_info_text(self, file_path, json_path, min_frame_ratio=0.0):
        """生成详细的视频信息文本 (支持动态置信度过滤)"""
        try:
            # 1. 获取视频基础信息 (Opencv + OS)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].replace('.', '')
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_size_kb = file_size_mb * 1024
            mtime = os.path.getmtime(file_path)
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

            width, height = 0, 0
            # 简单获取宽高，如果不频繁调用可以接受
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            info_text = (f"文件名: {file_name}    格式: {file_ext}\n"
                         f"拍摄日期: {date_str}    尺寸: {width}x{height}px    文件大小: {file_size_kb:.1f} KB")

            # 2. 处理检测结果
            if os.path.exists(json_path):
                # 如果没有缓存或路径变了，才重新加载 JSON，否则使用内存中的 self.current_preview_info 可能会更快
                # 这里为了数据一致性，还是读取文件或使用传入的 json_path
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                total_frames = data.get('total_frames_processed', 1)
                tracks = data.get('tracks', {})
                threshold = total_frames * min_frame_ratio

                # 统计有效 Track
                species_count = defaultdict(int)
                min_confidence = 1.0
                has_detections = False

                for t_id, points in tracks.items():
                    # A. 过滤掉帧数不足的目标
                    if len(points) < threshold:
                        continue

                    # B. 获取该轨迹的物种 (投票法)
                    s_list = [p.get('species') for p in points if p.get('species')]
                    if not s_list: continue
                    sp = Counter(s_list).most_common(1)[0][0]

                    # === C. 新增：置信度过滤 ===
                    # 获取该物种的当前阈值
                    thresh = self.species_conf_map.get(sp, self.species_conf_map.get("global", 0.25))

                    # 检查该轨迹中是否至少有一帧（或平均值）超过了阈值？
                    # 通常策略：如果整个轨迹的最高置信度都低于阈值，则视为误检
                    track_max_conf = max([float(p.get('confidence', 0)) for p in points])

                    if track_max_conf < thresh:
                        continue
                    # ========================

                    has_detections = True
                    species_count[sp] += 1

                    # 更新最低置信度 (仅统计通过筛选的)
                    if track_max_conf < min_confidence:
                        min_confidence = track_max_conf

                # 获取检测时间
                json_mtime = os.path.getmtime(json_path)
                detect_time = datetime.fromtimestamp(json_mtime).strftime("%Y-%m-%d %H:%M:%S")

                if has_detections:
                    res_parts = []
                    for sp, count in species_count.items():
                        res_parts.append(f"{sp}: {count}只")
                    res_str = " | ".join(res_parts)
                    result_text = f"\n检测结果: | {res_str} | 最低置信度: {min_confidence:.3f} | 检测于: {detect_time}"
                else:
                    result_text = f"\n检测结果: 当前条件下未检测到有效目标"

                info_text += result_text
            else:
                info_text += "\n检测结果: 暂无数据"

            self.info_text.setPlainText(info_text)

        except Exception as e:
            logger.error(f"生成视频信息失败: {e}")

    def _try_connect_settings_signal(self):
        """尝试连接高级设置页面的信号"""
        if not self.settings_connected and hasattr(self.controller, 'advanced_page'):
            try:
                self.controller.advanced_page.settings_changed.connect(self._on_global_settings_changed)
                self.settings_connected = True
            except Exception as e:
                logger.warning(f"Failed to connect settings signal: {e}")

    def showEvent(self, event):
        """显示事件"""
        super().showEvent(event)
        self._try_connect_settings_signal()  # 确保信号已连接

    def _on_global_settings_changed(self):
        """响应全局设置变化"""
        # 仅在视频模式且开启检测显示时处理
        if not (self.current_image_path and
                self.current_image_path.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS) and
                self.show_detection_checkbox.isChecked()):
            return

        # 获取最新的 min_frame_ratio
        new_ratio = 0.0
        if hasattr(self.controller, 'advanced_page'):
            new_ratio = self.controller.advanced_page.min_frame_ratio_var

        # 优化：检查过滤比例是否真的发生了变化（避免调节其他设置时导致视频重载）
        if self.video_thread:
            current_ratio = self.video_thread.min_frame_ratio
            if abs(current_ratio - new_ratio) < 1e-6:
                return

        # 获取当前播放位置，以便无缝衔接
        start_frame = 0
        if self.video_thread and self.video_thread.isRunning():
            start_frame = self.video_thread.current_frame_index

        # 准备路径
        temp_dir = self.controller.get_temp_photo_dir()
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        json_path = os.path.join(temp_dir, f"{base_name}.json")

        # 更新信息栏 (显示新的过滤统计结果)
        self._update_video_info_text(self.current_image_path, json_path, new_ratio)

        # 重启视频线程 (带上 start_frame 实现无缝切换)
        self._start_video_detection_thread(
            self.current_image_path,
            json_path,
            draw_boxes=True,
            start_frame=start_frame
        )

    def reload_and_apply_conf(self):
        """
        [新增] 从 conf.json 重新加载配置并强制刷新界面。
        用于页面切换时同步最新的置信度设置。
        """
        # 1. 从磁盘加载最新配置到内存 (self.species_conf_map)
        self._load_species_conf()

        # 2. 刷新下拉框和滑块状态
        # 这会将滑块位置更新为最新的数值，并更新 self.preview_conf_label
        self._on_species_selector_changed()

        # 3. 刷新视图 (如果显示检测框已开启)
        if self.show_detection_checkbox.isChecked():
            # A. 视频模式：更新线程内的 map
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.conf_map = self.species_conf_map
                # 如果需要更新信息面板
                temp_dir = self.controller.get_temp_photo_dir()
                if self.current_image_path:
                    base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                    json_path = os.path.join(temp_dir, f"{base_name}.json")
                    min_ratio = 0.0
                    if hasattr(self.controller, 'advanced_page'):
                        min_ratio = self.controller.advanced_page.min_frame_ratio_var
                    self._update_video_info_text(self.current_image_path, json_path, min_ratio)

            # B. 图片模式：重绘框和更新信息
            elif self.original_image and self.current_preview_info:
                self._redraw_preview_boxes_with_new_confidence(None)
                self._update_detection_info()
