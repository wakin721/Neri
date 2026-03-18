from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QLabel, QPushButton, QFrame, QGroupBox,
    QMessageBox, QFileDialog, QInputDialog, QComboBox,
    QSizePolicy, QApplication, QDialog, QLineEdit, QFormLayout,
    QScrollArea, QCompleter, QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QEvent, QRectF, QPoint, QUrl, QStringListModel, QItemSelectionModel
from PySide6.QtGui import (
    QFont, QPalette, QPixmap, QImage, QPainter, QColor,
    QKeySequence, QShortcut, QPainterPath, QDesktopServices,
    QIntValidator
)
from PySide6.QtSvg import QSvgRenderer
import openpyxl
import os
import json
import logging
import cv2
import time
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil

from system.config import SUPPORTED_IMAGE_EXTENSIONS, get_species_color
from system.gui.ui_components import (
    Win11Colors, ModernSlider, ModernGroupBox, ModernComboBox,
    ThemeManager, MaterialMessageBox, ModernLineEdit, ScrollingListDelegate, MarqueeButton
)
from system.data_processor import DataProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.utils import resource_path

logger = logging.getLogger(__name__)


class ValidationExportThread(QThread):
    """用于后台处理和导出数据的独立线程，防止界面卡死"""
    progress_updated = Signal(int, int, float, float, float)
    finished = Signal(bool, str)

    def __init__(self, temp_dir, source_dir, output_path, file_format, confidence_settings, columns_to_export, min_frame_ratio, supported_video_exts, save_to_source=False):
        super().__init__()
        self.temp_dir = temp_dir
        self.source_dir = source_dir
        self.output_path = output_path
        self.file_format = file_format
        self.confidence_settings = confidence_settings
        self.columns_to_export = columns_to_export
        self.min_frame_ratio = min_frame_ratio
        self.supported_video_exts = supported_video_exts
        self.save_to_source = save_to_source

    def run(self):
        import time
        import os
        import json
        from datetime import datetime
        from system.metadata_extractor import ImageMetadataExtractor
        from system.config import SUPPORTED_IMAGE_EXTENSIONS
        from system.data_processor import DataProcessor
        from system.detection_db import get_db_path, get_all_detections_with_validation

        start_time = time.time()
        try:
            temp_db_path = get_db_path(self.temp_dir)
            source_db_path = get_db_path(self.source_dir)
            rows = []

            # 1. 优先从源目录 SQLite 数据库加载
            if self.save_to_source and os.path.exists(source_db_path):
                try:
                    rows = get_all_detections_with_validation(source_db_path)
                except Exception as e:
                    pass

            # 2. 回退到软件缓存的 SQLite 数据库
            if not rows and os.path.exists(temp_db_path):
                try:
                    rows = get_all_detections_with_validation(temp_db_path)
                except Exception as e:
                    pass

            search_extensions = list(SUPPORTED_IMAGE_EXTENSIONS)
            if self.supported_video_exts:
                search_extensions.extend(list(self.supported_video_exts))

            # 3. 【兼容老版本】如果数据库没有数据，回退读取 temp 目录下的 JSON 文件
            if not rows:
                json_files = [f for f in os.listdir(self.temp_dir) if
                              f.lower().endswith('.json') and f not in ('validation.json', 'conf.json')]
                for jf in json_files:
                    base_name = os.path.splitext(jf)[0]
                    rows.append({
                        'base_name': base_name,
                        'image_filename': base_name,
                        'json_path': os.path.join(self.temp_dir, jf)
                    })

            if not rows:
                self.finished.emit(False, "未能成功读取到任何有效数据，无法导出。")
                return

            total_files = len(rows)
            total_work = total_files * 2

            all_image_data = []
            earliest_date = None
            processed_units = 0

            # --- 阶段 1: 遍历数据记录 ---
            for row in rows:
                base_name = row['base_name']
                image_filename = row['image_filename']
                found_path = os.path.join(self.source_dir, image_filename)

                # 如果记录的文件名在源目录找不到，尝试容错扩展名
                if not os.path.exists(found_path):
                    found_path = None
                    for ext in search_extensions:
                        temp_path = os.path.join(self.source_dir, base_name + ext)
                        if os.path.exists(temp_path):
                            found_path = temp_path
                            break

                if not found_path:
                    processed_units += 1
                    continue

                is_video = False
                if self.supported_video_exts and os.path.splitext(found_path)[1].lower() in self.supported_video_exts:
                    is_video = True

                try:
                    metadata = {}
                    if is_video:
                        metadata['文件名'] = os.path.basename(found_path)
                        metadata['格式'] = os.path.splitext(found_path)[1].replace('.', '').upper()
                        mtime = os.path.getmtime(found_path)
                        dt_obj = datetime.fromtimestamp(mtime)
                        metadata['拍摄日期'] = dt_obj.strftime("%Y-%m-%d")
                        metadata['拍摄时间'] = dt_obj.strftime("%H:%M:%S")
                        metadata['拍摄日期对象'] = dt_obj
                    else:
                        metadata, _ = ImageMetadataExtractor.extract_metadata(found_path, os.path.basename(found_path))

                    # 3. 【兼容老版本】优先读 SQLite 的字段，没有则读物理 JSON 文件
                    if 'detection_json' in row:
                        json_data = json.loads(row['detection_json'])
                    else:
                        with open(row['json_path'], 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                    metadata.update(json_data)
                    all_image_data.append(metadata)

                    date_taken = metadata.get('拍摄日期对象')
                    if date_taken:
                        if earliest_date is None or date_taken < earliest_date:
                            earliest_date = date_taken
                except Exception as e:
                    pass  # 忽略单文件错误，继续处理

                # 刷新进度
                processed_units += 1
                elapsed = time.time() - start_time
                speed = processed_units / elapsed if elapsed > 0 else 0
                remain = (total_work - processed_units) / speed if speed > 0 else 0
                self.progress_updated.emit(processed_units, total_work, elapsed, remain, speed)

            if not all_image_data:
                self.finished.emit(False, "数据读取成功，但未找到对应的物理文件。")
                return

            # --- 阶段 2: 数据预处理 ---
            processed_data = DataProcessor.process_independent_detection(
                all_image_data, self.confidence_settings, min_frame_ratio=self.min_frame_ratio
            )
            if earliest_date:
                processed_data = DataProcessor.calculate_working_days(processed_data, earliest_date)

            # --- 阶段 3: 执行导出 ---
            def progress_cb(current, total):
                current_total = total_files + current
                elapsed = time.time() - start_time
                speed = current_total / elapsed if elapsed > 0 else 0
                remain = (total_work - current_total) / speed if speed > 0 else 0
                self.progress_updated.emit(current_total, total_work, elapsed, remain, speed)

            success = DataProcessor.export_to_excel(
                processed_data, self.output_path, self.confidence_settings,
                file_format=self.file_format, columns_to_export=self.columns_to_export,
                min_frame_ratio=self.min_frame_ratio, progress_callback=progress_cb
            )

            if success:
                self.finished.emit(True, self.output_path)
            else:
                self.finished.emit(False, "导出文件时发生错误，请查看日志文件获取详情。")

        except Exception as e:
            self.finished.emit(False, f"导出过程中发生异常: {str(e)}")


class BackgroundUpdateThread(QThread):
    """用于在后台处理批量物种校验和数据保存的独立线程，防止UI卡死"""
    finished = Signal(bool)

    def __init__(self, update_tasks, temp_photo_dir, source_dir, save_to_source, full_validation_data):
        super().__init__()
        self.update_tasks = update_tasks
        self.temp_photo_dir = temp_photo_dir
        self.source_dir = source_dir
        self.save_to_source = save_to_source
        self.full_validation_data = full_validation_data.copy()

    def run(self):
        import sqlite3
        import json
        import os
        from datetime import datetime
        from system.detection_db import get_db_path, update_detection, init_db, upsert_validation, delete_validation, delete_validation_bulk

        try:
            db_path = get_db_path(self.temp_photo_dir)
            if not os.path.exists(db_path):
                init_db(db_path)

            source_db_path = None
            if self.save_to_source and self.source_dir and os.path.isdir(self.source_dir):
                source_db_path = get_db_path(self.source_dir)
                if not os.path.exists(source_db_path):
                    init_db(source_db_path)

            # 1. 批量更新检测数据 (SQLite & JSON 回退)
            for task in self.update_tasks:
                file_name = task['file_name']
                base_name = os.path.splitext(file_name)[0]
                action = task.get('action')

                if action in ('empty', 'update', 'unverified'):
                    detection_info = {}
                    try:
                        with sqlite3.connect(db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT detection_json FROM detections WHERE base_name=?", (base_name,))
                            row = cursor.fetchone()
                            if row:
                                detection_info = json.loads(row[0])
                    except Exception:
                        pass

                    json_path = os.path.join(self.temp_photo_dir, f"{base_name}.json")
                    if not detection_info and os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            detection_info = json.load(f)

                    changed = False
                    if action == 'unverified':
                        if detection_info.get('最低置信度') == '人工校验':
                            for key in ['最低置信度', '物种名称', '物种数量', '备注', '检测时间']:
                                if key in detection_info:
                                    detection_info.pop(key, None)
                                    changed = True
                    else:
                        if task.get('new_species') is not None:
                            detection_info['物种名称'] = task['new_species']
                            detection_info['最低置信度'] = '人工校验'
                            detection_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            changed = True
                        if task.get('new_count') is not None:
                            detection_info['物种数量'] = task['new_count']
                            changed = True
                        if task.get('new_remark') is not None:
                            detection_info['备注'] = task['new_remark']
                            changed = True

                    if changed:
                        update_detection(db_path, base_name, detection_info)
                        if source_db_path:
                            update_detection(source_db_path, base_name, detection_info)

            # 2. 批量处理 validation.json 校验状态
            files_to_remove = [t['file_name'] for t in self.update_tasks if t.get('action') == 'unverified']
            for f in files_to_remove:
                self.full_validation_data.pop(f, None)

            for filename, value in self.full_validation_data.items():
                if value is None:
                    delete_validation(db_path, filename)
                    if source_db_path:
                        delete_validation(source_db_path, filename)
                else:
                    upsert_validation(db_path, filename, bool(value))
                    if source_db_path:
                        upsert_validation(source_db_path, filename, bool(value))

            if files_to_remove:
                delete_validation_bulk(db_path, files_to_remove)
                if source_db_path:
                    delete_validation_bulk(source_db_path, files_to_remove)

            val_json_path = os.path.join(self.temp_photo_dir, "validation.json")
            with open(val_json_path, 'w', encoding='utf-8') as f:
                clean_dict = {k: v for k, v in self.full_validation_data.items() if v is not None}
                json.dump(clean_dict, f, ensure_ascii=False, indent=2, sort_keys=True)

            self.finished.emit(True)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"后台批量更新失败: {e}")
            self.finished.emit(False)


class QuantityInputDialog(QDialog):
    """自定义 Material You 风格的数量输入弹窗"""
    def __init__(self, parent=None, title="输入数量", prompt="请输入物种的数量 (1-999):", default_value=1):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.result_value = default_value

        # 获取当前主题状态，动态适应深/浅色模式
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD.name()
            border_color = Win11Colors.DARK_BORDER.name()
            text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            btn_bg = Win11Colors.DARK_ACCENT.name()
            btn_text = "#ffffff"
            btn_hover = Win11Colors.DARK_ACCENT.lighter(120).name()
            btn_pressed = Win11Colors.DARK_ACCENT.darker(110).name()
            cancel_bg = Win11Colors.DARK_SURFACE.name()
            cancel_text = Win11Colors.DARK_TEXT_PRIMARY.name()
            cancel_hover = Win11Colors.DARK_HOVER.name()
        else:
            bg_color = Win11Colors.LIGHT_CARD.name()
            border_color = Win11Colors.LIGHT_BORDER.name()
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            btn_bg = Win11Colors.LIGHT_ACCENT.name()
            btn_text = "#ffffff"
            btn_hover = Win11Colors.LIGHT_ACCENT.darker(110).name()
            btn_pressed = Win11Colors.LIGHT_ACCENT.darker(120).name()
            cancel_bg = Win11Colors.LIGHT_SURFACE.name()
            cancel_text = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            cancel_hover = Win11Colors.LIGHT_HOVER.name()

        # 动态主题样式 (移除了 QLineEdit 的硬编码样式，交由 ModernLineEdit 自身处理)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QLabel {{
                color: {text_color};
                font-size: 15px;
                font-weight: bold;
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {btn_text};
                border: none;
                padding: 8px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
            QPushButton:pressed {{
                background-color: {btn_pressed};
            }}
            QPushButton#cancelButton {{
                background-color: {cancel_bg};
                color: {cancel_text};
                border: 1px solid {border_color};
            }}
            QPushButton#cancelButton:hover {{
                background-color: {cancel_hover};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 20)

        self.prompt_label = QLabel(prompt)
        layout.addWidget(self.prompt_label)

        # 限制只能输入 1 到 999 的数字，替换为 ModernLineEdit
        self.input_field = ModernLineEdit()
        self.input_field.setText(str(default_value))
        self.input_field.setValidator(QIntValidator(1, 999, self))
        # 默认全选文本，方便用户直接覆盖输入
        self.input_field.selectAll()
        layout.addWidget(self.input_field)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept_input)
        self.ok_btn.setDefault(True)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.resize(300, 150)

    def accept_input(self):
        text = self.input_field.text().strip()
        if text and text.isdigit() and int(text) > 0:
            self.result_value = int(text)
            self.accept()
        else:
            self.input_field.setFocus()


class CorrectionDialog(QDialog):
    """用于修正物种信息的弹窗"""

    _cached_completer_items = None

    def __init__(self, parent, title="修正信息", original_info=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.parent = parent
        self.result = None
        self.original_info = original_info

        # 获取当前主题状态，动态适应深/浅色模式
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD.name()
            border_color = Win11Colors.DARK_BORDER.name()
            text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            btn_bg = Win11Colors.DARK_ACCENT.name()
            btn_text = "#ffffff"
            btn_hover = Win11Colors.DARK_ACCENT.lighter(120).name()
            btn_pressed = Win11Colors.DARK_ACCENT.darker(110).name()
            cancel_bg = Win11Colors.DARK_SURFACE.name()
            cancel_text = Win11Colors.DARK_TEXT_PRIMARY.name()
            cancel_hover = Win11Colors.DARK_HOVER.name()
        else:
            bg_color = Win11Colors.LIGHT_CARD.name()
            border_color = Win11Colors.LIGHT_BORDER.name()
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            btn_bg = Win11Colors.LIGHT_ACCENT.name()
            btn_text = "#ffffff"
            btn_hover = Win11Colors.LIGHT_ACCENT.darker(110).name()
            btn_pressed = Win11Colors.LIGHT_ACCENT.darker(120).name()
            cancel_bg = Win11Colors.LIGHT_SURFACE.name()
            cancel_text = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            cancel_hover = Win11Colors.LIGHT_HOVER.name()

        # 动态设置新的颜色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QLabel {{
                color: {text_color};
                font-size: 14px;
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {btn_text};
                border: none;
                padding: 6px 20px;
                border-radius: 16px;  /* Material You 药丸形圆角 */
                min-height: 32px;     /* 保证按钮有足够的高度 */
                font-size: 14px;
                font-weight: 600;     /* 字体稍微加粗 */
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
            QPushButton:pressed {{
                background-color: {btn_pressed};
            }}
            QPushButton#cancelButton {{
                background-color: {cancel_bg};
                color: {cancel_text};
                border: 1px solid {border_color};
                border-radius: 16px;  /* 取消按钮同样应用圆角 */
            }}
            QPushButton#cancelButton:hover {{
                background-color: {cancel_hover};
            }}
        """)

        try:
            self.db_path = resource_path(os.path.join("res", "species_database.db"))
        except NameError:
            self.db_path = os.path.join("res", "species_database.db")

        # 初始化输入框 - 替换为 ModernLineEdit
        self.species_name_edit = ModernLineEdit()
        self.species_count_edit = ModernLineEdit()
        self.species_type_edit = ModernLineEdit()
        self.remark_edit = ModernLineEdit()

        self.species_name_edit.textChanged.connect(self._auto_fill_type_from_db)
        self.species_name_edit.textChanged.connect(self._auto_update_count)

        # 如果有原始信息，则预先填充输入框
        if self.original_info:
            recalculated_info = {}
            if self.original_info.get('最低置信度') != '人工校验':
                # 直接读取父页面已计算好的标签文本
                info_text = parent.species_info_label.text() if hasattr(parent, 'species_info_label') else ""

                sp_name = ""
                sp_count = ""
                for part in info_text.split(" | "):
                    part = part.strip()
                    if part.startswith("物种:"):
                        sp_name = part.replace("物种:", "").strip()
                    elif part.startswith("数量:"):
                        sp_count = part.replace("数量:", "").strip()

                # 过滤掉无效或占位字符
                if sp_name in ["[未校验] 空", "[已校验] 空", "未知", "未检测", "需人工检验", ""]:
                    sp_name = "空"
                if sp_count in ["空", "未知", ""]:
                    sp_count = "空"

                recalculated_info['物种名称'] = sp_name
                recalculated_info['物种数量'] = sp_count
            else:
                recalculated_info['物种名称'] = self.original_info.get('物种名称', '')
                recalculated_info['物种数量'] = self.original_info.get('物种数量', '')

            self.species_name_edit.setText(recalculated_info.get('物种名称', ''))
            self.species_count_edit.setText(recalculated_info.get('物种数量', ''))
            self.remark_edit.setText(self.original_info.get('备注', ''))

            # 自动触发类型补全
            self._auto_fill_type_from_db()

        self._setup_completer()
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
        form_layout.addRow("物种类型:", self.species_type_edit)
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

        self.resize(396, 250)  # 稍微加高一点以适应有内边距的 ModernLineEdit

    def _setup_completer(self):
        """初始化拼音及中文自动补全器 (使用 SQLite)"""
        if CorrectionDialog._cached_completer_items is None:
            items = []
            try:
                import sqlite3
                if os.path.exists(self.db_path):
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 中文名 FROM species WHERE 中文名 IS NOT NULL AND 中文名 != ''")

                    try:
                        import pypinyin
                        has_pypinyin = True
                    except ImportError:
                        has_pypinyin = False
                        logger.warning("未安装 pypinyin，拼音首字母检索将不可用。请运行 pip install pypinyin")

                    for row in cursor.fetchall():
                        name = str(row[0]).strip()
                        if name:
                            if has_pypinyin:
                                pinyin_list = pypinyin.pinyin(name, style=pypinyin.Style.FIRST_LETTER, strict=False)
                                initials = "".join([p[0][0] for p in pinyin_list]).lower()
                                items.append(f"{name} ({initials})")
                            else:
                                items.append(name)
                    conn.close()
            except Exception as e:
                logger.error(f"加载自动补全词库失败: {e}")

            CorrectionDialog._cached_completer_items = list(set(items))

        self.species_completer = MultiSpeciesCompleter(CorrectionDialog._cached_completer_items, self)
        self.species_completer.activated.connect(lambda text: self._auto_fill_type_from_db())
        self.species_name_edit.setCompleter(self.species_completer)

    def _auto_update_count(self, text):
        """根据物种名称的数量，动态同步数量框的数字个数"""
        current_count = self.species_count_edit.text().strip()

        # 1. 提取当前输入的物种种类数
        names = [n.strip() for n in text.replace('，', ',').split(',') if n.strip()]
        num_species = len(names)

        if num_species == 0:
            self.species_count_edit.setText("")
            return

        # 2. 解析当前数量框内的数字
        if current_count == "空" or not current_count:
            current_counts = []
        else:
            current_counts = [c.strip() for c in current_count.replace('，', ',').split(',') if c.strip()]

        # 3. 对齐数字个数与物种个数
        if len(current_counts) < num_species:
            missing = num_species - len(current_counts)
            current_counts.extend(["1"] * missing)
        elif len(current_counts) > num_species:
            current_counts = current_counts[:num_species]

        # 4. 回填到输入框
        self.species_count_edit.setText(",".join(current_counts))

    def _auto_fill_type_from_db(self, text=None):
        full_name_str = self.species_name_edit.text().strip().replace('，', ',')
        if not full_name_str:
            self.species_type_edit.setText("")
            return

        species_names = [n.strip() for n in full_name_str.split(',') if n.strip()]
        found_types = []

        for name in species_names:
            sType = self.parent._get_species_info_from_db(name)
            found_types.append(sType if sType else "空")

        combined_type = ",".join(found_types)
        self.species_type_edit.setText(combined_type)

    def _update_sqlite_db(self, name_str, type_str):
        """更新或新增 SQLite 中的物种类型"""
        if not name_str or not type_str:
            return

        names = [n.strip() for n in name_str.replace('，', ',').split(',') if n.strip()]
        types = [t.strip() for t in type_str.replace('，', ',').split(',') if t.strip()]

        if len(names) != len(types):
            logger.warning(f"物种数量({len(names)})与类型数量({len(types)})不匹配，跳过自动更新数据库。")
            return

        if os.path.exists(self.db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                updated_any = False

                for s_name, s_type in zip(names, types):
                    if not s_name or not s_type:
                        continue

                    cursor.execute("SELECT 1 FROM species WHERE 中文名=?", (s_name,))
                    if cursor.fetchone():
                        cursor.execute("UPDATE species SET 物种类型=? WHERE 中文名=?", (s_type, s_name))
                        logger.info(f"更新物种数据库: {s_name} -> {s_type}")
                    else:
                        cursor.execute("INSERT INTO species (中文名, 物种类型) VALUES (?, ?)", (s_name, s_type))
                        logger.info(f"新增物种到数据库: {s_name} -> {s_type}")
                    updated_any = True

                if updated_any:
                    conn.commit()
                    logger.info("数据库已成功保存。")
                conn.close()
            except Exception as e:
                logger.error(f"更新物种数据库失败: {e}")

    def accept_input(self):
        species_name = self.species_name_edit.text().strip().replace('，', ',')
        species_count_str = self.species_count_edit.text().strip().replace('，', ',')
        remark = self.remark_edit.text().strip()
        species_type = self.species_type_edit.text().strip()

        if not species_name:
            MaterialMessageBox.warning(self, "输入错误", "物种名称不能为空。")
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

        if species_count_str.lower() != '空':
            try:
                counts = [int(c.strip()) for c in species_count_str.split(',')]
                if not all(c > 0 for c in counts):
                    raise ValueError("数量必须是正整数。")
            except ValueError:
                MaterialMessageBox.warning(
                    self,
                    "输入格式错误",
                    "物种数量必须为以下格式之一：\n\n"
                    "1. 单个正整数 (例如: 3)\n"
                    "2. 以英文逗号隔开的多个正整数 (例如: 5,2)\n"
                    '3. 文字"空"'
                )
                return

        if species_type:
            self._update_sqlite_db(species_name, species_type)

        self.result = (species_name, species_count_str, remark)
        self.accept()


class MultiSpeciesCompleter(QCompleter):
    """支持多物种（逗号分隔）、拼音过滤以及优先级排序的自动补全器"""

    def __init__(self, items, parent=None):
        super().__init__(items, parent)
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # 允许部分匹配 (输入 bm 能匹配到 豹猫 (bm))
        self.setFilterMode(Qt.MatchFlag.MatchContains)

        # 保存原始列表，并使用 QStringListModel，以便我们能动态刷新词库顺序
        self.original_items = items
        self.source_model = QStringListModel(self.original_items, self)
        self.setModel(self.source_model)

    def pathFromIndex(self, index):
        # 1. 获取选中的下拉项原始文本 (例如 "豹猫 (bm)")
        text = self.model().data(index, Qt.ItemDataRole.DisplayRole)
        # 2. 剥离拼音，只保留真实物种名
        clean_text = text.split(" (")[0] if " (" in text else text

        # 3. 将新选择的物种名拼接到当前输入框文本的最后一个逗号后面
        le = self.widget()
        if le:
            current_text = le.text()
            normalized_text = current_text.replace('，', ',')
            if ',' in normalized_text:
                parts = normalized_text.split(',')
                parts[-1] = clean_text
                return ",".join(parts)
        return clean_text

    def splitPath(self, path):
        # 规范化逗号，并获取当前正在搜索的部分
        normalized_path = path.replace('，', ',')
        if ',' in normalized_path:
            search_text = normalized_path.split(',')[-1].strip()
        else:
            search_text = normalized_path.strip()

        search_lower = search_text.lower()

        # 根据当前输入内容动态对基础词库进行排序
        if search_lower:
            def sort_key(item):
                item_lower = item.lower()
                # 安全提取拼音和中文名称
                if " (" in item_lower and item_lower.endswith(")"):
                    name, pinyin = item_lower.rsplit(" (", 1)
                    pinyin = pinyin[:-1]
                else:
                    name = item_lower
                    pinyin = ""

                # 优先级设定（数值越小越靠前）：
                # 0: 拼音完全匹配 (例如输入 "bl" 匹配 "白鹭 (bl)")
                if search_lower == pinyin: return 0
                # 1: 中文完全匹配 (例如输入 "白鹭" 匹配 "白鹭 (bl)")
                if search_lower == name: return 1
                # 2: 拼音首字母前缀匹配 (例如输入 "b" 匹配 "白鹭 (bl)")
                if pinyin.startswith(search_lower): return 2
                # 3: 中文前缀匹配 (例如输入 "白" 匹配 "白鹭 (bl)")
                if name.startswith(search_lower): return 3
                # 4: 拼音包含匹配 (例如输入 "bl" 匹配 "牛背鹭 (nbl)")
                if search_lower in pinyin: return 4
                # 5: 中文包含匹配
                if search_lower in name: return 5
                # 6: 其他兜底
                return 6

            # 先按优先级数值排序，如果优先级相同则按原本的首字母顺序(保证排序稳定性)
            sorted_items = sorted(self.original_items, key=lambda x: (sort_key(x), x))
            self.source_model.setStringList(sorted_items)
        else:
            # 输入为空时恢复默认字母顺序
            self.source_model.setStringList(sorted(self.original_items))

        return [search_text]


class NoArrowKeyListWidget(QListWidget):
    """一个将上下方向键事件传递给父控件的QListWidget子类"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Down:
            # 忽略事件，让它冒泡到父控件进行处理
            event.ignore()
        else:
            # 对于其他按键，保持默认行为
            super().keyPressEvent(event)


class KeepSelectionListWidget(QListWidget):
    """一个防止点击空白处丢失选中状态的 QListWidget"""

    def mousePressEvent(self, event):
        # 获取当前鼠标点击位置的 item
        item = self.itemAt(event.pos())

        # 如果没有点击到任何 item（即点击了空白区域）
        if not item:
            # 如果是右键点击，我们手动发射呼出菜单的信号
            if event.button() == Qt.MouseButton.RightButton:
                self.customContextMenuRequested.emit(event.pos())

            # 直接返回，不再向下传递事件。彻底阻止 Qt 底层清空选中状态！
            return

        # 如果点击到了具体的照片 item，按常规处理
        super().mousePressEvent(event)


class VideoPlayerThread(QThread):
    """
    视频播放线程：读取视频流，绘制检测框（支持中文），并发送 QPixmap。
    包含帧过滤和物种投票逻辑。
    """
    frame_ready = Signal(QPixmap)
    playback_finished = Signal()
    pause_state_changed = Signal(bool)

    def __init__(self, video_path, detection_data, conf_threshold, draw_boxes=True, min_frame_ratio=0.0, start_frame=0,
                 parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_data = detection_data
        self.conf_threshold = conf_threshold
        self.draw_boxes = draw_boxes
        self.min_frame_ratio = min_frame_ratio
        self.start_frame = start_frame
        self.running = False
        self.paused = False
        self.current_frame_index = 0
        self._needs_refresh = False  # 刷新标志

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

    def refresh_frame(self):
        """强制刷新当前帧（用于暂停时更新检测框）"""
        if self.paused:
            self._needs_refresh = True

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

        self._needs_refresh = False

        while self.running:
            # 暂停逻辑：支持强制刷新
            if self.paused:
                if self._needs_refresh:
                    # 回退一帧以重新读取当前帧
                    seek_target = max(0, self.current_frame_index - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    self._needs_refresh = False
                else:
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
                lookup_idx = self.current_frame_index - (self.current_frame_index % stride)
                if lookup_idx in detections:
                    self._draw_boxes_pil(pil_img, detections[lookup_idx])

            # 转换为 QImage/QPixmap
            img_data = pil_img.tobytes()
            w, h = pil_img.size
            bytes_per_line = 3 * w
            qt_image = QImage(img_data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            self.frame_ready.emit(QPixmap.fromImage(qt_image))

            if self.paused:
                continue

            process_time = time.time() - start_time
            wait_time = max(0, frame_delay - process_time)
            time.sleep(wait_time)

        cap.release()
        self.playback_finished.emit()

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

    def _parse_tracking_json(self):
        """解析内存中的跟踪数据字典，转换为以帧为索引的字典"""
        parsed_frames = {'frames': {}, 'stride': 1}

        # 变更为直接使用传入的字典，不再 open file
        if not self.detection_data:
            return parsed_frames

        try:
            data = self.detection_data
            parsed_frames['stride'] = data.get('vid_stride', 1)
            total_frames = data.get('total_frames_processed', 0)

            # 过滤逻辑
            tracks = data.get('tracks', {})
            min_frames_threshold = total_frames * self.min_frame_ratio

            for track_id, track_list in tracks.items():
                # 过滤掉帧数不足的目标
                if len(track_list) < min_frames_threshold:
                    continue

                # 计算该轨迹的最终物种（投票法）
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
                        point['species'] = final_species
                        parsed_frames['frames'][f_idx].append(point)

        except Exception as e:
            logger.error(f"JSON Parse Error: {e}")

        return parsed_frames

    def _draw_boxes_pil(self, pil_img, boxes):
        """在PIL图像上绘制检测框"""
        draw = ImageDraw.Draw(pil_img)
        img_w, img_h = pil_img.size

        for box in boxes:
            species = box.get('species', 'Unknown')
            track_id = box.get('track_id', '?')
            conf = box.get('confidence', 0)

            # 使用当前的置信度阈值进行过滤
            if conf < self.conf_threshold:
                continue

            bbox = box.get('bbox')
            if not bbox: continue

            try:
                x1_f, y1_f, x2_f, y2_f = map(float, bbox[:4])
                # 坐标反归一化处理
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

            # 绘制标签背景和文字
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
        # 确保 controller 有 confidence_settings 字典
        if not hasattr(self.controller, 'confidence_settings'):
            self.controller.confidence_settings = {"global": 0.25}

        # 加载 temp/conf.json 中的配置到 controller.confidence_settings
        self._load_species_conf()

        # 设置当前置信度变量 (优先使用加载后的 global 值)
        self.species_conf_var = self.controller.confidence_settings.get("global", 0.25)
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
        self._selected_species_names = []
        self._selected_counts = []
        self._selected_species_buttons = []
        self._selected_quantity_buttons = []

        self.species_validation_original_image = None

        # 防抖动定时器，用于滑动结束后的列表刷新
        self._list_refresh_timer = QTimer(self)
        self._list_refresh_timer.setSingleShot(True)
        self._list_refresh_timer.setInterval(600)  # 600毫秒无操作后刷新列表
        self._list_refresh_timer.timeout.connect(self._refresh_species_list_logic)

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
                # 如果创建失败，回退使用 controller 提供的目录
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

            # 1. 尝试加载现有文件并更新到 controller.confidence_settings
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.controller.confidence_settings.update(data)

            # 2. 确保内存中始终包含 'global' 键
            if "global" not in self.controller.confidence_settings:
                self.controller.confidence_settings["global"] = 0.25

            # 3. 如果文件不存在，则保存当前默认配置到文件
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
                json.dump(self.controller.confidence_settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 conf.json 失败: {e}")

    def _apply_theme(self):
        """应用主题样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        # 获取 Material You 风格的滚动条样式
        scrollbar_style = ThemeManager._get_scrollbar_style(is_dark)

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

            # 设置 Win11 风格并注入滚动条样式
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
                            padding: 6px 12px;
                            border-radius: 12px;
                            margin: 2px 4px 2px 4px;
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

                        {scrollbar_style}
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

            # 设置 Win11 风格并注入滚动条样式
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
                            padding: 6px 12px;
                            border-radius: 12px;
                            margin: 2px 4px 2px 4px;
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

                        {scrollbar_style}
                    """)

        # 刷新图片占位标签的独立样式
        if hasattr(self, 'species_image_label'):
            self.species_image_label.setStyleSheet(self._get_placeholder_style())

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 创建主要内容区域
        main_frame = QFrame()
        main_layout = QHBoxLayout(main_frame)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        layout.addWidget(main_frame)

        # 左侧面板
        self._create_left_panel(main_layout)

        # 右侧面板
        self._create_right_panel(main_layout)

    def _create_left_panel(self, parent_layout):
        """创建左侧面板"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_panel.setFixedWidth(200)

        # 物种列表
        species_list_group = ModernGroupBox("物种列表")
        species_list_layout = QVBoxLayout(species_list_group)
        self.species_listbox = NoArrowKeyListWidget()
        self.species_listbox.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.species_listbox.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.species_listbox.itemClicked.connect(self._on_species_selected)
        species_list_layout.addWidget(self.species_listbox)
        left_layout.addWidget(species_list_group)
        self._species_scroll_delegate = ScrollingListDelegate(self.species_listbox)
        self.species_listbox.setItemDelegate(self._species_scroll_delegate)

        # 照片文件列表
        photo_list_group = ModernGroupBox("照片文件")
        photo_list_layout = QVBoxLayout(photo_list_group)
        self.species_photo_listbox = KeepSelectionListWidget()
        self.species_photo_listbox.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.species_photo_listbox.setResizeMode(QListWidget.ResizeMode.Adjust)
        # 挂载滚动委托，文件名超出宽度时悬停自动滚动
        self._photo_scroll_delegate = ScrollingListDelegate(self.species_photo_listbox)
        self.species_photo_listbox.setItemDelegate(self._photo_scroll_delegate)

        # 开启多选模式 (支持 Shift 和 Ctrl)
        self.species_photo_listbox.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # 开启自定义右键菜单支持
        self.species_photo_listbox.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.species_photo_listbox.customContextMenuRequested.connect(self._show_photo_context_menu)

        self.species_photo_listbox.itemSelectionChanged.connect(self._on_species_photo_selected)
        photo_list_layout.addWidget(self.species_photo_listbox)
        left_layout.addWidget(photo_list_group)

        parent_layout.addWidget(left_panel)

    def _create_right_panel(self, parent_layout):
        """创建右侧面板"""
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 顶部区域
        top_area_frame = QWidget()
        top_layout = QHBoxLayout(top_area_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 图片显示区域
        self.species_image_display_frame = ModernGroupBox("图片显示")
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
        """创建操作按钮区域 (Material You 风格)"""
        action_buttons_group = ModernGroupBox("快速标记")
        action_buttons_layout = QVBoxLayout(action_buttons_group)
        action_buttons_group.setFixedWidth(110)

        # 稍微增加间距以适应圆润的按钮
        action_buttons_layout.setContentsMargins(6, 6, 6, 8)
        action_buttons_layout.setSpacing(4)

        # Material You 基础按钮样式 (药丸形状)
        material_btn_style = """
            QPushButton {
                max-width: 80px;
                min-width: 60px;
                min-height: 30px;
                padding: 4px 8px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 14px;
            }
        """

        correct_button = QPushButton("正确")
        correct_button.setStyleSheet(material_btn_style)
        correct_button.clicked.connect(lambda: self._mark_and_move_to_next(True))
        action_buttons_layout.addWidget(correct_button)

        empty_button = QPushButton("空")
        empty_button.setStyleSheet(material_btn_style)
        empty_button.clicked.connect(lambda: self._mark_and_move_to_next(species_name="空", count="空"))
        action_buttons_layout.addWidget(empty_button)

        # 创建一个 QScrollArea 来容纳物种按钮
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 物种按钮容器
        self.species_buttons_frame = QWidget()
        self.species_buttons_layout = QVBoxLayout(self.species_buttons_frame)
        self.species_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.species_buttons_layout.setSpacing(4)
        self.species_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area.setWidget(self.species_buttons_frame)
        action_buttons_layout.addWidget(scroll_area)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("border: none; background-color: rgba(128, 128, 128, 0.3); max-height: 1px; margin: 4px 0px;")
        action_buttons_layout.addWidget(separator)

        other_button = QPushButton("其他")
        other_button.setStyleSheet(material_btn_style)
        other_button.clicked.connect(self._mark_other_species)
        action_buttons_layout.addWidget(other_button)

        parent_layout.addWidget(action_buttons_group)

    def _create_quantity_buttons(self, parent_layout):
        """创建数量按钮区域 (Material You 风格)"""
        self.quantity_buttons_frame = ModernGroupBox("数量")
        main_layout = QVBoxLayout(self.quantity_buttons_frame)
        self.quantity_buttons_frame.setFixedWidth(80)

        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # 创建一个 QScrollArea 来容纳数量按钮，防止被挤压
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # 隐藏滚动条，保持界面简洁（可用滚轮滚动）

        # 数量按钮容器
        self.qty_scroll_widget = QWidget()
        quantity_buttons_layout = QVBoxLayout(self.qty_scroll_widget)
        quantity_buttons_layout.setContentsMargins(0, 0, 0, 0)
        quantity_buttons_layout.setSpacing(4)
        quantity_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 数量按钮的 Material 样式 (强制锁定高度 25px，圆角 12px)
        material_qty_style = """
            QPushButton {
                max-width: 58px;
                min-width: 45px;
                min-height: 25px;
                max-height: 25px;
                padding: 4px;
                font-size: 14px;
                font-weight: 600;
                text-align: center;
                border-radius: 12px;
            }
        """

        quantity_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50]
        for i in quantity_values:
            btn = QPushButton(str(i))
            btn.setStyleSheet(material_qty_style)
            # 绑定点击事件，将对应的数字和按钮对象传过去
            btn.clicked.connect(lambda checked, num=i, b=btn: self._on_quantity_button_press(str(num), b))
            quantity_buttons_layout.addWidget(btn)

        more_button = QPushButton("更多")
        more_button.setStyleSheet("""
            QPushButton {
                max-width: 58px;
                min-width: 45px;
                min-height: 25px;
                max-height: 25px;
                padding: 4px;
                font-size: 13px;
                font-weight: 600;
                text-align: center;
                border-radius: 12px;
            }
        """)
        more_button.clicked.connect(lambda: self._on_quantity_button_press("更多", more_button))
        quantity_buttons_layout.addWidget(more_button)

        # 将装有按钮的容器放入滚动区域，并将滚动区域加入主布局
        scroll_area.setWidget(self.qty_scroll_widget)
        main_layout.addWidget(scroll_area)

        # === 底部固定区域（分隔线与撤回按钮） ===
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(
            "border: none; background-color: rgba(128, 128, 128, 0.3); max-height: 1px; margin: 4px 0px;")
        main_layout.addWidget(separator)

        undo_button = QPushButton()
        undo_button.setToolTip("撤销上一步操作")

        try:
            from PySide6.QtGui import QIcon
            from system.utils import resource_path
            import os
            icon_path = resource_path(os.path.join("res", "icon", "return.svg"))
            white_icon_pixmap = self._generate_white_icon_pixmap(icon_path, 18)
            if white_icon_pixmap:
                undo_button.setIcon(QIcon(white_icon_pixmap))
            else:
                undo_button.setText("撤回")
        except Exception as e:
            logger.error(f"加载撤回图标失败: {e}")
            undo_button.setText("撤回")

        undo_button.setStyleSheet(material_qty_style)
        undo_button.clicked.connect(self._undo_last_action)
        main_layout.addWidget(undo_button)

        parent_layout.addWidget(self.quantity_buttons_frame)

    def _create_bottom_area(self, parent_layout):
        """创建底部区域"""
        bottom_area_frame = QWidget()
        bottom_layout = QHBoxLayout(bottom_area_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 检测信息与设置区域
        info_slider_group = ModernGroupBox("检测信息与设置")
        info_slider_layout = QVBoxLayout(info_slider_group)

        self.species_info_label = QLabel("物种:  | 数量:  | 类型:  | 置信度: ")
        info_slider_layout.addWidget(self.species_info_label)

        # 置信度控制
        conf_control_layout = QHBoxLayout()
        conf_control_layout.addWidget(QLabel("选择物种:"))
        self.species_selector = ModernComboBox()
        self.species_selector.addItem("全局设置 (Global)", "global")
        self.species_selector.currentIndexChanged.connect(self._on_species_selector_changed)
        conf_control_layout.addWidget(self.species_selector)

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

        # 1. 限制导出区域的最大宽度
        export_options_group.setMaximumWidth(300)

        export_layout = QHBoxLayout(export_options_group)
        export_layout.setContentsMargins(8, 8, 8, 8)  # 稍微收紧内边距
        export_layout.setSpacing(8)  # 缩小下拉框和按钮之间的间隙

        self.format_combo = ModernComboBox()
        self.format_combo.addItems(["CSV", "Excel", "错误照片"])
        self.format_combo.setCurrentText(self.export_format_var)
        self.format_combo.currentTextChanged.connect(self._on_export_format_changed)
        export_layout.addWidget(self.format_combo)

        # 导出按钮 (Material You)
        self.export_button = QPushButton("导出")

        self.export_button.setStyleSheet("""
                    QPushButton {
                        min-height: 15px;
                        padding: 12px;  
                        font-size: 14px;
                        font-weight: 600;
                        border-radius: 12px;
                    }
                """)
        self.export_button.clicked.connect(self._dispatch_export)
        export_layout.addWidget(self.export_button)

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
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            return f"""
                QLabel {{
                    border: 2px dashed {Win11Colors.DARK_BORDER.name()};
                    border-radius: 8px;
                    background-color: {Win11Colors.DARK_SURFACE.name()};
                    color: {Win11Colors.DARK_TEXT_SECONDARY.name()};
                    font-size: 16px;
                }}
            """
        else:
            return f"""
                QLabel {{
                    border: 2px dashed {Win11Colors.LIGHT_BORDER.name()};
                    border-radius: 8px;
                    background-color: {Win11Colors.LIGHT_SURFACE.name()};
                    color: {Win11Colors.LIGHT_TEXT_SECONDARY.name()};
                    font-size: 16px;
                }}
            """

    def _load_species_data(self):
        """拦截外部直接调用的加载方法，强制使用带有状态恢复的刷新逻辑，以在切换界面时保持选中状态"""
        self._refresh_species_list_logic()

    def _load_species_data_core(self):
        """核心加载物种数据逻辑（从 SQLite 读取，单次查询替代 N 次 JSON 读取）"""
        from system.detection_db import (
            get_db_path, init_db, migrate_from_json,
            get_all_detections_with_validation
        )

        photo_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.get_file_path()

        if not photo_dir or not os.path.exists(photo_dir) or not source_dir:
            self.species_listbox.clear()
            self.species_image_map.clear()
            return

        self.species_listbox.blockSignals(True)
        self.species_listbox.clear()
        self.species_listbox.blockSignals(False)
        self.species_image_map.clear()

        if hasattr(self, 'species_group_map'):
            self.species_group_map.clear()
        else:
            self.species_group_map = defaultdict(list)

        confidence_settings = self.controller.confidence_settings
        global_conf = confidence_settings.get("global", 0.25)

        # ── 获取源文件映射 ────────────────────────────────────────────
        try:
            source_files = [
                f for f in os.listdir(source_dir)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS) or
                   f.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)
            ]
        except Exception as e:
            logger.error(f"读取源目录失败: {e}")
            return

        image_basename_map = {os.path.splitext(f)[0]: f for f in source_files}

        # ── 初始化 / 选择优先加载的 SQLite ─────────────────────────────
        save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                 False)
        source_db_path = get_db_path(source_dir)
        temp_db_path = get_db_path(photo_dir)

        target_db_path = None

        if save_to_source and os.path.exists(source_db_path):
            # 优先从源目录读取
            target_db_path = source_db_path
        else:
            # 否则从 temp_dir 读取，并执行必要的初始化或同步
            target_db_path = temp_db_path
            if not os.path.exists(temp_db_path):
                init_db(temp_db_path)
                migrate_from_json(temp_db_path, photo_dir, image_basename_map)

                if save_to_source and os.path.exists(source_db_path):
                    try:
                        from system.detection_db import get_all_detections_with_validation, upsert_detection
                        img_rows = get_all_detections_with_validation(source_db_path)
                        for r in img_rows:
                            import json as _json
                            upsert_detection(temp_db_path, r["base_name"], r["image_filename"],
                                             _json.loads(r["detection_json"]))
                        logger.info(f"已从图像文件夹 db 恢复 {len(img_rows)} 条记录到软件缓存")
                    except Exception as e:
                        logger.warning(f"从图像文件夹 db 恢复失败: {e}")
            else:
                try:
                    json_files_on_disk = {
                        os.path.splitext(f)[0]
                        for f in os.listdir(photo_dir)
                        if f.lower().endswith('.json') and f != 'validation.json'
                    }
                    import sqlite3 as _sq
                    with _sq.connect(temp_db_path) as _c:
                        in_db = {r[0] for r in _c.execute("SELECT base_name FROM detections")}
                    if json_files_on_disk - in_db:
                        migrate_from_json(temp_db_path, photo_dir, image_basename_map)
                except Exception as e:
                    logger.warning(f"增量迁移检查失败: {e}")

        # ── 核心：单次查询取全量数据 ──────────────────────────────────
        try:
            rows = get_all_detections_with_validation(target_db_path)
        except Exception as e:
            logger.error(f"SQLite 读取失败: {e}")
            return

        # 同步内存 validation_data（保持与原逻辑兼容）
        for row in rows:
            if row["is_validated"] is not None:
                self.validation_data[row["image_filename"]] = bool(row["is_validated"])

        # ── 与原逻辑完全相同的物种归类计算 ───────────────────────────
        all_species_keys = set()
        processed_basenames = set()
        file_to_display_key = {}

        for row in rows:
            base_name = row["base_name"]
            image_filename = row["image_filename"]

            # 只处理在当前源目录中存在的文件
            if image_basename_map.get(base_name) != image_filename:
                continue

            processed_basenames.add(base_name)

            try:
                detection_info = json.loads(row["detection_json"])
            except Exception:
                continue

            is_validated = row["is_validated"] == 1
            if detection_info.get('最低置信度') == '人工校验':
                is_validated = True

            final_species_name = "[未校验] 空"

            if detection_info.get('最低置信度') == '人工校验':
                res_name = detection_info.get('物种名称', '')
                if res_name in ["[未校验] 空", "[已校验] 空", "", "未知", None]:
                    final_species_name = "[已校验] 空"
                else:
                    final_species_name = res_name

            elif 'tracks' in detection_info:
                tracks = detection_info.get('tracks', {})
                min_frame_ratio = 0.0
                if hasattr(self.controller, 'advanced_page'):
                    min_frame_ratio = self.controller.advanced_page.min_frame_ratio_var
                total_frames = detection_info.get('total_frames_processed', 1)
                threshold = total_frames * min_frame_ratio
                valid_votes = []
                for track_id, points in tracks.items():
                    if len(points) < threshold:
                        continue
                    valid_points = [
                        p['species'] for p in points
                        if p.get('confidence', 0) >= confidence_settings.get(
                            p.get('species', 'Unknown'), global_conf)
                    ]
                    if valid_points:
                        valid_votes.append(Counter(valid_points).most_common(1)[0][0])
                final_species_name = (
                    ",".join(sorted(set(valid_votes))) if valid_votes else "[未校验] 空"
                )

            else:
                boxes = detection_info.get('检测框', [])
                if not boxes:
                    boxes = detection_info.get('detect_results',
                                               detection_info.get('objects', []))
                valid_species_list = []
                is_ambiguous_image = False
                AMBIGUITY_THRESHOLD = 0.10
                for box in boxes:
                    candidates_pool = (
                        [{"name": c.get('name'), "conf": float(c.get('conf', 0))}
                         for c in box["候选项"]]
                        if "候选项" in box and box["候选项"]
                        else [{"name": box.get("物种", box.get("species", "未知")),
                               "conf": float(box.get("置信度", box.get("confidence", 0)))}]
                    )
                    valid_candidates = sorted(
                        [c for c in candidates_pool
                         if c['conf'] >= confidence_settings.get(c['name'], global_conf)],
                        key=lambda x: x['conf'], reverse=True
                    )
                    if not valid_candidates:
                        continue
                    if len(valid_candidates) >= 2 and (
                            valid_candidates[0]['conf'] - valid_candidates[1]['conf'] < AMBIGUITY_THRESHOLD
                    ):
                        is_ambiguous_image = True
                        break
                    valid_species_list.append(valid_candidates[0]['name'])

                if is_ambiguous_image:
                    final_species_name = "需人工检验"
                elif valid_species_list:
                    final_species_name = ",".join(sorted(set(valid_species_list)))
                else:
                    final_species_name = "[未校验] 空"

            # 归一化逗号
            if final_species_name not in ("需人工检验", "[已校验] 空", "未检测", "[未校验] 空"):
                normalized = final_species_name.replace('，', ',')
                if ',' in normalized:
                    parts = [p.strip() for p in normalized.split(',') if p.strip()]
                    if parts:
                        final_species_name = ",".join(sorted(parts))

            display_key = final_species_name
            if final_species_name == "[未校验] 空" and is_validated:
                display_key = "[已校验] 空"
            elif final_species_name not in ("[未校验] 空", "[已校验] 空", "未检测", "需人工检验"):
                display_key = f"[已校验] {final_species_name}" if is_validated \
                    else f"[未校验] {final_species_name}"

            file_to_display_key[image_filename] = display_key

        # 未检测（源目录中有文件但 DB 中无记录）
        undetected_basenames = set(image_basename_map.keys()) - processed_basenames
        if undetected_basenames:
            for base in undetected_basenames:
                self.species_image_map["未检测"].append(image_basename_map[base])
            all_species_keys.add("未检测")

        # ==================== 应用自动分组与归类逻辑 ====================
        auto_group = getattr(getattr(self.controller, 'advanced_page', None), 'auto_group_var', False)

        if auto_group:
            # 1. 字典序排列以还原相机的触发时间轴
            sorted_files = sorted(image_basename_map.values())
            groups = []
            current_group = []

            # 2. 模拟相机连拍分组（如最大3图+1视频为一个事件组）
            for idx, f in enumerate(sorted_files):
                current_group.append(f)
                is_video = f.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

                next_is_video = False
                if idx + 1 < len(sorted_files):
                    next_is_video = sorted_files[idx + 1].lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS)

                # 截断条件：当前是视频(一般连拍收尾)；或者已达到3图且下一张不是它配套的视频
                if is_video or (len(current_group) >= 3 and not next_is_video):
                    groups.append(current_group)
                    current_group = []
            if current_group:
                groups.append(current_group)

            # 3. 对每个组进行处理：合并物种，并且只有组内所有照片都校验才算该组已校验
            for grp in groups:
                keys_in_group = [file_to_display_key.get(f, "未检测") for f in grp]

                is_all_validated = True
                clean_valid_species = set()
                fallback_clean_keys = []

                for k in keys_in_group:
                    # 检查是否全部已校验 (只要有一个未校验或未检测，该组就保持未校验状态)
                    if not k.startswith("[已校验]"):
                        is_all_validated = False

                    # 剥离前缀，获取纯净的物种名称
                    clean_k = k.replace("[已校验] ", "").replace("[未校验] ", "")
                    fallback_clean_keys.append(clean_k)

                    if clean_k not in ["空", "未检测", "需人工检验"]:
                        # 处理单张照片中本身就有多个物种的情况
                        for sp in clean_k.replace('，', ',').split(','):
                            sp = sp.strip()
                            if sp:
                                clean_valid_species.add(sp)

                if clean_valid_species:
                    # 组合组内所有出现的有效物种（如 A,B）
                    combined_species = ",".join(sorted(list(clean_valid_species)))
                else:
                    # 如果全是背景/空/未检测，没有实体物种，走多数表决兜底
                    combined_species = Counter(fallback_clean_keys).most_common(1)[0][0]

                # 拼接最终的分组显示名称
                if combined_species in ["未检测", "需人工检验"]:
                    majority_key = combined_species
                elif combined_species == "空":
                    majority_key = "[已校验] 空" if is_all_validated else "[未校验] 空"
                else:
                    majority_key = f"[已校验] {combined_species}" if is_all_validated else f"[未校验] {combined_species}"

                all_species_keys.add(majority_key)

                if not hasattr(self, 'species_group_map'):
                    self.species_group_map = defaultdict(list)
                self.species_group_map[majority_key].append(grp)

                for f in grp:
                    self.species_image_map[majority_key].append(f)
        else:
            # 未开启自动分组，回退常规的单文件独立展示
            for f, key in file_to_display_key.items():
                all_species_keys.add(key)
                self.species_image_map[key].append(f)

        # ==================== 应用交替底色数据映射 ====================
        if not hasattr(self, 'species_group_map'):
            self.species_group_map = defaultdict(list)

        # 补全未走 auto_group 的情况 (或未检测的情况)
        for key in all_species_keys:
            if key not in self.species_group_map or not self.species_group_map[key]:
                sorted_files = sorted(set(self.species_image_map.get(key, [])))
                # 单张照片作为独立分组，实现单张交替颜色
                self.species_group_map[key] = [[f] for f in sorted_files]

        def sort_priority(name):
            if name == "需人工检验": return 0
            if name == "[未校验] 空": return 2
            if name.startswith("[未校验]"): return 1
            if name == "[已校验] 空": return 4
            if name.startswith("[已校验]"): return 3
            if name == "未检测": return 5
            return 6

        sorted_keys = sorted(all_species_keys, key=lambda x: (sort_priority(x), x))

        self.species_listbox.blockSignals(True)
        for key in sorted_keys:
            count = len(self.species_image_map[key])
            self.species_listbox.addItem(f"{key} ({count})")
        self.species_listbox.blockSignals(False)

    def _update_species_selector_items(self):
        """
        根据当前的检测结果更新下拉框内容。
        包含：候选项分析、视频轨迹投票、智能默认选中、人工校验优先
        """
        # 暂时阻断信号，防止清空时触发 change 事件
        self.species_selector.blockSignals(True)
        self.species_selector.clear()

        # 1. 恢复全局设置选项
        self.species_selector.addItem("全局设置 (Global)", "global")

        found_species = set()

        # 下拉框最小显示阈值
        MIN_DROPDOWN_CONF = 0.05

        # 始终添加当前左侧列表选中的分类
        # 如果是组合分类（如 "赤狐,狍子"），则拆分后添加各个单物种
        if self.current_selected_species and self.current_selected_species not in ["[已校验] 空", "[未校验] 空"]:
            if "," in self.current_selected_species:
                # 拆分并添加
                parts = self.current_selected_species.split(",")
                for p in parts:
                    found_species.add(p.strip())
            else:
                found_species.add(self.current_selected_species)

        # === 变量定义 ===
        # A. 有效最高置信度 (满足阈值)
        best_valid_species_name = None
        max_valid_confidence = -1.0

        # B. 绝对最高置信度 (无视阈值，作为兜底)
        best_absolute_species_name = None
        max_absolute_confidence = -1.0

        # 获取配置引用
        conf_settings = {}
        if hasattr(self.controller, 'confidence_settings'):
            conf_settings = self.controller.confidence_settings

        global_thresh = conf_settings.get("global", 0.25)

        # 【新增】标识是否为人工校验以及其校验的具体名称
        is_manual_validated = False
        manual_species_names = []

        # 从当前 JSON 数据中提取所有物种
        if self.current_species_info:
            # 【新增】检查是否为人工校验，将其物种名也加入 found_species 保证下拉框里有它
            if self.current_species_info.get('最低置信度') == '人工校验':
                is_manual_validated = True
                raw_manual_name = self.current_species_info.get('物种名称', '')
                if raw_manual_name and raw_manual_name not in ["空", "未知"]:
                    manual_species_names = [p.strip() for p in raw_manual_name.replace('，', ',').split(',') if p.strip()]
                    for sp in manual_species_names:
                        found_species.add(sp)

            # --- 情况 A: 处理图片 JSON 结构 ---
            boxes = self.current_species_info.get("检测框", [])
            if not boxes:
                boxes = self.current_species_info.get("detect_results", self.current_species_info.get("objects", []))

            for box in boxes:
                # 1. 获取原始信息
                raw_name = box.get("物种", box.get("species", box.get("class_name")))
                raw_conf = float(box.get("置信度", box.get("confidence", 0.0)))

                if not raw_name: continue

                # 默认情况下，最终显示的物种和置信度就是原始的
                final_name = raw_name
                final_conf = raw_conf

                # 2. 处理候选项逻辑
                is_candidate_match = False
                if "候选项" in box and box["候选项"]:
                    candidates = box["候选项"]
                    for cand in candidates:
                        c_name = cand.get('name')
                        c_conf = float(cand.get('conf', 0.0))
                        c_thresh = conf_settings.get(c_name, global_thresh)

                        if c_conf >= c_thresh:
                            final_name = c_name
                            final_conf = c_conf
                            is_candidate_match = True
                            break

                # 3. 将所有出现过的名字加入下拉列表 (增加 0.05 过滤)
                if final_conf >= MIN_DROPDOWN_CONF:
                    found_species.add(final_name)

                if "候选项" in box:
                    for c in box["候选项"]:
                        if c.get('name') and float(c.get('conf', 0)) >= MIN_DROPDOWN_CONF:
                            found_species.add(c['name'])

                # === 4. 更新绝对最大值 (兜底用) ===
                if final_conf > max_absolute_confidence:
                    max_absolute_confidence = final_conf
                    best_absolute_species_name = final_name

                # === 5. 更新有效最大值 (优先用) ===
                is_valid = False
                if is_candidate_match:
                    is_valid = True
                else:
                    thresh = conf_settings.get(final_name, global_thresh)
                    if final_conf >= thresh:
                        is_valid = True

                if is_valid:
                    if final_conf > max_valid_confidence:
                        max_valid_confidence = final_conf
                        best_valid_species_name = final_name

            # --- 情况 B: 处理视频 JSON 结构 (tracks) ---
            tracks = self.current_species_info.get("tracks", {})
            if tracks:
                min_ratio = 0.0
                if hasattr(self.controller, 'advanced_page'):
                    min_ratio = self.controller.advanced_page.min_frame_ratio_var

                total_frames = self.current_species_info.get('total_frames_processed', 1)
                threshold_frames = total_frames * min_ratio

                for t_list in tracks.values():
                    if len(t_list) < threshold_frames:
                        continue

                    s_list = [p.get('species') for p in t_list if p.get('species')]
                    if not s_list: continue
                    dominant_species = Counter(s_list).most_common(1)[0][0]

                    track_max_conf = max([float(p.get('confidence', 0.0)) for p in t_list])

                    # 增加 0.05 过滤
                    if track_max_conf >= MIN_DROPDOWN_CONF:
                        found_species.add(dominant_species)

                    if track_max_conf > max_absolute_confidence:
                        max_absolute_confidence = track_max_conf
                        best_absolute_species_name = dominant_species

                    sp_thresh = conf_settings.get(dominant_species, global_thresh)
                    if track_max_conf >= sp_thresh:
                        if track_max_conf > max_valid_confidence:
                            max_valid_confidence = track_max_conf
                            best_valid_species_name = dominant_species

        # 将发现的物种添加到下拉框
        for sp in sorted(list(found_species)):
            if self.species_selector.findText(sp) == -1:
                self.species_selector.addItem(sp, sp)

        self.species_selector.blockSignals(False)

        # === 核心逻辑：确定最终选中的目标 ===
        target_species_name = None

        # 策略0：优先保持刷新前的选择（防止调整置信度时跳变）
        preserved = getattr(self, '_preserve_dropdown_selection', None)
        if preserved and self.species_selector.findData(preserved) != -1:
            target_species_name = preserved

        if not target_species_name:
            # 【新增】策略1：如果是人工校验的照片，优先选中人工校验出来的物种（多物种时选第一个有效的）
            if is_manual_validated and manual_species_names:
                for sp in manual_species_names:
                    if self.species_selector.findData(sp) != -1:
                        target_species_name = sp
                        break

        if not target_species_name:
            # 策略2：优先选择“有效且置信度最高”的物种（AI原判断）
            if best_valid_species_name:
                target_species_name = best_valid_species_name
            # 策略3：如果所有物种都被过滤了，选择“绝对置信度最高”的物种
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
        current_species = self.species_selector.currentData()
        if not current_species:
            current_species = "global"

        # 获取保存的设置
        conf_settings = {}
        if hasattr(self.controller, 'confidence_settings'):
            conf_settings = self.controller.confidence_settings

        # 获取值：特定物种 -> 全局 -> 默认0.25
        saved_val = conf_settings.get(current_species, conf_settings.get("global", 0.25))

        # 更新滑块（阻断信号防止循环调用）
        self.species_conf_slider.blockSignals(True)
        self.species_conf_slider.setValue(int(saved_val * 100))
        self.species_conf_slider.blockSignals(False)

        # 更新标签和内部变量
        self.species_conf_label.setText(f"{saved_val:.2f}")
        # 更新内部变量
        self.species_conf_var = saved_val
        # 顺便刷新一下检测信息显示 (确保界面上的数量统计与新的置信度一致)
        self._update_detection_info_display()

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
        self.species_info_label.setText("物种:  | 数量:  | 类型:  | 置信度: ")

        # 解析选中的物种名称
        selected_text = selected_items[0].text()
        map_key = selected_text.split(' (')[0] if ' (' in selected_text else selected_text

        # 剥离前缀，还原真实的物种名称供逻辑使用
        species_name = map_key.replace('[已校验] ', '').replace('[未校验] ', '')

        self.current_selected_species = species_name
        image_files = sorted(self.species_image_map.get(map_key, []))

        photo_count = len(image_files)
        if hasattr(self.controller, 'status_bar'):
            self.controller.status_bar.status_label.setText(f"当前物种共有 {photo_count} 张照片")

        # 根据选择的物种来决定是否显示置信度滑块
        if species_name in ["[已校验] 空", "[未校验] 空", "空"]:
            self.species_conf_slider.setEnabled(False)
            self.species_conf_label.setText("N/A")
            self.species_selector.setEnabled(False)  # 禁用选择器
        else:
            self.species_conf_slider.setEnabled(True)
            self.species_selector.setEnabled(True)
            self._update_species_selector_items()

        # 获取当前主题，定义交替的浅灰色 (提高透明度数值使其清晰可见)
        from PySide6.QtGui import QBrush
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        # 调高 Alpha 值：深色模式 40，浅色模式 25
        alt_bg_color = QColor(255, 255, 255, 40) if is_dark else QColor(0, 0, 0, 25)
        alt_brush = QBrush(alt_bg_color)

        # 添加图片到列表（按分组交替底色）
        groups = getattr(self, 'species_group_map', {}).get(map_key, [])
        if groups:
            for i, grp in enumerate(groups):
                for image_file in sorted(grp):
                    self.species_photo_listbox.addItem(image_file)
                    # 为奇数组添加底色
                    if i % 2 == 1:
                        item = self.species_photo_listbox.item(self.species_photo_listbox.count() - 1)
                        item.setBackground(alt_brush)
                    logger.debug(f"添加图片到列表: {image_file}")
        else:
            # 兼容 fallback 情况
            for i, image_file in enumerate(image_files):
                self.species_photo_listbox.addItem(image_file)
                if i % 2 == 1:
                    item = self.species_photo_listbox.item(self.species_photo_listbox.count() - 1)
                    item.setBackground(alt_brush)
                logger.debug(f"添加图片到列表: {image_file}")

        # 如果照片列表不为空，则自动选择第一个
        if self.species_photo_listbox.count() > 0:
            self.species_photo_listbox.setCurrentRow(0)
            logger.info(f"物种 {species_name} 已选择，共 {photo_count} 张照片")
        else:
            logger.warning(f"物种 {species_name} 没有找到对应的照片")

    def _on_species_photo_selected(self):
        """当选择物种照片/视频时的处理"""
        # 1. 检查是否选择了同一个正在播放的视频
        selection = self.species_photo_listbox.selectedItems()
        if selection:
            file_name = selection[0].text()
            # 检查是否是同一个文件，且是视频，且线程正在运行
            if (hasattr(self, 'last_selected_species_image') and
                    self.last_selected_species_image == file_name and
                    file_name.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS) and
                    self.video_thread and self.video_thread.isRunning()):

                # 如果是同一个视频，只更新信息和下拉框，不重启播放
                if self.current_selected_species not in ["[已校验] 空", "[未校验] 空"]:
                    self._update_species_selector_items()
                return

        # 2. 原有清理工作
        self._stop_video_thread()  # 停止之前的视频线程

        # 判断是否是由撤回操作引发的刷新
        if getattr(self, '_is_undoing', False):
            self._is_undoing = False
            self._restore_button_styles()  # 恢复撤回栈中保留的半截按钮样式
        else:
            self._reset_quantity_buttons()  # 封装了重置按钮样式的逻辑
            self._reset_species_buttons()  # 重置物种按钮样式

        self.current_species_info = {}  # 重置当前信息

        # 切换文件时，必须强制清空上一张图片的缓存对象
        self.species_validation_original_image = None

        if not selection:
            self.species_info_label.setText("物种: - | 数量: - | 置信度: -")
            return

        file_name = selection[0].text()
        self.last_selected_species_image = file_name

        # 路径准备
        source_dir = self.controller.start_page.get_file_path()
        media_path = os.path.join(source_dir, file_name)

        # 使用SQLite 加载
        photo_dir = self.controller.get_temp_photo_dir()
        json_path = None
        if photo_dir:
            base_name = os.path.splitext(file_name)[0]
            json_path = os.path.join(photo_dir, f"{base_name}.json")

            loaded = False
            save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                     False)

            # 优先从各级数据库加载
            try:
                from system.detection_db import get_db_path, get_detection
                source_db_path = get_db_path(source_dir)
                temp_db_path = get_db_path(photo_dir)

                # 1. 优先从图像源目录 SQLite 加载
                if save_to_source and os.path.exists(source_db_path):
                    data = get_detection(source_db_path, base_name)
                    if data:
                        self.current_species_info = data
                        self._update_detection_info_display()
                        loaded = True

                # 2. 回退到软件缓存目录 SQLite 加载
                if not loaded and os.path.exists(temp_db_path):
                    data = get_detection(temp_db_path, base_name)
                    if data:
                        self.current_species_info = data
                        self._update_detection_info_display()
                        loaded = True
            except Exception as e:
                logger.warning(f"从 SQLite 加载检测信息失败: {e}")

            # 3. 回退到旧 JSON（仅当 SQLite 无记录时）
            if not loaded and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        self.current_species_info = json.load(f)
                    self._update_detection_info_display()
                    loaded = True
                except Exception as e:
                    logger.warning(f"从 JSON 加载检测信息失败: {e}")

            if not loaded:
                self.species_info_label.setText("物种:  | 数量:  | 类型:  | 置信度: ")

        # 加载图片信息后，刷新下拉框
        if self.current_selected_species not in ["[已校验] 空", "[未校验] 空"]:
            self._update_species_selector_items()

        # === 2. 判断文件类型 ===
        if file_name.lower().endswith(self.SUPPORTED_VIDEO_EXTENSIONS):
            if not os.path.exists(media_path):
                self.species_image_label.setText("视频文件不存在")
                return
            self.species_image_label.setText("正在加载视频...")
            self._start_video_thread(media_path)
        else:
            try:
                self.species_validation_original_image = Image.open(media_path)
                self._display_image_with_detection_boxes(
                    self.species_validation_original_image,
                    self.current_species_info,
                    self.species_conf_var
                )
            except Exception as e:
                logger.error(f"加载图像失败: {e}")
                self.species_image_label.setText("无法加载图像")

    def _reset_quantity_buttons(self):
        """重置数量按钮样式的辅助函数(支持多选)"""
        if hasattr(self, '_selected_quantity_buttons'):
            for btn in self._selected_quantity_buttons:
                try:
                    btn.setStyleSheet("""
                        QPushButton {
                            max-width: 58px;
                            min-width: 45px;
                            min-height: 25px;
                            max-height: 25px;
                            padding: 4px;
                            font-size: 14px;
                            font-weight: 600;
                            text-align: center;
                            border-radius: 12px;
                        }
                    """)
                except RuntimeError:
                    pass
        self._selected_quantity_buttons = []
        self._selected_counts = []

    def _reset_species_buttons(self):
        """重置快速标记物种按钮样式的辅助函数(支持多选)"""
        if hasattr(self, '_selected_species_buttons'):
            for btn in self._selected_species_buttons:
                try:
                    padding = btn.property("base_padding") or "2px 8px"
                    font_size = btn.property("base_font_size") or "13px"
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            max-width: 80px;
                            min-width: 60px;
                            min-height: 30px;
                            max-height: 30px;
                            padding: {padding};
                            font-size: {font_size};
                            font-weight: 600;
                            text-align: center;
                            border-radius: 12px;
                        }}
                    """)
                except RuntimeError:
                    pass
        self._selected_species_buttons = []
        self._selected_species_names = []

    def _restore_button_styles(self):
        """根据恢复的数组重新高亮物种和数量按钮"""
        self._selected_species_buttons = []
        if hasattr(self, 'species_buttons_layout'):
            for i in range(self.species_buttons_layout.count()):
                btn = self.species_buttons_layout.itemAt(i).widget()
                if isinstance(btn, QPushButton):
                    padding = btn.property("base_padding") or "2px 8px"
                    font_size = btn.property("base_font_size") or "13px"

                    # 恢复普通样式
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            max-width: 80px; min-width: 60px; min-height: 30px; max-height: 30px;
                            padding: {padding}; font-size: {font_size}; font-weight: 600;
                            text-align: center; border-radius: 12px;
                        }}
                    """)

                    # 如果在恢复的列表中，重新高亮
                    if btn.text() in getattr(self, '_selected_species_names', []):
                        self._selected_species_buttons.append(btn)
                        btn.setStyleSheet(f"""
                            QPushButton {{
                                max-width: 80px; min-width: 60px; min-height: 30px; max-height: 30px;
                                padding: {padding}; font-size: {font_size}; font-weight: bold;
                                text-align: center; border-radius: 12px;
                                background-color: #5d3a4f; color: white;
                            }}
                        """)

        self._selected_quantity_buttons = []
        if hasattr(self, 'qty_scroll_widget'):
            for btn in self.qty_scroll_widget.findChildren(QPushButton):
                if btn.text() == "更多":
                    btn.setStyleSheet("""
                        QPushButton {
                            max-width: 58px; min-width: 45px; min-height: 25px; max-height: 25px;
                            padding: 4px; font-size: 13px; font-weight: 600; text-align: center; border-radius: 12px;
                        }
                    """)
                else:
                    # 恢复普通样式
                    btn.setStyleSheet("""
                        QPushButton {
                            max-width: 58px; min-width: 45px; min-height: 25px; max-height: 25px;
                            padding: 4px; font-size: 14px; font-weight: 600; text-align: center; border-radius: 12px;
                        }
                    """)
                    # 如果在恢复的列表中，重新高亮
                    if btn.text() in getattr(self, '_selected_counts', []):
                        self._selected_quantity_buttons.append(btn)
                        btn.setStyleSheet("""
                            QPushButton {
                                max-width: 58px; min-width: 45px; min-height: 25px; max-height: 25px;
                                padding: 4px; font-size: 14px; font-weight: bold; text-align: center; border-radius: 12px;
                                background-color: #5d3a4f; color: white;
                            }
                        """)

    def _run_background_update(self, tasks, on_finished_callback=None):
        """挂起界面，启动后台更新线程"""
        self.controller.status_bar.show_progress()
        self.controller.status_bar.status_label.setText(f"正在后台保存 {len(tasks)} 项修改，请稍候...")
        self.species_photo_listbox.setEnabled(False)
        self.species_listbox.setEnabled(False)

        temp_photo_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.get_file_path()
        save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                 False)

        # =======================================================
        # 【关键修复】：防止 QThread 对象被 Python 过早垃圾回收导致闪退
        if not hasattr(self, '_active_bg_threads'):
            self._active_bg_threads = []

        # 实例化后台更新线程，不再使用单一的 self.bg_update_thread 变量去覆盖
        thread = BackgroundUpdateThread(
            tasks, temp_photo_dir, source_dir, save_to_source, self.validation_data
        )
        # 将线程加入托管列表，保持强引用
        self._active_bg_threads.append(thread)

        # =======================================================

        def thread_finished(success):
            self.controller.status_bar.hide_progress()
            self.species_photo_listbox.setEnabled(True)
            self.species_listbox.setEnabled(True)
            if success:
                self.controller.status_bar.status_label.setText("✅ 修改已成功保存")
            else:
                self.controller.status_bar.status_label.setText("❌ 保存过程出现异常，请查看日志")

            if on_finished_callback:
                on_finished_callback()

            # =======================================================
            # 【关键修复】：线程安全完成后，移除强引用并让 C++ 底层安全延后销毁
            if thread in getattr(self, '_active_bg_threads', []):
                self._active_bg_threads.remove(thread)
            thread.deleteLater()
            # =======================================================

        thread.finished.connect(thread_finished)
        thread.start()

    def _do_auto_advance(self):
        """满足个数匹配条件后自动跳转的通用逻辑"""
        if hasattr(self.controller, 'advanced_page') and self.controller.advanced_page.auto_sort_switch_row.isChecked():
            self._load_species_buttons()
            self.quick_marks_updated.emit()

        self._move_to_next_image()
        self.species_photo_listbox.setFocus()
        QTimer.singleShot(50, self._refresh_species_list_logic)

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

                q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                self.species_image_label.setPixmap(pixmap)
                self.species_image_label.setScaledContents(False)
                self.species_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            logger.error(f"显示图像失败: {e}")
            self.species_image_label.setText("无法显示图像")

    def _mark_and_move_to_next(self, is_correct=None, species_name=None, count=None, btn_widget=None):
        """处理标记逻辑并根据条件跳转到下一张图片。(支持批量后台)"""
        selection = self.species_photo_listbox.selectedItems()
        if not selection:
            return

        tasks = []
        if is_correct is True:
            for item in selection:
                f_name = item.text()
                self._push_to_undo_stack(f_name)
                self.validation_data[f_name] = True
                tasks.append({'file_name': f_name, 'action': 'correct'})
            self._run_background_update(tasks, self._do_auto_advance)
            return

        if species_name == "空" and count == "空":
            for item in selection:
                f_name = item.text()
                self._push_to_undo_stack(f_name)
                self.validation_data[f_name] = False
                tasks.append({'file_name': f_name, 'action': 'empty', 'new_species': '空', 'new_count': '空'})
            self._run_background_update(tasks, self._do_auto_advance)
            return

        if species_name and btn_widget:
            # 切换按钮选中状态
            if btn_widget in self._selected_species_buttons:
                self._selected_species_buttons.remove(btn_widget)
                if species_name in self._selected_species_names:
                    self._selected_species_names.remove(species_name)
                padding = btn_widget.property("base_padding") or "2px 8px"
                font_size = btn_widget.property("base_font_size") or "13px"
                btn_widget.setStyleSheet(f"""
                    QPushButton {{ max-width: 80px; min-width: 60px; min-height: 30px; max-height: 30px; padding: {padding}; font-size: {font_size}; font-weight: 600; text-align: center; border-radius: 12px; }}
                """)
            else:
                self._selected_species_buttons.append(btn_widget)
                self._selected_species_names.append(species_name)
                padding = btn_widget.property("base_padding") or "2px 8px"
                font_size = btn_widget.property("base_font_size") or "13px"
                btn_widget.setStyleSheet(f"""
                    QPushButton {{ max-width: 80px; min-width: 60px; min-height: 30px; max-height: 30px; padding: {padding}; font-size: {font_size}; font-weight: bold; text-align: center; border-radius: 12px; background-color: #5d3a4f; color: white; }}
                """)

            self._increment_quick_mark_count(species_name)
            new_species_str = ",".join(self._selected_species_names) if self._selected_species_names else None

            new_count_str = None
            if self._selected_counts:
                new_count_str = ",".join(self._selected_counts)
            else:
                count_str = str(self.current_species_info.get('物种数量', '')).strip()
                if not count_str or count_str in ['空', '未知']:
                    info_text = self.species_info_label.text() if hasattr(self, 'species_info_label') else ""
                    for part in info_text.split(" | "):
                        part = part.strip()
                        if part.startswith("数量:"):
                            count_str = part.replace("数量:", "").strip()
                            break
                if count_str and count_str not in ['空', '未知', '-']:
                    try:
                        counts = [int(c.strip()) for c in count_str.replace('，', ',').split(',') if c.strip()]
                        if counts:
                            new_count_str = ",".join(map(str, counts))
                    except (ValueError, TypeError):
                        new_count_str = None

            for item in selection:
                f_name = item.text()
                self._push_to_undo_stack(f_name)
                self.validation_data[f_name] = False
                tasks.append({'file_name': f_name, 'action': 'update', 'new_species': new_species_str,
                              'new_count': new_count_str})

            if new_species_str is not None:
                self.current_species_info['物种名称'] = new_species_str
                self.current_species_info['最低置信度'] = '人工校验'
            if new_count_str is not None:
                self.current_species_info['物种数量'] = new_count_str

            def on_finished():
                if len(self._selected_species_names) > 0 and len(self._selected_species_names) == len(
                        self._selected_counts):
                    self._do_auto_advance()
                else:
                    self.species_photo_listbox.setFocus()
                    self._update_detection_info_display()

            self._run_background_update(tasks, on_finished)

    def _on_quantity_button_press(self, count, btn_widget):
        """处理数量按钮点击事件 (支持多选及后台打标)"""
        selection = self.species_photo_listbox.selectedItems()
        if not selection:
            return

        final_count = str(count)
        if count == "更多":
            dialog = QuantityInputDialog(self, title="输入数量", prompt="请输入物种的数量:", default_value=1)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                final_count = str(dialog.result_value)
            else:
                self.species_photo_listbox.setFocus()
                return

        if count != "更多":
            self._selected_quantity_buttons.append(btn_widget)
            btn_widget.setStyleSheet("""
                QPushButton { max-width: 58px; min-width: 45px; min-height: 25px; max-height: 25px; padding: 4px; font-size: 14px; font-weight: bold; text-align: center; border-radius: 12px; background-color: #5d3a4f; color: white; }
            """)

        self._selected_counts.append(final_count)

        new_count_str = ",".join(self._selected_counts)
        new_species_str = ",".join(self._selected_species_names) if self._selected_species_names else None

        if new_species_str is None:
            sp_name = str(self.current_species_info.get('物种名称', '')).strip()
            if not sp_name or sp_name in ['未知', '空', '[未校验] 空', '[已校验] 空']:
                info_text = self.species_info_label.text() if hasattr(self, 'species_info_label') else ""
                for part in info_text.split(" | "):
                    part = part.strip()
                    if part.startswith("物种:"):
                        sp_name = part.replace("物种:", "").strip()
                        break

            if sp_name.startswith("[未校验] "):
                sp_name = sp_name.replace("[未校验] ", "")
            elif sp_name.startswith("[已校验] "):
                sp_name = sp_name.replace("[已校验] ", "")

            if sp_name and sp_name not in ['空', '未知', '-', '未检测', '需人工检验']:
                new_species_str = sp_name

        tasks = []
        for item in selection:
            f_name = item.text()
            self._push_to_undo_stack(f_name)
            self.validation_data[f_name] = False
            tasks.append(
                {'file_name': f_name, 'action': 'update', 'new_species': new_species_str, 'new_count': new_count_str})

        if new_species_str is not None:
            self.current_species_info['物种名称'] = new_species_str
            self.current_species_info['最低置信度'] = '人工校验'
        if new_count_str is not None:
            self.current_species_info['物种数量'] = new_count_str

        def on_finished():
            if len(self._selected_counts) > 0 and len(self._selected_counts) == len(self._selected_species_names):
                self._do_auto_advance()
            else:
                self.species_photo_listbox.setFocus()
                self._update_detection_info_display()

        self._run_background_update(tasks, on_finished)

    def _mark_other_species(self):
        """处理"其他"按钮的逻辑（支持多选）"""
        selected_items = self.species_photo_listbox.selectedItems()
        if not selected_items:
            return
        # 直接路由给批量逻辑，行为一致
        self._bulk_mark_other(selected_items)

    def _load_species_buttons(self):
        """根据自动排序设置，动态加载快速标记物种按钮，确保近期频次排序不丢失"""
        self._selected_species_button = None

        self._selected_species_buttons = []
        self._selected_species_names = []

        while self.species_buttons_layout.count():
            child = self.species_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not hasattr(self.controller, 'settings_manager'):
            return

        quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
        use_auto_sort = False
        if hasattr(self.controller, 'advanced_page'):
            use_auto_sort = self.controller.advanced_page.auto_sort_switch_row.isChecked()

        if use_auto_sort:
            # 【核心修复】：动态实时计算近期频次排序，防止切换界面时被旧缓存的 list_auto 覆盖
            recent_history = quick_marks_data.get("recent_history", [])
            from collections import Counter

            # 统计近期频次 (反转列表以保证频次相同时，越晚使用的越靠前)
            recent_counter = Counter(reversed(recent_history))
            recent_sorted = [item[0] for item in recent_counter.most_common()]

            # 统计总数频次作兜底
            excluded_keys = {'list', 'list_auto', 'auto', 'recent_history'}
            total_counts = {k: v for k, v in quick_marks_data.items() if
                            k not in excluded_keys and isinstance(v, (int, float))}
            total_sorted = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
            total_sorted_species = [item[0] for item in total_sorted]

            # 获取固定列表的长度作为按钮总数参考（通常为 11 个）
            target_btn_count = len(quick_marks_data.get("list", []))
            if target_btn_count == 0:
                target_btn_count = 11

            # 组合最终列表：优先填充近期常用，空余部分用总数最高的补齐
            species_to_display = []
            for sp in recent_sorted:
                if len(species_to_display) < target_btn_count:
                    species_to_display.append(sp)

            for sp in total_sorted_species:
                if len(species_to_display) < target_btn_count and sp not in species_to_display:
                    species_to_display.append(sp)
        else:
            # 如果未开启自动排序，则使用固定的 list
            species_to_display = quick_marks_data.get("list", [])

        for species in species_to_display:
            btn = MarqueeButton(species)

            # 根据文字长度动态调整字体大小和水平边距，但高度统一锁定为25px
            text_length = len(species)
            if text_length <= 3:
                padding = "2px 8px"
                font_size = "13px"
            elif text_length <= 6:
                padding = "2px 6px"
                font_size = "12px"
            else:
                padding = "2px 4px"
                font_size = "11px"

            # 将计算好的样式保存到按钮的自定义属性中，方便恢复
            btn.setProperty("base_padding", padding)
            btn.setProperty("base_font_size", font_size)

            btn.setStyleSheet(f"""
                        QPushButton {{
                            max-width: 80px;
                            min-width: 60px;
                            min-height: 30px;
                            max-height: 30px;
                            padding: {padding};
                            font-size: {font_size};
                            font-weight: 600;
                            text-align: center;
                            border-radius: 12px;
                        }}
                    """)

            # 将按钮对象 b=btn 传入 lambda 表达式
            btn.clicked.connect(
                lambda checked=False, s=species, b=btn: self._mark_and_move_to_next(species_name=s, btn_widget=b))
            self.species_buttons_layout.addWidget(btn)

    def _on_confidence_slider_changed(self, value):
        """处理置信度滑块值的变化"""
        self._update_confidence_label(value)

        # 转换值为小数
        new_conf = value / 100.0 if isinstance(value, int) else value

        # 获取当前针对的物种Key
        current_species_key = self.species_selector.currentData()
        if not current_species_key:
            current_species_key = "global"

        # 1. 实时更新当前视图（保持流畅）
        if hasattr(self, 'current_species_info') and self.current_species_info:
            self._update_detection_info_display()

        if (hasattr(self, 'species_validation_original_image') and
                self.species_validation_original_image and
                hasattr(self, 'current_species_info') and
                self.current_species_info):
            self._display_image_with_detection_boxes(
                self.species_validation_original_image,
                self.current_species_info,
                new_conf
            )

        # 实时更新视频阈值并请求刷新
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.conf_threshold = new_conf
            # 如果视频暂停，请求刷新当前帧以显示新的框
            if self.video_thread.paused:
                self.video_thread.refresh_frame()

        # 2. 保存设置
        if self.current_selected_species in ["[已校验] 空", "[未校验] 空"]:
            return

        if hasattr(self.controller, 'confidence_settings'):
            self.controller.confidence_settings[current_species_key] = new_conf

        # 3. 启动/重置定时器，准备刷新左侧列表
        self._list_refresh_timer.start()

    def _export_validation_data(self):
        """配置并启动后台线程进行数据导出"""
        temp_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.get_file_path()

        if not temp_dir or not os.path.exists(temp_dir) or not source_dir:
            MaterialMessageBox.critical(self, "错误", "无法找到临时文件或源文件路径，请确保已进行批处理并且路径设置正确。")
            return

        json_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.json') and f != 'validation.json']
        if not json_files:
            MaterialMessageBox.information(self, "提示", "没有找到任何处理后的数据，无法导出。")
            return

        file_format = self.export_format_var.lower()
        if file_format == 'excel':
            file_types = "Excel 文件 (*.xlsx);;所有文件 (*.*)"
            file_extension = ".xlsx"
        elif file_format == 'csv':
            file_types = "CSV 文件 (*.csv);;所有文件 (*.*)"
            file_extension = ".csv"
        else:
            return

        folder_name = os.path.basename(os.path.normpath(source_dir))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"{folder_name}_validation_data_{timestamp}{file_extension}"
        default_save_path = os.path.join(source_dir, default_filename)

        output_path, _ = QFileDialog.getSaveFileName(self, "选择表格保存位置", default_save_path, file_types)
        if not output_path:
            return

        # 准备导出所需的所有参数
        confidence_settings = self.controller.settings_manager.load_confidence_settings() or {}
        min_frame_ratio = self.controller.advanced_page.min_frame_ratio_var if hasattr(self.controller, 'advanced_page') else 0.0
        columns_to_export = self.controller.advanced_page.get_selected_export_columns() if hasattr(self.controller, 'advanced_page') else None

        # 获取设置状态
        save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                 False)

        # 锁定 UI 防止重复点击
        self.export_button.setEnabled(False)
        self.format_combo.setEnabled(False)
        self.controller.status_bar.show_progress()
        self.controller.status_bar.status_label.setText(f"正在读取文件并导出 {file_format.upper()}，请稍候...")

        # 创建并启动后台导出线程 (增加 save_to_source 参数)
        self.export_thread = ValidationExportThread(
            temp_dir=temp_dir,
            source_dir=source_dir,
            output_path=output_path,
            file_format=file_format,
            confidence_settings=confidence_settings,
            columns_to_export=columns_to_export,
            min_frame_ratio=min_frame_ratio,
            supported_video_exts=getattr(self, 'SUPPORTED_VIDEO_EXTENSIONS', ()),
            save_to_source=save_to_source
        )
        self.export_thread.progress_updated.connect(self.controller.status_bar.update_progress)
        self.export_thread.finished.connect(self._on_export_finished)
        self.export_thread.start()

    def _on_export_finished(self, success, result_message_or_path):
        """导出线程结束的回调"""
        # 恢复 UI 和进度条
        self.controller.status_bar.hide_progress()
        self.export_button.setEnabled(True)
        self.format_combo.setEnabled(True)

        if success:
            self.controller.status_bar.status_label.setText("✅ 数据导出成功")
            reply = MaterialMessageBox.question(self, "成功", f"数据已成功导出到:\n{result_message_or_path}\n\n是否立即打开文件？",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.startfile(result_message_or_path)
                except Exception as e:
                    MaterialMessageBox.critical(self, "错误", f"无法打开文件: {e}")
        else:
            self.controller.status_bar.status_label.setText("❌ 数据导出失败")
            MaterialMessageBox.critical(self, "导出失败", result_message_or_path)

    def _export_error_images(self):
        """导出被标记为错误的照片"""
        try:
            source_dir = self.controller.start_page.get_file_path()
            if not source_dir or not os.path.exists(source_dir):
                MaterialMessageBox.warning(self, "错误", "源图像目录未设置或不存在。")
                return

            temp_photo_dir = self.controller.get_temp_photo_dir()
            if not temp_photo_dir or not os.path.exists(temp_photo_dir):
                MaterialMessageBox.warning(self, "错误", "临时目录不存在，无法获取检测数据。")
                return

            # 获取目标目录
            dest_dir = self.controller.start_page.get_save_path()
            if not dest_dir or not os.path.isdir(dest_dir):
                dest_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
                if not dest_dir:
                    return  # 用户取消

            error_dir = os.path.join(dest_dir, "error")
            os.makedirs(error_dir, exist_ok=True)

            copied_count = 0

            # --- 关键修改：优先从 SQLite 获取数据，回退读取 JSON ---
            from system.detection_db import get_db_path, get_all_detections_with_validation

            save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                     False)
            source_db_path = get_db_path(source_dir)
            temp_db_path = get_db_path(temp_photo_dir)

            rows = []
            # 1. 源目录数据库
            if save_to_source and os.path.exists(source_db_path):
                try:
                    rows = get_all_detections_with_validation(source_db_path)
                except Exception as e:
                    logger.warning(f"读取源目录 SQLite 失败: {e}")

            # 2. 临时目录数据库
            if not rows and os.path.exists(temp_db_path):
                try:
                    rows = get_all_detections_with_validation(temp_db_path)
                except Exception as e:
                    logger.warning(f"读取缓存 SQLite 失败: {e}")

            # 3. JSON 兼容
            if not rows:
                json_files = [f for f in os.listdir(temp_photo_dir) if
                              f.lower().endswith('.json') and f not in ('validation.json', 'conf.json')]
                for jf in json_files:
                    base_name = os.path.splitext(jf)[0]
                    rows.append({
                        'base_name': base_name,
                        'image_filename': base_name,
                        'json_path': os.path.join(temp_photo_dir, jf)
                    })

            if rows:
                for row in rows:
                    try:
                        # 兼容处理
                        if 'detection_json' in row:
                            data = json.loads(row['detection_json'])
                        else:
                            with open(row['json_path'], 'r', encoding='utf-8') as f:
                                data = json.load(f)

                        # 判断是否为人工校验（错检数据）
                        if data.get("最低置信度") == "人工校验":
                            species_name = data.get("物种名称", "unknown")
                            if species_name == "[未校验] 空":
                                species_name = "[未校验] 空"

                            species_dir = os.path.join(error_dir, species_name)
                            os.makedirs(species_dir, exist_ok=True)

                            image_filename = row['image_filename']
                            source_path = os.path.join(source_dir, image_filename)

                            # 复制物理文件
                            if os.path.exists(source_path):
                                dest_path = os.path.join(species_dir, image_filename)
                                shutil.copy2(source_path, dest_path)
                                copied_count += 1
                            else:
                                # 容错逻辑：尝试其他扩展名
                                image_filename_base = row['base_name']
                                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                                    fallback_filename = image_filename_base + ext
                                    fallback_path = os.path.join(source_dir, fallback_filename)
                                    if os.path.exists(fallback_path):
                                        dest_path = os.path.join(species_dir, fallback_filename)
                                        shutil.copy2(fallback_path, dest_path)
                                        copied_count += 1
                                        break
                    except Exception as e:
                        logger.error(f"处理数据记录 {row.get('base_name')} 时出错: {e}")
            else:
                logger.warning("未找到有效的数据源（数据库与JSON均为空），无法导出错误照片。")

            if copied_count > 0:
                MaterialMessageBox.information(self, "成功",
                                               f"已成功导出 {copied_count} 张错误照片到:\n{error_dir}")
            else:
                MaterialMessageBox.information(self, "提示", "没有错误照片可供导出。")

        except Exception as e:
            logger.error(f"导出错误照片失败: {e}")
            MaterialMessageBox.critical(self, "导出失败", f"导出错误照片时发生错误: {e}")

    def _dispatch_export(self):
        """根据下拉框的选择来分派导出任务"""
        export_type = self.export_format_var
        if export_type == "错误照片":
            self._export_error_images()
        elif export_type in ["Excel", "CSV"]:
            self._export_validation_data()
        else:
            MaterialMessageBox.warning(self, "错误", f"未知的导出格式: {export_type}")

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

        # 遍历并更新所有自定义子组件的主题（如分组框、下拉框、滑块等）
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'update_theme') and callable(widget.update_theme):
                widget.update_theme()
            # 兼容只有 _setup_style 方法的组件
            elif hasattr(widget, '_setup_style') and callable(widget._setup_style):
                widget._setup_style()

    def _mark_as_error_and_save(self, file_name):
        """标记文件为错误并保存验证数据"""
        if not hasattr(self, 'validation_data'):
            self.validation_data = {}

        self.validation_data[file_name] = False  # 标记为错误
        self._save_validation_data()

    def _save_validation_data(self):
        """保存校验状态到 SQLite（如果设置开启，同时同步到图像文件夹的 SQLite）"""
        from system.detection_db import get_db_path, init_db, upsert_validation, delete_validation

        try:
            temp_photo_dir = self.controller.get_temp_photo_dir()
            source_dir = self.controller.start_page.get_file_path()
            if not temp_photo_dir:
                return

            db_path = get_db_path(temp_photo_dir)
            if not os.path.exists(db_path):
                init_db(db_path)

            # 判断是否需要同步到源文件夹
            save_to_source = getattr(
                getattr(self.controller, 'advanced_page', None),
                'save_cache_to_image_folder_var', False
            )
            source_db_path = None
            if save_to_source and source_dir and os.path.isdir(source_dir):
                source_db_path = get_db_path(source_dir)
                if not os.path.exists(source_db_path):
                    init_db(source_db_path)

            # 将内存中的最新状态批量写入 SQLite (双写逻辑)
            for filename, value in self.validation_data.items():
                if value is None:
                    delete_validation(db_path, filename)
                    if source_db_path:
                        delete_validation(source_db_path, filename)
                else:
                    upsert_validation(db_path, filename, bool(value))
                    if source_db_path:
                        upsert_validation(source_db_path, filename, bool(value))

            # 同时保留 validation.json（兼容旧版本读取）
            validation_file_path = os.path.join(temp_photo_dir, "validation.json")
            with open(validation_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_data, f, ensure_ascii=False,
                          indent=2, sort_keys=True)

        except Exception as e:
            logger.error(f"保存验证数据失败: {e}")

    def _increment_quick_mark_count(self, species_name):
        """增加快速标记物种的使用次数，并更新历史记录"""
        if hasattr(self.controller, 'settings_manager'):
            quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
            # 1. 确保中文逗号被替换为英文逗号
            normalized_name = species_name.replace('，', ',')

            # 2. 按逗号拆分并去除首尾空格
            species_list = [s.strip() for s in normalized_name.split(',') if s.strip()]

            if "recent_history" not in quick_marks_data:
                quick_marks_data["recent_history"] = []

            # 3. 对拆分后的每个独立物种分别计数
            for single_species in species_list:
                # 更新总计数
                if single_species in quick_marks_data:
                    quick_marks_data[single_species] = quick_marks_data.get(single_species, 0) + 1
                else:
                    quick_marks_data[single_species] = 1

                # 追加到近期记录队列
                quick_marks_data["recent_history"].append(single_species)

            # 保持近期记录最大长度为 200
            max_recent_len = 200
            if len(quick_marks_data["recent_history"]) > max_recent_len:
                quick_marks_data["recent_history"] = quick_marks_data["recent_history"][-max_recent_len:]

            # 保存更新后的历史记录数据
            self.controller.settings_manager.save_quick_mark_species(quick_marks_data)

    def _push_to_undo_stack(self, file_name):
        """将当前文件的状态压入撤回栈"""
        if not hasattr(self, 'undo_stack'):
            self.undo_stack = []

        try:
            temp_photo_dir = self.controller.get_temp_photo_dir()
            if not temp_photo_dir: return

            json_path = os.path.join(temp_photo_dir, f"{os.path.splitext(file_name)[0]}.json")
            old_json = None
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    old_json = json.load(f)

            old_val = self.validation_data.get(file_name)

            self.undo_stack.append({
                'file_name': file_name,
                'json_data': old_json,
                'validation_data': old_val,
                'selected_species_names': list(getattr(self, '_selected_species_names', [])),
                'selected_counts': list(getattr(self, '_selected_counts', []))
            })

            # 限制栈大小为 50 步，防止占用过多内存
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)
        except Exception as e:
            logger.error(f"加入撤回栈失败: {e}")

    def _undo_last_action(self):
        """执行撤回操作并自动刷新界面，并在状态栏提示"""
        if not hasattr(self, 'undo_stack') or not self.undo_stack:
            MaterialMessageBox.information(self, "提示", "没有可撤回的操作记录。")
            return

        try:
            last_state = self.undo_stack.pop()
            file_name = last_state['file_name']
            old_json = last_state['json_data']
            old_val = last_state['validation_data']

            # [新增] 恢复按钮数组，并打上撤回标志位以便恢复高亮样式
            self._selected_species_names = last_state.get('selected_species_names', [])
            self._selected_counts = last_state.get('selected_counts', [])
            self._is_undoing = True

            temp_photo_dir = self.controller.get_temp_photo_dir()
            if not temp_photo_dir: return
            json_path = os.path.join(temp_photo_dir, f"{os.path.splitext(file_name)[0]}.json")

            # 还原 JSON
            if old_json is not None:
                try:
                    from system.detection_db import get_db_path, update_detection
                    base_name = os.path.splitext(file_name)[0]
                    update_detection(get_db_path(temp_photo_dir), base_name, old_json)
                except Exception as db_err:
                    logger.warning(f"撤回时同步 SQLite 失败: {db_err}")

            # 还原 内存验证状态 并 保存
            if old_val is not None:
                self.validation_data[file_name] = old_val
                self._save_validation_data()
            else:
                self.validation_data.pop(file_name, None)
                validation_file_path = os.path.join(temp_photo_dir, "validation.json")
                if os.path.exists(validation_file_path):
                    try:
                        with open(validation_file_path, 'r', encoding='utf-8') as f:
                            disk_data = json.load(f)

                        if file_name in disk_data:
                            disk_data.pop(file_name, None)
                            with open(validation_file_path, 'w', encoding='utf-8') as f:
                                json.dump(disk_data, f, ensure_ascii=False, indent=2, sort_keys=True)
                    except Exception as e:
                        logger.error(f"撤回时清理 validation.json 失败: {e}")

            # 触发强制选中，自动刷新界面
            self._force_select_files = [file_name]
            self._refresh_species_list_logic()

            if hasattr(self.controller, 'status_bar'):
                current_text = self.controller.status_bar.status_label.text()
                self.controller.status_bar.status_label.setText(
                    f"✅ 已成功撤回上一步操作  |  {current_text}"
                )

        except Exception as e:
            logger.error(f"撤回失败: {e}")

    def _update_json_file(self, file_name, new_species=None,
                          new_count=None, new_remark=None):
        """更新物种信息，直接同步到软件缓存 SQLite（如果设置开启，同时同步到图像文件夹的 SQLite）"""
        from system.detection_db import get_db_path, update_detection, init_db

        try:
            temp_photo_dir = self.controller.get_temp_photo_dir()
            source_dir = self.controller.start_page.get_file_path()
            if not temp_photo_dir:
                return

            base_name = os.path.splitext(file_name)[0]
            json_path = os.path.join(temp_photo_dir, f"{base_name}.json")
            db_path = get_db_path(temp_photo_dir)

            # --- 优先从SQLite读取数据，或作为兼容从旧JSON读取 ---
            detection_info = {}
            import sqlite3

            save_to_source = getattr(getattr(self.controller, 'advanced_page', None), 'save_cache_to_image_folder_var',
                                     False)
            source_db_path = get_db_path(source_dir) if source_dir else None

            # 1. 优先从图像文件夹的 DB 读取
            if save_to_source and source_db_path and os.path.exists(source_db_path):
                try:
                    with sqlite3.connect(source_db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT detection_json FROM detections WHERE base_name=?", (base_name,))
                        row = cursor.fetchone()
                        if row:
                            detection_info = json.loads(row[0])
                except Exception:
                    pass

            # 2. 如果没有，则尝试从软件缓存区的 DB 读取
            if not detection_info and os.path.exists(db_path):
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT detection_json FROM detections WHERE base_name=?", (base_name,))
                        row = cursor.fetchone()
                        if row:
                            detection_info = json.loads(row[0])
                except Exception:
                    pass

            # 3. 如果数据库都没数据，尝试从 JSON 读取
            if not detection_info and os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    detection_info = json.load(f)

            # ── 原有字段更新逻辑 ──
            if new_species is not None:
                detection_info['物种名称'] = new_species
                detection_info['最低置信度'] = '人工校验'
                detection_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if new_count is not None:
                detection_info['物种数量'] = new_count
            if new_remark is not None:
                detection_info['备注'] = new_remark

            # ── 同步到软件缓存 SQLite ──
            try:
                update_detection(db_path, base_name, detection_info)
            except Exception as db_err:
                logger.warning(f"_update_json_file 同步缓存 SQLite 失败: {db_err}")

            # ── 同步到图像源文件夹 SQLite (如果高级设置中开启) ──
            try:
                save_to_source = getattr(
                    getattr(self.controller, 'advanced_page', None),
                    'save_cache_to_image_folder_var', False
                )
                if save_to_source and source_dir and os.path.isdir(source_dir):
                    source_db_path = get_db_path(source_dir)
                    if not os.path.exists(source_db_path):
                        init_db(source_db_path)
                    update_detection(source_db_path, base_name, detection_info)
            except Exception as e:
                logger.warning(f"_update_json_file 同步源文件夹 SQLite 失败: {e}")

        except Exception as e:
            logger.error(f"更新数据失败: {e}")

    def _update_detection_info_display(self):
        """更新检测信息显示"""
        if not (hasattr(self, 'species_info_label') and self.current_species_info):
            return

        try:
            # 1. 准备配置 Map (关键修复：构建全量物种的阈值表)
            # 获取当前的全局配置
            conf_map = {}
            if hasattr(self.controller, 'confidence_settings'):
                conf_map = self.controller.confidence_settings.copy()
            else:
                conf_map = {"global": 0.25}

            # 将当前滑块的值应用到 Map 中 (实现实时拖动预览)
            # 这样不仅能调整当前选中物种，也能正确保持其他物种的阈值不变
            current_species_key = "global"
            if hasattr(self, 'species_selector'):
                current_species_key = self.species_selector.currentData() or "global"

            # 使用当前滑块的值覆盖配置中的值
            conf_map[current_species_key] = self.species_conf_var

            # 默认值
            species_name = "未知"
            species_count = "空"
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

                    # 2. 过滤掉置信度不足的点 (关键修复：使用 conf_map 针对性过滤)
                    valid_points = []
                    for p in points:
                        sp = p.get('species', 'Unknown')
                        conf = p.get('confidence', 0)
                        # 获取该特定物种的阈值
                        thresh = conf_map.get(sp, conf_map.get("global", 0.25))

                        if conf >= thresh:
                            valid_points.append(p)

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
                    sorted_species = sorted(species_counts_map.keys())
                    species_name = ','.join(sorted_species)
                    species_count = ','.join([str(species_counts_map[s]) for s in sorted_species])
                    confidence = f"{min_conf_val:.2f}"
                else:
                    species_name = "空"
                    species_count = "空"
                    confidence = "N/A"

            # === 情况C：图片数据 (检测框) ===
            else:
                detection_boxes = self.current_species_info.get('检测框', [])
                if not detection_boxes:
                    detection_boxes = self.current_species_info.get('detect_results',
                                                                    self.current_species_info.get('objects', []))

                if detection_boxes:
                    species_counts_map = Counter()
                    min_conf_val = float('inf')
                    valid_boxes_found = False

                    for box in detection_boxes:
                        # 复用与画框一致的逻辑：检查候选项
                        species_found = None
                        conf_found = 0.0

                        # A. 检查候选项
                        if "候选项" in box and box["候选项"]:
                            for cand in box["候选项"]:
                                c_name = cand.get('name')
                                c_conf = float(cand.get('conf', 0))
                                # 关键修复：使用 map 获取该候选物种的特定阈值
                                c_thresh = conf_map.get(c_name, conf_map.get("global", 0.25))

                                if c_conf >= c_thresh:
                                    species_found = c_name
                                    conf_found = c_conf
                                    break

                        # B. 无候选项或候选项未匹配
                        if not species_found:
                            raw_name = box.get("物种", box.get("species", "未知"))
                            raw_conf = float(box.get("置信度", box.get("confidence", 0)))
                            # 关键修复：使用 map 获取该主物种的特定阈值
                            thresh = conf_map.get(raw_name, conf_map.get("global", 0.25))

                            if raw_conf >= thresh:
                                species_found = raw_name
                                conf_found = raw_conf

                        if species_found:
                            valid_boxes_found = True
                            species_counts_map[species_found] += 1
                            min_conf_val = min(min_conf_val, conf_found)

                    if valid_boxes_found:
                        sorted_species = sorted(species_counts_map.keys())
                        species_name = ','.join(sorted_species)
                        species_count = ','.join([str(species_counts_map[s]) for s in sorted_species])
                        confidence = f"{min_conf_val:.2f}"
                    else:
                        species_name = "空"
                        species_count = "空"
                        confidence = "N/A"
                else:
                    # 如果JSON本身没有框数据，回退到读取原始字段
                    species_name = self.current_species_info.get('物种名称', '未知')
                    species_count = self.current_species_info.get('物种数量', '未知')
                    confidence = self.current_species_info.get('最低置信度', '未知')

            # 统一拦截处理：如果置信度为 None 或 "None"、空字符串，则强制转换为 "N/A"
            if confidence is None or str(confidence).strip().lower() in ["none", ""]:
                confidence = "N/A"

            # 更新 UI 标签
            species_type_str = "空"
            if species_name and species_name not in ["[未校验] 空", "未知", "N/A", "[已校验] 空"]:
                type_list = []
                # 支持多物种（逗号分隔）
                names = species_name.replace('，', ',').split(',')
                for name in names:
                    name = name.strip()
                    if name:
                        # 直接查 SQLite 数据库
                        sType = self._get_species_info_from_db(name)
                        type_list.append(sType if sType else "空")

                if type_list:
                    species_type_str = ",".join(type_list)

            # 更新 UI 标签
            info_text = f"物种: {species_name} | 数量: {species_count} | 类型: {species_type_str} | 置信度: {confidence}"
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
            # 如果当前已经是列表最后一张
            # 判断列表中是否还有其他照片（总数 > 1 说明除了当前这张，还有别的没处理完）
            if self.species_photo_listbox.count() > 1:
                # 循环跳转回当前分组的第一张照片
                self.species_photo_listbox.setCurrentRow(0)
            else:
                # 真正完全没有其他照片了，才显示完成提示
                MaterialMessageBox.information(self, "提示", "当前物种的所有图像已处理完成！")

    def _show_photo_context_menu(self, pos):
        """显示照片列表的右键菜单"""
        selected_items = self.species_photo_listbox.selectedItems()
        if not selected_items:
            return

        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        # 获取当前主题状态，动态适应深/浅色模式
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD.name()
            border_color = Win11Colors.DARK_BORDER.name()
            text_color = Win11Colors.DARK_TEXT_PRIMARY.name()
            accent_color = Win11Colors.DARK_ACCENT.name()
        else:
            bg_color = Win11Colors.LIGHT_CARD.name()
            border_color = Win11Colors.LIGHT_BORDER.name()
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY.name()
            accent_color = Win11Colors.LIGHT_ACCENT.name()

        # 动态应用 Material You 风格样式
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
                padding: 6px;
                color: {text_color};
                font-size: 14px;
                font-weight: 500;
            }}
            QMenu::item {{
                padding: 8px 28px;
                border-radius: 8px;
                margin: 2px 4px;
            }}
            QMenu::item:selected {{
                background-color: {accent_color};
                color: #ffffff;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {border_color};
                margin: 4px 8px;
            }}
        """)

        # 动态显示选中了多少项
        action_correct = menu.addAction(f"标记为正确 ({len(selected_items)}项)")
        action_empty = menu.addAction(f"标记为空 ({len(selected_items)}项)")
        action_other = menu.addAction(f"标注为其他 ({len(selected_items)}项)")
        action_unverified = menu.addAction(f"改为未校验 ({len(selected_items)}项)")

        menu.addSeparator()

        # 重新检测选项
        action_redetect = menu.addAction(f"重新检测 ({len(selected_items)}项)")

        action = menu.exec(self.species_photo_listbox.mapToGlobal(pos))

        if action == action_correct:
            self._bulk_mark_correct(selected_items)
        elif action == action_empty:
            self._bulk_mark_empty(selected_items)
        elif action == action_other:
            self._bulk_mark_other(selected_items)
        elif action == action_unverified:
            self._bulk_mark_unverified(selected_items)

        # 点击重新检测后调用新的处理函数
        elif action == action_redetect:
            self._bulk_redetect(selected_items)

    def _bulk_mark_correct(self, items):
        """批量标记为正确"""
        file_names = []
        tasks = []
        for item in items:
            file_name = item.text()
            file_names.append(file_name)
            self._push_to_undo_stack(file_name)
            self.validation_data[file_name] = True
            tasks.append({'file_name': file_name, 'action': 'correct'})

        self._force_select_files = file_names
        self._run_background_update(tasks, lambda: QTimer.singleShot(50, self._refresh_species_list_logic))

    def _bulk_mark_empty(self, items):
        """批量标记为空"""
        file_names = []
        tasks = []
        for item in items:
            file_name = item.text()
            file_names.append(file_name)
            self._push_to_undo_stack(file_name)
            self.validation_data[file_name] = False
            tasks.append({'file_name': file_name, 'action': 'empty', 'new_species': '空', 'new_count': '空'})

        self._force_select_files = file_names
        self._run_background_update(tasks, lambda: QTimer.singleShot(50, self._refresh_species_list_logic))

    def _bulk_mark_other(self, items):
        """批量标注为其他物种"""
        dialog = CorrectionDialog(self, title=f"批量输入其他物种信息 ({len(items)}项)", original_info=self.current_species_info)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result:
            species_name, species_count, remark = dialog.result

            file_names = []
            tasks = []
            for item in items:
                file_name = item.text()
                file_names.append(file_name)
                self._push_to_undo_stack(file_name)
                self.validation_data[file_name] = False
                tasks.append({'file_name': file_name, 'action': 'update', 'new_species': species_name, 'new_count': species_count, 'new_remark': remark})

            self._increment_quick_mark_count(species_name)
            self._force_select_files = file_names

            def on_finished():
                if hasattr(self.controller, 'advanced_page') and self.controller.advanced_page.auto_sort_switch_row.isChecked():
                    self._load_species_buttons()
                    self.quick_marks_updated.emit()
                QTimer.singleShot(50, self._refresh_species_list_logic)

            self._run_background_update(tasks, on_finished)

    def _bulk_mark_unverified(self, items):
        """批量将状态重置为未校验"""
        file_names = []
        tasks = []
        for item in items:
            file_name = item.text()
            file_names.append(file_name)
            self._push_to_undo_stack(file_name)
            self.validation_data.pop(file_name, None)
            tasks.append({'file_name': file_name, 'action': 'unverified'})

        self._force_select_files = file_names
        self._run_background_update(tasks, lambda: QTimer.singleShot(50, self._refresh_species_list_logic))

    def _bulk_redetect(self, items):
        """批量重新检测选中的文件"""
        file_names = [item.text() for item in items]
        source_dir = self.controller.start_page.get_file_path()
        temp_photo_dir = self.controller.get_temp_photo_dir()

        if not source_dir or not temp_photo_dir:
            return

        # 推入撤回栈以便日后恢复
        for file_name in file_names:
            self._push_to_undo_stack(file_name)

        # 禁用列表控件防止检测期间被点击扰乱状态
        self.species_photo_listbox.setEnabled(False)
        self.species_listbox.setEnabled(False)

        # 直接调用主窗口的统一接口
        self.controller.redetect_files(file_names, callback=self._on_redetect_finished)

    def _on_redetect_finished(self, success, file_names):
        """重新检测完成后的后置处理"""
        # 恢复底部进度条状态及界面控件
        self.controller.status_bar.hide_progress()
        self.species_photo_listbox.setEnabled(True)
        self.species_listbox.setEnabled(True)

        if success:
            self.controller.status_bar.status_label.setText("✅ 选定文件重新检测完成")
        else:
            self.controller.status_bar.status_label.setText("❌ 重新检测过程出现错误，请查看日志")

        # 重测完成后，必须抹除掉这些文件以前带有的人工检验标志(以便界面上恢复到自动判断的状态)
        temp_photo_dir = self.controller.get_temp_photo_dir()
        files_to_remove = []

        for file_name in file_names:
            files_to_remove.append(file_name)
            # 清除内存状态
            self.validation_data.pop(file_name, None)

        # 彻底移除本地 validation.json 中的标记，防止自动合并时老数据又被合并回来
        if temp_photo_dir and files_to_remove:
            # 👇 这里补充导入 update_detection
            from system.detection_db import get_db_path, delete_validation_bulk, update_detection

            db_path = get_db_path(temp_photo_dir)
            if os.path.exists(db_path):
                delete_validation_bulk(db_path, files_to_remove)

                # ==========================================
                # 👇 【新增修复】：将重测后磁盘上最新的 JSON 数据强制同步更新到 SQLite 数据库中
                for f_name in file_names:
                    base_name = os.path.splitext(f_name)[0]
                    json_path = os.path.join(temp_photo_dir, f"{base_name}.json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                new_detection_info = json.load(f)
                            # 覆盖更新 SQLite 缓存中的旧检测结果
                            update_detection(db_path, base_name, new_detection_info)
                        except Exception as e:
                            logger.error(f"同步重测结果到 SQLite 失败: {f_name}, {e}")

            # 同时保持 validation.json 同步（可选，兼容性用）
            validation_file_path = os.path.join(temp_photo_dir, "validation.json")
            if os.path.exists(validation_file_path):
                try:
                    with open(validation_file_path, 'r', encoding='utf-8') as f:
                        disk_data = json.load(f)
                    modified = False
                    for f_name in files_to_remove:
                        if disk_data.pop(f_name, None) is not None:
                            modified = True
                    if modified:
                        with open(validation_file_path, 'w', encoding='utf-8') as f:
                            json.dump(disk_data, f, ensure_ascii=False, indent=2, sort_keys=True)
                except Exception as e:
                    logger.error(f"清理 validation.json 失败: {e}")

        # 设置重新选中标记 (如果有之前选择的文件在这个列表里，界面刷新后会自动保持选中)
        if len(file_names) > 0:
            self._force_select_files = file_names

        # 重新加载左侧列表和对应显示
        QTimer.singleShot(50, self._refresh_species_list_logic)

    def _update_confidence_label(self, value):
        """更新置信度标签"""
        if hasattr(self, 'species_conf_label'):
            confidence_value = value / 100.0 if isinstance(value, int) else value
            self.species_conf_label.setText(f"{confidence_value:.2f}")
            self.species_conf_var = confidence_value

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
        """
        在图像上绘制检测框
        """
        try:
            # 1. 准备配置 Map (用于候选项筛选)
            # 获取当前的全局配置
            conf_map = {}
            if hasattr(self.controller, 'confidence_settings'):
                conf_map = self.controller.confidence_settings.copy()
            else:
                conf_map = {"global": 0.25}

            # 将当前滑块的值应用到 Map 中 (实现实时拖动预览)
            # 这样不仅能调整当前选中物种，也能正确过滤其他背景物种
            current_species_key = "global"
            if hasattr(self, 'species_selector'):
                current_species_key = self.species_selector.currentData() or "global"

            conf_map[current_species_key] = conf_threshold

            # 2. 准备绘图工具
            # 确保在副本上绘制，且为 RGB 模式
            if img.mode != 'RGB':
                img_to_draw = img.convert('RGB')
            else:
                img_to_draw = img.copy()

            draw = ImageDraw.Draw(img_to_draw)
            img_width, img_height = img.size

            # 兼容多种键名
            boxes_info = detection_info.get("检测框", [])
            if not boxes_info:
                boxes_info = detection_info.get("detect_results", detection_info.get("objects", []))

            if not boxes_info:
                return img_to_draw

            # 3. 加载字体 (动态大小: 图像短边的 2%，保持与 PreviewPage 一致)
            try:
                from system.utils import resource_path
                font_path = resource_path(os.path.join("res", "AlibabaPuHuiTi-3-65-Medium.ttf"))
                font_size = max(12, int(0.02 * min(img_width, img_height)))
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

            # 4. 遍历并绘制
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

            return img_to_draw

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
            if not item: continue

            # ==================== [修改] 忽略前缀进行匹配 ====================
            item_text = item.text()
            map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
            clean_name = map_key.replace('[已校验] ', '').replace('[未校验] ', '')

            # 检查物种名称是否匹配 (如"赤狐")
            if clean_name == species_name:
                # 进一步检查要选中的图像是否正处于这个 [校验/未校验] 分组内
                if image_filename in self.species_image_map.get(map_key, []):
                    # 设置当前选中项
                    self.species_listbox.setCurrentItem(item)
                    # 滚动以确保可见
                    self.species_listbox.scrollToItem(item)

                    # 手动调用选中逻辑以填充照片列表
                    self._on_species_selected()

                    # 2. 定义一个内部函数来选中照片
                    def select_image_item():
                        for j in range(self.species_photo_listbox.count()):
                            photo_item = self.species_photo_listbox.item(j)
                            if photo_item and photo_item.text() == image_filename:
                                self.species_photo_listbox.setCurrentItem(photo_item)
                                self.species_photo_listbox.scrollToItem(photo_item)
                                break

                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(100, select_image_item)
                    return

    def _start_video_thread(self, video_path):
        """启动视频播放线程，直接传递内存中基于 SQLite 的检测数据"""
        self._stop_video_thread()
        self._is_video_paused = False

        # 获取高级设置中的过滤比例 (如果可用)
        min_ratio = 0.0
        if hasattr(self.controller, 'advanced_page'):
            min_ratio = self.controller.advanced_page.min_frame_ratio_var

        self.video_thread = VideoPlayerThread(
            video_path,
            self.current_species_info, # 直接传字典，取代原本的 json_path
            self.species_conf_var,
            draw_boxes=True,
            min_frame_ratio=min_ratio
        )
        self.video_thread.frame_ready.connect(self._on_video_frame_ready)
        self.video_thread.pause_state_changed.connect(self._on_video_pause_state_changed)
        self.video_thread.start()

    def _stop_video_thread(self):
        """停止视频线程"""
        if self.video_thread:
            # 1. 关键修复：先断开信号连接。
            # 即使线程还在后台由于惯性跑完最后一帧，也无法触发 UI 更新去覆盖静态图。
            try:
                self.video_thread.frame_ready.disconnect(self._on_video_frame_ready)
                self.video_thread.pause_state_changed.disconnect(self._on_video_pause_state_changed)
            except Exception:
                # 忽略如果没有连接时的异常
                pass

            # 2. 停止线程运行
            if self.video_thread.isRunning():
                self.video_thread.stop()
                # 注意：不要在这里使用 wait() 阻塞主线程，因为我们已经断开了信号，
                # 让线程自己在后台安全退出即可。

            self.video_thread.deleteLater()
            self.video_thread = None

        # 3. 清除缓存的视频帧，防止 resizeEvent 意外重绘旧视频帧
        self._current_video_frame_pixmap = None
        self._is_video_paused = False

    def _on_video_frame_ready(self, pixmap):
        """接收视频帧并显示"""
        if not self.video_thread or not self.isVisible():
            return

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
        """处理点击事件：单击左键暂停/播放视频，双击左键或单击右键打开系统默认查看器"""
        # 仅处理针对图片显示标签的事件
        if source == self.species_image_label:
            
            # === 1. 双击事件 (左键) ===
            if event.type() == QEvent.Type.MouseButtonDblClick:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._open_media_externally()
                    return True

            # === 2. 鼠标按下事件 ===
            elif event.type() == QEvent.Type.MouseButtonPress:
                
                # 情况 A: 右键单击 -> 调用系统默认软件打开
                if event.button() == Qt.MouseButton.RightButton:
                    self._open_media_externally()
                    return True
                
                # 情况 B: 左键单击 -> 视频暂停/播放逻辑
                elif event.button() == Qt.MouseButton.LeftButton:
                    # 注意：双击时会先触发一次 Press，这里会让视频暂停一下随后立即被外部打开，这是正常现象
                    if self.video_thread and self.video_thread.isRunning():
                        self.video_thread.toggle_pause()
                        return True
                        
        return super().eventFilter(source, event)

    def _open_media_externally(self):
        """调用系统默认程序打开当前显示的图片或视频"""
        
        # === 在打开外部程序前，确保内部视频播放器处于暂停状态 ===
        if self.video_thread and self.video_thread.isRunning():
            # 如果当前没有暂停（即正在播放），则执行暂停操作
            if not self.video_thread.paused:
                self.video_thread.toggle_pause()

        # 检查是否有选中的文件
        if not hasattr(self, 'last_selected_species_image') or not self.last_selected_species_image:
            return

        # 获取文件完整路径
        source_dir = self.controller.start_page.get_file_path()
        file_path = os.path.join(source_dir, self.last_selected_species_image)

        if os.path.exists(file_path):
            try:
                # 使用 QDesktopServices 打开本地文件，这是跨平台调用系统默认软件的标准方式
                QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                logger.info(f"已调用系统默认软件打开: {file_path}")
            except Exception as e:
                logger.error(f"无法打开外部文件: {e}")
        else:
            logger.warning(f"文件不存在，无法打开: {file_path}")

    def _generate_white_icon_pixmap(self, icon_path, size):
        """生成白色图标 (复用 PreviewPage 的逻辑)"""
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

    def _refresh_species_list_logic(self):
        """重新加载物种列表，并尽可能恢复之前的选中状态"""
        # 保存配置到磁盘 (延迟保存)
        self._save_species_conf()

        # 1. 记住当前的选中状态
        current_species = self.current_selected_species

        # 记住当前左侧确切选中的完整分组名
        current_group_exact = None
        current_item = self.species_listbox.currentItem()
        if current_item:
            current_group_exact = current_item.text().split(' (')[
                0] if ' (' in current_item.text() else current_item.text()

        # 使用列表支持多选恢复
        force_files = getattr(self, '_force_select_files', None)
        current_photo_names = []
        if force_files:
            current_photo_names = force_files
            self._force_select_files = None  # 消费掉标志，避免影响后续的常规刷新
        else:
            # 兼容旧的单选标志位
            force_file = getattr(self, '_force_select_file', None)
            if force_file:
                current_photo_names = [force_file]
                self._force_select_file = None
            else:
                selected_items = self.species_photo_listbox.selectedItems()
                if selected_items:
                    current_photo_names = [item.text() for item in selected_items]

        current_photo_row = self.species_photo_listbox.currentRow()

        # 2. 重新加载物种数据
        # 暂时阻断信号，避免在 clear() 和 addItem() 期间触发不必要的闪烁
        self.species_listbox.blockSignals(True)
        self._load_species_data_core()
        self.species_listbox.blockSignals(False)

        # 3. 尝试恢复物种选中状态
        if current_species:
            target_item = None
            # 获取是否开启了自动分组
            auto_group = getattr(getattr(self.controller, 'advanced_page', None), 'auto_group_var', False)

            # 策略A：如果是批量操作 (有 force_files 标志)，优先追踪照片跳转到新分组
            if force_files and current_photo_names:
                for current_photo_name in current_photo_names:
                    for idx in range(self.species_listbox.count()):
                        item = self.species_listbox.item(idx)
                        item_text = item.text()
                        map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                        if current_photo_name in self.species_image_map.get(map_key, []):
                            target_item = item
                            break
                    if target_item:
                        break

            if auto_group:
                # ========================================================
                # 开启了事件分组模式：物种的修改会导致分组名称实时变化（如合并成 狍子,赤狐）
                # 因此必须优先跟随当前处理的照片，防止因为组名变化导致跳错分组
                # ========================================================
                if not target_item and current_photo_names:
                    for current_photo_name in current_photo_names:
                        for idx in range(self.species_listbox.count()):
                            item = self.species_listbox.item(idx)
                            item_text = item.text()
                            map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                            if current_photo_name in self.species_image_map.get(map_key, []):
                                target_item = item
                                break
                        if target_item:
                            break

                # 兜底：如果照片不见了，再尝试找原分组名
                if not target_item and current_group_exact:
                    for idx in range(self.species_listbox.count()):
                        item = self.species_listbox.item(idx)
                        item_text = item.text()
                        map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                        if map_key == current_group_exact:
                            target_item = item
                            break
            else:
                # ========================================================
                # 未开启事件分组模式：照片是按单一物种聚合的。如果修改了某张照片的物种，
                # 照片会被踢出当前组。此时应该优先留在原物种组，继续处理其他同类照片。
                # ========================================================
                if not target_item and current_group_exact:
                    for idx in range(self.species_listbox.count()):
                        item = self.species_listbox.item(idx)
                        item_text = item.text()
                        map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                        if map_key == current_group_exact:
                            target_item = item
                            break

                # 兜底：如果原物种组因为处理完了而消失，才去跟随照片去向
                if not target_item and current_photo_names:
                    for current_photo_name in current_photo_names:
                        for idx in range(self.species_listbox.count()):
                            item = self.species_listbox.item(idx)
                            item_text = item.text()
                            map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                            if current_photo_name in self.species_image_map.get(map_key, []):
                                target_item = item
                                break
                        if target_item:
                            break

            # 策略D：终极兜底，尝试按前缀找回原来的物种大类
            if not target_item:
                for prefix in ["[未校验] ", "[已校验] ", ""]:
                    search_key = f"{prefix}{current_species}"
                    for idx in range(self.species_listbox.count()):
                        item = self.species_listbox.item(idx)
                        item_text = item.text()
                        map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                        if map_key == search_key:
                            target_item = item
                            break
                    if target_item:
                        break

            if target_item:
                self.species_listbox.blockSignals(True)
                self.species_listbox.setCurrentItem(target_item)
                self.species_listbox.scrollToItem(target_item)
                self.species_listbox.blockSignals(False)

                if hasattr(self, '_species_scroll_delegate'):
                    self._species_scroll_delegate._update_active_rows()

                # 手动恢复照片列表
                item_text = target_item.text()
                map_key = item_text.split(' (')[0] if ' (' in item_text else item_text
                self.current_selected_species = map_key.replace('[已校验] ', '').replace('[未校验] ', '')
                image_files = sorted(self.species_image_map.get(map_key, []))

                self.species_photo_listbox.blockSignals(True)
                self.species_photo_listbox.clear()

                from PySide6.QtGui import QBrush
                app = QApplication.instance()
                is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
                alt_bg_color = QColor(255, 255, 255, 40) if is_dark else QColor(0, 0, 0, 25)
                alt_brush = QBrush(alt_bg_color)

                groups = getattr(self, 'species_group_map', {}).get(map_key, [])
                if groups:
                    for i, grp in enumerate(groups):
                        for img in sorted(grp):
                            self.species_photo_listbox.addItem(img)
                            if i % 2 == 1:
                                item = self.species_photo_listbox.item(self.species_photo_listbox.count() - 1)
                                item.setBackground(alt_brush)
                else:
                    for i, img in enumerate(image_files):
                        self.species_photo_listbox.addItem(img)
                        if i % 2 == 1:
                            item = self.species_photo_listbox.item(self.species_photo_listbox.count() - 1)
                            item.setBackground(alt_brush)

                self.species_photo_listbox.blockSignals(False)

                if hasattr(self, '_photo_scroll_delegate'):
                    self._photo_scroll_delegate._update_active_rows()

                # 恢复照片选中 (多选逻辑)
                if current_photo_names:
                    # 1. 先找出需要被设为当前焦点（CurrentItem）的第一项
                    first_item_found = None
                    for photo_name in current_photo_names:
                        items = self.species_photo_listbox.findItems(photo_name, Qt.MatchFlag.MatchExactly)
                        if items:
                            first_item_found = items[0]
                            break

                    # 2. 优先设置 CurrentItem（此时 Qt 底层无论怎么重置现有多选状态都没关系）
                    if first_item_found:
                        self.species_photo_listbox.setCurrentItem(first_item_found)
                        self.species_photo_listbox.scrollToItem(first_item_found)

                    # 3. 最后遍历所有需要选中的项，安全地将其强行设为选中状态
                    for photo_name in current_photo_names:
                        items = self.species_photo_listbox.findItems(photo_name, Qt.MatchFlag.MatchExactly)
                        if items:
                            items[0].setSelected(True)

                # 更新状态栏
                if hasattr(self.controller, 'status_bar'):
                    self.controller.status_bar.status_label.setText(f"当前物种共有 {len(image_files)} 张照片")

        self.species_photo_listbox.setFocus()

    def reload_and_apply_conf(self):
        """
        从 conf.json 重新加载配置并强制刷新界面。
        用于页面切换时同步最新的置信度设置。
        """
        # 1. 从磁盘加载最新配置到内存 (self.controller.confidence_settings)
        self._load_species_conf()

        # 2. 刷新下拉框和滑块状态
        self._on_species_selector_changed()

        # 3. 核心修复：强制调用带状态恢复的列表刷新逻辑。
        self._refresh_species_list_logic()

        # 4. 如果当前有加载视频且正在运行，需单独更新其阈值并刷新 (图片重绘已在上一条的选中恢复逻辑中自动处理)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.conf_threshold = self.species_conf_var
            if self.video_thread.paused:
                self.video_thread.refresh_frame()

    def _get_species_info_from_db(self, species_name):
        """直接从 SQLite 查询物种类型"""
        if not species_name:
            return "空"

        species_type = "空"
        try:
            db_path = resource_path(os.path.join("res", "species_database.db"))
            if os.path.exists(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 物种类型 FROM species WHERE 中文名=?", (species_name,))
                row = cursor.fetchone()
                if row:
                    species_type = str(row[0]).strip() if row[0] else ""
                conn.close()
        except Exception as e:
            logger.error(f"查询数据库失败: {e}")

        return species_type
