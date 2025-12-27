from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QSpacerItem, QApplication, QLabel
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer


from system.gui.ui_components import (
    Win11Colors, RoundedButton,
    ModernLineEdit, PathInputWidget,
    ModernGroupBox, ModernComboBox
)
import os
from system.utils import resource_path


class StartPage(QWidget):
    """开始处理页面 - PySide6版本"""

    # 信号定义
    browse_file_path_requested = Signal()
    browse_save_path_requested = Signal()
    file_path_changed = Signal(str)
    save_path_changed = Signal(str)
    toggle_processing_requested = Signal()
    settings_changed = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._load_icons()
        self._setup_ui()
        self._setup_connections()

    def _load_icons(self):
        """加载图标"""
        self.play_icon = self._load_svg_icon(resource_path("res/icon/play.svg"))
        self.stop_icon = self._load_svg_icon(resource_path("res/icon/stop.svg"))

    def _load_svg_icon(self, icon_path):
        """加载SVG图标"""
        if os.path.exists(icon_path):
            renderer = QSvgRenderer(icon_path)
            if renderer.isValid():
                pixmap = QPixmap(16, 16)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                return pixmap
        return None

    def _create_colored_pixmap(self, original_pixmap, color):
        """根据颜色创建着色的图标"""
        if not original_pixmap:
            return QIcon()

        colored_pixmap = QPixmap(original_pixmap.size())
        colored_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(colored_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        painter.end()
        return QIcon(colored_pixmap)

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 基础设置组 (原路径设置 + 快速设置)
        self._create_basic_settings_group(layout)

        # 添加控制台组件（初始隐藏）
        self._create_console_widget(layout)

        # 添加较小的弹性空间（当控制台隐藏时使用）
        # 当控制台显示时，它会占据可用空间
        self.spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addItem(self.spacer)

        # 底部控制区域
        self._create_bottom_controls(layout)

    def _create_basic_settings_group(self, parent_layout):
        """创建基础设置组 (包含路径、模型和跳帧)"""
        # 分组框名称改为 "基础设置"
        basic_group = ModernGroupBox("基础设置")
        basic_layout = QVBoxLayout(basic_group)
        basic_layout.setSpacing(20)
        basic_layout.setContentsMargins(16, 16, 16, 16)

        # --- 1. 图像文件路径 (占满一行) ---
        self.file_path_widget = PathInputWidget(
            "图像文件路径:",
            "请选择包含图像文件的文件夹"
        )
        basic_layout.addWidget(self.file_path_widget)

        # --- 2. 模型选择与跳帧设置 (下方水平并排) ---
        settings_row_layout = QHBoxLayout()
        settings_row_layout.setSpacing(20)

        # A. 模型选择
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)

        model_label = QLabel("选择模型:")
        model_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        model_layout.addWidget(model_label)

        self.model_combo = ModernComboBox()
        self.model_combo.setMinimumWidth(200)
        self._populate_models()  # 填充模型列表
        model_layout.addWidget(self.model_combo)

        settings_row_layout.addLayout(model_layout)

        # B. 跳帧设置
        stride_layout = QVBoxLayout()
        stride_layout.setSpacing(8)

        stride_label = QLabel("视频跳帧 (Frame Stride):")
        stride_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        stride_layout.addWidget(stride_label)

        self.stride_combo = ModernComboBox()
        self.stride_combo.setMinimumWidth(150)
        # 默认选项
        self.default_strides = ["5", "10", "15", "20", "25", "30"]
        self.stride_combo.addItems(self.default_strides)
        stride_layout.addWidget(self.stride_combo)

        settings_row_layout.addLayout(stride_layout)

        # 添加弹性空间填充右侧，保持左对齐
        settings_row_layout.addStretch()

        # 将水平布局添加到主垂直布局中
        basic_layout.addLayout(settings_row_layout)

        parent_layout.addWidget(basic_group)

    def _create_console_widget(self, parent_layout):
        """创建控制台组件"""
        from PySide6.QtWidgets import QTextEdit

        # 控制台容器
        self.console_container = ModernGroupBox("处理控制台")
        self.console_container.hide()  # 初始隐藏

        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(16, 16, 16, 16)
        console_layout.setSpacing(12)

        # 文本输出区域
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        # 移除固定高度限制，让它能够自适应
        self.console_output.setMinimumHeight(200)
        # 移除 setMaximumHeight，让控制台可以扩展

        # 设置尺寸策略为可扩展
        self.console_output.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 设置控制台样式
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            border_color = Win11Colors.DARK_BORDER
        else:
            bg_color = Win11Colors.LIGHT_CARD
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            border_color = Win11Colors.LIGHT_BORDER

        self.console_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_color.name()};
                color: {text_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }}
        """)

        console_layout.addWidget(self.console_output)

        self.console_container.setLayout(console_layout)

        # 设置控制台容器的尺寸策略，允许垂直扩展
        self.console_container.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )

        parent_layout.addWidget(self.console_container)

    def show_console(self):
        """显示控制台"""
        self.console_container.show()
        # 当显示控制台时，减少弹性空间
        if hasattr(self, 'spacer'):
            self.layout().removeItem(self.spacer)
            self.spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            # 在底部控制区域之前插入弹性空间
            self.layout().insertItem(self.layout().count() - 1, self.spacer)

    def hide_console(self):
        """隐藏控制台"""
        self.console_container.hide()
        # 当隐藏控制台时，恢复弹性空间
        if hasattr(self, 'spacer'):
            self.layout().removeItem(self.spacer)
            self.spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            # 在底部控制区域之前插入弹性空间
            self.layout().insertItem(self.layout().count() - 1, self.spacer)

    def append_console_log(self, message: str, color: str = None):
        """添加控制台日志"""
        if color:
            self.console_output.append(f'<span style="color: {color};">{message}</span>')
        else:
            self.console_output.append(message)

        # 强制刷新显示
        self.console_output.repaint()
        QApplication.processEvents()  # 立即处理事件

        # 自动滚动到底部
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


    def _populate_models(self):
        """扫描并填充模型列表"""
        self.model_combo.clear()
        try:
            model_dir = resource_path(os.path.join("res", "model"))
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.pt')]
                model_files.sort()
                if model_files:
                    self.model_combo.addItems(model_files)
                else:
                    self.model_combo.addItem("未找到模型")
            else:
                self.model_combo.addItem("模型目录不存在")
        except Exception as e:
            self.model_combo.addItem("加载失败")
            print(f"Error loading models: {e}")

    def _create_bottom_controls(self, parent_layout):
        """创建底部控制区域"""
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)

        # 开始按钮容器
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 添加弹性空间，使按钮右对齐
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # 开始/停止按钮
        self.start_stop_button = RoundedButton("开始处理")
        self.start_stop_button.setMinimumSize(160, 50)
        self.start_stop_button.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))

        # 设置按钮样式
        self._update_button_style(False)

        button_layout.addWidget(self.start_stop_button)
        bottom_layout.addWidget(button_container)

        parent_layout.addWidget(bottom_widget)

    def _update_button_style(self, is_processing):
        """更新按钮样式"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        icon_color = QColor("#ffffff")

        if is_processing:
            # 停止状态 - 使用红色
            bg_color = "#e74c3c"
            hover_color = "#c0392b"
            pressed_color = "#a93226"
            text_color = "#ffffff"
            self.start_stop_button.setText(" 停止处理")
            if self.stop_icon:
                self.start_stop_button.setIcon(self._create_colored_pixmap(self.stop_icon, icon_color))
        else:
            # 开始状态 - 使用主题色
            if is_dark:
                bg_color = Win11Colors.DARK_ACCENT.name()
                hover_color = Win11Colors.DARK_ACCENT.lighter(120).name()
                pressed_color = Win11Colors.DARK_ACCENT.darker(110).name()
            else:
                bg_color = Win11Colors.LIGHT_ACCENT.name()
                hover_color = Win11Colors.LIGHT_ACCENT.darker(110).name()
                pressed_color = Win11Colors.LIGHT_ACCENT.darker(120).name()
            text_color = "#ffffff"
            self.start_stop_button.setText(" 开始处理")
            if self.play_icon:
                self.start_stop_button.setIcon(self._create_colored_pixmap(self.play_icon, icon_color))


        self.start_stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)

    def _setup_connections(self):
        """设置信号连接"""
        # 路径输入控件
        self.file_path_widget.browse_requested.connect(self.browse_file_path_requested.emit)
        self.file_path_widget.path_changed.connect(self.file_path_changed.emit)

        # 使用 lambda 丢弃 textChanged 发出的字符串参数，避免 TypeError
        self.model_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        self.stride_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())

        # 开始/停止按钮
        self.start_stop_button.clicked.connect(self.toggle_processing_requested.emit)

    def set_processing_state(self, is_processing: bool):
        """设置UI的处理状态"""
        # 禁用或启用设置控件
        self.file_path_widget.set_enabled(not is_processing)

        # 处理时禁用快速设置
        self.model_combo.setEnabled(not is_processing)
        self.stride_combo.setEnabled(not is_processing)

        # 更新处理按钮的样式和文本
        self._update_button_style(is_processing)

    def update_progress(self, value, total, speed, remaining_time, current_file=None):
        """更新进度"""
        if hasattr(self.progress_frame, 'update_progress'):
            self.progress_frame.update_progress(value, total, speed, remaining_time, current_file)

    def set_processing_enabled(self, enabled):
        """设置处理功能的启用状态"""
        self.start_stop_button.setEnabled(enabled)

    # 获取器方法
    def get_file_path(self):
        """获取文件路径"""
        return self.file_path_widget.get_path()

    def get_save_path(self):
        """获取保存路径"""
        return None

    # 设置器方法
    def set_file_path(self, path):
        """设置文件路径"""
        self.file_path_widget.set_path(path)

    def set_save_path(self, path):
        """设置保存路径"""
        pass

    # 设置和加载方法
    def get_settings(self):
        """获取页面设置"""
        return {
            "file_path": self.get_file_path(),
            "selected_model": self.model_combo.currentText(),
            "vid_stride": int(self.stride_combo.currentText()) if self.stride_combo.currentText().isdigit() else 1 # [新增]
        }

    def load_settings(self, settings):
        """加载页面设置"""
        if "file_path" in settings and settings["file_path"] and os.path.exists(settings["file_path"]):
            self.set_file_path(settings["file_path"])

            # 加载模型选择
            if "selected_model" in settings and settings["selected_model"]:
                # 尝试选中设置中的模型
                index = self.model_combo.findText(settings["selected_model"])
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

            # 加载跳帧设置
            if "vid_stride" in settings:
                stride_val = str(settings["vid_stride"])

                self.stride_combo.blockSignals(True)  # 暂时屏蔽信号

                # 1. 清空当前列表
                self.stride_combo.clear()
                # 2. 重新添加默认选项
                self.stride_combo.addItems(self.default_strides)

                # 3. 如果当前设置的值不在默认列表中，则单独添加
                if stride_val not in self.default_strides:
                    self.stride_combo.addItem(stride_val)

                # 4. 选中当前值
                self.stride_combo.setCurrentText(stride_val)

                self.stride_combo.blockSignals(False)  # 恢复信号

    def update_theme(self):
        """更新主题"""
        # 重新应用样式到所有子组件
        for child in self.findChildren(ModernGroupBox):
            child._setup_style()
        for child in self.findChildren(ModernLineEdit):
            child._setup_style()
        for child in self.findChildren(PathInputWidget):
            if hasattr(child, 'update_theme'):
                child.update_theme()
        # 更新下拉框样式
        for child in self.findChildren(ModernComboBox):
            if hasattr(child, 'update_theme'):
                child.update_theme()

        # 更新按钮样式
        self._update_button_style(self.controller.is_processing if hasattr(self.controller, 'is_processing') else False)

    def update_quick_settings(self, model_name, stride):
        """从外部更新快速设置控件状态（不触发信号）"""
        # 1. 更新模型选择
        if model_name:
            self.model_combo.blockSignals(True)
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)

        # 2. 更新跳帧设置
        if stride is not None:
            self.stride_combo.blockSignals(True)

            stride_str = str(stride)

            # 每次更新都重置列表，确保不残留历史数值
            self.stride_combo.clear()
            self.stride_combo.addItems(self.default_strides)

            # 如果当前下拉框里没有这个数值（非默认值），添加进去
            if stride_str not in self.default_strides:
                self.stride_combo.addItem(stride_str)

            self.stride_combo.setCurrentText(stride_str)

            self.stride_combo.blockSignals(False)