from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QSpacerItem, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer


from system.gui.ui_components import (
    Win11Colors, RoundedButton,
    ModernLineEdit, PathInputWidget, ModernGroupBox
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

        # 路径设置组
        self._create_paths_group(layout)

        # 添加控制台组件（初始隐藏）
        self._create_console_widget(layout)

        # 添加较小的弹性空间（当控制台隐藏时使用）
        # 当控制台显示时，它会占据可用空间
        self.spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addItem(self.spacer)

        # 底部控制区域
        self._create_bottom_controls(layout)

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
        self.console_output.clear()
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

    def _create_paths_group(self, parent_layout):
        """创建路径设置组"""
        paths_group = ModernGroupBox("路径设置")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setSpacing(16)
        paths_layout.setContentsMargins(16, 16, 16, 16)

        # 图像文件路径
        self.file_path_widget = PathInputWidget(
            "图像文件路径:",
            "请选择包含图像文件的文件夹"
        )
        paths_layout.addWidget(self.file_path_widget)

        # 结果保存路径
        self.save_path_widget = PathInputWidget(
            "结果保存路径:",
            "请选择保存处理结果的文件夹"
        )
        paths_layout.addWidget(self.save_path_widget)

        parent_layout.addWidget(paths_group)

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
        self.save_path_widget.browse_requested.connect(self.browse_save_path_requested.emit)
        self.save_path_widget.path_changed.connect(self.save_path_changed.emit)

        # 开始/停止按钮
        self.start_stop_button.clicked.connect(self.toggle_processing_requested.emit)

    def set_processing_state(self, is_processing: bool):
        """设置UI的处理状态"""
        # 禁用或启用设置控件
        self.file_path_widget.set_enabled(not is_processing)
        self.save_path_widget.set_enabled(not is_processing)

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
        return self.save_path_widget.get_path()

    # 设置器方法
    def set_file_path(self, path):
        """设置文件路径"""
        self.file_path_widget.set_path(path)

    def set_save_path(self, path):
        """设置保存路径"""
        self.save_path_widget.set_path(path)

    # 设置和加载方法
    def get_settings(self):
        """获取页面设置"""
        return {
            "file_path": self.get_file_path(),
            "save_path": self.get_save_path(),
        }

    def load_settings(self, settings):
        """加载页面设置"""
        if "file_path" in settings and settings["file_path"] and os.path.exists(settings["file_path"]):
            self.set_file_path(settings["file_path"])

        if "save_path" in settings and settings["save_path"]:
            self.set_save_path(settings["save_path"])

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

        # 更新按钮样式
        self._update_button_style(self.controller.is_processing if hasattr(self.controller, 'is_processing') else False)