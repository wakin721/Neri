import sys
import platform
import logging
from typing import Callable, Optional, List
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QApplication, QGraphicsDropShadowEffect,
    QSizePolicy, QSpacerItem, QCheckBox, QLineEdit, QGroupBox,
    QSlider, QComboBox
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QRect, QSize, Signal, Property, QParallelAnimationGroup, QPoint
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QFont, QFontMetrics,
    QPalette, QLinearGradient, QBrush, QPen
)

logger = logging.getLogger(__name__)


class Win11Colors:
    """Windows 11 设计系统颜色 - 自定义主题色"""
    # 自定义主题色
    CUSTOM_DARK_ACCENT = QColor(93, 58, 79)  # #5d3a4f - 深色模式主色调
    CUSTOM_LIGHT_ACCENT = QColor(219, 188, 194)  # #dbbcc2 - 浅色模式主色调

    # 亮色主题 - 基于自定义颜色调整
    LIGHT_BACKGROUND = QColor(252, 250, 251)  # 调整为与主题色协调
    LIGHT_SURFACE = QColor(247, 243, 244)  # 微调表面色
    LIGHT_CARD = QColor(255, 255, 255)  # 保持纯白
    LIGHT_TEXT_PRIMARY = QColor(32, 31, 30)  # 保持文字可读性
    LIGHT_TEXT_SECONDARY = QColor(96, 94, 92)  # 保持次要文字可读性
    LIGHT_ACCENT = CUSTOM_LIGHT_ACCENT  # 使用自定义浅色主题色
    LIGHT_HOVER = QColor(240, 234, 236)  # 悬停色调整
    LIGHT_BORDER = QColor(226, 216, 219)  # 边框色调整

    # 暗色主题 - 基于自定义颜色调整
    DARK_BACKGROUND = QColor(26, 20, 23)  # 深化背景色以配合主题
    DARK_SURFACE = QColor(38, 28, 32)  # 调整表面色
    DARK_CARD = QColor(48, 36, 41)  # 调整卡片色
    DARK_TEXT_PRIMARY = QColor(255, 255, 255)  # 保持白色文字
    DARK_TEXT_SECONDARY = QColor(200, 190, 194)  # 调整次要文字色
    DARK_ACCENT = CUSTOM_DARK_ACCENT  # 使用自定义深色主题色
    DARK_HOVER = QColor(58, 42, 48)  # 悬停色调整
    DARK_BORDER = QColor(66, 50, 57)  # 边框色调整


class ModernSwitch(QCheckBox):
    """现代化开关组件 - Win11风格"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedSize(50, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 动画属性
        self._animation_duration = 200
        self._track_opacity = 1.0
        self._thumb_position = 0.0

        self._setup_animations()
        self._update_stylesheet()

    def _setup_animations(self):
        """设置动画"""
        self._position_animation = QPropertyAnimation(self, b"thumb_position")
        self._position_animation.setDuration(self._animation_duration)
        self._position_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._opacity_animation = QPropertyAnimation(self, b"track_opacity")
        self._opacity_animation.setDuration(self._animation_duration)
        self._opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    @Property(float)
    def thumb_position(self):
        return self._thumb_position

    @thumb_position.setter
    def thumb_position(self, value):
        self._thumb_position = value
        self.update()

    @Property(float)
    def track_opacity(self):
        return self._track_opacity

    @track_opacity.setter
    def track_opacity(self, value):
        self._track_opacity = value
        self.update()

    def _update_stylesheet(self):
        """更新样式"""
        # 隐藏原始复选框样式
        self.setStyleSheet("""
            QCheckBox {
                background: transparent;
                spacing: 0px;
            }
            QCheckBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)

    def nextCheckState(self):
        """处理状态切换"""
        super().nextCheckState()
        self._animate_to_state()

    def _animate_to_state(self):
        """动画到当前状态"""
        if self.isChecked():
            # 切换到开启状态
            self._position_animation.setStartValue(self._thumb_position)
            self._position_animation.setEndValue(1.0)
        else:
            # 切换到关闭状态
            self._position_animation.setStartValue(self._thumb_position)
            self._position_animation.setEndValue(0.0)

        self._position_animation.start()

    def paintEvent(self, event):
        """自定义绘制"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 获取主题颜色
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            track_color_off = Win11Colors.DARK_BORDER
            track_color_on = Win11Colors.DARK_ACCENT
            thumb_color = QColor(255, 255, 255)
            shadow_color = QColor(0, 0, 0, 0)
        else:
            track_color_off = Win11Colors.LIGHT_BORDER
            track_color_on = Win11Colors.LIGHT_ACCENT
            thumb_color = QColor(255, 255, 255)
            shadow_color = QColor(0, 0, 0, 0)

        # 计算尺寸
        rect = self.rect()
        track_width = rect.width()
        track_height = rect.height()
        track_radius = track_height // 2
        thumb_radius = track_radius - 2

        # 绘制轨道
        track_rect = QRect(0, 0, track_width, track_height)
        track_path = QPainterPath()
        track_path.addRoundedRect(track_rect, track_radius, track_radius)

        # 根据状态混合颜色
        if self.isChecked():
            current_color = track_color_on
        else:
            current_color = track_color_off

        painter.fillPath(track_path, current_color)

        # 计算滑块位置
        thumb_x = int(2 + self._thumb_position * (track_width - 2 * thumb_radius - 4))
        thumb_y = 2

        # 绘制滑块阴影
        shadow_rect = QRect(thumb_x + 1, thumb_y + 1, thumb_radius * 2, thumb_radius * 2)
        shadow_path = QPainterPath()
        shadow_path.addEllipse(shadow_rect)
        painter.fillPath(shadow_path, shadow_color)

        # 绘制滑块
        thumb_rect = QRect(thumb_x, thumb_y, thumb_radius * 2, thumb_radius * 2)
        thumb_path = QPainterPath()
        thumb_path.addEllipse(thumb_rect)
        painter.fillPath(thumb_path, thumb_color)

    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
            self._animate_to_state()
        super().mousePressEvent(event)

    def setChecked(self, checked):
        """设置选中状态"""
        super().setChecked(checked)
        # 直接设置位置，不使用动画
        self._thumb_position = 1.0 if checked else 0.0
        self.update()

    def update_theme(self):
        """更新主题"""
        self._update_stylesheet()
        self.update()


class SwitchRow(QWidget):
    """开关行组件 - 包含标签和开关的完整行"""

    toggled = Signal(bool)  # 开关状态改变信号

    def __init__(self, text: str = "", checked: bool = False, parent: QWidget = None):
        super().__init__(parent)
        self._text = text
        self._checked = checked
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        # 创建标签
        self._label = QLabel(self._text)
        self._label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self._label)

        # 添加弹性空间
        layout.addStretch()

        # 创建开关
        self._switch = ModernSwitch()
        self._switch.setChecked(self._checked)
        self._switch.toggled.connect(self.toggled.emit)
        layout.addWidget(self._switch)

        # 设置透明背景
        self.setStyleSheet("SwitchRow { background-color: transparent; }")

    def setText(self, text: str):
        """设置文本"""
        self._text = text
        self._label.setText(text)

    def text(self) -> str:
        """获取文本"""
        return self._text

    def setChecked(self, checked: bool):
        """设置选中状态"""
        self._checked = checked
        self._switch.setChecked(checked)

    def isChecked(self) -> bool:
        """获取选中状态"""
        return self._switch.isChecked()

    def switch(self) -> ModernSwitch:
        """获取开关组件"""
        return self._switch

    def label(self) -> QLabel:
        """获取标签组件"""
        return self._label

    def setEnabled(self, enabled: bool):
        """设置启用状态"""
        super().setEnabled(enabled)
        self._switch.setEnabled(enabled)
        self._label.setEnabled(enabled)

    def update_theme(self):
        """更新主题"""
        self._switch.update_theme()
        # 更新标签颜色
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
        self._label.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")


class RoundedButton(QPushButton):
    """Win11 风格圆角按钮 - 自定义主题色版本"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self._is_active = False
        self._corner_radius = 6  # Win11 标准圆角
        self._animation_duration = 150
        self._setup_ui()
        self._setup_animations()

    def _setup_ui(self):
        """设置UI样式"""
        self.setMinimumSize(120, 36)
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 设置样式表
        self._update_stylesheet()

        # 添加微妙的阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def _setup_animations(self):
        """设置动画效果"""
        self._hover_animation = QPropertyAnimation(self, b"geometry")
        self._hover_animation.setDuration(self._animation_duration)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _update_stylesheet(self):
        """根据当前状态更新样式表"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            hover_color = Win11Colors.DARK_HOVER
            accent_color = Win11Colors.DARK_ACCENT
        else:
            bg_color = Win11Colors.LIGHT_CARD
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            hover_color = Win11Colors.LIGHT_HOVER
            accent_color = Win11Colors.LIGHT_ACCENT

        active_bg = accent_color if self._is_active else bg_color
        active_text = QColor(255, 255, 255) if self._is_active else text_color

        # 为激活状态计算更好的悬停色
        hover_active_color = accent_color.lighter(120) if self._is_active else hover_color
        pressed_active_color = accent_color.darker(110) if self._is_active else hover_color.darker(105)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {active_bg.name()};
                color: {active_text.name()};
                border: 1px solid {"transparent" if self._is_active else Win11Colors.LIGHT_BORDER.name() if not is_dark else Win11Colors.DARK_BORDER.name()};
                border-radius: {self._corner_radius}px;
                padding: 8px 16px;
                font-weight: {"600" if self._is_active else "400"};
            }}
            QPushButton:hover {{
                background-color: {hover_active_color.name()};
            }}
            QPushButton:pressed {{
                background-color: {pressed_active_color.name()};
            }}
        """)

    def set_active(self, active: bool = True):
        """设置按钮激活状态"""
        if self._is_active != active:
            self._is_active = active
            self._update_stylesheet()

    def is_active(self) -> bool:
        """返回按钮是否处于激活状态"""
        return self._is_active


class ModernFrame(QFrame):
    """现代风格框架 - 自定义主题色版本"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置UI样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD
            border_color = Win11Colors.DARK_BORDER
        else:
            bg_color = Win11Colors.LIGHT_CARD
            border_color = Win11Colors.LIGHT_BORDER

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 8px;
            }}
        """)


class InfoBar(QFrame):
    """信息栏 - 自定义主题色版本，支持 tqdm 风格进度条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.status_label)

        # 添加弹性空间
        layout.addStretch()

        # Tqdm 风格进度条容器（初始隐藏）
        self.progress_container = QWidget()
        self.progress_container.hide()
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)

        # 进度描述标签
        self.progress_desc_label = QLabel("处理进度:")
        self.progress_desc_label.setFont(QFont("Segoe UI", 9))
        progress_layout.addWidget(self.progress_desc_label)

        # 进度百分比标签
        self.progress_percent_label = QLabel("0.00%")  # 修改：默认显示两位小数
        self.progress_percent_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.progress_percent_label.setMinimumWidth(50)  # 修改：增加宽度以容纳小数
        progress_layout.addWidget(self.progress_percent_label)

        # 进度条 - 修改：使用更高精度
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(10000)  # 修改：使用10000以支持0.01%精度
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setMinimumWidth(200)
        progress_layout.addWidget(self.progress_bar)

        # 进度详情标签（n/total [elapsed<remaining, speed]）
        self.progress_detail_label = QLabel("0/0 [00:00<00:00, 0.00张/秒]")
        self.progress_detail_label.setFont(QFont("Consolas", 9))
        self.progress_detail_label.setMinimumWidth(250)
        progress_layout.addWidget(self.progress_detail_label)

        layout.addWidget(self.progress_container)

        # 应用自定义主题样式
        self._apply_style()

    def _apply_style(self):
        """应用样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE
            text_color = Win11Colors.DARK_TEXT_SECONDARY
            accent_color = Win11Colors.DARK_ACCENT
            border_color = Win11Colors.DARK_BORDER
        else:
            bg_color = Win11Colors.LIGHT_SURFACE
            text_color = Win11Colors.LIGHT_TEXT_SECONDARY
            accent_color = Win11Colors.LIGHT_ACCENT
            border_color = Win11Colors.LIGHT_BORDER

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color.name()};
            }}
            QLabel {{
                color: {text_color.name()};
                background-color: {bg_color.name()};
            }}
            QProgressBar {{
                border: 1px solid {border_color.name()};
                border-radius: 4px;
                background-color: {bg_color.lighter(110).name()};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {accent_color.name()};
                border-radius: 3px;
            }}
            QWidget {{
                background-color: {bg_color.name()};
            }}
        """)

    def show_progress(self):
        """显示进度条"""
        self.progress_container.show()

    def hide_progress(self):
        """隐藏进度条"""
        self.progress_container.hide()

    def update_progress(self, current: int, total: int, elapsed_seconds: float,
                        remaining_seconds: float, speed: float):
        """
        更新 tqdm 风格的进度条

        Args:
            current: 当前处理数量
            total: 总数量
            elapsed_seconds: 已用时间(秒)
            remaining_seconds: 剩余时间(秒)
            speed: 处理速度(张/秒)
        """
        # 计算百分比 - 保留两位小数
        if total > 0:
            percentage = (current / total * 100)
        else:
            percentage = 0

        # 更新进度条 - 使用10000作为最大值，实现0.01%的精度
        self.progress_bar.setMaximum(10000)
        progress_value = int(percentage * 100)  # 50.25% -> 5025
        self.progress_bar.setValue(progress_value)

        # 百分比标签显示两位小数
        self.progress_percent_label.setText(f"{percentage:.2f}%")

        # 格式化时间 - 修改这里
        elapsed_str = self._format_time(elapsed_seconds)

        # 判断是否需要显示"计算中..."
        if remaining_seconds <= 0 or remaining_seconds == float('inf') or speed <= 0 or current < 2:
            remaining_str = "计算中..."
        else:
            remaining_str = self._format_time(remaining_seconds)

        # 判断速度是否有效
        if speed <= 0 or current < 2:
            speed_str = "计算中..."
        else:
            speed_str = f"{speed:.2f}张/秒"

        # 更新详情标签
        detail_text = f"{current}/{total} [{elapsed_str}<{remaining_str}, {speed_str}]"
        self.progress_detail_label.setText(detail_text)

    def _format_time(self, seconds: float) -> str:
        """格式化时间为 HH:MM:SS 或 MM:SS"""
        if seconds == float('inf') or seconds < 0:
            return "计算中..."

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def update_theme(self):
        """更新主题"""
        self._apply_style()


class SpeedProgressBar(QFrame):
    """现代化进度条组件 - 内嵌信息版"""

    def __init__(self, parent: QWidget = None, accent_color: QColor = None):
        super().__init__(parent)
        self._progress = 0
        self._total = 100
        self._speed = 0.0
        self._remaining_time = 0
        self._current_file = ""

        # 使用自定义主题色作为默认强调色
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self._accent_color = accent_color or (Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 创建自定义进度条 - 增加高度以容纳更多信息
        self._progress_widget = QWidget()
        self._progress_widget.setFixedHeight(50)  # 增加高度
        self._progress_widget.setStyleSheet("background: transparent;")
        layout.addWidget(self._progress_widget)

        # 设置整体样式
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD
            border_color = Win11Colors.DARK_BORDER
        else:
            bg_color = Win11Colors.LIGHT_CARD
            border_color = Win11Colors.LIGHT_BORDER

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 8px;
            }}
        """)

    def set_current_file(self, filename: str):
        """设置当前处理的文件名"""
        self._current_file = filename
        self.update()

    def paintEvent(self, event):
        """自定义绘制"""
        super().paintEvent(event)

        if not self.isVisible():
            return

        painter = QPainter(self._progress_widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self._progress_widget.rect()

        # 获取当前主题
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            progress_text_color = QColor(255, 255, 255)  # 进度条内文字为白色
        else:
            bg_color = Win11Colors.LIGHT_SURFACE
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            progress_text_color = QColor(255, 255, 255)  # 进度条内文字为白色

        # 绘制背景（带圆角）
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        painter.fillPath(path, bg_color)

        # 绘制进度（带圆角）
        progress_ratio = self._progress / max(self._total, 1)
        progress_width = int(rect.width() * progress_ratio)

        if progress_width > 4:  # 只有当进度条有一定宽度时才绘制
            progress_rect = QRect(0, 0, progress_width, rect.height())
            progress_path = QPainterPath()
            progress_path.addRoundedRect(progress_rect, 6, 6)
            painter.fillPath(progress_path, self._accent_color)

        # 绘制文本信息
        self._draw_progress_text(painter, rect, progress_width, text_color, progress_text_color)

    def _draw_progress_text(self, painter, rect, progress_width, text_color, progress_text_color):
        """绘制进度条内的文本信息"""
        # 计算百分比
        percentage = (self._progress / max(self._total, 1)) * 100

        # 第一行：百分比和文件计数
        main_text = f"{percentage:.1f}% ({self._progress}/{self._total})"

        # 第二行：当前文件名（截断过长的文件名）
        file_text = ""
        if self._current_file:
            file_text = self._current_file
            # 如果文件名太长，截断显示
            if len(file_text) > 40:
                file_text = file_text[:37] + "..."

        # 第三行：速度和剩余时间
        stats_text = self._format_stats_text()

        # 设置字体
        main_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        file_font = QFont("Segoe UI", 9)
        stats_font = QFont("Segoe UI", 8)

        # 计算文本位置
        y_offset = rect.height() // 2

        # 绘制主要进度信息（居中）
        painter.setFont(main_font)
        main_rect = QRect(rect.x(), rect.y() + y_offset - 20, rect.width(), 20)

        # 绘制进度条外的文字（深色）
        painter.setPen(QPen(text_color))
        painter.drawText(main_rect, Qt.AlignmentFlag.AlignCenter, main_text)

        # 绘制进度条内的文字（浅色）- 使用裁剪
        if progress_width > 50:  # 只有进度条足够宽时才显示内部文字
            painter.save()
            clip_rect = QRect(0, 0, progress_width, rect.height())
            painter.setClipRect(clip_rect)
            painter.setPen(QPen(progress_text_color))
            painter.drawText(main_rect, Qt.AlignmentFlag.AlignCenter, main_text)
            painter.restore()

        # 绘制文件名（较小字体，居中偏上）
        if file_text:
            painter.setFont(file_font)
            file_rect = QRect(rect.x() + 10, rect.y() + y_offset - 5, rect.width() - 20, 15)

            # 进度条外的文件名
            painter.setPen(QPen(text_color))
            painter.drawText(file_rect, Qt.AlignmentFlag.AlignCenter, file_text)

            # 进度条内的文件名
            if progress_width > 100:
                painter.save()
                clip_rect = QRect(0, 0, progress_width, rect.height())
                painter.setClipRect(clip_rect)
                painter.setPen(QPen(progress_text_color))
                painter.drawText(file_rect, Qt.AlignmentFlag.AlignCenter, file_text)
                painter.restore()

        # 绘制统计信息（较小字体，居中偏下）
        if stats_text:
            painter.setFont(stats_font)
            stats_rect = QRect(rect.x() + 10, rect.y() + y_offset + 10, rect.width() - 20, 12)

            # 进度条外的统计信息
            painter.setPen(QPen(text_color.darker(120)))
            painter.drawText(stats_rect, Qt.AlignmentFlag.AlignCenter, stats_text)

            # 进度条内的统计信息
            if progress_width > 150:
                painter.save()
                clip_rect = QRect(0, 0, progress_width, rect.height())
                painter.setClipRect(clip_rect)
                painter.setPen(QPen(progress_text_color.darker(120)))
                painter.drawText(stats_rect, Qt.AlignmentFlag.AlignCenter, stats_text)
                painter.restore()

    def _format_stats_text(self):
        """格式化统计文本"""
        speed_text = f"{self._speed:.2f} 张/秒" if self._speed > 0 else "计算中..."

        if isinstance(self._remaining_time, (int, float)) and self._remaining_time > 0:
            if self._remaining_time == float('inf') or self._remaining_time > 3600 * 24:
                time_text = "计算中..."
            elif self._remaining_time > 3600:
                hours = int(self._remaining_time // 3600)
                minutes = int((self._remaining_time % 3600) // 60)
                time_text = f"{hours}h{minutes}m"
            elif self._remaining_time > 60:
                minutes = int(self._remaining_time // 60)
                seconds = int(self._remaining_time % 60)
                time_text = f"{minutes}m{seconds}s"
            else:
                time_text = f"{int(self._remaining_time)}s"
        else:
            time_text = "计算中..."

        return f"{speed_text} | 剩余 {time_text}"

    def update_progress(self, value: int, total: int = None, speed: float = None, remaining_time=None,
                        current_file: str = None):
        """更新进度"""
        self._progress = value
        if total is not None:
            self._total = max(1, total)

        if speed is not None:
            self._speed = speed

        if remaining_time is not None:
            self._remaining_time = remaining_time

        if current_file is not None:
            self._current_file = current_file

        self.update()


class CollapsiblePanel(QFrame):
    """现代化可折叠面板组件 - 自定义主题色版本"""

    toggled = Signal(bool)  # 折叠状态改变信号

    def __init__(self, parent: QWidget = None, title: str = "", subtitle: str = "", icon: str = None):
        super().__init__(parent)
        self._is_expanded = False
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._animation_duration = 200
        self._is_animating = False
        self._setup_ui()
        self._setup_animations()

    def _setup_ui(self):
        """设置UI"""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # 创建头部
        self._create_header()

        # 创建内容区域
        self._content_frame = QFrame()
        self._content_frame.setFixedHeight(0)  # 初始高度为0
        self._content_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # 设置内容区域无边框但有背景色
        self._content_frame.setFrameStyle(QFrame.Shape.NoFrame)

        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(20, 10, 20, 20)

        self._main_layout.addWidget(self._content_frame)

        # 应用样式
        self._update_styles()

    def _create_header(self):
        """创建头部区域"""
        self._header_frame = QFrame()
        self._header_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header_frame.mousePressEvent = self._on_header_clicked

        header_layout = QHBoxLayout(self._header_frame)
        header_layout.setContentsMargins(15, 12, 15, 12)

        # 图标
        if self._icon:
            self._icon_label = QLabel(self._icon)
            font_name = "Segoe UI Emoji" if platform.system() == "Windows" else "Apple Color Emoji"
            self._icon_label.setFont(QFont(font_name, 16))
            header_layout.addWidget(self._icon_label)

        # 标题容器
        title_container = QVBoxLayout()

        self._title_label = QLabel(self._title)
        self._title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        title_container.addWidget(self._title_label)

        if self._subtitle:
            self._subtitle_label = QLabel(self._subtitle)
            self._subtitle_label.setFont(QFont("Segoe UI", 9))
            title_container.addWidget(self._subtitle_label)

        header_layout.addLayout(title_container)
        header_layout.addStretch()

        # 展开/收起指示器
        self._toggle_indicator = QLabel("▼")
        self._toggle_indicator.setFont(QFont("Segoe UI", 12))
        header_layout.addWidget(self._toggle_indicator)

        self._main_layout.addWidget(self._header_frame)

    def _setup_animations(self):
        """设置动画"""
        self._height_animation = QPropertyAnimation(self._content_frame, b"maximumHeight")
        self._height_animation.setDuration(self._animation_duration)
        self._height_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 连接动画完成信号
        self._height_animation.finished.connect(self._on_animation_finished)

    def _on_animation_finished(self):
        """动画完成回调"""
        self._is_animating = False

        if not self._is_expanded:
            # 折叠完成后立即设置最终状态
            self._content_frame.setFixedHeight(0)
            self._content_frame.setMaximumHeight(0)
            # 立即更新样式
            self._update_header_style()
        else:
            # 展开完成，设置为自动高度
            self._content_frame.setMaximumHeight(16777215)

    def expand(self):
        """展开面板"""
        if self._is_expanded or self._is_animating:
            return

        self._is_expanded = True
        self._is_animating = True

        # 先更新头部样式为展开状态
        self._update_header_style()

        # 计算目标高度
        self._content_frame.setMaximumHeight(16777215)
        self._content_frame.adjustSize()
        target_height = self._content_frame.sizeHint().height()

        # 设置起始状态
        self._content_frame.setFixedHeight(0)
        self._content_frame.setMaximumHeight(0)

        # 启动动画
        self._height_animation.setStartValue(0)
        self._height_animation.setEndValue(target_height)
        self._height_animation.start()

        # 更新指示器
        self._toggle_indicator.setText("▲")
        self._update_indicator_style()

        # 立即更新内容区域样式
        self._update_content_style()

        self.toggled.emit(True)

    def collapse(self):
        """收起面板"""
        if not self._is_expanded or self._is_animating:
            return

        self._is_expanded = False
        self._is_animating = True

        # 获取当前高度
        current_height = self._content_frame.height()
        if current_height <= 0:
            current_height = self._content_frame.sizeHint().height()
            self._content_frame.setFixedHeight(current_height)

        # 启动动画
        self._height_animation.setStartValue(current_height)
        self._height_animation.setEndValue(0)
        self._height_animation.start()

        # 更新指示器
        self._toggle_indicator.setText("▼")
        self._update_indicator_style()

        self.toggled.emit(False)

    def toggle(self):
        """切换展开/收起状态"""
        # 防止动画期间的重复操作
        if self._is_animating:
            return

        if self._is_expanded:
            self.collapse()
        else:
            self.expand()

    def _on_header_clicked(self, event):
        """头部点击事件"""
        self.toggle()

    def _update_header_style(self):
        """更新头部样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            header_bg = Win11Colors.DARK_SURFACE
            hover_color = Win11Colors.DARK_HOVER
        else:
            header_bg = Win11Colors.LIGHT_SURFACE
            hover_color = Win11Colors.LIGHT_HOVER

        # 根据展开状态设置圆角
        border_radius = "8px 8px 0px 0px" if self._is_expanded else "8px 8px 8px 8px"

        self._header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {header_bg.name()};
                border: none;
                border-radius: {border_radius};
            }}
            QFrame:hover {{
                background-color: {hover_color.name()};
            }}
        """)

    def _update_content_style(self):
        """更新内容区域样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        content_bg = Win11Colors.DARK_CARD if is_dark else Win11Colors.LIGHT_CARD
        header_bg = Win11Colors.DARK_SURFACE if is_dark else Win11Colors.LIGHT_SURFACE
        border_color = Win11Colors.DARK_BORDER if is_dark else Win11Colors.LIGHT_BORDER
        hover_color = Win11Colors.DARK_HOVER if is_dark else Win11Colors.LIGHT_HOVER
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY

        if self._is_expanded:
            self._content_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {content_bg.name()};
                    border: none;
                    border-radius: 0px 0px 8px 8px;
                }}
                QWidget {{
                    background-color: transparent;
                }}
                QLabel {{
                    background-color: transparent;
                    color: {text_color.name()};
                }}
                QLineEdit {{
                    background-color: {content_bg.lighter(105).name()};
                    border: 1px solid {border_color.name()};
                    border-radius: 4px;
                    padding: 6px;
                    color: {text_color.name()};
                }}
                QLineEdit:focus {{
                    border-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                }}
                QSlider {{
                    background-color: transparent;
                }}
                QCheckBox {{
                    background-color: transparent;
                    color: {text_color.name()};
                }}
                ModernSwitch {{
                    background-color: transparent;
                }}
                QComboBox {{
                    background-color: {content_bg.lighter(105).name()};
                    border: 1px solid {border_color.name()};
                    border-radius: 4px;
                    padding: 6px 8px;
                    color: {text_color.name()};
                    min-height: 20px;
                }}
                QComboBox:hover {{
                    background-color: {hover_color.name()};
                    border-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                }}
                QComboBox:focus {{
                    border-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                }}
                QComboBox::drop-down {{
                    width: 0px;
                    border: none;
                    background-color: transparent;
                }}
                QComboBox::down-arrow {{
                    width: 0px;
                    height: 0px;
                    border: none;
                    background: transparent;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {content_bg.name()};
                    border: 1px solid {border_color.name()};
                    border-radius: 4px;
                    selection-background-color: {hover_color.name()};
                    selection-color: {text_color.name()};
                    color: {text_color.name()};
                    outline: none;
                }}
                QComboBox QAbstractItemView::item {{
                    padding: 8px 12px;
                    background-color: {content_bg.name()};
                    color: {text_color.name()};
                    border: none;
                }}
                QComboBox QAbstractItemView::item:hover {{
                    background-color: {hover_color.name()};
                }}
                QComboBox QAbstractItemView::item:selected {{
                    background-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                    color: {"white" if is_dark else "white"};
                }}
                QPushButton {{
                    background-color: {header_bg.name()};
                    border: 1px solid {border_color.name()};
                    border-radius: 4px;
                    padding: 6px 12px;
                    color: {text_color.name()};
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {hover_color.name()};
                    border-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                }}
                QPushButton:pressed {{
                    background-color: {Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()};
                    color: white;
                }}
            """)

    def _update_indicator_style(self):
        """更新指示器样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        indicator_color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT
        self._toggle_indicator.setStyleSheet(f"color: {indicator_color.name()}; background-color: transparent;")

    def _update_styles(self):
        """更新样式 - 使用自定义主题色"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            content_bg = Win11Colors.DARK_CARD
            text_primary = Win11Colors.DARK_TEXT_PRIMARY
            text_secondary = Win11Colors.DARK_TEXT_SECONDARY
            border_color = Win11Colors.DARK_BORDER
        else:
            content_bg = Win11Colors.LIGHT_CARD
            text_primary = Win11Colors.LIGHT_TEXT_PRIMARY
            text_secondary = Win11Colors.LIGHT_TEXT_SECONDARY
            border_color = Win11Colors.LIGHT_BORDER

        # 主容器样式
        self.setStyleSheet(f"""
            CollapsiblePanel {{
                border: 1px solid {border_color.name()};
                border-radius: 8px;
            }}
        """)

        # 更新各部分样式
        self._update_header_style()
        if self._is_expanded:
            self._update_content_style()
        self._update_indicator_style()

        # 更新文字标签样式
        self._title_label.setStyleSheet(f"color: {text_primary.name()}; background-color: transparent;")
        if hasattr(self, '_subtitle_label'):
            self._subtitle_label.setStyleSheet(f"color: {text_secondary.name()}; background-color: transparent;")

        # 更新图标标签
        if hasattr(self, '_icon_label'):
            self._icon_label.setStyleSheet("QLabel { background-color: transparent; }")

        # 递归更新所有子组件的背景
        if self._is_expanded:
            self._update_child_widgets_background(content_bg)

    def _update_child_widgets_background(self, content_bg):
        """递归更新所有子组件的背景色"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        border_color = Win11Colors.DARK_BORDER if is_dark else Win11Colors.LIGHT_BORDER
        hover_color = Win11Colors.DARK_HOVER if is_dark else Win11Colors.LIGHT_HOVER
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
        accent_color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT

        def update_widget_recursive(widget):
            if widget:
                class_name = widget.__class__.__name__

                if class_name in ['QWidget', 'QFrame']:
                    # 容器组件设置透明背景
                    current_style = widget.styleSheet()
                    if 'background-color' not in current_style:
                        widget.setStyleSheet(f"{current_style}background-color: transparent;")

                elif class_name == 'QLabel':
                    # 标签组件透明背景
                    current_style = widget.styleSheet()
                    if 'background-color' not in current_style:
                        widget.setStyleSheet(
                            f"{current_style}background-color: transparent; color: {text_color.name()};")

                elif class_name == 'SwitchRow':
                    # 开关行组件
                    widget.setStyleSheet("SwitchRow { background-color: transparent; }")
                    widget.update_theme()

                elif class_name == 'ModernSwitch':
                    # 开关组件保持透明背景，让自定义绘制生效
                    widget.setStyleSheet("ModernSwitch { background-color: transparent; }")
                    # 更新开关的主题
                    widget.update_theme()


                elif class_name == 'QComboBox':

                    # 下拉菜单需要特殊处理，确保下拉列表有正确的背景

                    widget.setStyleSheet(f"""

                        QComboBox {{

                            background-color: {content_bg.lighter(105).name()};

                            border: 1px solid {border_color.name()};

                            border-radius: 4px;

                            padding: 6px 8px;

                            color: {text_color.name()};

                            min-height: 20px;

                        }}

                        QComboBox:hover {{

                            background-color: {hover_color.name()};

                            border-color: {accent_color.name()};

                        }}

                        QComboBox:focus {{

                            border-color: {accent_color.name()};

                        }}

                        QComboBox::drop-down {{

                            width: 0px;

                            border: none;

                            background-color: transparent;

                        }}

                        QComboBox::down-arrow {{

                            width: 0px;

                            height: 0px;

                            border: none;

                            background: transparent;

                        }}

                        QComboBox QAbstractItemView {{

                            background-color: {content_bg.name()};

                            border: 1px solid {border_color.name()};

                            border-radius: 4px;

                            selection-background-color: {hover_color.name()};

                            selection-color: {text_color.name()};

                            color: {text_color.name()};

                            outline: none;

                            show-decoration-selected: 1;

                        }}

                        QComboBox QAbstractItemView::item {{

                            padding: 8px 12px;

                            background-color: transparent;

                            color: {text_color.name()};

                            border: none;

                            min-height: 20px;

                        }}

                        QComboBox QAbstractItemView::item:hover {{

                            background-color: {hover_color.name()};

                        }}

                        QComboBox QAbstractItemView::item:selected {{

                            background-color: {accent_color.name()};

                            color: white;

                        }}

                    """)

                elif class_name == 'QLineEdit':

                    # 文本输入框处理...

                    widget.setStyleSheet(f"""

                                    QLineEdit {{

                                        background-color: {content_bg.lighter(105).name()};

                                        border: 1px solid {border_color.name()};

                                        border-radius: 4px;

                                        padding: 6px;

                                        color: {text_color.name()};

                                    }}

                                    QLineEdit:focus {{

                                        border-color: {accent_color.name()};

                                    }}

                                """)


                elif class_name in ['QSlider', 'QCheckBox']:

                    # 传统复选框和滑块保持透明背景

                    current_style = widget.styleSheet()

                    if 'background-color' not in current_style:
                        widget.setStyleSheet(

                            f"{current_style}background-color: transparent; color: {text_color.name()};")


                elif class_name == 'QPushButton':

                    # 按钮组件处理...

                    current_style = widget.styleSheet()

                    if 'background-color' not in current_style:
                        widget.setStyleSheet(f"""

                                        QPushButton {{

                                            background-color: {content_bg.lighter(110).name()};

                                            border: 1px solid {border_color.name()};

                                            border-radius: 4px;

                                            padding: 6px 12px;

                                            color: {text_color.name()};

                                            font-weight: 500;

                                        }}

                                        QPushButton:hover {{

                                            background-color: {hover_color.name()};

                                            border-color: {accent_color.name()};

                                        }}

                                        QPushButton:pressed {{

                                            background-color: {accent_color.name()};

                                            color: white;

                                        }}

                                    """)

                # 递归处理子组件
                for child in widget.findChildren(QWidget):
                    update_widget_recursive(child)

        # 从内容区域开始递归更新
        update_widget_recursive(self._content_frame)

    def content_widget(self) -> QWidget:
        """获取内容区域Widget，用于添加子控件"""
        content_widget = QWidget()
        self._content_layout.addWidget(content_widget)
        return content_widget

    def add_content_widget(self, widget: QWidget):
        """添加内容控件"""
        self._content_layout.addWidget(widget)

        # 立即为新添加的组件设置背景色
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        content_bg = Win11Colors.DARK_CARD if is_dark else Win11Colors.LIGHT_CARD

        # 为新组件设置透明背景
        widget.setStyleSheet(f"{widget.styleSheet()}; background-color: transparent;")

        # 为新组件的子控件也设置合适的背景
        self._setup_widget_background(widget, content_bg)

    def _setup_widget_background(self, widget, content_bg):
        """为单个组件及其子组件设置背景"""

        def setup_recursive(w):
            if w:
                class_name = w.__class__.__name__
                current_style = w.styleSheet()

                if class_name in ['QWidget', 'QFrame', 'QLabel']:
                    if 'background-color' not in current_style:
                        w.setStyleSheet(f"{current_style}; background-color: transparent;")
                elif class_name in ['QLineEdit', 'QComboBox']:
                    if 'background-color' not in current_style:
                        w.setStyleSheet(f"{current_style}; background-color: {content_bg.name()};")

                for child in w.findChildren(QWidget):
                    setup_recursive(child)

        setup_recursive(widget)

    def is_expanded(self) -> bool:
        """返回是否已展开"""
        return self._is_expanded

    def update_theme(self):
        """更新主题"""
        self._update_styles()


class ThemeManager:
    """主题管理器，用于统一管理自定义Win11样式"""

    @staticmethod
    def apply_win11_style(app: QApplication):
        """应用自定义Win11样式到整个应用程序"""
        # 检测系统主题
        palette = app.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            ThemeManager._apply_dark_theme(app)
        else:
            ThemeManager._apply_light_theme(app)

    @staticmethod
    def _apply_light_theme(app: QApplication):
        """应用自定义亮色主题"""
        style = f"""
        QWidget {{
            background-color: {Win11Colors.LIGHT_BACKGROUND.name()};
            color: {Win11Colors.LIGHT_TEXT_PRIMARY.name()};
            font-family: "Segoe UI";
        }}

        QFrame {{
            border: none;
        }}

        QScrollBar:vertical {{
            background: transparent;
            width: 12px;
            border-radius: 6px;
        }}

        QScrollBar::handle:vertical {{
            background: {Win11Colors.LIGHT_BORDER.name()};
            border-radius: 6px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {Win11Colors.LIGHT_ACCENT.name()};
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}

        /* 自定义滚动条样式 */
        QScrollArea {{
            border: none;
        }}
        """
        app.setStyleSheet(style)

    @staticmethod
    def _apply_dark_theme(app: QApplication):
        """应用自定义暗色主题"""
        style = f"""
        QWidget {{
            background-color: {Win11Colors.DARK_BACKGROUND.name()};
            color: {Win11Colors.DARK_TEXT_PRIMARY.name()};
            font-family: "Segoe UI";
        }}

        QFrame {{
            border: none;
        }}

        QScrollBar:vertical {{
            background: transparent;
            width: 12px;
            border-radius: 6px;
        }}

        QScrollBar::handle:vertical {{
            background: {Win11Colors.DARK_BORDER.name()};
            border-radius: 6px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {Win11Colors.DARK_ACCENT.name()};
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}

        /* 自定义滚动条样式 */
        QScrollArea {{
            border: none;
        }}
        """
        app.setStyleSheet(style)


class ModernGroupBox(QGroupBox):
    """现代化的分组框，使用Win11风格"""

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self._setup_style()

    def _setup_style(self):
        """设置Win11风格样式"""
        # 检测当前主题
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            border_color = Win11Colors.DARK_BORDER
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            title_bg_color = Win11Colors.DARK_SURFACE
        else:
            border_color = Win11Colors.LIGHT_BORDER
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            title_bg_color = Win11Colors.LIGHT_SURFACE

        self.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 600;
                font-size: 14px;
                color: {text_color.name()};
                border: 2px solid {border_color.name()};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 24px;
                background-color: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 6px 12px 6px 12px;
                background-color: {title_bg_color.name()};
                border: 2px solid {border_color.name()};
                border-radius: 6px;
                color: {text_color.name()};
            }}
        """)


class ModernLineEdit(QLineEdit):
    """现代化的输入框，使用Win11风格"""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        if placeholder:
            self.setPlaceholderText(placeholder)
        self._setup_style()

    def _setup_style(self):
        """设置Win11风格样式"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_CARD
            border_color = Win11Colors.DARK_BORDER
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            focus_color = Win11Colors.DARK_ACCENT
            placeholder_color = Win11Colors.DARK_TEXT_SECONDARY
        else:
            bg_color = Win11Colors.LIGHT_CARD
            border_color = Win11Colors.LIGHT_BORDER
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            focus_color = Win11Colors.LIGHT_ACCENT
            placeholder_color = Win11Colors.LIGHT_TEXT_SECONDARY

        self.setStyleSheet(f"""
            QLineEdit {{
                padding: 8px 12px;
                border: 2px solid {border_color.name()};
                border-radius: 6px;
                background-color: {bg_color.name()};
                color: {text_color.name()};
                font-size: 14px;
                selection-background-color: {focus_color.name()};
            }}
            QLineEdit:focus {{
                border-color: {focus_color.name()};
            }}
            QLineEdit:disabled {{
                background-color: {border_color.name()};
                color: {placeholder_color.name()};
            }}
            QLineEdit::placeholder {{
                color: {placeholder_color.name()};
            }}
        """)

    def update_theme(self):
        """更新主题"""
        self._setup_style()


class PathInputWidget(QWidget):
    """路径输入组件"""

    path_changed = Signal(str)
    browse_requested = Signal()

    def __init__(self, label_text="", placeholder="", parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self._setup_ui(placeholder)
        self._setup_connections()
        self._setup_style()

    def _setup_style(self):
        """设置组件背景样式"""
        # 确保组件背景透明，继承父组件背景
        self.setStyleSheet("QWidget { background-color: transparent; }")

    def _setup_ui(self, placeholder):
        """设置UI"""
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
        from PySide6.QtGui import QFont

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 标签
        if self.label_text:
            label = QLabel(self.label_text)
            label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))

            # 设置标签颜色和背景
            palette = self.palette()
            is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
            text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
            label.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")

            layout.addWidget(label)

        # 输入框和按钮
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.line_edit = ModernLineEdit(placeholder)
        input_layout.addWidget(self.line_edit, 1)

        self.browse_button = RoundedButton("浏览")
        self.browse_button.setMinimumWidth(80)
        self.browse_button.setMaximumWidth(80)
        input_layout.addWidget(self.browse_button)

        layout.addLayout(input_layout)

    def _setup_connections(self):
        """设置信号连接"""
        self.line_edit.textChanged.connect(self.path_changed.emit)
        self.browse_button.clicked.connect(self.browse_requested.emit)

    def get_path(self):
        """获取路径"""
        return self.line_edit.text().strip()

    def set_path(self, path):
        """设置路径"""
        self.line_edit.setText(path)

    def set_enabled(self, enabled):
        """设置启用状态"""
        self.line_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def update_theme(self):
        """更新主题"""
        self._setup_style()
        if hasattr(self.line_edit, '_setup_style'):
            self.line_edit._setup_style()

        # 更新标签颜色
        from PySide6.QtWidgets import QLabel
        for label in self.findChildren(QLabel):
            if label.text() == self.label_text:
                palette = self.palette()
                is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
                text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
                label.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")


class ModernSlider(QSlider):
    """现代化滑块组件 - WinUI 3 风格，带悬停和拖动动画"""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMinimumHeight(40)
        self.setFixedHeight(40)

        # 状态变量
        self._is_dragging = False
        self._is_hovering = False

        # 尺寸定义
        self._track_height = 4
        self._thumb_radius = 10  # 外圈半径
        self._thumb_border_width = 2

        # 根据新要求调整内圈动画半径
        self._inner_radius_default = 6  # 正常状态
        self._inner_radius_hover = 8  # 悬停状态
        self._inner_radius_pressed = 5  # 拖动状态

        self._animated_inner_radius = self._inner_radius_default

        self._setup_style()
        self._setup_animation()

    @Property(float)
    def innerRadius(self):
        return self._animated_inner_radius

    @innerRadius.setter
    def innerRadius(self, value):
        self._animated_inner_radius = value
        self.update()

    def _setup_animation(self):
        """设置内圈半径动画"""
        self._radius_animation = QPropertyAnimation(self, b"innerRadius")
        # 统一使用更快的动画时长，让所有状态反馈都更“清脆”
        self._radius_animation.setDuration(150)
        # OutCubic 曲线（先快后慢）比 InOut 曲线更适合提供快速的交互反馈
        self._radius_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _setup_style(self):
        """设置基础样式 - 隐藏默认外观"""
        # 隐藏原始的QSlider样式，以便完全自定义绘制
        self.setStyleSheet("QSlider { background: transparent; border: none; }")

    def paintEvent(self, event):
        """自定义绘制"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 获取当前主题颜色 - 修正：使用 QApplication 的全局 palette
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            track_bg = Win11Colors.DARK_SURFACE
            progress_color = Win11Colors.DARK_ACCENT
            thumb_color = QColor(255, 255, 255)
            thumb_border = Win11Colors.DARK_BORDER
        else:
            # 为浅色模式添加颜色定义
            track_bg = Win11Colors.LIGHT_SURFACE
            progress_color = Win11Colors.LIGHT_ACCENT
            thumb_color = QColor(255, 255, 255)
            thumb_border = Win11Colors.LIGHT_BORDER

        # --- 1. 绘制轨道 ---
        rect = self.rect()
        track_margin = self._thumb_radius
        track_rect = QRect(
            track_margin,
            (rect.height() - self._track_height) // 2,
            rect.width() - 2 * track_margin,
            self._track_height
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(track_bg))
        painter.drawRoundedRect(track_rect, self._track_height / 2, self._track_height / 2)

        # --- 2. 绘制进度 ---
        if self.maximum() > self.minimum():
            ratio = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
        else:
            ratio = 0

        if ratio > 0:
            progress_width = int(track_rect.width() * ratio)
            progress_rect = QRect(
                track_rect.x(), track_rect.y(),
                progress_width, track_rect.height()
            )
            painter.setBrush(QBrush(progress_color))
            painter.drawRoundedRect(progress_rect, self._track_height / 2, self._track_height / 2)

        # --- 3. 绘制滑块 ---
        thumb_x = track_margin + ratio * track_rect.width()
        thumb_y = rect.height() / 2
        thumb_center_point = QPoint(int(thumb_x), int(thumb_y))

        # 绘制外圈（边框和背景）
        painter.setPen(QPen(thumb_border, 1))
        painter.setBrush(QBrush(thumb_color))
        painter.drawEllipse(thumb_center_point, self._thumb_radius, self._thumb_radius)

        # 绘制内圈（根据状态变化）
        current_inner_radius = self._animated_inner_radius
        if self._is_dragging:
            current_inner_radius = self._inner_radius_pressed  # 拖动时，使用最小半径

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(progress_color))
        painter.drawEllipse(thumb_center_point, current_inner_radius, current_inner_radius)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 1. 标记拖动状态已开始
            self._is_dragging = True

            # 2. 触发“按下”动画，此时滑块的值和位置不发生改变
            self._animate_radius_to(self._inner_radius_pressed)

            # 3. 接受事件，阻止 QSlider 的默认点击行为（即立即跳转到点击位置）
            event.accept()
        else:
            # 对于其他鼠标按钮，继续使用父类的默认行为
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self._is_dragging:
            self._update_value_from_position(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            # 检查鼠标是否仍在组件上，以决定恢复到悬停还是默认状态
            if self.underMouse():
                self._is_hovering = True
                self._animate_radius_to(self._inner_radius_hover)
            else:
                self._is_hovering = False
                self._animate_radius_to(self._inner_radius_default)
            # update() 会在动画过程中被调用，这里不需要手动调用
        super().mouseReleaseEvent(event)


    def enterEvent(self, event):
        """鼠标进入事件"""
        super().enterEvent(event) # 调用父类实现
        self._is_hovering = True
        # 仅当没有在拖动时，才触发悬停动画
        if not self._is_dragging:
            self._animate_radius_to(self._inner_radius_hover)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        super().leaveEvent(event) # 调用父类实现
        self._is_hovering = False
        # 仅当没有在拖动时，才触发离开动画
        if not self._is_dragging:
            self._animate_radius_to(self._inner_radius_default)

    def _animate_radius_to(self, target_radius):
        """启动半径动画到目标值"""
        if self._radius_animation.state() == QPropertyAnimation.State.Running:
            self._radius_animation.stop()

        self._radius_animation.setStartValue(self._animated_inner_radius)
        self._radius_animation.setEndValue(target_radius)
        self._radius_animation.start()

    def _update_value_from_position(self, x):
        """根据鼠标位置更新值"""
        track_width = self.width() - 2 * self._thumb_radius
        relative_x = x - self._thumb_radius

        if track_width > 0:
            ratio = max(0, min(1, relative_x / track_width))
            new_value = self.minimum() + ratio * (self.maximum() - self.minimum())
            self.setValue(int(new_value))

    def update_theme(self):
        """更新主题"""
        self._setup_style()
        self.update()


class ModernComboBox(QComboBox):
    """现代化下拉框组件 - Win11风格"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_style()

    def _setup_style(self):
        """设置Win11风格样式"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE
            border_color = Win11Colors.DARK_BORDER
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            focus_color = Win11Colors.DARK_ACCENT
            hover_color = Win11Colors.DARK_HOVER
        else:
            bg_color = Win11Colors.LIGHT_CARD
            border_color = Win11Colors.LIGHT_BORDER
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            focus_color = Win11Colors.LIGHT_ACCENT
            hover_color = Win11Colors.LIGHT_HOVER

        self.setStyleSheet(f"""
            QComboBox {{
                border: 2px solid {border_color.name()};
                border-radius: 6px;
                padding: 6px 12px;
                background-color: {bg_color.name()};
                color: {text_color.name()};
                min-width: 100px;
                font-size: 14px;
            }}
            QComboBox:hover {{
                border-color: {focus_color.name()};
                background-color: {hover_color.name()};
            }}
            QComboBox:focus {{
                border-color: {focus_color.name()};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 6px;
                selection-background-color: {focus_color.name()};
            }}
        """)

    def update_theme(self):
        """更新主题"""
        self._setup_style()


class ModernCheckBox(QCheckBox):
    """现代化复选框 - WinUI 3 风格"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(28)

        self._bg_color = QColor("transparent")
        self._check_opacity = 0.0  # 0.0 to 1.0

        # 动画属性
        self._animation_duration = 180
        self._bg_color_anim = QPropertyAnimation(self, b"backgroundColor")
        self._check_opacity_anim = QPropertyAnimation(self, b"checkOpacity")

        self._bg_color = QColor("transparent")
        self._check_opacity = 0.0  # 0.0 to 1.0

        self._setup_animations()
        self._update_stylesheet()

        self.stateChanged.connect(self._animate_state_change)

    def _setup_animations(self):
        """设置动画"""
        # 背景颜色动画
        self._bg_color_anim.setDuration(self._animation_duration)
        self._bg_color_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 勾选标记透明度动画
        self._check_opacity_anim.setDuration(self._animation_duration)
        self._check_opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    @Property(QColor)
    def backgroundColor(self):
        return self._bg_color

    @backgroundColor.setter
    def backgroundColor(self, color):
        self._bg_color = color
        self.update()

    @Property(float)
    def checkOpacity(self):
        return self._check_opacity

    @checkOpacity.setter
    def checkOpacity(self, opacity):
        self._check_opacity = opacity
        self.update()

    def _update_stylesheet(self):
        """更新样式 - 隐藏默认指示器"""
        self.setStyleSheet("""
            QCheckBox {
                spacing: 8px; /* 文本和复选框之间的间距 */
                background: transparent;
            }
            QCheckBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)

    def _get_theme_colors(self):
        """获取当前主题的颜色"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            return {
                "border": Win11Colors.DARK_BORDER,
                "hover_bg": QColor(255, 255, 255, 15),
                "checked_bg": Win11Colors.DARK_ACCENT,
                "checked_hover_bg": Win11Colors.DARK_ACCENT.lighter(120),
                "text": Win11Colors.DARK_TEXT_PRIMARY,
                "checkmark": QColor(255, 255, 255)
            }
        else:
            return {
                "border": Win11Colors.LIGHT_BORDER,
                "hover_bg": QColor(0, 0, 0, 10),
                "checked_bg": Win11Colors.LIGHT_ACCENT,
                "checked_hover_bg": Win11Colors.LIGHT_ACCENT.darker(105),
                "text": Win11Colors.LIGHT_TEXT_PRIMARY,
                "checkmark": QColor(255, 255, 255)
            }

    def _animate_state_change(self, state):
        """根据状态启动动画"""
        colors = self._get_theme_colors()

        if state == Qt.CheckState.Checked.value:
            # 勾选动画
            self._bg_color_anim.setStartValue(self.backgroundColor)
            self._bg_color_anim.setEndValue(colors["checked_bg"])
            self._check_opacity_anim.setStartValue(self._check_opacity)
            self._check_opacity_anim.setEndValue(1.0)
        else:
            # 取消勾选动画
            self._bg_color_anim.setStartValue(self.backgroundColor)
            # 目标颜色取决于是否悬停
            target_bg = colors["hover_bg"] if self.underMouse() else QColor("transparent")
            self._bg_color_anim.setEndValue(target_bg)
            self._check_opacity_anim.setStartValue(self._check_opacity)
            self._check_opacity_anim.setEndValue(0.0)

        group = QParallelAnimationGroup(self)
        group.addAnimation(self._bg_color_anim)
        group.addAnimation(self._check_opacity_anim)
        group.start()

    def setChecked(self, checked):
        """设置选中状态，跳过动画"""
        super().setChecked(checked)
        colors = self._get_theme_colors()
        if checked:
            self._bg_color = colors["checked_bg"]
            self._check_opacity = 1.0
        else:
            self._bg_color = QColor("transparent")
            self._check_opacity = 0.0
        self.update()

    def paintEvent(self, event):
        """自定义绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        colors = self._get_theme_colors()

        # --- 绘制复选框 ---
        rect = self.rect()
        box_size = 20
        box_y = (rect.height() - box_size) // 2
        box_rect = QRect(0, box_y, box_size, box_size)
        box_radius = 4

        # 绘制背景
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.backgroundColor))
        painter.drawRoundedRect(box_rect, box_radius, box_radius)

        # 绘制边框
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(colors["border"], 2)
        painter.setPen(pen)
        painter.drawRoundedRect(box_rect.adjusted(1, 1, -1, -1), box_radius, box_radius)

        # --- 绘制勾选标记 ---
        if self._check_opacity > 0:
            painter.setOpacity(self._check_opacity)
            check_pen = QPen(colors["checkmark"], 2)
            check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)

            # 定义勾选路径
            path = QPainterPath()
            path.moveTo(box_rect.x() + 5, box_rect.y() + 10)
            path.lineTo(box_rect.x() + 9, box_rect.y() + 14)
            path.lineTo(box_rect.x() + 15, box_rect.y() + 6)
            painter.drawPath(path)

        painter.setOpacity(1.0) # 重置透明度

        # --- 绘制文本 ---
        text_rect = rect.adjusted(box_size + 8, 0, 0, 0)
        font = self.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QPen(colors["text"]))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self.text())


    def enterEvent(self, event):
        """鼠标进入事件"""
        super().enterEvent(event)
        if not self.isChecked():
            colors = self._get_theme_colors()
            self._bg_color_anim.setStartValue(self.backgroundColor)
            self._bg_color_anim.setEndValue(colors["hover_bg"])
            self._bg_color_anim.start()

    def leaveEvent(self, event):
        """鼠标离开事件"""
        super().leaveEvent(event)
        if not self.isChecked():
            self._bg_color_anim.setStartValue(self.backgroundColor)
            self._bg_color_anim.setEndValue(QColor("transparent"))
            self._bg_color_anim.start()

    def update_theme(self):
        """更新主题"""
        self.setChecked(self.isChecked()) # 更新颜色