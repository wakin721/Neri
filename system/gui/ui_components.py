import sys
import platform
import logging
from typing import Callable, Optional, List
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QApplication, QGraphicsDropShadowEffect,
    QSizePolicy, QSpacerItem, QCheckBox, QLineEdit, QGroupBox,
    QSlider, QComboBox, QDialog, QStyledItemDelegate
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QRect, QRectF, QSize, Signal, Property, QParallelAnimationGroup, QPoint,
    QObject, QEvent
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QFont, QFontMetrics,
    QPalette, QLinearGradient, QBrush, QPen, QRegion
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
    """Material You (M3) 风格开关组件"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation_duration = 200
        self._thumb_position = 0.0   # 0.0=左 1.0=右
        self._thumb_scale   = 0.0   # 0.0=小(关闭) 1.0=大(开启)

        self._setup_animations()
        self._update_stylesheet()

    def _setup_animations(self):
        self._pos_anim = QPropertyAnimation(self, b"thumb_position")
        self._pos_anim.setDuration(self._animation_duration)
        self._pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._scale_anim = QPropertyAnimation(self, b"thumb_scale")
        self._scale_anim.setDuration(self._animation_duration)
        self._scale_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ── 动画属性 ──────────────────────────────────────────────────
    @Property(float)
    def thumb_position(self):
        return self._thumb_position

    @thumb_position.setter
    def thumb_position(self, value):
        self._thumb_position = value
        self.update()

    @Property(float)
    def thumb_scale(self):
        return self._thumb_scale

    @thumb_scale.setter
    def thumb_scale(self, value):
        self._thumb_scale = value
        self.update()

    def _update_stylesheet(self):
        self.setStyleSheet("""
            QCheckBox { background: transparent; spacing: 0px; }
            QCheckBox::indicator { width: 0px; height: 0px; }
        """)

    def nextCheckState(self):
        super().nextCheckState()
        self._animate_to_state()

    def _animate_to_state(self):
        target = 1.0 if self.isChecked() else 0.0

        self._pos_anim.setStartValue(self._thumb_position)
        self._pos_anim.setEndValue(target)
        self._pos_anim.start()

        self._scale_anim.setStartValue(self._thumb_scale)
        self._scale_anim.setEndValue(target)
        self._scale_anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        # ── 颜色定义 ───────────────────────────────────────────────
        if is_dark:
            accent          = Win11Colors.DARK_ACCENT       # 轨道开启填充 / 关闭滑块色
            track_off_fill  = Win11Colors.DARK_SURFACE      # 关闭态轨道填充
            track_border    = Win11Colors.DARK_BORDER       # 关闭态轨道边框
            thumb_on_color  = Win11Colors.DARK_TEXT_PRIMARY # 开启态滑块（白）
            thumb_off_color = Win11Colors.DARK_TEXT_SECONDARY
        else:
            accent          = Win11Colors.LIGHT_ACCENT
            track_off_fill  = Win11Colors.LIGHT_SURFACE
            track_border    = Win11Colors.LIGHT_BORDER
            thumb_on_color  = Win11Colors.LIGHT_CARD        # 开启态滑块（白）
            thumb_off_color = Win11Colors.LIGHT_TEXT_SECONDARY

        w, h = self.width(), self.height()
        track_r = h / 2          # 轨道圆角（完整胶囊）
        border_w = 2             # 关闭态边框宽度

        # ── 绘制轨道 ───────────────────────────────────────────────
        from PySide6.QtCore import QRectF
        track_rect = QRectF(0, 0, w, h)
        track_path = QPainterPath()
        track_path.addRoundedRect(track_rect, track_r, track_r)

        if self.isChecked():
            # 开启：实心填充
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(accent))
            painter.drawPath(track_path)
        else:
            # 关闭：填充 surface + 2px 边框描边
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(track_off_fill))
            painter.drawPath(track_path)

            pen = QPen(track_border, border_w)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            inner = track_rect.adjusted(border_w / 2, border_w / 2,
                                        -border_w / 2, -border_w / 2)
            inner_path = QPainterPath()
            inner_path.addRoundedRect(inner, track_r - border_w / 2,
                                               track_r - border_w / 2)
            painter.drawPath(inner_path)

        # ── 滑块尺寸（关闭=小 开启=大）────────────────────────────
        thumb_min = h * 0.40   # 关闭态直径
        thumb_max = h * 0.72   # 开启态直径
        thumb_d   = thumb_min + self._thumb_scale * (thumb_max - thumb_min)
        thumb_r   = thumb_d / 2

        # 运动范围：从左侧内边距到右侧内边距
        margin   = (h - thumb_max) / 2        # 保证最大滑块不超出轨道
        travel   = w - 2 * margin - thumb_max  # 可移动距离
        thumb_cx = margin + thumb_r + self._thumb_position * travel
        thumb_cy = h / 2

        thumb_rect = QRectF(thumb_cx - thumb_r, thumb_cy - thumb_r,
                            thumb_d, thumb_d)
        thumb_path = QPainterPath()
        thumb_path.addEllipse(thumb_rect)

        # 滑块颜色：关闭→主题色，开启→白色，中间插值
        r = int(thumb_off_color.red()   + self._thumb_scale * (thumb_on_color.red()   - thumb_off_color.red()))
        g = int(thumb_off_color.green() + self._thumb_scale * (thumb_on_color.green() - thumb_off_color.green()))
        b = int(thumb_off_color.blue()  + self._thumb_scale * (thumb_on_color.blue()  - thumb_off_color.blue()))
        thumb_color = QColor(r, g, b)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(thumb_color))
        painter.drawPath(thumb_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
            self._animate_to_state()
        super().mousePressEvent(event)

    def setChecked(self, checked):
        super().setChecked(checked)
        # 直接跳到目标状态，不使用动画
        self._thumb_position = 1.0 if checked else 0.0
        self._thumb_scale    = 1.0 if checked else 0.0
        self.update()

    # track_opacity 保留以防外部代码引用
    @Property(float)
    def track_opacity(self):
        return 1.0

    @track_opacity.setter
    def track_opacity(self, value):
        pass

    def update_theme(self):
        self._update_stylesheet()
        self.update()


class MaterialMessageBox(QDialog):
    """自定义 Material You 风格的全局提示框，完美替换 QMessageBox"""

    def __init__(self, parent=None, title="", text="", is_question=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)

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

        # 应用动态主题颜色
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QLabel {{
                color: {text_color};
                font-size: 15px;
                font-weight: normal;
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {btn_text};
                border: none;
                padding: 8px 20px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
                min-width: 60px;
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

        # 提示文本 (支持自动换行)
        self.msg_label = QLabel(text)
        self.msg_label.setTextFormat(Qt.TextFormat.MarkdownText)
        self.msg_label.setOpenExternalLinks(True)
        self.msg_label.setWordWrap(True)
        layout.addWidget(self.msg_label)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if is_question:
            self.yes_btn = QPushButton("是")
            self.yes_btn.clicked.connect(self.accept)

            self.no_btn = QPushButton("否")
            self.no_btn.setObjectName("cancelButton")  # 绑定浅色/次要按钮样式
            self.no_btn.clicked.connect(self.reject)

            btn_layout.addWidget(self.yes_btn)
            btn_layout.addWidget(self.no_btn)
        else:
            self.ok_btn = QPushButton("确定")
            self.ok_btn.clicked.connect(self.accept)
            btn_layout.addWidget(self.ok_btn)

        layout.addLayout(btn_layout)
        self.setMinimumWidth(320)

    # ================= 兼容原 QMessageBox 的静态方法 =================
    @classmethod
    def information(cls, parent, title, text):
        dialog = cls(parent, title, text, is_question=False)
        dialog.exec()

    @classmethod
    def warning(cls, parent, title, text):
        dialog = cls(parent, title, text, is_question=False)
        dialog.exec()

    @classmethod
    def critical(cls, parent, title, text):
        dialog = cls(parent, title, text, is_question=False)
        dialog.exec()

    @classmethod
    def question(cls, parent, title, text, buttons=None):
        dialog = cls(parent, title, text, is_question=True)
        result = dialog.exec()
        # 完美兼容代码中原有的 QMessageBox.StandardButton.Yes 返回值逻辑
        if result == QDialog.DialogCode.Accepted:
            from PySide6.QtWidgets import QMessageBox
            return QMessageBox.StandardButton.Yes
        else:
            from PySide6.QtWidgets import QMessageBox
            return QMessageBox.StandardButton.No


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
    """Material You (M3) 风格药丸按钮"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self._is_active = False
        self._corner_radius = 14
        self._animation_duration = 150
        self._setup_ui()
        self._setup_animations()

    def _setup_ui(self):
        """设置UI样式"""
        self.setMinimumSize(120, 36)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium)) # 稍微提升字重
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 设置样式表
        self._update_stylesheet()

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 1)
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

        # 核心修改：直接使用 Accent 作为按钮背景色，并调整文字对比度
        if is_dark:
            base_bg = Win11Colors.DARK_ACCENT      # #5d3a4f
            text_color = QColor(255, 255, 255)     # 深色背景配白字
            hover_bg = base_bg.lighter(120)
            pressed_bg = base_bg.darker(110)
            disabled_bg = QColor(60, 60, 60)
            disabled_text = QColor(120, 120, 120)
        else:
            base_bg = Win11Colors.LIGHT_ACCENT     # #dbbcc2
            text_color = QColor(255, 255, 255)  # 浅色背景配深色字保证可读性
            hover_bg = base_bg.darker(105)
            pressed_bg = base_bg.darker(115)
            disabled_bg = QColor(230, 230, 230)
            disabled_text = QColor(150, 150, 150)

        active_bg = pressed_bg if self._is_active else base_bg

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {active_bg.name()};
                color: {text_color.name()};
                border: none;
                border-radius: {self._corner_radius}px;
                padding: 8px 16px;
                font-weight: {"600" if self._is_active else "500"};
            }}
            QPushButton:hover {{
                background-color: {hover_bg.name()};
            }}
            QPushButton:pressed {{
                background-color: {pressed_bg.name()};
            }}
            QPushButton:disabled {{
                background-color: {disabled_bg.name()};
                color: {disabled_text.name()};
            }}
        """)

    def is_active(self) -> bool:
        """返回按钮是否处于激活状态"""
        return self._is_active

    def update_theme(self):
        """更新主题"""
        self._update_stylesheet()


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
            track_color = Win11Colors.DARK_ACCENT.darker(300)
            border_color = Win11Colors.DARK_BORDER  # 新增：深色边框色
        else:
            bg_color = Win11Colors.LIGHT_SURFACE
            text_color = Win11Colors.LIGHT_TEXT_SECONDARY
            accent_color = Win11Colors.LIGHT_ACCENT
            track_color = Win11Colors.LIGHT_ACCENT.lighter(140)
            border_color = Win11Colors.LIGHT_BORDER  # 新增：浅色边框色

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color.name()};
            }}
            QLabel {{
                color: {text_color.name()};
                background-color: {bg_color.name()};
            }}
            /* Material You 风格进度条（带外边框） */
            QProgressBar {{
                border: 1px solid {border_color.name()}; /* 增加外边框 */
                border-radius: 8px; 
                background-color: {track_color.name()};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {accent_color.name()};
                border-radius: 7px; /* 稍微缩小一点圆角以适应边框内径 */
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
                        border: none; /* 去除硬边框 */
                        border-radius: 16px; /* Material You 大圆角 */
                    }}
                """)

    def set_current_file(self, filename: str):
        """设置当前处理的文件名"""
        self._current_file = filename
        self.update()

    def paintEvent(self, event):
        """自定义绘制"""
        from PySide6.QtCore import QRectF  # 确保导入了 QRectF

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
            track_bg = Win11Colors.DARK_ACCENT.darker(300)
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            progress_text_color = QColor(255, 255, 255)
            border_color = Win11Colors.DARK_BORDER  # 新增：深色边框色
        else:
            track_bg = Win11Colors.LIGHT_ACCENT.lighter(140)
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            progress_text_color = QColor(255, 255, 255)
            border_color = Win11Colors.LIGHT_BORDER  # 新增：浅色边框色

        # 将绘制区域向内收缩 0.5 像素，防止边框被边缘裁剪掉
        draw_rect = QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)

        # 绘制背景（Material You 大圆角）
        path = QPainterPath()
        path.addRoundedRect(draw_rect, 16, 16)
        painter.fillPath(path, track_bg)

        # 绘制外边框
        painter.setPen(QPen(border_color, 1))
        painter.drawPath(path)

        # 绘制进度（带大圆角）
        progress_ratio = self._progress / max(self._total, 1)
        progress_width = draw_rect.width() * progress_ratio

        if progress_width > 4:  # 只有当进度条有一定宽度时才绘制
            progress_rect = QRectF(draw_rect.x(), draw_rect.y(), progress_width, draw_rect.height())
            progress_path = QPainterPath()
            progress_path.addRoundedRect(progress_rect, 16, 16)
            painter.fillPath(progress_path, self._accent_color)

        # 绘制文本信息
        self._draw_progress_text(painter, rect, int(progress_width), text_color, progress_text_color)

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


class _RotatingChevron(QWidget):
    """可旋转的 chevron 指示器，用于 CollapsiblePanel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self._rotation = 0.0

    @Property(float)
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT

        # 绕中心旋转
        cx, cy = self.width() / 2, self.height() / 2
        painter.translate(cx, cy)
        painter.rotate(self._rotation)
        painter.translate(-cx, -cy)

        # 绘制 chevron（向下的 "∨"）
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        path.moveTo(6, 9)
        path.lineTo(12, 15)
        path.lineTo(18, 9)
        painter.drawPath(path)


class CollapsiblePanel(QFrame):
    """现代化可折叠面板组件 - Material You 风格版本"""

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
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # 修复 Header 向下伸展：强制主布局靠上对齐。
        # 这样在收起动画期间，即使外部父容器还没来得及缩小，
        # 内部多出的空隙也会留在底部，绝对不会去拉伸 Header。
        self._main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._create_header()

        self._content_frame = QFrame()
        self._content_frame.setMinimumHeight(0)
        # 修复默认展开状态：强制初始的最大高度为0
        self._content_frame.setMaximumHeight(0)
        self._content_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._content_frame.setFrameStyle(QFrame.Shape.NoFrame)

        # ── 透明度特效层 ──────────────────────────────────────
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._opacity_effect = QGraphicsOpacityEffect(self._content_frame)
        self._opacity_effect.setOpacity(0.0)
        self._content_frame.setGraphicsEffect(self._opacity_effect)

        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(20, 10, 20, 20)

        self._main_layout.addWidget(self._content_frame)
        self._update_styles()

    def _create_header(self):
        self._header_frame = QFrame()

        self._header_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._header_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header_frame.mousePressEvent = self._on_header_clicked

        header_layout = QHBoxLayout(self._header_frame)
        header_layout.setContentsMargins(16, 16, 16, 16)

        if self._icon:
            self._icon_label = QLabel(self._icon)
            font_name = "Segoe UI Emoji" if platform.system() == "Windows" else "Apple Color Emoji"
            self._icon_label.setFont(QFont(font_name, 16))
            header_layout.addWidget(self._icon_label)

        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        self._title_label = QLabel(self._title)
        self._title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        title_container.addWidget(self._title_label)
        if self._subtitle:
            self._subtitle_label = QLabel(self._subtitle)
            self._subtitle_label.setFont(QFont("Segoe UI", 9))
            title_container.addWidget(self._subtitle_label)

        header_layout.addLayout(title_container)
        header_layout.addStretch()

        # ── 新增：自定义可旋转指示器 ────────────────────────────────
        self._toggle_indicator = _RotatingChevron()
        header_layout.addWidget(self._toggle_indicator)

        self._main_layout.addWidget(self._header_frame)

    def _setup_animations(self):
        self._animation_duration = 300  # M3 标准时长

        # 高度动画
        self._height_animation = QPropertyAnimation(self._content_frame, b"maximumHeight")
        self._height_animation.setDuration(self._animation_duration)

        # 透明度动画
        self._opacity_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._opacity_animation.setDuration(self._animation_duration)

        # 指示器旋转动画
        self._rotation_animation = QPropertyAnimation(self._toggle_indicator, b"rotation")
        self._rotation_animation.setDuration(self._animation_duration)

        # 并行动画组
        self._anim_group = QParallelAnimationGroup()
        self._anim_group.addAnimation(self._height_animation)
        self._anim_group.addAnimation(self._opacity_animation)
        self._anim_group.addAnimation(self._rotation_animation)
        self._anim_group.finished.connect(self._on_animation_finished)

    def expand(self):
        if self._is_expanded:
            return

        if self._anim_group.state() == QParallelAnimationGroup.State.Running:
            self._anim_group.stop()

        self._is_expanded = True
        self._update_header_style()

        # 【新增】：在动画开始前锁死 Header 的最小高度
        # 确保在布局引擎由于空间不足而“恐慌”时，绝对无法压缩 Header
        if self._header_frame.minimumHeight() == 0:
            self._header_frame.setMinimumHeight(self._header_frame.sizeHint().height())

        # 1. 记录当前高度（兼容动画中途打断的情况）
        current_h = self._content_frame.maximumHeight()
        if current_h >= 16777215:
            current_h = self._content_frame.height()

        # 2. 直接通过内部布局获取目标高度！
        # 不再来回修改 maximumHeight，彻底切断 Qt 强制同步重绘的触发条件
        target_height = self._content_layout.sizeHint().height()

        # 展开：先快后慢（OutQuart）
        self._height_animation.setEasingCurve(QEasingCurve.Type.OutQuart)
        self._height_animation.setStartValue(current_h)
        self._height_animation.setEndValue(target_height)
        # ─── 修改的部分结束 ──────────────────────────────────────────

        self._opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._opacity_animation.setStartValue(self._opacity_effect.opacity())
        self._opacity_animation.setEndValue(1.0)

        self._rotation_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._rotation_animation.setStartValue(self._toggle_indicator.rotation)
        self._rotation_animation.setEndValue(180.0)

        self._is_animating = True
        self._anim_group.start()
        self.toggled.emit(True)

    def collapse(self):
        if not self._is_expanded:
            return

        if self._anim_group.state() == QParallelAnimationGroup.State.Running:
            self._anim_group.stop()

        self._is_expanded = False

        # 注意：这里结合了上一条回复中建议的 current_h = self._content_frame.height() 修复
        current_h = self._content_frame.maximumHeight()
        if current_h >= 16777215:
            current_h = self._content_frame.height()  # 修改为使用实际渲染高度
        self._content_frame.setMaximumHeight(current_h)

        # 收起：平滑减速（OutCubic）
        self._height_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._height_animation.setStartValue(current_h)
        self._height_animation.setEndValue(0)

        self._opacity_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self._opacity_animation.setStartValue(self._opacity_effect.opacity())
        self._opacity_animation.setEndValue(0.0)

        self._rotation_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._rotation_animation.setStartValue(self._toggle_indicator.rotation)
        self._rotation_animation.setEndValue(0.0)

        # 新增：标记动画正在进行中，防止重复点击
        self._is_animating = True
        self._anim_group.start()
        self._update_header_style()
        self.toggled.emit(False)

    def _on_animation_finished(self):
        self._is_animating = False
        if not self._is_expanded:
            self._content_frame.setMaximumHeight(0)
        else:
            self._content_frame.setMaximumHeight(16777215)
            self._update_content_style()

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

        # Material You 偏爱大圆角，使用 16px
        border_radius = "16px 16px 0px 0px" if self._is_expanded else "16px"

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
        border_color = Win11Colors.DARK_BORDER if is_dark else Win11Colors.LIGHT_BORDER
        hover_color = Win11Colors.DARK_HOVER if is_dark else Win11Colors.LIGHT_HOVER
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
        accent_color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT

        self._content_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {content_bg.name()};
                border: none;
                border-radius: 16px 16px 16px 16px;
            }}
            .QWidget {{
                background-color: transparent;
            }}
            QLabel {{
                background-color: transparent;
                color: {text_color.name()};
            }}

            QLineEdit {{
                background-color: {content_bg.lighter(105).name()};
                border: 2px solid {border_color.name()};
                border-radius: 12px;
                padding: 10px 16px;
                color: {text_color.name()};
            }}
            QLineEdit:hover {{
                background-color: {hover_color.name()};
            }}
            QLineEdit:focus {{
                border-color: {accent_color.name()};
            }}

            QSlider {{ background-color: transparent; }}
            QCheckBox {{ background-color: transparent; color: {text_color.name()}; }}
            ModernSwitch {{ background-color: transparent; }}

            /* Material You 风格下拉框 */
            QComboBox {{
                background-color: {content_bg.lighter(105).name()};
                border: 2px solid {border_color.name()};
                border-radius: 12px;
                padding: 8px 16px;
                color: {text_color.name()};
                min-height: 24px;
            }}
            QComboBox:hover {{
                background-color: {hover_color.name()};
                border-color: {accent_color.name()};
            }}
            QComboBox:focus {{
                border-color: {accent_color.name()};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border: none;
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
                border: none;
                background: transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {content_bg.name()};
                border: 1px solid {border_color.name()};
                border-radius: 8px;
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView QWidget {{
                background-color: {content_bg.name()};
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 32px;
                border-radius: 6px;
                padding-left: 12px;
                margin: 2px 4px;
                background-color: transparent;
                color: {text_color.name()};
                border: none;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {hover_color.name()};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {accent_color.name()};
                color: white;
            }}
        """)

    def _update_indicator_style(self):
        """更新指示器样式"""
        self._toggle_indicator.update()

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

        # 主容器样式 - Material You 16px 大圆角
        self.setStyleSheet(f"""
            CollapsiblePanel {{
                border: 1px solid {border_color.name()};
                border-radius: 16px;
            }}
        """)

        # 更新各部分样式
        self._update_header_style()
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
        self._update_child_widgets_background(content_bg)

    def _update_child_widgets_background(self, content_bg):
        """递归更新所有子组件的背景色，同步 Material You 样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        border_color = Win11Colors.DARK_BORDER if is_dark else Win11Colors.LIGHT_BORDER
        hover_color = Win11Colors.DARK_HOVER if is_dark else Win11Colors.LIGHT_HOVER
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
        accent_color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT

        def update_widget_recursive(widget):
            if not widget:
                return

            class_name = widget.__class__.__name__

            if class_name in ['QWidget', 'QFrame']:
                current_style = widget.styleSheet()
                current_style = current_style.replace("background-color: transparent;", "")
                widget.setStyleSheet(f"{current_style} background-color: transparent;")

            elif class_name == 'QLabel':
                current_style = widget.styleSheet()
                current_style = current_style.replace("background-color: transparent;", "")
                current_style = current_style.replace(f"color: {Win11Colors.LIGHT_TEXT_PRIMARY.name()};", "")
                current_style = current_style.replace(f"color: {Win11Colors.DARK_TEXT_PRIMARY.name()};", "")

                if 'color:' not in current_style:
                    widget.setStyleSheet(
                        f"{current_style} background-color: transparent; color: {text_color.name()};")
                else:
                    widget.setStyleSheet(f"{current_style} background-color: transparent;")

            elif class_name == 'SwitchRow':
                widget.setStyleSheet("SwitchRow { background-color: transparent; }")
                widget.update_theme()

            elif class_name == 'ModernSwitch':
                widget.setStyleSheet("ModernSwitch { background-color: transparent; }")
                widget.update_theme()

            elif class_name in ['QComboBox', 'ModernComboBox']:
                widget.setStyleSheet(f"""
                    QComboBox {{
                        background-color: {content_bg.lighter(105).name()};
                        border: 2px solid {border_color.name()};
                        border-radius: 12px;
                        padding: 8px 16px;
                        color: {text_color.name()};
                        min-height: 24px;
                    }}
                    QComboBox:hover {{
                        background-color: {hover_color.name()};
                        border-color: {accent_color.name()};
                    }}
                    QComboBox:focus {{
                        border-color: {accent_color.name()};
                    }}
                    QComboBox::drop-down {{
                        width: 32px;
                        border: none;
                        background-color: transparent;
                    }}
                    QComboBox::down-arrow {{
                        width: 12px;
                        height: 12px;
                        border: none;
                        background: transparent;
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: {content_bg.name()};
                        border: 1px solid {border_color.name()};
                        border-radius: 8px;
                        padding: 4px;
                        outline: none;
                        show-decoration-selected: 1;
                    }}
                    QComboBox QAbstractItemView QWidget {{
                        background-color: {content_bg.name()};
                    }}
                    QComboBox QAbstractItemView::item {{
                        min-height: 32px;
                        border-radius: 6px;
                        padding-left: 12px;
                        margin: 2px 4px;
                        background-color: transparent;
                        color: {text_color.name()};
                        border: none;
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
                widget.setStyleSheet(f"""
                    QLineEdit {{
                        background-color: {content_bg.lighter(105).name()};
                        border: 2px solid {border_color.name()};
                        border-radius: 12px;
                        padding: 10px 16px;
                        color: {text_color.name()};
                    }}
                    QLineEdit:hover {{
                        background-color: {hover_color.name()};
                    }}
                    QLineEdit:focus {{
                        border-color: {accent_color.name()};
                    }}
                """)

            elif class_name in ['QSlider', 'QCheckBox']:
                current_style = widget.styleSheet()
                current_style = current_style.replace("background-color: transparent;", "")
                current_style = current_style.replace(f"color: {Win11Colors.LIGHT_TEXT_PRIMARY.name()};", "")
                current_style = current_style.replace(f"color: {Win11Colors.DARK_TEXT_PRIMARY.name()};", "")

                if 'color:' not in current_style:
                    widget.setStyleSheet(
                        f"{current_style} background-color: transparent; color: {text_color.name()};")
                else:
                    widget.setStyleSheet(f"{current_style} background-color: transparent;")


            elif class_name == 'QPushButton':

                # 核心修改：使用主题色作为面板内普通按钮的背景

                if is_dark:

                    btn_bg = Win11Colors.DARK_ACCENT

                    btn_text = QColor(255, 255, 255)

                    btn_hover = btn_bg.lighter(120)

                    btn_pressed = btn_bg.darker(110)

                    disabled_bg = QColor(60, 60, 60)

                    disabled_text = QColor(120, 120, 120)

                else:
                    btn_bg = Win11Colors.LIGHT_ACCENT
                    btn_text = Win11Colors.LIGHT_TEXT_PRIMARY
                    btn_hover = btn_bg.darker(105)
                    btn_pressed = btn_bg.darker(115)
                    disabled_bg = QColor(230, 230, 230)
                    disabled_text = QColor(150, 150, 150)
                widget.setStyleSheet(f"""
                                QPushButton {{
                                    background-color: {btn_bg.name()};
                                    border: none;
                                    border-radius: 18px;
                                    padding: 8px 16px;
                                    color: {btn_text.name()};
                                    font-weight: 500;
                                }}
                                QPushButton:hover {{
                                    background-color: {btn_hover.name()};
                                }}
                                QPushButton:pressed {{
                                    background-color: {btn_pressed.name()};
                                }}
                                QPushButton:disabled {{
                                    background-color: {disabled_bg.name()};
                                    color: {disabled_text.name()};
                                }}
                            """)

            if class_name in ['QComboBox', 'ModernComboBox', 'QLineEdit', 'ModernLineEdit', 'QSlider', 'ModernSlider',
                              'QCheckBox', 'ModernCheckBox', 'QPushButton', 'RoundedButton', 'ModernSwitch',
                              'SwitchRow']:
                return

            # 使用 children() 获取直系子组件，避免 findChildren 造成的深层破坏和重复计算
            for child in widget.children():
                if isinstance(child, QWidget):
                    update_widget_recursive(child)

        update_widget_recursive(self._content_frame)

    def content_widget(self) -> QWidget:
        """获取内容区域Widget，用于添加子控件"""
        content_widget = QWidget()
        self._content_layout.addWidget(content_widget)
        return content_widget

    def add_content_widget(self, widget: QWidget):
        """添加内容控件"""
        self._content_layout.addWidget(widget)

        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        content_bg = Win11Colors.DARK_CARD if is_dark else Win11Colors.LIGHT_CARD

        # 让后面的 _setup_widget_background 根据组件类型 (class_name) 智能分配背景色
        self._setup_widget_background(widget, content_bg)

    def _setup_widget_background(self, widget, content_bg):
        """为单个组件及其子组件设置背景"""

        def setup_recursive(w):
            if not w:
                return

            class_name = w.__class__.__name__
            current_style = w.styleSheet()

            if class_name in ['QWidget', 'QFrame', 'QLabel']:
                if 'background-color' not in current_style:
                    w.setStyleSheet(f"{current_style}; background-color: transparent;")
            elif class_name in ['QLineEdit', 'QComboBox', 'ModernComboBox']:
                if 'background-color' not in current_style:
                    w.setStyleSheet(f"{current_style}; background-color: {content_bg.name()};")

            # 同样在这里设置屏障，避免向下渗透
            if class_name in [
                'QComboBox', 'ModernComboBox', 'QLineEdit', 'ModernLineEdit',
                'QSlider', 'ModernSlider', 'QCheckBox', 'ModernCheckBox',
                'QPushButton', 'RoundedButton', 'ModernSwitch', 'SwitchRow',
                'PathInputWidget'
            ]:
                return

            for child in w.children():
                if isinstance(child, QWidget):
                    setup_recursive(child)

        setup_recursive(widget)

    def is_expanded(self) -> bool:
        """返回是否已展开"""
        return self._is_expanded

    def update_theme(self):
        """更新主题"""
        self._update_styles()


class ThemeManager:
    """主题管理器"""
    @staticmethod
    def apply_win11_style(app: QApplication, force_dark: bool = None):
        """应用自定义Win11样式到整个应用程序"""
        
        if force_dark is not None:
            palette = app.palette()
            if force_dark:
                # 欺骗其他组件，告诉它们现在是深色背景
                palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30)) 
            else:
                # 欺骗其他组件，告诉它们现在是浅色背景
                palette.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250)) 
            app.setPalette(palette)
            is_dark = force_dark
        else:
            # 自动模式：检测系统主题
            palette = app.palette()
            is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            ThemeManager._apply_dark_theme(app)
        else:
            ThemeManager._apply_light_theme(app)

    @staticmethod
    def _get_scrollbar_style(is_dark):
        """生成 Material You 风格的滚动条样式"""
        accent = Win11Colors.DARK_ACCENT.name() if is_dark else Win11Colors.LIGHT_ACCENT.name()
        # 悬停时稍微加亮或变深
        hover_accent = Win11Colors.DARK_ACCENT.lighter(120).name() if is_dark else Win11Colors.LIGHT_ACCENT.darker(
            110).name()

        return f"""
        /* 整个滚动条区域 */
        QScrollBar:vertical {{
            background: transparent;
            width: 12px;
            margin: 4px 2px 4px 2px;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 12px;
            margin: 2px 4px 2px 4px;
        }}

        /* 滚动条滑块 - 药丸形状 */
        QScrollBar::handle:vertical {{
            background: {accent};
            min-height: 40px;
            border-radius: 4px; /* 初始状态较细 */
        }}
        QScrollBar::handle:horizontal {{
            background: {accent};
            min-width: 40px;
            border-radius: 4px;
        }}

        /* 悬停状态 - 宽度微增，颜色变化 */
        QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed {{
            background: {hover_accent};
            border-radius: 4px;
        }}

        /* 隐藏滚动条上下按钮（Material 风格不需要） */
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
            height: 0px;
            background: none;
        }}

        /* 隐藏轨道背景 */
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
        """

    @staticmethod
    def _apply_light_theme(app: QApplication):
        style = f"""
        QWidget {{
            background-color: {Win11Colors.LIGHT_BACKGROUND.name()};
            color: {Win11Colors.LIGHT_TEXT_PRIMARY.name()};
            font-family: "Segoe UI", "PingFang SC";
        }}
        {ThemeManager._get_scrollbar_style(False)}
        QScrollArea {{ border: none; background: transparent; }}
        """
        app.setStyleSheet(style)

    @staticmethod
    def _apply_dark_theme(app: QApplication):
        style = f"""
        QWidget {{
            background-color: {Win11Colors.DARK_BACKGROUND.name()};
            color: {Win11Colors.DARK_TEXT_PRIMARY.name()};
            font-family: "Segoe UI", "PingFang SC";
        }}
        {ThemeManager._get_scrollbar_style(True)}
        QScrollArea {{ border: none; background: transparent; }}
        """
        app.setStyleSheet(style)


class ModernGroupBox(QGroupBox):
    """现代化的分组框，使用 Material You 风格"""

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self._setup_style()

    def _setup_style(self):
        """设置 Material You 风格样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            border_color = Win11Colors.DARK_BORDER
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            title_bg_color = Win11Colors.DARK_SURFACE
            accent_color = Win11Colors.DARK_ACCENT  # 引入主题色
        else:
            border_color = Win11Colors.LIGHT_BORDER
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            title_bg_color = Win11Colors.LIGHT_SURFACE
            accent_color = Win11Colors.LIGHT_ACCENT # 引入主题色

        self.setStyleSheet(f"""
            QGroupBox {{
                font-family: "Segoe UI Variable", "Segoe UI", "PingFang SC";
                font-weight: 600;
                font-size: 14px;
                color: {text_color.name()};
                border: 2px solid {border_color.name()};
                border-radius: 12px;  /* Material You 偏爱更大的整体圆角 */
                margin-top: 14px;     /* 为标题留出切割空间 */
                padding-top: 24px;
                background-color: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;           /* 标题整体稍微往右靠，增加呼吸感 */
                padding: 4px 16px;    /* 增加左右内边距，形成经典的药丸形状 */
                background-color: {title_bg_color.name()};
                border: 2px solid {accent_color.name()}; /* 使用主题强调色作为独立边框 */
                border-radius: 14px;  /* 强圆角，呈现药丸形独立边框 */
                color: {text_color.name()};
            }}
        """)

    def update_theme(self):
        """更新主题"""
        self._setup_style()


class ModernLineEdit(QLineEdit):
    """现代化的输入框，使用 Material You 风格"""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        if placeholder:
            self.setPlaceholderText(placeholder)
        self._setup_style()

    def _setup_style(self):
        """设置 Material You 风格样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        # 调整颜色映射以贴合 Material You
        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE        # 使用 Surface 色作为背景
            border_color = Win11Colors.DARK_BORDER
            text_color = Win11Colors.DARK_TEXT_PRIMARY
            focus_color = Win11Colors.DARK_ACCENT
            placeholder_color = Win11Colors.DARK_TEXT_SECONDARY
            hover_bg = Win11Colors.DARK_HOVER          # 引入悬停色
        else:
            bg_color = Win11Colors.LIGHT_SURFACE       # 使用 Surface 色作为背景
            border_color = Win11Colors.LIGHT_BORDER
            text_color = Win11Colors.LIGHT_TEXT_PRIMARY
            focus_color = Win11Colors.LIGHT_ACCENT
            placeholder_color = Win11Colors.LIGHT_TEXT_SECONDARY
            hover_bg = Win11Colors.LIGHT_HOVER         # 引入悬停色

        self.setStyleSheet(f"""
            QLineEdit {{
                padding: 10px 16px;   /* 增加上下左右的内边距，提供呼吸感 */
                border: 2px solid {border_color.name()};
                border-radius: 12px;  /* 增大圆角，与 ModernGroupBox 保持一致 */
                background-color: {bg_color.name()};
                color: {text_color.name()};
                font-size: 14px;
                selection-background-color: {focus_color.name()};
                selection-color: white; /* 选中文本保持白色以确保对比度 */
                transition: all 0.2s ease-in-out; /* 平滑过渡效果 (如果Qt版本支持) */
            }}
            QLineEdit:hover {{
                background-color: {hover_bg.name()}; /* 增加悬停时的背景变色反馈 */
            }}
            QLineEdit:focus {{
                border: 2px solid {focus_color.name()}; /* 聚焦时边框变为强调色 */
                background-color: {bg_color.name()};    /* 聚焦时背景保持清晰 */
            }}
            QLineEdit:disabled {{
                background-color: transparent;
                border: 2px dashed {border_color.name()}; /* 禁用状态改用虚线边框，视觉更轻量 */
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
            app = QApplication.instance()
            is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
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

        if hasattr(self, 'browse_button'):
            if hasattr(self.browse_button, 'update_theme'):
                self.browse_button.update_theme()
            elif hasattr(self.browse_button, '_update_stylesheet'):
                self.browse_button._update_stylesheet()

        # 更新标签颜色
        from PySide6.QtWidgets import QLabel
        for label in self.findChildren(QLabel):
            if label.text() == self.label_text:
                app = QApplication.instance()
                is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
                text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY
                label.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")


class ModernSlider(QSlider):
    """Material You (M3) 风格滑块组件"""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMinimumHeight(40)
        self.setFixedHeight(40)

        self._is_dragging = False
        self._is_hovering = False

        # 轨道
        self._track_height = 6

        # 滑块主体半径（固定，不随状态变化）
        self._thumb_radius = 10

        # 状态层半径：0=隐藏  hover=20  pressed=16
        self._state_layer_radius = 0.0

        self._setup_style()
        self._setup_animation()

    # ── 动画属性：状态层半径 ──────────────────────────────────────
    @Property(float)
    def statLayerRadius(self):
        return self._state_layer_radius

    @statLayerRadius.setter
    def statLayerRadius(self, value):
        self._state_layer_radius = value
        self.update()

    def _setup_animation(self):
        self._layer_anim = QPropertyAnimation(self, b"statLayerRadius")
        self._layer_anim.setDuration(150)
        self._layer_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _setup_style(self):
        self.setStyleSheet("QSlider { background: transparent; border: none; }")

    def _animate_layer_to(self, target):
        if self._layer_anim.state() == QPropertyAnimation.State.Running:
            self._layer_anim.stop()
        self._layer_anim.setStartValue(self._state_layer_radius)
        self._layer_anim.setEndValue(float(target))
        self._layer_anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            track_bg       = Win11Colors.DARK_SURFACE
            accent         = Win11Colors.DARK_ACCENT
            state_layer    = QColor(accent.red(), accent.green(), accent.blue(), 30)
        else:
            track_bg       = Win11Colors.LIGHT_SURFACE
            accent         = Win11Colors.LIGHT_ACCENT
            state_layer    = QColor(accent.red(), accent.green(), accent.blue(), 40)

        rect = self.rect()
        tm   = self._thumb_radius  # 轨道两端留白

        # ── 1. 轨道背景 ────────────────────────────────────────────
        from PySide6.QtCore import QRectF
        th = self._track_height
        tr = QRectF(tm, (rect.height() - th) / 2,
                    rect.width() - 2 * tm, th)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(track_bg))
        painter.drawRoundedRect(tr, th / 2, th / 2)

        # ── 2. 进度填充 ────────────────────────────────────────────
        ratio = ((self.value() - self.minimum()) /
                 (self.maximum() - self.minimum())
                 if self.maximum() > self.minimum() else 0)

        if ratio > 0:
            pr = QRectF(tr.x(), tr.y(), tr.width() * ratio, th)
            painter.setBrush(QBrush(accent))
            painter.drawRoundedRect(pr, th / 2, th / 2)

        # ── 3. 滑块位置 ────────────────────────────────────────────
        cx = tm + ratio * tr.width()
        cy = rect.height() / 2

        # 状态层（hover / press 时展开的半透明光晕）
        if self._state_layer_radius > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(state_layer))
            sr = self._state_layer_radius
            painter.drawEllipse(
                QRectF(cx - sr, cy - sr, sr * 2, sr * 2)
            )

        # 滑块主体：M3 实心主题色圆
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        r = self._thumb_radius
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

    # ── 交互事件 ────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._animate_layer_to(16)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            self._update_value_from_position(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            if self.underMouse():
                self._is_hovering = True
                self._animate_layer_to(20)
            else:
                self._is_hovering = False
                self._animate_layer_to(0)
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._is_hovering = True
        if not self._is_dragging:
            self._animate_layer_to(20)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._is_hovering = False
        if not self._is_dragging:
            self._animate_layer_to(0)

    def _update_value_from_position(self, x):
        track_width = self.width() - 2 * self._thumb_radius
        relative_x  = x - self._thumb_radius
        if track_width > 0:
            ratio = max(0.0, min(1.0, relative_x / track_width))
            self.setValue(int(self.minimum() + ratio * (self.maximum() - self.minimum())))

    # innerRadius 保留以防外部引用
    @Property(float)
    def innerRadius(self):
        return 0.0

    @innerRadius.setter
    def innerRadius(self, value):
        pass

    def update_theme(self):
        self._setup_style()
        self.update()


class _DropdownPainter(QObject):
    """
    为 ModernComboBox 弹出窗口提供真实圆角绘制。

    原理：拦截弹出窗口（popup window）的 Paint 事件，
    手动绘制圆角背景 + 边框，并返回 True 消费该事件，
    阻止系统默认的矩形背景覆盖圆角效果。
    列表项的 Paint 事件属于子组件，不受影响，仍正常绘制。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.WinIdChange, QEvent.Type.Show, QEvent.Type.WindowActivate):
            try:
                obj.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
                obj.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            except Exception:
                pass
            return False  # ← 明确返回 bool

        if event.type() == QEvent.Type.Paint:
            try:
                app = QApplication.instance()
                is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
                bg_color = Win11Colors.DARK_CARD if is_dark else Win11Colors.LIGHT_CARD
                border_color = Win11Colors.DARK_ACCENT if is_dark else Win11Colors.LIGHT_ACCENT

                painter = QPainter(obj)
                if not painter.isActive():  # ← 防止绑定无效设备时抛出异常
                    return True

                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                rect = QRectF(obj.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
                path = QPainterPath()
                path.addRoundedRect(rect, 16, 16)
                painter.fillPath(path, bg_color)
                painter.setPen(QPen(border_color, 2.0))
                painter.drawPath(path)
                painter.end()
            except Exception:
                pass  # 出现任何异常时静默处理，但仍返回 True 消费事件

            return True  # ← 无论是否抛出异常，都明确返回 bool

        return False  # ← 所有其他事件，明确返回 bool


class ModernComboBox(QComboBox):
    """现代化下拉框组件 - Material You 风格（真实圆角弹出框版本）"""

    def __init__(self, parent=None):
        super().__init__(parent)

        view = self.view()
        view_window = view.window()

        view_window.setWindowFlags(
            Qt.WindowType.Popup |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.NoDropShadowWindowHint
        )
        view_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        view_window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)  # ← 新增：彻底关闭系统背景/阴影

        view.setAutoFillBackground(False)
        view.viewport().setAutoFillBackground(False)
        view.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._dropdown_painter = _DropdownPainter(self)
        view_window.installEventFilter(self._dropdown_painter)

        self._setup_style()

    def _setup_style(self):
        """设置 ComboBox 按钮本体 + 列表项样式"""
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark:
            bg_color     = Win11Colors.DARK_SURFACE
            border_color = Win11Colors.DARK_BORDER
            text_color   = Win11Colors.DARK_TEXT_PRIMARY
            focus_color  = Win11Colors.DARK_ACCENT
            hover_color  = Win11Colors.DARK_HOVER
        else:
            bg_color     = Win11Colors.LIGHT_SURFACE
            border_color = Win11Colors.LIGHT_BORDER
            text_color   = Win11Colors.LIGHT_TEXT_PRIMARY
            focus_color  = Win11Colors.LIGHT_ACCENT
            hover_color  = Win11Colors.LIGHT_HOVER

        self.setStyleSheet(f"""
            /* ── 按钮本体 ── */
            QComboBox {{
                background-color: {bg_color.name()};
                border: 2px solid {border_color.name()};
                border-radius: 12px;
                padding: 8px 16px;
                color: {text_color.name()};
                min-width: 120px;
                min-height: 24px;
                font-size: 14px;
                font-weight: 500;
            }}
            QComboBox:hover  {{ background-color: {hover_color.name()}; }}
            QComboBox:focus  {{ border-color: {focus_color.name()}; }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 32px;
                border: none;
            }}

            /* ── 弹出列表：背景透明，由 _DropdownPainter 提供圆角背景 ── */
            QComboBox QAbstractItemView {{
                background-color: transparent;
                border: none;
                padding: 6px;
                outline: none;
                color: {text_color.name()};
            }}
            QComboBox QAbstractItemView::item {{
                padding: 5px 16px;        /* 原为 10px 28px，缩小上下内边距 */
                border-radius: 10px;
                margin: 1px 4px;          /* 原为 2px 4px，同步缩小垂直间距 */
                background-color: transparent;
                min-height: 18px;         /* 原为 24px，降低最小高度 */
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {hover_color.name()};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {focus_color.name()};
                color: #ffffff;
            }}
        """)

    def showPopup(self):
        """覆写弹出方法，确保每次弹出时阴影标志都生效"""
        view_window = self.view().window()
        view_window.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
        view_window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        view_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)  # 补充：确保背景彻底透明

        super().showPopup()

        # ── super().showPopup() 之后 HWND 才稳定，在此彻底压制 Windows 系统阴影 ──
        if platform.system() == "Windows":
            try:
                import ctypes
                from ctypes import wintypes

                hwnd = int(view_window.winId())

                # 1. 禁用 DWM 非客户区渲染 (处理部分 Windows 版本的阴影)
                DWMWA_NCRENDERING_POLICY = 2
                DWMNCRP_DISABLED = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_NCRENDERING_POLICY,
                    ctypes.byref(DWMNCRP_DISABLED),
                    ctypes.sizeof(DWMNCRP_DISABLED)
                )

                # 2. 移除 Windows 传统菜单的 CS_DROPSHADOW 样式 (解决图中深色直角阴影的关键)
                GCL_STYLE = -26
                CS_DROPSHADOW = 0x00020000
                user32 = ctypes.windll.user32

                # 兼容 32 位和 64 位系统 API
                if ctypes.sizeof(ctypes.c_void_p) == 8:
                    GetClassLong = user32.GetClassLongPtrW
                    GetClassLong.argtypes = [wintypes.HWND, ctypes.c_int]
                    GetClassLong.restype = ctypes.c_void_p

                    SetClassLong = user32.SetClassLongPtrW
                    SetClassLong.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
                    SetClassLong.restype = ctypes.c_void_p
                else:
                    GetClassLong = user32.GetClassLongW
                    GetClassLong.argtypes = [wintypes.HWND, ctypes.c_int]
                    GetClassLong.restype = ctypes.c_long

                    SetClassLong = user32.SetClassLongW
                    SetClassLong.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
                    SetClassLong.restype = ctypes.c_long

                # 获取当前样式并剥离 CS_DROPSHADOW
                style = GetClassLong(hwnd, GCL_STYLE)
                if style is not None and (style & CS_DROPSHADOW):
                    SetClassLong(hwnd, GCL_STYLE, style & ~CS_DROPSHADOW)

            except Exception:
                pass

    def update_theme(self):
        """更新主题"""
        self._setup_style()
        # 强制弹出窗口重绘，使过滤器用新主题色重绘
        self.view().window().update()

    def wheelEvent(self, event):
        # 禁用滚轮切换选项，但允许事件继续传递给父级（如滚动区域）
        event.ignore()


class ModernCheckBox(QCheckBox):
    """Material You (M3) 风格复选框"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(32)

        # 动画属性初始值
        self._bg_color      = QColor("transparent")
        self._check_progress = 0.0   # 0.0=未绘制  1.0=完整勾
        self._state_layer_r  = 0.0   # 状态层半径

        self._animation_duration = 200
        self._setup_animations()
        self._update_stylesheet()
        self.stateChanged.connect(self._animate_state_change)

    # ── 动画属性 ──────────────────────────────────────────────────
    @Property(QColor)
    def backgroundColor(self):
        return self._bg_color

    @backgroundColor.setter
    def backgroundColor(self, color):
        self._bg_color = color
        self.update()

    @Property(float)
    def checkProgress(self):
        return self._check_progress

    @checkProgress.setter
    def checkProgress(self, v):
        self._check_progress = max(0.0, min(1.0, v))
        self.update()

    @Property(float)
    def stateLayerRadius(self):
        return self._state_layer_r

    @stateLayerRadius.setter
    def stateLayerRadius(self, v):
        self._state_layer_r = v
        self.update()

    def _setup_animations(self):
        self._bg_anim = QPropertyAnimation(self, b"backgroundColor")
        self._bg_anim.setDuration(self._animation_duration)
        self._bg_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._check_anim = QPropertyAnimation(self, b"checkProgress")
        self._check_anim.setDuration(self._animation_duration)
        self._check_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._layer_anim = QPropertyAnimation(self, b"stateLayerRadius")
        self._layer_anim.setDuration(150)
        self._layer_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _update_stylesheet(self):
        self.setStyleSheet("""
            QCheckBox { spacing: 10px; background: transparent; }
            QCheckBox::indicator { width: 0px; height: 0px; }
        """)

    def _get_theme_colors(self):
        app = QApplication.instance()
        is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        if is_dark:
            return {
                "border":       Win11Colors.DARK_BORDER,
                "checked_bg":   Win11Colors.DARK_ACCENT,
                "text":         Win11Colors.DARK_TEXT_PRIMARY,
                "checkmark":    Win11Colors.DARK_TEXT_PRIMARY,
                "state_layer":  QColor(Win11Colors.DARK_ACCENT.red(),
                                       Win11Colors.DARK_ACCENT.green(),
                                       Win11Colors.DARK_ACCENT.blue(), 30),
            }
        else:
            return {
                "border":       Win11Colors.LIGHT_BORDER,
                "checked_bg":   Win11Colors.LIGHT_ACCENT,
                "text":         Win11Colors.LIGHT_TEXT_PRIMARY,
                "checkmark":    Win11Colors.LIGHT_TEXT_PRIMARY,
                "state_layer":  QColor(Win11Colors.LIGHT_ACCENT.red(),
                                       Win11Colors.LIGHT_ACCENT.green(),
                                       Win11Colors.LIGHT_ACCENT.blue(), 40),
            }

    def _animate_state_change(self, state):
        colors = self._get_theme_colors()
        checked = (state == Qt.CheckState.Checked.value)

        # 背景色过渡
        self._bg_anim.stop()
        self._bg_anim.setStartValue(self._bg_color)
        self._bg_anim.setEndValue(
            colors["checked_bg"] if checked else QColor("transparent")
        )
        self._bg_anim.start()

        # 勾选路径进度
        self._check_anim.stop()
        self._check_anim.setStartValue(self._check_progress)
        self._check_anim.setEndValue(1.0 if checked else 0.0)
        self._check_anim.start()

    def setChecked(self, checked):
        """直接设置状态，跳过动画"""
        super().setChecked(checked)
        colors = self._get_theme_colors()
        self._bg_color       = colors["checked_bg"] if checked else QColor("transparent")
        self._check_progress = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event):
        from PySide6.QtCore import QRectF, QPointF
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        colors = self._get_theme_colors()
        rect    = self.rect()
        box_size = 18
        box_x    = 0
        box_y    = (rect.height() - box_size) // 2
        cx       = box_x + box_size / 2
        cy       = box_y + box_size / 2

        # ── 1. 状态层（hover / press 光晕）────────────────────────
        if self._state_layer_r > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(colors["state_layer"]))
            sr = self._state_layer_r
            painter.drawEllipse(QRectF(cx - sr, cy - sr, sr * 2, sr * 2))

        # ── 2. 复选框背景 ─────────────────────────────────────────
        box_rect = QRectF(box_x, box_y, box_size, box_size)
        box_r    = 6.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(box_rect, box_r, box_r)

        # ── 3. 边框（仅未选中时显示）─────────────────────────────
        if self._check_progress < 1.0:
            border_alpha = int(255 * (1.0 - self._check_progress))
            border_color = QColor(colors["border"])
            border_color.setAlpha(border_alpha)
            pen = QPen(border_color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                box_rect.adjusted(1, 1, -1, -1), box_r, box_r
            )

        # ── 4. 勾选路径（逐笔绘出动画）──────────────────────────
        if self._check_progress > 0:
            # M3 checkmark 两段：短段(起点→拐点) 长段(拐点→终点)
            p1 = QPointF(box_x + 4,  cy)
            p2 = QPointF(box_x + 7.5, box_y + box_size - 4.5)
            p3 = QPointF(box_x + box_size - 3.5, box_y + 4)

            seg1_len = ((p2.x()-p1.x())**2 + (p2.y()-p1.y())**2) ** 0.5
            seg2_len = ((p3.x()-p2.x())**2 + (p3.y()-p2.y())**2) ** 0.5
            total    = seg1_len + seg2_len
            progress_px = self._check_progress * total

            check_pen = QPen(colors["checkmark"], 2)
            check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            if progress_px <= seg1_len:
                # 仅绘制第一段的一部分
                t = progress_px / seg1_len
                mid = QPointF(p1.x() + t * (p2.x() - p1.x()),
                              p1.y() + t * (p2.y() - p1.y()))
                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(mid)
                painter.drawPath(path)
            else:
                # 第一段完整 + 第二段部分
                t = (progress_px - seg1_len) / seg2_len
                end = QPointF(p2.x() + t * (p3.x() - p2.x()),
                              p2.y() + t * (p3.y() - p2.y()))
                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(p2)
                path.lineTo(end)
                painter.drawPath(path)

        # ── 5. 文本 ───────────────────────────────────────────────
        if self.text():
            text_rect = rect.adjusted(box_size + 10, 0, 0, 0)
            painter.setPen(QPen(colors["text"]))
            font = self.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self.text())

    # ── 状态层交互 ────────────────────────────────────────────────
    def _animate_layer(self, target):
        self._layer_anim.stop()
        self._layer_anim.setStartValue(self._state_layer_r)
        self._layer_anim.setEndValue(float(target))
        self._layer_anim.start()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._animate_layer(18)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._animate_layer(0)

    def mousePressEvent(self, event):
        self._animate_layer(14)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._animate_layer(18 if self.underMouse() else 0)
        super().mouseReleaseEvent(event)

    # checkOpacity 保留兼容
    @Property(float)
    def checkOpacity(self):
        return self._check_progress

    @checkOpacity.setter
    def checkOpacity(self, v):
        self._check_progress = v
        self.update()

    def update_theme(self):
        self._update_stylesheet()
        self.setChecked(self.isChecked())


class MarqueeButton(QPushButton):
    """文字超出按钮宽度时自动向左滚动显示的按钮"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._scroll_offset = 0
        self._pause_ticks = 0
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(35)
        self._scroll_timer.timeout.connect(self._tick)

    def _text_width(self):
        return self.fontMetrics().horizontalAdvance(self.text())

    def _avail_width(self):
        return max(1, self.width() - 16)

    def _check_and_start(self):
        if self._text_width() > self._avail_width():
            self._scroll_offset = 0
            self._pause_ticks = 25
            if not self._scroll_timer.isActive():
                self._scroll_timer.start()
        else:
            self._scroll_timer.stop()
            self._scroll_offset = 0
            self.update()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self._check_and_start)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._check_and_start()

    def _tick(self):
        if self._pause_ticks > 0:
            self._pause_ticks -= 1
            # 新增：当末尾的暂停倒计时结束时，直接重置回开头，并再次短暂暂停
            if self._pause_ticks == 0 and self._scroll_offset > 0:
                self._scroll_offset = 0
                self._pause_ticks = 25
                self.update()
            return

        max_offset = self._text_width() - self._avail_width()
        if max_offset <= 0:
            self._scroll_timer.stop()
            self._scroll_offset = 0
            self.update()
            return

        # 完美循环逻辑：步进 -> 触顶 -> 暂停 -> 重置
        if self._scroll_offset < max_offset:
            self._scroll_offset += 1
        else:
            self._scroll_offset = max_offset
            self._pause_ticks = 40  # 触顶后进入40帧的停顿

        self.update()

    def _reset_to_start(self):
        self._scroll_offset = 0
        self._pause_ticks = 25
        self.update()

    def paintEvent(self, event):
        from PySide6.QtWidgets import QStyleOptionButton, QStyle
        if self._text_width() <= self._avail_width():
            super().paintEvent(event)
            return
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 画按钮背景/边框，不含文字
        opt.text = ""
        self.style().drawControl(QStyle.ControlElement.CE_PushButton, opt, painter, self)
        # 获取文字区域并裁剪
        text_rect = self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, opt, self)
        painter.setClipRect(text_rect)
        painter.setFont(self.font())
        color = opt.palette.buttonText().color() if self.isEnabled() else opt.palette.mid().color()
        painter.setPen(color)
        draw_rect = text_rect.adjusted(-self._scroll_offset, 0, self._text_width(), 0)
        painter.drawText(draw_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        painter.end()

    def changeEvent(self, event):
        """监听样式和字体变化，字体加粗可能导致文字超宽，需要重新计算"""
        super().changeEvent(event)
        # 如果字体改变了（通过 QSS 加粗引发），重新检测是否需要滚动
        if event.type() in [QEvent.Type.FontChange, QEvent.Type.StyleChange]:
            self._check_and_start()


class ScrollingListDelegate(QStyledItemDelegate):
    """支持文字超长时：未选中/未悬浮显示省略号，选中/悬浮时自动滚动显示的列表委托"""

    def __init__(self, list_widget):
        super().__init__(list_widget)
        self._lw = list_widget
        self._hovered_row = -1

        # 使用字典按 item 的内存地址 (id) 独立跟踪多项的滚动状态
        # 结构： {id(item): {'offset': 0, 'pause_ticks': 20, 'waiting_for_reset': False, 'item': item}}
        self._scroll_states = {}

        self._timer = QTimer(self)
        self._timer.setInterval(35)
        self._timer.timeout.connect(self._tick)

        list_widget.viewport().setMouseTracking(True)
        list_widget.viewport().installEventFilter(self)
        list_widget.destroyed.connect(self._on_lw_destroyed)

        # 监听列表项的选中状态变化
        list_widget.itemSelectionChanged.connect(self._update_active_rows)

        self._lw_alive = True

    def _on_lw_destroyed(self):
        self._lw_alive = False
        self._timer.stop()
        self._lw = None

    def _is_valid(self):
        return self._lw_alive and self._lw is not None

    def eventFilter(self, obj, event):
        if not self._is_valid():
            return False
        try:
            if obj == self._lw.viewport():
                if event.type() == QEvent.Type.MouseMove:
                    pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
                    item = self._lw.itemAt(pos)
                    self._set_hovered_row(self._lw.row(item) if item else -1)
                elif event.type() == QEvent.Type.Leave:
                    self._set_hovered_row(-1)
        except RuntimeError:
            self._lw_alive = False
        return super().eventFilter(obj, event)

    def _set_hovered_row(self, row):
        if not self._is_valid():
            return
        if row == self._hovered_row:
            return
        self._hovered_row = row
        self._update_active_rows()

        if row >= 0:
            self._lw.update(self._lw.model().index(row, 0))

    def _update_active_rows(self):
        """核心：更新当前需要滚动的激活项（选中 + 悬浮）"""
        if not self._is_valid():
            return

        # 修复：使用字典映射 id -> item，彻底避免把 QListWidgetItem 放入 set 中引发 unhashable 错误
        active_items_map = {}

        # 1. 获取悬浮项
        if self._hovered_row >= 0:
            hovered_item = self._lw.item(self._hovered_row)
            if hovered_item:
                active_items_map[id(hovered_item)] = hovered_item

        # 2. 获取所有选中项
        for item in self._lw.selectedItems():
            active_items_map[id(item)] = item

        # 3. 提取所有激活项的 ID 集合，并初始化新激活项的状态
        active_ids = set(active_items_map.keys())
        for item_id, item in active_items_map.items():
            if item_id not in self._scroll_states:
                self._scroll_states[item_id] = {'offset': 0, 'pause_ticks': 20, 'waiting_for_reset': False, 'item': item}

        # 4. 移除不再激活的项（恢复省略号）
        for item_id in list(self._scroll_states.keys()):
            if item_id not in active_ids:
                item_to_update = self._scroll_states[item_id]['item']
                del self._scroll_states[item_id]
                try:
                    # 获取底层的对象来刷新 UI
                    row = self._lw.row(item_to_update)
                    if row >= 0:
                        self._lw.update(self._lw.model().index(row, 0))  # 重绘以显示省略号
                except RuntimeError:
                    pass

        # 控制定时器启停
        if self._scroll_states:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _tick(self):
        """每一帧推进各激活项的动画偏移"""
        if not self._is_valid():
            self._timer.stop()
            return

        try:
            for item_id, state in list(self._scroll_states.items()):
                item = state['item']
                row = self._lw.row(item)

                if row < 0:
                    del self._scroll_states[item_id]
                    continue

                if state['pause_ticks'] > 0:
                    state['pause_ticks'] -= 1
                    if state['pause_ticks'] == 0 and state['waiting_for_reset']:
                        state['offset'] = 0
                        state['pause_ticks'] = 20
                        state['waiting_for_reset'] = False
                        self._lw.update(self._lw.model().index(row, 0))
                    continue

                fm = self._lw.fontMetrics()

                # 修复核心：换回稳定的 viewport().width()，避免 visualItemRect 的判定误差
                max_offset = fm.horizontalAdvance(item.text()) - (self._lw.viewport().width() - 32)

                if max_offset <= 0:
                    state['offset'] = 0
                    continue

                # 完美平滑滚动：步进 -> 触达最大值 -> 打上标记并暂停
                if state['offset'] < max_offset:
                    state['offset'] += 1
                else:
                    state['offset'] = max_offset
                    state['pause_ticks'] = 40
                    state['waiting_for_reset'] = True

                self._lw.update(self._lw.model().index(row, 0))
        except RuntimeError:
            self._lw_alive = False
            self._timer.stop()

    def paint(self, painter, option, index):
        if not self._is_valid():
            super().paint(painter, option, index)
            return
        try:
            item = self._lw.item(index.row())
            if not item:
                super().paint(painter, option, index)
                return

            text = item.text()
            fm = option.fontMetrics
            avail_width = option.rect.width() - 32

            # 使用 id(item) 来检查状态
            item_id = id(item)
            is_active = item_id in self._scroll_states

            # ===== 核心判断 =====
            # 无论文字是否超出宽度，只要处于激活状态（悬停/选中），
            # 就统一使用自定义绘制，以此保证短文本也能拥有相同的悬停浮动（偏移）效果。
            if not is_active:
                super().paint(painter, option, index)
                return

            # ===== 激活状态，自行绘制 =====
            from PySide6.QtWidgets import QApplication, QStyle
            painter.save()

            # 手动补齐悬停状态，确保底层样式能准确渲染悬停背景色
            if self._hovered_row == index.row():
                option.state |= QStyle.StateFlag.State_MouseOver

            # 1. 绘制带有圆角的背景/焦点框（复用QSS样式）
            QApplication.style().drawPrimitive(
                QStyle.PrimitiveElement.PE_PanelItemViewItem, option, painter, self._lw)

            # 2. 获取该项独立的滚动偏移 (使用 id(item) 作为键)
            state = self._scroll_states.get(item_id, {'offset': 0})

            # 判断文字是否真超长，未超长的短文本强制偏移量为0，只应用边距浮动效果
            is_truncated = fm.horizontalAdvance(text) > avail_width
            offset = state.get('offset', 0) if is_truncated else 0

            # 3. 裁剪避免文字画出边界
            painter.setClipRect(option.rect.adjusted(16, 0, -16, 0))

            # 4. 根据选中状态设定文字颜色
            if option.state & QStyle.StateFlag.State_Selected:
                painter.setPen(option.palette.highlightedText().color())
            else:
                painter.setPen(option.palette.text().color())

            painter.setFont(option.font)

            # 5. 应用滚动偏移量并绘制文字
            draw_rect = option.rect.adjusted(16 - offset, 0, fm.horizontalAdvance(text), 0)
            painter.drawText(draw_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            painter.restore()

        except RuntimeError:
            self._lw_alive = False
            super().paint(painter, option, index)