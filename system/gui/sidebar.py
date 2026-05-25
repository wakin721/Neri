from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect, QPushButton
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect, QTimer, QParallelAnimationGroup
from PySide6.QtGui import (
    QPixmap, QFont, QPainter, QPainterPath, QColor, QBrush,
    QPen, QLinearGradient, QFontMetrics, QPalette, QIcon
)
from PySide6.QtSvg import QSvgRenderer
from PIL import Image
import os

from system.config import APP_DISPLAY_VERSION
from system.utils import resource_path
from system.gui.ui_components import Win11Colors


class FluentIcon:
    """Fluent Design图标字符"""
    HAMBURGER = "☰"
    CHEVRON_RIGHT = "☰"
    HOME = "⌂"
    IMAGE = "📷"
    SETTINGS = "⚙"
    INFO = "ℹ"
    UPDATE = "🔄"


class ModernNavigationButton(QWidget):
    """现代化导航按钮 - 自动适应浅色/深色主题"""

    clicked = Signal()

    def __init__(self, text="", icon_text="", icon_path=None, parent=None):
        super().__init__(parent)
        self._text = text
        self._icon_text = icon_text
        self._icon_path = icon_path
        self._icon_pixmap = None
        self._is_active = False
        self._is_hovered = False
        self._is_collapsed = True  # 默认折叠状态
        self._corner_radius = 14
        self._animation_duration = 150

        # 设置基本属性
        self.setFixedHeight(40)
        self.setMinimumWidth(160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # 设置透明背景，继承父组件背景
        self.setStyleSheet("QWidget { background-color: transparent; }")

        # 加载SVG图标
        self._load_svg_icon()

        # 设置动画
        self._setup_animations()

        # 初始样式
        self._update_colors()

        # 添加这行：立即应用折叠状态
        self.set_collapsed(True)

    def _load_svg_icon(self):
        """加载SVG图标"""
        if self._icon_path and os.path.exists(self._icon_path):
            try:
                # 使用QSvgRenderer渲染SVG
                renderer = QSvgRenderer(self._icon_path)
                if renderer.isValid():
                    # 创建20x20的图标
                    self._icon_pixmap = QPixmap(20, 20)
                    self._icon_pixmap.fill(Qt.GlobalColor.transparent)

                    painter = QPainter(self._icon_pixmap)
                    renderer.render(painter)
                    painter.end()
                else:
                    print(f"SVG渲染失败: {self._icon_path}")
                    self._icon_pixmap = None
            except Exception as e:
                print(f"加载SVG图标失败 {self._icon_path}: {e}")
                self._icon_pixmap = None

    def _setup_animations(self):
        """设置动画效果"""
        self._hover_animation = QPropertyAnimation(self, b"geometry")
        self._hover_animation.setDuration(self._animation_duration)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def _update_colors(self):
        """更新颜色方案 - 使用 Win11Colors 统一色彩系统"""
        is_dark = self._is_dark_theme()

        if is_dark:
            self._bg_normal = QColor(0, 0, 0, 0)
            self._bg_hover = Win11Colors.DARK_HOVER  # #3a2a30
            self._bg_active = Win11Colors.DARK_ACCENT  # #5d3a4f
            self._text_normal = Win11Colors.DARK_TEXT_SECONDARY  # rgba柔和白
            self._text_active = Win11Colors.DARK_TEXT_PRIMARY  # 纯白
            self._icon_normal = Win11Colors.DARK_TEXT_SECONDARY
            self._icon_active = Win11Colors.DARK_TEXT_PRIMARY
        else:
            self._bg_normal = QColor(0, 0, 0, 0)
            self._bg_hover = Win11Colors.LIGHT_HOVER  # #f0eaec
            self._bg_active = Win11Colors.LIGHT_ACCENT  # #dbbcc2
            self._text_normal = Win11Colors.LIGHT_TEXT_SECONDARY
            self._text_active = Win11Colors.LIGHT_TEXT_PRIMARY
            self._icon_normal = Win11Colors.LIGHT_TEXT_SECONDARY
            self._icon_active = Win11Colors.LIGHT_TEXT_PRIMARY

        self.update()

    def _create_colored_pixmap(self, original_pixmap, color):
        """根据颜色创建着色的图标"""
        if not original_pixmap:
            return None

        colored_pixmap = QPixmap(original_pixmap.size())
        colored_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(colored_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # 绘制原始图标
        painter.drawPixmap(0, 0, original_pixmap)

        # 应用颜色覆盖
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)

        painter.end()
        return colored_pixmap

    def set_collapsed(self, collapsed):
        """设置折叠状态"""
        self._is_collapsed = collapsed
        if collapsed:
            self.setFixedWidth(48)
            self.setToolTip(self._text)
        else:
            self.setMinimumWidth(160)
            self.setToolTip("")
        self.update()

    def set_active(self, active):
        """设置激活状态"""
        if self._is_active != active:
            self._is_active = active
            self.update()

    def is_active(self):
        """返回是否激活"""
        return self._is_active

    def enterEvent(self, event):
        """鼠标进入事件"""
        if not self._is_hovered:
            self._is_hovered = True
            self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        if self._is_hovered:
            self._is_hovered = False
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        """绘制事件 - Material You 胶囊风格"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 2, -4, -2)

        if self._is_active:
            bg_color = self._bg_active
            text_color = self._text_active
            icon_color = self._icon_active
        elif self._is_hovered:
            bg_color = self._bg_hover
            text_color = self._text_normal
            icon_color = self._icon_normal
        else:
            bg_color = self._bg_normal
            text_color = self._text_normal
            icon_color = self._icon_normal

        # 绘制胶囊背景（激活或悬停时）
        if bg_color.alpha() > 0:
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.PenStyle.NoPen)
            path = QPainterPath()
            path.addRoundedRect(rect, self._corner_radius, self._corner_radius)
            painter.drawPath(path)

        # ← 原来左侧指示条的绘制代码全部删除

        # 绘制 SVG 图标
        if self._icon_pixmap:
            colored_icon = self._create_colored_pixmap(self._icon_pixmap, icon_color)
            if colored_icon:
                if self._is_collapsed:
                    icon_x = rect.center().x() - 10
                    icon_y = rect.center().y() - 10
                else:
                    icon_x = rect.x() + 20
                    icon_y = rect.center().y() - 10
                painter.drawPixmap(icon_x, icon_y, colored_icon)
        elif self._icon_text:
            painter.setPen(QPen(icon_color))
            icon_font = QFont("Segoe UI Symbol", 16)
            painter.setFont(icon_font)
            if self._is_collapsed:
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._icon_text)
            else:
                icon_rect = QRect(rect.x() + 16, rect.y(), 24, rect.height())
                painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, self._icon_text)

        # 绘制文本（仅展开状态）
        if not self._is_collapsed and self._text:
            painter.setPen(QPen(text_color))
            text_font = QFont("Segoe UI", 9,
                              QFont.Weight.DemiBold if self._is_active else QFont.Weight.Normal)
            painter.setFont(text_font)
            text_rect = rect.adjusted(48, 0, -16, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self._text)

    def update_theme(self):
        """更新主题"""
        self._update_colors()


class CollapseButton(QPushButton):
    """折叠按钮 - 自动适应浅色/深色主题"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._is_collapsed = True  # 默认为折叠状态

        # 加载SVG图标
        self._bars_icon_path = resource_path(os.path.join("res", "icon", "bars.svg"))
        self._bars_pixmap = None
        self._load_bars_icon()

        self._setup_style()

    def _load_bars_icon(self):
        """加载bars.svg图标"""
        if os.path.exists(self._bars_icon_path):
            try:
                from PySide6.QtSvg import QSvgRenderer
                renderer = QSvgRenderer(self._bars_icon_path)
                if renderer.isValid():
                    # 创建16x16的图标
                    self._bars_pixmap = QPixmap(16, 16)
                    self._bars_pixmap.fill(Qt.GlobalColor.transparent)

                    painter = QPainter(self._bars_pixmap)
                    renderer.render(painter)
                    painter.end()
                else:
                    print(f"SVG渲染失败: {self._bars_icon_path}")
            except Exception as e:
                print(f"加载bars.svg图标失败: {e}")

    def _create_colored_pixmap(self, original_pixmap, color):
        """根据颜色创建着色的图标"""
        if not original_pixmap:
            return None

        colored_pixmap = QPixmap(original_pixmap.size())
        colored_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(colored_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        painter.end()

        return colored_pixmap

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def set_collapsed(self, collapsed):
        """设置折叠状态"""
        self._is_collapsed = collapsed
        self._update_icon()

    def _setup_style(self):
        """设置样式"""
        is_dark = self._is_dark_theme()

        if is_dark:
            # 深色主题
            hover_color = "rgba(255, 255, 255, 13)"
            press_color = "rgba(255, 255, 255, 9)"
        else:
            # 浅色主题
            hover_color = "rgba(0, 0, 0, 13)"
            press_color = "rgba(0, 0, 0, 9)"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {press_color};
            }}
        """)

        self._update_icon()

    def _update_icon(self):
        """更新图标"""
        is_dark = self._is_dark_theme()

        if self._bars_pixmap:
            # 使用SVG图标
            if is_dark:
                text_color = QColor(255, 255, 255, 220)
            else:
                text_color = QColor(0, 0, 0, 200)

            colored_icon = self._create_colored_pixmap(self._bars_pixmap, text_color)
            if colored_icon:
                icon = QIcon(colored_icon)
                self.setIcon(icon)
                self.setIconSize(colored_icon.size())
                self.setText("")  # 清空文字
            else:
                # 回退到文字图标
                self._set_text_icon()
        else:
            # 回退到文字图标
            self._set_text_icon()

        # 设置工具提示
        if self._is_collapsed:
            self.setToolTip("展开导航栏")
        else:
            self.setToolTip("折叠导航栏")

    def _set_text_icon(self):
        """设置文字图标作为回退"""
        is_dark = self._is_dark_theme()

        self.setIcon(QIcon())  # 清空图标
        if self._is_collapsed:
            self.setText(FluentIcon.CHEVRON_RIGHT)
        else:
            self.setText(FluentIcon.HAMBURGER)

        # 根据主题设置文字颜色
        if is_dark:
            text_color = "rgba(255, 255, 255, 220)"
        else:
            text_color = "rgba(0, 0, 0, 200)"

        # 重新设置文字样式，确保透明背景
        self.setStyleSheet(self.styleSheet() + f"""
            QPushButton {{
                font-family: "Segoe UI Symbol";
                font-size: 14px;
                color: {text_color};
                background-color: transparent;
            }}
        """)

    def update_theme(self):
        """更新主题"""
        self._setup_style()


class ModernSeparator(QFrame):
    """现代化分隔线 - 自动适应浅色/深色主题"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self._update_style()

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def _update_style(self):
        """更新样式 - 使用 Win11Colors 边框色"""
        is_dark = self._is_dark_theme()

        if is_dark:
            border_color = Win11Colors.DARK_BORDER.name()  # #42323a
        else:
            border_color = Win11Colors.LIGHT_BORDER.name()  # #e2d8db

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {border_color};
                border: none;
            }}
        """)

    def update_theme(self):
        """更新主题"""
        self._update_style()


class UpdateNotificationLabel(QLabel):
    """更新通知标签 - 自动适应浅色/深色主题"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Segoe UI", 8, QFont.Weight.Medium))
        self._setup_style()
        self.hide()

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def _setup_style(self):
        """设置样式"""
        # 黄色通知配色在两种主题下都保持一致
        bg_color = "rgba(255, 193, 7, 20)"
        text_color = "rgb(255, 193, 7)"
        border_color = "rgba(255, 193, 7, 40)"

        self.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                background-color: {bg_color};
                border: 1px solid {border_color};
                padding: 4px 8px;
                border-radius: 4px;
            }}
        """)

    def show_notification(self, text="发现新版本"):
        """显示通知"""
        self.setText(text)
        self.show()
        self._create_fade_effect()

    def _create_fade_effect(self):
        """创建淡入效果"""
        self._fade_timer = QTimer()
        self._fade_count = 0
        self._fade_timer.timeout.connect(self._fade)
        self._fade_timer.start(300)

    def _fade(self):
        """淡入实现"""
        self._fade_count += 1
        if self._fade_count <= 4:
            self.setVisible(not self.isVisible())
        else:
            self._fade_timer.stop()
            self.show()

    def update_theme(self):
        """更新主题"""
        self._setup_style()


class AppLogoWidget(QWidget):
    """应用Logo组件 - 自动适应浅色/深色主题"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_collapsed = True  # 默认为折叠状态
        self._setup_ui()
        self._setup_style()

    def _setup_style(self):
        """设置背景样式"""
        self.setStyleSheet("QWidget { background-color: transparent; border: none; }")

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def _setup_ui(self):
        """设置UI"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Logo容器 - 固定高度确保位置稳定
        self.logo_container = QWidget()
        self.logo_container.setFixedHeight(40)
        self.logo_container.setStyleSheet("QWidget { background-color: transparent; border: none; }")
        self.logo_layout = QHBoxLayout(self.logo_container)
        self.logo_layout.setContentsMargins(8, 4, 8, 4)
        self.logo_layout.setSpacing(8)

        # 尝试加载应用图标
        self.icon_label = None
        try:
            ico_path = resource_path(os.path.join("res", "ico.ico"))
            if os.path.exists(ico_path):
                pil_image = Image.open(ico_path)
                pil_image = pil_image.resize((32, 32), Image.Resampling.LANCZOS)

                import io
                byte_array = io.BytesIO()
                pil_image.save(byte_array, format='PNG')
                byte_array.seek(0)

                pixmap = QPixmap()
                pixmap.loadFromData(byte_array.getvalue())

                self.icon_label = QLabel()
                self.icon_label.setPixmap(pixmap)
                self.icon_label.setFixedSize(32, 32)
                self.icon_label.setStyleSheet("QLabel { background-color: transparent; border: none; }")
                self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.logo_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        except Exception as e:
            print(f"加载图标失败: {e}")

        # 应用名称
        self.app_name = QLabel("Neri")
        self.app_name.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        self.app_name.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._update_text_color(self.app_name)
        self.logo_layout.addWidget(self.app_name, 0, Qt.AlignmentFlag.AlignVCenter)

        self.logo_layout.addStretch()

        # 添加到主布局时使用顶部对齐，确保位置固定
        self.layout.addWidget(self.logo_container, 0, Qt.AlignmentFlag.AlignTop)

        # 初始化折叠状态 - 放在UI构建完成后
        self.set_collapsed(self._is_collapsed)

        self.updateGeometry()

    def set_collapsed(self, collapsed):
        """设置折叠状态"""
        self._is_collapsed = collapsed
        if self.app_name:
            self.app_name.setVisible(not collapsed)

    def _update_text_color(self, label):
        """更新文本颜色"""
        is_dark = self._is_dark_theme()

        if is_dark:
            text_color = "rgba(255, 255, 255, 220)"
        else:
            text_color = "rgba(0, 0, 0, 200)"

        label.setStyleSheet(f"color: {text_color}; background-color: transparent; border: none;")

    def update_theme(self):
        """更新主题"""
        self._setup_style()
        if hasattr(self, 'app_name'):
            self._update_text_color(self.app_name)


class Sidebar(QWidget):
    """侧边栏导航 - 自动适应浅色/深色主题，支持折叠，默认折叠，展开时悬浮"""

    # 信号定义
    page_requested = Signal(str)
    collapse_toggled = Signal(bool)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.nav_buttons = {}
        self._is_collapsed = True  # 默认为折叠状态
        self._animation_duration = 250

        self._setup_ui()
        self._apply_theme()
        self._setup_animations()

        # 设置悬浮效果和阴影
        self._setup_floating_style()

    def _is_dark_theme(self):
        """检测是否为深色主题"""
        return get_theme_aware_color(False, True)

    def _setup_ui(self):
        """设置UI"""
        # 默认为折叠状态的宽度
        self.setFixedWidth(64)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        # 顶部区域（包含折叠按钮和Logo）
        top_widget = QWidget()
        top_widget.setStyleSheet("QWidget { background-color: transparent; }")
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # 折叠按钮
        button_container = QWidget()
        button_container.setStyleSheet("QWidget { background-color: transparent; }")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(4, 4, 4, 4)

        self.collapse_button = CollapseButton()
        self.collapse_button.clicked.connect(self._toggle_collapse)
        button_layout.addWidget(self.collapse_button)
        button_layout.addStretch()

        top_layout.addWidget(button_container)

        # Logo区域
        self.logo_widget = AppLogoWidget()
        top_layout.addWidget(self.logo_widget)

        layout.addWidget(top_widget)

        # 更新通知
        self.update_notification = UpdateNotificationLabel()
        layout.addWidget(self.update_notification, 0, Qt.AlignmentFlag.AlignCenter)

        # 分隔线 - 在折叠状态下减少间距
        layout.addSpacing(4)  # 减少间距
        separator = ModernSeparator()
        layout.addWidget(separator)
        layout.addSpacing(4)  # 减少间距

        # 导航按钮区域
        nav_container = QWidget()
        nav_container.setStyleSheet("QWidget { background-color: transparent; }")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(2)

        self._create_navigation_buttons(nav_layout)
        layout.addWidget(nav_container)

        # 弹性空间
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # 版本信息
        self.version_label = QLabel(APP_DISPLAY_VERSION)
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.version_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        self._update_version_color(self.version_label)
        # 默认隐藏版本信息（因为默认是折叠状态）
        self.version_label.hide()
        layout.addWidget(self.version_label)

        # 立即初始化所有组件的折叠状态
        self._update_collapsed_state()
        # 强制更新布局和重绘
        self.updateGeometry()
        self.update()
        self.repaint()

    def _setup_floating_style(self):
        """设置悬浮样式和阴影效果"""
        # 设置为悬浮窗口属性
        self.setWindowFlags(Qt.WindowType.Widget)

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(2, 0)
        self.setGraphicsEffect(shadow)

        # 提升层级，确保在其他widget上方
        self.raise_()

    def _setup_animations(self):
        """设置动画"""
        self._width_animation = QPropertyAnimation(self, b"maximumWidth")
        self._width_animation.setDuration(self._animation_duration)
        self._width_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._min_width_animation = QPropertyAnimation(self, b"minimumWidth")
        self._min_width_animation.setDuration(self._animation_duration)
        self._min_width_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 动画组
        self._animation_group = QParallelAnimationGroup()
        self._animation_group.addAnimation(self._width_animation)
        self._animation_group.addAnimation(self._min_width_animation)

    def _create_navigation_buttons(self, layout):
        """创建导航按钮"""
        menu_items = [
            ("settings", "开始", FluentIcon.HOME, resource_path(os.path.join("res", "icon", "home.svg"))),
            ("preview", "图像预览", FluentIcon.IMAGE, resource_path(os.path.join("res", "icon", "image.svg"))),
            ("species_validation", "物种校验", "🏷️", resource_path(os.path.join("res", "icon", "tag.svg"))),
            ("advanced", "高级设置", FluentIcon.SETTINGS, resource_path(os.path.join("res", "icon", "setting.svg"))),
            ("about", "关于", FluentIcon.INFO, resource_path(os.path.join("res", "icon", "info.svg")))
        ]

        for page_id, page_name, icon_text, icon_path in menu_items:
            button = ModernNavigationButton(page_name, icon_text, icon_path)
            button.clicked.connect(lambda checked=False, pid=page_id: self.page_requested.emit(pid))
            button.set_collapsed(True)
            layout.addWidget(button)
            self.nav_buttons[page_id] = button

    def _toggle_collapse(self):
        """切换折叠状态"""
        self._is_collapsed = not self._is_collapsed

        # 更新按钮状态
        self.collapse_button.set_collapsed(self._is_collapsed)

        # 设置动画目标值
        if self._is_collapsed:
            target_width = 64
        else:
            target_width = 180

        # 执行动画
        self._width_animation.setStartValue(self.width())
        self._width_animation.setEndValue(target_width)
        self._min_width_animation.setStartValue(self.minimumWidth())
        self._min_width_animation.setEndValue(target_width)

        self._animation_group.finished.connect(self._on_animation_finished)
        self._animation_group.start()

        # 立即更新子组件状态
        self._update_collapsed_state()

        # 发送信号
        self.collapse_toggled.emit(self._is_collapsed)

        # 更新悬浮效果
        self._update_floating_effect()

    def _update_floating_effect(self):
        """更新悬浮效果"""
        if not self._is_collapsed:
            # 展开时增强阴影效果
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(30)
            shadow.setColor(QColor(0, 0, 0, 120))
            shadow.setOffset(3, 0)
            self.setGraphicsEffect(shadow)

            # 确保在最上层
            self.raise_()
        else:
            # 折叠时减少阴影效果
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(1, 0)
            self.setGraphicsEffect(shadow)

    def _on_animation_finished(self):
        """动画完成回调"""
        self._animation_group.finished.disconnect()
        if self._is_collapsed:
            self.setFixedWidth(64)
        else:
            self.setFixedWidth(180)

    def _update_collapsed_state(self):
        """更新折叠状态"""
        # 更新Logo
        self.logo_widget.set_collapsed(self._is_collapsed)

        # 更新导航按钮
        for button in self.nav_buttons.values():
            button.set_collapsed(self._is_collapsed)

        # 更新版本标签
        if self._is_collapsed:
            self.version_label.hide()
        else:
            self.version_label.show()

    def _apply_theme(self):
        """应用主题 - 使用 Win11Colors 统一背景色"""
        is_dark = self._is_dark_theme()

        if is_dark:
            bg_color = Win11Colors.DARK_SURFACE.name()  # #261c20
        else:
            bg_color = Win11Colors.LIGHT_SURFACE.name()  # #f7f3f4

        self.setStyleSheet(f"""
            Sidebar {{
                background-color: {bg_color};
                border: none;
                border-radius: 8px;
            }}
            QWidget {{
                background-color: transparent;
            }}
        """)

    def _update_version_color(self, label):
        """更新版本标签颜色"""
        is_dark = self._is_dark_theme()

        if is_dark:
            text_color = "rgba(255, 255, 255, 140)"
        else:
            text_color = "rgba(0, 0, 0, 140)"

        label.setStyleSheet(f"color: {text_color}; background-color: transparent; border: none;")

    def set_active_button(self, page_id):
        """设置激活按钮"""
        for pid, button in self.nav_buttons.items():
            button.set_active(pid == page_id)

    def set_processing_state(self, is_processing):
        """设置处理状态"""
        for page_id, button in self.nav_buttons.items():
            if page_id in ["settings"]:
                button.setEnabled(True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                button.setEnabled(not is_processing)
                if is_processing:
                    button.setCursor(Qt.CursorShape.ForbiddenCursor)
                else:
                    button.setCursor(Qt.CursorShape.PointingHandCursor)

    def show_update_notification(self, message="发现新版本"):
        """显示更新通知"""
        self.update_notification.show_notification(message)

    def is_collapsed(self):
        """返回是否折叠"""
        return self._is_collapsed

    def update_theme(self):
        """更新主题"""
        self._apply_theme()

        # 更新所有子组件的主题
        if hasattr(self, 'logo_widget'):
            self.logo_widget.update_theme()

        if hasattr(self, 'collapse_button'):
            self.collapse_button.update_theme()

        if hasattr(self, 'update_notification'):
            self.update_notification.update_theme()

        for button in self.nav_buttons.values():
            if hasattr(button, 'update_theme'):
                button.update_theme()

        for separator in self.findChildren(ModernSeparator):
            separator.update_theme()

        if hasattr(self, 'version_label'):
            self._update_version_color(self.version_label)


# 主题感知的工具函数
def get_theme_aware_color(light_color, dark_color):
    """根据当前主题返回合适的颜色"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        palette = app.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        return dark_color if is_dark else light_color
    return light_color
