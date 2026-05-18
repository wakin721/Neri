# about_page.py

import os
from PIL import Image
import io

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import (
    Qt, Signal, QUrl, QPropertyAnimation, QEasingCurve, QPoint, QTimer,
    QParallelAnimationGroup
)
from PySide6.QtGui import (
    QPixmap, QFont, QPainter, QPainterPath, QColor, QDesktopServices,
    QPalette
)

# 假设这些模块存在于您的项目中
from system.config import APP_TITLE, APP_VERSION
from system.utils import resource_path
from system.gui.ui_components import Win11Colors


class Theme:
    """集中管理UI样式常量，便于全局调整和主题切换"""
    # 字体
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_XL = 28
    FONT_SIZE_L = 14
    FONT_SIZE_M = 12
    FONT_SIZE_S = 10
    FONT_SIZE_XS = 9

    # 尺寸与边距
    BORDER_RADIUS = 12
    CARD_PADDING = 24
    SPACING_L = 24
    SPACING_M = 15
    SPACING_S = 8

    # 动画
    ANIM_DURATION = 500
    ANIM_STAGGER_DELAY = 75

    @staticmethod
    def get_color(role, is_dark):
        """根据角色和主题模式获取颜色"""
        colors = {
            "bg_start": (Win11Colors.LIGHT_BACKGROUND, Win11Colors.DARK_BACKGROUND.lighter(110)),
            "bg_end": (Win11Colors.LIGHT_BACKGROUND.darker(105), Win11Colors.DARK_BACKGROUND),
            "card_bg": (Win11Colors.LIGHT_CARD, Win11Colors.DARK_CARD),
            "text_primary": (Win11Colors.LIGHT_TEXT_PRIMARY, Win11Colors.DARK_TEXT_PRIMARY),
            "text_secondary": (Win11Colors.LIGHT_TEXT_SECONDARY, Win11Colors.DARK_TEXT_SECONDARY),
            "text_tertiary": (
            Win11Colors.LIGHT_TEXT_SECONDARY.darker(110), Win11Colors.DARK_TEXT_SECONDARY.darker(110)),
            "accent": (Win11Colors.LIGHT_ACCENT, Win11Colors.DARK_ACCENT),
            "accent_hover": (Win11Colors.LIGHT_ACCENT.darker(110), Win11Colors.DARK_ACCENT.lighter(120)),
            "border": (Win11Colors.LIGHT_BORDER, Win11Colors.DARK_BORDER),
            "scrollbar": (Win11Colors.LIGHT_BORDER, Win11Colors.DARK_BORDER),
            "scrollbar_hover": (Win11Colors.LIGHT_ACCENT, Win11Colors.DARK_ACCENT),
        }
        return colors[role][1] if is_dark else colors[role][0]


class ThemedWidgetMixin:
    """提供主题更新能力的混入类"""

    def is_dark_theme(self):
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def update_theme(self):
        if hasattr(self, '_apply_about_page_theme'):
            self._apply_about_page_theme()
        for child in self.findChildren(QWidget):
            if isinstance(child, ThemedWidgetMixin) and child != self:
                child.update_theme()


class ModernCard(QFrame, ThemedWidgetMixin):
    """现代化卡片组件，具有圆角、边框和阴影效果"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shadow_effect = self._create_shadow()
        self.setGraphicsEffect(self.shadow_effect)
        self._apply_about_page_theme()

    def _apply_about_page_theme(self):
        is_dark = self.is_dark_theme()
        bg_color = Theme.get_color("card_bg", is_dark)
        border_color = Theme.get_color("border", is_dark)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: {Theme.BORDER_RADIUS}px;
                padding: {Theme.CARD_PADDING}px;
            }}
        """)
        self.shadow_effect.setColor(QColor(0, 0, 0, 50 if is_dark else 20))

    def _create_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setOffset(0, 5)
        return shadow


class ClickableLabel(QLabel, ThemedWidgetMixin):
    """可点击的标签，用于实现超链接效果"""
    clicked = Signal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._is_hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_S))
        self._apply_about_page_theme()

    def _apply_about_page_theme(self):
        is_dark = self.is_dark_theme()
        normal_color = Theme.get_color("accent", is_dark)
        hover_color = Theme.get_color("accent_hover", is_dark)
        color = hover_color if self._is_hovered else normal_color
        self.setStyleSheet(f"color: {color.name()}; text-decoration: {'underline' if self._is_hovered else 'none'};")

    def enterEvent(self, event):
        self._is_hovered = True;
        self._apply_about_page_theme();
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False;
        self._apply_about_page_theme();
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.clicked.emit()
        super().mousePressEvent(event)


class LogoWidget(QLabel, ThemedWidgetMixin):
    """
    显示应用Logo或名称的组件。
    这个类直接继承自 QLabel，使得它本身就是图像显示区域，
    从而避免了任何外部容器（“组件框”），让Logo直接显示在父窗口上。
    """
    def __init__(self, parent=None):
        # 初始化时，它就是一个 QLabel
        super().__init__(parent)
        self._setup_ui()
        self._apply_about_page_theme()

    def _setup_ui(self):
        # 所有的设置都直接应用于 self (即这个QLabel本身)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        try:
            # 加载和处理图像的逻辑不变
            logo_path = resource_path(os.path.join("res", "ico.ico"))
            pil_image = Image.open(logo_path)
            pil_image.thumbnail((120, 120), Image.Resampling.LANCZOS)

            byte_array = io.BytesIO()
            pil_image.save(byte_array, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(byte_array.getvalue())

            # 将图像、尺寸、阴影效果都直接设置给 self
            self.setMinimumSize(pixmap.size())
            self.setPixmap(pixmap)

            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 0))
            shadow.setOffset(0, 6)
            self.setGraphicsEffect(shadow)

        except Exception as e:
            # 如果图片加载失败，备用的文本也会直接显示在 self 上
            print(f"加载Logo失败: {e}. 使用备用文本。")
            self.setText(APP_TITLE)
            self.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_XL, QFont.Weight.Bold))
            self.setFixedSize(120, 120)

    def _apply_about_page_theme(self):
        # 仅在没有图片（显示备用文本）时才设置字体颜色
        if not self.pixmap():
            is_dark = self.is_dark_theme()
            text_color = Theme.get_color("text_primary", is_dark)
            # 确保背景透明
            self.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")


class InfoSection(QWidget, ThemedWidgetMixin):
    """信息区块组件，包含标题和内容"""

    def __init__(self, title, content, is_link=False, link_url="", parent=None):
        super().__init__(parent)
        self.link_url = link_url
        layout = QVBoxLayout(self);
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setSpacing(Theme.SPACING_S)
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_M, QFont.Weight.DemiBold))
        layout.addWidget(self.title_label)
        if is_link:
            self.content_label = ClickableLabel(content);
            self.content_label.clicked.connect(self._open_link)
        else:
            self.content_label = QLabel(content);
            self.content_label.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_S))
            self.content_label.setWordWrap(True);
            self.content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.content_label);
        self._apply_about_page_theme()

    def _apply_about_page_theme(self):
        is_dark = self.is_dark_theme()
        self.title_label.setStyleSheet(f"color: {Theme.get_color('text_primary', is_dark).name()};")
        if not isinstance(self.content_label, ClickableLabel):
            self.content_label.setStyleSheet(f"color: {Theme.get_color('text_secondary', is_dark).name()};")

    def _open_link(self):
        if self.link_url: QDesktopServices.openUrl(QUrl(self.link_url))


class AboutPage(QWidget, ThemedWidgetMixin):
    """关于页面，展示应用、开发者和技术信息"""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.cards = []
        self._setup_ui()
        self.update_theme()
        self.start_animations()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self);
        main_layout.setContentsMargins(0, 0, 0, 0);
        self.setObjectName("AboutPage")
        scroll_area = QScrollArea();
        scroll_area.setWidgetResizable(True);
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content_widget = QWidget();
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget);
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(Theme.SPACING_L);
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._create_all_cards(content_layout)
        main_layout.addWidget(scroll_area)

    def _create_all_cards(self, layout):
        main_card = self._create_main_card();
        layout.addWidget(main_card);
        self.cards.append(main_card)
        info_layout = QHBoxLayout();
        info_layout.setSpacing(20)
        cards_data = [
            {"title": "👨‍💻 开发者信息", "sections": [{"title": "作者", "content": "和錦わきん"},
                                                     {"title": "GitHub", "content": "🔗 查看源代码", "is_link": True,
                                                      "url": "https://github.com/wakin721/Neri"}]},
            {"title": "🔧 技术信息", "sections": [{"title": "版本", "content": f"v{APP_VERSION}"}, {"title": "技术栈",
                                                                                                   "content": "• Python + PySide6\n• PyTorch + YOLO\n• OpenCV + PIL"}]},
            {"title": "📖 使用指南",
             "sections": [{"title": "快速开始", "content": "1. 设置图像文件路径\n2. 选择结果保存位置\n3. 开始批量处理"},
                          {"title": "支持格式", "content": "JPG, PNG, BMP, TIFF"}]}
        ]
        for data in cards_data:
            card = self._create_info_card(data["title"], data["sections"]);
            info_layout.addWidget(card);
            self.cards.append(card)
        layout.addLayout(info_layout)
        copyright_card = self._create_copyright_card();
        layout.addWidget(copyright_card);
        self.cards.append(copyright_card)

    def _create_main_card(self):
        card = ModernCard()
        main_layout = QVBoxLayout(card)
        main_layout.setSpacing(Theme.SPACING_M)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(Theme.SPACING_M)

        # Logo
        logo = LogoWidget()
        top_layout.addWidget(logo)

        # 应用名称和副标题
        title_layout = QVBoxLayout()
        app_name = QLabel("Neri")
        app_name.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_XL, QFont.Weight.Bold))
        app_subtitle = QLabel("红外相机图像智能处理工具")
        app_subtitle.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_L))
        title_layout.addWidget(app_name)
        title_layout.addWidget(app_subtitle)
        title_layout.addStretch()

        top_layout.addLayout(title_layout)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # 软件介绍
        description = QLabel("Neri是一款专为处理红外相机影像数据设计的智能桌面应用")
        description.setFont(QFont(Theme.FONT_FAMILY, 11))
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        description.setMaximumWidth(650)
        main_layout.addWidget(description)

        main_layout.addStretch()

        return card

    def _create_info_card(self, title, sections_data):
        card = ModernCard();
        layout = QVBoxLayout(card);
        card_title = QLabel(title)
        card_title.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_L, QFont.Weight.DemiBold));
        layout.addWidget(card_title)
        for section in sections_data:
            layout.addWidget(InfoSection(section["title"], section["content"], section.get("is_link", False),
                                         section.get("url", "")))
        layout.addStretch();
        return card

    def _create_copyright_card(self):
        card = ModernCard();
        layout = QVBoxLayout(card)
        copyright_text = QLabel("© 2024 和錦わきん. 遵循开源协议，仅供学习研究使用。")
        copyright_text.setFont(QFont(Theme.FONT_FAMILY, Theme.FONT_SIZE_XS));
        copyright_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_text);
        return card

    def start_animations(self):
        for i, card in enumerate(self.cards):
            card.setWindowOpacity(0)
            anim_group = self._create_fade_slide_animation(card)
            QTimer.singleShot(i * Theme.ANIM_STAGGER_DELAY, anim_group.start)

    def _create_fade_slide_animation(self, widget):
        group = QParallelAnimationGroup(self);
        opacity_anim = QPropertyAnimation(widget, b"windowOpacity", self)
        opacity_anim.setDuration(Theme.ANIM_DURATION);
        opacity_anim.setStartValue(0);
        opacity_anim.setEndValue(1)
        pos_anim = QPropertyAnimation(widget, b"pos", self);
        pos_anim.setDuration(Theme.ANIM_DURATION)
        pos_anim.setStartValue(widget.pos() + QPoint(0, 40));
        pos_anim.setEndValue(widget.pos())
        pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic);
        group.addAnimation(opacity_anim);
        group.addAnimation(pos_anim)
        return group

    def _apply_theme_colors(self):
        is_dark = self.is_dark_theme()
        self.setStyleSheet(f"""
            QWidget#AboutPage {{
                background-color: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1,
                    stop:0 {Theme.get_color('bg_start', is_dark).name()}, stop:1 {Theme.get_color('bg_end', is_dark).name()});
            }}
            QScrollArea {{ background-color: transparent; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; }}
            QScrollBar::handle:vertical {{ background: {Theme.get_color('scrollbar', is_dark).name()}; border-radius: 4px; }}
            QScrollBar::handle:vertical:hover {{ background: {Theme.get_color('scrollbar_hover', is_dark).name()}; }}
        """)
        self._update_card_text_colors()

    def _update_card_text_colors(self):
        is_dark = self.is_dark_theme()
        primary_color = Theme.get_color("text_primary", is_dark)
        secondary_color = Theme.get_color("text_secondary", is_dark)
        tertiary_color = Theme.get_color("text_tertiary", is_dark)
        for label in self.findChildren(QLabel):
            if isinstance(label, ClickableLabel): continue
            font_size = label.font().pointSize()
            color = secondary_color
            if font_size >= Theme.FONT_SIZE_L:
                color = primary_color
            elif font_size <= Theme.FONT_SIZE_XS:
                color = tertiary_color
            label.setStyleSheet(f"color: {color.name()};")