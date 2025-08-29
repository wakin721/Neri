import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import webbrowser

from system.config import APP_TITLE, NORMAL_FONT
from system.utils import resource_path


class AboutPage(ttk.Frame):
    """关于页面"""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self._create_widgets()
        self.update_theme()

    def _create_widgets(self) -> None:
        """创建关于页面的控件"""
        about_content = ttk.Frame(self)
        about_content.pack(fill="both", expand=True, padx=20, pady=20)

        # 应用Logo
        try:
            logo_path = resource_path(os.path.join("res", "ico.ico"))
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((120, 120), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(about_content, image=self.logo_photo)
            logo_label.pack(pady=(20, 10))
        except Exception:
            logo_label = ttk.Label(about_content, text=APP_TITLE, font=('Segoe UI', 18, 'bold'))
            logo_label.pack(pady=(20, 10))

        # 应用名称
        app_name = ttk.Label(about_content, text="Neri - 红外相机图像智能处理工具", font=("Segoe UI", 16, "bold"))
        app_name.pack(pady=5)

        # 应用描述
        desc_label = ttk.Label(
            about_content,
            text="Neri (NERI Enables Rapid Identification) 是一款专为处理红外相机影像数据设计的智能桌面应用。它基于目标检测模型，能够高效、自动地识别和处理大批量由红外相机拍摄的野生动物照片。本工具旨在为生态保护工作者、野生动物研究人员和爱好者提供一个强大的数据整理和分析平台，将繁琐的手动筛选工作自动化，极大地提升科研和监测效率。",
            font=NORMAL_FONT,
            wraplength=500,
            justify="center"
        )
        desc_label.pack(pady=15)

        # 作者信息
        author_label = ttk.Label(about_content, text="作者：和錦わきん", font=NORMAL_FONT)
        author_label.pack(pady=5)

        # GitHub 链接
        self.github_link = ttk.Label(about_content, text="GitHub Repository", cursor="hand2", font=NORMAL_FONT)
        self.github_link.pack(pady=5)
        self.github_link.bind("<Button-1>", lambda e: self.open_link("https://github.com/wakin721/Neri"))

    def open_link(self, url: str):
        """打开网页链接"""
        webbrowser.open_new(url)

    def update_theme(self):
        """更新关于页面的主题颜色"""
        if hasattr(self, 'github_link'):
            # 根据当前是深色还是浅色模式设置链接颜色
            if self.controller.is_dark_mode:
                link_color = "#dbbcc2"  # 深色模式下的颜色
            else:
                link_color = "#5d3a4f"  # 浅色模式下的颜色
            self.github_link.configure(foreground=link_color)