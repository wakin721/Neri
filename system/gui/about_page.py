import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import shutil

from system.config import APP_TITLE, NORMAL_FONT
from system.utils import resource_path


class AboutPage(ttk.Frame):
    """关于页面"""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self._create_widgets()

    def _create_widgets(self) -> None:
        """创建关于页面的控件"""
        about_content = ttk.Frame(self)
        about_content.pack(fill="both", expand=True, padx=20, pady=20)

        # 应用Logo
        try:
            logo_path = resource_path(os.path.join("res", "logo.png"))
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((120, 120), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(about_content, image=self.logo_photo)
            logo_label.pack(pady=(20, 10))
        except Exception:
            logo_label = ttk.Label(about_content, text=APP_TITLE, font=('Segoe UI', 18, 'bold'))
            logo_label.pack(pady=(20, 10))

        # 应用名称
        app_name = ttk.Label(about_content, text="物种信息检测系统", font=("Segoe UI", 16, "bold"))
        app_name.pack(pady=5)

        # 应用描述
        desc_label = ttk.Label(
            about_content,
            text="一款高效的物种信息检测应用程序，支持图像物种识别、探测图片保存、Excel输出和图像分类功能。",
            font=NORMAL_FONT,
            wraplength=500,
            justify="center"
        )
        desc_label.pack(pady=15)

        # 作者信息
        author_label = ttk.Label(about_content, text="作者：和錦わきん", font=NORMAL_FONT)
        author_label.pack(pady=5)

        # 移除了按钮容器和按钮