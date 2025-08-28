import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

from system.config import APP_VERSION
from system.utils import resource_path
from system.gui.ui_components import RoundedButton


class Sidebar(ttk.Frame):
    """侧边栏导航"""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, style="Sidebar.TFrame", width=180, **kwargs)
        self.controller = controller
        self.grid_propagate(False)
        self.nav_buttons = {}
        self._create_widgets()

    def _create_widgets(self):
        logo_frame = ttk.Frame(self, style="Sidebar.TFrame")
        logo_frame.pack(fill="x", pady=(20, 10))
        try:
            logo_path = resource_path("res/logo.png")
            logo_img = Image.open(logo_path).resize((50, 50), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            ttk.Label(logo_frame, image=self.logo_photo, background=self.controller.sidebar_bg).pack(pady=(0, 5))
        except Exception:
            pass

        ttk.Label(
            logo_frame, text="动物检测系统", font=("Segoe UI", 12, "bold"),
            foreground=self.controller.sidebar_fg, background=self.controller.sidebar_bg
        ).pack()

        # 使用StringVar来确保UI更新
        self.update_notification_text = tk.StringVar()
        self.update_notification_label = ttk.Label(
            logo_frame, textvariable=self.update_notification_text, font=("Segoe UI", 9, "bold"),
            foreground="#FFFF00",  # 使用明确的亮黄色
            background=self.controller.sidebar_bg
        )
        self.update_notification_label.pack(pady=(5, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=15, pady=10)

        buttons_frame = tk.Frame(self, bg=self.controller.sidebar_bg)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        menu_items = [
            ("settings", "开始"),
            ("preview", "图像预览"),
            ("advanced", "高级设置"),
            ("about", "关于")
        ]
        for page_id, page_name in menu_items:
            button = RoundedButton(
                buttons_frame,
                text=page_name,
                command=lambda p=page_id: self.controller._show_page(p),
                bg=self.controller.sidebar_bg,
                fg=self.controller.sidebar_fg,
                width=160,
                height=40,
                radius=10,
                highlight_color=self.controller.highlight_color
            )
            button.pack(fill="x", pady=3)
            self.nav_buttons[page_id] = button

        ttk.Frame(self, style="Sidebar.TFrame").pack(fill="both", expand=True)
        ttk.Label(
            self, text=f"V{APP_VERSION}", foreground=self.controller.sidebar_fg,
            background=self.controller.sidebar_bg, font=("Segoe UI", 8)
        ).pack(pady=(0, 10))

    def set_active_button(self, page_id):
        for pid, button in self.nav_buttons.items():
            button.set_active(pid == page_id)

    def set_processing_state(self, is_processing):
        for page_id, button in self.nav_buttons.items():
            if page_id != "preview":
                button.configure(state="disabled" if is_processing else "normal")

    def show_update_notification(self, message="发现新版本"):
        # 通过StringVar更新文本，这是Tkinter中最可靠的文本更新方式
        self.update_notification_text.set(message)

    def update_theme(self):
        # 1. Update the style definition. This is the proper way to style ttk widgets.
        style = ttk.Style()
        style.configure("Sidebar.TFrame", background=self.controller.sidebar_bg)

        # 2. Apply the updated style to the main sidebar frame and its ttk children.
        self.configure(style="Sidebar.TFrame")

        # 3. Manually update the background of specific widgets.
        # This is necessary for standard tk widgets or complex custom widgets.
        for widget in self.winfo_children():
            # Update standard tk.Frame (used as a container for buttons)
            if isinstance(widget, tk.Frame):
                widget.configure(bg=self.controller.sidebar_bg)

            # Update ttk.Labels directly.
            elif isinstance(widget, ttk.Label):
                widget.configure(background=self.controller.sidebar_bg)

            # Update children of the ttk.Frame that holds the logo
            elif isinstance(widget, ttk.Frame):
                for sub_widget in widget.winfo_children():
                    if isinstance(sub_widget, ttk.Label):
                        sub_widget.configure(background=self.controller.sidebar_bg)

        # 4. Update the custom RoundedButton widgets with their new colors.
        for button in self.nav_buttons.values():
            button.bg = self.controller.sidebar_bg
            button.fg = self.controller.sidebar_fg
            button.highlight_color = self.controller.highlight_color
            button.parent_bg = self.controller.sidebar_bg
            button.configure(bg=self.controller.sidebar_bg)  # Update the canvas background
            button.set_active(button.active)  # Redraw the button with new colors

        # 5. Re-set the active button to ensure highlighting is correct.
        self.set_active_button(self.controller.current_page)