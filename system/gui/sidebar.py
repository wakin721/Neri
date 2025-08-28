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
        self.logo_frame = ttk.Frame(self, style="Sidebar.TFrame")
        self.logo_frame.pack(fill="x", pady=(20, 10))
        # Center the content within logo_frame
        self.logo_frame.columnconfigure(0, weight=1)

        try:
            # 加载并显示ico.ico图标
            ico_path = resource_path(os.path.join("res", "ico.ico"))
            icon_image = Image.open(ico_path).resize((30, 30), Image.LANCZOS)
            self.app_icon = ImageTk.PhotoImage(icon_image)

            # A container frame to hold icon and text together for centering
            container = tk.Frame(self.logo_frame, bg=self.controller.sidebar_bg) # Use tk.Frame for explicit bg
            container.grid(row=0, column=0)

            self.icon_label = ttk.Label(container, image=self.app_icon, background=self.controller.sidebar_bg)
            self.icon_label.pack(side="left", padx=(0, 5))

            self.text_label = ttk.Label(
                container, text="Neri", font=("Segoe UI", 20, "bold"), # Increased font size
                foreground=self.controller.sidebar_fg, background=self.controller.sidebar_bg
            )
            self.text_label.pack(side="left")

        except Exception as e:
            print(f"Error loading icon: {e}")
            # Fallback if icon fails to load
            self.text_label = ttk.Label(
                self.logo_frame, text="Neri", font=("Segoe UI", 14, "bold"), # Increased font size
                foreground=self.controller.sidebar_fg, background=self.controller.sidebar_bg
            )
            self.text_label.grid(row=0, column=0)


        # 使用StringVar来确保UI更新
        self.update_notification_text = tk.StringVar()
        self.update_notification_label = ttk.Label(
            self, textvariable=self.update_notification_text, font=("Segoe UI", 9, "bold"),
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
        style = ttk.Style()
        style.configure("Sidebar.TFrame", background=self.controller.sidebar_bg)
        self.configure(style="Sidebar.TFrame")

        # Manually update backgrounds of key components
        self.logo_frame.configure(style="Sidebar.TFrame")

        # Update the tk.Frame container and its children
        if hasattr(self, 'icon_label'):
            container = self.icon_label.master
            container.configure(bg=self.controller.sidebar_bg)
            self.icon_label.configure(background=self.controller.sidebar_bg)

        if hasattr(self, 'text_label'):
             self.text_label.configure(background=self.controller.sidebar_bg, foreground=self.controller.sidebar_fg)

        self.update_notification_label.configure(background=self.controller.sidebar_bg)
        # Version label is the last child
        self.nametowidget(self.winfo_children()[-1]).configure(background=self.controller.sidebar_bg, foreground=self.controller.sidebar_fg)

        # The buttons frame is a tk.Frame
        buttons_frame = self.nav_buttons["settings"].master
        buttons_frame.configure(bg=self.controller.sidebar_bg)

        # Update the custom RoundedButton widgets
        for button in self.nav_buttons.values():
            button.bg = self.controller.sidebar_bg
            button.fg = self.controller.sidebar_fg
            button.highlight_color = self.controller.highlight_color
            button.parent_bg = self.controller.sidebar_bg
            button.configure(bg=self.controller.sidebar_bg)  # Update the canvas background
            button.set_active(button.active)  # Redraw the button with new colors

        # Re-set the active button to ensure highlighting is correct.
        self.set_active_button(self.controller.current_page)