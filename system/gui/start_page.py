import tkinter as tk
from tkinter import ttk

from system.gui.ui_components import SpeedProgressBar, RoundedButton


class StartPage(ttk.Frame):
    """开始处理页面"""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self._create_widgets()

    def update_theme(self):
        """更新开始页面的主题。"""
        # 这会更新按钮的画布背景
        self.start_stop_button.update_theme()

        # 这会更新按钮本身的颜色并重绘它
        self.set_processing_state(self.controller.is_processing)

    def _create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        paths_frame = ttk.LabelFrame(self, text="路径设置")
        paths_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        file_path_frame = ttk.Frame(paths_frame)
        file_path_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(file_path_frame, text="图像文件路径:").pack(side="top", anchor="w")
        file_path_entry_frame = ttk.Frame(file_path_frame)
        file_path_entry_frame.pack(fill="x", pady=5)
        self.file_path_entry = ttk.Entry(file_path_entry_frame)
        self.file_path_entry.pack(side="left", fill="x", expand=True)
        self.file_path_button = ttk.Button(
            file_path_entry_frame, text="浏览", command=self.controller.browse_file_path, width=8)
        self.file_path_button.pack(side="right", padx=(5, 0))

        save_path_frame = ttk.Frame(paths_frame)
        save_path_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(save_path_frame, text="结果保存路径:").pack(side="top", anchor="w")
        save_path_entry_frame = ttk.Frame(save_path_frame)
        save_path_entry_frame.pack(fill="x", pady=5)
        self.save_path_entry = ttk.Entry(save_path_entry_frame)
        self.save_path_entry.pack(side="left", fill="x", expand=True)
        self.save_path_button = ttk.Button(
            save_path_entry_frame, text="浏览", command=self.controller.browse_save_path, width=8)
        self.save_path_button.pack(side="right", padx=(5, 0))

        options_frame = ttk.LabelFrame(self, text="功能选项")
        options_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.save_detect_image_var = tk.BooleanVar(value=True)
        self.copy_img_var = tk.BooleanVar(value=False)
        options_container = ttk.Frame(options_frame)
        options_container.pack(fill="x", padx=10, pady=10)

        ttk.Checkbutton(
            options_container, text="保存探测结果图片", variable=self.save_detect_image_var
        ).grid(row=0, column=0, sticky="w", pady=5, padx=10)

        ttk.Checkbutton(
            options_container, text="按物种分类图片", variable=self.copy_img_var
        ).grid(row=1, column=0, sticky="w", pady=5, padx=10)

        ttk.Frame(self).grid(row=3, column=0, sticky="nsew")

        bottom_frame = ttk.Frame(self)
        bottom_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 20))
        bottom_frame.columnconfigure(0, weight=1)
        progress_container = ttk.Frame(bottom_frame, height=50)
        progress_container.grid(row=1, column=0, sticky="ew")
        progress_container.grid_propagate(False)

        self.progress_frame = SpeedProgressBar(progress_container, accent_color=self.controller.accent_color)
        self.progress_frame.pack(fill="both", expand=True)
        self.progress_frame.hide()

        button_container = ttk.Frame(bottom_frame)
        button_container.grid(row=0, column=0, sticky="ew", pady=(10, 0))
        button_container.columnconfigure(1, weight=1)
        self.start_stop_button = RoundedButton(
            button_container,
            text="▶️开始处理",
            bg=self.controller.sidebar_bg,
            fg=self.controller.sidebar_fg,
            width=160, height=50, radius=15,
            command=self.controller.toggle_processing_state,
            show_indicator=False
        )
        self.start_stop_button.grid(row=0, column=1, sticky="e")

    def set_processing_state(self, is_processing):
        self.progress_frame.show() if is_processing else self.progress_frame.hide()
        self.start_stop_button.bg = "#e74c3c" if is_processing else self.controller.sidebar_bg
        self.start_stop_button.text = "停止处理" if is_processing else "▶️开始处理"
        self.start_stop_button._draw_button("normal")
        for widget in [self.file_path_entry, self.file_path_button, self.save_path_entry, self.save_path_button]:
            widget["state"] = "disabled" if is_processing else "normal"