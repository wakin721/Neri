import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import json
import logging
import cv2
import threading
import re
import numpy as np

from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

from system.data_processor import DataProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.config import NORMAL_FONT, SUPPORTED_IMAGE_EXTENSIONS
from system.utils import resource_path

logger = logging.getLogger(__name__)


# In system/gui/preview_page.py
class CorrectionDialog(tk.Toplevel):
    """用于修正物种信息的弹窗"""

    def __init__(self, parent, title="修正信息", original_info=None):
        super().__init__(parent)
        self.transient(parent)
        self.title(title)
        self.parent = parent
        self.result = None
        self.original_info = original_info

        # 初始化输入框变量
        self.species_name_var = tk.StringVar()
        self.species_count_var = tk.StringVar()
        self.remark_var = tk.StringVar()

        # 创建窗口内容
        body = ttk.Frame(self)
        self.initial_focus = self.create_body(body)
        body.pack(padx=15, pady=15)

        self.create_buttons()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        # 协议和窗口位置设置
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry(f"+{parent.winfo_rootx() + 60}+{parent.winfo_rooty() + 60}")

        self.initial_focus.focus_set()
        self.wait_window(self)

    def create_body(self, master):
        """创建弹窗主体，包含输入框"""
        ttk.Label(master, text="正确物种名称:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        species_name_entry = ttk.Entry(master, textvariable=self.species_name_var, width=25)
        species_name_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(master, text="物种数量:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        species_count_entry = ttk.Entry(master, textvariable=self.species_count_var, width=25)
        species_count_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(master, text="备注:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        remark_entry = ttk.Entry(master, textvariable=self.remark_var, width=25)
        remark_entry.grid(row=2, column=1, padx=5, pady=5)

        return species_name_entry

    def create_buttons(self):
        """创建“确定”和“取消”按钮"""
        box = ttk.Frame(self)
        ok_button = ttk.Button(box, text="确定", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(side=tk.LEFT, padx=10, pady=10)
        cancel_button = ttk.Button(box, text="取消", width=10, command=self.cancel)
        cancel_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 绑定快捷键
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def ok(self, event=None):
        """“确定”按钮的回调函数"""
        species_name = self.species_name_var.get().strip()
        species_count_str = self.species_count_var.get().strip()
        remark = self.remark_var.get().strip()

        # 校验物种名称
        if not species_name:
            messagebox.showwarning("输入错误", "物种名称不能为空。", parent=self)
            return

        if not species_count_str:
            if self.original_info and self.original_info.get('物种数量'):
                try:
                    original_counts = [int(c.strip()) for c in self.original_info['物种数量'].split(',')]
                    species_count_str = str(sum(original_counts))
                except (ValueError, TypeError):
                    species_count_str = '1'
            else:
                species_count_str = '1'


        # 检查物种数量格式
        if species_count_str.lower() != '空':
            try:
                # 尝试按逗号分割并转换为整数
                counts = [int(c.strip()) for c in species_count_str.split(',')]
                # 检查是否所有数字都为正数
                if not all(c > 0 for c in counts):
                    raise ValueError("数量必须是正整数。")
            except ValueError:
                messagebox.showwarning(
                    "输入格式错误",
                    "物种数量必须为以下格式之一：\n\n"
                    "1. 单个正整数 (例如: 3)\n"
                    "2. 以英文逗号隔开的多个正整数 (例如: 5,2)\n"
                    "3. 文字“空”",
                    parent=self
                )
                return

        self.result = (species_name, species_count_str, remark)
        self.destroy()

    def cancel(self, event=None):
        """“取消”按钮的回调函数"""
        self.result = None
        self.destroy()

class PreviewPage(ttk.Frame):
    """图像预览和校验页面"""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self.validation_data = {}
        self.original_image = None
        self.validation_original_image = None
        self.species_validation_original_image = None
        self.current_image_path = None
        self.current_detection_results = None

        self.last_selected_preview_image = None
        self.last_selected_validation_image = None
        self.last_selected_species_image = None

        self.current_preview_info = {}
        self.active_keybinds = []
        self._is_navigating = False

        self.species_image_map = defaultdict(list)

        self._species_marked = None
        self._count_marked = None
        self._selected_species_button = None
        self._selected_quantity_button = None

        self.color_palette = [
            '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3CD4F5',
            '#0052FF', '#6541D1', '#A777F5', '#701F57', '#FDE8DC', '#FFEADD', '#FFDBAD', '#FCDB6D',
            '#E4D884', '#B6EE93', '#ECF2C2', '#C4E4E8', '#ABCDFF', '#C8B0F4'
        ]
        self.species_color_map = {}

        global_conf = self.controller.confidence_settings.get("global", 0.25)
        self.validation_conf_var = tk.DoubleVar(value=global_conf)

        self.preview_conf_var = tk.DoubleVar(value=global_conf)
        self.current_validation_info = {}
        self.validation_conf_label = None
        self.current_selected_validation_species = None
        self._selected_validation_species_button = None
        self._selected_validation_quantity_button = None
        self.validation_progress_var = tk.StringVar(value="0/0")

        self.export_format_var = tk.StringVar(value="CSV")

        self._species_validation_marked = None
        self._count_validation_marked = None
        self._selected_validation_species_button = None
        self._selected_validation_quantity_button = None


        self.species_conf_var = tk.DoubleVar(value=0.25)
        self.current_species_info = {}
        self.species_conf_label = None
        self.current_selected_species = None

        self._create_widgets()

    def _create_widgets(self):
        self.preview_notebook = ttk.Notebook(self)
        self.preview_notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_preview_tab = ttk.Frame(self.preview_notebook)
        self.validation_tab = ttk.Frame(self.preview_notebook)
        self.species_validation_tab = ttk.Frame(self.preview_notebook)

        self.preview_notebook.add(self.image_preview_tab, text="图像预览")
        self.preview_notebook.add(self.validation_tab, text="检验校验(时间)")
        self.preview_notebook.add(self.species_validation_tab, text="检验校验(物种)")
        self.preview_notebook.bind("<<NotebookTabChanged>>", self._on_preview_tab_changed)

        self._create_image_preview_content(self.image_preview_tab)
        self._create_validation_content(self.validation_tab)
        self._create_species_validation_content(self.species_validation_tab)

    def clear_previews(self):
        """Clears content from all preview tabs to reset the state."""
        # Clear image preview tab
        self.file_listbox.delete(0, tk.END)
        self.image_label.config(image='', text="请从左侧列表选择图像")
        if hasattr(self.image_label, 'image'):
            self.image_label.image = None
        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state="disabled")
        self.current_image_path = None
        self.current_detection_results = None
        self.show_detection_var.set(False)

        # Clear validation check tab
        self.validation_listbox.delete(0, tk.END)
        self.validation_image_label.config(image='', text="请从左侧列表选择处理后的图像")
        if hasattr(self.validation_image_label, 'image'):
            self.validation_image_label.image = None
        self.validation_info_text.config(state="normal")
        self.validation_info_text.delete(1.0, tk.END)
        self.validation_info_text.config(state="disabled")
        self.validation_status_label.config(text="未校验")
        self.validation_progress_var.set("0/0")
        self.validation_data.clear()

    def _create_image_preview_content(self, parent):
        preview_content = ttk.Frame(parent)
        preview_content.pack(fill="both", expand=True)
        preview_content.columnconfigure(1, weight=1) # 让右侧列扩展
        preview_content.rowconfigure(0, weight=1) # 让第一行扩展

        list_frame = ttk.LabelFrame(preview_content, text="图像文件")
        list_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        self.file_listbox = tk.Listbox(list_frame, width=25, font=NORMAL_FONT,
                                       selectbackground=self.controller.sidebar_bg,
                                       selectforeground=self.controller.sidebar_fg,
                                       exportselection=False)

        self.file_listbox.pack(side="left", fill="both", expand=True)
        file_list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        file_list_scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=file_list_scrollbar.set)

        self.file_listbox.bind("<Up>", self._navigate_listbox_up)
        self.file_listbox.bind("<Down>", self._navigate_listbox_down)

        preview_right = ttk.Frame(preview_content)
        preview_right.grid(row=0, column=1, sticky="nsew")
        preview_right.columnconfigure(0, weight=1)
        preview_right.rowconfigure(0, weight=1) # 图片行将扩展

        image_frame = ttk.LabelFrame(preview_right, text="图像预览")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, text="请从左侧列表选择图像", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_label.bind('<Configure>', self._on_resize)


        info_frame = ttk.LabelFrame(preview_right, text="图像信息")
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.info_text = tk.Text(info_frame, height=4, font=NORMAL_FONT, wrap="word")
        self.info_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.info_text.config(state="disabled")

        control_frame = ttk.Frame(preview_right)
        control_frame.grid(row=2, column=0, sticky="ew")
        control_frame.columnconfigure(1, weight=1) # 让滑块区域扩展

        self.show_detection_var = tk.BooleanVar(value=False)
        show_detection_switch = ttk.Checkbutton(
            control_frame,
            text="显示检测结果",
            variable=self.show_detection_var,
            command=self.toggle_detection_preview
        )
        show_detection_switch.pack(side="left")

        # 新增滑块和标签
        self.preview_conf_slider = ttk.Scale(
            control_frame, from_=0.05, to=0.95, orient="horizontal",
            variable=self.preview_conf_var,
            command=self._on_preview_confidence_slider_changed
        )
        self.preview_conf_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.preview_conf_label = ttk.Label(control_frame, text=f"{self.preview_conf_var.get():.2f}")
        self.preview_conf_label.pack(side="left")

        self.detect_button = ttk.Button(
            control_frame,
            text="检测当前图像",
            command=self.detect_current_image,
            width=12
        )
        self.detect_button.pack(side="right")

    def _create_validation_content(self, parent):
        """创建“检验校验(时间)”标签页内容，使其与物种校验页一致"""
        style = ttk.Style()

        text_color = "white" if self.controller.is_dark_mode else "black"
        style.map("Selected.TButton",
                  background=[('!active', self.controller.accent_color), ('active', self.controller.accent_color)],
                  foreground=[('!active', text_color), ('active', text_color)])

        validation_content = ttk.Frame(parent)
        validation_content.pack(fill="both", expand=True)
        validation_content.columnconfigure(1, weight=1)
        validation_content.rowconfigure(0, weight=1)

        # Left side: File List
        list_frame = ttk.LabelFrame(validation_content, text="处理后图像")
        list_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.validation_listbox = tk.Listbox(list_frame, width=25, font=NORMAL_FONT,
                                             selectbackground=self.controller.sidebar_bg,
                                             selectforeground=self.controller.sidebar_fg,
                                             exportselection=False)

        self.validation_listbox.grid(row=0, column=0, sticky="nsew")
        validation_list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.validation_listbox.yview)
        validation_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.validation_listbox.config(yscrollcommand=validation_list_scrollbar.set)
        self.validation_listbox.bind("<<ListboxSelect>>", self._on_validation_file_selected)

        self.validation_listbox.bind("<Up>", self._navigate_listbox_up)
        self.validation_listbox.bind("<Down>", self._navigate_listbox_down)

        # Right side: Content Area
        right_pane = ttk.Frame(validation_content)
        right_pane.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(0, weight=1)

        top_area_frame = ttk.Frame(right_pane)
        top_area_frame.grid(row=0, column=0, sticky="nsew")
        top_area_frame.columnconfigure(0, weight=1)
        top_area_frame.rowconfigure(0, weight=1)

        validation_image_display_frame = ttk.LabelFrame(top_area_frame, text="图片显示")
        validation_image_display_frame.grid(row=0, column=0, sticky="nsew")
        validation_image_display_frame.columnconfigure(0, weight=1)
        validation_image_display_frame.rowconfigure(0, weight=1)

        self.validation_image_label = ttk.Label(validation_image_display_frame, anchor="center")
        self.validation_image_label.grid(row=0, column=0, sticky="nsew")
        self.validation_image_label.bind('<Configure>', self._on_resize)
        self.validation_image_label.bind("<Double-1>", self.on_image_double_click)

        action_buttons_frame = ttk.LabelFrame(top_area_frame, text="快速标记")
        action_buttons_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        correct_button = ttk.Button(action_buttons_frame, text="正确", command=lambda: self._mark_validation(True))
        correct_button.pack(fill="x", pady=5, padx=5)

        empty_button = ttk.Button(action_buttons_frame, text="空",
                                  command=lambda: self._mark_validation_and_move_to_next(species_name="空", count="空"))
        empty_button.pack(fill="x", pady=5, padx=5)

        self.validation_species_buttons_frame = ttk.Frame(action_buttons_frame)
        self.validation_species_buttons_frame.pack(fill="x", pady=5, padx=5)

        ttk.Separator(action_buttons_frame, orient="horizontal").pack(fill="x", pady=5)

        other_button = ttk.Button(action_buttons_frame, text="其他", command=self._mark_validation_other_species)
        other_button.pack(fill="x", pady=5, padx=5)

        self.validation_quantity_buttons_frame = ttk.LabelFrame(top_area_frame, text="数量")
        self.validation_quantity_buttons_frame.grid(row=0, column=2, sticky="ns", padx=(10, 0))

        for i in range(1, 11):
            def create_command(num, btn):
                return lambda: self._on_validation_quantity_button_press(num, btn)

            btn = ttk.Button(self.validation_quantity_buttons_frame, text=str(i))
            btn['command'] = create_command(i, btn)
            btn.pack(fill="x", pady=2, padx=5)

        more_button = ttk.Button(self.validation_quantity_buttons_frame, text="更多")
        more_button['command'] = lambda b=more_button: self._on_validation_quantity_button_press("更多", b)
        more_button.pack(fill="x", pady=2, padx=5)

        bottom_area_frame = ttk.Frame(right_pane)
        bottom_area_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        bottom_area_frame.columnconfigure(0, weight=1)

        info_slider_frame = ttk.LabelFrame(bottom_area_frame, text="检测信息与设置", padding=(30, 5))
        info_slider_frame.grid(row=0, column=0, sticky="ew")
        info_slider_frame.columnconfigure(0, weight=1)

        self.validation_status_label = ttk.Label(info_slider_frame, text="未校验", font=NORMAL_FONT)
        self.validation_status_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        conf_slider = ttk.Scale(info_slider_frame, from_=0.05, to=0.95, orient="horizontal",
                                variable=self.validation_conf_var,
                                command=self._on_validation_confidence_slider_changed)
        conf_slider.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=(0, 5))

        self.validation_conf_label = ttk.Label(info_slider_frame, text="0.25", font=NORMAL_FONT)
        self.validation_conf_label.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=(0, 5))

        export_options_frame = ttk.LabelFrame(bottom_area_frame, text="导出选项", padding=(10, 5))
        export_options_frame.grid(row=0, column=1, sticky="e", padx=(10, 0))

        format_combo = ttk.Combobox(
            export_options_frame,
            textvariable=self.export_format_var,
            values=["CSV", "Excel", "错误照片"],
            width=8,
            state="readonly",
            takefocus=False
        )
        format_combo.pack(side="left", padx=(0, 5), pady=5)

        export_button = ttk.Button(export_options_frame, text="导出",
                                   command=self._dispatch_export,
                                   takefocus=False)
        export_button.pack(side="left", padx=(0, 5), pady=5)

    def _on_preview_tab_changed(self, event):
        selected_tab = self.preview_notebook.select()
        tab_text = self.preview_notebook.tab(selected_tab, "text")

        if tab_text == "图像预览":
            # 切换到预览页时，尝试恢复之前的选择
            if self.last_selected_preview_image:
                try:
                    all_files = self.file_listbox.get(0, tk.END)
                    if self.last_selected_preview_image in all_files:
                        idx = all_files.index(self.last_selected_preview_image)
                        self.file_listbox.selection_set(idx)
                        self.file_listbox.see(idx)
                        self.file_listbox.event_generate("<<ListboxSelect>>")
                except ValueError:
                    self.last_selected_preview_image = None  # 如果找不到，则清除状态

        elif tab_text == "检验校验(时间)":
            self._load_processed_images()
            self.validation_listbox.focus_set()
            self._load_validation_species_buttons()

            # 加载全局置信度设置
            global_conf = self.controller.confidence_settings.get("global", 0.25)
            self.validation_conf_var.set(global_conf)
            if self.validation_conf_label:
                self.validation_conf_label.config(text=f"{global_conf:.2f}")

            # 尝试恢复之前的选择
            restored = False
            if self.last_selected_validation_image:
                try:
                    all_files = self.validation_listbox.get(0, tk.END)
                    if self.last_selected_validation_image in all_files:
                        idx = all_files.index(self.last_selected_validation_image)
                        self.validation_listbox.selection_set(idx)
                        self.validation_listbox.see(idx)
                        self.validation_listbox.event_generate("<<ListboxSelect>>")
                        restored = True
                except ValueError:
                    self.last_selected_validation_image = None

            # 如果没有恢复之前的选择，则执行默认选择逻辑
            if not restored and self.validation_listbox.size() > 0:
                unvalidated_index = next(
                    (i for i, f in enumerate(self.validation_listbox.get(0, tk.END)) if f not in self.validation_data),
                    -1)
                if unvalidated_index != -1:
                    self.validation_listbox.selection_set(unvalidated_index)
                    self.validation_listbox.see(unvalidated_index)
                else:
                    self.validation_listbox.selection_set(0)
                # 触发事件以加载图片
                self.validation_listbox.event_generate("<<ListboxSelect>>")

        elif tab_text == "检验校验(物种)":
            self._load_species_data()
            # 恢复之前选择的物种焦点
            if self.current_selected_species:
                try:
                    all_species = self.species_listbox.get(0, tk.END)
                    if self.current_selected_species in all_species:
                        idx = all_species.index(self.current_selected_species)
                        self.species_listbox.selection_set(idx)
                        self.species_listbox.see(idx)
                        self.species_listbox.event_generate("<<ListboxSelect>>")

                        # 在恢复物种后，进一步恢复该物种下的图片选择
                        if self.last_selected_species_image:
                            self.master.after(50, self._restore_species_photo_selection)

                except ValueError:
                    self.current_selected_species = None
                    self.species_photo_listbox.delete(0, tk.END)

    def _restore_species_photo_selection(self):
        """在物种列表加载后，恢复照片列表的选择"""
        try:
            all_photos = self.species_photo_listbox.get(0, tk.END)
            if self.last_selected_species_image in all_photos:
                idx = all_photos.index(self.last_selected_species_image)
                self.species_photo_listbox.selection_set(idx)
                self.species_photo_listbox.see(idx)
                self.species_photo_listbox.event_generate("<<ListboxSelect>>")
        except ValueError:
            self.last_selected_species_image = None

    def update_file_list(self, directory: str):
        # The clearing is now done in clear_previews, called from main_window
        if not os.path.isdir(directory):
            return

        try:
            image_files = [f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            image_files.sort()
            for file in image_files:
                self.file_listbox.insert(tk.END, file)
        except Exception as e:
            logger.error(f"更新文件列表失败: {e}")

    def on_file_selected(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return

        self.controller.master.update_idletasks()

        file_name = self.file_listbox.get(selection[0])
        self.last_selected_preview_image = file_name
        file_path = os.path.join(self.controller.start_page.file_path_entry.get(), file_name)

        # Reset states for the new image
        self.current_image_path = file_path
        self.current_detection_results = None
        self.current_preview_info = {}
        self.show_detection_var.set(False)

        # Load and display the original image without any boxes first
        self.update_image_preview(file_path)
        # Update text-based info
        self.update_image_info(file_path, file_name)

        # Check if a JSON file with detection results exists
        photo_path = self.controller.get_temp_photo_dir()
        if not photo_path: return

        base_name, _ = os.path.splitext(file_name)
        json_path = os.path.join(photo_path, f"{base_name}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    # Load the detection info
                    self.current_preview_info = json.load(f)

                # Update the text part of the info display
                self._update_detection_info(self.current_preview_info)

                # Since we have results, check the "show detection" box
                # The trace on this var will trigger toggle_detection_preview, which will then call the redraw function
                self.show_detection_var.set(True)
            except Exception as e:
                logger.error(f"读取检测JSON失败: {e}")
                self.current_preview_info = {}
                self._update_detection_info({}) # Clear text info as well

    def update_image_preview(self, file_path: str, show_detection: bool = False, detection_results=None,
                             is_temp_result: bool = False):
        if hasattr(self.image_label, 'image'):
            self.image_label.image = None

        try:
            if is_temp_result:
                img = Image.open(file_path)
            elif show_detection and detection_results:
                result_img = detection_results[0].plot()
                img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(file_path)
            self.original_image = img
            resized_img = self._resize_image_to_fit(img, self.image_label.winfo_width(),
                                                    self.image_label.winfo_height())
            photo = ImageTk.PhotoImage(resized_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            logger.error(f"更新图像预览失败: {e}")
            self.image_label.config(image='', text="无法加载图像")
            self.original_image = None

    def update_image_info(self, file_path: str, file_name: str):
        from system.metadata_extractor import ImageMetadataExtractor
        image_info, _ = ImageMetadataExtractor.extract_metadata(file_path, file_name)
        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        info1 = f"文件名: {image_info.get('文件名', '')}    格式: {image_info.get('格式', '')}"
        info2 = f"拍摄日期: {image_info.get('拍摄日期', '未知')} {image_info.get('拍摄时间', '')}    "
        try:
            with Image.open(file_path) as img:
                info2 += f"尺寸: {img.width}x{img.height}px    文件大小: {os.path.getsize(file_path) / 1024:.1f} KB"
        except:
            pass
        self.info_text.insert(tk.END, info1 + "\n" + info2)
        # Keep the text box disabled for user interaction, but allow code to modify it.
        # self.info_text.config(state="disabled")

    def toggle_detection_preview(self, *args):
        if self.controller.is_processing:
            self.show_detection_var.set(True)
            return
        selection = self.file_listbox.curselection()
        if not selection:
            self.show_detection_var.set(False)
            return

        if self.show_detection_var.get():
            # 我们需要检测信息来绘制检测框
            if self.current_preview_info:
                self._redraw_preview_boxes_with_new_confidence(self.preview_conf_var.get())
            else:
                messagebox.showinfo("提示", '当前图像尚未检测，请点击"检测当前图像"按钮。')
                self.show_detection_var.set(False)
        else:
            # 只显示不带检测框的原始图片
            self.update_image_preview(self.current_image_path)

    def detect_current_image(self):
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一张图像。")
            return
        file_name = self.file_listbox.get(selection[0])
        file_path = os.path.join(self.controller.start_page.file_path_entry.get(), file_name)
        # self.controller.status_bar.status_label.config(text="正在检测图像...")
        self.detect_button.config(state="disabled")
        threading.Thread(target=self._detect_image_thread, args=(file_path, file_name), daemon=True).start()

    def _detect_image_thread(self, img_path, filename):
        try:
            from datetime import datetime
            results = self.controller.image_processor.detect_species(img_path,
                                                                     self.controller.advanced_page.controller.use_fp16_var.get(),
                                                                     self.controller.advanced_page.controller.iou_var.get(),
                                                                     self.controller.advanced_page.controller.conf_var.get(),
                                                                     self.controller.advanced_page.controller.use_augment_var.get(),
                                                                     self.controller.advanced_page.controller.use_agnostic_nms_var.get())
            self.current_detection_results = results['detect_results']
            species_info = {k: v for k, v in results.items() if k != 'detect_results'}
            species_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if self.current_detection_results:
                temp_photo_dir = self.controller.get_temp_photo_dir()
                # 保存JSON文件
                json_path = self.controller.image_processor.save_detection_info_json(self.current_detection_results, filename, species_info,
                                                                      temp_photo_dir)

                # 从刚保存的JSON中读回数据以填充 self.current_preview_info
                with open(json_path, 'r', encoding='utf-8') as f:
                    loaded_detection_info = json.load(f)

                # 在主线程中更新UI
                self.master.after(0, lambda: setattr(self, 'current_preview_info', loaded_detection_info))
                self.master.after(0, lambda: self._update_detection_info(species_info))
                self.master.after(0, lambda: self.show_detection_var.set(True)) # 这将触发重绘
        except Exception as err:
            logger.error(f"检测图像失败: {err}")
            self.master.after(0, lambda msg=str(err): messagebox.showerror("错误", f"检测图像失败: {msg}"))
        finally:
            self.master.after(0, lambda: self.detect_button.config(state="normal"))

    def _update_detection_info(self, species_info):
        self.info_text.config(state="normal")
        current_text_lines = self.info_text.get(1.0, tk.END).strip().split('\n')
        basic_info = "\n".join(current_text_lines[:2])

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, basic_info)

        detection_parts = ["检测结果:"]
        if species_info and species_info.get('物种名称') and species_info['物种名称'] != '空':
            names = species_info['物种名称'].split(',')
            counts = species_info.get('物种数量', '').split(',')
            info_parts = [f"{n}: {c}只" for n, c in zip(names, counts)]
            detection_parts.append(", ".join(info_parts))
            if species_info.get('最低置信度'):
                detection_parts.append(f"最低置信度: {species_info['最低置信度']}")
            if species_info.get('检测时间'):
                detection_parts.append(f"检测于: {species_info['检测时间']}")
        else:
            detection_parts.append("未检测到已知物种")

        self.info_text.insert(tk.END, "\n" + " | ".join(detection_parts))
        self.info_text.config(state="disabled")

    def _resize_image_to_fit(self, img, max_width, max_height):
        if not all([max_width > 0, max_height > 0]):
            max_width, max_height = 400, 300
        w, h = img.size
        if w == 0 or h == 0: return img
        scale = min(max_width / w, max_height / h)
        if scale >= 1: return img
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))
        return img.resize((new_width, new_height), Image.LANCZOS)

    def on_image_double_click(self, event):
        pass

    def _load_processed_images(self):
        photo_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.file_path_entry.get()

        if not photo_dir or not os.path.exists(photo_dir) or not source_dir:
            self.validation_listbox.delete(0, tk.END) # 如果路径无效，则清空列表
            return

        self.validation_listbox.delete(0, tk.END)

        # 获取temp目录下的所有json文件
        try:
            json_files = [f for f in os.listdir(photo_dir) if f.lower().endswith('.json')]
        except FileNotFoundError:
            logger.error(f"临时目录未找到: {photo_dir}")
            return

        # 获取源目录下的所有支持的图片文件，并创建一个从基础文件名到完整文件名的映射
        try:
            source_images = [f for f in os.listdir(source_dir) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            image_basename_map = {os.path.splitext(f)[0]: f for f in source_images}
        except FileNotFoundError:
            logger.error(f"源目录未找到: {source_dir}")
            return

        processed_images = []
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            image_filename = image_basename_map.get(base_name)
            if image_filename:
                processed_images.append(image_filename)

        processed_images.sort()

        for file in processed_images:
            self.validation_listbox.insert(tk.END, file)

        self._update_validation_progress()

    def _on_validation_file_selected(self, event):
        self._species_validation_marked = None
        self._count_validation_marked = None

        if self._selected_validation_species_button and self._selected_validation_species_button.winfo_exists():
            self._selected_validation_species_button.configure(style="TButton")
            self._selected_validation_species_button = None
        if self._selected_validation_quantity_button and self._selected_validation_quantity_button.winfo_exists():
            self._selected_validation_quantity_button.configure(style="TButton")
            self._selected_validation_quantity_button = None

        selection = self.validation_listbox.curselection()
        if not selection:
            self.validation_status_label.config(text="请从左侧列表选择处理后的图像")
            return

        file_name = self.validation_listbox.get(selection[0])
        self.last_selected_validation_image = file_name
        photo_dir = self.controller.get_temp_photo_dir()
        if not photo_dir: return

        original_image_path = os.path.join(self.controller.start_page.file_path_entry.get(), file_name)
        try:
            self.validation_original_image = Image.open(original_image_path)
        except Exception as e:
            logger.error(f"加载原始校验图像失败: {e}")
            self.validation_original_image = None
            self.validation_image_label.config(image='', text="无法加载原始图像")
            if hasattr(self.validation_image_label, 'image'):
                self.validation_image_label.image = None
            return

        json_path = os.path.join(photo_dir, f"{os.path.splitext(file_name)[0]}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.current_validation_info = json.load(f)

                self._recalculate_and_update_info_label(
                    self.validation_status_label,
                    self.current_validation_info,
                    self.validation_conf_var.get()
                )

                self._redraw_validation_boxes_with_new_confidence(self.validation_conf_var.get())

            except Exception as e:
                logger.error(f"加载JSON信息失败: {e}")
                self.validation_status_label.config(text="加载信息失败")
        else:
            self.validation_status_label.config(text="物种: - | 数量: - | 置信度: -")
            self._redraw_validation_boxes_with_new_confidence(1.1)

    def _mark_validation(self, is_correct):
        selection = self.validation_listbox.curselection()
        if not selection:
            return
        file_name = self.validation_listbox.get(selection[0])

        if not is_correct:
            # 弹出修正对话框
            dialog = CorrectionDialog(self)
            # 如果用户点击了“确定”并输入了有效值
            if dialog.result:
                correct_species_name, correct_species_count = dialog.result
                self._update_json_file(file_name, correct_species_name, correct_species_count)
                # 即使修正了，也标记为错误，以便导出
                self.validation_data[file_name] = False
                # 刷新信息显示
                self._on_validation_file_selected(None)
            else:
                # 如果用户取消或关闭了对话框，则不进行任何操作
                return
        else:
            self.validation_data[file_name] = True

        # 更新状态标签并保存
        self.validation_status_label.config(text=f"已标记: {'正确 ✅' if self.validation_data.get(file_name) else '错误 ❌'}")
        self._save_validation_data()
        self._update_validation_progress()

        # 自动选择下一张图片
        self._select_next_image()
        self.validation_listbox.focus_set()

    def _update_validation_progress(self):
        total = self.validation_listbox.size()
        validated = len(self.validation_data)
        self.validation_progress_var.set(f"{validated}/{total}")

    def _save_validation_data(self):
        temp_dir = self.controller.get_temp_photo_dir()
        if not temp_dir: return
        with open(os.path.join(temp_dir, "validation.json"), 'w', encoding='utf-8') as f:
            json.dump(self.validation_data, f, indent=2)

    def _load_validation_data(self):
        temp_dir = self.controller.get_temp_photo_dir()
        if not temp_dir: return
        path = os.path.join(temp_dir, "validation.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.validation_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load validation data: {e}")
                self.validation_data = {}
        else:
            self.validation_data = {}

    def _update_json_file(self, file_name: str, new_species: str = None, new_count: str = None, new_remark: str = None):
        """根据弹窗输入更新JSON文件"""
        photo_dir = self.controller.get_temp_photo_dir()
        if not photo_dir: return

        base_name, _ = os.path.splitext(file_name)
        json_path = os.path.join(photo_dir, f"{base_name}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)

                    if new_species is not None:
                        data['物种名称'] = new_species
                    if new_count is not None:
                        data['物种数量'] = str(new_count)
                    if new_remark is not None:
                        data['备注'] = new_remark


                    # 只要有手动修改，就更新置信度和时间
                    data['最低置信度'] = '人工校验'
                    data['检测时间'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}(人工校验)"
                    if new_species or new_count == "空":
                        data['检测框'] = []  # 清空检测框

                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.truncate()

                    self.current_species_info = data
                    self._update_species_info_label()

            except Exception as e:
                logger.error(f"更新JSON文件失败 ({file_name}): {e}")
                messagebox.showerror("错误", f"更新JSON文件失败: {e}", parent=self)

        # 立即刷新右侧信息显示
        self._on_validation_file_selected(None)

    def _update_species_info_label(self):
        """辅助函数，用于从 self.current_species_info 更新物种信息标签"""
        if self.species_info_label and self.current_species_info:
            info_text = (f"物种: {self.current_species_info.get('物种名称', 'N/A')} | "
                         f"数量: {self.current_species_info.get('物种数量', 'N/A')} | "
                         f"置信度: {self.current_species_info.get('最低置信度', 'N/A')}")
            self.species_info_label.config(text=info_text)

    def _export_error_images(self):
        error_files = [f for f, v in self.validation_data.items() if v is False]
        if not error_files:
            messagebox.showinfo("提示", "没有标记为错误的图片。", parent=self)
            return

        source_dir = self.controller.start_page.file_path_entry.get()
        save_dir = self.controller.start_page.save_path_entry.get()
        if not all([source_dir, save_dir]):
            messagebox.showerror("错误", "请先在“开始”页面设置源路径和保存路径。", parent=self)
            return

        error_folder = os.path.join(save_dir, "error")
        os.makedirs(error_folder, exist_ok=True)
        from shutil import copy

        temp_photo_dir = self.controller.get_temp_photo_dir()
        copied_count = 0
        failed_files = []

        for file in error_files:
            try:
                json_path = os.path.join(temp_photo_dir, f"{os.path.splitext(file)[0]}.json")
                corrected_species_name = "未分类错误" # 默认文件夹

                # 如果是人工校验过的，按修正后的物种名分类
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get('最低置信度') == '人工校验':
                        corrected_species_name = data.get('物种名称', corrected_species_name)

                # 创建物种分类子文件夹
                species_folder = os.path.join(error_folder, corrected_species_name)
                os.makedirs(species_folder, exist_ok=True)

                # 复制原图
                source_image_path = os.path.join(source_dir, file)
                if os.path.exists(source_image_path):
                    copy(source_image_path, species_folder)
                    copied_count += 1
                else:
                    logger.warning(f"源图片未找到，无法复制: {source_image_path}")
                    failed_files.append(file)

            except Exception as e:
                logger.error(f"导出错误图片失败 ({file}): {e}")
                failed_files.append(file)

        message = f"成功导出 {copied_count} 张错误图片到以下文件夹:\n{error_folder}"
        if failed_files:
            message += f"\n\n有 {len(failed_files)} 个文件导出失败，请检查日志获取详细信息。"
            messagebox.showwarning("导出完成", message, parent=self)
        else:
            messagebox.showinfo("成功", message, parent=self)

    def _export_validation_data(self):
        """从校验页面的数据导出为表格文件（Excel或CSV）"""
        temp_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.file_path_entry.get()

        if not temp_dir or not os.path.exists(temp_dir) or not source_dir:
            messagebox.showerror("错误", "无法找到临时文件或源文件路径，请确保已进行批处理并且路径设置正确。",
                                 parent=self)
            return

        json_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.json')]
        if not json_files:
            messagebox.showinfo("提示", "没有找到任何处理后的数据，无法导出。", parent=self)
            return

        # 根据下拉框选择确定文件类型
        file_format = self.export_format_var.get().lower()
        if file_format == 'excel':
            file_types = [("Excel 文件", "*.xlsx"), ("所有文件", "*.*")]
            default_extension = ".xlsx"
            initial_file = "校验结果.xlsx"
        elif file_format == 'csv':
            file_types = [("CSV 文件", "*.csv"), ("所有文件", "*.*")]
            default_extension = ".csv"
            initial_file = "校验结果.csv"
        else:
            return  # 如果格式未知则不执行操作

        # 弹出文件保存对话框
        output_path = filedialog.asksaveasfilename(
            title="选择表格保存位置",
            defaultextension=default_extension,
            filetypes=file_types,
            initialfile=initial_file,
            parent=self
        )

        # 如果用户取消了选择，则不执行任何操作
        if not output_path:
            return

        # 加载置信度配置文件
        confidence_settings = self.controller.settings_manager.load_confidence_settings()
        if not confidence_settings:
            confidence_settings = {}

        all_image_data = []
        earliest_date = None

        for json_file in json_files:
            json_path = os.path.join(temp_dir, json_file)
            image_filename = os.path.splitext(json_file)[0] + ".jpg"
            image_path = os.path.join(source_dir, image_filename)

            if not os.path.exists(image_path):
                found_image = False
                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                    temp_path = os.path.join(source_dir, os.path.splitext(json_file)[0] + ext)
                    if os.path.exists(temp_path):
                        image_path = temp_path
                        found_image = True
                        break
                if not found_image:
                    logger.warning(f"找不到原始图片: {image_filename}")
                    continue

            try:
                metadata, _ = ImageMetadataExtractor.extract_metadata(image_path, os.path.basename(image_path))
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                metadata.update(json_data)
                all_image_data.append(metadata)
                date_taken = metadata.get('拍摄日期对象')
                if date_taken:
                    if earliest_date is None or date_taken < earliest_date:
                        earliest_date = date_taken
            except Exception as e:
                logger.error(f"处理文件 {json_file} 时出错: {e}")

        if not all_image_data:
            messagebox.showerror("错误", "未能成功处理任何数据，无法导出。", parent=self)
            return

        processed_data = DataProcessor.process_independent_detection(all_image_data, confidence_settings)
        if earliest_date:
            processed_data = DataProcessor.calculate_working_days(processed_data, earliest_date)

        # 传递选择的文件格式
        success = DataProcessor.export_to_excel(processed_data, output_path, confidence_settings,
                                                file_format=file_format)

        if success:
            if messagebox.askyesno("成功", f"数据已成功导出到:\n{output_path}\n\n是否立即打开文件？", parent=self):
                try:
                    os.startfile(output_path)
                except Exception as e:
                    messagebox.showerror("错误", f"无法打开文件: {e}", parent=self)
        else:
            messagebox.showerror("导出失败", "导出文件时发生错误，请查看日志文件获取详情。", parent=self)

    def _on_resize(self, event):
        # 确定是哪个标签触发了事件
        image_to_resize = None
        label_widget = None

        if event.widget == self.image_label:
            image_to_resize = self.original_image
            label_widget = self.image_label
        elif event.widget == self.validation_image_label:
            image_to_resize = self.validation_original_image
            label_widget = self.validation_image_label
        elif event.widget == self.species_image_label:  # 新增的判断逻辑
            image_to_resize = self.species_validation_original_image
            label_widget = self.species_image_label
        else:
            return

        # 如果有原始图片，则根据新大小重新缩放
        if image_to_resize:
            # 获取标签的新尺寸
            width, height = event.width, event.height
            if width < 2 or height < 2: return  # 避免尺寸过小时出错

            # 重新缩放并更新图片
            resized_img = self._resize_image_to_fit(image_to_resize, width, height)
            photo = ImageTk.PhotoImage(resized_img)
            label_widget.config(image=photo)
            label_widget.image = photo

    def _on_species_photo_selected(self, event):
        self._count_marked = None

        if self._selected_species_button and self._selected_species_button.winfo_exists():
            self._selected_species_button.configure(style="TButton")
            self._selected_species_button = None
        if self._selected_quantity_button and self._selected_quantity_button.winfo_exists():
            self._selected_quantity_button.configure(style="TButton")
            self._selected_quantity_button = None

        selection = self.species_photo_listbox.curselection()
        if not selection:
            self.species_info_label.config(text="物种: - | 数量: - | 置信度: -")
            return

        file_name = self.species_photo_listbox.get(selection[0])
        self.last_selected_species_image = file_name
        photo_dir = self.controller.get_temp_photo_dir()
        if not photo_dir:
            return

        original_image_path = os.path.join(self.controller.start_page.file_path_entry.get(), file_name)
        try:
            self.species_validation_original_image = Image.open(original_image_path)
        except Exception as e:
            logger.error(f"加载原始物种校验图像失败: {e}")
            self.species_validation_original_image = None
            self.species_image_label.config(image='', text="无法加载原始图像")
            if hasattr(self.species_image_label, 'image'):
                self.species_image_label.image = None
            return

        json_path = os.path.join(photo_dir, f"{os.path.splitext(file_name)[0]}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.current_species_info = json.load(f)

                self._recalculate_and_update_info_label(
                    self.species_info_label,
                    self.current_species_info,
                    self.species_conf_var.get()
                )

                self._redraw_boxes_with_new_confidence(self.species_conf_var.get())

            except Exception as e:
                logger.error(f"加载JSON信息失败: {e}")
                self.species_info_label.config(text="加载信息失败")
        else:
            self.species_info_label.config(text="物种: - | 数量: - | 置信度: -")
            self._redraw_boxes_with_new_confidence(1.1)


    def _create_species_validation_content(self, parent):
        style = ttk.Style()

        text_color = "white" if self.controller.is_dark_mode else "black"
        style.map("Selected.TButton",
                  background=[('!active', self.controller.accent_color), ('active', self.controller.accent_color)],
                  foreground=[('!active', text_color), ('active', text_color)])

        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left_pane = ttk.Frame(content)
        left_pane.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))
        left_pane.rowconfigure(0, weight=1)
        left_pane.rowconfigure(1, weight=1)

        species_list_frame = ttk.LabelFrame(left_pane, text="物种")
        species_list_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        species_list_frame.rowconfigure(0, weight=1)
        species_list_frame.columnconfigure(0, weight=1)

        self.species_listbox = tk.Listbox(species_list_frame, width=25, font=NORMAL_FONT,
                                          selectbackground=self.controller.sidebar_bg,
                                          selectforeground=self.controller.sidebar_fg,
                                          exportselection=False)
        self.species_listbox.grid(row=0, column=0, sticky="nsew")

        species_scrollbar = ttk.Scrollbar(species_list_frame, orient="vertical", command=self.species_listbox.yview)
        species_scrollbar.grid(row=0, column=1, sticky="ns")
        self.species_listbox.config(yscrollcommand=species_scrollbar.set)
        self.species_listbox.bind("<<ListboxSelect>>", self._on_species_selected)

        photo_list_frame = ttk.LabelFrame(left_pane, text="照片文件")
        photo_list_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        photo_list_frame.rowconfigure(0, weight=1)
        photo_list_frame.columnconfigure(0, weight=1)

        self.species_photo_listbox = tk.Listbox(photo_list_frame,
                                                width=25,
                                                font=NORMAL_FONT,
                                                selectbackground=self.controller.sidebar_bg,
                                                selectforeground=self.controller.sidebar_fg,
                                                exportselection=False
                                                )
        self.species_photo_listbox.grid(row=0, column=0, sticky="nsew")

        photo_scrollbar = ttk.Scrollbar(photo_list_frame, orient="vertical", command=self.species_photo_listbox.yview)
        photo_scrollbar.grid(row=0, column=1, sticky="ns")
        self.species_photo_listbox.config(yscrollcommand=photo_scrollbar.set)
        self.species_photo_listbox.bind("<<ListboxSelect>>", self._on_species_photo_selected)

        self.species_photo_listbox.bind("<Up>", self._navigate_listbox_up)
        self.species_photo_listbox.bind("<Down>", self._navigate_listbox_down)

        right_pane = ttk.Frame(content)
        right_pane.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(0, weight=1)

        top_area_frame = ttk.Frame(right_pane)
        top_area_frame.grid(row=0, column=0, sticky="nsew")
        top_area_frame.columnconfigure(0, weight=1)
        top_area_frame.rowconfigure(0, weight=1)

        self.species_image_display_frame = ttk.LabelFrame(top_area_frame, text="图片显示")
        self.species_image_display_frame.grid(row=0, column=0, sticky="nsew")
        self.species_image_display_frame.columnconfigure(0, weight=1)
        self.species_image_display_frame.rowconfigure(0, weight=1)

        self.species_image_label = ttk.Label(self.species_image_display_frame, anchor="center")
        self.species_image_label.grid(row=0, column=0, sticky="nsew")
        self.species_image_label.bind('<Configure>', self._on_resize)

        action_buttons_frame = ttk.LabelFrame(top_area_frame, text="快速标记")
        action_buttons_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        correct_button = ttk.Button(action_buttons_frame, text="正确",
                                    command=lambda: self._mark_and_move_to_next(is_correct=True))
        correct_button.pack(fill="x", pady=5, padx=5)

        empty_button = ttk.Button(action_buttons_frame, text="空",
                                  command=lambda: self._mark_and_move_to_next(species_name="空", count="空"))
        empty_button.pack(fill="x", pady=5, padx=5)

        self.species_buttons_frame = ttk.Frame(action_buttons_frame)
        self.species_buttons_frame.pack(fill="x", pady=5, padx=5)

        ttk.Separator(action_buttons_frame, orient="horizontal").pack(fill="x", pady=5)

        other_button = ttk.Button(action_buttons_frame, text="其他", command=self._mark_other_species)
        other_button.pack(fill="x", pady=5, padx=5)

        self.quantity_buttons_frame = ttk.LabelFrame(top_area_frame, text="数量")
        self.quantity_buttons_frame.grid(row=0, column=2, sticky="ns", padx=(10, 0))

        for i in range(1, 11):
            def create_command(num, btn):
                return lambda: self._on_quantity_button_press(num, btn)

            btn = ttk.Button(self.quantity_buttons_frame, text=str(i))
            btn['command'] = create_command(i, btn)
            btn.pack(fill="x", pady=2, padx=5)

        more_button = ttk.Button(self.quantity_buttons_frame, text="更多")
        more_button['command'] = lambda b=more_button: self._on_quantity_button_press("更多", b)
        more_button.pack(fill="x", pady=2, padx=5)

        bottom_area_frame = ttk.Frame(right_pane)
        bottom_area_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        bottom_area_frame.columnconfigure(0, weight=1)

        info_slider_frame = ttk.LabelFrame(bottom_area_frame, text="检测信息与设置", padding=(30, 5))
        info_slider_frame.grid(row=0, column=0, sticky="ew")
        info_slider_frame.columnconfigure(0, weight=1)

        self.species_info_label = ttk.Label(info_slider_frame, text="物种:  | 数量:  | 置信度: ", font=NORMAL_FONT)
        self.species_info_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        self.species_conf_slider = ttk.Scale(info_slider_frame, from_=0.05, to=0.95, orient="horizontal",
                                             variable=self.species_conf_var, command=self._on_confidence_slider_changed)
        self.species_conf_slider.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=(0, 5))

        self.species_conf_label = ttk.Label(info_slider_frame, text="0.25", font=NORMAL_FONT)
        self.species_conf_label.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=(0, 5))

        export_options_frame = ttk.LabelFrame(bottom_area_frame, text="导出选项", padding=(10, 5))
        export_options_frame.grid(row=0, column=1, sticky="e", padx=(10, 0))

        format_combo = ttk.Combobox(
            export_options_frame,
            textvariable=self.export_format_var,
            values=["CSV", "Excel", "错误照片"],
            width=8,
            state="readonly",
            takefocus=False
        )
        format_combo.pack(side="left", padx=(0, 5), pady=5)

        export_button = ttk.Button(export_options_frame, text="导出",
                                   command=self._dispatch_export,
                                   takefocus=False)
        export_button.pack(side="left", padx=(0, 5), pady=5)

    def _load_species_data(self):
        photo_dir = self.controller.get_temp_photo_dir()
        source_dir = self.controller.start_page.file_path_entry.get()

        if not photo_dir or not os.path.exists(photo_dir) or not source_dir:
            return

        self.species_listbox.delete(0, tk.END)
        self.species_image_map.clear()

        confidence_settings = self.controller.confidence_settings

        try:
            json_files = [f for f in os.listdir(photo_dir) if f.lower().endswith('.json') and f != 'validation.json']
        except FileNotFoundError:
            logger.error(f"临时目录未找到: {photo_dir}")
            return

        try:
            source_images = [f for f in os.listdir(source_dir) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            image_basename_map = {os.path.splitext(f)[0]: f for f in source_images}
        except FileNotFoundError:
            logger.error(f"源目录未找到: {source_dir}")
            return

        all_species_keys = set()

        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            image_filename = image_basename_map.get(base_name)

            if not image_filename:
                continue

            json_path = os.path.join(photo_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 存储这张图片对应的有效物种名称
                final_species_for_image = set()

                # 如果是人工校验过的，直接使用其物种名称
                if data.get('最低置信度') == '人工校验':
                    species_names_list = data.get("物种名称", "").split(',')
                    # 清理并去重
                    final_species_for_image = {s.strip() for s in species_names_list if s.strip() and s.strip() != '空'}

                # 如果不是人工校验，则根据置信度阈值过滤
                else:
                    confidences = data.get('all_confidences', [])
                    classes = data.get('all_classes', [])
                    names_map = data.get('names_map', {})

                    if confidences and classes and names_map:
                        for cls, conf in zip(classes, confidences):
                            species_name = names_map.get(str(int(cls)))
                            if species_name:
                                threshold = confidence_settings.get(species_name,
                                                                    confidence_settings.get("global", 0.25))
                                if conf >= threshold:
                                    final_species_for_image.add(species_name)

                # 根据过滤或解析后的物种列表，生成唯一的key并分类
                if not final_species_for_image:
                    species_key = "标记为空"
                else:
                    # 通过排序和组合，为物种组合创建唯一的键 (e.g., "物种A,物种B,物种C")
                    # 这个逻辑对任意数量的物种都有效
                    species_key = ",".join(sorted(list(final_species_for_image)))

                all_species_keys.add(species_key)
                self.species_image_map[species_key].append(image_filename)

            except Exception as e:
                logger.error(f"加载物种数据失败 ({json_file}): {e}")

        # 将物种列表排序，并确保“标记为空”在列表末尾
        sorted_species = sorted(list(all_species_keys), key=lambda x: (x == "标记为空", x))

        for species in sorted_species:
            self.species_listbox.insert(tk.END, species)

        self._load_species_buttons()

    def _on_species_selected(self, event):
        selection = self.species_listbox.curselection()
        if not selection:
            return

        # 清空照片列表和信息显示
        self.species_photo_listbox.delete(0, tk.END)
        self.species_image_label.config(image='')
        if hasattr(self.species_image_label, 'image'):
            self.species_image_label.image = None
        self.species_info_label.config(text="物种:  | 数量:  | 置信度: ")

        species_name_key = self.species_listbox.get(selection[0])
        self.current_selected_species = species_name_key
        image_files = self.species_image_map.get(species_name_key, [])

        photo_count = len(image_files)
        self.controller.status_bar.status_label.config(text=f"当前物种共有 {photo_count} 张照片")

        # 根据选择的物种（或组合）来决定是否显示置信度滑块
        if species_name_key == "标记为空":
            self.species_conf_slider.grid_remove()
            self.species_conf_label.grid_remove()
        else:
            self.species_conf_slider.grid()
            self.species_conf_label.grid()
            # 更新滑块的值为当前物种的设置，如果不存在则使用默认值
            # 注意：对于组合物种，我们暂时使用全局设置
            default_conf = self.controller.confidence_settings.get(species_name_key, self.controller.confidence_settings.get("global", 0.25))
            self.species_conf_var.set(default_conf)
            self._update_confidence_label(default_conf)

        for image_file in image_files:
            self.species_photo_listbox.insert(tk.END, image_file)

        # 如果照片列表不为空，则自动选择第一个并触发加载
        if self.species_photo_listbox.size() > 0:
            self.species_photo_listbox.selection_set(0)
            self.species_photo_listbox.event_generate("<<ListboxSelect>>")

    def _load_species_buttons(self):
        """从/res/model.json加载物种按钮"""
        for widget in self.species_buttons_frame.winfo_children():
            widget.destroy()

        try:
            model_json_path = resource_path("res/model.json")
            if os.path.exists(model_json_path):
                with open(model_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    species_list = data.get("species", [])
                    for species_name in species_list:
                        # 使用lambda来捕获按钮实例
                        def create_command(s, b):
                            return lambda: self._on_species_button_press(s, b)

                        btn = ttk.Button(self.species_buttons_frame, text=species_name)
                        btn['command'] = create_command(species_name, btn)
                        btn.pack(fill="x", pady=2)
            else:
                ttk.Label(self.species_buttons_frame, text="未找到model.json").pack()
        except Exception as e:
            logger.error(f"加载物种按钮失败: {e}")
            ttk.Label(self.species_buttons_frame, text="加载物种按钮失败").pack()

    def _on_species_button_press(self, species_name, btn_widget):
        """处理物种按钮点击事件，并管理按钮状态"""
        selection = self.species_photo_listbox.curselection()
        if not selection:
            return

        file_name = self.species_photo_listbox.get(selection[0])

        # 取消上一个物种按钮的选中状态
        if self._selected_species_button and self._selected_species_button.winfo_exists():
            self._selected_species_button.configure(style="TButton")

        # 设置新按钮为选中状态
        btn_widget.configure(style="Selected.TButton")
        self._selected_species_button = btn_widget
        self._species_marked = species_name

        # 只要点击了快速标记按钮，就将其标记为错误
        self._mark_as_error_and_save(file_name)

        # 如果数量已经标记，则更新JSON并跳转
        if self._count_marked is not None:
            self._update_json_file(file_name, new_species=self._species_marked, new_count=str(self._count_marked))
            self._move_to_next_image()
        else:
            # 如果只点击了物种，且原始识别包含多个物种，则将数量设置为总和
            species_names_original = self.current_species_info.get('物种名称', '').split(',')
            species_counts_original_str = self.current_species_info.get('物种数量', '').split(',')

            if len(species_names_original) > 1 and species_names_original[0] != '空':
                try:
                    species_counts_original = [int(c.strip()) for c in species_counts_original_str]
                    new_count = sum(species_counts_original)
                    self._update_json_file(file_name, new_species=self._species_marked, new_count=str(new_count))
                except (ValueError, IndexError):
                    # 如果转换失败，则仅更新物种名称
                    self._update_json_file(file_name, new_species=self._species_marked)
            else:
                # 如果原始识别只有一个物种，或为空，则仅更新物种名称 (数量维持原样)
                self._update_json_file(file_name, new_species=self._species_marked)

    def _mark_and_move_to_next(self, is_correct=None, species_name=None, count=None):
        """
        处理标记逻辑并根据条件跳转到下一张图片。
        """
        selection = self.species_photo_listbox.curselection()
        if not selection:
            return

        file_name = self.species_photo_listbox.get(selection[0])

        # 处理“正确”按钮
        if is_correct is True:
            self.validation_data[file_name] = True
            self._save_validation_data()
            self._move_to_next_image()
            return

        # 处理“空”按钮
        if species_name == "空" and count == "空":
            self._update_json_file(file_name, new_species="空", new_count="空")
            self.validation_data[file_name] = False  # 标记为 False
            self._save_validation_data()
            self._move_to_next_image()
            return

        # 处理物种按钮点击
        if species_name:
            self._species_marked = species_name
            # 如果数量已经标记，则更新并跳转
            if self._count_marked is not None:
                self._update_json_file(file_name, new_species=self._species_marked, new_count=str(self._count_marked))
                self.validation_data[file_name] = False  # 标记为 False
                self._save_validation_data()
                self._move_to_next_image()
            else:
                # 否则，只更新物种名称
                self._update_json_file(file_name, new_species=self._species_marked)
        # 如果这是一个由数量按钮触发的跳转
        elif self._species_marked and self._count_marked is not None:
            self.validation_data[file_name] = False  # 标记为 False
            self._save_validation_data()
            self._move_to_next_image()

    def _on_quantity_button_press(self, count, btn_widget):
        """处理数量按钮点击事件，并管理按钮状态"""
        selection = self.species_photo_listbox.curselection()
        if not selection:
            return

        file_name = self.species_photo_listbox.get(selection[0])

        final_count = count
        if count == "更多":
            from tkinter import simpledialog
            result = simpledialog.askinteger("输入数量", "请输入物种的数量:", parent=self)
            if result is not None:
                final_count = result
            else:
                return

        # 取消上一个数量按钮的选中状态
        if self._selected_quantity_button and self._selected_quantity_button.winfo_exists():
            self._selected_quantity_button.configure(style="TButton")

        # 设置新按钮为选中状态
        btn_widget.configure(style="Selected.TButton")
        self._selected_quantity_button = btn_widget
        self._count_marked = final_count

        # 只要点击了数量按钮，就将其标记为错误
        self._mark_as_error_and_save(file_name)

        # 如果物种已经标记，则更新JSON并跳转
        if self._species_marked:
            self._update_json_file(file_name, new_species=self._species_marked, new_count=str(self._count_marked))
            self._move_to_next_image()
        else:
            # 否则，只更新JSON中的数量
            self._update_json_file(file_name, new_count=str(final_count))

    def _mark_other_species(self):
        """处理“其他”按钮的逻辑，弹出对话框"""
        selection = self.species_photo_listbox.curselection()
        if not selection:
            return

        file_name = self.species_photo_listbox.get(selection[0])

        dialog = CorrectionDialog(self, title="输入其他物种信息", original_info=self.current_species_info)
        if dialog.result:
            species_name, species_count, remark = dialog.result
            self._update_json_file(file_name, new_species=species_name, new_count=species_count, new_remark=remark)
            # 标记为错误并跳转
            self._mark_as_error_and_save(file_name)
            self._move_to_next_image()

    def _move_to_next_image(self):
        """重置状态并智能跳转到当前物种列表中的下一张未校验图片"""
        # 重置按钮和标记状态
        self._species_marked = None
        self._count_marked = None

        if self._selected_species_button and self._selected_species_button.winfo_exists():
            self._selected_species_button.configure(style="TButton")
        self._selected_species_button = None

        if self._selected_quantity_button and self._selected_quantity_button.winfo_exists():
            self._selected_quantity_button.configure(style="TButton")
        self._selected_quantity_button = None

        # --- 新的智能跳转逻辑 ---
        selection = self.species_photo_listbox.curselection()
        current_index = selection[0] if selection else -1

        total_files = self.species_photo_listbox.size()
        if total_files == 0:
            return

        all_files_list = list(self.species_photo_listbox.get(0, tk.END))

        # 1. 优先从当前位置向后查找下一个未被校验的项目
        for i in range(current_index + 1, total_files):
            if all_files_list[i] not in self.validation_data:
                self.species_photo_listbox.selection_clear(0, tk.END)
                self.species_photo_listbox.selection_set(i)
                self.species_photo_listbox.see(i)
                self.species_photo_listbox.event_generate("<<ListboxSelect>>")
                return

        # 2. 如果后面没有，再从头开始查找未被校验的项目
        for i in range(current_index + 1):
            if all_files_list[i] not in self.validation_data:
                self.species_photo_listbox.selection_clear(0, tk.END)
                self.species_photo_listbox.selection_set(i)
                self.species_photo_listbox.see(i)
                self.species_photo_listbox.event_generate("<<ListboxSelect>>")
                return

        # 3. 如果当前物种所有图片都已校验，则清空预览并提示
        self.species_image_label.config(image='', text="当前物种所有图片已校验完成")
        if hasattr(self.species_image_label, 'image'):
            self.species_image_label.image = None

    def _on_confidence_slider_changed(self, value):
        """处理置信度滑块值的变化"""
        self._update_confidence_label(value)

        # V V V V V V V V V V V V V V V V V V V V V V V V V V V V
        # MODIFICATION START
        # V V V V V V V V V V V V V V V V V V V V V V V V V V V V
        # 重新计算并更新信息标签
        self._recalculate_and_update_info_label(
            self.species_info_label,
            self.current_species_info,
            self.species_conf_var.get()
        )
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        # MODIFICATION END
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        self._redraw_boxes_with_new_confidence(value)

        if not self.current_selected_species or self.current_selected_species == "标记为空":
            return

        species_name = self.current_selected_species

        new_conf = round(float(value), 2)
        self.controller.confidence_settings[species_name] = new_conf
        self.controller.settings_manager.save_confidence_settings(self.controller.confidence_settings)

    def _update_confidence_label(self, value):
        """更新置信度滑块旁边的数值标签"""
        if self.species_conf_label:
            self.species_conf_label.config(text=f"{float(value):.2f}")

    def _load_validation_species_buttons(self):
        """为“检验校验(时间)”页面加载物种按钮"""
        for widget in self.validation_species_buttons_frame.winfo_children():
            widget.destroy()

        try:
            model_json_path = resource_path("res/model.json")
            if os.path.exists(model_json_path):
                with open(model_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    species_list = data.get("species", [])
                    for species_name in species_list:
                        def create_command(s, b):
                            return lambda: self._on_validation_species_button_press(s, b)

                        btn = ttk.Button(self.validation_species_buttons_frame, text=species_name)
                        btn['command'] = create_command(species_name, btn)
                        btn.pack(fill="x", pady=2)
            else:
                ttk.Label(self.validation_species_buttons_frame, text="未找到model.json").pack()
        except Exception as e:
            logger.error(f"加载物种按钮失败: {e}")
            ttk.Label(self.validation_species_buttons_frame, text="加载物种按钮失败").pack()

    def _on_validation_species_button_press(self, species_name, btn_widget):
        """处理时间校验页的物种按钮点击事件"""
        selection = self.validation_listbox.curselection()
        if not selection: return

        file_name = self.validation_listbox.get(selection[0])

        if self._selected_validation_species_button and self._selected_validation_species_button.winfo_exists():
            self._selected_validation_species_button.configure(style="TButton")

        btn_widget.configure(style="Selected.TButton")
        self._selected_validation_species_button = btn_widget
        self._species_validation_marked = species_name

        # 修复： 只要点击了快速标记按钮，就将其标记为错误
        self._mark_as_error_and_save(file_name)

        if self._count_validation_marked is not None:
            self._update_json_file(file_name, new_species=self._species_validation_marked,
                                   new_count=str(self._count_validation_marked))
            self._move_to_next_validation_image()
        else:
            # 如果只点击了物种，且原始识别包含多个物种，则将数量设置为总和
            species_names_original = self.current_validation_info.get('物种名称', '').split(',')
            species_counts_original_str = self.current_validation_info.get('物种数量', '').split(',')

            if len(species_names_original) > 1 and species_names_original[0] != '空':
                try:
                    species_counts_original = [int(c.strip()) for c in species_counts_original_str]
                    new_count = sum(species_counts_original)
                    self._update_json_file(file_name, new_species=self._species_validation_marked,
                                           new_count=str(new_count))
                except (ValueError, IndexError):
                    # 如果转换失败，则仅更新物种名称
                    self._update_json_file(file_name, new_species=self._species_validation_marked)
            else:
                # 如果原始识别只有一个物种，或为空，则仅更新物种名称 (数量维持原样)
                self._update_json_file(file_name, new_species=self._species_validation_marked)

    def _on_validation_quantity_button_press(self, count, btn_widget):
        """处理时间校验页的数量按钮点击事件"""
        selection = self.validation_listbox.curselection()
        if not selection: return

        file_name = self.validation_listbox.get(selection[0])

        final_count = count
        if count == "更多":
            from tkinter import simpledialog
            result = simpledialog.askinteger("输入数量", "请输入物种的数量:", parent=self)
            if result is not None:
                final_count = result
            else:
                return

        if self._selected_validation_quantity_button and self._selected_validation_quantity_button.winfo_exists():
            self._selected_validation_quantity_button.configure(style="TButton")

        btn_widget.configure(style="Selected.TButton")
        self._selected_validation_quantity_button = btn_widget
        self._count_validation_marked = final_count

        # 修复： 只要点击了数量按钮，就将其标记为错误
        self._mark_as_error_and_save(file_name)

        if self._species_validation_marked:
            self._update_json_file(file_name, new_species=self._species_validation_marked,
                                   new_count=str(self._count_validation_marked))
            self._move_to_next_validation_image()
        else:
            self._update_json_file(file_name, new_count=str(final_count))

    def _mark_validation_other_species(self):
        selection = self.validation_listbox.curselection()
        if not selection: return

        file_name = self.validation_listbox.get(selection[0])
        dialog = CorrectionDialog(self, title="输入其他物种信息", original_info=self.current_validation_info)
        if dialog.result:
            species_name, species_count, remark = dialog.result
            self._mark_validation_and_move_to_next(species_name=species_name, count=species_count, remark=remark)

    def _mark_validation_and_move_to_next(self, is_correct=None, species_name=None, count=None, remark=None):
        selection = self.validation_listbox.curselection()
        if not selection: return

        file_name = self.validation_listbox.get(selection[0])

        if is_correct is True:
            self.validation_data[file_name] = True
        else:
            self.validation_data[file_name] = False
            if species_name or count or remark:
                self._update_json_file(file_name, new_species=species_name, new_count=count, new_remark=remark)

        self._save_validation_data()
        self._move_to_next_validation_image()

    def _move_to_next_validation_image(self):
        self._species_validation_marked = None
        self._count_validation_marked = None

        if self._selected_validation_species_button and self._selected_validation_species_button.winfo_exists():
            self._selected_validation_species_button.configure(style="TButton")
        self._selected_validation_species_button = None

        if self._selected_validation_quantity_button and self._selected_validation_quantity_button.winfo_exists():
            self._selected_validation_quantity_button.configure(style="TButton")
        self._selected_validation_quantity_button = None

        self._select_next_image()

    def _on_validation_confidence_slider_changed(self, value):
        """处理时间校验页置信度滑块值的变化"""
        if self.validation_conf_label:
            self.validation_conf_label.config(text=f"{float(value):.2f}")

        # 重新计算并更新信息标签
        self._recalculate_and_update_info_label(
            self.validation_status_label,
            self.current_validation_info,
            self.validation_conf_var.get()
        )

        self._redraw_validation_boxes_with_new_confidence(value)

        new_conf = round(float(value), 2)
        self.controller.confidence_settings["global"] = new_conf
        self.controller.settings_manager.save_confidence_settings(self.controller.confidence_settings)

    def _on_preview_confidence_slider_changed(self, value):
        """处理预览页置信度滑块值的变化"""
        if self.preview_conf_label:
            self.preview_conf_label.config(text=f"{float(value):.2f}")
        if self.show_detection_var.get():
            self._redraw_preview_boxes_with_new_confidence(value)

    def _dispatch_export(self):
        """根据下拉框的选择来分派导出任务"""
        export_type = self.export_format_var.get()
        if export_type == "错误照片":
            self._export_error_images()
        elif export_type in ["Excel", "CSV"]:
            self._export_validation_data()
        else:
            messagebox.showwarning("提示", "未知的导出类型。", parent=self)

        self.validation_image_label.focus_set()

    def _draw_detection_boxes(self, image_label, original_image, detection_info, conf_threshold_str):
        """
        根据给定的置信度阈值，在指定的原始图像上绘制检测框，并更新对应的UI标签。

        Args:
            image_label (ttk.Label): 要更新图像的UI组件。
            original_image (PIL.Image): 用于绘制的原始PIL图像。
            detection_info (dict): 包含检测框信息的字典。
            conf_threshold_str (str): 置信度阈值的字符串表示。
        """
        # 1. 检查是否有原始图像
        if not original_image:
            placeholder_text = "请从左侧列表选择图像"
            if image_label == self.validation_image_label:
                placeholder_text = "请从左侧列表选择处理后的图像"
            image_label.config(image='', text=placeholder_text)
            if hasattr(image_label, 'image'):
                image_label.image = None
            return

        # 如果有图像但没有检测信息，则只显示原始图片
        if not detection_info or not detection_info.get("检测框"):
            resized_img = self._resize_image_to_fit(original_image, image_label.winfo_width(),
                                                    image_label.winfo_height())
            photo = ImageTk.PhotoImage(resized_img)
            image_label.config(image=photo)
            image_label.image = photo
            return

        try:
            conf_threshold = float(conf_threshold_str)
        except (ValueError, TypeError):
            return

        # --- 字体加载 ---
        try:
            font_path = resource_path("res/AlibabaPuHuiTi-3-65-Medium.ttf")
            font_size = int(0.03 * original_image.height)
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning("中文字体文件 res/AlibabaPuHuiTi-3-65-Medium.ttf 未找到。")
            font = ImageFont.load_default()

        # --- 绘制逻辑 ---
        img_to_draw = original_image.copy()
        draw = ImageDraw.Draw(img_to_draw)
        boxes_info = detection_info.get("检测框", [])

        for box in boxes_info:
            confidence = box.get("置信度", 0)
            if confidence >= conf_threshold:
                bbox = box.get("边界框")
                species_name = box.get("物种")
                label = f"{species_name} {confidence:.2f}"
                color = self._get_color_for_species(species_name)

                # 转换十六进制颜色为RGB元组
                hex_color = color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

                # 计算亮度 (YIQ a formula)
                brightness = ((r * 299) + (g * 587) + (b * 114)) / 1000

                # 根据亮度选择字体颜色
                text_color = "#FFFFFF" if brightness < 128 else "#000000"

                p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                draw.rectangle([p1, p2], outline=color, width=7)

                try:
                    text_bbox = draw.textbbox((p1[0], p1[1] - font_size - 3), label, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((p1[0], p1[1] - font_size - 3), label, fill=text_color, font=font)
                except AttributeError:
                    draw.text((p1[0], p1[1] - 15), label, fill=color, font=font)

        # --- 更新UI ---
        resized_img = self._resize_image_to_fit(img_to_draw, image_label.winfo_width(), image_label.winfo_height())
        photo = ImageTk.PhotoImage(resized_img)
        image_label.config(image=photo)
        image_label.image = photo

    def _redraw_preview_boxes_with_new_confidence(self, conf_threshold_str):
        """根据新的置信度阈值，在预览图像上重新绘制检测框"""
        self._draw_detection_boxes(
            image_label=self.image_label,
            original_image=self.original_image,
            detection_info=self.current_preview_info,
            conf_threshold_str=conf_threshold_str
        )

    def _redraw_validation_boxes_with_new_confidence(self, conf_threshold_str):
        """根据新的置信度阈值，在时间校验图像上重新绘制检测框"""
        self._draw_detection_boxes(
            image_label=self.validation_image_label,
            original_image=self.validation_original_image,
            detection_info=self.current_validation_info,
            conf_threshold_str=conf_threshold_str
        )

    def _redraw_boxes_with_new_confidence(self, conf_threshold_str):
        """根据新的置信度阈值，在物种校验图像上重新绘制检测框"""
        self._draw_detection_boxes(
            image_label=self.species_image_label,
            original_image=self.species_validation_original_image,
            detection_info=self.current_species_info,
            conf_threshold_str=conf_threshold_str
        )

    def _get_color_for_species(self, species_name):
        """为物种分配一个固定的颜色"""
        if species_name not in self.species_color_map:
            # 使用hash确保同一物种总能得到相同的颜色索引
            color_index = hash(species_name) % len(self.color_palette)
            self.species_color_map[species_name] = self.color_palette[color_index]
        return self.species_color_map[species_name]

    def on_image_double_click(self, event):
        pass

    def _select_next_image(self):
        """在“检验校验(时间)”列表中智能选择下一张图片进行校验。"""
        selection = self.validation_listbox.curselection()
        current_index = selection[0] if selection else -1

        total_files = self.validation_listbox.size()
        if total_files == 0:
            return

        # 1. 优先从当前位置向后查找下一个未被校验的项目
        all_files_list = list(self.validation_listbox.get(0, tk.END))
        for i in range(current_index + 1, total_files):
            if all_files_list[i] not in self.validation_data:
                self.validation_listbox.selection_clear(0, tk.END)
                self.validation_listbox.selection_set(i)
                self.validation_listbox.see(i)
                self.validation_listbox.event_generate("<<ListboxSelect>>")
                return

        # 2. 如果后面没有，再从头开始查找未被校验的项目
        for i in range(current_index + 1):
            if all_files_list[i] not in self.validation_data:
                self.validation_listbox.selection_clear(0, tk.END)
                self.validation_listbox.selection_set(i)
                self.validation_listbox.see(i)
                self.validation_listbox.event_generate("<<ListboxSelect>>")
                return

        # 3. 如果所有图片都已校验，则按顺序选择下一张图片
        if current_index < total_files - 1:
            next_index = current_index + 1
            self.validation_listbox.selection_clear(0, tk.END)
            self.validation_listbox.selection_set(next_index)
            self.validation_listbox.see(next_index)
            self.validation_listbox.event_generate("<<ListboxSelect>>")
        else:
            # 如果已经是最后一张，可以给出提示或清空预览
            self.validation_image_label.config(image='', text="所有图片已校验完成")
            if hasattr(self.validation_image_label, 'image'):
                self.validation_image_label.image = None

    def _update_validation_progress(self):
        total = self.validation_listbox.size()
        validated = len(self.validation_data)
        self.validation_progress_var.set(f"{validated}/{total}")

    def _mark_as_error_and_save(self, file_name):
        """将当前文件标记为不正确并保存验证数据。"""
        if file_name:
            self.validation_data[file_name] = False
            self._save_validation_data()
            self._update_validation_progress()

    def navigate_listbox(self, direction: str):
        """
        根据方向键（'up'或'down'）在当前激活的列表框中导航。
        """
        if self._is_navigating:
            return  # 如果正在导航，则忽略新的请求

        self._is_navigating = True

        notebook = self.preview_notebook
        current_tab_index = notebook.index(notebook.select())

        listbox_to_navigate = None
        if current_tab_index == 0:  # 图像预览
            listbox_to_navigate = self.file_listbox
        elif current_tab_index == 1:  # 检验校验(时间)
            listbox_to_navigate = self.validation_listbox
        elif current_tab_index == 2:  # 检验校验(物种)
            listbox_to_navigate = self.species_photo_listbox

        if not listbox_to_navigate or listbox_to_navigate.size() == 0:
            return

        current_selection = listbox_to_navigate.curselection()
        current_index = current_selection[0] if current_selection else -1

        if direction == 'down':
            next_index = 0 if current_index == -1 else (current_index + 1) % listbox_to_navigate.size()
        elif direction == 'up':
            next_index = 0 if current_index == -1 else (
                                                                   current_index - 1 + listbox_to_navigate.size()) % listbox_to_navigate.size()
        else:
            return

        # 只有在索引实际改变时才触发事件，避免不必要的重载
        if next_index != current_index:
            listbox_to_navigate.selection_clear(0, tk.END)
            listbox_to_navigate.selection_set(next_index)
            listbox_to_navigate.see(next_index)
            listbox_to_navigate.event_generate("<<ListboxSelect>>")

        self.master.after(50, lambda: setattr(self, '_is_navigating', False))

    def _recalculate_and_update_info_label(self, label_widget, detection_info, conf_threshold):
        """
        根据置信度阈值重新计算物种数量和最低置信度，并更新指定的标签。
        """
        if not detection_info or '检测框' not in detection_info:
            info_text = (f"物种: {detection_info.get('物种名称', 'N/A')} | "
                         f"数量: {detection_info.get('物种数量', 'N/A')} | "
                         f"置信度: {detection_info.get('最低置信度', 'N/A')}")
            label_widget.config(text=info_text)
            return

        if detection_info.get('最低置信度') == '人工校验':
            info_text = (f"物种: {detection_info.get('物种名称', 'N/A')} | "
                         f"数量: {detection_info.get('物种数量', 'N/A')} | "
                         f"置信度: 人工校验")
            label_widget.config(text=info_text)
            return

        boxes_info = detection_info.get("检测框", [])
        filtered_species_counts = Counter()
        valid_confidences = []

        for box in boxes_info:
            confidence = box.get("置信度", 0)
            if confidence >= conf_threshold:
                species_name = box.get("物种")
                if species_name:
                    filtered_species_counts[species_name] += 1
                    valid_confidences.append(confidence)

        min_conf_text = ''
        if not filtered_species_counts:
            species_text = "空"
            count_text = "空"
        else:
            species_text = ",".join(filtered_species_counts.keys())
            count_text = ",".join(map(str, filtered_species_counts.values()))
            if valid_confidences:
                min_conf_text = f"{min(valid_confidences):.3f}"

        info_text = (f"物种: {species_text} | "
                     f"数量: {count_text} | "
                     f"置信度: {min_conf_text}")

        if label_widget == self.validation_status_label:
            selection = self.validation_listbox.curselection()
            if selection:
                file_name = self.validation_listbox.get(selection[0])
                status = self.validation_data.get(file_name)
                status_text = f"  已标记: {'正确 ✅' if status is True else '错误 ❌' if status is False else '未校验'}"
                info_text += status_text

        label_widget.config(text=info_text)

    def _navigate_listbox_up(self, event=None):
        """处理向上箭头键事件，用于在当前激活的列表框中向上导航。"""
        self.navigate_listbox('up')
        return "break"  # 阻止事件传播

    def _navigate_listbox_down(self, event=None):
        """处理向下箭头键事件，用于在当前激活的列表框中向下导航。"""
        self.navigate_listbox('down')
        return "break"  # 阻止事件传播

