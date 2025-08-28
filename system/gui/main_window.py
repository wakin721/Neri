import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import platform
import logging
import threading
import json
import time
from datetime import datetime
import gc
import sv_ttk
import hashlib
import shutil
from PIL import Image, ImageTk

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS
from system.utils import resource_path
from system.image_processor import ImageProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.data_processor import DataProcessor
from system.settings_manager import SettingsManager
from system.update_checker import check_for_updates, get_latest_version_info, compare_versions, start_download_thread, \
    _show_messagebox

# Import GUI components
from system.gui.sidebar import Sidebar
from system.gui.start_page import StartPage
from system.gui.preview_page import PreviewPage
from system.gui.advanced_page import AdvancedPage
from system.gui.about_page import AboutPage
from system.gui.ui_components import InfoBar

logger = logging.getLogger(__name__)


class ObjectDetectionGUI:
    """主应用程序窗口"""

    def __init__(self, master: tk.Tk, settings_manager: SettingsManager, settings: dict, resume_processing: bool,
                 cache_data: dict):
        self.master = master
        self.settings_manager = settings_manager
        self.settings = settings
        self.resume_processing = resume_processing
        self.cache_data = cache_data
        self.current_temp_photo_dir = None
        self.is_dark_mode = False
        self.accent_color = "#dbbcc2"  # 默认使用浅色模式的颜色
        import torch
        self.cuda_available = torch.cuda.is_available()
        self.is_processing = False
        self.processing_stop_flag = threading.Event()
        self.excel_data = []
        self.current_page = "settings"
        self.update_channel_var = tk.StringVar(value="稳定版 (Release)")
        self.model_var = tk.StringVar()  # <<< 新增：用于跟踪所选模型的变量
        self.confidence_settings = self.settings_manager.load_confidence_settings()

        self._apply_system_theme()
        self._setup_window()
        self._initialize_model(settings)
        self._setup_styles()
        self._create_ui_elements()
        self._bind_events()

        if self.settings:
            self._load_settings_to_ui(self.settings)
        else:
            # 如果没有找到配置文件，则立即用默认值创建一个
            logging.info("未找到配置文件，正在使用默认值创建 'setting.json'。")
            # 1. 从UI控件获取所有默认设置
            default_settings = self._get_current_settings()
            # 2. 保存这些默认设置到文件
            self.settings_manager.save_settings(default_settings)
            # 3. 将新创建的默认设置赋给当前实例，以确保程序后续部分能正常运行
            self.settings = default_settings
            # 4. (可选) 加载新创建的默认主题
            if hasattr(self, 'advanced_page'):
                self.change_theme()

        # 确保UI完全加载后再执行启动检查
        self._check_for_updates(silent=True)

        if not self.image_processor.model:
            messagebox.showerror("错误", "未找到有效的模型文件(.pt)。请在res目录中放入至少一个模型文件。")
            if hasattr(self, 'start_page') and hasattr(self.start_page, 'start_stop_button'):
                self.start_page.start_stop_button["state"] = "disabled"
        if self.resume_processing and self.cache_data:
            self.master.after(1000, self._resume_processing)
        self.setup_theme_monitoring()
        if hasattr(self, 'preview_page'):
            self.preview_page._load_validation_data()

    # --- 更新检查逻辑 ---

    def _check_for_updates(self, silent=False):
        """
        在程序启动时，根据用户设置的通道静默检查更新。
        这个方法现在是启动时检查的唯一入口。
        """

        def _startup_check_thread():
            """后台线程，用于处理启动时的静默更新检查。"""
            try:
                channel_selection = self.update_channel_var.get()
                channel = 'preview' if '预览版' in channel_selection else 'stable'
                latest_info = get_latest_version_info(channel)

                if not latest_info:
                    return  # Silently fail

                remote_version = latest_info['version']

                if compare_versions(APP_VERSION, remote_version):
                    if self.master.winfo_exists():
                        # 调用主窗口的方法来更新侧边栏
                        self.master.after(0, self.show_update_notification_on_sidebar)

                # 非静默模式下弹窗提示 (will not trigger on startup)
                if not silent and self.master.winfo_exists():
                    update_message = f"新版本 ({remote_version}) 可用，是否前往高级设置进行更新？"
                    _show_messagebox(self.master, "发现新版本", update_message, "info")

            except Exception as e:
                logger.error(f"启动时检查更新失败: {e}")
                if not silent and self.master.winfo_exists():
                    _show_messagebox(self.master, "更新错误", f"检查更新失败: {e}", "error")

        self.master.after(2000, lambda: threading.Thread(target=_startup_check_thread, daemon=True).start())

    def show_update_notification_on_sidebar(self):
        """这是一个专门从后台线程安全调用UI更新的方法。"""
        if hasattr(self, 'sidebar'):
            self.sidebar.show_update_notification()

    def check_for_updates_from_ui(self):
        """从高级设置UI手动触发的更新检查。"""
        channel_selection = self.update_channel_var.get()
        channel = 'preview' if '预览版' in channel_selection else 'stable'

        button = self.advanced_page.check_update_button
        status_label = self.advanced_page.update_status_label

        button.config(state="disabled")
        status_label.config(text=f"正在检查 '{channel_selection}' ...")

        threading.Thread(target=self._manual_update_check_thread, args=(channel, status_label, button),
                         daemon=True).start()

    def _manual_update_check_thread(self, channel, status_label, button):
        """后台线程，用于处理手动点击“检查更新”的逻辑。"""
        try:
            latest_info = get_latest_version_info(channel)

            if not latest_info:
                if self.master.winfo_exists():
                    self.master.after(0, lambda: status_label.config(text="检查失败，请重试。"))
                    _show_messagebox(self.master, "更新错误", "无法获取远程版本信息。", "error")
                return

            remote_version = latest_info['version']

            if compare_versions(APP_VERSION, remote_version):
                if self.master.winfo_exists():
                    # 调用主窗口的方法来更新侧边栏
                    self.master.after(0, self.show_update_notification_on_sidebar)
                    self.master.after(0, lambda: status_label.config(text=f"发现新版本: {remote_version}"))
                    update_message = f"发现新版本: {remote_version}\n\n更新日志:\n{latest_info.get('notes', '无')}\n\n是否立即下载并安装？"
                    if messagebox.askyesno("发现新版本", update_message, parent=self.master):
                        start_download_thread(self.master, latest_info['url'])
            else:
                if self.master.winfo_exists():
                    self.master.after(0, lambda: status_label.config(text=f"当前已是最新版本 ({APP_VERSION})"))
                    _show_messagebox(self.master, "无更新", "您目前使用的是最新版本。", "info")

        except Exception as e:
            logger.error(f"UI检查更新失败: {e}")
            if self.master.winfo_exists():
                self.master.after(0, lambda: status_label.config(text="检查更新时出错。"))
                _show_messagebox(self.master, "更新错误", f"检查更新时发生错误: {e}", "error")
        finally:
            if self.master.winfo_exists() and button.winfo_exists():
                self.master.after(0, lambda: button.config(state="normal"))

    def change_theme(self):
        """根据用户选择更改应用程序主题。"""
        selected_theme = self.advanced_page.theme_var.get()

        if selected_theme == "自动":
            self._apply_system_theme()
        elif selected_theme == "深色":
            sv_ttk.set_theme("dark")
            self.is_dark_mode = True
            self.accent_color = "#5d3a4f"
        else:  # "浅色"
            sv_ttk.set_theme("light")
            self.is_dark_mode = False
            self.accent_color = "#dbbcc2"

        # 使用 "after" 来延迟UI更新，确保sv_ttk有时间应用主题
        self.master.after(50, self._finalize_theme_change)

    def _finalize_theme_change(self):
        """在主题设置后完成UI更新。"""
        self._setup_styles()
        self._update_ui_theme()
        self._save_current_settings()

    def _apply_system_theme(self):
        try:
            import darkdetect
            system_theme = darkdetect.theme().lower()
            if system_theme == 'dark':
                sv_ttk.set_theme("dark")
                self.is_dark_mode = True
                self.accent_color = "#5d3a4f"
            else:
                sv_ttk.set_theme("light")
                self.is_dark_mode = False
                self.accent_color = "#dbbcc2"
        except Exception as e:
            sv_ttk.set_theme("light")
            self.is_dark_mode = False
            self.accent_color = "#dbbcc2"
            logger.warning(f"无法检测系统主题: {e}")

    def _detect_system_accent_color(self):
        pass

    def setup_theme_monitoring(self):
        if platform.system() in ["Windows", "Darwin"]:
            self._check_theme_change()

    def _check_theme_change(self):
        try:
            if self.advanced_page.theme_var.get() == "自动":
                import darkdetect
                current_theme = darkdetect.theme().lower()
                if (current_theme == 'dark' and not self.is_dark_mode) or \
                        (current_theme == 'light' and self.is_dark_mode):
                    self._apply_system_theme()
                    # 延迟最终的样式和UI更新
                    self.master.after(50, self._finalize_theme_change)
        except Exception as e:
            logger.warning(f"检查主题变化失败: {e}")
        self.master.after(10000, self._check_theme_change)

    def _update_ui_theme(self):
        self.sidebar.update_theme()
        if hasattr(self, 'start_page') and hasattr(self.start_page, 'update_theme'):
            self.start_page.update_theme()
        if hasattr(self, 'advanced_page') and hasattr(self.advanced_page, 'update_theme'):
            self.advanced_page.update_theme()
        self._show_page(self.current_page)

    def _setup_window(self):
        self.master.title(APP_TITLE)
        width, height = 1100, 750
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.master.geometry(f"{width}x{height}+{x}+{y}")
        self.master.minsize(width, height)

        # --- 设置任务栏和窗口图标 ---
        try:
            ico_path = resource_path("res/ico.ico")
            # 使用更可靠的 iconphoto 方法
            icon_image = Image.open(ico_path)
            self.app_icon = ImageTk.PhotoImage(icon_image)
            self.master.iconphoto(True, self.app_icon)
        except Exception as e:
            logger.warning(f"无法加载窗口图标: {e}")
            # 如果新方法失败，尝试旧方法
            try:
                self.master.iconbitmap(ico_path)
            except Exception as e2:
                logger.warning(f"备用图标加载方法也失败: {e2}")

    def _initialize_model(self, settings: dict):
        """根据设置初始化模型，优先加载已保存的模型。"""
        saved_model_name = settings.get("selected_model") if settings else None
        model_path = None
        res_dir = resource_path("res")

        # 1. 尝试从设置中加载模型
        if saved_model_name:
            potential_path = os.path.join(res_dir, saved_model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                logger.info(f"从设置加载模型: {saved_model_name}")
            else:
                logger.warning(f"设置中保存的模型文件不存在: {saved_model_name}。将尝试加载默认模型。")

        # 2. 如果设置中没有模型或文件不存在，则查找第一个可用的模型作为后备
        if not model_path:
            model_path = self._find_model_file()  # 此方法会查找第一个.pt文件
            if model_path:
                logger.info(f"加载找到的第一个模型: {os.path.basename(model_path)}")

        # 3. 初始化 ImageProcessor
        self.image_processor = ImageProcessor(model_path)
        if model_path:
            self.image_processor.model_path = model_path
            # 更新 model_var，以便UI（如下拉框）能同步显示正确的模型名称
            self.model_var.set(os.path.basename(model_path))
        else:
            # 处理未找到任何模型文件的情况
            self.image_processor.model = None
            self.image_processor.model_path = None
            self.model_var.set("")
            logger.error("在 res 目录中未找到任何有效的模型文件 (.pt)。")

    def _find_model_file(self) -> str or None:
        try:
            res_dir = resource_path("res")
            if not os.path.exists(res_dir) or not os.path.isdir(res_dir):
                return None
            model_files = [f for f in os.listdir(res_dir) if f.lower().endswith('.pt')]
            if not model_files:
                return None
            return os.path.join(res_dir, model_files[0])
        except Exception as e:
            logger.error(f"查找模型文件时出错: {e}")
            return None

    def _create_ui_elements(self):
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self.master, self)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.content_frame = ttk.Frame(self.master)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)

        self.start_page = StartPage(self.content_frame, self)
        self.advanced_page = AdvancedPage(self.content_frame, self)
        self.preview_page = PreviewPage(self.content_frame, self)
        self.about_page = AboutPage(self.content_frame, self)

        self.status_bar = InfoBar(self.master)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._show_page("settings")

        if hasattr(self, 'advanced_page') and hasattr(self.advanced_page, 'model_combobox'):
            self.advanced_page._refresh_model_list()

    def _setup_styles(self):
        style = ttk.Style()
        sidebar_bg = self.accent_color

        try:
            hex_color = sidebar_bg.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            sidebar_fg = "#000000" if brightness > 128 else "#FFFFFF"
        except:
            sidebar_fg = "#FFFFFF"  # 如果颜色计算失败，默认为白色

        sidebar_fg = "#FFFFFF"
        highlight_color = "#FFFFFF"
        self.sidebar_bg = sidebar_bg
        self.sidebar_fg = sidebar_fg
        self.highlight_color = highlight_color
        try:
            r, g, b = self.master.winfo_rgb(sidebar_bg)
            r, g, b = r // 257, g // 257, b // 257
            self.sidebar_hover_bg = f"#{min(255, r + 30):02x}{min(255, g + 30):02x}{min(255, b + 30):02x}"
        except tk.TclError:
            self.sidebar_hover_bg = sidebar_bg

        style.configure("Sidebar.TFrame", background=sidebar_bg)
        style.configure("Sidebar.TLabel", background=sidebar_bg, foreground=sidebar_fg)
        style.configure("Sidebar.Version.TLabel", background=sidebar_bg, foreground=sidebar_fg, font=("Segoe UI", 8))
        style.configure("Sidebar.Notification.TLabel", background=sidebar_bg, foreground="#FFFF00",
                        font=("Segoe UI", 9, "bold"))
        style.configure("Sidebar.Title.TLabel", background=sidebar_bg, foreground=sidebar_fg,
                        font=("Segoe UI", 12, "bold"))

        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), padding=(0, 10, 0, 10))
        style.configure("Process.TButton", font=("Segoe UI", 11), padding=(10, 5))

        style.map("Selected.TButton",
                  background=[('!active', self.accent_color), ('active', self.accent_color)],
                  foreground=[('!active', sidebar_fg), ('active', sidebar_fg)])

    def _show_page(self, page_id: str):
        self.sidebar.set_active_button(page_id)
        self.start_page.pack_forget()
        self.preview_page.pack_forget()
        self.advanced_page.pack_forget()
        self.about_page.pack_forget()

        if page_id == "settings":
            self.start_page.pack(fill="both", expand=True)
            self.status_bar.status_label.config(text="就绪")
        elif page_id == "preview":
            self.preview_page.pack(fill="both", expand=True)
            if hasattr(self, 'start_page'):
                file_path = self.start_page.file_path_entry.get()
                if file_path and os.path.isdir(file_path):
                    if self.preview_page.file_listbox.size() == 0:
                        self.preview_page.update_file_list(file_path)

                    file_count = self.preview_page.file_listbox.size()
                    self.status_bar.status_label.config(text=f"当前文件夹下有 {file_count} 个图像文件")

                    if self.preview_page.file_listbox.size() > 0 and not self.preview_page.file_listbox.curselection():
                        self.preview_page.file_listbox.selection_set(0)
                        self.preview_page.on_file_selected(None)
                else:
                    self.status_bar.status_label.config(text="请在“开始”页面中设置有效的图像文件路径")
        elif page_id == "advanced":
            self.advanced_page.pack(fill="both", expand=True)
            self.status_bar.status_label.config(text="就绪")
        elif page_id == "about":
            self.about_page.pack(fill="both", expand=True)
            self.status_bar.status_label.config(text="就绪")

        if page_id != "preview" and not self.is_processing:
            self.status_bar.status_label.config(text="就绪")

        self.current_page = page_id

    def _bind_events(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.start_page.file_path_entry.bind("<Return>", self._validate_and_update_file_path)
        self.start_page.save_path_entry.bind("<Return>", self._validate_and_update_save_path)
        self.preview_page.file_listbox.bind("<<ListboxSelect>>", self.preview_page.on_file_selected)
        self.preview_page.image_label.bind("<Double-1>", self.preview_page.on_image_double_click)
        self.preview_page.show_detection_var.trace("w", self.preview_page.toggle_detection_preview)
        self.start_page.save_detect_image_var.trace("w", lambda *args: self._save_current_settings())
        self.start_page.copy_img_var.trace("w", lambda *args: self._save_current_settings())
        self.advanced_page.controller.use_fp16_var.trace("w", lambda *args: self._save_current_settings())
        self.advanced_page.controller.iou_var.trace("w", lambda *args: self._save_current_settings())
        self.advanced_page.controller.conf_var.trace("w", lambda *args: self._save_current_settings())
        self.advanced_page.controller.use_augment_var.trace("w", lambda *args: self._save_current_settings())
        self.advanced_page.controller.use_agnostic_nms_var.trace("w", lambda *args: self._save_current_settings())
        self.update_channel_var.trace("w", lambda *args: self._save_current_settings())
        self.preview_page.export_format_var.trace("w", lambda *args: self._save_current_settings())

        self.master.bind("<Up>", self._navigate_image_up)
        self.master.bind("<Down>", self._navigate_image_down)

    def _save_current_settings(self):
        if not self.settings_manager: return
        settings = self._get_current_settings()
        if self.settings_manager.save_settings(settings): logger.info("设置已保存")

    def _get_current_settings(self):
        settings = {"file_path": self.start_page.file_path_entry.get(),
                    "save_path": self.start_page.save_path_entry.get(),
                    "save_detect_image": self.start_page.save_detect_image_var.get(),
                    "copy_img": self.start_page.copy_img_var.get(),
                    "use_fp16": self.advanced_page.controller.use_fp16_var.get(),
                    "iou": self.advanced_page.controller.iou_var.get(),
                    "conf": self.advanced_page.controller.conf_var.get(),
                    "use_augment": self.advanced_page.controller.use_augment_var.get(),
                    "use_agnostic_nms": self.advanced_page.controller.use_agnostic_nms_var.get(),
                    "update_channel": self.update_channel_var.get(),
                    "theme": self.advanced_page.theme_var.get(),
                    "selected_model": self.model_var.get()}

        if hasattr(self, 'preview_page'):
            settings["export_format"] = self.preview_page.export_format_var.get()

        return settings

    def _load_settings_to_ui(self, settings: dict):
        if not settings:
            return
        try:
            if "file_path" in settings and settings["file_path"] and os.path.exists(settings["file_path"]):
                if hasattr(self, 'preview_page') and hasattr(self.preview_page, 'file_listbox'):
                    self.preview_page.file_listbox.delete(0, tk.END)

                self.start_page.file_path_entry.delete(0, tk.END)
                self.start_page.file_path_entry.insert(0, settings["file_path"])
                self.get_temp_photo_dir(update=True)
                self.preview_page.update_file_list(settings["file_path"])
            if "save_path" in settings and settings["save_path"]:
                self.start_page.save_path_entry.delete(0, tk.END)
                self.start_page.save_path_entry.insert(0, settings["save_path"])
            self.start_page.save_detect_image_var.set(settings.get("save_detect_image", True))
            self.start_page.copy_img_var.set(settings.get("copy_img", False))
            self.advanced_page.controller.use_fp16_var.set(settings.get("use_fp16", False))
            iou_value = settings.get("iou", 0.3)
            conf_value = settings.get("conf", 0.25)
            self.advanced_page.controller.iou_var.set(iou_value)
            self.advanced_page.controller.conf_var.set(conf_value)
            self.advanced_page._update_iou_label(iou_value)
            self.advanced_page._update_conf_label(conf_value)
            self.advanced_page.controller.use_augment_var.set(settings.get("use_augment", True))
            self.advanced_page.controller.use_agnostic_nms_var.set(settings.get("use_agnostic_nms", True))
            self.advanced_page._update_iou_label(settings.get("iou", 0.3))
            self.advanced_page._update_conf_label(settings.get("conf", 0.25))
            self.update_channel_var.set(settings.get("update_channel", "稳定版 (Release)"))


            # Load theme
            self.advanced_page.theme_var.set(settings.get("theme", "自动"))
            self.change_theme()

            if hasattr(self, 'preview_page'):
                self.preview_page.export_format_var.set(settings.get("export_format", "CSV"))

            '''# <<< 新增：加载并应用模型选择 >>>
            saved_model = settings.get("selected_model", "")
            available_models = self.advanced_page.model_combobox.cget('values')

            # 检查保存的模型是否存在于可用模型列表中
            if saved_model and saved_model in available_models:
                self.model_var.set(saved_model)
            elif available_models:
                # 如果没有保存的模型或模型文件已不存在，则默认选择列表中的第一个
                self.model_var.set(available_models[0])

            # 手动调用模型更改处理函数，以确保后端ImageProcessor使用正确的模型
            #self.advanced_page._change_model()
            # <<< 新增结束 >>>'''

        except Exception as e:
            logger.error(f"加载设置到UI失败: {e}")

    def on_closing(self):
        if self.is_processing:
            if not messagebox.askyesno("确认退出", "图像处理正在进行中，确定要退出吗？"): return
            self.processing_stop_flag.set()
        if hasattr(self, 'preview_page'): self.preview_page._save_validation_data()
        self._save_current_settings()
        self.master.destroy()

    def browse_file_path(self):
        folder_selected = filedialog.askdirectory(title="选择图像文件所在文件夹")
        if folder_selected:
            self.start_page.file_path_entry.delete(0, tk.END)
            self.start_page.file_path_entry.insert(0, folder_selected)
            self._validate_and_update_file_path()
        else:
            # Handle user cancelling the dialog by clearing the path
            self.start_page.file_path_entry.delete(0, tk.END)
            self._validate_and_update_file_path()

    def _validate_and_update_file_path(self, event=None):
        folder_selected = self.start_page.file_path_entry.get().strip()

        # Always clear previews first to handle path changes correctly
        self.preview_page.clear_previews()

        if not folder_selected:
            self.status_bar.status_label.config(text="文件路径已清除")
            self._save_current_settings()
            return

        if os.path.isdir(folder_selected):
            self.get_temp_photo_dir(update=True)
            self.preview_page.update_file_list(folder_selected)
            file_count = self.preview_page.file_listbox.size()
            self.status_bar.status_label.config(text=f"文件路径已设置，找到 {file_count} 个图像文件。")
            self._save_current_settings()
            if self.current_page == "preview":
                self._show_page("preview")
        else:
            messagebox.showerror("路径错误", f"提供的图像文件路径不存在或不是一个文件夹:\n'{folder_selected}'")
            self.status_bar.status_label.config(text="无效的文件路径")

    def browse_save_path(self):
        folder_selected = filedialog.askdirectory(title="选择结果保存文件夹")
        if folder_selected:
            self.start_page.save_path_entry.delete(0, tk.END)
            self.start_page.save_path_entry.insert(0, folder_selected)
            self._validate_and_update_save_path()

    def _validate_and_update_save_path(self, event=None):
        save_path = self.start_page.save_path_entry.get().strip()
        if not save_path: return
        if not os.path.isdir(save_path):
            if messagebox.askyesno("确认创建路径", f"结果保存路径不存在，是否要创建它？\n\n{save_path}"):
                try:
                    os.makedirs(save_path, exist_ok=True)
                    self.start_page.save_path_entry.delete(0, tk.END)
                    self.start_page.save_path_entry.insert(0, save_path)
                    self.status_bar.status_label.config(text=f"结果保存路径已创建: {save_path}")
                    self._save_current_settings()
                except Exception as e:
                    messagebox.showerror("路径错误", f"无法创建结果保存路径:\n{e}")
                    self.status_bar.status_label.config(text="结果保存路径创建失败")
            else:
                self.status_bar.status_label.config(text="操作已取消，请输入有效的结果保存路径。")
                pass
        else:
            self.start_page.save_path_entry.delete(0, tk.END)
            self.start_page.save_path_entry.insert(0, save_path)
            self.status_bar.status_label.config(text=f"结果保存路径已设置: {save_path}")
            self._save_current_settings()

    def show_params_help(self):
        help_text = """
        **检测阈值设置**
        - **IOU阈值:** 控制对象检测中非极大值抑制（NMS）的重叠阈值。较高的值会减少重叠框，但可能导致部分目标漏检。
        - **置信度阈值:** 检测对象的最小置信度分数。较高的值只显示高置信度的检测结果，减少误检。

        **模型加速选项**
        - **使用FP16加速:** 使用半精度浮点数进行推理，可以加快速度但可能会略微降低精度。需要兼容的NVIDIA GPU。

        **高级检测选项**
        - **使用数据增强:** 在测试时使用数据增强（TTA），通过对输入图像进行多种变换并综合结果，可能会提高准确性，但会显著降低处理速度。
        - **使用类别无关NMS:** 在所有类别上一起执行NMS，对于检测多种相互重叠的物种可能有用。
        """
        messagebox.showinfo("参数说明", help_text, parent=self.master)

    def get_temp_photo_dir(self, update=False):
        source_path = self.start_page.file_path_entry.get()
        if not source_path: return None
        path_hash = hashlib.md5(source_path.encode()).hexdigest()
        base_dir = self.settings_manager.base_dir
        temp_dir = os.path.join(base_dir, "temp", "photo", path_hash)
        if update:
            self.current_temp_photo_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def clear_image_cache(self):
        cache_dir = os.path.join(self.settings_manager.base_dir, "temp", "photo")
        if messagebox.askyesno("确认清除缓存",
                               f"是否清空图片缓存？\n\n此操作将删除以下文件夹及其所有内容：\n{cache_dir}\n\n注意：这不会影响您的原始图片或已保存的结果。",
                               parent=self.master):
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    messagebox.showinfo("成功", "图片缓存已成功清除。", parent=self.master)
                except Exception as e:
                    messagebox.showerror("错误", f"清除缓存时发生错误：\n{e}", parent=self.master)
            else:
                messagebox.showinfo("提示", "缓存目录不存在，无需清除。", parent=self.master)

    def toggle_processing_state(self):
        if not self.is_processing:
            self.check_for_cache_and_process()
        else:
            self.stop_processing()

    def check_for_cache_and_process(self):
        cache_file = os.path.join(self.settings_manager.settings_dir, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if 'processed_files' in cache_data and 'total_files' in cache_data:
                    processed = cache_data.get('processed_files', 0)
                    total = cache_data.get('total_files', 0)
                    file_path = cache_data.get('file_path', '')
                    if messagebox.askyesno("发现未完成任务",
                                           f"检测到上次有未完成的任务，是否继续？\n已处理：{processed}/{total} at {file_path}"):
                        self._load_cache_data_from_file(cache_data)
                        self.start_processing(resume_from=processed)
                        return
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}")
        self.start_processing()

    def start_processing(self, resume_from=0):
        file_path = self.start_page.file_path_entry.get()
        save_path = self.start_page.save_path_entry.get()
        save_detect_image = self.start_page.save_detect_image_var.get()
        copy_img = self.start_page.copy_img_var.get()
        use_fp16 = self.advanced_page.controller.use_fp16_var.get()

        if not self._validate_inputs(file_path, save_path): return
        if self.is_processing: return

        selected_tab_id = self.preview_page.preview_notebook.select()
        if selected_tab_id:
            tab_text = self.preview_page.preview_notebook.tab(selected_tab_id, "text")
            if tab_text == "检验校验(时间)" or tab_text == "检验校验(物种)":
                self.preview_page.preview_notebook.select(self.preview_page.image_preview_tab)

        self._set_processing_state(True)
        self._show_page("preview")
        if resume_from == 0:
            self.excel_data = []
            self._clear_current_validation_file()

        threading.Thread(
            target=self._process_images_thread,
            args=(file_path, save_path, save_detect_image, copy_img, use_fp16, resume_from),
            daemon=True
        ).start()

    def stop_processing(self):
        if messagebox.askyesno("停止确认", "确定要停止图像处理吗？\n处理进度将被保存，下次可以继续。"):
            self.processing_stop_flag.set()
            self.status_bar.status_label.config(text="正在停止处理...")
        else:
            messagebox.showinfo("信息", "处理继续进行。")

    def _process_images_thread(self, file_path, save_path, save_detect_image, copy_img, use_fp16,
                               resume_from=0):
        start_time = time.time()
        excel_data = [] if resume_from == 0 else self.excel_data
        processed_files = resume_from
        stopped_manually = False
        earliest_date = None
        temp_photo_dir = self.get_temp_photo_dir()

        try:
            iou = self.advanced_page.controller.iou_var.get()
            conf = self.advanced_page.controller.conf_var.get()
            augment = self.advanced_page.controller.use_augment_var.get()
            agnostic_nms = self.advanced_page.controller.use_agnostic_nms_var.get()
            image_files = sorted([f for f in os.listdir(file_path) if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)])
            total_files = len(image_files)
            if resume_from > 0:
                image_files = image_files[resume_from:]
                if excel_data:
                    valid_dates = [item['拍摄日期对象'] for item in excel_data if item.get('拍摄日期对象')]
                    if valid_dates:
                        earliest_date = min(valid_dates)

            for idx, filename in enumerate(image_files):
                if self.processing_stop_flag.is_set():
                    stopped_manually = True
                    break

                if self.master.winfo_exists():
                    self.master.after(0, lambda f=filename: self.status_bar.status_label.config(text=f"正在处理: {f}"))
                try:
                    listbox_idx = self.preview_page.file_listbox.get(0, "end").index(filename)
                    if self.master.winfo_exists():
                        self.master.after(0, lambda i=listbox_idx: (
                            self.preview_page.file_listbox.selection_clear(0, "end"),
                            self.preview_page.file_listbox.selection_set(i),
                            self.preview_page.file_listbox.see(i)
                        ))
                except ValueError:
                    pass

                elapsed_time = time.time() - start_time
                speed = (processed_files - resume_from + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_time = (total_files - (processed_files + 1)) / speed if speed > 0 else float('inf')

                if self.master.winfo_exists():
                    self.master.after(0, lambda p=processed_files + 1, t=total_files, s=speed, r=remaining_time:
                    self.start_page.progress_frame.update_progress(value=p, total=t, speed=s, remaining_time=r))

                try:
                    img_path = os.path.join(file_path, filename)
                    image_info, img = ImageMetadataExtractor.extract_metadata(img_path, filename)
                    species_info = self.image_processor.detect_species(img_path, bool(use_fp16), iou, conf, augment,
                                                                       agnostic_nms)
                    species_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detect_results = species_info.get('detect_results')
                    if detect_results:
                        #self.image_processor.save_detection_temp(detect_results, filename, temp_photo_dir)
                        self.image_processor.save_detection_info_json(detect_results, filename, species_info,
                                                                      temp_photo_dir)
                        if self.master.winfo_exists():
                            self.master.after(0, lambda p=img_path, d=detect_results, info=species_info.copy(): (
                                self.preview_page.update_image_preview(p, show_detection=True, detection_results=d),
                                self.preview_page.update_image_info(p, os.path.basename(p)),
                                self.preview_page._update_detection_info(info)
                            ))
                    if save_detect_image: self.image_processor.save_detection_result(detect_results, filename,
                                                                                     save_path)
                    if copy_img and img: self._copy_image_by_species(img_path, save_path,
                                                                     species_info['物种名称'].split(','))
                    if 'detect_results' in species_info: del species_info['detect_results']
                    image_info.update(species_info)
                    excel_data.append(image_info)
                except Exception as e:
                    logger.error(f"处理文件 {filename} 失败: {e}")
                processed_files += 1
                if processed_files % 10 == 0: self._save_processing_cache(excel_data, file_path, save_path,
                                                                          save_detect_image, True, copy_img,
                                                                          use_fp16, processed_files, total_files,
                                                                          iou, conf, augment, agnostic_nms)
                try:
                    del img_path, image_info, img, species_info, detect_results
                except NameError:
                    pass
                gc.collect()

            if not stopped_manually:
                if self.master.winfo_exists():
                    self.master.after(0, lambda: self.start_page.progress_frame.update_progress(value=total_files,
                                                                                                total=total_files,
                                                                                                speed=0,
                                                                                                remaining_time="已完成"))
                self.excel_data = excel_data
                excel_data = DataProcessor.process_independent_detection(excel_data, self.confidence_settings)
                if earliest_date: excel_data = DataProcessor.calculate_working_days(excel_data, earliest_date)
                #if excel_data and output_excel: self._export_and_open_excel(excel_data, save_path)
                self._delete_processing_cache()
                if self.master.winfo_exists(): self.status_bar.status_label.config(text="处理完成！")
                messagebox.showinfo("成功", "图像处理完成！")
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}")
            messagebox.showerror("错误", f"处理过程中发生错误: {e}")
        finally:
            if self.master.winfo_exists():
                self._set_processing_state(False)
            gc.collect()

    def _set_processing_state(self, is_processing: bool):
        self.is_processing = is_processing
        self.start_page.set_processing_state(is_processing)
        self.sidebar.set_processing_state(is_processing)
        if is_processing:
            self.preview_page.show_detection_var.set(True)
            self.processing_stop_flag.clear()
        else:
            if self.processing_stop_flag.is_set():
                if self.master.winfo_exists(): self.status_bar.status_label.config(text="处理已停止")
            elif self.master.winfo_exists() and self.status_bar.status_label.cget("text") != "处理完成！":
                self.status_bar.status_label.config(text="就绪")

    def _validate_inputs(self, file_path: str, save_path: str) -> bool:
        if not file_path or not os.path.isdir(file_path):
            messagebox.showerror("错误", "请提供有效的源文件夹路径。")
            return False
        if not save_path or not os.path.isdir(save_path):
            messagebox.showerror("错误", "请提供有效的保存文件夹路径。")
            return False
        return True

    def _copy_image_by_species(self, img_path: str, save_path: str, species_names: list):
        for name in species_names:
            if name:
                to_path = os.path.join(save_path, name)
                os.makedirs(to_path, exist_ok=True)
                from shutil import copy
                copy(img_path, to_path)

    def _export_and_open_excel(self, excel_data, save_path):
        from system.config import DEFAULT_EXCEL_FILENAME
        output_file_path = os.path.join(save_path, DEFAULT_EXCEL_FILENAME)
        if DataProcessor.export_to_excel(excel_data, output_file_path):
            if messagebox.askyesno("成功", f"数据已导出到 {output_file_path}\n是否立即打开?"):
                try:
                    os.startfile(output_file_path)
                except Exception as e:
                    messagebox.showerror("错误", f"无法打开文件: {e}")

    def _save_processing_cache(self, excel_data, file_path, save_path, save_detect_image, output_excel, copy_img,
                               use_fp16, processed_files, total_files, iou, conf, use_augment, use_agnostic_nms):
        def make_serializable(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items() if k != 'detect_results'}
            if isinstance(obj, list): return [make_serializable(i) for i in obj]
            if hasattr(obj, '__dict__'): return None
            return obj

        serializable_excel_data = make_serializable(excel_data)
        cache_data = {'file_path': file_path, 'save_path': save_path, 'save_detect_image': save_detect_image,
                      'output_excel': output_excel, 'copy_img': copy_img, 'use_fp16': use_fp16,
                      'processed_files': processed_files, 'total_files': total_files,
                      'excel_data': serializable_excel_data,
                      'iou': iou,
                      'conf': conf,
                      'use_augment': use_augment,
                      'use_agnostic_nms': use_agnostic_nms}
        cache_file = os.path.join(self.settings_manager.settings_dir, "cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _delete_processing_cache(self):
        cache_file = os.path.join(self.settings_manager.settings_dir, "cache.json")
        if os.path.exists(cache_file): os.remove(cache_file)

    def _load_cache_data_from_file(self, cache_data):
        self._load_settings_to_ui(cache_data)
        self.excel_data = cache_data.get('excel_data', [])
        for item in self.excel_data:
            if '拍摄日期对象' in item and isinstance(item['拍摄日期对象'], str):
                try:
                    item['拍摄日期对象'] = datetime.fromisoformat(item['拍摄日期对象'])
                except ValueError:
                    item['拍摄日期对象'] = None

    def _resume_processing(self):
        self._load_cache_data_from_file(self.cache_data)
        self.start_processing(resume_from=self.cache_data.get('processed_files', 0))

    def _clear_current_validation_file(self):
        """删除当前所选文件夹的validation.json文件。"""
        temp_photo_dir = self.get_temp_photo_dir()
        if temp_photo_dir:
            validation_file_path = os.path.join(temp_photo_dir, "validation.json")
            if os.path.exists(validation_file_path):
                try:
                    os.remove(validation_file_path)
                    logger.info(f"已清除旧的校验文件: {validation_file_path}")
                except Exception as e:
                    logger.error(f"清除旧的校验文件失败: {e}")
        # 同时清除内存中的数据
        if hasattr(self, 'preview_page'):
            self.preview_page.validation_data.clear()

    def _navigate_image_up(self, event=None):
        """处理向上箭头键事件，用于在预览页面切换图片。"""
        if self.current_page == "preview":
            self.preview_page.navigate_listbox('up')
            return "break"  # 新增：阻止事件继续传播

    def _navigate_image_down(self, event=None):
        """处理向下箭头键事件，用于在预览页面切换图片。"""
        if self.current_page == "preview":
            self.preview_page.navigate_listbox('down')
            return "break"  # 新增：阻止事件继续传播