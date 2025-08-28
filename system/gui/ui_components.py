import tkinter as tk
from tkinter import ttk
import logging
import sys
import platform  # 导入 platform 模块

logger = logging.getLogger(__name__)


class RoundedButton(tk.Canvas):
    """圆角按钮实现 - 选中时在左侧显示高亮指示条"""

    def __init__(self, parent, text="", command=None, bg="#2c3e50", fg="#ffffff",
                 width=160, height=40, radius=10, highlight_color="#ffffff",
                 show_indicator=True, **kwargs):
        """初始化圆角按钮

        Args:
            parent: 父级窗口
            text: 按钮文本
            command: 点击回调函数
            bg: 背景色
            fg: 文字颜色
            width: 按钮宽度
            height: 按钮高度
            radius: 圆角半径
            highlight_color: 高亮指示条颜色
        """
        # 安全地获取父组件背景色
        # 根据深色模式设置默认背景色
        is_dark_mode = False
        try:
            # 尝试确定是否为深色模式
            if hasattr(parent, 'is_dark_mode'):
                is_dark_mode = parent.is_dark_mode
            elif hasattr(parent, 'master') and hasattr(parent.master, 'is_dark_mode'):
                is_dark_mode = parent.master.is_dark_mode
            # 根据ttk样式判断
            else:
                try:
                    style = ttk.Style()
                    current_theme = style.theme_use()
                    is_dark_mode = (current_theme == "dark" or
                                    current_theme == "sun-valley-dark" or
                                    "dark" in current_theme.lower())
                except:
                    pass
        except:
            pass

        # 根据深色模式设置默认背景色
        parent_bg = '#1C1C1C' if is_dark_mode else '#FAFAFA'

        try:
            # 尝试多种方法获取父组件背景色
            if hasattr(parent, 'winfo_rgb'):
                try:
                    # 使用标准的tk方法获取实际背景色
                    parent_bg = parent.cget('background')
                except Exception:
                    # 某些ttk组件使用'bg'而不是'background'
                    try:
                        parent_bg = parent.cget('bg')
                    except Exception:
                        pass
        except Exception:
            pass  # 忽略所有错误，使用默认背景色

        # 创建Canvas
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent_bg,  # 使用获取的背景色
            highlightthickness=0,
            bd=0,  # 无边框
            **kwargs
        )

        self.bg = bg
        self.fg = fg
        self.command = command
        self.radius = radius
        self.text = text
        self.width = width
        self.height = height
        self.active = False  # 默认非激活状态
        self.highlight_color = highlight_color
        self.parent_bg = parent_bg  # 保存父组件背景色
        self.show_indicator = show_indicator  # 是否显示左侧高亮指示条

        # 计算悬停状态的颜色
        r, g, b = [int(self.bg[i:i + 2], 16) for i in (1, 3, 5)]
        brightness = (r * 299 + g * 587 + b * 114) / 1000

        if brightness < 128:  # 深色背景
            # 变亮
            self.hover_bg = f"#{min(255, int(r * 1.3)):02x}{min(255, int(g * 1.3)):02x}{min(255, int(b * 1.3)):02x}"
        else:
            # 变暗
            self.hover_bg = f"#{max(0, int(r * 0.9)):02x}{max(0, int(g * 0.9)):02x}{max(0, int(b * 0.9)):02x}"

        # 绘制初始状态
        self._draw_button("normal")

        # 绑定事件
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def update_theme(self):
        """更新按钮的主题以匹配父主题。"""
        try:
            # 这对于tk和ttk小部件都应该有效
            new_parent_bg = self.master.cget('background')
            self.config(bg=new_parent_bg)
        except Exception:
            # 复杂小部件或主题问题的后备方案
            style = ttk.Style()
            new_parent_bg = style.lookup('TFrame', 'background')
            self.config(bg=new_parent_bg)

        self.parent_bg = self.cget('background')

        # 根据当前状态（活动或正常）重新绘制按钮
        self.set_active(self.active)

    def _on_press(self, event):
        """按钮按下事件处理"""
        self._draw_button("active")

    def _on_release(self, event):
        """按钮释放事件处理"""
        self._draw_button("hover")
        if self.command:
            self.command()

    def _on_enter(self, event):
        """鼠标进入事件处理"""
        self._draw_button("hover")

    def _on_leave(self, event):
        """鼠标离开事件处理"""
        if not self.active:
            self._draw_button("normal")
        else:
            self._draw_button("active")

    def set_active(self, active=True):
        """设置按钮激活状态

        Args:
            active: 是否激活
        """
        self.active = active
        if active:
            self._draw_button("active")
        else:
            self._draw_button("normal")

    def _draw_button(self, state="normal"):
        """绘制按钮，根据状态使用不同的背景色

        Args:
            state: 按钮状态，可选值为 "normal", "hover", "active"
        """
        # 清除所有内容
        self.delete("all")

        # 根据状态选择背景色
        if state == "active":
            bg_color = self.bg  # 选中时使用原始背景色
            font_style = ("Segoe UI", 11, "bold")  # 选中时文字加粗
        elif state == "hover":
            bg_color = self.hover_bg
            font_style = ("Segoe UI", 11)
        else:  # normal
            bg_color = self.bg
            font_style = ("Segoe UI", 11)

        # 绘制圆角矩形
        radius = self.radius
        width = self.width
        height = self.height

        # 绘制主体矩形
        self.create_rectangle(
            radius, 0,
            width - radius, height,
            fill=bg_color, outline="", tags="body"
        )

        # 绘制左右侧矩形
        self.create_rectangle(
            0, radius,
            radius, height - radius,
            fill=bg_color, outline="", tags="body"
        )

        self.create_rectangle(
            width - radius, radius,
            width, height - radius,
            fill=bg_color, outline="", tags="body"
        )

        # 绘制四个角的圆弧 - 使用按钮当前背景色，而不是父组件背景色
        self.create_arc(
            0, 0, radius * 2, radius * 2,
            start=90, extent=90, fill=bg_color, outline=""
        )

        self.create_arc(
            width - radius * 2, 0, width, radius * 2,
            start=0, extent=90, fill=bg_color, outline=""
        )

        self.create_arc(
            width - radius * 2, height - radius * 2, width, height,
            start=270, extent=90, fill=bg_color, outline=""
        )

        self.create_arc(
            0, height - radius * 2, radius * 2, height,
            start=180, extent=90, fill=bg_color, outline=""
        )

        # 如果是激活状态，添加左侧高亮指示条
        if state == "active" and self.show_indicator:
            # 设置缩短的高亮指示条参数
            indicator_width = 2  # 指示条宽度
            indicator_height = height * 0.6  # 指示条高度为按钮高度的60%
            indicator_y_offset = (height - indicator_height) / 2  # 垂直居中

            # 使用固定坐标确保上下半圆对齐
            x_left = 0
            x_right = indicator_width

            # 计算半圆直径
            circle_diameter = indicator_width * 2

            # 计算矩形的上下边界
            rect_top = indicator_y_offset + circle_diameter / 2
            rect_bottom = indicator_y_offset + indicator_height - circle_diameter / 2

            # 绘制主体矩形部分
            if rect_bottom > rect_top:  # 确保有中间部分
                self.create_rectangle(
                    x_left,
                    rect_top,
                    x_right * 2,
                    rect_bottom,
                    fill=self.highlight_color,
                    outline="",
                    tags="highlight"
                )

            # 绘制上半圆 - 确保位置精确对齐
            self.create_arc(
                x_left,
                indicator_y_offset,
                circle_diameter,
                indicator_y_offset + circle_diameter,
                start=0,
                extent=180,  # 180度弧，形成半圆
                fill=self.highlight_color,
                outline="",
                tags="highlight"
            )

            # 绘制下半圆 - 确保与上半圆完全对齐
            self.create_arc(
                x_left,  # 与上半圆使用完全相同的x坐标
                indicator_y_offset + indicator_height - circle_diameter,
                circle_diameter * 0.75,  # 修正：使用相同的直径确保对齐
                indicator_y_offset + indicator_height,
                start=180,
                extent=180,  # 180度弧，形成半圆
                fill=self.highlight_color,
                outline="",
                tags="highlight"
            )

        # 绘制文本
        self.text_id = self.create_text(
            width // 2,
            height // 2,
            text=self.text,
            fill=self.fg,
            font=font_style,
            anchor="center",
            tags="text"
        )


class ModernFrame(ttk.Frame):
    """现代风格框架"""
    pass


class InfoBar(ttk.Frame):
    """信息栏"""

    def __init__(self, parent, **kwargs):
        """初始化信息栏"""
        super().__init__(parent, **kwargs)

        self.status_label = ttk.Label(self, text="就绪", padding=(10, 5))
        self.status_label.pack(side="left")


class SpeedProgressBar(ttk.Frame):
    def __init__(self, parent, accent_color="#2ecc71"):
        """自定义进度框架组件"""
        super().__init__(parent)
        self.parent = parent
        self.progress_var = tk.IntVar(value=0)
        self.total_var = tk.IntVar(value=100)
        self.accent_color = accent_color
        # V V V V V V V V V V V V V V V V V V V V
        # MODIFICATION: Add attributes to store speed and time text
        # V V V V V V V V V V V V V V V V V V V V
        self.speed_text = "速度: 0.00 张/秒"
        self.time_text = "剩余: N/A"
        # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        self._create_widgets()
        self.hide()

    def _create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(main_frame, height=25, bg="#E0E0E0", highlightthickness=0)
        self.canvas.pack(side="top", fill="x", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_progressbar())

    # V V V V V V V V V V V V V V V V V V V V
    # MODIFICATION: Update drawing logic to include speed and time
    # V V V V V V V V V V V V V V V V V V V V
    def _draw_progressbar(self):
        self.canvas.delete("all")
        current = self.progress_var.get()
        total = self.total_var.get() if self.total_var.get() > 0 else 1
        current = min(current, total)
        progress_ratio = current / total
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        bar_width = int(canvas_width * progress_ratio)

        # Draw background
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#E0E0E0", outline="")

        # Draw progress bar
        if bar_width > 0:
            self.canvas.create_rectangle(0, 0, bar_width, canvas_height, fill=self.accent_color, outline="")

        # --- Center Text (Percentage and Count) ---
        percentage = progress_ratio * 100
        center_text_color = "white" if bar_width > canvas_width / 2 else "black"
        self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2,
            text=f"{percentage:.1f}% ({current}/{total})",
            fill=center_text_color,
            font=("Segoe UI", 9)
        )

        # --- Right-aligned Text (Speed and Time) ---
        right_text = f"{self.speed_text} | {self.time_text}"
        right_text_x = canvas_width - 10  # 10px padding from the right

        # Estimate text width to determine color
        # This is an approximation, but works well enough
        est_text_width = len(right_text) * 7
        right_text_color = "white" if bar_width > (right_text_x - est_text_width) else "black"

        self.canvas.create_text(
            right_text_x,
            canvas_height / 2,
            text=right_text,
            fill=right_text_color,
            font=("Segoe UI", 9),
            anchor="e"  # Anchor to the east (right)
        )

    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

    def show(self):
        self.pack(fill="x", expand=True, padx=5, pady=5)

    def hide(self):
        self.pack_forget()

    # V V V V V V V V V V V V V V V V V V V V
    # MODIFICATION: Update text attributes
    # V V V V V V V V V V V V V V V V V V V V
    def update_progress(self, value, total=None, speed=None, remaining_time=None):
        if total is not None:
            self.total_var.set(max(1, total))
        self.progress_var.set(value)

        if speed is not None:
            self.speed_text = f"速度: {speed:.2f} 张/秒"

        if remaining_time is not None:
            # 检查 remaining_time 在比较前是否为数字
            if isinstance(remaining_time, (int, float)):
                if remaining_time == float('inf') or remaining_time > 3600 * 24:  # 避免过大的数字
                    self.time_text = "剩余: 计算中"
                elif remaining_time > 60:
                    minutes = int(remaining_time // 60)
                    seconds = int(remaining_time % 60)
                    self.time_text = f"剩余: {minutes}分{seconds}秒"
                else:
                    self.time_text = f"剩余: {int(remaining_time)}秒"
            else:
                # 如果是字符串 (例如 "已完成"), 直接显示
                self.time_text = f"剩余: {remaining_time}"

        self._draw_progressbar()
        self.update_idletasks()
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^


class CollapsiblePanel(ttk.Frame):
    """现代化可折叠面板组件"""

    def __init__(self, parent, title, subtitle="", icon=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.toggle_callbacks = []
        self._initialize_colors()

        self.header_frame = tk.Frame(self, bg=self.header_bg, cursor="hand2")
        self.header_frame.pack(fill="x", expand=False, pady=(0, 1))

        self.icon_label = None
        if icon:
            try:
                if isinstance(icon, str):
                    font_name = "Segoe UI Emoji" if platform.system() == "Windows" else "Apple Color Emoji" if platform.system() == "Darwin" else "Noto Color Emoji"
                    self.icon_label = tk.Label(self.header_frame, text=icon,
                                               font=(font_name, 20),
                                               bg=self.header_bg, fg=self.text_color)
                else:
                    self.icon_label = tk.Label(self.header_frame, image=icon,
                                               bg=self.header_bg)
                self.icon_label.pack(side="left", padx=(15, 10), pady=12)
            except Exception as e:
                logger.error(f"Failed to load icon: {e}")

        title_container = tk.Frame(self.header_frame, bg=self.header_bg)
        title_container.pack(side="left", fill="both", expand=True, pady=10)
        self.title_label = tk.Label(title_container, text=title, font=("Segoe UI", 11, "bold"), bg=self.header_bg,
                                    fg=self.text_color)
        self.title_label.pack(side="top", anchor="w")
        if subtitle:
            self.subtitle_label = tk.Label(title_container, text=subtitle, font=("Segoe UI", 9), bg=self.header_bg,
                                           fg=self.text_color)
            self.subtitle_label.pack(side="top", anchor="w")

        self.is_expanded = False
        self.toggle_button = tk.Label(self.header_frame, text="▼", font=("Segoe UI", 9), bg=self.header_bg,
                                      fg=self.text_color)
        self.toggle_button.pack(side="right", padx=(0, 15), pady=10)

        self.content_frame = ttk.Frame(self)
        self.content_padding = ttk.Frame(self.content_frame)
        self.content_padding.pack(fill="both", expand=True, padx=20, pady=(10, 20))

        self.header_frame.bind("<Button-1>", self.toggle)
        self.title_label.bind("<Button-1>", self.toggle)
        self.toggle_button.bind("<Button-1>", self.toggle)
        if self.icon_label:
            self.icon_label.bind("<Button-1>", self.toggle)

    def _initialize_colors(self):
        """根据当前主题初始化颜色变量"""
        self.style = ttk.Style()
        current_theme = self.style.theme_use()
        self.is_dark_mode = 'dark' in current_theme
        self.bg_color = "#2b2b2b" if self.is_dark_mode else "#f5f5f5"
        self.header_bg = "#333333" if self.is_dark_mode else "#e5e5e5"
        self.text_color = "#ffffff" if self.is_dark_mode else "#000000"
        self.hover_color = "#3a3a3a" if self.is_dark_mode else "#e0e0e0"

    def update_theme(self):
        """更新组件颜色以匹配新的主题"""
        self.configure(style='TFrame')  # 强制面板本身应用新主题的样式
        self._initialize_colors()

        self.header_frame.config(bg=self.header_bg)
        self.toggle_button.config(bg=self.header_bg, fg=self.text_color)

        if self.icon_label:
            self.icon_label.config(bg=self.header_bg, fg=self.text_color)

        # 更新标题和副标题的容器及标签
        title_container = self.title_label.master
        title_container.config(bg=self.header_bg)
        self.title_label.config(bg=self.header_bg, fg=self.text_color)
        if hasattr(self, 'subtitle_label'):
            self.subtitle_label.config(bg=self.header_bg, fg=self.text_color)

        # 更新内容区域的样式
        self.content_frame.configure(style="TFrame")
        self.content_padding.configure(style="TFrame")

    def toggle(self, event=None):
        if self.is_expanded:
            self.collapse()
        else:
            self.expand()
        for callback in self.toggle_callbacks:
            callback(self, self.is_expanded)

    def expand(self):
        self.content_frame.pack(fill="both", expand=True)
        self.toggle_button.configure(text="▲")
        self.is_expanded = True

    def collapse(self):
        self.content_frame.pack_forget()
        self.toggle_button.configure(text="▼")
        self.is_expanded = False

    def bind_toggle_callback(self, callback):
        if callback not in self.toggle_callbacks:
            self.toggle_callbacks.append(callback)