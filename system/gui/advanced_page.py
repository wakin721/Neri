from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QSlider, QCheckBox, QComboBox,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QSpacerItem, QMessageBox, QInputDialog, QGroupBox,
    QLineEdit, QApplication
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QRect, QObject
from PySide6.QtGui import QFont, QPalette, QPainter, QPainterPath, QColor, QPen, QBrush

import os
import platform
import re
import subprocess
import logging
import threading
import sys

from system.gui.ui_components import (
    CollapsiblePanel, Win11Colors, RoundedButton,
    ModernSlider, ModernComboBox, SwitchRow,
    ModernLineEdit, ModernGroupBox, ModernCheckBox
)
from system.utils import resource_path
from system.config import APP_VERSION, NORMAL_FONT

logger = logging.getLogger(__name__)

class ModelLoadWorker(QObject):
    finished = Signal(str, str)  # model_name, error_string

    def __init__(self, controller, model_path, model_name):
        super().__init__()
        self.controller = controller
        self.model_path = model_path
        self.model_name = model_name

    def run(self):
        """加载模型并更新控制器属性。"""
        try:
            self.controller.image_processor.load_model(self.model_path)
            self.controller.image_processor.model_path = self.model_path
            if hasattr(self.controller, 'model_var'):
                self.controller.model_var = self.model_name
            else:
                setattr(self.controller, 'model_var', self.model_name)
            self.finished.emit(self.model_name, None)
        except Exception as e:
            logger.error(f"自动加载模型失败: {e}")
            self.finished.emit(self.model_name, str(e))



class AdvancedPage(QWidget):
    """高级设置页面 - PySide6版本"""

    # 信号定义
    settings_changed = Signal()
    update_check_requested = Signal()
    theme_changed = Signal()
    params_help_requested = Signal()
    cache_clear_requested = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.is_dark_mode = False

        # 检测系统主题
        palette = self.palette()
        self.is_dark_mode = palette.color(QPalette.ColorRole.Window).lightness() < 128

        # 初始化变量
        self.iou_var = 0.3
        self.conf_var = 0.25
        self.use_fp16_var = self.controller.cuda_available if hasattr(controller, 'cuda_available') else False
        self.use_augment_var = True
        self.use_agnostic_nms_var = True
        self.vid_stride_var = 1  # 默认值为1 (处理每一帧)
        self.min_frame_ratio_var = 0.0  # 默认 0%
        self.theme_var = "自动"
        self.cache_size_var = "正在计算..."
        self.update_channel_var = "稳定版 (Release)"
        self.pytorch_version_var = "2.9.0 (CUDA 13.0)"
        self.package_var = ""
        self.version_constraint_var = ""
        self.pytorch_status_var = "未检查"
        self.model_status_var = ""
        self.package_status_var = ""
        self.auto_sort_var = False

        # 存储引用以便主题更新
        self.components_to_update = []

        # 设置Win11风格
        self._apply_win11_style()

        self._create_widgets()
        self._setup_connections()

        # 初始化数据
        QTimer.singleShot(100, self._post_init)

    def _apply_win11_style(self):
        """应用Win11风格"""
        palette = self.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        bg_color = Win11Colors.DARK_BACKGROUND if is_dark else Win11Colors.LIGHT_BACKGROUND
        text_color = Win11Colors.DARK_TEXT_PRIMARY if is_dark else Win11Colors.LIGHT_TEXT_PRIMARY

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color.name()};
                color: {text_color.name()};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
        """)

    def _create_widgets(self):
        """创建控件"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 创建滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll_area)

        # 内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # 模型参数设置
        model_params_group = ModernGroupBox("模型参数设置")
        content_layout.addWidget(model_params_group)
        self.model_params_layout = QVBoxLayout(model_params_group)
        self._create_model_params_content()

        # 视频检测设置
        video_settings_group = ModernGroupBox("视频检测设置")
        content_layout.addWidget(video_settings_group)
        self.video_settings_layout = QVBoxLayout(video_settings_group)
        self._create_video_settings_content()

        # 环境维护
        env_maintenance_group = ModernGroupBox("环境维护")
        content_layout.addWidget(env_maintenance_group)
        self.env_maintenance_layout = QVBoxLayout(env_maintenance_group)
        self._create_env_maintenance_content()

        # 软件设置
        software_settings_group = ModernGroupBox("软件设置")
        content_layout.addWidget(software_settings_group)
        self.software_settings_layout = QVBoxLayout(software_settings_group)
        self._create_software_settings_content()


    def _create_model_params_content(self):
        """创建模型参数设置内容"""
        # 主内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(12)

        # 模型管理面板
        self.model_panel = CollapsiblePanel(
            title="模型管理",
            subtitle="管理用于识别的模型",
            icon="🔧"
        )

        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        model_layout.setSpacing(15)

        # 选择模型
        select_model_label = QLabel("选择可用模型")
        select_model_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        model_layout.addWidget(select_model_label)

        self.model_combo = ModernComboBox()
        self.components_to_update.append(self.model_combo)
        model_layout.addWidget(self.model_combo)

        # 状态和按钮
        model_bottom_frame = QFrame()
        model_bottom_layout = QHBoxLayout(model_bottom_frame)

        self.model_status_label = QLabel(self.model_status_var)
        self.model_status_label.setFont(QFont("Segoe UI", 10))

        refresh_model_button = RoundedButton("刷新列表")
        refresh_model_button.setMinimumWidth(80)
        refresh_model_button.clicked.connect(self._refresh_model_list)

        model_bottom_layout.addWidget(self.model_status_label)
        model_bottom_layout.addStretch()
        model_bottom_layout.addWidget(refresh_model_button)

        model_layout.addWidget(model_bottom_frame)

        self.model_panel.add_content_widget(model_widget)
        content_layout.addWidget(self.model_panel)

        # 检测阈值设置面板
        self.threshold_panel = CollapsiblePanel(
            title="检测阈值设置",
            subtitle="调整目标检测的置信度和重叠度阈值",
            icon="🎯"
        )

        threshold_widget = QWidget()
        threshold_layout = QVBoxLayout(threshold_widget)
        threshold_layout.setSpacing(15)

        # IOU阈值
        iou_frame = QFrame()
        iou_layout = QVBoxLayout(iou_frame)

        iou_label_frame = QFrame()
        iou_label_layout = QHBoxLayout(iou_label_frame)
        iou_label_layout.setContentsMargins(0, 0, 0, 0)

        iou_title = QLabel("IOU阈值")
        iou_title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.iou_label = QLabel("0.30")
        self.iou_label.setFont(QFont("Segoe UI", 10))

        iou_label_layout.addWidget(iou_title)
        iou_label_layout.addStretch()
        iou_label_layout.addWidget(self.iou_label)

        self.iou_slider = ModernSlider()
        self.iou_slider.setRange(10, 90)
        self.iou_slider.setValue(int(self.iou_var * 100))
        self.components_to_update.append(self.iou_slider)

        iou_layout.addWidget(iou_label_frame)
        iou_layout.addWidget(self.iou_slider)
        threshold_layout.addWidget(iou_frame)

        # 置信度阈值
        conf_frame = QFrame()
        conf_layout = QVBoxLayout(conf_frame)

        conf_label_frame = QFrame()
        conf_label_layout = QHBoxLayout(conf_label_frame)
        conf_label_layout.setContentsMargins(0, 0, 0, 0)

        conf_title = QLabel("置信度阈值")
        conf_title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.conf_label = QLabel("0.25")
        self.conf_label.setFont(QFont("Segoe UI", 10))

        conf_label_layout.addWidget(conf_title)
        conf_label_layout.addStretch()
        conf_label_layout.addWidget(self.conf_label)

        self.conf_slider = ModernSlider()
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(int(self.conf_var * 100))
        self.components_to_update.append(self.conf_slider)

        conf_layout.addWidget(conf_label_frame)
        conf_layout.addWidget(self.conf_slider)
        threshold_layout.addWidget(conf_frame)

        self.threshold_panel.add_content_widget(threshold_widget)
        content_layout.addWidget(self.threshold_panel)

        # 模型加速选项面板
        self.accel_panel = CollapsiblePanel(
            title="模型加速选项",
            subtitle="控制推理速度与精度的平衡",
            icon="⚡"
        )

        accel_widget = QWidget()
        accel_layout = QVBoxLayout(accel_widget)

        # 替换为开关行
        self.fp16_switch_row = SwitchRow("使用FP16加速 (需要支持CUDA)", checked=self.use_fp16_var)
        self.fp16_switch_row.switch().setEnabled(
            self.controller.cuda_available if hasattr(self.controller, 'cuda_available') else False)
        self.fp16_switch_row.toggled.connect(self._on_setting_changed)
        self.components_to_update.append(self.fp16_switch_row)
        accel_layout.addWidget(self.fp16_switch_row)

        if not (hasattr(self.controller, 'cuda_available') and self.controller.cuda_available):
            cuda_warning = QLabel("未检测到CUDA，FP16加速已禁用")
            cuda_warning.setStyleSheet("color: #e74c3c; font-size: 12px;")
            accel_layout.addWidget(cuda_warning)

        self.accel_panel.add_content_widget(accel_widget)
        content_layout.addWidget(self.accel_panel)

        # 高级检测选项面板
        self.advanced_detect_panel = CollapsiblePanel(
            title="高级检测选项",
            subtitle="配置增强检测功能和特殊选项",
            icon="🔍"
        )

        advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(advanced_widget)

        # 替换为开关行
        self.augment_switch_row = SwitchRow("使用数据增强 (Test-Time Augmentation)", checked=self.use_augment_var)
        self.augment_switch_row.toggled.connect(self._on_setting_changed)
        self.components_to_update.append(self.augment_switch_row)
        advanced_layout.addWidget(self.augment_switch_row)

        self.agnostic_switch_row = SwitchRow("使用类别无关NMS (Class-Agnostic NMS)", checked=self.use_agnostic_nms_var)
        self.agnostic_switch_row.toggled.connect(self._on_setting_changed)
        self.components_to_update.append(self.agnostic_switch_row)
        advanced_layout.addWidget(self.agnostic_switch_row)

        self.advanced_detect_panel.add_content_widget(advanced_widget)
        content_layout.addWidget(self.advanced_detect_panel)

        # 底部按钮
        content_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)

        help_button = RoundedButton("参数说明")
        help_button.setMinimumWidth(120)
        help_button.clicked.connect(self.params_help_requested.emit)

        reset_button = RoundedButton("重置为默认值")
        reset_button.setMinimumWidth(120)
        reset_button.clicked.connect(self._reset_model_params)

        button_layout.addWidget(help_button)
        button_layout.addStretch()
        button_layout.addWidget(reset_button)

        content_layout.addWidget(button_frame)
        self.model_params_layout.addWidget(content_widget)

    def _create_video_settings_content(self):
        """创建视频检测设置内容"""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(12)

        # 跳帧处理面板
        self.frame_skip_panel = CollapsiblePanel(
            title="跳帧处理",
            subtitle="设置视频检测时的跳帧间隔 (vid_stride)",
            icon="⏩"
        )

        skip_widget = QWidget()
        skip_layout = QVBoxLayout(skip_widget)
        skip_layout.setSpacing(15)

        # 间隔设置
        stride_frame = QFrame()
        stride_layout = QVBoxLayout(stride_frame)

        stride_label_frame = QFrame()
        stride_label_layout = QHBoxLayout(stride_label_frame)
        stride_label_layout.setContentsMargins(0, 0, 0, 0)

        stride_title = QLabel("帧间隔 (Frame Stride)")
        stride_title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.stride_label = QLabel(str(self.vid_stride_var))
        self.stride_label.setFont(QFont("Segoe UI", 10))

        stride_label_layout.addWidget(stride_title)
        stride_label_layout.addStretch()
        stride_label_layout.addWidget(self.stride_label)

        # 创建滑块，范围设为 1-30
        self.stride_slider = ModernSlider()
        self.stride_slider.setRange(1, 30)
        self.stride_slider.setValue(self.vid_stride_var)
        self.components_to_update.append(self.stride_slider)

        # 连接信号
        self.stride_slider.valueChanged.connect(self._update_stride_label)
        self.stride_slider.valueChanged.connect(self._on_setting_changed)

        stride_layout.addWidget(stride_label_frame)
        stride_layout.addWidget(self.stride_slider)

        # 说明文本
        stride_info = QLabel(
            "值为 1 表示处理每一帧；值为 5 表示每 5 帧处理一次。增加此值可显著提高长视频的处理速度，但可能会降低时间精度。")
        stride_info.setStyleSheet("color: #888888; font-size: 12px;")
        stride_info.setWordWrap(True)
        stride_layout.addWidget(stride_info)

        skip_layout.addWidget(stride_frame)

        self.frame_skip_panel.add_content_widget(skip_widget)
        content_layout.addWidget(self.frame_skip_panel)

        self.video_settings_layout.addWidget(content_widget)

        self.frame_ratio_panel = CollapsiblePanel(
            title="检测过滤",
            subtitle="设置检测到的最低帧数比例",
            icon="🛡️"
        )

        ratio_widget = QWidget()
        ratio_layout = QVBoxLayout(ratio_widget)
        ratio_layout.setSpacing(15)

        # 比例设置
        ratio_frame = QFrame()
        ratio_frame_layout = QVBoxLayout(ratio_frame)

        ratio_label_frame = QFrame()
        ratio_label_layout = QHBoxLayout(ratio_label_frame)
        ratio_label_layout.setContentsMargins(0, 0, 0, 0)

        ratio_title = QLabel("最低帧数比例 (Minimum Frame Ratio)")
        ratio_title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.ratio_label = QLabel(f"{int(self.min_frame_ratio_var * 100)}%")
        self.ratio_label.setFont(QFont("Segoe UI", 10))

        ratio_label_layout.addWidget(ratio_title)
        ratio_label_layout.addStretch()
        ratio_label_layout.addWidget(self.ratio_label)

        # 创建滑块，范围 0-30 (%)
        self.ratio_slider = ModernSlider()
        self.ratio_slider.setRange(0, 30)
        self.ratio_slider.setValue(int(self.min_frame_ratio_var * 100))
        self.ratio_label.setText(f"{int(self.min_frame_ratio_var * 100)}%")
        self.components_to_update.append(self.ratio_slider)

        # 连接信号
        self.ratio_slider.valueChanged.connect(self._update_ratio_label)
        self.ratio_slider.valueChanged.connect(self._on_setting_changed)

        ratio_frame_layout.addWidget(ratio_label_frame)
        ratio_frame_layout.addWidget(self.ratio_slider)

        # 说明文本
        ratio_info = QLabel(
            "如果某个目标（Track ID）在视频中出现的总帧数占视频总帧数的比例低于此值，"
            "则该目标将被视为误检或无效目标，不会在结果中显示。")
        ratio_info.setStyleSheet("color: #888888; font-size: 12px;")
        ratio_info.setWordWrap(True)
        ratio_frame_layout.addWidget(ratio_info)

        ratio_layout.addWidget(ratio_frame)
        self.frame_ratio_panel.add_content_widget(ratio_widget)
        content_layout.addWidget(self.frame_ratio_panel)

    def _update_stride_label(self, value):
        """更新跳帧标签"""
        self.vid_stride_var = value
        self.stride_label.setText(str(value))

    def _update_ratio_label(self, value):
        """更新比例标签"""
        self.min_frame_ratio_var = value / 100.0
        self.ratio_label.setText(f"{value}%")

    def _create_env_maintenance_content(self):
        """创建环境维护标签页内容"""
        # 主内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(12)

        # PyTorch安装面板
        self.pytorch_panel = CollapsiblePanel(
            title="安装 PyTorch",
            subtitle="安装或修复 PyTorch",
            icon="📦"
        )

        pytorch_widget = QWidget()
        pytorch_layout = QVBoxLayout(pytorch_widget)
        pytorch_layout.setSpacing(15)

        # 版本选择
        version_label = QLabel("选择版本")
        version_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        pytorch_layout.addWidget(version_label)

        self.pytorch_version_combo = ModernComboBox()
        versions = [
            "2.9.1 (CUDA 13.0)",
            "2.9.1 (CUDA 12.8)",
            "2.9.1 (CUDA 12.6)",
            "2.9.1 (CPU Only)",
            "2.7.1 (CUDA 12.8)",
            "2.7.1 (CUDA 12.6)",
            "2.7.1 (CUDA 11.8)",
            "2.7.1 (CPU Only)",
        ]
        self.pytorch_version_combo.addItems(versions)
        self.pytorch_version_combo.setCurrentText(self.pytorch_version_var)
        self.components_to_update.append(self.pytorch_version_combo)
        pytorch_layout.addWidget(self.pytorch_version_combo)

        # 说明文本
        warning_label = QLabel("将先卸载现有的torch、torchvision、torchaudio模块再重新安装")
        warning_label.setStyleSheet("color: #666666; font-size: 12px;")
        warning_label.setWordWrap(True)
        pytorch_layout.addWidget(warning_label)

        # 状态和安装按钮
        pytorch_bottom_frame = QFrame()
        pytorch_bottom_layout = QHBoxLayout(pytorch_bottom_frame)

        self.pytorch_status_label = QLabel(self.pytorch_status_var)
        self.pytorch_status_label.setFont(QFont("Segoe UI", 10))

        self.install_pytorch_button = RoundedButton("安装")
        self.install_pytorch_button.setMinimumWidth(80)
        self.install_pytorch_button.clicked.connect(self._install_pytorch)

        pytorch_bottom_layout.addWidget(self.pytorch_status_label)
        pytorch_bottom_layout.addStretch()
        pytorch_bottom_layout.addWidget(self.install_pytorch_button)

        pytorch_layout.addWidget(pytorch_bottom_frame)

        self.pytorch_panel.add_content_widget(pytorch_widget)
        content_layout.addWidget(self.pytorch_panel)

        # Python包管理面板
        self.python_panel = CollapsiblePanel(
            title="重装单个 Python 组件",
            subtitle="重新安装单个 Pip 软件包",
            icon="🐍"
        )

        python_widget = QWidget()
        python_layout = QVBoxLayout(python_widget)
        python_layout.setSpacing(15)

        # 包名称
        package_label = QLabel("输入包名称")
        package_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        python_layout.addWidget(package_label)

        self.package_edit = ModernLineEdit("例如: numpy")
        self.components_to_update.append(self.package_edit)
        python_layout.addWidget(self.package_edit)

        # 版本约束
        version_constraint_label = QLabel("版本约束 (可选)")
        version_constraint_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        python_layout.addWidget(version_constraint_label)

        self.version_constraint_edit = ModernLineEdit("例如: >=1.0.0")
        self.components_to_update.append(self.version_constraint_edit)
        python_layout.addWidget(self.version_constraint_edit)

        version_example = QLabel("示例: ==1.0.0, >=2.0.0, <3.0.0")
        version_example.setStyleSheet("color: #888888; font-size: 12px;")
        python_layout.addWidget(version_example)

        # 状态和安装按钮
        python_bottom_frame = QFrame()
        python_bottom_layout = QHBoxLayout(python_bottom_frame)

        self.package_status_label = QLabel(self.package_status_var)
        self.package_status_label.setFont(QFont("Segoe UI", 10))

        self.install_package_button = RoundedButton("安装")
        self.install_package_button.setMinimumWidth(80)
        self.install_package_button.clicked.connect(self._install_python_package)

        python_bottom_layout.addWidget(self.package_status_label)
        python_bottom_layout.addStretch()
        python_bottom_layout.addWidget(self.install_package_button)

        python_layout.addWidget(python_bottom_frame)

        self.python_panel.add_content_widget(python_widget)
        content_layout.addWidget(self.python_panel)

        content_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.env_maintenance_layout.addWidget(content_widget)

    def _create_software_settings_content(self):
        """创建软件设置标签页内容"""
        # 主内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(12)

        # 快速标记设置面板
        self.quick_mark_panel = CollapsiblePanel(
            title="快速标记设置",
            subtitle="手动增减、更改快速标记",
            icon="🏷️"
        )

        quick_mark_widget = QWidget()
        quick_mark_layout = QVBoxLayout(quick_mark_widget)

        # 自动排序开关
        self.auto_sort_switch_row = SwitchRow("自动排序", checked=self.auto_sort_var)
        self.auto_sort_switch_row.toggled.connect(self._on_auto_sort_changed)  # 确保这一行存在
        self.components_to_update.append(self.auto_sort_switch_row)
        quick_mark_layout.addWidget(self.auto_sort_switch_row)

        # 清空排序数据按钮 - 新的一行，靠右对齐
        reset_mark_button_frame = QFrame()
        reset_mark_button_layout = QHBoxLayout(reset_mark_button_frame)
        reset_mark_button_layout.setContentsMargins(0, 4, 0, 8)  # 上边距小一些，下边距大一些

        reset_mark_button = RoundedButton("清空排序数据")
        reset_mark_button.setMinimumWidth(120)
        reset_mark_button.clicked.connect(self._reset_quick_mark_data)

        reset_mark_button_layout.addStretch()  # 添加弹性空间，将按钮推到右边
        reset_mark_button_layout.addWidget(reset_mark_button)

        quick_mark_layout.addWidget(reset_mark_button_frame)

        # 物种列表标题
        species_header_frame = QFrame()
        species_header_layout = QHBoxLayout(species_header_frame)

        order_header = QLabel("排列序号")
        order_header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        order_header.setFixedWidth(80)

        name_header = QLabel("物种名称")
        name_header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))

        species_header_layout.addWidget(order_header)
        species_header_layout.addWidget(name_header, 1)
        species_header_layout.addWidget(QLabel("操作"))  # 为删除按钮预留空间

        quick_mark_layout.addWidget(species_header_frame)

        # 物种列表容器
        self.species_list_frame = QFrame()
        self.species_list_layout = QVBoxLayout(self.species_list_frame)
        self.species_list_layout.setContentsMargins(0, 0, 0, 0)
        quick_mark_layout.addWidget(self.species_list_frame)

        # 底部按钮区域（新增和保存按钮）
        quick_mark_buttons_frame = QFrame()
        quick_mark_buttons_layout = QHBoxLayout(quick_mark_buttons_frame)

        add_species_button = RoundedButton("新增")
        add_species_button.setMinimumWidth(80)
        add_species_button.clicked.connect(self._add_new_quick_mark_row)

        save_species_button = RoundedButton("保存更改")
        save_species_button.setMinimumWidth(100)
        save_species_button.clicked.connect(self.save_quick_mark_settings)

        quick_mark_buttons_layout.addStretch()
        quick_mark_buttons_layout.addWidget(add_species_button)
        quick_mark_buttons_layout.addWidget(save_species_button)

        quick_mark_layout.addWidget(quick_mark_buttons_frame)

        self.quick_mark_panel.add_content_widget(quick_mark_widget)
        content_layout.addWidget(self.quick_mark_panel)

        # 导出设置面板
        self.export_settings_panel = CollapsiblePanel(
            title="导出设置",
            subtitle="自定义导出表格中的列",
            icon="📤"
        )
        export_widget = QWidget()
        # 使用网格布局以更好地对齐多列复选框
        export_layout = QGridLayout(export_widget)
        export_layout.setSpacing(10)

        # 创建“全选”复选框
        self.select_all_checkbox = ModernCheckBox("全选/全不选")
        self.select_all_checkbox.setChecked(True)  # 默认全选
        self.select_all_checkbox.stateChanged.connect(self._toggle_all_columns)
        # 将其放置在网格布局的第一行，并让它跨越所有列
        export_layout.addWidget(self.select_all_checkbox, 0, 0, 1, -1)

        self.all_export_columns = [
                '文件名', '格式', '拍摄日期', '拍摄时间', '工作天数',
                '物种名称', '学名', '目名', '目拉丁名', '科名', '科拉丁名', '属名', '属拉丁名',
                '物种类型', '物种数量', '最低置信度', '独立探测首只', '备注']

        self.export_checkboxes = {}
        columns_per_row = 3  # 每行显示3个选项

        for i, col_name in enumerate(self.all_export_columns):
            checkbox = ModernCheckBox(col_name)
            checkbox.setChecked(True)  # 默认全部选中
            checkbox.stateChanged.connect(self._update_select_all_state)
            self.export_checkboxes[col_name] = checkbox
            row = i // columns_per_row + 1
            col = i % columns_per_row
            export_layout.addWidget(checkbox, row, col)

        self.export_settings_panel.add_content_widget(export_widget)
        content_layout.addWidget(self.export_settings_panel)

        # 主题设置面板
        self.theme_panel = CollapsiblePanel(
            title="深色模式",
            subtitle="选择应用的主题模式",
            icon="🎨"
        )

        theme_widget = QWidget()
        theme_layout = QVBoxLayout(theme_widget)

        self.theme_combo = ModernComboBox()
        self.theme_combo.addItems(["浅色", "深色", "自动"])
        self.theme_combo.setCurrentText(self.theme_var)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.components_to_update.append(self.theme_combo)
        theme_layout.addWidget(self.theme_combo)

        self.theme_panel.add_content_widget(theme_widget)
        content_layout.addWidget(self.theme_panel)

        # 缓存管理面板
        self.cache_panel = CollapsiblePanel(
            title="缓存管理",
            subtitle="清除应用程序生成的临时文件",
            icon="🗑️"
        )

        cache_widget = QWidget()
        cache_layout = QVBoxLayout(cache_widget)

        self.cache_size_label = QLabel(self.cache_size_var)
        self.cache_size_label.setFont(QFont("Segoe UI", 10))
        cache_layout.addWidget(self.cache_size_label)

        cache_buttons_frame = QFrame()
        cache_buttons_layout = QHBoxLayout(cache_buttons_frame)

        refresh_cache_button = RoundedButton("刷新大小")
        refresh_cache_button.setMinimumWidth(80)
        refresh_cache_button.clicked.connect(self.update_cache_size)

        clear_cache_button = RoundedButton("清除缓存")
        clear_cache_button.setMinimumWidth(80)
        clear_cache_button.clicked.connect(self._clear_image_cache_with_refresh)

        cache_buttons_layout.addStretch()
        cache_buttons_layout.addWidget(refresh_cache_button)
        cache_buttons_layout.addWidget(clear_cache_button)

        cache_layout.addWidget(cache_buttons_frame)

        self.cache_panel.add_content_widget(cache_widget)
        content_layout.addWidget(self.cache_panel)

        # 软件更新面板
        self.update_panel = CollapsiblePanel(
            title="软件更新",
            subtitle="检查、更新和管理软件版本",
            icon="🔄"
        )

        update_widget = QWidget()
        update_layout = QVBoxLayout(update_widget)
        update_layout.setSpacing(15)

        # 更新通道选择
        channel_label = QLabel("选择更新通道")
        channel_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        update_layout.addWidget(channel_label)

        self.update_channel_combo = ModernComboBox()
        self.update_channel_combo.addItems(["稳定版 (Release)", "预览版 (Preview)"])
        # 确保设置初始值
        self.update_channel_combo.setCurrentText(self.update_channel_var)
        self.components_to_update.append(self.update_channel_combo)
        update_layout.addWidget(self.update_channel_combo)

        # 状态和检查按钮
        update_bottom_frame = QFrame()
        update_bottom_layout = QHBoxLayout(update_bottom_frame)

        self.update_status_label = QLabel(f"当前版本: {APP_VERSION}")
        self.update_status_label.setFont(QFont("Segoe UI", 10))

        self.check_update_button = RoundedButton("检查更新")
        self.check_update_button.setMinimumWidth(100)
        self.check_update_button.clicked.connect(self.update_check_requested.emit)

        update_bottom_layout.addWidget(self.update_status_label)
        update_bottom_layout.addStretch()
        update_bottom_layout.addWidget(self.check_update_button)

        update_layout.addWidget(update_bottom_frame)

        self.update_panel.add_content_widget(update_widget)
        content_layout.addWidget(self.update_panel)

        content_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.software_settings_layout.addWidget(content_widget)

    def _setup_connections(self):
        """设置信号连接"""
        # 滑块连接 - 确保同时连接标签更新和设置保存
        self.iou_slider.valueChanged.connect(self._update_iou_label)
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        self.auto_sort_switch_row.toggled.connect(self._on_auto_sort_changed)

        # 确保滑块变化时立即触发设置保存
        self.iou_slider.valueChanged.connect(self._on_setting_changed)
        self.conf_slider.valueChanged.connect(self._on_setting_changed)

        # 复选框连接
        self.fp16_switch_row.toggled.connect(self._on_setting_changed)
        self.augment_switch_row.toggled.connect(self._on_setting_changed)
        self.agnostic_switch_row.toggled.connect(self._on_setting_changed)
        self.auto_sort_switch_row.toggled.connect(self._on_auto_sort_changed)

        # 下拉框连接
        self.pytorch_version_combo.currentTextChanged.connect(self._on_pytorch_version_changed)
        self.update_channel_combo.currentTextChanged.connect(self._on_update_channel_changed)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)

        # 输入框连接
        self.package_edit.textChanged.connect(self._on_package_changed)
        self.version_constraint_edit.textChanged.connect(self._on_version_constraint_changed)

        self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)

    def _on_model_selection_changed(self, model_name):
        """处理模型选择变化。"""
        if not model_name:
            return

        current_model = ""
        if hasattr(self.controller, 'image_processor') and hasattr(self.controller.image_processor, 'model_path'):
            if self.controller.image_processor.model_path:
                current_model = os.path.basename(self.controller.image_processor.model_path)

        if model_name == current_model:
            logger.info(f"模型 {model_name} 已经在使用中")
            self.model_status_label.setText(f"当前使用: {model_name}")
            return

        model_path = resource_path(os.path.join("res", "model", model_name))
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            self.model_status_label.setText("模型文件不存在")
            return

        self.model_status_label.setText("正在加载...")
        if hasattr(self.controller, 'start_page'):
            self.controller.start_page.set_processing_enabled(False)

        # 使用QThread和Worker模式
        self.thread = QThread()
        self.worker = ModelLoadWorker(self.controller, model_path, model_name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_model_loaded)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_model_loaded(self, model_name, error_string):
        """处理模型加载完成的结果。"""
        if error_string:
            self.model_status_label.setText(f"加载失败: {error_string}")
        else:
            self.model_status_label.setText(f"已应用: {model_name}")
            self.model_combo.setToolTip(f"当前使用的模型: {model_name}")
            self._on_setting_changed()  # 加载成功后保存设置
            logger.info(f"模型自动加载成功: {model_name}")

        if hasattr(self.controller, 'start_page'):
            self.controller.start_page.set_processing_enabled(True)

    def _save_settings_immediately(self):
        """立即保存设置到JSON文件"""
        try:
            # 发出设置变更信号
            self.settings_changed.emit()

            # 直接调用设置管理器保存设置
            if hasattr(self.controller, 'settings_manager'):
                current_settings = self.get_settings()
                self.controller.settings_manager.save_settings(current_settings)
                logger.info("设置已立即保存到JSON文件")

            # 如果controller有save_settings方法，也调用它
            if hasattr(self.controller, 'save_settings'):
                self.controller.save_settings()

        except Exception as e:
            logger.error(f"立即保存设置失败: {e}")

    def _post_init(self):
        """后期初始化"""
        self._check_pytorch_status()
        self._refresh_model_list()
        self.load_quick_mark_settings()
        QTimer.singleShot(100, self.update_cache_size)

    def _update_iou_label(self, value):
        """更新IOU标签"""
        rounded_value = round(float(value) / 100, 2)
        self.iou_var = rounded_value
        self.iou_label.setText(f"{rounded_value:.2f}")
        self._on_setting_changed()

    def _update_conf_label(self, value):
        """更新置信度标签"""
        rounded_value = round(float(value) / 100, 2)
        self.conf_var = rounded_value
        self.conf_label.setText(f"{rounded_value:.2f}")
        self._on_setting_changed()

    def _reset_model_params(self):
        """重置模型参数"""
        self.iou_var = 0.3
        self.conf_var = 0.25
        self.use_fp16_var = self.controller.cuda_available if hasattr(self.controller, 'cuda_available') else False
        self.use_augment_var = True
        self.use_agnostic_nms_var = True

        self.iou_slider.setValue(int(self.iou_var * 100))
        self.conf_slider.setValue(int(self.conf_var * 100))
        # 替换为开关行的设置方法
        self.fp16_switch_row.setChecked(self.use_fp16_var)
        self.augment_switch_row.setChecked(self.use_augment_var)
        self.agnostic_switch_row.setChecked(self.use_agnostic_nms_var)

        self._update_iou_label(int(self.iou_var * 100))
        self._update_conf_label(int(self.conf_var * 100))

        self.min_frame_ratio_var = 0.0
        self.ratio_slider.setValue(0)
        self.ratio_label.setText("0%")

        QMessageBox.information(self, "参数重置", "已重置所有参数到默认值")

    def _check_pytorch_status(self):
        """检查PyTorch安装状态"""
        try:
            import torch
            version = torch.__version__
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            self.pytorch_status_var = f"已安装 v{version} ({device})"
            self.pytorch_status_label.setText(self.pytorch_status_var)
        except ImportError:
            self.pytorch_status_var = "未安装"
            self.pytorch_status_label.setText(self.pytorch_status_var)
        except Exception as e:
            self.pytorch_status_var = f"检查失败: {str(e)}"
            self.pytorch_status_label.setText(self.pytorch_status_var)

    def _install_pytorch(self):
        """安装PyTorch"""
        version = self.pytorch_version_combo.currentText()
        if not version:
            QMessageBox.critical(self, "错误", "请选择PyTorch版本")
            return

        message = f"将安装 PyTorch {version}。\n\n此操作会强制卸载任何现有版本，是否继续？"

        reply = QMessageBox.question(
            self, "确认安装", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        is_cuda = "CPU" not in version
        cuda_version = None
        if is_cuda:
            cuda_match = re.search(r"CUDA (\d+\.\d+)", version)
            if cuda_match:
                cuda_version = cuda_match.group(1)

        pytorch_match = re.search(r"(\d+\.\d+\.\d+)", version)
        if pytorch_match:
            pytorch_version = pytorch_match.group(1)
        else:
            QMessageBox.critical(self, "错误", "无法解析PyTorch版本")
            return

        self.install_pytorch_button.setEnabled(False)
        self.pytorch_status_label.setText("准备安装...")

        # 启动安装线程
        threading.Thread(
            target=self._run_pytorch_install,
            args=(pytorch_version, cuda_version),
            daemon=True
        ).start()

    def _run_pytorch_install(self, pytorch_version, cuda_version=None):
        """运行PyTorch安装"""
        try:
            QTimer.singleShot(0, lambda: self.pytorch_status_label.setText("正在启动安装..."))

            pip_command_prefix = self._get_python_command_prefix()

            if cuda_version:
                cuda_str_map = {"11.8": "cu118", "12.1": "cu121", "12.6": "cu126", "12.8": "cu128"}
                cuda_str = cuda_str_map.get(cuda_version, f"cu{cuda_version.replace('.', '')}")
                install_cmd = f"{pip_command_prefix} install torch=={pytorch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_str}"
            else:
                install_cmd = f"{pip_command_prefix} install torch=={pytorch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

            command = (
                f"echo 正在卸载现有PyTorch... && "
                f"{pip_command_prefix} uninstall -y torch torchvision torchaudio && "
                f"echo 卸载完成，开始安装新版本... && "
                f"{install_cmd} && "
                f"echo. && echo 安装完成！窗口将在5秒后自动关闭... && "
                f"timeout /t 5"
            )

            QTimer.singleShot(0, lambda: self.pytorch_status_label.setText("安装已启动，请查看命令行窗口"))

            if platform.system() == "Windows":
                subprocess.Popen(f"start cmd /C \"{command}\"", shell=True)
            else:
                # Unix系统处理
                if platform.system() == "Darwin":
                    mac_command = command.replace("timeout /t 5", "sleep 5")
                    subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "{mac_command}"'])
                else:
                    linux_command = command.replace("timeout /t 5", "sleep 5")
                    for terminal in ["gnome-terminal", "konsole", "xterm"]:
                        try:
                            if terminal == "gnome-terminal":
                                subprocess.Popen([terminal, "--", "bash", "-c", f"{linux_command}"])
                            elif terminal == "konsole":
                                subprocess.Popen([terminal, "-e", f"bash -c '{linux_command}'"])
                            elif terminal == "xterm":
                                subprocess.Popen([terminal, "-e", f"bash -c '{linux_command}'"])
                            break
                        except FileNotFoundError:
                            continue

            QTimer.singleShot(2000, lambda: self.install_pytorch_button.setEnabled(True))
            QTimer.singleShot(2000, lambda: QMessageBox.information(
                self, "安装已启动",
                "PyTorch安装已在命令行窗口中启动，\n"
                "请查看命令行窗口了解安装进度，\n"
                "安装完成后，重启程序以使更改生效。\n"
                "命令执行完成后窗口将在5秒后自动关闭。"
            ))

            version_text = f"{pytorch_version} {'(CUDA ' + cuda_version + ')' if cuda_version else '(CPU)'}"
            QTimer.singleShot(3000, lambda: self.pytorch_status_label.setText(f"已完成安装 PyTorch {version_text}"))

        except Exception as e:
            logger.error(f"安装PyTorch出错: {e}")
            QTimer.singleShot(0, lambda: self.pytorch_status_label.setText(f"安装失败: {str(e)}"))
            QTimer.singleShot(0, lambda: self.install_pytorch_button.setEnabled(True))
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, "安装错误", f"安装PyTorch失败：\n{str(e)}"))

    def _get_python_command_prefix(self):
        """获取用于调用pip的python.exe命令前缀"""
        program_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        python_exe_path = os.path.join(program_root_dir, "toolkit", "python.exe")

        if not os.path.exists(python_exe_path):
            print(f"警告: 未在 {program_root_dir}\\toolkit 找到 python.exe, 将回退到默认python。")
            return f'"{sys.executable}" -m pip'
        else:
            return f'"{python_exe_path}" -m pip'

    def _refresh_model_list(self):
        """刷新可用模型列表"""
        model_dir = os.path.join(resource_path("res") ,"model")
        try:
            # 保存当前选择
            current_selection = self.model_combo.currentText()
            current_model_path = None
            current_model = "未指定"

            if hasattr(self.controller, 'image_processor') and hasattr(self.controller.image_processor, 'model_path'):
                current_model_path = self.controller.image_processor.model_path
                current_model = os.path.basename(current_model_path) if current_model_path else "未指定"

            # 暂时断开信号连接，避免在刷新时触发自动加载
            self.model_combo.currentTextChanged.disconnect()

            self.model_combo.clear()
            self.model_combo.setToolTip(f"当前使用的模型: {current_model}")

            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.pt')]
                if model_files:
                    model_files.sort()
                    self.model_combo.addItems(model_files)

                    # 尝试恢复之前的选择或当前正在使用的模型
                    if current_model in model_files:
                        self.model_combo.setCurrentText(current_model)
                    elif current_selection and current_selection in model_files:
                        self.model_combo.setCurrentText(current_selection)

                    # 更新状态标签
                    if current_model in model_files:
                        self.model_status_label.setText(f"当前使用: {current_model}")
                    else:
                        self.model_status_label.setText(f"找到 {len(model_files)} 个模型文件")
                else:
                    self.model_status_label.setText("未找到任何模型文件")
            else:
                self.model_status_label.setText("模型目录不存在")

            # 重新连接信号
            self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)

        except Exception as e:
            logger.error(f"刷新模型列表失败: {e}")
            self.model_status_label.setText(f"刷新失败: {str(e)}")
            # 确保重新连接信号
            try:
                self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)
            except:
                pass

    def _install_python_package(self):
        """安装Python包"""
        package = self.package_edit.text().strip()
        if not package:
            QMessageBox.critical(self, "错误", "请输入包名称")
            return

        version_constraint = self.version_constraint_edit.text().strip()
        package_spec = f"{package}{version_constraint}" if version_constraint else package

        reply = QMessageBox.question(
            self, "确认安装", f"将安装 {package_spec}\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.package_status_label.setText("准备安装...")

        threading.Thread(
            target=self._run_pip_install,
            args=(package_spec,),
            daemon=True
        ).start()

    def _run_pip_install(self, package_spec):
        """使用弹出命令行窗口安装Python包"""
        try:
            QTimer.singleShot(0, lambda: self.package_status_label.setText("正在启动安装..."))

            pip_command_prefix = self._get_python_command_prefix()

            install_cmd = f"{pip_command_prefix} install {package_spec}"
            command = (
                f"echo 正在安装 {package_spec}... && "
                f"{install_cmd} && "
                f"echo. && echo 安装完成！窗口将在5秒后自动关闭... && "
                f"timeout /t 5"
            )

            QTimer.singleShot(0, lambda: self.package_status_label.setText("安装已启动，请查看命令行窗口"))

            if platform.system() == "Windows":
                subprocess.Popen(f"start cmd /C \"{command}\"", shell=True)
            else:
                if platform.system() == "Darwin":
                    mac_command = command.replace("timeout /t 5", "sleep 5")
                    subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "{mac_command}"'])
                else:
                    linux_command = command.replace("timeout /t 5", "sleep 5")
                    for terminal in ["gnome-terminal", "konsole", "xterm"]:
                        try:
                            if terminal == "gnome-terminal":
                                subprocess.Popen([terminal, "--", "bash", "-c", f"{linux_command}"])
                            elif terminal == "konsole":
                                subprocess.Popen([terminal, "-e", f"bash -c '{linux_command}'"])
                            elif terminal == "xterm":
                                subprocess.Popen([terminal, "-e", f"bash -c '{linux_command}'"])
                            break
                        except FileNotFoundError:
                            continue

            QTimer.singleShot(3000, lambda: self.package_status_label.setText(f"已完成安装 {package_spec}"))

        except Exception as e:
            logger.error(f"安装Python包出错: {e}")
            QTimer.singleShot(0, lambda: self.package_status_label.setText(f"安装失败: {str(e)}"))
            QTimer.singleShot(0, lambda: QMessageBox.critical(
                self, "安装错误", f"安装Python包失败：\n{str(e)}"
            ))

    def update_cache_size(self):
        """计算并更新缓存大小显示"""
        # 立即显示"正在计算..."状态
        self.cache_size_label.setText("缓存大小: 正在计算...")

        # 使用QTimer延迟执行计算，让UI有时间更新显示
        QTimer.singleShot(250, self._calculate_cache_size_async)

    def _calculate_cache_size_async(self):
        """异步计算缓存大小"""

        # 创建缓存大小计算工作线程
        class CacheSizeWorker(QObject):
            finished = Signal(int, str, str)  # size, error_message, size_str

            def __init__(self, controller):
                super().__init__()
                self.controller = controller

            def run(self):
                try:
                    # 获取缓存目录
                    if hasattr(self.controller, 'settings_manager'):
                        cache_dir = os.path.join(self.controller.settings_manager.base_dir, "temp", "photo")
                    else:
                        cache_dir = os.path.join(os.path.expanduser("~"), ".neri", "temp", "photo")

                    logger.info(f"计算缓存目录大小: {cache_dir}")

                    # 检查目录是否存在
                    if not os.path.exists(cache_dir):
                        self.finished.emit(0, None, "0 Bytes (目录不存在)")
                        return

                    total_size = 0
                    file_count = 0

                    # 遍历目录计算大小
                    for dirpath, dirnames, filenames in os.walk(cache_dir):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            try:
                                if os.path.isfile(file_path) and not os.path.islink(file_path):
                                    file_size = os.path.getsize(file_path)
                                    total_size += file_size
                                    file_count += 1
                            except (OSError, IOError) as e:
                                logger.warning(f"无法获取文件大小 {file_path}: {e}")
                                continue

                    # 格式化文件大小
                    if total_size < 1024:
                        size_str = f"{total_size} Bytes"
                    elif total_size < 1024 ** 2:
                        size_str = f"{total_size / 1024:.2f} KB"
                    elif total_size < 1024 ** 3:
                        size_str = f"{total_size / 1024 ** 2:.2f} MB"
                    else:
                        size_str = f"{total_size / 1024 ** 3:.2f} GB"

                    logger.info(f"缓存大小: {size_str} ({file_count} 个文件)")
                    self.finished.emit(total_size, None, size_str)

                except Exception as e:
                    logger.error(f"计算缓存大小失败: {e}")
                    self.finished.emit(0, str(e), f"计算失败 ({str(e)})")

        # 创建工作线程
        self.cache_thread = QThread()
        self.cache_worker = CacheSizeWorker(self.controller)
        self.cache_worker.moveToThread(self.cache_thread)

        # 连接信号
        self.cache_thread.started.connect(self.cache_worker.run)
        self.cache_worker.finished.connect(self._on_cache_size_calculated)
        self.cache_worker.finished.connect(self.cache_thread.quit)
        self.cache_worker.finished.connect(self.cache_worker.deleteLater)
        self.cache_thread.finished.connect(self.cache_thread.deleteLater)

        # 启动线程
        self.cache_thread.start()

    def _on_cache_size_calculated(self, total_size, error_message, size_str):
        """处理缓存大小计算结果"""
        try:
            if error_message:
                self.cache_size_label.setText(f"缓存大小: {size_str}")
            else:
                self.cache_size_label.setText(f"缓存大小: {size_str}")

            logger.info(f"缓存大小更新完成: {size_str}")

        except Exception as e:
            logger.error(f"更新缓存大小显示失败: {e}")
            self.cache_size_label.setText("缓存大小: 更新失败")

    def _clear_image_cache_with_refresh(self):
        """清除图像缓存并刷新大小"""
        self.cache_clear_requested.emit()
        QTimer.singleShot(500, self.update_cache_size)

    def load_quick_mark_settings(self):
        """加载快速标记设置并显示在UI中"""
        # 清空现有控件
        while self.species_list_layout.count():
            child = self.species_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.quick_marks_entries = {}

        if hasattr(self.controller, 'settings_manager'):
            quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
        else:
            quick_marks_data = {"list": [], "auto": False}

            # 临时阻塞信号，防止在程序设置开关状态时意外触发 _on_auto_sort_changed 方法
            self.auto_sort_switch_row.blockSignals(True)
            self.auto_sort_switch_row.setChecked(quick_marks_data.get("auto", False))
            self.auto_sort_switch_row.blockSignals(False)

        species_list_to_display = []
        if self.auto_sort_switch_row.isChecked():
            species_list_to_display = self.update_auto_sorted_list()
        else:
            species_list_to_display = quick_marks_data.get("list", [])

        if species_list_to_display:
            for i, species in enumerate(species_list_to_display):
                self._create_species_row(i + 1, species)

    def _create_species_row(self, order, species_name):
        """创建物种行（修改后）"""
        row_frame = QFrame()
        row_layout = QHBoxLayout(row_frame)
        row_layout.setContentsMargins(0, 2, 0, 2)

        # 排序号
        order_edit = ModernLineEdit()
        order_edit.setText(str(order))
        order_edit.setReadOnly(True)
        order_edit.setFixedWidth(80)
        row_layout.addWidget(order_edit)

        # 物种名称
        species_edit = ModernLineEdit()
        species_edit.setText(species_name)
        self.components_to_update.append(species_edit)
        row_layout.addWidget(species_edit, 1)

        # 删除按钮
        delete_button = RoundedButton("删除")
        delete_button.setMinimumWidth(60)
        # 修改连接：让按钮直接引用它所在的行(row_frame)，以便删除
        delete_button.clicked.connect(lambda: self._delete_species_row(row_frame))
        row_layout.addWidget(delete_button)

        self.species_list_layout.addWidget(row_frame)
        # 注意：quick_marks_entries的逻辑在保存时处理，这里不再需要对新行进行特殊处理
        if species_name: # 只为已存在的物种添加条目
            self.quick_marks_entries[species_name] = (species_edit, str(order))

    def _add_new_quick_mark_row(self):
        """新增物种行（修改后）"""
        # 根据当前UI中的行数计算新序号，确保序号总是递增的
        new_order = self.species_list_layout.count() + 1
        self._create_species_row(new_order, "")

    def _delete_species_row(self, row_widget):
        """删除物种行（修改后）"""
        # 从布局中移除并删除该行控件
        self.species_list_layout.removeWidget(row_widget)
        row_widget.deleteLater()

        # 延迟执行，确保控件删除后，再重新为所有剩余行排序
        QTimer.singleShot(0, self._update_quick_mark_order)

    def _update_quick_mark_order(self):
        """更新所有快速标记行的显示序号"""
        for i in range(self.species_list_layout.count()):
            row_widget = self.species_list_layout.itemAt(i).widget()
            if row_widget:
                # 找到序号输入框（通常是第一个 ModernLineEdit）
                order_edit = row_widget.findChildren(ModernLineEdit)[0]
                order_edit.setText(str(i + 1))

    def update_auto_sorted_list(self):
        """根据使用次数对物种进行排序"""
        if hasattr(self.controller, 'settings_manager'):
            quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
        else:
            return []

        species_counts = {k: v for k, v in quick_marks_data.items() if k not in ["list", "list_auto", "auto"]}
        sorted_species = sorted(species_counts.items(), key=lambda item: item[1], reverse=True)
        num_to_take = len(quick_marks_data.get("list", []))
        list_auto = [species for species, count in sorted_species[:num_to_take]]

        quick_marks_data["list_auto"] = list_auto
        if hasattr(self.controller, 'settings_manager'):
            self.controller.settings_manager.save_quick_mark_species(quick_marks_data)
        return list_auto

    def save_quick_mark_settings(self):
        """保存快速标记设置"""
        if not hasattr(self.controller, 'settings_manager'):
            QMessageBox.critical(self, "错误", "设置管理器不可用")
            return

        current_marks = self.controller.settings_manager.load_quick_mark_species()
        new_list = []

        # 收集所有物种名称
        for i in range(self.species_list_layout.count()):
            row_widget = self.species_list_layout.itemAt(i).widget()
            if row_widget:
                species_edits = row_widget.findChildren(ModernLineEdit)
                if len(species_edits) >= 2:
                    species_edit = species_edits[1]
                    species_name = species_edit.text().strip()
                    if species_name:
                        new_list.append(species_name)
                        # 如果是新物种，则在文件中为其添加一个计数为0的条目
                        if species_name not in current_marks:
                            current_marks[species_name] = 0

        current_marks["list"] = new_list
        current_marks["auto"] = self.auto_sort_switch_row.isChecked()

        if self.controller.settings_manager.save_quick_mark_species(current_marks):
            QMessageBox.information(self, "成功", "快速标记设置已保存")
            self.load_quick_mark_settings()  # 重新加载以更新显示
        else:
            QMessageBox.critical(self, "错误", "保存快速标记设置失败")

    def _reset_quick_mark_data(self):
        """清空排序数据并恢复为默认值"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要将快速标记列表恢复为默认设置吗？\n\n此操作将清除所有物种的使用计数和自定义列表。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if hasattr(self.controller, 'settings_manager'):
                # 定义默认的快速标记数据
                default_marks = {
                    "list": [
                        "骆驼", "北山羊", "狗", "蒙古野驴", "鹅喉羚",
                        "马", "中亚兔", "猞猁", "盘羊", "赤狐", "狼"
                    ],
                    "list_auto": [
                        "骆驼", "北山羊", "狗", "蒙古野驴", "鹅喉羚",
                        "马", "中亚兔", "猞猁", "盘羊", "赤狐", "狼"
                    ],
                    "auto": True,
                    "骆驼": 0, "北山羊": 0, "狗": 0, "蒙古野驴": 0,
                    "鹅喉羚": 0, "马": 0, "中亚兔": 0, "猞猁": 0,
                    "盘羊": 0, "赤狐": 0, "狼": 0
                }

                # 使用默认数据覆盖现有文件
                self.controller.settings_manager.save_quick_mark_species(default_marks)
                QMessageBox.information(self, "成功", "快速标记设置已恢复为默认值。")
                self.load_quick_mark_settings()  # 重新加载UI以显示默认值

    # 事件处理函数
    def _on_setting_changed(self):
        """设置改变处理 - 立即保存"""
        try:
            # 获取当前设置
            current_settings = self.get_settings()

            # 发出设置变更信号
            self.settings_changed.emit()

            # 立即保存到JSON文件
            if hasattr(self.controller, 'settings_manager'):
                self.controller.settings_manager.save_settings(current_settings)
                logger.debug("设置已实时保存")

            # 如果controller有save_settings方法，也调用它
            if hasattr(self.controller, 'save_settings'):
                self.controller.save_settings()

        except Exception as e:
            logger.error(f"保存设置失败: {e}")

    def _on_auto_sort_changed(self, checked):
        """自动排序开关改变并立即保存"""
        self.auto_sort_var = checked
        if hasattr(self.controller, 'settings_manager'):
            # 加载当前的快速标记设置
            quick_marks_data = self.controller.settings_manager.load_quick_mark_species()
            # 更新 "auto" 的值
            quick_marks_data["auto"] = checked
            # 立即保存回 quick_mark.json 文件
            self.controller.settings_manager.save_quick_mark_species(quick_marks_data)

        # 重新加载列表以根据新的排序方式更新UI显示
        self.load_quick_mark_settings()

    def _on_theme_changed(self, theme_text):
        """主题改变处理"""
        self.theme_var = theme_text
        self._on_setting_changed()
        self.theme_changed.emit()

    def _on_pytorch_version_changed(self, version):
        """PyTorch版本改变"""
        self.pytorch_version_var = version
        self._on_setting_changed()

    def _on_update_channel_changed(self, channel):
        """更新通道改变"""
        self.update_channel_var = channel
        self._on_setting_changed()

    def _on_package_changed(self, package):
        """包名称改变"""
        self.package_var = package
        self._on_setting_changed()

    def _on_version_constraint_changed(self, constraint):
        """版本约束改变"""
        self.version_constraint_var = constraint
        self._on_setting_changed()

    # 获取器和设置器方法
    def get_use_fp16(self):
        """获取是否使用FP16"""
        return self.fp16_switch_row.isChecked()

    def get_theme_selection(self):
        """获取主题选择"""
        return self.theme_combo.currentText()

    def set_theme_selection(self, theme):
        """设置主题选择"""
        self.theme_combo.setCurrentText(theme)
        self.theme_var = theme

    def get_settings(self):
        """获取页面设置"""
        # 获取当前选择的模型 - 优先级顺序
        selected_model = ""

        # 1. 首先尝试从controller的model_var获取
        if hasattr(self.controller, 'model_var') and self.controller.model_var:
            selected_model = self.controller.model_var
        # 2. 其次从下拉框获取当前选择
        elif self.model_combo.currentText():
            selected_model = self.model_combo.currentText()
        # 3. 最后从image_processor的model_path获取
        elif (hasattr(self.controller, 'image_processor') and
              hasattr(self.controller.image_processor, 'model_path') and
              self.controller.image_processor.model_path):
            selected_model = os.path.basename(self.controller.image_processor.model_path)

        return {
            "iou_threshold": self.iou_var,
            "conf_threshold": self.conf_var,
            "use_fp16": self.get_use_fp16(),
            "use_augment": self.augment_switch_row.isChecked(),
            "use_agnostic_nms": self.agnostic_switch_row.isChecked(),
            "vid_stride": self.vid_stride_var,
            "min_frame_ratio": self.min_frame_ratio_var,
            "theme": self.get_theme_selection(),
            "auto_sort": self.auto_sort_switch_row.isChecked(),
            "update_channel": self.update_channel_combo.currentText(),  # 直接从下拉框获取当前值
            "pytorch_version": self.pytorch_version_combo.currentText(),  # 同样修复
            "package": self.package_edit.text().strip(),  # 直接从输入框获取当前值
            "version_constraint": self.version_constraint_edit.text().strip(),  # 直接从输入框获取当前值
            "selected_model": selected_model,
            "export_columns": [name for name, cb in self.export_checkboxes.items() if cb.isChecked()],
        }

    def load_settings(self, settings):
        """加载页面设置"""
        # 加载模型参数
        if "iou_threshold" in settings:
            self.iou_var = settings["iou_threshold"]
            self.iou_slider.setValue(int(self.iou_var * 100))
            self._update_iou_label(int(self.iou_var * 100))

        if "conf_threshold" in settings:
            self.conf_var = settings["conf_threshold"]
            self.conf_slider.setValue(int(self.conf_var * 100))
            self._update_conf_label(int(self.conf_var * 100))

        if "use_fp16" in settings:
            self.use_fp16_var = settings["use_fp16"]
            self.fp16_switch_row.setChecked(self.use_fp16_var)

        if "use_augment" in settings:
            self.use_augment_var = settings["use_augment"]
            self.augment_switch_row.setChecked(self.use_augment_var)

        if "use_agnostic_nms" in settings:
            self.use_agnostic_nms_var = settings["use_agnostic_nms"]
            self.agnostic_switch_row.setChecked(self.use_agnostic_nms_var)

        if "vid_stride" in settings:
            self.vid_stride_var = int(settings["vid_stride"])
            self.stride_slider.setValue(self.vid_stride_var)
            self.stride_label.setText(str(self.vid_stride_var))

        if "min_frame_ratio" in settings:
            self.min_frame_ratio_var = settings["min_frame_ratio"]
            self.ratio_slider.setValue(int(self.min_frame_ratio_var * 100))
            self.ratio_label.setText(f"{int(self.min_frame_ratio_var * 100)}%")

        # 加载主题设置
        if "theme" in settings:
            self.set_theme_selection(settings["theme"])

        # 加载快速标记设置
        if "auto_sort" in settings:
            self.auto_sort_var = settings["auto_sort"]
            self.auto_sort_switch_row.setChecked(self.auto_sort_var)

        # 加载其他设置 - 修复更新通道设置
        if "update_channel" in settings:
            self.update_channel_var = settings["update_channel"]
            # 确保同时更新下拉框的选择
            self.update_channel_combo.setCurrentText(self.update_channel_var)

        if "pytorch_version" in settings:
            self.pytorch_version_var = settings["pytorch_version"]
            self.pytorch_version_combo.setCurrentText(self.pytorch_version_var)

        if "package" in settings:
            self.package_var = settings["package"]
            self.package_edit.setText(self.package_var)

        if "version_constraint" in settings:
            self.version_constraint_var = settings["version_constraint"]
            self.version_constraint_edit.setText(self.version_constraint_var)

        if "selected_model" in settings and settings["selected_model"]:
            selected_model = settings["selected_model"]
            # 使用定时器延迟设置，确保模型列表已经加载
            QTimer.singleShot(200, lambda: self._set_selected_model(selected_model))

        if "export_columns" in settings:
            selected_columns = settings["export_columns"]
            for name, cb in self.export_checkboxes.items():
                # 根据配置文件中的列表来设置复选框的选中状态
                if name in selected_columns:
                    cb.setChecked(True)
                else:
                    cb.setChecked(False)

            # 加载后，根据单个复选框的状态更新“全选”框的状态
            self._update_select_all_state()

    def _set_selected_model(self, model_name):
        """设置选定的模型"""
        try:
            # 暂时断开信号连接
            self.model_combo.currentTextChanged.disconnect()

            # 查找并设置模型
            for i in range(self.model_combo.count()):
                if self.model_combo.itemText(i) == model_name:
                    self.model_combo.setCurrentIndex(i)
                    self.model_status_label.setText(f"已加载: {model_name}")
                    break
            else:
                logger.warning(f"设置中的模型 {model_name} 在可用模型列表中未找到")

            # 重新连接信号
            self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)

        except Exception as e:
            logger.error(f"设置选定模型失败: {e}")
            # 确保重新连接信号
            try:
                self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)
            except:
                pass

    def update_theme(self):
        """更新主题"""
        # 重新应用主题
        self._apply_win11_style()

        # 更新所有自定义组件的主题
        for component in self.components_to_update:
            if hasattr(component, 'update_theme'):
                component.update_theme()

        # 更新所有可折叠面板
        for panel in [self.threshold_panel, self.accel_panel, self.advanced_detect_panel,
                      self.frame_skip_panel, self.frame_ratio_panel, self.pytorch_panel, self.model_panel, self.python_panel,
                      self.quick_mark_panel, self.theme_panel, self.cache_panel, self.update_panel,
                      self.export_settings_panel]:
            if hasattr(panel, 'update_theme'):
                panel.update_theme()

    def clear_validation_data(self):
        """清除验证数据"""
        # 这个方法可能被其他地方调用，提供空实现
        pass

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 可以在这里处理窗口大小改变时的逻辑
        pass

    def showEvent(self, event):
        """显示事件"""
        super().showEvent(event)
        # 页面显示时可能需要的初始化逻辑
        pass

    def hideEvent(self, event):
        """隐藏事件"""
        super().hideEvent(event)
        # 页面隐藏时可能需要的清理逻辑
        pass

    def get_selected_export_columns(self):
        """获取用户选择的要导出的列名列表"""
        if not hasattr(self, 'export_checkboxes'):
            return self.all_export_columns  # 如果UI未完全初始化，返回所有列
        return [name for name, cb in self.export_checkboxes.items() if cb.isChecked()]

    def _toggle_all_columns(self, state):
        """
        响应“全选/全不选”复选框的点击事件。
        """
        # 临时断开单个复选框的信号连接，防止循环触发
        for checkbox in self.export_checkboxes.values():
            checkbox.blockSignals(True)

        # 设置所有单个复选框的状态
        is_checked = (state == Qt.CheckState.Checked.value)
        for checkbox in self.export_checkboxes.values():
            checkbox.setChecked(is_checked)

        # 恢复信号连接
        for checkbox in self.export_checkboxes.values():
            checkbox.blockSignals(False)

        # 手动触发一次设置保存
        self._on_setting_changed()

    def _update_select_all_state(self):
        """
        当单个导出列复选框状态改变时，更新“全选/全不选”复选框的状态。
        """
        # 检查是否所有复选框都被选中
        all_checked = all(cb.isChecked() for cb in self.export_checkboxes.values())

        # 临时断开“全选”复选框的信号连接，防止它反过来调用 _toggle_all_columns
        self.select_all_checkbox.blockSignals(True)

        self.select_all_checkbox.setChecked(all_checked)

        # 恢复信号连接
        self.select_all_checkbox.blockSignals(False)

        # 触发设置保存
        self._on_setting_changed()
