"""
配置模块 - 包含应用程序的常量定义和配置参数
"""

# 应用信息常量
APP_TITLE = "Neri v2.6"
APP_VERSION = "2.6.1-release"
DEFAULT_EXCEL_FILENAME = "物种检测信息.xlsx"

# 文件支持相关常量
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
DATE_FORMATS = ['%Y:%m:%d %H:%M:%S', '%Y:%d:%m %H:%M:%S', '%Y-%m-%d %H:%M:%S']
INDEPENDENT_DETECTION_THRESHOLD = 30 * 60  # 30分钟，单位：秒

# 界面相关常量
PADDING = 10
BUTTON_WIDTH = 14
LARGE_FONT = ('Segoe UI', 11)
NORMAL_FONT = ('Segoe UI', 10)
SMALL_FONT = ('Segoe UI', 9)

