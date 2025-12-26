"""
配置模块 - 包含应用程序的常量定义和配置参数
"""
import zlib  # Add this import for stable hashing

# 应用信息常量
APP_TITLE = "Neri v2.2.7"
APP_VERSION = "2.2.7-beta"
DEFAULT_EXCEL_FILENAME = "物种检测信息.xlsx"

# 文件支持相关常量
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.ts')
DATE_FORMATS = ['%Y:%m:%d %H:%M:%S', '%Y:%d:%m %H:%M:%S', '%Y-%m-%d %H:%M:%S']
INDEPENDENT_DETECTION_THRESHOLD = 30 * 60  # 30分钟，单位：秒

# 界面相关常量
PADDING = 10
BUTTON_WIDTH = 14
LARGE_FONT = ('Segoe UI', 11)
NORMAL_FONT = ('Segoe UI', 10)
SMALL_FONT = ('Segoe UI', 9)

# === Unified Species Color Palette (Added) ===
# A rich palette of distinct colors
SPECIES_COLOR_PALETTE = [
    '#0078d4', '#d13438', '#00bcf2', '#107c10', '#5c2d91', '#ff8c00',
    '#e3008c', '#00b294', '#40e0d0', '#f7630c', '#ffb900', '#bad80a',
    '#00b4ff', '#e74856', '#ffd700', '#32cd32', '#da70d6', '#4682b4',
    '#8E44AD', '#2ECC71', '#F39C12', '#D35400', '#C0392B', '#16A085'
]


def get_species_color(species_name, return_rgb=False):
    """
    Get a consistent color for a specific species name.
    Uses zlib.adler32 for stable hashing across sessions.

    Args:
        species_name (str): The name of the species.
        return_rgb (bool): If True, returns (r, g, b) tuple. Else returns Hex string.
    """
    if not species_name:
        return (128, 128, 128) if return_rgb else '#808080'

    # Use adler32 for a deterministic hash that stays same across app restarts
    hash_val = zlib.adler32(species_name.encode('utf-8'))
    hex_color = SPECIES_COLOR_PALETTE[hash_val % len(SPECIES_COLOR_PALETTE)]

    if return_rgb:
        # Convert Hex to RGB Tuple
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


    return hex_color
