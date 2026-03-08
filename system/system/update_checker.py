# system/update_checker.py

import requests
import re
import os
import zipfile
import io
import shutil
import sys
import subprocess
import time
import threading
import platform
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox
# Make sure Slot is imported
from PySide6.QtCore import Qt, QThread, Signal, QMetaObject, Q_ARG, QObject, Slot
from system.config import APP_VERSION

# GitHub仓库信息
GITHUB_USER = "wakin721"
GITHUB_REPO = "Neri"


def get_icon_path():
    """获取图标文件的绝对路径。"""
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "res", "ico.ico")

def parse_version(version_string):
    """
    解析版本字符串，返回用于比较的元组。
    支持格式: major.minor.patch-prerelease[number] (例如: 1.2.3-beta1)
    """
    prerelease_priority = {'alpha': 1, 'beta': 2, 'rc': 3, 'release': 4}

    prerelease_part = 'release'
    prerelease_num = 0

    if '-' in version_string:
        main_version, prerelease_full = version_string.split('-', 1)
        # 使用正则表达式从预发布字符串中分离出字母和数字
        match = re.match(r"([a-zA-Z]+)(\d*)", prerelease_full)
        if match:
            prerelease_part = match.group(1).lower()
            if match.group(2):
                prerelease_num = int(match.group(2))
        else:
            prerelease_part = prerelease_full.lower()

    else:
        main_version = version_string
        prerelease_part = 'release'

    try:
        version_parts = list(map(int, main_version.split('.')))
        prerelease_value = prerelease_priority.get(prerelease_part, 0)
        # 将预发布版本号添加到元组中进行比较
        return tuple(version_parts + [prerelease_value, prerelease_num])
    except ValueError:
        # Fallback for non-standard version strings
        return (0,)

def compare_versions(current_version, remote_version):
    """比较两个版本，如果远程版本更新则返回True。"""
    current_tuple = parse_version(current_version)
    remote_tuple = parse_version(remote_version)
    return remote_tuple > current_tuple


def get_latest_version_info(channel='stable', mirror='official'):
    """
    通过GitHub API获取最新的版本信息。
    支持镜像源替换，并优先使用Assets中的zip包。
    """
    base_api_url = "https://api.github.com"
    verify_ssl = True

    if mirror == 'kkgithub':
        base_api_url = "https://api.kkgithub.com"
        verify_ssl = False

    api_url = f"{base_api_url}/repos/{GITHUB_USER}/{GITHUB_REPO}/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(api_url, headers=headers, timeout=15, verify=verify_ssl)
        response.raise_for_status()
        releases = response.json()
        if not releases:
            return None

        latest_release = None
        if channel == 'stable':
            for release in releases:
                if not release.get('prerelease') and not release.get('draft'):
                    latest_release = release
                    break
        else:  # channel == 'preview'
            for release in releases:
                if not release.get('draft'):
                    latest_release = release
                    break

        if not latest_release:
            return None

        tag_name = latest_release.get('tag_name', 'v0.0.0')
        notes = latest_release.get('body') or '无更新说明。'

        # 1. 默认获取源码包下载链接
        download_url = latest_release.get('zipball_url')

        # 2. [修改] 优先检查 Assets 里的 exe 文件
        assets = latest_release.get('assets', [])
        for asset in assets:
            # 查找以 .exe 结尾的文件
            if asset.get('name', '').lower().endswith('.exe'):
                download_url = asset.get('browser_download_url')
                # 记录文件名，方便后续保存
                latest_info_filename = asset.get('name')
                break
        else:
            # 如果没找到 exe，回退到 zip 或源码包
            latest_info_filename = "update.zip"

        # 域名替换逻辑
        if download_url and mirror != 'official':
            if mirror == 'kkgithub':
                if "kkgithub.com" in download_url:
                    download_url = download_url.replace("kkgithub.com", "github.com")
                download_url = download_url.replace("github.com", "kkgithub.com")

        return {
            'version': tag_name.lstrip('v'),
            'notes': notes,
            'url': download_url,
            'filename': latest_info_filename
        }

    except requests.RequestException as e:
        print(f"从GitHub (镜像: {mirror}) 获取版本信息失败: {e}")
        return None

def check_for_updates(parent, silent=False, channel='preview', mirror='official'):
    """
    在后台线程中检查是否有新版本。
    警告：此函数在后台线程运行，绝不能直接调用 UI 控件（如 QMessageBox），否则会导致闪退。
    """
    try:
        latest_info = get_latest_version_info(channel=channel, mirror=mirror)

        if not latest_info:
            if not silent and parent:
                # [关键修复] 不再直接调用 _show_messagebox (会导致崩溃)
                # 而是更新状态栏提示错误
                err_msg = f"检查更新失败: 无法连接到服务器 ({mirror})"
                QMetaObject.invokeMethod(parent, "set_status_bar_message", Qt.QueuedConnection,
                                         Q_ARG(str, err_msg))
            return

        remote_version = latest_info['version']

        # 如果发现新版本
        if compare_versions(APP_VERSION, remote_version):
            if parent:
                QMetaObject.invokeMethod(parent, "set_status_bar_message", Qt.QueuedConnection,
                                         Q_ARG(str, f"发现新版本：{remote_version}"))

            if parent:
                download_url = latest_info.get('url')
                release_notes = latest_info.get('notes') or '无更新说明。'
                release_notes = str(release_notes)

                # 使用 invokeMethod 安全地在主线程弹出更新确认框
                QMetaObject.invokeMethod(
                    parent,
                    "prompt_for_update",
                    Qt.QueuedConnection,
                    Q_ARG(str, remote_version),
                    Q_ARG(str, download_url),
                    Q_ARG(str, release_notes)
                )
        else:
            # 未发现新版本
            if not silent and parent:
                # [关键修复] 使用状态栏提示代替弹窗，防止线程崩溃
                QMetaObject.invokeMethod(parent, "set_status_bar_message", Qt.QueuedConnection,
                                         Q_ARG(str, f"当前已是最新版本 ({APP_VERSION})"))

    except Exception as e:
        if not silent and parent:
            # [关键修复] 使用状态栏提示错误
            QMetaObject.invokeMethod(parent, "set_status_bar_message", Qt.QueuedConnection,
                                     Q_ARG(str, f"检查更新出错: {e}"))

def start_download_thread(parent, download_url):
    """启动下载更新的线程。"""
    if not download_url:
        _show_messagebox(parent, "错误", "下载链接无效！", "error")
        return

    progress_dialog = UpdateProgressDialog(parent, download_url)
    progress_dialog.exec()


class UpdateWorker(QObject):
    """
    更新工作线程，负责下载和解压文件，并通过信号与主线程通信。
    包含断线重试机制。
    """
    status_changed = Signal(str)
    progress_updated = Signal(int)
    progress_max_set = Signal(int)
    finished = Signal(bool, str)  # success/fail, message

    def __init__(self, download_url):
        super().__init__()
        self.download_url = download_url
        self._is_running = True

    def _format_speed(self, bytes_per_sec):
        """辅助函数：格式化速度显示"""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.1f} B/s"
        elif bytes_per_sec < 1024 * 1024:
            return f"{bytes_per_sec / 1024:.1f} KB/s"
        else:
            return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"

    def run(self):
        """执行下载、解压、安装的完整流程。"""
        # --- 配置 ---
        max_retries = 3  # 最大重试次数
        retry_delay = 3  # 重试间隔(秒)
        verify_ssl = True
        if 'kkgithub' in self.download_url:
            verify_ssl = False

        download_success = False
        file_buffer = io.BytesIO()
        last_error = ""
        self.last_ui_update = 0

        # =======================
        # 阶段 1: 下载 (带重试)
        # =======================
        for attempt in range(max_retries):
            try:
                if not self._is_running:
                    self.finished.emit(False, "用户取消操作")
                    return

                # 每次尝试前重置缓冲区
                file_buffer.seek(0)
                file_buffer.truncate(0)

                # 更新状态
                if attempt > 0:
                    self.status_changed.emit(f"下载中断，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    self.status_changed.emit("正在连接服务器...")

                # 发起请求
                response = requests.get(self.download_url, stream=True, timeout=(5, 30), verify=verify_ssl)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                self.progress_max_set.emit(total_size if total_size > 0 else 100)

                downloaded_size = 0

                # --- 速度计算初始化 ---
                start_time = time.time()
                last_time = start_time
                last_downloaded_size = 0
                update_interval = 0.5
                # --------------------

                self.status_changed.emit("开始下载...")

                for chunk in response.iter_content(chunk_size=8192):
                    if not self._is_running:
                        self.finished.emit(False, "用户取消操作")
                        return
                    if chunk:
                        file_buffer.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            self.progress_updated.emit(downloaded_size)

                        # --- 实时计算下载速度 ---
                        current_time = time.time()
                        if current_time - self.last_ui_update > 0.1:  # 限制每秒最多更新10次UI
                            self.progress_updated.emit(downloaded_size)
                            self.last_ui_update = current_time
                            speed = (downloaded_size - last_downloaded_size) / (current_time - last_time)
                            speed_str = self._format_speed(speed)

                            # 计算百分比
                            if total_size > 0:
                                percent = int(downloaded_size * 100 / total_size)
                                status_msg = f"正在下载... {percent}% ({speed_str})"
                            else:
                                downloaded_mb = downloaded_size / (1024 * 1024)
                                status_msg = f"正在下载... {downloaded_mb:.1f} MB ({speed_str})"

                            self.status_changed.emit(status_msg)
                            last_time = current_time
                            last_downloaded_size = downloaded_size
                        # ----------------------

                # 如果循环正常结束，说明下载完成
                download_success = True
                break  # 跳出重试循环

            except Exception as e:
                # 捕获所有下载异常（包括 IncompleteRead, ConnectionError, Timeout 等）
                last_error = str(e)
                print(f"下载尝试 {attempt + 1} 失败: {e}")
                # 继续下一次循环

        if not download_success:
            self.finished.emit(False, f"更新下载失败 (重试{max_retries}次): {last_error}")
            return

        # =======================
        # 阶段 2: 解压与安装
        # =======================
        try:
            self.status_changed.emit("下载完成，准备安装...")

            # 获取下载的文件名后缀
            is_exe = self.download_url.lower().endswith('.exe')
            import tempfile
            # 确定保存路径（如果是 exe，建议保存到用户临时文件夹或程序目录）
            save_path = os.path.join(tempfile.gettempdir(), "Neri_Update")
            os.makedirs(save_path, exist_ok=True)

            if is_exe:
                # 获取文件名并保存
                file_name = self.download_url.split('/')[-1]
                target_file = os.path.join(save_path, file_name)

                with open(target_file, "wb") as f:
                    f.write(file_buffer.getvalue())

                # 将最终路径传给 finished 信号，方便重启调用
                self.finished.emit(True, target_file)

            else:
                self.progress_max_set.emit(0)  # 忙碌模式

                if getattr(sys, 'frozen', False):
                    app_root = os.path.dirname(sys.executable)
                else:
                    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                file_buffer.seek(0)

                with zipfile.ZipFile(file_buffer) as z:
                    # 智能判断是否需要去除顶层目录
                    all_names = z.namelist()
                    root_folder_to_strip = ""

                    if all_names:
                        common_prefix = os.path.commonprefix(all_names)
                        if '/' in common_prefix:
                            root_folder_to_strip = common_prefix

                    for member in z.infolist():
                        if not self._is_running:
                            self.finished.emit(False, "用户取消操作")
                            return

                        if root_folder_to_strip:
                            path_in_zip = member.filename.replace(root_folder_to_strip, '', 1)
                        else:
                            path_in_zip = member.filename

                        if not path_in_zip or ".git" in path_in_zip: continue

                        target_path = os.path.join(app_root, path_in_zip)
                        if member.is_dir():
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            target_dir = os.path.dirname(target_path)
                            os.makedirs(target_dir, exist_ok=True)
                            with z.open(member) as source, open(target_path, "wb") as target:
                                shutil.copyfileobj(source, target)

                self.finished.emit(True, "更新成功")

        except Exception as e:
            self.finished.emit(False, f"解压安装失败: {str(e)}")

    def stop(self):
        self._is_running = False


class UpdateProgressDialog(QDialog):
    def __init__(self, parent, download_url):
        super().__init__(parent)
        self.setWindowTitle("正在更新...")
        self.setWindowIcon(parent.windowIcon())
        self.setModal(True)
        self.finished_successfully = False

        self.layout = QVBoxLayout(self)
        self.label = QLabel("正在连接到服务器...", self)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)

        # 使用QThread来管理工作线程
        self.thread = QThread(self)
        self.worker = UpdateWorker(download_url)
        self.worker.moveToThread(self.thread)

        # 连接信号与槽
        self.worker.status_changed.connect(self.label.setText)
        # --- FIX STARTS HERE ---
        # 连接到新的辅助槽，而不是直接连接到 setRange
        self.worker.progress_max_set.connect(self.on_set_progress_range)
        # --- FIX ENDS HERE ---
        self.worker.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker.finished.connect(self.on_finished)
        self.thread.started.connect(self.worker.run)

        self.thread.start()
        self.new_exe_path = None

    # --- FIX STARTS HERE ---
    @Slot(int)
    def on_set_progress_range(self, max_value):
        """这个槽函数接收一个参数，并用它来正确调用setRange(min, max)"""
        if max_value > 0:
            # 设置正常的进度范围
            self.progress_bar.setRange(0, max_value)
        else:
            # 设置为不确定模式（min=0, max=0）
            self.progress_bar.setRange(0, 0)

    # --- FIX ENDS HERE ---

    def on_finished(self, success, message):
        self.thread.quit()
        self.thread.wait()
        self.close()

        if success:
            self.finished_successfully = True
            # 如果 message 是一个真实存在的 exe 路径
            if message.lower().endswith('.exe') and os.path.exists(message):
                self.new_exe_path = message
            else:
                self.new_exe_path = None
        else:
            if "用户取消操作" not in message:
                _show_messagebox(self.parent(), "更新失败", f"错误: {message}", "error")

    def exec(self):
        """重写exec，在对话框关闭后执行重启逻辑"""
        super().exec()
        if self.finished_successfully:
            self.ask_restart()

    def closeEvent(self, event):
        """关闭对话框时，尝试停止工作线程"""
        if self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        event.accept()

    def ask_restart(self):
        # 只有确实拿到了新路径才执行直接启动逻辑
        if self.new_exe_path and self.new_exe_path.lower().endswith('.exe'):
            try:
                if platform.system() == "Windows":
                    # 使用 subprocess.DETACHED_PROCESS 标志让新进程完全独立（可选）
                    subprocess.Popen(f'start "" "{self.new_exe_path}"', shell=True)

                # 退出当前程序
                if self.parent(): self.parent().close()
                QApplication.instance().quit()
                sys.exit(0)
            except Exception as e:
                _show_messagebox(self.parent(), "启动失败", f"无法运行更新程序: {e}", "error")

        else:
            reply = QMessageBox.question(self.parent(), "更新成功",
                                         "程序已成功更新！\n是否立即重启应用程序以应用更改？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    # 获取应用程序根目录
                    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    # 指定启动 Neri.exe
                    exe_path = os.path.join(app_root, "Neri.exe")

                    # 检查 Neri.exe 是否存在 (针对非 frozen 情况)
                    if not getattr(sys, 'frozen', False) and not os.path.exists(exe_path):
                        _show_messagebox(self.parent(), "重启错误", f"找不到可执行文件: {exe_path}", "error")
                        return

                    # 执行重启逻辑
                    if platform.system() == "Windows":
                        # 使用 start 命令启动，确保与当前进程分离
                        subprocess.Popen(f'start "" "{exe_path}"', shell=True)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.Popen(["open", exe_path])
                    else:  # Linux
                        subprocess.Popen([exe_path])

                    # 关闭当前的应用程序实例
                    self.parent().close()
                    QApplication.instance().quit()

                except Exception as e:
                    _show_messagebox(self.parent(), "重启失败", f"无法重新启动应用程序: {e}", "error")


def _show_messagebox(parent, title, message, msg_type):
    """内部辅助函数，确保在主线程中调用messagebox。"""
    if msg_type == "info":
        QMessageBox.information(parent, title, message)
    elif msg_type == "error":
        QMessageBox.critical(parent, title, message)
    elif msg_type == "askyesno":
        return QMessageBox.question(parent, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)