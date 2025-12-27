# system/update_checker.py

import requests
import re
import os
import zipfile
import io
import shutil
import sys
import subprocess
import threading
import platform
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
    支持镜像源替换。
    """
    # 根据镜像源构建 API URL
    base_api_url = "https://api.github.com"

    if mirror == 'kkgithub':
        base_api_url = "https://api.kkgithub.com"

    api_url = f"{base_api_url}/repos/{GITHUB_USER}/{GITHUB_REPO}/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(api_url, headers=headers, timeout=15)  # 稍微增加超时时间
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
        notes = latest_release.get('body')
        if notes is None:
            notes = '无更新说明。'

        # 获取原始下载链接
        download_url = latest_info = latest_release.get('zipball_url')

        if download_url and mirror != 'official':
            # 简单的域名替换逻辑
            if mirror == 'kkgithub':
                download_url = download_url.replace("api.github.com", "api.kkgithub.com")
                download_url = download_url.replace("github.com", "kkgithub.com")

        return {
            'version': tag_name.lstrip('v'),
            'notes': notes,
            'url': download_url
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
    """
    status_changed = Signal(str)
    progress_updated = Signal(int)
    progress_max_set = Signal(int)
    finished = Signal(bool, str)  # success/fail, message

    def __init__(self, download_url):
        super().__init__()
        self.download_url = download_url
        self._is_running = True

    def run(self):
        """执行下载、解压、安装的完整流程。"""
        try:
            self.status_changed.emit("正在从GitHub下载更新...")
            response = requests.get(self.download_url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            self.progress_max_set.emit(total_size if total_size > 0 else 100)

            downloaded_size = 0
            file_buffer = io.BytesIO()

            for chunk in response.iter_content(chunk_size=8192):
                if not self._is_running:
                    self.finished.emit(False, "用户取消操作")
                    return
                if chunk:
                    file_buffer.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        self.progress_updated.emit(downloaded_size)

            self.status_changed.emit("下载完成，正在解压并安装...")
            self.progress_max_set.emit(0)  # 进入不确定模式

            if getattr(sys, 'frozen', False):
                app_root = os.path.dirname(sys.executable)
            else:
                app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            file_buffer.seek(0)

            with zipfile.ZipFile(file_buffer) as z:
                root_folder_in_zip = z.namelist()[0]
                for member in z.infolist():
                    if not self._is_running:
                        self.finished.emit(False, "用户取消操作")
                        return
                    path_in_zip = member.filename.replace(root_folder_in_zip, '', 1)
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
            self.finished.emit(False, str(e))

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
        """处理工作线程完成的信号"""
        self.thread.quit()
        self.thread.wait()
        self.close()

        if success:
            self.finished_successfully = True
        else:
            if "用户取消操作" not in message:
                _show_messagebox(self.parent(), "更新失败", f"更新过程中发生错误: {message}", "error")

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
        reply = QMessageBox.question(self.parent(), "更新成功",
                                     "程序已成功更新！\n是否立即重启应用程序以应用更改？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                # 确定重启命令的参数
                if getattr(sys, 'frozen', False):
                    # 对于打包后的程序（frozen=True），重启逻辑保持不变，直接重启主程序
                    args = [sys.executable]
                    if platform.system() == "Windows":
                        cmd = ' '.join(f'"{arg}"' for arg in args)
                        subprocess.Popen(f'start "Restarting Application" {cmd}', shell=True)
                    elif platform.system() == "Darwin":  # macOS
                        cmd = ' '.join(f'"{arg}"' for arg in args)
                        mac_cmd = cmd.replace("\"", "\\\"")
                        subprocess.Popen(
                            ["osascript", "-e", f'tell app "Terminal" to do script "{mac_cmd}"'],
                            close_fds=True
                        )
                    else:  # Linux
                        terminal_found = False
                        for terminal in ["gnome-terminal", "konsole", "xterm"]:
                            try:
                                if terminal == "gnome-terminal":
                                    subprocess.Popen([terminal, "--"] + args, close_fds=True)
                                elif terminal == "konsole":
                                    subprocess.Popen([terminal, "-e"] + args, close_fds=True)
                                elif terminal == "xterm":
                                    subprocess.Popen([terminal, "-e"] + args, close_fds=True)
                                terminal_found = True
                                break
                            except FileNotFoundError:
                                continue
                        if not terminal_found:
                            _show_messagebox(self.parent(), "重启提示",
                                             "无法自动打开新终端，请手动重启程序以应用更新。",
                                             "info")
                else:
                    # 对于开发环境（直接运行 .py 文件）
                    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    python_exe_path = os.path.join(app_root, "toolkit", "python.exe")
                    pythonw_exe_path = os.path.join(app_root, "toolkit", "pythonw.exe")
                    checker_script_path = os.path.join(app_root, 'checker.py')
                    main_script_path = os.path.join(app_root, 'gui.py')

                    if not os.path.exists(python_exe_path):
                        python_exe_path = sys.executable
                        # 尝试基于 sys.executable 推断 pythonw.exe 的路径
                        pythonw_exe_path = python_exe_path.replace("python.exe", "pythonw.exe")

                    if not os.path.exists(pythonw_exe_path):
                        # 如果 pythonw.exe 不存在，则回退到 python.exe
                        pythonw_exe_path = python_exe_path

                    if not os.path.exists(checker_script_path):
                        _show_messagebox(self.parent(), "重启错误", f"找不到脚本: {checker_script_path}", "error")
                        return
                    if not os.path.exists(main_script_path):
                        _show_messagebox(self.parent(), "重启错误", f"找不到主脚本: {main_script_path}", "error")
                        return

                    # 在新的控制台中重新启动应用程序
                    if platform.system() == "Windows":
                        # 修改后的 Windows 重启逻辑，确保 checker.py 在可见的命令行窗口中运行
                        # 使用 start 命令打开新的命令行窗口，并在其中运行 checker.py
                        # /WAIT 参数确保等待 checker.py 执行完毕
                        # 然后使用 pythonw.exe 启动 gui.py（无窗口）
                        cmd_sequence = f'start "Neri checker" /WAIT "{python_exe_path}" "{checker_script_path}" && "{pythonw_exe_path}" "{main_script_path}"'

                        # 使用 cmd /c 执行命令序列
                        subprocess.Popen(f'cmd /c "{cmd_sequence}"', shell=True)

                    elif platform.system() == "Darwin":  # macOS
                        # 使用bash执行命令序列
                        bash_cmd = f'''
if "{python_exe_path}" "{checker_script_path}"; then 
"{pythonw_exe_path}" "{main_script_path}" & 
fi; 
exit
'''.strip().replace('\n', ' ')

                        subprocess.Popen([
                            "osascript", "-e",
                            f'tell app "Terminal" to do script "{bash_cmd.replace(chr(34), chr(92) + chr(34))}"'
                        ], close_fds=True)

                    else:  # Linux
                        # 使用bash执行命令序列
                        bash_cmd = f'''
if "{python_exe_path}" "{checker_script_path}"; then 
"{pythonw_exe_path}" "{main_script_path}" & 
fi; 
exit
'''.strip().replace('\n', ' ')

                        terminal_found = False
                        for terminal in ["gnome-terminal", "konsole", "xterm"]:
                            try:
                                if terminal == "gnome-terminal":
                                    subprocess.Popen([terminal, "--", "bash", "-c", bash_cmd], close_fds=True)
                                elif terminal == "konsole":
                                    subprocess.Popen([terminal, "-e", "bash", "-c", bash_cmd], close_fds=True)
                                elif terminal == "xterm":
                                    subprocess.Popen([terminal, "-e", "bash", "-c", bash_cmd], close_fds=True)
                                terminal_found = True
                                break
                            except FileNotFoundError:
                                continue
                        if not terminal_found:
                            _show_messagebox(self.parent(), "重启提示",
                                             "无法自动打开新终端，请手动重启程序以应用更新。",
                                             "info")

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