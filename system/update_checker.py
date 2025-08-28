# system/update_checker.py

import requests
import re  # 确保导入 re 模块
import os
import zipfile
import io
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import subprocess
import threading
import platform
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


def get_latest_version_info(channel='stable'):
    """
    通过GitHub API获取最新的版本信息。
    channel: 'stable' 获取最新的正式版, 'preview' 获取最新的版本(包括预发布版)。
    """
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
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
        return {
            'version': tag_name.lstrip('v'),
            'notes': latest_release.get('body', '无更新说明。'),
            'url': latest_release.get('zipball_url')
        }

    except requests.RequestException as e:
        print(f"从GitHub获取版本信息失败: {e}")
        return None


def check_for_updates(parent, silent=False, channel='preview'):
    """
    在后台线程中检查是否有新版本。
    此函数由主窗口在启动时调用。
    """
    try:
        latest_info = get_latest_version_info(channel=channel)
        if not latest_info:
            if not silent and parent.winfo_exists():
                _show_messagebox(parent, "更新错误", "无法在远程仓库中找到版本信息。", "error")
            return

        remote_version = latest_info['version']

        if compare_versions(APP_VERSION, remote_version):
            # 只有在找到更新时才与UI交互
            if parent.winfo_exists():
                # 安全地调用主窗口的方法来更新侧边栏
                parent.after(0, parent.show_update_notification_on_sidebar)

            # 非静默模式下弹窗提示
            if not silent and parent.winfo_exists():
                update_message = f"新版本 ({remote_version}) 可用，是否前往高级设置进行更新？"
                _show_messagebox(parent, "发现新版本", update_message, "info")

    except Exception as e:
        if not silent and parent.winfo_exists():
            _show_messagebox(parent, "更新错误", f"检查更新失败: {e}", "error")


def start_download_thread(parent, download_url):
    """启动下载更新的线程。"""
    if not download_url:
        _show_messagebox(parent, "错误", "下载链接无效！", "error")
        return
    download_thread = threading.Thread(target=download_and_install_update, args=(parent, download_url), daemon=True)
    download_thread.start()


def download_and_install_update(parent, download_url):
    """创建带进度条的更新窗口。"""

    def create_progress_window():
        global progress_window, label, progress_bar
        progress_window = tk.Toplevel(parent)
        progress_window.title("正在更新...")
        progress_window.geometry("350x120")

        icon_path = get_icon_path()
        if os.path.exists(icon_path):
            try:
                progress_window.iconbitmap(icon_path)
            except tk.TclError:
                pass

        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (350 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (120 // 2)
        progress_window.geometry(f'+{x}+{y}')
        progress_window.transient(parent)
        progress_window.grab_set()

        label = ttk.Label(progress_window, text="正在连接到服务器...")
        label.pack(padx=20, pady=10)
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode='determinate')
        progress_bar.pack(padx=20, pady=5)

        threading.Thread(target=perform_download, args=(parent, download_url), daemon=True).start()

    if parent.winfo_exists():
        parent.after(0, create_progress_window)


def perform_download(parent_window, download_url):
    """执行下载、解压、安装和重启的完整流程。"""

    def update_ui(text=None, p_value=None, p_mode=None, start=False, stop=False):
        if 'progress_window' in globals() and progress_window.winfo_exists():
            if text: label.config(text=text)
            if p_value is not None: progress_bar['value'] = p_value
            if p_mode: progress_bar['mode'] = p_mode
            if start: progress_bar.start(10)
            if stop: progress_bar.stop()
            progress_window.update_idletasks()

    try:
        parent_window.after(0, lambda: update_ui(text="正在从GitHub下载更新..."))
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        parent_window.after(0, lambda: progress_bar.config(maximum=total_size if total_size > 0 else 100))

        downloaded_size = 0
        file_buffer = io.BytesIO()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_buffer.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    parent_window.after(0, lambda v=downloaded_size: update_ui(p_value=v))

        parent_window.after(0, lambda: update_ui(text="下载完成，正在解压并安装...", p_mode='indeterminate', start=True))

        if getattr(sys, 'frozen', False):
            app_root = os.path.dirname(sys.executable)
        else:
            app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        file_buffer.seek(0)

        with zipfile.ZipFile(file_buffer) as z:
            root_folder_in_zip = z.namelist()[0]
            for member in z.infolist():
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

        parent_window.after(0, lambda: update_ui(stop=True))
        parent_window.after(0,
                            lambda: progress_window.destroy() if 'progress_window' in globals() and progress_window.winfo_exists() else None)

        def ask_restart():
            if messagebox.askyesno("更新成功", "程序已成功更新！\n是否立即重启应用程序以应用更改？",
                                   parent=parent_window):
                # 确定重启命令的参数
                if getattr(sys, 'frozen', False):
                    args = [sys.executable]
                else:
                    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    python_exe_path = os.path.join(app_root, "toolkit", "python.exe")
                    main_script_path = os.path.join(app_root, 'gui.py')

                    if not os.path.exists(python_exe_path):
                        # 如果 toolkit/python.exe 不存在，则回退到 sys.executable
                        python_exe_path = sys.executable

                    if not os.path.exists(main_script_path):
                        _show_messagebox(parent_window, "重启错误", f"找不到主脚本: {main_script_path}", "error")
                        return
                    args = [python_exe_path, main_script_path]

                try:
                    # 在新的控制台中重新启动应用程序
                    if platform.system() == "Windows":
                        # 在Windows上，使用'start'命令在一个新窗口中创建一个独立的进程。
                        # 这能正确地模拟用户双击应用程序的效果, 弹出命令行并在之后自动关闭。
                        cmd = ' '.join(f'"{arg}"' for arg in args)
                        subprocess.Popen(f'start "Restarting Application" {cmd}', shell=True)
                    elif platform.system() == "Darwin":  # macOS
                        # 在macOS上，使用AppleScript在新Terminal窗口中运行命令。
                        cmd = ' '.join(f'"{arg}"' for arg in args)
                        mac_cmd = cmd.replace("\"", "\\\"")
                        subprocess.Popen(
                            ["osascript", "-e", f'tell app "Terminal" to do script "{mac_cmd}"'],
                            close_fds=True
                        )
                    else:  # Linux
                        # 在Linux上，尝试在新的终端模拟器中启动。
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
                            # 如果找不到终端，则显示消息提示用户手动重启
                            _show_messagebox(parent_window, "重启提示", "无法自动打开新终端，请手动重启程序以应用更新。",
                                             "info")

                    # 关闭当前的应用程序实例
                    parent_window.destroy()
                    sys.exit()

                except Exception as e:
                    _show_messagebox(parent_window, "重启失败", f"无法重新启动应用程序: {e}", "error")

        parent_window.after(0, ask_restart)

    except Exception as e:
        parent_window.after(0,
                            lambda: progress_window.destroy() if 'progress_window' in globals() and progress_window.winfo_exists() else None)
        _show_messagebox(parent_window, "更新失败", f"更新过程中发生错误: {e}", "error")


def _show_messagebox(parent, title, message, msg_type):
    """内部辅助函数，确保在主线程中调用messagebox。"""

    def show_message():
        if not parent.winfo_exists(): return
        transient_parent = tk.Toplevel(parent)
        transient_parent.withdraw()
        if msg_type == "info":
            messagebox.showinfo(title, message, parent=transient_parent)
        elif msg_type == "error":
            messagebox.showerror(title, message, parent=transient_parent)
        elif msg_type == "askyesno":
            messagebox.askyesno(title, message, parent=transient_parent)
        if transient_parent.winfo_exists():
            transient_parent.destroy()

    if parent.winfo_exists():

        parent.after(0, show_message)
