#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neri 深度学习模型 SSH 远程连接与训练管理工具 (Python 跨平台版)

功能:
1. 模型服务端口映射隧道 (TensorBoard / Jupyter / WebUI / FastAPI / Neri Service 等)
2. 训练防掉线会话管理 (自动附加/新建 Tmux、SSH KeepAlive 保活)
3. 远程 GPU 监控与 PyTorch/CUDA 深度学习环境诊断
4. 模型权重 (Weights: *.pt, *.onnx, *.safetensors) 与数据集双向快速同步 (SCP)
5. Neri 云端模型分发服务管理 (状态查询、日志流、重启)
6. 一键导出 VS Code Remote-SSH 连接配置
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "model_ssh_config.json"


def load_config(config_path: Path = DEFAULT_CONFIG_FILE) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_data: dict, config_path: Path = DEFAULT_CONFIG_FILE) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 配置文件已更新: {config_path}")


def resolve_profile(config: dict, name: str = "", overrides: dict = None) -> dict:
    overrides = overrides or {}
    target_name = name or config.get("default_profile", "neri-cloud")
    profiles = config.get("profiles", {})
    if target_name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"未找到配置项 '{target_name}'。可用配置: {available}")

    base = dict(profiles[target_name])
    base["name"] = target_name

    # 应用命令行覆盖
    if overrides.get("host"):
        base["host"] = overrides["host"]
    if overrides.get("port"):
        base["port"] = int(overrides["port"])
    if overrides.get("user"):
        base["user"] = overrides["user"]
    if overrides.get("key_path"):
        base["key_path"] = overrides["key_path"]
    if overrides.get("tmux_session"):
        base["tmux_session"] = overrides["tmux_session"]

    if overrides.get("no_forward"):
        base["forward_ports"] = []
    elif overrides.get("forward_ports"):
        base["forward_ports"] = overrides["forward_ports"]
    else:
        base["forward_ports"] = base.get("forward_ports", [])

    return base


def build_ssh_args(profile: dict, extra_cmd: list = None, forward: bool = True) -> list:
    args = ["ssh"]

    # 保持会话活跃，防止长周期模型训练过程中连接中断
    args.extend([
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=3",
        "-o", "TCPKeepAlive=yes",
        "-o", "StrictHostKeyChecking=accept-new"
    ])

    port = profile.get("port", 22)
    if port and port != 22:
        args.extend(["-p", str(port)])

    key_path = profile.get("key_path")
    if key_path and Path(key_path).expanduser().is_file():
        args.extend(["-i", str(Path(key_path).expanduser())])

    if forward and profile.get("forward_ports"):
        for p in profile["forward_ports"]:
            args.extend(["-L", f"{p}:127.0.0.1:{p}"])

    user = profile.get("user")
    host = profile["host"]
    target = f"{user}@{host}" if user else host
    args.append(target)

    if extra_cmd:
        args.extend(extra_cmd)

    return args


def run_ssh(profile: dict, extra_cmd: list = None, with_tmux: bool = False) -> int:
    print(f"\n[SSH] 连接服务器: {profile['name']} ({profile.get('user', '')}@{profile['host']}:{profile.get('port', 22)})")
    ports = profile.get("forward_ports", [])
    if ports:
        print(f"[Tunnel] 已开启端口映射: {', '.join(f'http://localhost:{p}' for p in ports)}")

    final_extra = []
    if with_tmux:
        session_name = profile.get("tmux_session", "model-train")
        print(f"[Tmux] 自动连接/创建训练会话: '{session_name}'")
        final_extra = ["-t", f"tmux new-session -A -s {session_name}"]
    elif extra_cmd:
        final_extra = extra_cmd

    cmd = build_ssh_args(profile, final_extra)
    return subprocess.call(cmd)


def show_gpu_status(profile: dict) -> int:
    print(f"\n[GPU] 查询远程 GPU 显卡状态 ({profile['name']})...")
    check_cmd = (
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "  nvidia-smi; "
        "else "
        "  echo '[提示] 未检测到 nvidia-smi 命令，该服务器可能无 NVIDIA GPU 或未安装显卡驱动。'; "
        "fi"
    )
    cmd = build_ssh_args(profile, ["-t", check_cmd], forward=False)
    return subprocess.call(cmd)


def show_pytorch_env(profile: dict) -> int:
    print(f"\n[Env] 检测远程 Python 与 PyTorch/CUDA 深度学习环境 ({profile['name']})...")
    py_probe = (
        'python3 -c "'
        'import sys; '
        'print(\'=== Python 运行时 ===\'); '
        'print(\'Python 版本:\', sys.version.split()[0]); '
        'try: '
        '  import torch; '
        '  print(\'\\n=== PyTorch 深度学习框架 ===\'); '
        '  print(\'PyTorch 版本   :\', torch.__version__); '
        '  print(\'CUDA 是否可用  :\', torch.cuda.is_available()); '
        '  if torch.cuda.is_available(): '
        '    print(\'CUDA 编译版本  :\', torch.version.cuda); '
        '    print(\'cuDNN 版本     :\', torch.backends.cudnn.version()); '
        '    print(\'GPU 数量       :\', torch.cuda.device_count()); '
        '    for i in range(torch.cuda.device_count()): '
        '      p = torch.cuda.get_device_properties(i); '
        '      print(f\'  [GPU {i}] {p.name} (显存: {round(p.total_memory / (1024**3), 2)} GB)\'); '
        '  else: '
        '    print(\'当前仅支持 CPU 推理\'); '
        'except ImportError: '
        '  print(\'\\n[提示] 未安装 PyTorch (torch) 模块\'); '
        'try: '
        '  import ultralytics; '
        '  print(\'\\n=== YOLO 目标检测 ===\'); '
        '  print(\'Ultralytics 版本:\', ultralytics.__version__); '
        'except ImportError: pass; '
        '"'
    )
    cmd = build_ssh_args(profile, ["-t", py_probe], forward=False)
    return subprocess.call(cmd)


def download_weights(profile: dict, remote_path: str = "", local_path: str = "") -> int:
    print(f"\n[Download] 从远程拉取模型权重/文件 ({profile['name']})")
    if not remote_path:
        default_remote = profile.get("remote_weights_dir", "/root")
        user_input = input(f"请输入远程文件或目录路径 (默认: {default_remote}): ").strip()
        remote_path = user_input if user_input else default_remote

    if not local_path:
        default_local = profile.get("local_weights_dir", "runs")
        user_input = input(f"请输入本地保存目录 (默认: {default_local}): ").strip()
        local_path = user_input if user_input else default_local

    dest_dir = Path(local_path)
    dest_dir.mkdir(parents=True, exist_ok=True)

    scp_cmd = ["scp", "-r", "-C"]
    port = profile.get("port", 22)
    if port and port != 22:
        scp_cmd.extend(["-P", str(port)])
    key_path = profile.get("key_path")
    if key_path and Path(key_path).expanduser().is_file():
        scp_cmd.extend(["-i", str(Path(key_path).expanduser())])

    user = profile.get("user")
    host = profile["host"]
    source_spec = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    scp_cmd.extend([source_spec, str(dest_dir)])

    print(f"[SCP] 执行: {' '.join(scp_cmd)}")
    ret = subprocess.call(scp_cmd)
    if ret == 0:
        print(f"[OK] 文件已成功下载至本地: {dest_dir.resolve()}")
    else:
        print(f"[Error] 下载失败，退出码: {ret}")
    return ret


def upload_weights(profile: dict, local_path: str = "", remote_path: str = "") -> int:
    print(f"\n[Upload] 向远程上传模型权重/数据集/脚本 ({profile['name']})")
    if not local_path:
        local_path = input("请输入要上传的本地文件或目录路径: ").strip()

    local_p = Path(local_path)
    if not local_p.exists():
        print(f"[Error] 本地路径不存在: {local_path}")
        return 1

    if not remote_path:
        default_remote = profile.get("remote_weights_dir", "/root")
        user_input = input(f"请输入远程目标目录 (默认: {default_remote}): ").strip()
        remote_path = user_input if user_input else default_remote

    scp_cmd = ["scp", "-r", "-C"]
    port = profile.get("port", 22)
    if port and port != 22:
        scp_cmd.extend(["-P", str(port)])
    key_path = profile.get("key_path")
    if key_path and Path(key_path).expanduser().is_file():
        scp_cmd.extend(["-i", str(Path(key_path).expanduser())])

    user = profile.get("user")
    host = profile["host"]
    target_spec = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    scp_cmd.extend([str(local_p), target_spec])

    print(f"[SCP] 执行: {' '.join(scp_cmd)}")
    ret = subprocess.call(scp_cmd)
    if ret == 0:
        print(f"[OK] 文件已成功上传至远程: {remote_path}")
    else:
        print(f"[Error] 上传失败，退出码: {ret}")
    return ret


def manage_neri_service(profile: dict) -> int:
    service_name = profile.get("service_name", "neri-models")
    print(f"\n=== Neri 模型分发服务管理 ({service_name}) ===")
    print("1. 查看服务运行状态 (systemctl status)")
    print("2. 重启服务 (sudo systemctl restart)")
    print("3. 实时跟踪最近日志 (journalctl -u ... -f)")
    print("4. 测试公网健康检查接口 (https://myneri.top/api/models/health)")
    print("0. 返回")

    choice = input("请选择操作编号 (0-4): ").strip()
    if choice == "1":
        cmd = build_ssh_args(profile, ["-t", f"systemctl status {service_name}"], forward=False)
        return subprocess.call(cmd)
    elif choice == "2":
        cmd = build_ssh_args(profile, ["-t", f"sudo systemctl restart {service_name} && systemctl status {service_name}"], forward=False)
        return subprocess.call(cmd)
    elif choice == "3":
        cmd = build_ssh_args(profile, ["-t", f"journalctl -u {service_name} -n 50 -f"], forward=False)
        return subprocess.call(cmd)
    elif choice == "4":
        url = "https://myneri.top/api/models/health"
        print(f"请求 {url} ...")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.load(r)
                print("[Health Response]:", json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        except Exception as e:
            print("[Error] 健康检查失败:", e)
            return 1
    return 0


def export_vscode_config(profile: dict, auto_write: bool = False) -> None:
    name = profile["name"]
    host = profile["host"]
    user = profile.get("user", "")
    port = profile.get("port", 22)
    key_path = profile.get("key_path", "")
    forward_ports = profile.get("forward_ports", [])

    lines = [
        f"Host {name}",
        f"    HostName {host}",
    ]
    if user:
        lines.append(f"    User {user}")
    if port and port != 22:
        lines.append(f"    Port {port}")
    if key_path:
        lines.append(f"    IdentityFile {key_path}")
    lines.extend([
        "    ServerAliveInterval 60",
        "    ServerAliveCountMax 3"
    ])
    for p in forward_ports:
        lines.append(f"    LocalForward {p} 127.0.0.1:{p}")

    block = "\n".join(lines)
    print("\n[VS Code Remote-SSH 配置片段]:")
    print("-" * 50)
    print(block)
    print("-" * 50)

    ssh_config_file = Path.home() / ".ssh" / "config"
    write_it = auto_write
    if not write_it:
        ans = input(f"\n是否自动将该配置追加到 {ssh_config_file}? [y/N]: ").strip().lower()
        write_it = ans in ("y", "yes")

    if write_it:
        try:
            ssh_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ssh_config_file, "a", encoding="utf-8") as f:
                f.write(f"\n# Added by Neri Model SSH\n{block}\n")
            print(f"[OK] 已写入 {ssh_config_file}")
            print(f"可在 VS Code 中直接按 F1 -> Remote-SSH: Connect to Host -> 选择 '{name}' 连接！")
        except Exception as e:
            print(f"[提示] 无法自动写入 {ssh_config_file} ({e})。请手动复制上方配置片段粘贴至该文件中。")


def interactive_menu(config: dict, profile_name: str = "") -> None:
    current_profile = profile_name or config.get("default_profile", "neri-cloud")

    while True:
        profile = resolve_profile(config, current_profile)
        print("\n" + "=" * 58)
        print("  Neri 深度学习模型 SSH 远程连接与训练管理")
        print("=" * 58)
        print(f"当前配置      : [{profile['name']}] ({profile.get('description', '')})")
        print(f"连接目标      : {profile.get('user', '')}@{profile['host']}:{profile.get('port', 22)}")
        ports = profile.get("forward_ports", [])
        if ports:
            print(f"端口映射隧道  : {', '.join(map(str, ports))} (本地浏览器直连)")
        print("-" * 58)
        print(" [1] 连接交互式终端 (带端口映射与会话保护)")
        print(" [2] 连接并附加 Tmux 训练会话 (训练防掉线)")
        print(" [3] 查看远程 GPU 显卡状态 (nvidia-smi)")
        print(" [4] 检测远程 PyTorch / CUDA 深度学习环境")
        print(" [5] 从远程下载模型权重 (Pull Checkpoints)")
        print(" [6] 向远程上传模型权重/数据 (Push Files)")
        print(" [7] 管理 Neri 模型分发服务 (Status/Restart/Logs)")
        print(" [8] 导出 VS Code Remote-SSH 连接配置")
        print(" [9] 切换或添加服务器配置")
        print(" [0] 退出")
        print("-" * 58)

        choice = input("请输入操作编号 (0-9): ").strip()
        if choice == "1":
            run_ssh(profile)
        elif choice == "2":
            run_ssh(profile, with_tmux=True)
        elif choice == "3":
            show_gpu_status(profile)
        elif choice == "4":
            show_pytorch_env(profile)
        elif choice == "5":
            download_weights(profile)
        elif choice == "6":
            upload_weights(profile)
        elif choice == "7":
            manage_neri_service(profile)
        elif choice == "8":
            export_vscode_config(profile)
        elif choice == "9":
            profiles = config.get("profiles", {})
            names = list(profiles.keys())
            print("\n可用服务器配置项:")
            for idx, p_name in enumerate(names, 1):
                p_info = profiles[p_name]
                print(f"  [{idx}] {p_name} - {p_info.get('description', '')} ({p_info.get('user', '')}@{p_info['host']})")
            print("  [A] 添加新配置 (例如 AutoDL / 个人工作站)")
            p_sel = input("请输入编号选择或输入 A 添加: ").strip()
            if p_sel.lower() == "a":
                new_name = input("请输入新配置唯一名称 (如 autodl-4090): ").strip()
                if new_name:
                    new_host = input("请输入主机名或 IP: ").strip()
                    new_port = input("请输入 SSH 端口 (默认 22): ").strip()
                    new_port = int(new_port) if new_port else 22
                    new_user = input("请输入 SSH 用户名 (如 root): ").strip()
                    new_desc = input("请输入描述信息: ").strip()
                    new_ports_raw = input("请输入要映射的端口 (逗号分隔，默认 6006,7860,8888): ").strip()
                    new_ports = [int(p.strip()) for p in new_ports_raw.split(",") if p.strip()] if new_ports_raw else [6006, 7860, 8888]
                    config["profiles"][new_name] = {
                        "name": new_name,
                        "description": new_desc,
                        "host": new_host,
                        "port": new_port,
                        "user": new_user,
                        "key_path": "",
                        "remote_model_dir": "/root",
                        "remote_weights_dir": "/root",
                        "local_weights_dir": "runs",
                        "forward_ports": new_ports,
                        "tmux_session": "model-train",
                        "service_name": ""
                    }
                    save_config(config)
                    current_profile = new_name
            elif p_sel.isdigit() and 1 <= int(p_sel) <= len(names):
                current_profile = names[int(p_sel) - 1]
        elif choice == "0":
            break
        else:
            print("[提示] 无效选项，请重新选择。")


def main():
    parser = argparse.ArgumentParser(
        description="Neri 深度学习模型 SSH 远程连接与训练管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/model_ssh.py
      打开交互式终端菜单

  python scripts/model_ssh.py --profile neri-cloud --action connect
      直接连接 Neri 云端模型服务器 (自动映射 8002 端口)

  python scripts/model_ssh.py --profile gpu-training --action gpu-status
      查看远程显卡状态 (nvidia-smi)

  python scripts/model_ssh.py --profile gpu-training --ports 6006 7860 8888 --tmux
      连接 GPU 训练节点，映射 TensorBoard/WebUI/Jupyter 并进入 Tmux 训练会话

  python scripts/model_ssh.py --action download --remote-path /root/autodl-tmp/best.pt --local-path runs/
      从远程下载最佳模型权重到本地
        """
    )
    parser.add_argument("--profile", "-p", default="", help="服务器配置名称 (neri-cloud, gpu-training 等)")
    parser.add_argument("--action", "-a", choices=["connect", "gpu-status", "env-check", "tmux", "download", "upload", "service", "vscode", "list"], default="", help="执行指定动作")
    parser.add_argument("--ports", type=int, nargs="*", default=None, help="自定义本地映射端口列表，如 --ports 6006 7860 8888")
    parser.add_argument("--no-forward", action="store_true", help="禁用所有端口映射")
    parser.add_argument("--tmux", action="store_true", help="自动创建或附加 Tmux 训练保护会话")
    parser.add_argument("--tmux-session", default="", help="指定 Tmux 会话名称")
    parser.add_argument("--remote-path", default="", help="远程模型/权重文件路径 (下载或上传)")
    parser.add_argument("--local-path", default="", help="本地模型/权重文件路径 (下载或上传)")
    parser.add_argument("--host", default="", help="覆盖服务器主机地址/IP")
    parser.add_argument("--port", type=int, default=0, help="覆盖 SSH 端口")
    parser.add_argument("--user", default="", help="覆盖 SSH 用户名")
    parser.add_argument("--key", default="", help="覆盖 SSH 私钥路径")
    parser.add_argument("--command", "-c", default="", help="执行单条远程命令并退出")
    parser.add_argument("--yes", "-y", action="store_true", help="自动确认提示")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="自定义配置文件路径")

    args = parser.parse_args()
    config_file = Path(args.config)
    config = load_config(config_file)

    if args.action == "list":
        profiles = config.get("profiles", {})
        print("\n=== 已配置的服务器列表 ===")
        for p_name, p_info in profiles.items():
            ports_str = ", ".join(map(str, p_info.get("forward_ports", []))) or "无"
            print(f"  * {p_name}: {p_info.get('description', '')}")
            print(f"    目标: {p_info.get('user', '')}@{p_info['host']}:{p_info.get('port', 22)}")
            print(f"    映射端口: {ports_str}")
        return

    if not args.action and not args.profile and not args.command:
        interactive_menu(config, args.profile)
        return

    overrides = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "key_path": args.key,
        "tmux_session": args.tmux_session,
        "no_forward": args.no_forward,
        "forward_ports": args.ports,
    }
    profile = resolve_profile(config, args.profile, overrides)

    if args.command:
        cmd = build_ssh_args(profile, ["-t", args.command], forward=not args.no_forward)
        sys.exit(subprocess.call(cmd))

    action = args.action or "connect"
    if action == "connect":
        sys.exit(run_ssh(profile, with_tmux=args.tmux))
    elif action == "tmux":
        sys.exit(run_ssh(profile, with_tmux=True))
    elif action == "gpu-status":
        sys.exit(show_gpu_status(profile))
    elif action == "env-check":
        sys.exit(show_pytorch_env(profile))
    elif action == "download":
        sys.exit(download_weights(profile, args.remote_path, args.local_path))
    elif action == "upload":
        sys.exit(upload_weights(profile, args.local_path, args.remote_path))
    elif action == "service":
        sys.exit(manage_neri_service(profile))
    elif action == "vscode":
        export_vscode_config(profile, auto_write=args.yes)


if __name__ == "__main__":
    main()
