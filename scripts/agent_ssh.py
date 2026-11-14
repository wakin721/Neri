#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/agent_ssh.py - AI 模型专用 SSH 远程安全连接与执行脚本

【设计目标】
专门供 AI 智能体 / 大语言模型 (LLM) 在终端中调用，安全地对远程服务器进行运维、
调试和文件传输，从技术原理上彻底防止模型接触、索取、存储或泄露服务器密码。

【核心安全防线】
1. 强制免密锁 (BatchMode Enforced):
   注入 -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no
   彻底关闭密码交互通道。若公钥认证未就绪，立即退出并报警，绝不触发密码输入，
   也绝不给模型任何索要或输入密码的机会。
2. 零密码存储 (Zero Password Storage):
   本脚本及配置文件均无 password 字段，只接受 Ed25519/RSA 密钥或 OpenSSH Host 别名。
3. 毁灭性命令防御拦截 (Destructive Command Guardrails):
   内置危险指令特征过滤库，主动拦截 rm -rf /、格式化磁盘、破坏 authorized_keys 等误操作。
4. 结构化 JSON 输出 (--json):
   支持输出格式化 JSON，便于 AI 模型和自动化 Agent 准确解析 exit_code、stdout 与 stderr。
5. 执行超时保护 (--timeout):
   默认 60 秒超时，防止交互式命令或挂起进程导致模型会话无限等待。
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 默认配置文件路径
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "model_ssh_config.json"

# 毁灭性危险命令特征黑名单 (直接拒绝执行，保护远程系统安全)
CRITICAL_DANGEROUS_PATTERNS = [
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(?:/\s*$|/\*|\s+/\s+)", "尝试递归强制删除系统根目录 (rm -rf /)"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+~\s*$", "尝试递归强制删除用户主目录 (rm -rf ~)"),
    (r"\bmkfs(?:\.[a-zA-Z0-9]+)?\b", "尝试格式化磁盘分区 (mkfs)"),
    (r"\bdd\s+if=/dev/(?:zero|urandom)\s+of=/dev/[a-z0-9]+", "尝试使用 dd 覆盖物理磁盘设备"),
    (r"(?:>|>>)\s*(?:~|/home/[^/]+|/root)/\.ssh/authorized_keys", "尝试修改或覆盖远程 authorized_keys 密钥文件"),
    (r"\brm\s+.*(?:~|/home/[^/]+|/root)/\.ssh", "尝试删除远程 .ssh 密钥目录"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "检测到 Fork 炸弹恶意代码"),
    (r"\b(?:shutdown|poweroff|halt|init\s+0)\b", "尝试关闭远程服务器电源"),
]

# 常见需要警告的高危命令
WARNING_DANGEROUS_PATTERNS = [
    (r"\breboot\b", "系统重启命令"),
    (r"\bkillall\s+-9\b", "强制杀死所有匹配进程"),
    (r"\biptables\s+-F\b", "清空防火墙规则可能导致失联"),
]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """加载服务器配置文件，若不存在则返回内置默认配置"""
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            sys.stderr.write(f"[WARN] 无法解析配置文件 {config_path}: {e}\n")
    return {
        "default_profile": "neri-cloud",
        "profiles": {
            "neri-cloud": {
                "name": "neri-cloud",
                "host": "47.108.14.199",
                "port": 22,
                "user": "codex",
                "key_path": "",
                "description": "Neri 官方云端模型分发服务器",
            }
        }
    }


def find_agent_key() -> Optional[str]:
    """在标准位置自动探测专门给 Agent 使用的 SSH 密钥"""
    home = Path.home()
    candidates = [
        home / ".ssh" / "id_ed25519_agent",
        home / ".ssh" / "id_ed25519_neri_agent",
        home / ".ssh" / "id_rsa_agent",
        SCRIPT_DIR / "id_ed25519_agent",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except (PermissionError, OSError):
            pass
    return None


def resolve_profile(config: dict, name: str = "", overrides: dict = None) -> dict:
    """解析目标服务器配置，并应用命令行覆盖参数"""
    overrides = overrides or {}
    target_name = name or config.get("default_profile", "neri-cloud")
    profiles = config.get("profiles", {})

    if target_name in profiles:
        base = dict(profiles[target_name])
    else:
        # 如果未在配置中找到，允许直接使用主机名或别名
        base = {
            "name": target_name,
            "host": target_name,
            "port": 22,
            "user": "",
            "key_path": "",
            "description": "自定义主机配置",
        }

    base["name"] = target_name
    if overrides.get("host"):
        base["host"] = overrides["host"]
    if overrides.get("port"):
        base["port"] = int(overrides["port"])
    if overrides.get("user"):
        base["user"] = overrides["user"]
    if overrides.get("key_path"):
        base["key_path"] = overrides["key_path"]

    # 若未指定密钥，自动检查是否存在专用的 agent 密钥
    if not base.get("key_path"):
        auto_key = find_agent_key()
        if auto_key:
            base["key_path"] = auto_key

    return base


def audit_command(command: str, force: bool = False) -> Tuple[bool, str]:
    """对模型要执行的远程命令进行安全审计，防止误操作或恶意操作"""
    for pattern, desc in CRITICAL_DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"[安全拦截] 禁止执行毁灭性危险指令: {desc} (匹配规则: {pattern})"

    if not force:
        for pattern, desc in WARNING_DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"[安全警告] 该指令存在较高运维风险: {desc}。若确认执行请追加 --force 参数。"

    return True, ""


def build_safe_ssh_args(profile: dict, extra_args: List[str] = None) -> List[str]:
    """
    构造供模型使用的强制免密安全 SSH 命令参数列表
    核心保障：强制开启 BatchMode，禁止密码提示与键盘交互
    """
    ssh_exe = shutil.which("ssh") or "ssh"
    args = [
        ssh_exe,
        # 强制非交互模式：认证失败立即退出，绝不提示密码
        "-o", "BatchMode=yes",
        # 彻底关闭密码认证
        "-o", "PasswordAuthentication=no",
        # 彻底关闭键盘交互认证
        "-o", "KbdInteractiveAuthentication=no",
        # 自动接受新主机的公钥指纹，防止模型首次连接被交互提示挂起
        "-o", "StrictHostKeyChecking=accept-new",
        # 连接超时 (秒)
        "-o", "ConnectTimeout=10",
        # 保活心跳
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]

    port = profile.get("port", 22)
    if port and port != 22:
        args.extend(["-p", str(port)])

    key_path = profile.get("key_path")
    if key_path:
        try:
            expanded = Path(key_path).expanduser()
            is_file = expanded.is_file()
        except (PermissionError, OSError):
            is_file = True
            expanded = Path(key_path)

        if is_file:
            args.extend(["-i", str(expanded)])
        else:
            sys.stderr.write(f"[WARN] 指定的密钥文件不存在: {expanded}\n")

    user = profile.get("user")
    host = profile["host"]
    target = f"{user}@{host}" if user else host
    args.append(target)

    if extra_args:
        args.extend(extra_args)

    return args


def build_safe_scp_args(profile: dict) -> List[str]:
    """构造安全 SCP 命令基础参数"""
    scp_exe = shutil.which("scp") or "scp"
    args = [
        scp_exe,
        "-r",
        "-C",  # 压缩传输
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
    ]

    port = profile.get("port", 22)
    if port and port != 22:
        args.extend(["-P", str(port)])

    key_path = profile.get("key_path")
    if key_path:
        try:
            expanded = Path(key_path).expanduser()
            is_file = expanded.is_file()
        except (PermissionError, OSError):
            is_file = True
            expanded = Path(key_path)

        if is_file:
            args.extend(["-i", str(expanded)])

    return args


def run_command(args: List[str], timeout: Optional[int] = 60) -> Tuple[int, str, str, float]:
    """在子进程中安全执行命令并捕获超时与输出"""
    start_time = time.time()
    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        duration = round(time.time() - start_time, 2)
        return proc.returncode, stdout, stderr, duration
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        duration = round(time.time() - start_time, 2)
        return -1, stdout, f"[TIMEOUT] 执行超过设定的超时时间 ({timeout}s)，进程已被强制终止。", duration
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        return -2, "", f"[EXCEPTION] 执行异常: {str(e)}", duration


def explain_auth_failure(profile: dict, stderr: str) -> str:
    """针对模型提供清晰友好的认证失败与防密码暴露解释"""
    key_info = profile.get("key_path") or "未显式指定密钥 (使用系统默认/ssh-agent)"
    return (
        f"[安全锁定 - 免密认证失败]\n"
        f"目标服务器: {profile.get('user', '')}@{profile['host']}:{profile.get('port', 22)}\n"
        f"当前密钥: {key_info}\n"
        f"底层报错: {stderr.strip()}\n\n"
        f"【核心防护说明】\n"
        f"本脚本为 AI 模型专用，已强制开启 BatchMode 禁用密码交互，防止模型接触、索取或泄露密码。\n"
        f"请勿在对话中提供密码！请通知人类管理员在本地终端运行以下脚本完成公钥授权：\n"
        f"  PowerShell: .\\scripts\\setup_agent_ssh.ps1\n"
    )


def action_test(profile: dict, timeout: int = 15) -> Dict[str, Any]:
    """测试 SSH 免密连通性并获取基本响应"""
    test_cmd = "echo '__NERI_SSH_TEST_OK__'"
    args = build_safe_ssh_args(profile, [test_cmd])
    code, stdout, stderr, duration = run_command(args, timeout=timeout)

    success = (code == 0 and "__NERI_SSH_TEST_OK__" in stdout)
    result = {
        "action": "test",
        "profile": profile["name"],
        "target": f"{profile.get('user', '')}@{profile['host']}:{profile.get('port', 22)}",
        "key_path": profile.get("key_path", ""),
        "success": success,
        "exit_code": code,
        "duration_seconds": duration,
        "latency_ms": int(duration * 1000),
    }

    if success:
        result["message"] = "SSH 免密连接成功！模型可在无密码暴露状态下安全执行运维命令。"
    else:
        result["error"] = explain_auth_failure(profile, stderr)

    return result


def action_exec(profile: dict, command: str, timeout: int = 60, force: bool = False) -> Dict[str, Any]:
    """在远程服务器上安全执行单条指令"""
    if not command.strip():
        return {
            "action": "exec",
            "success": False,
            "error": "未提供要执行的命令 (-c / --command)。"
        }

    # 执行命令安全审计
    safe, reason = audit_command(command, force=force)
    if not safe:
        return {
            "action": "exec",
            "command": command,
            "success": False,
            "blocked": True,
            "error": reason
        }

    args = build_safe_ssh_args(profile, [command])
    code, stdout, stderr, duration = run_command(args, timeout=timeout)

    result = {
        "action": "exec",
        "profile": profile["name"],
        "command": command,
        "exit_code": code,
        "success": (code == 0),
        "duration_seconds": duration,
        "stdout": stdout,
        "stderr": stderr
    }

    # 针对 255 退出码进行专门诊断 (通常是 SSH 认证失败)
    if code == 255 and ("Permission denied" in stderr or "timed out" in stderr):
        result["auth_error"] = explain_auth_failure(profile, stderr)

    return result


def action_env(profile: dict, timeout: int = 30) -> Dict[str, Any]:
    """探测远程系统基础环境、GPU 显卡以及 Python 深度学习运行时"""
    probe_script = (
        "echo '=== HOST INFO ==='; "
        "hostname; uname -a; whoami; uptime; "
        "echo '=== GPU INFO ==='; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "  nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,driver_version --format=csv,noheader; "
        "else "
        "  echo 'NO_NVIDIA_GPU'; "
        "fi; "
        "echo '=== PYTHON & PYTORCH ==='; "
        "python3 -c 'import sys; print(\"Python:\", sys.version.split()[0]); "
        "exec(\"try:\\n  import torch\\n  print(\\\"PyTorch:\\\", torch.__version__)\\n  print(\\\"CUDA:\\\", torch.cuda.is_available())\\nexcept Exception as e:\\n  print(\\\"PyTorch:\\\", e)\")' 2>&1"
    )
    return action_exec(profile, probe_script, timeout=timeout)


def action_upload(profile: dict, local_path: str, remote_path: str, timeout: int = 300) -> Dict[str, Any]:
    """向远程服务器上传文件或目录"""
    local_p = Path(local_path).resolve()
    if not local_p.exists():
        return {
            "action": "upload",
            "success": False,
            "error": f"本地路径不存在: {local_p}"
        }

    scp_args = build_safe_scp_args(profile)
    user = profile.get("user")
    host = profile["host"]
    target_spec = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    scp_args.extend([str(local_p), target_spec])

    code, stdout, stderr, duration = run_command(scp_args, timeout=timeout)
    return {
        "action": "upload",
        "profile": profile["name"],
        "local_path": str(local_p),
        "remote_path": remote_path,
        "success": (code == 0),
        "exit_code": code,
        "duration_seconds": duration,
        "stdout": stdout,
        "stderr": stderr
    }


def action_download(profile: dict, remote_path: str, local_path: str, timeout: int = 300) -> Dict[str, Any]:
    """从远程服务器下载文件或目录"""
    dest_dir = Path(local_path).resolve()
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    scp_args = build_safe_scp_args(profile)
    user = profile.get("user")
    host = profile["host"]
    src_spec = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
    scp_args.extend([src_spec, str(dest_dir)])

    code, stdout, stderr, duration = run_command(scp_args, timeout=timeout)
    return {
        "action": "download",
        "profile": profile["name"],
        "remote_path": remote_path,
        "local_path": str(dest_dir),
        "success": (code == 0),
        "exit_code": code,
        "duration_seconds": duration,
        "stdout": stdout,
        "stderr": stderr
    }


def action_list(config: dict) -> Dict[str, Any]:
    """列出所有预设服务器配置信息"""
    profiles = config.get("profiles", {})
    summary = []
    for k, v in profiles.items():
        summary.append({
            "name": k,
            "description": v.get("description", ""),
            "host": v.get("host", ""),
            "port": v.get("port", 22),
            "user": v.get("user", ""),
            "has_key": bool(v.get("key_path")),
            "key_path": v.get("key_path", ""),
        })
    return {
        "action": "list",
        "default_profile": config.get("default_profile", ""),
        "profiles": summary
    }


def main():
    parser = argparse.ArgumentParser(
        description="AI 模型专用 SSH 远程安全连接与执行脚本 (零密码接触/安全沙箱)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
【模型调用示例】:
  1. 测试免密连通性 (不会弹出密码输入):
     python scripts/agent_ssh.py --action test

  2. 远程执行命令 (安全拦截 + 超时防护):
     python scripts/agent_ssh.py --action exec -c "nvidia-smi"

  3. 结构化 JSON 输出 (推荐大模型调用):
     python scripts/agent_ssh.py --action exec -c "df -h" --json

  4. 探测远程环境 (GPU / PyTorch / CUDA):
     python scripts/agent_ssh.py --action env --json

  5. 上传与下载文件:
     python scripts/agent_ssh.py --action upload --local runs/best.pt --remote /Neri_Data/Model/
     python scripts/agent_ssh.py --action download --remote /Neri_Data/Model/best.pt --local runs/
        """
    )
    parser.add_argument("--action", "-a", choices=["test", "exec", "env", "upload", "download", "list"], default="test", help="要执行的操作类型")
    parser.add_argument("--profile", "-p", default="", help="目标服务器配置名称 (默认使用 neri-cloud)")
    parser.add_argument("--command", "-c", default="", help="要执行的远程命令 (用于 exec 操作)")
    parser.add_argument("--local", default="", help="本地路径 (用于 upload/download 操作)")
    parser.add_argument("--remote", default="", help="远程路径 (用于 upload/download 操作)")
    parser.add_argument("--timeout", type=int, default=60, help="执行超时时间 (秒，默认 60s)")
    parser.add_argument("--force", action="store_true", help="忽略非毁灭性高危命令警告")
    parser.add_argument("--json", action="store_true", help="以格式化 JSON 形式输出结果 (推荐模型使用)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="指定服务器配置文件路径")

    parser.add_argument("--host", default="", help="覆盖服务器主机名/IP")
    parser.add_argument("--port", type=int, default=0, help="覆盖 SSH 端口")
    parser.add_argument("--user", default="", help="覆盖 SSH 用户名")
    parser.add_argument("--key", default="", help="覆盖 SSH 私钥路径")

    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.action == "list":
        res = action_list(config)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(f"默认服务器配置: {res['default_profile']}")
            for p in res["profiles"]:
                key_status = f"(密钥: {p['key_path']})" if p['has_key'] else "(未指定独立私钥)"
                print(f" - [{p['name']}] {p['user']}@{p['host']}:{p['port']} {key_status} : {p['description']}")
        return 0

    overrides = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "key_path": args.key,
    }
    profile = resolve_profile(config, args.profile, overrides)

    result = {}
    if args.action == "test":
        result = action_test(profile, timeout=args.timeout)
    elif args.action == "exec":
        result = action_exec(profile, args.command, timeout=args.timeout, force=args.force)
    elif args.action == "env":
        result = action_env(profile, timeout=args.timeout)
    elif args.action == "upload":
        result = action_upload(profile, args.local, args.remote, timeout=args.timeout)
    elif args.action == "download":
        result = action_download(profile, args.remote, args.local, timeout=args.timeout)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if not result.get("success", False):
            print(f"[FAILED] 操作 '{args.action}' 失败 (退出码: {result.get('exit_code', -1)})")
            if result.get("error"):
                print(result["error"])
            if result.get("auth_error"):
                print(result["auth_error"])
            if result.get("stderr"):
                print("\n[STDERR]:\n" + result["stderr"].strip())
        else:
            print(f"[SUCCESS] 操作 '{args.action}' 执行成功 (耗时: {result.get('duration_seconds', 0)}s)")
            if result.get("message"):
                print(result["message"])
            if result.get("stdout"):
                print("\n[STDOUT]:\n" + result["stdout"].strip())

    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
