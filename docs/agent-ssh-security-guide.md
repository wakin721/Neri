# AI 模型 SSH 远程连接安全方案 (防止模型接触密码)

本指南针对在 AI 编码助手 / 大语言模型 (LLM) / 自动化 Agent 执行环境中进行远程服务器运维的场景，提供一套 **从技术原理上彻底杜绝模型接触、索取、存储或泄露服务器密码** 的免密安全解决方案。

---

## 1. 为什么必须阻止 AI 模型接触服务器密码？

在日常开发中，如果让 AI 模型直接参与基于密码的远程 SSH 交互，存在严重的系统性安全隐患：

1. **对话上下文与 Token 外泄 (Token & Log Exposure)**：
   如果脚本要求输入密码，用户往往会在聊天窗口输入密码。该密码会被写入对话历史、云端 Token 缓存、API 调用日志和客户端本地会话记录，极易被多轮 Prompt 窃取或日志留痕。
2. **脚本明文泄露 (Credential Leakage in Files)**：
   使用 `sshpass -p` 或在配置文件（如 `.json`、`.yaml`、`.env`）中写入明文密码，拥有文件读取权限的 AI 模型可直接读取该文件并泄露其内容。
3. **进程挂起与超时 (Terminal Hangs)**：
   AI 模型在非交互式（Headless）终端中运行时，任何交互式密码提示（如 `user@host's password:`）都会导致 SSH 阻塞，最终因超时导致任务失败。
4. **毁灭性命令误操作 (Destructive Command Hallucination)**：
   模型若拥有不受约束的高权限账户连接，可能会因幻觉执行误操作（如 `rm -rf /` 或格式化磁盘）。

---

## 2. 零密码（Zero-Password）安全架构设计

```
+--------------------------------------------------------------------------+
|                        管理员本地计算机 (Windows/Linux)                    |
|                                                                          |
|  【人类管理员】(执行一次)                                                  |
|    .\scripts\setup_agent_ssh.ps1                                         |
|          |                                                               |
|          +---> 生成专属密钥: ~/.ssh/id_ed25519_agent (无密码私钥)         |
|          +---> 人工在本地安全终端输入一次密码，将公钥推送到远程 authorized_keys|
|          +---> 自动写入 ~/.ssh/config 别名与 model_ssh_config.json        |
|                                                                          |
|  【AI 模型 / Agent】(日常执行)                                            |
|    python scripts/agent_ssh.py --action exec -c "..." --json             |
|          |                                                               |
|          +---> 强制注入: -o BatchMode=yes -o PasswordAuthentication=no   |
|          +---> 危险命令拦截 (rm -rf /, mkfs, 覆盖 authorized_keys 等)     |
|          +---> 密钥免密鉴权 (完全不经过任何密码流程)                       |
|          +---> 超时控制与格式化 JSON 结果解析                             |
+--------------------------------------------------------------------------+
                                     | (SSH / SCP 端口 22)
                                     v
+--------------------------------------------------------------------------+
|                        远程 Linux 服务器 (GPU / 云端)                     |
|                                                                          |
|  ~/.ssh/authorized_keys:                                                 |
|    no-port-forwarding ssh-ed25519 AAAAC3... neri-agent                  |
|                                                                          |
|  系统权限限制 (可选加固):                                                 |
|    独立低权限用户 codex-agent，仅配置必要的白名单 sudo 权限                |
+--------------------------------------------------------------------------+
```

### 核心安全防线

| 防线 | 技术手段 | 防御效果 |
| :--- | :--- | :--- |
| **防线 1：强制免密锁** | `-o BatchMode=yes -o PasswordAuthentication=no` | 彻底切断密码交互通道。公钥未就绪时立即报错退出，绝不给模型任何索要或输入密码的机会 |
| **防线 2：人机密钥隔离** | 独立生成 `id_ed25519_agent` 专属密钥 | 不与人类管理员的主密钥混用，随时可在服务端注销该单项公钥 |
| **防线 3：密码零存储** | 配置文件和代码中无任何 `password` 字段 | 即使模型读取本地所有脚本和配置文件，也无法获得任何服务器凭证 |
| **防线 4：本地指令护栏** | 正则匹配拦截高危命令 | 本地实时阻断 `rm -rf /`、磁盘格式化、篡改 `authorized_keys`、关机等操作 |
| **防线 5：超时与结构化** | 默认 60s 超时与 `--json` 输出 | 防止模型会话无限挂起，保证智能体精准感知执行结果 |

---

## 3. 落地实施指南

### 步骤一：人类管理员一键安全初始化 (只需执行 1 次)

由**人类管理员**打开自己的本地 PowerShell 终端，执行初始化脚本：

```powershell
.\scripts\setup_agent_ssh.ps1
```

该向导会自动执行以下动作：
1. 自动生成专属于 AI 模型的 Ed25519 密钥对（`~/.ssh/id_ed25519_agent`）；
2. 通过原生 OpenSSH 将公钥安全推送到远程服务器（默认 `codex@47.108.14.199:22`，支持自定义）；
3. **在此步骤中，终端由 OpenSSH 提示管理员输入一次服务器密码**（密码由操作系统内核终端捕获，不写入任何日志或文件，AI 模型完全隔离）；
4. 自动更新 `scripts/model_ssh_config.json` 中的 `key_path`；
5. 自动配置 `~/.ssh/config` 安全主机别名；
6. 以强制免密模式（`BatchMode=yes`）进行回环验证，确保初始化成功。

---

### 步骤二：AI 模型在日常任务中安全调用

配置完成后，AI 模型可以通过 `scripts/agent_ssh.py` 或 `scripts/agent_ssh.ps1` 自由执行远程任务，完全无需任何密码。

#### 1. 连通性测试 (Test)
```bash
# Python 版本 (跨平台，推荐大模型调用)
python scripts/agent_ssh.py --action test --json

# PowerShell 原生版本
.\scripts\agent_ssh.ps1 -Action Test -Json
```
**返回示例**：
```json
{
  "action": "test",
  "profile": "neri-cloud",
  "target": "codex@47.108.14.199:22",
  "key_path": "C:\\Users\\wakin\\.ssh\\id_ed25519_agent",
  "success": true,
  "exit_code": 0,
  "duration_seconds": 0.42,
  "latency_ms": 420,
  "message": "SSH 免密连接成功！模型可在无密码暴露状态下安全执行运维命令。"
}
```

#### 2. 安全执行远程命令 (Exec)
```bash
# 查看 GPU 显卡状态
python scripts/agent_ssh.py --action exec -c "nvidia-smi" --json

# 检查远程磁盘空间
python scripts/agent_ssh.py --action exec -c "df -h" --json

# 查看服务进程
python scripts/agent_ssh.py --action exec -c "ps aux | grep neri" --json
```

#### 3. 危险指令自动拦截验证 (Guardrails Demo)
如果模型因误操作生成了危险指令：
```bash
python scripts/agent_ssh.py --action exec -c "rm -rf /" --json
```
**拦截结果**：
```json
{
  "action": "exec",
  "command": "rm -rf /",
  "success": false,
  "blocked": true,
  "error": "[安全拦截] 禁止执行毁灭性危险指令: 尝试递归强制删除系统根目录 (rm -rf /)"
}
```
指令在本地直接被拦截，根本不会发送到远程服务器！

#### 4. 远程环境一键探测 (Env)
自动检测远程操作系统、显卡型号、驱动版本、CUDA 及 PyTorch 深度学习运行时：
```bash
python scripts/agent_ssh.py --action env --json
```

#### 5. 文件与模型权重安全传输 (Upload / Download)
```bash
# 上传本地模型权重到远程 (全程 BatchMode 免密传输)
python scripts/agent_ssh.py --action upload --local runs/best.pt --remote /Neri_Data/Model/

# 从远程下载最佳检查点到本地
python scripts/agent_ssh.py --action download --remote /Neri_Data/Model/best.pt --local runs/
```

---

## 4. 远程服务器权限最小化加固 (建议管理员选配)

为了达到企业级最高安全标准，建议在 Linux 服务器端对模型使用的账户进行权限限制：

### 方案 A：在 `authorized_keys` 中限制模型公钥功能
在远程主机的 `~/.ssh/authorized_keys` 中，在模型专属公钥前加入限制前缀：
```text
no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... neri-agent
```
这样该密钥将无法被用于端口转发跳板或 X11 转发。

### 方案 B：创建专用的低权限系统用户
避免让模型直接使用 `root` 账户登录，可在服务器上创建专门的 `agent` 用户：
```bash
# 1. 创建无密码低权限账户
sudo adduser --disabled-password --gecos "AI Agent Worker" codex-agent

# 2. 设置公钥认证目录
sudo mkdir -p /home/codex-agent/.ssh
sudo chmod 700 /home/codex-agent/.ssh
sudo cp ~/.ssh/authorized_keys /home/codex-agent/.ssh/
sudo chown -R codex-agent:codex-agent /home/codex-agent/.ssh

# 3. (可选) 配置 sudo 白名单命令 (仅允许重启或查询特定模型服务，禁止提权 bash)
# 编辑 /etc/sudoers.d/codex-agent:
# codex-agent ALL=(ALL) NOPASSWD: /bin/systemctl status neri-models, /bin/systemctl restart neri-models
```

---

## 5. 文件索引

| 文件路径 | 用途 | 适用角色 |
| :--- | :--- | :--- |
| `scripts/setup_agent_ssh.ps1` | 生成专属密钥并向远程部署公钥的一键向导 | **人类管理员** (运行一次) |
| `scripts/agent_ssh.py` | 跨平台免密安全执行脚本 (命令拦截、超时保护、JSON 输出) | **AI 模型 / 自动化脚本** |
| `scripts/agent_ssh.ps1` | Windows PowerShell 原生免密安全执行脚本 | **AI 模型 / Windows 用户** |
| `scripts/model_ssh_config.json` | 服务器节点与端口映射统一配置表 (无任何密码字段) | 共享配置 |
| `docs/agent-ssh-security-guide.md` | 本安全方案文档 | 参考查阅 |
