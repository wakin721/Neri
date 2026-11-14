# Neri 深度学习模型 SSH 远程连接与训练管理指南

本文档介绍专为 **深度学习模型训练、微调与模型服务运维** 打造的 SSH 远程连接工具套件。

包含两个核心脚本与一个配置文件：
- **PowerShell 原生脚本**: `scripts/connect_model_ssh.ps1`（Windows 终端开箱即用，无需第三方库，内置交互式控制台菜单）
- **Python 跨平台脚本**: `scripts/model_ssh.py`（纯标准库，支持 Windows / Linux / macOS，适合自动化脚本与流水线调用）
- **服务器与端口配置文件**: `scripts/model_ssh_config.json`（统一管理服务器节点配置与端口映射预设）

---

## 核心特性与模型场景适配

### 1. 模型服务端口映射隧道 (Port Forwarding)
远程训练或部署模型时，通常需访问各种基于 Web 的可视化与推理服务。本工具默认开启隧道映射（`LocalForward`），在本地浏览器即可直接打开：
- **TensorBoard** (`6006`): 训练 Loss、学习率曲线监控
- **Gradio / WebUI** (`7860`): YOLO / SD / LLM 的 Web 交互界面
- **Streamlit** (`8501`): 交互式数据与模型仪表盘
- **Jupyter Lab / Notebook** (`8888`): 远程交互式模型调试
- **FastAPI / vLLM** (`8000`): 模型推理服务接口
- **Neri Model Service** (`8002`): Neri 官方云端模型分发服务
- **Ollama** (`11434`): 本地大语言模型服务
- **MLflow** (`5000`): 实验追踪

### 2. 训练防掉线会话保护 (Tmux KeepAlive)
模型训练往往耗时数小时至数天：
- **自动 SSH 保活**: 自动注入 `-o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes`，防止路由器或云防火墙因长连接空闲而切断连接。
- **自动 Tmux 训练会话**: 一键进入或恢复远程 `tmux` 会话（如 `model-train`），即使本地电脑断网或关机，远程训练进程仍持续运行。

### 3. 显卡与深度学习环境诊断
- **GPU 状态**: 一键执行并展示远程显卡信息（`nvidia-smi`、显存占用、温度与 CUDA 驱动版本）。
- **PyTorch/YOLO 环境**: 自动探测远程 Python 版本、PyTorch 版本、CUDA 是否可用、可用显卡数量与设备型号，以及 Ultralytics (YOLO) 版本。

### 4. 模型权重 (Weights) 与数据双向同步
- **下载模型权重**: 从远程训练目录（如 `/root/autodl-tmp/best.pt`）一键拉取模型权重、ONNX、safetensors、TensorRT engine 至本地 `runs/` 目录。
- **上传数据/模型**: 一键将本地数据集或模型文件同步至远程指定目录，支持压缩传输加速（`scp -r -C`）。

### 5. Neri 云端模型分发服务专属运维
针对 Neri 官方模型分发服务器（IP `47.108.14.199`）：
- 检查 `neri-models.service` 运行状态
- 查看实时运行日志 (`journalctl -u neri-models -f`)
- 重启服务与验证公共健康接口 (`https://myneri.top/api/models/health`)

### 6. VS Code Remote-SSH 一键配置
一键生成标准 `~/.ssh/config` 配置块，并支持自动追加写入，便于在 VS Code 中直接使用 Remote-SSH 插件进行可视化开发。

---

## 快速上手

### 方式 A: PowerShell 终端（推荐 Windows 用户）

#### 1. 交互式菜单模式
直接运行脚本，出现全功能中文菜单：
```powershell
.\scripts\connect_model_ssh.ps1
```
在菜单中可按 `1-9` 快捷键执行：
- `[1]` 连接交互式终端（带默认端口映射）
- `[2]` 连接并自动附加 Tmux 训练会话
- `[3]` 查看远程 GPU 显卡状态 (`nvidia-smi`)
- `[4]` 诊断远程 PyTorch / CUDA 环境
- `[5]` 从远程下载模型权重 (Pull Checkpoints)
- `[6]` 向远程上传模型权重/数据 (Push Files)
- `[7]` 管理 Neri 模型分发服务
- `[8]` 导出 VS Code Remote-SSH 配置
- `[9]` 切换或添加新服务器节点配置

#### 2. 命令行直连与常用参数
```powershell
# 查看所有已配置服务器节点
.\scripts\connect_model_ssh.ps1 -Action ListProfiles

# 连接默认服务器 (neri-cloud)，映射 8002 端口
.\scripts\connect_model_ssh.ps1 -Profile neri-cloud -Action Connect

# 连接 GPU 训练节点，自定义映射 TensorBoard(6006) 和 WebUI(7860)，并进入 Tmux
.\scripts\connect_model_ssh.ps1 -Profile gpu-training -ForwardPorts 6006,7860 -AutoTmux

# 查看远程 GPU 显卡
.\scripts\connect_model_ssh.ps1 -Profile gpu-training -Action GpuStatus

# 诊断远程 PyTorch 与 CUDA 环境
.\scripts\connect_model_ssh.ps1 -Profile gpu-training -Action EnvCheck

# 从远程拉取模型权重
.\scripts\connect_model_ssh.ps1 -Profile gpu-training -Action DownloadModel -RemotePath "/root/autodl-tmp/best.pt" -LocalPath "runs/best.pt"

# 导出 VS Code 远程连接配置
.\scripts\connect_model_ssh.ps1 -Profile gpu-training -Action VsCodeConfig -Yes
```

---

### 方式 B: Python 命令行

可使用系统 Python 或 Neri 依赖 Python 运行：

```bash
# 打开交互式菜单
python scripts/model_ssh.py

# 查看所有配置节点
python scripts/model_ssh.py --action list

# 查看远程 GPU 状态
python scripts/model_ssh.py --profile gpu-training --action gpu-status

# 诊断远程深度学习环境
python scripts/model_ssh.py --profile gpu-training --action env-check

# 连接并映射端口 6006 7860，自动启动 tmux
python scripts/model_ssh.py --profile gpu-training --ports 6006 7860 --tmux

# 下载模型权重
python scripts/model_ssh.py --profile gpu-training --action download --remote-path /root/autodl-tmp/best.pt --local-path runs/

# 执行单条远程训练指令
python scripts/model_ssh.py --profile gpu-training --command "python -m training.train --epochs 50"
```

---

## 配置文件说明 (`scripts/model_ssh_config.json`)

配置文件预置了两个经典模板：

```json
{
  "default_profile": "neri-cloud",
  "profiles": {
    "neri-cloud": {
      "name": "neri-cloud",
      "description": "Neri 云端模型分发服务器 (Model Distribution & DINOv3)",
      "host": "47.108.14.199",
      "port": 22,
      "user": "codex",
      "key_path": "",
      "remote_model_dir": "/opt/neri-models/server/model_distribution",
      "remote_weights_dir": "/Neri_Data/Model",
      "local_weights_dir": "runs",
      "forward_ports": [8002],
      "tmux_session": "neri-model",
      "service_name": "neri-models"
    },
    "gpu-training": {
      "name": "gpu-training",
      "description": "远程GPU模型训练/微调节点 (AutoDL/云GPU/本地工作站)",
      "host": "127.0.0.1",
      "port": 22,
      "user": "root",
      "key_path": "",
      "remote_model_dir": "/root/models",
      "remote_weights_dir": "/root/autodl-tmp",
      "local_weights_dir": "runs",
      "forward_ports": [6006, 7860, 8888, 8000],
      "tmux_session": "model-train",
      "service_name": ""
    }
  }
}
```

### 接入 AutoDL 或第三方 GPU 云算力机示例
如果使用 AutoDL，通常在控制台会看到类似指令：
`ssh -p 12345 root@region-9.autodl.pro`

只需在 `scripts/model_ssh_config.json` 中添加或通过交互菜单按 `9 -> A` 添加：
```json
"autodl-4090": {
  "name": "autodl-4090",
  "description": "AutoDL RTX 4090 算力机",
  "host": "region-9.autodl.pro",
  "port": 12345,
  "user": "root",
  "key_path": "",
  "forward_ports": [6006, 7860, 8888],
  "tmux_session": "yolo-train"
}
```
保存后即可：
- 运行 `.\scripts\connect_model_ssh.ps1 -Profile autodl-4090 -Action Connect`
- 浏览器访问 `http://localhost:6006` 查看 TensorBoard
- 浏览器访问 `http://localhost:7860` 打开 WebUI
- 运行 `.\scripts\connect_model_ssh.ps1 -Profile autodl-4090 -Action DownloadModel` 拉取训练好的权重
