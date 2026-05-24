<div align="center">
  <img src="res/logo.png" alt="Neri Logo" width="110" height="110">
  <h1>Neri</h1>
  <p><strong>NERI Enables Rapid Identification</strong></p>
  <p>红外相机图像智能处理工具</p>
</div>

<div align="center">

[![Stars](https://img.shields.io/github/stars/wakin721/Neri?style=for-the-badge&label=Stars&color=FFD9DC&labelColor=FFEBEB&logo=github&logoColor=black)](https://github.com/wakin721/Neri)
[![Website](https://img.shields.io/badge/官网-myneri.top-blue?style=for-the-badge&labelColor=E8F4FD&color=B3D9F7&logo=googlechrome&logoColor=black)](https://myneri.top/)
[![License](https://img.shields.io/github/license/wakin721/Neri?style=for-the-badge&colorA=F0FFF0&colorB=C8F0C8&logoColor=black)](LICENSE)
[![Release](https://img.shields.io/github/v/release/wakin721/Neri?style=for-the-badge&colorA=FFF8E8&colorB=FFE8A8&logo=github&logoColor=black)](https://github.com/wakin721/Neri/releases)

</div>

<div align="center">

[中文](README.md) &nbsp;·&nbsp; [English](res/demo/README_en.md) &nbsp;·&nbsp; [更新日志](res/demo/README_Update.md) &nbsp;·&nbsp; [🌐 官网](https://myneri.top/)

</div>

<br>

<p align="center">
  <img src="https://github.com/wakin721/Neri/blob/main/res/demo/demo1.png" width="780px">
</p>

<br>

## 📖 项目简介

**Neri** 是一款专为处理红外相机影像数据设计的智能桌面应用。它基于 **YOLO (You Only Look Once)** 目标检测模型，能够高效、自动地识别和处理大批量由红外相机拍摄的野生动物照片。

本工具旨在为生态保护工作者、野生动物研究人员和爱好者提供一个强大的数据整理和分析平台，将繁琐的手动筛选工作自动化，极大地提升科研和监测效率。

<br>

## ✨ 主要功能

| 功能 | 描述 |
|------|------|
| 🎯 **智能识别** | 采用先进的 YOLO 模型，快速准确地识别图像中的野生动物 |
| 🖼️ **批量处理** | 支持一次性导入整个文件夹的图片或视频，实现全自动化的数据处理流程 |
| 📄 **EXIF 提取** | 自动读取并整合照片的 EXIF 元数据，如拍摄时间等关键信息 |
| 📊 **灵活导出** | 可将识别结果一键导出为 `.csv` 或 `.xlsx` 格式，便于后续统计分析 |
| ⚙️ **高度可定制** | 提供高级选项，允许用户替换或更新 YOLO 模型，适应不同地区和物种的识别需求 |

<br>

## 🚀 快速开始

> 本程序**无需安装**，解压即用。

**第一步：** 前往 [Releases 页面](https://github.com/wakin721/Neri/releases) 下载最新版本的 `.zip` 或 `.7z` 压缩包。

**第二步：** 将下载的压缩包解压到您希望存放程序的任意位置。

**第三步：** 进入解压后的文件夹，双击运行 `Neri.exe` 即可启动程序。

> ⚠️ 首次使用需联网下载相关依赖，后续可断网使用。

<br>

## 💡 使用指南

### 快速使用

```
1. 启动程序后，点击 "选择图片" 按钮，选择包含红外相机照片的文件夹
2. 点击 "开始处理" 按钮，程序将自动开始批量识别图像中的动物，并在界面上实时显示进度
3. 识别完成后，点击 "校验检验" 按钮，可以详细查看每张照片的识别结果，并根据识别情况进行纠正
4. 在预览页面，点击 "导出" 即可将所有分析数据保存为 .xlsx 或 .csv 文件
```

---

### 高级设置

高级设置分为四个分组框：**模型参数设置**、**视频检测设置**、**环境维护**和**软件设置**。

#### 🔧 模型参数设置

**模型管理**

- 模型目录位于 `res/model` 以及 `res/cls_model` 文件夹下
- 目前仅支持 `.pt` 结尾的模型文件，可自行更换

**检测阈值设置**

| 参数 | 说明 |
|------|------|
| **IOU 阈值** | 控制 NMS 的重叠阈值。较高的值会减少重叠框，但可能导致部分目标漏检 |
| **置信度阈值** | 检测对象的最小置信度分数。较高的值减少误检，但可能漏掉低置信度的真实目标 |

**加速与高级选项**

| 选项 | 说明 |
|------|------|
| **FP16 加速** | 使用半精度浮点数进行推理，加快速度但可能略微降低精度（需要兼容的 NVIDIA GPU） |
| **数据增强 (TTA)** | 通过多种变换综合结果提高准确性，但会显著降低处理速度 |
| **类别无关 NMS** | 在所有类别上一起执行 NMS，对检测多种相互重叠的物种可能有用 |

#### 🎬 视频检测设置

| 参数 | 说明 |
|------|------|
| **帧间隔** | 设置视频处理时的跳帧间隔，以加快处理视频的速度 |
| **最低帧数比例** | 若某个目标在视频中出现的总帧数占视频总帧数低于此值，则该目标将被视为误检，不会在结果中显示 |

---

### 使用 NVIDIA CUDA / Intel XPU 加速

> 建议（但非强制）配备 NVIDIA GPU，以使用更高精度的模型并获得更快的推理速度。

**自动检测逻辑：**

```
首次运行时自动检测
  ├── 检测到 NVIDIA GPU → 自动安装对应 CUDA 版本的 PyTorch
  ├── 未检测到 NVIDIA → 继续检测 Intel GPU
  │     ├── 支持的 Intel Arc 独显 / Core Ultra 平台 → 安装 XPU 版本
  │     └── 其他 Intel 显卡 → 回退到 CPU 版本
  └── 安装失败 → 可尝试手动安装
```

> 使用 Intel GPU 时，请确保已安装 Intel 显卡驱动：[Intel GPU 驱动安装页面](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
>
> 参考：[Intel GPU PyTorch 官方入门指南](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)

**如何查看 CUDA 版本并安装对应 PyTorch：**

**步骤 1：** 右键点击桌面，选择 "NVIDIA 控制面板"

**步骤 2：** 在菜单栏选择 "帮助" → "系统信息"

<p align="center">
  <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda1.png" width="720px">
</p>

**步骤 3：** 在弹出的窗口中找到 "NVCUDA64.DLL" 相关文件，即可查看对应的 CUDA 版本

<p align="center">
  <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda2.png">
</p>

**步骤 4：** 在 **高级设置 → 环境维护 → 安装 PyTorch** 中选择对应版本安装

<p align="center">
  <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda3.png" width="720px">
</p>

**模型选择建议：**

| 硬件配置 | 推荐模型 | 说明 |
|----------|----------|------|
| 仅 CPU | `11n` / `11s` | 防止推理过慢 |
| 配备 GPU | `11x` | 获得最高的推理精度 |

<br>

## 🗺️ 未来蓝图

- [x] 引入更高效的 YOLO 模型版本，进一步提升识别速度和准确率（已加入对 YOLO26 的支持）
- [x] 增加对红外相机视频数据的识别和分析功能
- [x] 支持更多样化的数据筛选和标记功能
- [ ] 开发物种丰度、活动规律等自动化数据分析和图表生成模块

<br>

## 🧪 Flutter Material 3 前端与 Python 后端（实验性）

本分支新增了面向跨平台客户端的前后端拆分实现：

- **`system/backend/`** — 基于 FastAPI 的 Python 后端，复用现有 `system/` 中的配置、EXIF 元数据提取和 YOLO 图像处理模块
- **`frontend/`** — 基于 Flutter 的 Material 3 前端，使用 NavigationRail 区分各界面，支持进度轮询、路径记忆、模型文件夹自动识别和自适应图像预览排布

### 启动 Python 后端

```bash
python -m pip install -r requirements.txt
python -m uvicorn system.backend.main:app --reload --app-dir . --host 127.0.0.1 --port 721
```

后端文档地址：<http://127.0.0.1:721/docs>

### 启动 Flutter 前端

```bash
cd frontend
flutter pub get
flutter run
```

> 默认连接 `http://127.0.0.1:721`。当前前端使用本机文件夹路径提交任务，桌面端运行体验最佳。

<br>

## ⚠️ 注意事项

> 项目仍处于开发中，请勿过分依赖本项目的识别结果。程序内置的模型仅供测试使用，暂不提供更多物种的识别。

<br>

## 📧 联系我们

**作者：** 和錦わきん

**官网：** [https://myneri.top/](https://myneri.top/)

如果您在使用过程中遇到任何问题或有任何建议，欢迎通过 [Issues](https://github.com/wakin721/Neri/issues) 与我们联系。

<br>

## 🙏 鸣谢

感谢 **北纬44度的Suger** 为本项目提供 Logo

<br>

---

<div align="center">
  <sub>Made with ❤️ by 和錦わきん &nbsp;·&nbsp; <a href="https://myneri.top/">myneri.top</a></sub>
</div>
