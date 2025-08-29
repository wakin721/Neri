<div align="left">
<a href="/README.md">中文</a>&nbsp;|&nbsp;
<a href="/res/demo/README_en.md">English</a> &nbsp;|&nbsp;
<a href="/res/demo/README_Update.md">Changelog</a> &nbsp;
</div>

<div align="center">
<img src="res/logo.png" alt="Logo" width="120" height="120">
<h1 align="center">Neri - Infrared Camera Image Intelligent Processing Tool</h1>
</div>

<div align="center">

</div>
</div>

</p>
<p align="center">
 <img src="https://img.shields.io/github/stars/wakin721/Neri?style=for-the-badge&colorA=FFEBEB&colorB=FFD9DC&logo=github&logoColor=black">
  </a>
</p>

### 📖 Project Introduction
Neri (NERI Enables Rapid Identification) is an intelligent desktop application specifically designed for processing infrared camera image data. Based on the YOLO (You Only Look Once) object detection model, it can efficiently and automatically identify wildlife in images.

<p align="center">
   <img src="https://github.com/wakin721/Neri/blob/main/res/demo/demo1.png" width="750px">
</p>

## Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [User Guide](#-user-guide)
  - [Quick Usage](#quick-usage)
  - [Advanced Settings](#advanced-settings)
  - [Using CUDA Acceleration](#using-cuda-acceleration)
- [Future Roadmap](#%EF%B8%8F-future-roadmap)
- [Warning](#%EF%B8%8F-warning)
- [Contact Us](#-contact-us)
- [Acknowledgments](#acknowledgments)

## ✨ Key Features

🎯 **YOLO-based Intelligent Recognition**: Utilizes advanced YOLO models for fast and accurate wildlife identification in images.

🖼️ **Powerful Batch Processing**: Supports importing entire folders of images for fully automated data processing workflows.

📄 **Detailed EXIF Data Extraction**: Automatically reads and integrates photo EXIF metadata, including key information like capture time.

📊 **Flexible Result Export**: One-click export of recognition results (species, count, time, etc.) and metadata to Excel (.xlsx) format for subsequent statistical analysis and report writing.

⚙️ **Highly Customizable Models**: Provides advanced options allowing users to replace or update YOLO models to meet different regional and species identification needs.

## 🚀 Quick Start

This program requires no installation - just extract and run.

**Download the Program**: Visit the [Releases](https://github.com/wakin721/Neri/releases) page and download the latest version .zip or .7z archive.

**Extract Files**: Extract the downloaded archive to any location where you want to store the program.

**Run the Program**: Navigate to the extracted folder and double-click Neri.exe to launch the program.

## 💡 User Guide

### Quick Usage
After launching the program, click the "Select Images" button and choose a folder containing infrared camera photos.

Click the "Start Processing" button, and the program will automatically begin batch identification of animals in the images, displaying real-time progress on the interface.

After identification is complete, click the "Verification" button to view detailed recognition results for each photo and make corrections based on the identification accuracy.

In the preview page, click "Export" to save all analysis data as .xlsx or .csv files for further use.

### Advanced Settings

Advanced settings are divided into three tabs: Model Parameter Settings, Environment Maintenance, and Software Settings.

### Using CUDA Acceleration

We recommend (but don't require) that your Windows system be equipped with an NVIDIA GPU, as this enables the use of higher precision models and faster processing.

How to check CUDA version and install CUDA-supported PyTorch?

1. Right-click on the desktop and select "NVIDIA Control Panel".

2. In the control panel menu bar, select "Help", then click "System Information".
<p align="center">
   <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda1.png" width="750px">
</p>

3. In the popup window, check the information under "Display" and find "NVCUDA64.DLL" and other related files to see the corresponding CUDA version.
<p align="center">
   <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda2.png">
</p>

4. In Advanced Settings - Environment Maintenance - Install PyTorch, select the corresponding version for installation.
<p align="center">
   <img src="https://github.com/wakin721/Neri/blob/main/res/demo/cuda3.png" width="750px">
</p>

## 🗺️ Future Roadmap

We plan to add more features in future versions:

[1] Introduce more efficient YOLO model versions to further improve recognition speed and accuracy.

[2] Add recognition and analysis functionality for infrared camera video data.

[3] Develop automated data analysis and chart generation modules for species abundance, activity patterns, etc.

[4] Support more diverse data filtering and labeling functions.

## ⚠️ Warning

The project is still under development - do not rely too heavily on this project. The built-in model is for testing purposes only and does not currently provide identification for many species.

## 📧 Contact Us

Author: 和錦わきん

If you encounter any problems during use or have any suggestions, please feel free to contact us through Issues.

## Acknowledgments

Thanks to Suger from 44°N for providing the logo for this project.
