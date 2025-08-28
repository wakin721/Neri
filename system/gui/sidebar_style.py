"""
侧边栏样式模块 - 提供左侧导航栏的样式定义
"""

import tkinter as tk
from tkinter import ttk

def apply_sidebar_style():
    """应用侧边栏样式"""
    style = ttk.Style()
    
    # 创建侧边栏框架样式
    style.configure("Sidebar.TFrame", background="#f0f0f0")
    
    # 创建选中状态的导航按钮样式
    style.map("Nav.TButton",
        background=[('pressed', '#e1e1e1'), ('active', '#e9e9e9')],
        relief=[('pressed', 'sunken'), ('!pressed', 'flat')]
    )
    
    # 配置导航按钮样式
    style.configure("Nav.TButton", 
                   padding=10, 
                   relief="flat",
                   background="#f0f0f0",
                   anchor="center")
    
    # 未激活状态下的导航按钮样式
    style.map("Nav.TButton",
              background=[('active', '#e1e1e1')])