#!/usr/bin/env python3
"""
模型验证脚本
============
检查模型文件是否已正确安装
"""

import os
from pathlib import Path

print("=" * 60)
print("模型验证工具")
print("=" * 60)

# 检查 checkpoints 目录
checkpoints_dir = Path("./checkpoints")
if not checkpoints_dir.exists():
    print("\n[ERROR] checkpoints 目录不存在")
    print("请创建目录并放入模型文件")
    exit(1)

# 检查模型文件
models = {
    "propainter.pth": "ProPainter 主模型",
    "recurrent_flow_completion.pth": "光流补全模型",
}

print("\n检查模型文件:")
all_ok = True
for filename, description in models.items():
    filepath = checkpoints_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  [OK] {filename} ({size_mb:.1f} MB) - {description}")
    else:
        print(f"  [MISSING] {filename} - {description}")
        all_ok = False

# 检查可选模型
optional_models = {
    "GFPGANv1.4.pth": "GFPGAN 人脸修复模型",
    "detection_Resnet50_Final.pth": "人脸检测模型",
}

print("\n可选模型:")
for filename, description in optional_models.items():
    filepath = checkpoints_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  [OK] {filename} ({size_mb:.1f} MB) - {description}")
    else:
        print(f"  [NOT FOUND] {filename} - {description}")

# 总结
print("\n" + "=" * 60)
if all_ok:
    print("验证通过！所有必需模型已安装。")
    print("\n现在可以使用以下命令处理视频:")
    print("  python main.py --input video.mp4 --output output.mp4")
else:
    print("验证失败！缺少必需模型。")
    print("\n请下载模型文件并放到 checkpoints/ 目录:")
    print("  1. 访问: https://github.com/sczhou/ProPainter/releases")
    print("  2. 下载: propainter.pth 和 recurrent_flow_completion.pth")
    print("  3. 放到: video_demosiac/checkpoints/")
print("=" * 60)
