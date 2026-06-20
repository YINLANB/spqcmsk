@echo off
chcp 65001 >nul
title 下载 ProPainter 模型

echo ============================================================
echo     ProPainter 模型下载指南
echo ============================================================
echo.
echo 由于网络问题无法自动下载，请按以下步骤手动下载：
echo.
echo ------------------------------------------------------------
echo 步骤1: 下载模型文件
echo ------------------------------------------------------------
echo.
echo 方式1 - GitHub (推荐):
echo   访问: https://github.com/sczhou/ProPainter/releases
echo   下载: propainter.pth 和 recurrent_flow_completion.pth
echo.
echo 方式2 - 百度网盘:
echo   链接: https://pan.baidu.com/s/1MxLQf8rNkHhRzGNlLZfVwA?pwd=1234
echo.
echo 方式3 - Google Drive:
echo   链接: https://drive.google.com/file/d/1OwJBLUjh4wEUv4W9GKqbYMa8dPHqZMw1/view
echo.
echo ------------------------------------------------------------
echo 步骤2: 放置模型文件
echo ------------------------------------------------------------
echo.
echo 将下载的文件放到以下目录:
echo   %~dp0video_demosiac\checkpoints\
echo.
echo 文件结构:
echo   video_demosiac/
echo   └── checkpoints/
echo       ├── propainter.pth
echo       └── recurrent_flow_completion.pth
echo.
echo ------------------------------------------------------------
echo 步骤3: 验证安装
echo ------------------------------------------------------------
echo.
echo 运行以下命令验证:
echo   python verify_model.py
echo.
echo ============================================================
echo.

pause
