@echo off
REM 视频马赛克去除工具启动脚本
REM ==========================

echo.
echo ============================================================
echo 视频马赛克去除工具 v1.0.0
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查参数
if "%~1"=="" (
    echo 使用方法：
    echo   run.bat --input video.mp4 --output output.mp4
    echo.
    echo 或运行快速开始：
    echo   run.bat --quick
    echo.
    pause
    exit /b 0
)

REM 快速开始模式
if "%~1"=="--quick" (
    python quick_start.py
    pause
    exit /b 0
)

REM 运行主程序
python main.py %*
pause
