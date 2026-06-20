#!/bin/bash
# 视频马赛克去除工具启动脚本
# ==========================

echo ""
echo "============================================================"
echo "视频马赛克去除工具 v1.0.0"
echo "============================================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查参数
if [ "$#" -eq 0 ]; then
    echo "使用方法："
    echo "  ./run.sh --input video.mp4 --output output.mp4"
    echo ""
    echo "或运行快速开始："
    echo "  ./run.sh --quick"
    echo ""
    exit 0
fi

# 快速开始模式
if [ "$1" = "--quick" ]; then
    python3 quick_start.py
    exit 0
fi

# 运行主程序
python3 main.py "$@"
