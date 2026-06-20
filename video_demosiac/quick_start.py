#!/usr/bin/env python3
"""
快速开始脚本
============
帮助用户快速设置和运行视频马赛克去除工具。
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def check_python_version():
    """检查 Python 版本"""
    if sys.version_info < (3, 8):
        print("错误：需要 Python 3.8 或更高版本")
        print(f"当前版本：{sys.version}")
        return False
    print(f"✓ Python 版本：{sys.version}")
    return True


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "yaml",
        "tqdm"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "yaml":
                import yaml
            else:
                __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")

    return len(missing_packages) == 0, missing_packages


def check_cuda():
    """检查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用，设备：{torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA 不可用，将使用 CPU 处理")
            return False
    except ImportError:
        print("⚠ 无法导入 torch，CUDA 状态未知")
        return False


def check_model():
    """检查模型是否已下载"""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_files = ["propainter.pth", "e2fgvi.pth", "sttn.pth"]
    found_models = []

    for model_file in model_files:
        model_path = checkpoint_dir / model_file
        if model_path.exists():
            found_models.append(model_file)
            print(f"✓ 找到模型：{model_file}")
        else:
            print(f"✗ 未找到模型：{model_file}")

    return len(found_models) > 0, found_models


def install_dependencies():
    """安装依赖包"""
    print("\n正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖包安装失败：{e}")
        return False


def download_model():
    """下载模型"""
    print("\n正在下载模型...")
    try:
        subprocess.check_call([sys.executable, "download_models.py", "--model", "propainter"])
        print("✓ 模型下载完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 模型下载失败：{e}")
        return False


def create_test_video():
    """创建测试视频"""
    print("\n正在创建测试视频...")
    try:
        import cv2
        import numpy as np

        # 创建一个简单的测试视频
        width, height = 640, 480
        fps = 30
        duration = 5  # 秒
        total_frames = fps * duration

        # 创建 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_input.mp4', fourcc, fps, (width, height))

        # 创建带有马赛克的测试帧
        for i in range(total_frames):
            # 创建彩色背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (200, 150, 100)  # BGR

            # 添加移动的矩形
            x = int(100 + 100 * np.sin(2 * np.pi * i / total_frames))
            y = int(100 + 50 * np.cos(2 * np.pi * i / total_frames))
            cv2.rectangle(frame, (x, y), (x+100, y+80), (0, 255, 0), -1)

            # 添加马赛克区域
            mosaic_x, mosaic_y = 400, 200
            mosaic_size = 20
            for my in range(mosaic_y, mosaic_y + 100, mosaic_size):
                for mx in range(mosaic_x, mosaic_x + 100, mosaic_size):
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    cv2.rectangle(frame, (mx, my), (mx+mosaic_size, my+mosaic_size), color, -1)

            # 添加文本
            cv2.putText(frame, f"Frame: {i}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print("✓ 测试视频已创建：test_input.mp4")
        return True

    except Exception as e:
        print(f"✗ 创建测试视频失败：{e}")
        return False


def run_test():
    """运行测试"""
    print("\n正在运行测试...")
    try:
        subprocess.check_call([
            sys.executable, "main.py",
            "--input", "test_input.mp4",
            "--output", "test_output.mp4",
            "--auto-detect",
            "--device", "cuda" if check_cuda() else "cpu"
        ])
        print("✓ 测试完成，输出文件：test_output.mp4")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 测试失败：{e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("视频马赛克去除工具 - 快速开始")
    print("="*60)

    # 检查 Python 版本
    if not check_python_version():
        return

    # 检查依赖包
    print("\n检查依赖包...")
    deps_installed, missing_deps = check_dependencies()

    if not deps_installed:
        print(f"\n缺少以下依赖包：{', '.join(missing_deps)}")
        response = input("是否自动安装依赖包？(y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("请先安装依赖包：pip install -r requirements.txt")
            return

    # 检查 CUDA
    print("\n检查 CUDA...")
    has_cuda = check_cuda()

    # 检查模型
    print("\n检查模型...")
    models_found, found_models = check_model()

    if not models_found:
        print("\n未找到预训练模型")
        response = input("是否自动下载模型？(y/n): ")
        if response.lower() == 'y':
            if not download_model():
                print("模型下载失败，将使用默认配置")
        else:
            print("请先下载模型：python download_models.py --model propainter")

    # 创建测试视频
    print("\n创建测试视频...")
    response = input("是否创建测试视频？(y/n): ")
    if response.lower() == 'y':
        if create_test_video():
            # 运行测试
            response = input("是否运行测试？(y/n): ")
            if response.lower() == 'y':
                run_test()

    # 显示使用说明
    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    print("1. 基本使用：")
    print("   python main.py --input input.mp4 --output output.mp4")
    print()
    print("2. 自动检测马赛克：")
    print("   python main.py --input input.mp4 --output output.mp4 --auto-detect")
    print()
    print("3. 使用配置文件：")
    print("   python main.py --config configs/config.yaml")
    print()
    print("4. 批量处理：")
    print("   python batch_process.py --input-dir ./videos --output-dir ./output")
    print()
    print("5. 查看完整说明：")
    print("   cat README.md")
    print("="*60)


if __name__ == "__main__":
    main()
