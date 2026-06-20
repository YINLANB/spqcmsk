#!/usr/bin/env python3
"""
ProPainter 模型下载脚本
======================
直接下载 ProPainter 预训练模型
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# 下载链接
DOWNLOAD_URLS = {
    "propainter": "https://github.com/sczhou/ProPainter/releases/download/v1.0/propainter.pth",
    "recurrent_flow": "https://github.com/sczhou/ProPainter/releases/download/v1.0/recurrent_flow_completion.pth",
}

def download_file(url, output_path):
    """
    下载文件（支持进度显示）

    Args:
        url: 下载链接
        output_path: 输出路径
    """
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        # 下载
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"下载完成: {output_path}")
        return True

    except Exception as e:
        print(f"下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("ProPainter 模型下载工具")
    print("=" * 60)

    # 创建 checkpoints 目录
    checkpoints_dir = Path("./checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # 下载模型
    success_count = 0
    for name, url in DOWNLOAD_URLS.items():
        output_path = checkpoints_dir / f"{name}.pth"

        if output_path.exists():
            print(f"模型已存在: {output_path}")
            success_count += 1
            continue

        if download_file(url, output_path):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(DOWNLOAD_URLS):
        print("所有模型下载完成！")
        print("\n现在可以使用以下命令处理视频:")
        print("  python main.py --input video.mp4 --output output.mp4")
    else:
        print("部分模型下载失败")
        print("\n请手动下载模型:")
        print("  1. 访问: https://github.com/sczhou/ProPainter/releases")
        print("  2. 下载 propainter.pth 和 recurrent_flow_completion.pth")
        print("  3. 放到 checkpoints/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
