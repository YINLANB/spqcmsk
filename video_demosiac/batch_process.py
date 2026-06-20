#!/usr/bin/env python3
"""
批量处理脚本
============
批量处理多个视频文件。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_processor import VideoProcessor
from utils.mask_generator import MaskGenerator
from utils.model_runner import ModelRunner
from utils.logger import setup_logger

logger = setup_logger(__name__)


def find_videos(input_dir: str, extensions: List[str] = None) -> List[str]:
    """
    查找目录中的所有视频文件

    Args:
        input_dir: 输入目录
        extensions: 视频文件扩展名列表

    Returns:
        list: 视频文件路径列表
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"目录不存在: {input_dir}")
        return []

    video_files = []
    for ext in extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted([str(v) for v in video_files])


def process_single_video(video_path: str, output_path: str, config: dict):
    """
    处理单个视频

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        config: 配置字典
    """
    # 更新配置
    config["base"]["input_video"] = video_path
    config["base"]["output_video"] = output_path

    logger.info(f"处理视频: {video_path}")
    logger.info(f"输出到: {output_path}")

    try:
        # 创建组件
        mask_generator = MaskGenerator(config)
        model_runner = ModelRunner(config)
        video_processor = VideoProcessor(config, mask_generator, model_runner)

        # 处理视频
        video_processor.process()

        logger.info(f"视频处理完成: {video_path}")
        return True

    except Exception as e:
        logger.error(f"视频处理失败: {video_path}, 错误: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量处理视频文件")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入视频目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出视频目录"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["propainter", "e2fgvi", "sttn"],
        default="propainter",
        help="选择模型 (默认: propainter)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="运行设备 (默认: cuda)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用半精度推理"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="处理块大小 (默认: 10)"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="启用自动马赛克检测"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="视频文件扩展名"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录"
    )

    args = parser.parse_args()

    # 查找视频文件
    logger.info(f"正在搜索视频文件: {args.input_dir}")
    video_files = find_videos(args.input_dir, args.extensions)

    if not video_files:
        logger.error("未找到视频文件")
        sys.exit(1)

    logger.info(f"找到 {len(video_files)} 个视频文件")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 基础配置
    base_config = {
        "base": {
            "input_video": "",
            "output_video": "",
            "mask_file": ""
        },
        "model": {
            "name": args.model,
            "checkpoint_dir": "./checkpoints",
            "device": args.device,
            "use_fp16": args.fp16
        },
        "process": {
            "chunk_size": args.chunk_size,
            "neighbor_frames": 10,
            "bidirectional": True,
            "dilate_kernel": 5
        },
        "detection": {
            "auto_detect": args.auto_detect,
            "sensitivity": 0.7,
            "min_area": 100,
            "morphology": True,
            "dilate_iterations": 2,
            "close_kernel": 5
        },
        "output": {
            "codec": "mp4v",
            "quality": 95,
            "save_frames": False,
            "show_progress": True
        }
    }

    # 处理每个视频
    success_count = 0
    fail_count = 0

    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n处理进度: {i}/{len(video_files)}")

        # 生成输出路径
        video_name = Path(video_path).stem
        output_path = os.path.join(args.output_dir, f"{video_name}_fixed.mp4")

        # 处理视频
        if process_single_video(video_path, output_path, base_config.copy()):
            success_count += 1
        else:
            fail_count += 1

    # 打印统计信息
    print("\n" + "="*60)
    print("批量处理完成")
    print("="*60)
    print(f"  总视频数: {len(video_files)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出目录: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
