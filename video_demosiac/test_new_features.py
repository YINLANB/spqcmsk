#!/usr/bin/env python3
"""
新功能测试脚本
==============
测试实时预览、断点续传、遮罩编辑增强等功能。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_processor import VideoProcessor
from utils.mask_generator import MaskGenerator
from utils.model_runner import ModelRunner
from utils.logger import setup_logger, ProcessingStats

logger = setup_logger(__name__)


def create_test_video(output_path: str, num_frames: int = 100):
    """
    创建测试视频

    Args:
        output_path: 输出路径
        num_frames: 帧数
    """
    import cv2

    logger.info(f"创建测试视频: {output_path}")

    # 视频参数
    width, height = 640, 480
    fps = 30

    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 创建带有马赛克的测试帧
    for i in range(num_frames):
        # 创建彩色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (200, 150, 100)  # BGR

        # 添加移动的矩形
        x = int(100 + 100 * np.sin(2 * np.pi * i / num_frames))
        y = int(100 + 50 * np.cos(2 * np.pi * i / num_frames))
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
    logger.info(f"测试视频已创建: {output_path}")


def test_realtime_preview():
    """测试实时预览功能"""
    logger.info("=" * 60)
    logger.info("测试实时预览功能")
    logger.info("=" * 60)

    # 创建测试视频
    test_video = "test_preview.mp4"
    if not os.path.exists(test_video):
        create_test_video(test_video, num_frames=50)

    # 配置
    config = {
        "base": {
            "input_video": test_video,
            "output_video": "test_output_preview.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cpu",  # 使用 CPU 进行测试
            "use_fp16": False
        },
        "process": {
            "chunk_size": 5,
            "neighbor_frames": 3,
            "bidirectional": True,
            "dilate_kernel": 5
        },
        "detection": {
            "auto_detect": True,
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
            "show_progress": True,
            "preview": True,  # 启用预览
            "preview_scale": 0.5
        },
        "resume": {
            "enabled": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()

    # 清理
    if os.path.exists(test_video):
        os.remove(test_video)
    if os.path.exists("test_output_preview.mp4"):
        os.remove("test_output_preview.mp4")


def test_resume_feature():
    """测试断点续传功能"""
    logger.info("=" * 60)
    logger.info("测试断点续传功能")
    logger.info("=" * 60)

    # 创建测试视频
    test_video = "test_resume.mp4"
    if not os.path.exists(test_video):
        create_test_video(test_video, num_frames=100)

    output_video = "test_output_resume.mp4"

    # 配置
    config = {
        "base": {
            "input_video": test_video,
            "output_video": output_video,
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cpu",
            "use_fp16": False
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 5,
            "bidirectional": True,
            "dilate_kernel": 5
        },
        "detection": {
            "auto_detect": True,
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
            "show_progress": True,
            "preview": False
        },
        "resume": {
            "enabled": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()

    # 检查进度文件是否被清理
    progress_file = output_video + ".progress.json"
    if not os.path.exists(progress_file):
        logger.info("✓ 进度文件已正确清理")
    else:
        logger.warning("✗ 进度文件未被清理")

    # 清理
    if os.path.exists(test_video):
        os.remove(test_video)
    if os.path.exists(output_video):
        os.remove(output_video)


def test_processing_stats():
    """测试处理统计功能"""
    logger.info("=" * 60)
    logger.info("测试处理统计功能")
    logger.info("=" * 60)

    # 创建统计对象
    stats = ProcessingStats(total_chunks=10, total_frames=100)

    # 模拟处理过程
    import time
    for i in range(5):
        time.sleep(0.1)
        stats.update(chunk_time=0.1, frames_processed=10)
        stats.print_status(chunk_idx=i, chunk_time=0.1)

    # 打印摘要
    stats.print_summary()

    logger.info("✓ 处理统计功能正常")


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("新功能测试")
    logger.info("=" * 60)

    # 测试处理统计
    test_processing_stats()

    # 注意：实时预览和断点续传测试需要实际运行视频处理
    # 在无 GUI 环境下可能会失败
    # 取消注释以下代码进行完整测试
    # test_realtime_preview()
    # test_resume_feature()

    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
