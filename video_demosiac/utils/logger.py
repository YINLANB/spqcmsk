#!/usr/bin/env python3
"""
日志工具模块
============
提供统一的日志记录功能，支持文件日志和丰富的统计信息。
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name, level=logging.INFO, log_file: Optional[str] = None):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 创建格式器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建文件处理器（如果指定）
    if log_file:
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"无法创建日志文件 {log_file}: {e}")

    return logger


def log_progress(current, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    打印进度条

    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文本
        suffix: 后缀文本
        decimals: 小数位数
        length: 进度条长度
        fill: 填充字符
    """
    if total == 0:
        return

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')

    # 打印新行当完成时
    if current == total:
        print()


def log_processing_info(video_path, info):
    """
    打印视频处理信息

    Args:
        video_path: 视频路径
        info: 视频信息字典
    """
    print("\n" + "=" * 60)
    print(f"视频信息: {video_path}")
    print("=" * 60)
    print(f"  分辨率: {info.get('width', 'N/A')} x {info.get('height', 'N/A')}")
    print(f"  帧率: {info.get('fps', 'N/A')} FPS")
    print(f"  总帧数: {info.get('total_frames', 'N/A')}")
    print(f"  时长: {info.get('duration', 'N/A'):.2f} 秒")
    print(f"  编码格式: {info.get('codec', 'N/A')}")
    print("=" * 60 + "\n")


class ProcessingStats:
    """
    处理统计类
    用于跟踪和报告处理进度、速度、ETA等信息
    """

    def __init__(self, total_chunks: int, total_frames: int):
        """
        初始化处理统计

        Args:
            total_chunks: 总块数
            total_frames: 总帧数
        """
        self.total_chunks = total_chunks
        self.total_frames = total_frames
        self.completed_chunks = 0
        self.processed_frames = 0
        self.chunk_times = []
        self.start_time = datetime.now()

    def update(self, chunk_time: float, frames_processed: int):
        """
        更新统计信息

        Args:
            chunk_time: 块处理时间（秒）
            frames_processed: 本块处理的帧数
        """
        self.chunk_times.append(chunk_time)
        self.completed_chunks += 1
        self.processed_frames += frames_processed

    def get_average_time(self) -> float:
        """获取平均处理时间"""
        if not self.chunk_times:
            return 0.0
        return sum(self.chunk_times) / len(self.chunk_times)

    def get_eta_seconds(self) -> float:
        """获取预计剩余时间（秒）"""
        if not self.chunk_times:
            return 0.0
        avg_time = self.get_average_time()
        remaining_chunks = self.total_chunks - self.completed_chunks
        return avg_time * remaining_chunks

    def get_progress_percent(self) -> float:
        """获取进度百分比"""
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_frames) * 100

    def get_processing_speed(self) -> float:
        """获取处理速度（帧/秒）"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.processed_frames / elapsed

    def get_elapsed_time(self) -> str:
        """获取已用时间字符串"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_eta_string(self) -> str:
        """获取预计剩余时间字符串"""
        eta = self.get_eta_seconds()
        hours = int(eta // 3600)
        minutes = int((eta % 3600) // 60)
        seconds = int(eta % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_gpu_info(self) -> str:
        """获取 GPU 信息"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"GPU: {gpu_memory:.1f}/{gpu_total:.1f}GB"
        except ImportError:
            pass
        return "GPU: N/A"

    def print_status(self, chunk_idx: int, chunk_time: float):
        """
        打印当前状态

        Args:
            chunk_idx: 当前块索引
            chunk_time: 当前块处理时间
        """
        progress = self.get_progress_percent()
        avg_time = self.get_average_time()
        eta = self.get_eta_string()
        elapsed = self.get_elapsed_time()
        speed = self.get_processing_speed()
        gpu = self.get_gpu_info()

        status_line = (
            f"[{chunk_idx+1}/{self.total_chunks}] "
            f"Progress: {progress:.1f}% | "
            f"Time: {chunk_time:.1f}s | "
            f"Avg: {avg_time:.1f}s/chunk | "
            f"Speed: {speed:.1f} fps | "
            f"ETA: {eta} | "
            f"Elapsed: {elapsed} | "
            f"{gpu}"
        )

        print(f"\r{status_line}", end='', flush=True)

        # 每10个块打印一次换行
        if (chunk_idx + 1) % 10 == 0:
            print()

    def print_summary(self):
        """打印处理摘要"""
        elapsed = self.get_elapsed_time()
        speed = self.get_processing_speed()
        avg_time = self.get_average_time()

        print("\n" + "=" * 60)
        print("处理摘要")
        print("=" * 60)
        print(f"  总块数: {self.total_chunks}")
        print(f"  处理帧数: {self.processed_frames}")
        print(f"  总耗时: {elapsed}")
        print(f"  平均块处理时间: {avg_time:.2f} 秒")
        print(f"  平均处理速度: {speed:.2f} 帧/秒")
        print(f"  GPU: {self.get_gpu_info()}")
        print("=" * 60)


def create_log_file(output_path: str) -> str:
    """
    创建日志文件路径

    Args:
        output_path: 输出视频路径

    Returns:
        str: 日志文件路径
    """
    # 在输出文件同目录创建日志文件
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"processing_{timestamp}.log")

    return log_file
