#!/usr/bin/env python3
"""
多格式输出写入器
================
支持多种视频和图片序列输出格式。
"""

import os
import cv2
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

from .logger import setup_logger

logger = setup_logger(__name__)


class OutputWriter(ABC):
    """输出写入器基类"""

    def __init__(self, output_path: str, video_info: dict, config: dict):
        """
        初始化输出写入器

        Args:
            output_path: 输出路径
            video_info: 视频信息
            config: 配置字典
        """
        self.output_path = output_path
        self.video_info = video_info
        self.config = config
        self.frame_count = 0

    @abstractmethod
    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        pass

    @abstractmethod
    def release(self):
        """释放资源"""
        pass

    def get_frame_count(self) -> int:
        """获取已写入的帧数"""
        return self.frame_count


class VideoWriterMP4(OutputWriter):
    """MP4/AVI/MKV 视频写入器"""

    def __init__(self, output_path: str, video_info: dict, config: dict):
        super().__init__(output_path, video_info, config)

        # 获取编码格式
        codec_str = config.get("output", {}).get("codec", "mp4v")
        if codec_str == "mp4v":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif codec_str == "XVID":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif codec_str == "H264":
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建写入器
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info.get("fps", 30),
            (video_info.get("width", 640), video_info.get("height", 480))
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {output_path}")

    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        """释放资源"""
        if self.writer:
            self.writer.release()


class VideoWriterGIF(OutputWriter):
    """GIF 写入器（使用 ffmpeg）"""

    def __init__(self, output_path: str, video_info: dict, config: dict):
        super().__init__(output_path, video_info, config)

        # 创建临时目录
        self.temp_dir = Path(output_path).parent / ".temp_gif"
        self.temp_dir.mkdir(exist_ok=True)

        # 临时 AVI 文件
        self.temp_avi = self.temp_dir / "temp.avi"
        self.temp_writer = cv2.VideoWriter(
            str(self.temp_avi),
            cv2.VideoWriter_fourcc(*'XVID'),
            video_info.get("fps", 30),
            (video_info.get("width", 640), video_info.get("height", 480))
        )

        # GIF 参数
        self.fps = video_info.get("fps", 30)
        self.gif_fps = min(15, self.fps)  # GIF 最大 15 fps

    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        self.temp_writer.write(frame)
        self.frame_count += 1

    def release(self):
        """释放资源并转换为 GIF"""
        if self.temp_writer:
            self.temp_writer.release()

        # 使用 ffmpeg 转换
        try:
            cmd = [
                "ffmpeg",
                "-i", str(self.temp_avi),
                "-vf", f"fps={self.gif_fps},scale=480:-1:flags=lanczos",
                "-loop", "0",
                "-y",
                self.output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"GIF 已保存: {self.output_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"GIF 转换失败: {e}")
            # 备用方案：直接使用 AVI
            import shutil
            shutil.copy2(str(self.temp_avi), self.output_path)

        except FileNotFoundError:
            logger.warning("ffmpeg 未找到，将保存为 AVI 格式")
            import shutil
            shutil.copy2(str(self.temp_avi), self.output_path)

        finally:
            # 清理临时文件
            self._cleanup()

    def _cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_avi.exists():
                self.temp_avi.unlink()
            if self.temp_dir.exists():
                self.temp_dir.rmdir()
        except Exception:
            pass


class ImageSequenceWriter(OutputWriter):
    """图片序列写入器"""

    def __init__(self, output_path: str, video_info: dict, config: dict):
        super().__init__(output_path, video_info, config)

        # 创建输出目录
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 图片格式
        self.image_format = config.get("output", {}).get("image_format", "png")
        self.quality = config.get("output", {}).get("quality", 95)

    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        frame_path = self.output_dir / f"frame_{self.frame_count:06d}.{self.image_format}"

        if self.image_format == "jpg" or self.image_format == "jpeg":
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        elif self.image_format == "png":
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(str(frame_path), frame)

        self.frame_count += 1

    def release(self):
        """释放资源"""
        logger.info(f"图片序列已保存: {self.output_dir} ({self.frame_count} 帧)")


class VideoWriterWebM(OutputWriter):
    """WebM 写入器（使用 ffmpeg）"""

    def __init__(self, output_path: str, video_info: dict, config: dict):
        super().__init__(output_path, video_info, config)

        # 创建临时目录
        self.temp_dir = Path(output_path).parent / ".temp_webm"
        self.temp_dir.mkdir(exist_ok=True)

        # 临时 AVI 文件
        self.temp_avi = self.temp_dir / "temp.avi"
        self.temp_writer = cv2.VideoWriter(
            str(self.temp_avi),
            cv2.VideoWriter_fourcc(*'XVID'),
            video_info.get("fps", 30),
            (video_info.get("width", 640), video_info.get("height", 480))
        )

    def write_frame(self, frame: np.ndarray):
        """写入一帧"""
        self.temp_writer.write(frame)
        self.frame_count += 1

    def release(self):
        """释放资源并转换为 WebM"""
        if self.temp_writer:
            self.temp_writer.release()

        # 使用 ffmpeg 转换
        try:
            cmd = [
                "ffmpeg",
                "-i", str(self.temp_avi),
                "-c:v", "libvpx-vp9",
                "-crf", "30",
                "-b:v", "0",
                "-an",
                "-y",
                self.output_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"WebM 已保存: {self.output_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"WebM 转换失败: {e}")
            # 备用方案：直接使用 AVI
            import shutil
            shutil.copy2(str(self.temp_avi), self.output_path)

        except FileNotFoundError:
            logger.warning("ffmpeg 未找到，将保存为 AVI 格式")
            import shutil
            shutil.copy2(str(self.temp_avi), self.output_path)

        finally:
            # 清理临时文件
            self._cleanup()

    def _cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_avi.exists():
                self.temp_avi.unlink()
            if self.temp_dir.exists():
                self.temp_dir.rmdir()
        except Exception:
            pass


def create_output_writer(output_path: str, video_info: dict, config: dict) -> OutputWriter:
    """
    创建输出写入器的工厂函数

    Args:
        output_path: 输出路径
        video_info: 视频信息
        config: 配置字典

    Returns:
        OutputWriter: 输出写入器实例
    """
    # 获取输出格式
    output_format = config.get("output", {}).get("format", "mp4")

    # 根据格式创建对应的写入器
    if output_format == "mp4":
        return VideoWriterMP4(output_path, video_info, config)
    elif output_format == "avi":
        return VideoWriterMP4(output_path, video_info, config)
    elif output_format == "mkv":
        return VideoWriterMP4(output_path, video_info, config)
    elif output_format == "gif":
        return VideoWriterGIF(output_path, video_info, config)
    elif output_format == "webm":
        return VideoWriterWebM(output_path, video_info, config)
    elif output_format == "image_sequence":
        return ImageSequenceWriter(output_path, video_info, config)
    else:
        logger.warning(f"不支持的输出格式: {output_format}，使用 MP4")
        return VideoWriterMP4(output_path, video_info, config)


def convert_video(input_path: str, output_path: str, format: str, **kwargs) -> bool:
    """
    转换视频格式

    Args:
        input_path: 输入路径
        output_path: 输出路径
        format: 目标格式
        **kwargs: 额外参数

    Returns:
        bool: 是否成功
    """
    try:
        if format == "gif":
            fps = kwargs.get("fps", 15)
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"fps={fps},scale=480:-1:flags=lanczos",
                "-loop", "0",
                "-y",
                output_path
            ]
        elif format == "webm":
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libvpx-vp9",
                "-crf", "30",
                "-b:v", "0",
                "-an",
                "-y",
                output_path
            ]
        else:
            # 使用 ffmpeg 的通用转换
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-y",
                output_path
            ]

        subprocess.run(cmd, capture_output=True, check=True)
        return True

    except Exception as e:
        logger.error(f"视频转换失败: {e}")
        return False
