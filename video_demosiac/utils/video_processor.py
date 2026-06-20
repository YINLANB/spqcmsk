#!/usr/bin/env python3
"""
视频处理器模块
==============
负责视频的读取、处理和写入。
支持长视频分块处理、实时预览、断点续传。
"""

import os
import cv2
import json
import time
import signal
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Callable

from .logger import setup_logger, log_progress, log_processing_info

logger = setup_logger(__name__)


class VideoProcessor:
    """
    视频处理器
    支持长视频分块处理、实时预览、断点续传
    """

    def __init__(self, config, mask_generator, model_runner):
        """
        初始化视频处理器

        Args:
            config: 配置字典
            mask_generator: 遮罩生成器实例
            model_runner: 模型运行器实例
        """
        self.config = config
        self.mask_generator = mask_generator
        self.model_runner = model_runner

        # 处理参数
        self.input_video = config["base"]["input_video"]
        self.output_video = config["base"]["output_video"]
        self.chunk_size = config["process"]["chunk_size"]
        self.neighbor_frames = config["process"]["neighbor_frames"]
        self.bidirectional = config["process"].get("bidirectional", True)

        # 输出参数
        self.save_frames = config["output"]["save_frames"]
        self.frames_dir = config["output"].get("frames_dir", "./output_frames")
        self.show_progress = config["output"]["show_progress"]

        # 预览参数
        self.preview_enabled = config["output"].get("preview", False)
        self.preview_scale = config["output"].get("preview_scale", 0.5)
        self.preview_window = "Video Preview (Press Q to quit)"

        # 断点续传参数
        self.progress_file = None
        self.temp_dir = None
        self.resume_enabled = config.get("resume", {}).get("enabled", True)

        # 初始化视频捕获
        self.cap = None
        self.writer = None
        self.video_info = {}

        # 信号处理
        self.stop_flag = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信号处理器，支持优雅退出"""
        def signal_handler(signum, frame):
            logger.info("\n收到中断信号，正在保存进度...")
            self.stop_flag = True

        # Windows 下只支持 SIGINT
        try:
            signal.signal(signal.SIGINT, signal_handler)
        except (OSError, ValueError):
            pass

    def _init_video_capture(self):
        """初始化视频捕获对象"""
        self.cap = cv2.VideoCapture(self.input_video)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.input_video}")

        # 获取视频信息
        self.video_info = {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": self._get_codec_name()
        }

        # 计算时长
        self.video_info["duration"] = (
            self.video_info["total_frames"] / self.video_info["fps"]
            if self.video_info["fps"] > 0 else 0
        )

        return self.video_info

    def _get_codec_name(self):
        """获取视频编码名称"""
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec

    def _create_video_writer(self, output_path=None):
        """创建视频写入器"""
        # 获取编码格式
        codec_str = self.config["output"]["codec"]
        if codec_str == "mp4v":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif codec_str == "XVID":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 创建输出目录
        output = output_path or self.output_video
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        # 创建写入器
        writer = cv2.VideoWriter(
            output,
            fourcc,
            self.video_info["fps"],
            (self.video_info["width"], self.video_info["height"])
        )

        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {output}")

        return writer

    def _read_frames(self, num_frames):
        """
        读取指定数量的帧

        Args:
            num_frames: 要读取的帧数

        Returns:
            list: 帧列表，每帧为 numpy 数组
        """
        frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _get_frame_range(self, chunk_idx):
        """
        获取块的帧范围（包含邻近帧）

        Args:
            chunk_idx: 块索引

        Returns:
            tuple: (start_frame, end_frame)
        """
        # 计算当前块的帧范围
        chunk_start = chunk_idx * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.video_info["total_frames"])

        # 扩展范围以包含邻近帧
        if self.bidirectional:
            start = max(0, chunk_start - self.neighbor_frames)
            end = min(self.video_info["total_frames"], chunk_end + self.neighbor_frames)
        else:
            start = chunk_start
            end = min(self.video_info["total_frames"], chunk_end + self.neighbor_frames)

        return start, end

    def _extract_chunk(self, start_frame, end_frame):
        """
        提取视频块

        Args:
            start_frame: 起始帧
            end_frame: 结束帧

        Returns:
            tuple: (frames, masks)
        """
        # 定位到起始帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 读取帧
        frames = self._read_frames(end_frame - start_frame)

        # 生成遮罩
        masks = self.mask_generator.generate_masks(frames)

        return frames, masks

    def _process_chunk(self, frames, masks, chunk_start, chunk_size):
        """
        处理一个视频块

        Args:
            frames: 帧列表
            masks: 遮罩列表
            chunk_start: 块起始帧索引
            chunk_size: 块大小

        Returns:
            list: 处理后的帧列表
        """
        # 使用模型进行修复
        repaired_frames = self.model_runner.repair_frames(frames, masks)

        # 如果使用双向传播，只取中间部分
        if self.bidirectional and len(repaired_frames) > chunk_size:
            # 计算需要保留的帧数
            offset = (len(repaired_frames) - chunk_size) // 2
            repaired_frames = repaired_frames[offset:offset + chunk_size]

        return repaired_frames

    def _show_preview(self, original_frames, repaired_frames, masks, chunk_idx):
        """
        显示预览窗口

        Args:
            original_frames: 原始帧列表
            repaired_frames: 修复后的帧列表
            masks: 遮罩列表
            chunk_idx: 当前块索引
        """
        if not self.preview_enabled or not repaired_frames:
            return

        # 显示第一帧的预览
        original = original_frames[0] if original_frames else repaired_frames[0]
        repaired = repaired_frames[0]
        mask = masks[0] if masks else np.zeros(repaired.shape[:2], dtype=np.uint8)

        # 创建对比视图：原图 | 遮罩 | 修复后
        h, w = original.shape[:2]
        preview_w = int(w * self.preview_scale)
        preview_h = int(h * self.preview_scale)

        # 缩放帧
        original_small = cv2.resize(original, (preview_w, preview_h))
        repaired_small = cv2.resize(repaired, (preview_w, preview_h))
        mask_small = cv2.resize(mask, (preview_w, preview_h))

        # 将遮罩转为彩色（红色半透明）
        mask_color = np.zeros_like(repaired_small)
        mask_color[:, :, 2] = mask_small  # 红色通道
        mask_overlay = cv2.addWeighted(repaired_small, 0.7, mask_color, 0.3, 0)

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label_h = 25

        # 创建带标签的帧
        def add_label(frame, label):
            labeled = np.zeros((frame.shape[0] + label_h, frame.shape[1], 3), dtype=np.uint8)
            labeled[label_h:] = frame
            cv2.putText(labeled, label, (5, 18), font, font_scale, (255, 255, 255), font_thickness)
            return labeled

        original_labeled = add_label(original_small, "Original")
        mask_labeled = add_label(mask_overlay, "Mask + Repaired")
        repaired_labeled = add_label(repaired_small, "Repaired")

        # 拼接三个视图
        preview = np.hstack([original_labeled, mask_labeled, repaired_labeled])

        # 添加顶部信息栏
        info_height = 35
        info_bar = np.zeros((info_height, preview.shape[1], 3), dtype=np.uint8)
        info_bar[:] = (50, 50, 50)

        # 添加信息文本
        total_frames = self.video_info["total_frames"]
        current_frame = chunk_idx * self.chunk_size
        progress_pct = (current_frame / total_frames * 100) if total_frames > 0 else 0

        info_text = f"Chunk: {chunk_idx+1} | Frame: {current_frame}/{total_frames} | Progress: {progress_pct:.1f}%"
        cv2.putText(info_bar, info_text, (10, 25), font, 0.6, (0, 255, 0), 1)

        # 合并信息栏和预览
        preview_with_info = np.vstack([info_bar, preview])

        # 显示预览窗口
        cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
        cv2.imshow(self.preview_window, preview_with_info)

        # 非阻塞等待，检测是否按下 Q 键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("用户请求退出预览")
            self.stop_flag = True

    def _init_progress_tracking(self):
        """初始化断点续传的进度跟踪"""
        # 设置进度文件和临时目录路径
        self.progress_file = self.output_video + ".progress.json"
        self.temp_dir = self.output_video + "_chunks"

        # 创建临时目录
        if self.resume_enabled:
            os.makedirs(self.temp_dir, exist_ok=True)

    def _save_progress(self, chunk_idx, total_chunks, processed_frames):
        """
        保存处理进度

        Args:
            chunk_idx: 当前块索引
            total_chunks: 总块数
            processed_frames: 已处理帧数
        """
        if not self.resume_enabled or not self.progress_file:
            return

        progress = {
            "input_video": self.input_video,
            "output_video": self.output_video,
            "total_chunks": total_chunks,
            "completed_chunks": chunk_idx + 1,
            "chunk_size": self.chunk_size,
            "processed_frames": processed_frames,
            "video_info": self.video_info,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def _load_progress(self):
        """
        加载处理进度

        Returns:
            dict: 进度信息，如果没有则返回 None
        """
        if not self.resume_enabled or not self.progress_file:
            return None

        if not os.path.exists(self.progress_file):
            return None

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

            # 验证进度文件是否匹配当前任务
            if progress.get("input_video") != self.input_video:
                logger.warning("进度文件与当前输入视频不匹配，将重新开始")
                return None

            return progress

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"读取进度文件失败: {e}，将重新开始")
            return None

    def _save_chunk_video(self, chunk_idx, repaired_frames):
        """
        保存单个块的视频片段

        Args:
            chunk_idx: 块索引
            repaired_frames: 修复后的帧列表
        """
        if not self.resume_enabled or not self.temp_dir:
            return

        chunk_path = os.path.join(self.temp_dir, f"chunk_{chunk_idx:06d}.mp4")
        writer = self._create_video_writer(chunk_path)

        for frame in repaired_frames:
            writer.write(frame)

        writer.release()

    def _merge_chunks(self):
        """合并所有分块视频到最终输出"""
        if not self.resume_enabled or not self.temp_dir:
            return

        # 查找所有分块文件
        chunk_files = sorted([
            os.path.join(self.temp_dir, f)
            for f in os.listdir(self.temp_dir)
            if f.startswith("chunk_") and f.endswith(".mp4")
        ])

        if not chunk_files:
            logger.warning("没有找到分块文件")
            return

        logger.info(f"正在合并 {len(chunk_files)} 个分块...")

        # 创建最终输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.output_video,
            fourcc,
            self.video_info["fps"],
            (self.video_info["width"], self.video_info["height"])
        )

        total_merged_frames = 0

        for chunk_file in chunk_files:
            cap = cv2.VideoCapture(chunk_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                total_merged_frames += 1
            cap.release()

        writer.release()

        logger.info(f"合并完成，共 {total_merged_frames} 帧")

        # 清理临时文件
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理临时文件"""
        import shutil

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")

        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
                logger.debug(f"已清理进度文件: {self.progress_file}")
            except Exception as e:
                logger.warning(f"清理进度文件失败: {e}")

    def process(self):
        """
        处理整个视频（支持断点续传和实时预览）
        """
        try:
            # 初始化视频捕获
            logger.info("正在初始化视频捕获...")
            video_info = self._init_video_capture()
            log_processing_info(self.input_video, video_info)

            # 初始化进度跟踪
            self._init_progress_tracking()

            # 计算总块数
            total_chunks = (
                self.video_info["total_frames"] + self.chunk_size - 1
            ) // self.chunk_size

            # 检查是否有未完成的任务
            start_chunk = 0
            processed_frames = 0

            if self.resume_enabled:
                progress = self._load_progress()
                if progress:
                    start_chunk = progress["completed_chunks"]
                    processed_frames = progress.get("processed_frames", 0)
                    logger.info(f"发现未完成的任务，将从第 {start_chunk} 个块继续")
                    logger.info(f"已处理 {processed_frames} 帧")

                    # 验证总块数是否一致
                    if progress.get("total_chunks") != total_chunks:
                        logger.warning("视频长度已变化，将重新开始")
                        start_chunk = 0
                        processed_frames = 0

            logger.info(f"视频将被分为 {total_chunks} 个块进行处理")
            logger.info(f"每块大小: {self.chunk_size} 帧")
            logger.info(f"邻近帧数: {self.neighbor_frames} 帧")
            logger.info(f"实时预览: {'启用' if self.preview_enabled else '禁用'}")
            logger.info(f"断点续传: {'启用' if self.resume_enabled else '禁用'}")

            # 创建视频写入器
            if start_chunk == 0:
                logger.info("正在创建输出视频...")
                self.writer = self._create_video_writer()
            else:
                # 恢复模式：使用分块写入
                logger.info("恢复模式：使用分块写入")

            # 性能统计
            chunk_times = []
            total_start_time = time.time()

            # 处理每个块
            for chunk_idx in range(start_chunk, total_chunks):
                # 检查是否收到停止信号
                if self.stop_flag:
                    logger.info("收到停止信号，正在保存进度...")
                    break

                chunk_start_time = time.time()

                # 获取帧范围
                start_frame, end_frame = self._get_frame_range(chunk_idx)
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, self.video_info["total_frames"])

                logger.debug(f"处理块 {chunk_idx + 1}/{total_chunks}: "
                           f"帧 {start_frame}-{end_frame} "
                           f"(输出: {chunk_start}-{chunk_end})")

                # 提取块
                frames, masks = self._extract_chunk(start_frame, end_frame)

                if not frames:
                    logger.warning(f"块 {chunk_idx + 1} 没有读取到帧")
                    continue

                # 处理块
                repaired_frames = self._process_chunk(
                    frames, masks, chunk_start, self.chunk_size
                )

                # 显示预览
                self._show_preview(frames, repaired_frames, masks, chunk_idx)

                # 保存输出
                if self.resume_enabled and start_chunk > 0:
                    # 恢复模式：保存到分块文件
                    self._save_chunk_video(chunk_idx, repaired_frames)
                else:
                    # 正常模式：直接写入输出视频
                    for frame in repaired_frames:
                        self.writer.write(frame)

                    # 保存帧（如果启用）
                    if self.save_frames:
                        for i, frame in enumerate(repaired_frames):
                            frame_path = os.path.join(
                                self.frames_dir,
                                f"frame_{chunk_start + i:06d}.png"
                            )
                            cv2.imwrite(frame_path, frame)

                # 更新统计
                chunk_time = time.time() - chunk_start_time
                chunk_times.append(chunk_time)
                processed_frames += len(repaired_frames)

                # 计算统计信息
                avg_time = np.mean(chunk_times)
                remaining_chunks = total_chunks - chunk_idx - 1
                eta_seconds = avg_time * remaining_chunks

                # 显存使用（如果可用）
                gpu_info = ""
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        gpu_info = f" | GPU: {gpu_memory:.1f}GB"
                except ImportError:
                    pass

                # 打印详细进度
                if self.show_progress:
                    logger.info(
                        f"Chunk {chunk_idx+1}/{total_chunks} | "
                        f"Time: {chunk_time:.1f}s | "
                        f"Avg: {avg_time:.1f}s/chunk | "
                        f"ETA: {eta_seconds/60:.1f}min"
                        f"{gpu_info}"
                    )

                # 保存进度
                if self.resume_enabled:
                    self._save_progress(chunk_idx, total_chunks, processed_frames)

            # 清理预览窗口
            if self.preview_enabled:
                cv2.destroyAllWindows()

            # 合并分块（如果是恢复模式）
            if self.resume_enabled and start_chunk > 0 and not self.stop_flag:
                self._merge_chunks()

            # 释放资源
            if self.writer:
                self.writer.release()
            if self.cap:
                self.cap.release()

            # 打印处理统计
            total_time = time.time() - total_start_time
            self._print_stats(processed_frames, total_time)

        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            raise
        finally:
            # 确保释放资源
            if self.cap:
                self.cap.release()
            if self.writer:
                self.writer.release()
            # 清理预览窗口
            if self.preview_enabled:
                cv2.destroyAllWindows()

    def _print_stats(self, processed_frames, total_time=None):
        """打印处理统计"""
        print("\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        print(f"  输入视频: {self.input_video}")
        print(f"  输出视频: {self.output_video}")
        print(f"  处理帧数: {processed_frames}")
        print(f"  视频时长: {self.video_info['duration']:.2f} 秒")
        print(f"  分辨率: {self.video_info['width']}x{self.video_info['height']}")
        if total_time:
            print(f"  处理时间: {total_time:.2f} 秒")
            if processed_frames > 0:
                fps = processed_frames / total_time
                print(f"  处理速度: {fps:.2f} 帧/秒")
        print("=" * 60)

    def __del__(self):
        """析构函数，确保释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
