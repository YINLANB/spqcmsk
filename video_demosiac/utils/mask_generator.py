#!/usr/bin/env python3
"""
遮罩生成器模块
==============
负责生成和处理马赛克区域的遮罩。
支持自动检测和手动指定遮罩。
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from .logger import setup_logger

logger = setup_logger(__name__)


class MaskGenerator:
    """
    遮罩生成器
    支持自动检测马赛克区域和手动指定遮罩
    """

    def __init__(self, config):
        """
        初始化遮罩生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.auto_detect = config["detection"]["auto_detect"]
        self.sensitivity = config["detection"]["sensitivity"]
        self.min_area = config["detection"]["min_area"]
        self.morphology = config["detection"]["morphology"]
        self.dilate_iterations = config["detection"]["dilate_iterations"]
        self.close_kernel = config["detection"]["close_kernel"]
        self.dilate_kernel = config["process"]["dilate_kernel"]

        # 加载手动遮罩（如果提供）
        self.manual_masks = None
        if config["base"].get("mask_file"):
            self._load_manual_masks(config["base"]["mask_file"])

    def _load_manual_masks(self, mask_file):
        """
        加载手动指定的遮罩

        Args:
            mask_file: 遮罩文件路径（JSON 格式）

        JSON 格式示例：
        {
            "frames": [
                {
                    "frame_idx": 0,
                    "masks": [
                        {
                            "x": 100,
                            "y": 100,
                            "width": 200,
                            "height": 150
                        }
                    ]
                }
            ],
            "global_masks": [
                {
                    "x": 50,
                    "y": 50,
                    "width": 100,
                    "height": 80,
                    "start_frame": 0,
                    "end_frame": 100
                }
            ]
        }
        """
        try:
            with open(mask_file, 'r', encoding='utf-8') as f:
                self.manual_masks = json.load(f)
            logger.info(f"已加载手动遮罩文件: {mask_file}")
        except Exception as e:
            logger.error(f"加载遮罩文件失败: {e}")
            self.manual_masks = None

    def generate_masks(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        为帧列表生成遮罩

        Args:
            frames: 帧列表

        Returns:
            list: 遮罩列表
        """
        if self.manual_masks:
            return self._generate_manual_masks(frames)
        elif self.auto_detect:
            return self._auto_detect_masks(frames)
        else:
            # 返回全白遮罩（无马赛克区域）
            return [np.ones(frame.shape[:2], dtype=np.uint8) * 255 for frame in frames]

    def _generate_manual_masks(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        根据手动指定的区域生成遮罩

        Args:
            frames: 帧列表

        Returns:
            list: 遮罩列表
        """
        masks = []
        frame_height, frame_width = frames[0].shape[:2]

        # 获取全局遮罩
        global_masks = self.manual_masks.get("global_masks", [])
        frame_masks = {m["frame_idx"]: m["masks"] for m in self.manual_masks.get("frames", [])}

        for i, frame in enumerate(frames):
            # 创建空白遮罩
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # 添加帧特定遮罩
            if i in frame_masks:
                for m in frame_masks[i]:
                    x, y, w, h = m["x"], m["y"], m["width"], m["height"]
                    # 裁剪到有效范围
                    x = max(0, min(x, frame_width))
                    y = max(0, min(y, frame_height))
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)
                    mask[y:y+h, x:x+w] = 255

            # 添加全局遮罩
            for gm in global_masks:
                start = gm.get("start_frame", 0)
                end = gm.get("end_frame", len(frames) - 1)
                if start <= i <= end:
                    x, y, w, h = gm["x"], gm["y"], gm["width"], gm["height"]
                    # 裁剪到有效范围
                    x = max(0, min(x, frame_width))
                    y = max(0, min(y, frame_height))
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)
                    mask[y:y+h, x:x+w] = 255

            # 膨胀遮罩
            if self.dilate_kernel > 0:
                kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            masks.append(mask)

        return masks

    def _auto_detect_masks(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        自动检测马赛克区域并生成遮罩

        Args:
            frames: 帧列表

        Returns:
            list: 遮罩列表
        """
        masks = []

        for i, frame in enumerate(frames):
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 检测马赛克区域
            mosaic_regions = self._detect_mosaic_regions(gray)

            # 创建遮罩
            mask = np.zeros(gray.shape, dtype=np.uint8)

            for region in mosaic_regions:
                x, y, w, h = region
                mask[y:y+h, x:x+w] = 255

            # 形态学处理
            if self.morphology:
                mask = self._apply_morphology(mask)

            # 膨胀遮罩
            if self.dilate_kernel > 0:
                kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)

            masks.append(mask)

        return masks

    def _detect_mosaic_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测灰度图中的马赛克区域

        Args:
            gray: 灰度图

        Returns:
            list: 检测到的区域列表 [(x, y, w, h), ...]
        """
        # 使用 Laplacian 检测块状伪影
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # 计算局部方差
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray.astype(np.float64) ** 2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean ** 2

        # 马赛克区域通常具有：
        # 1. 低方差（块内颜色均匀）
        # 2. 高梯度（块边界清晰）

        # 归一化
        laplacian_norm = np.abs(laplacian)
        laplacian_norm = (laplacian_norm / laplacian_norm.max() * 255).astype(np.uint8) if laplacian_norm.max() > 0 else laplacian_norm.astype(np.uint8)

        var_norm = (local_var / local_var.max() * 255).astype(np.uint8) if local_var.max() > 0 else local_var.astype(np.uint8)

        # 组合特征
        # 马赛克区域：低方差 + 高梯度
        mosaic_score = (255 - var_norm).astype(np.float64) * 0.5 + laplacian_norm.astype(np.float64) * 0.5

        # 应用阈值
        threshold = 255 * (1 - self.sensitivity)
        _, binary = cv2.threshold(mosaic_score.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤小区域
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= self.min_area:
                regions.append((x, y, w, h))

        return regions

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学操作优化遮罩

        Args:
            mask: 输入遮罩

        Returns:
            numpy.ndarray: 优化后的遮罩
        """
        # 闭运算填充孔洞
        close_kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # 开运算去除小噪点
        open_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        return mask

    def create_mask_from_video(self, video_path: str, output_json: str):
        """
        从视频创建遮罩标注文件

        Args:
            video_path: 视频路径
            output_json: 输出 JSON 文件路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"视频信息: {total_frames} 帧, {fps} FPS")
        print("请在视频窗口中用鼠标框选马赛克区域，按 'n' 进入下一帧，按 'q' 退出")

        frame_idx = 0
        masks_data = {"frames": [], "global_masks": []}
        current_rect = None
        drawing = False
        start_x, start_y = 0, 0

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_x, start_y, current_rect

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_x, start_y = x, y
                current_rect = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    current_rect = (start_x, start_y, x - start_x, y - start_y)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if current_rect:
                    x, y, w, h = current_rect
                    # 确保宽高为正
                    if w < 0:
                        x += w
                        w = -w
                    if h < 0:
                        y += h
                        h = -h
                    current_rect = (x, y, w, h)

        cv2.namedWindow("Mask Creator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mask Creator", mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 显示当前帧
            display = frame.copy()

            # 绘制已有遮罩
            for fm in masks_data["frames"]:
                if fm["frame_idx"] == frame_idx:
                    for m in fm["masks"]:
                        x, y, w, h = m["x"], m["y"], m["width"], m["height"]
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 绘制当前正在画的矩形
            if current_rect:
                x, y, w, h = current_rect
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # 显示帧信息
            cv2.putText(display, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Draw rectangle on mosaic, press 'n' for next frame, 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Mask Creator", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # 保存当前帧的遮罩
                if current_rect:
                    x, y, w, h = current_rect
                    frame_masks = {"frame_idx": frame_idx, "masks": [{"x": x, "y": y, "width": w, "height": h}]}
                    masks_data["frames"].append(frame_masks)
                    current_rect = None
                frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        # 保存遮罩文件
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(masks_data, f, indent=2, ensure_ascii=False)

        print(f"遮罩文件已保存到: {output_json}")

    def visualize_masks(self, frames: List[np.ndarray], masks: List[np.ndarray]):
        """
        可视化遮罩

        Args:
            frames: 帧列表
            masks: 遮罩列表
        """
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            # 创建可视化图像
            vis = frame.copy()

            # 将遮罩区域显示为红色半透明
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            red_mask = np.zeros_like(mask_3ch)
            red_mask[:, :, 2] = 255  # 红色通道

            # 应用遮罩
            mask_bool = mask > 0
            vis[mask_bool] = cv2.addWeighted(
                vis[mask_bool], 0.5,
                red_mask[mask_bool], 0.5, 0
            )

            # 显示
            cv2.imshow(f"Mask Visualization - Frame {i}", vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
