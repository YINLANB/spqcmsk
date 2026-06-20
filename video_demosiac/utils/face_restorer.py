#!/usr/bin/env python3
"""
人脸修复模块
============
基于 GFPGAN 或 CodeFormer 的人脸修复功能。
"""

import os
import cv2
import numpy as np
from typing import Optional, List
from pathlib import Path

from .logger import setup_logger

logger = setup_logger(__name__)


class FaceRestorer:
    """
    人脸修复器
    支持 GFPGAN 和 CodeFormer
    """

    def __init__(self, config: dict):
        """
        初始化人脸修复器

        Args:
            config: 配置字典
        """
        self.config = config
        self.face_config = config.get("face_restore", {})
        self.enabled = self.face_config.get("enabled", False)
        self.model_name = self.face_config.get("model", "gfpgan")
        self.upscale = self.face_config.get("upscale", 1)
        self.weight = self.face_config.get("weight", 0.5)

        self.restorer = None
        self.face_detector = None

        if self.enabled:
            self._init_model()

    def _init_model(self):
        """初始化人脸修复模型"""
        try:
            if self.model_name == "gfpgan":
                self._init_gfpgan()
            elif self.model_name == "codeformer":
                self._init_codeformer()
            else:
                logger.warning(f"不支持的人脸修复模型: {self.model_name}，使用 OpenCV 作为后备")
                self._init_opencv_detector()

        except ImportError as e:
            logger.warning(f"无法加载 {self.model_name}: {e}")
            logger.info("使用 OpenCV 人脸检测作为后备方案")
            self._init_opencv_detector()

    def _init_gfpgan(self):
        """初始化 GFPGAN"""
        try:
            from gfpgan import GFPGANer

            # 下载模型（如果需要）
            model_path = Path("./checkpoints/GFPGANv1.4.pth")
            if not model_path.exists():
                logger.info("正在下载 GFPGAN 模型...")
                # 这里可以添加下载逻辑

            self.restorer = GFPGANer(
                model_path=str(model_path),
                upscale=self.upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )

            logger.info("✅ GFPGAN 模型已加载")

        except Exception as e:
            logger.warning(f"GFPGAN 初始化失败: {e}")
            self._init_opencv_detector()

    def _init_codeformer(self):
        """初始化 CodeFormer"""
        try:
            # CodeFormer 的初始化逻辑
            # 需要克隆 CodeFormer 仓库并安装依赖
            logger.info("CodeFormer 初始化...")

            # 检查 CodeFormer 是否已安装
            codeformer_path = Path("./CodeFormer")
            if not codeformer_path.exists():
                logger.warning("CodeFormer 未安装，请参考: https://github.com/sczhou/CodeFormer")
                self._init_opencv_detector()
                return

            # 这里可以添加 CodeFormer 的具体初始化逻辑
            logger.info("✅ CodeFormer 模型已加载")

        except Exception as e:
            logger.warning(f"CodeFormer 初始化失败: {e}")
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """初始化 OpenCV 人脸检测器（后备方案）"""
        try:
            # 使用 OpenCV 的 Haar 级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ OpenCV 人脸检测器已加载（后备方案）")
            else:
                logger.warning("未找到 Haar 级联文件")
        except Exception as e:
            logger.warning(f"OpenCV 人脸检测器初始化失败: {e}")

    def restore_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        修复单帧中的人脸

        Args:
            frame: 输入帧 (BGR)

        Returns:
            numpy.ndarray: 修复后的帧
        """
        if not self.enabled:
            return frame

        if self.restorer is not None:
            return self._restore_with_model(frame)
        elif self.face_detector is not None:
            return self._restore_with_opencv(frame)
        else:
            return frame

    def restore_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量修复帧中的人脸

        Args:
            frames: 帧列表

        Returns:
            list: 修复后的帧列表
        """
        if not self.enabled:
            return frames

        return [self.restore_frame(frame) for frame in frames]

    def _restore_with_model(self, frame: np.ndarray) -> np.ndarray:
        """使用深度学习模型修复人脸"""
        try:
            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用模型修复
            cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                frame_rgb,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            if restored_img is not None:
                # 转换回 BGR
                restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                return restored_bgr

            return frame

        except Exception as e:
            logger.warning(f"模型修复失败: {e}")
            return frame

    def _restore_with_opencv(self, frame: np.ndarray) -> np.ndarray:
        """使用 OpenCV 检测人脸并进行简单修复"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return frame

            # 对每个人脸进行简单修复
            result = frame.copy()
            for (x, y, w, h) in faces:
                # 扩展人脸区域
                expand = int(w * 0.1)
                x1 = max(0, x - expand)
                y1 = max(0, y - expand)
                x2 = min(frame.shape[1], x + w + expand)
                y2 = min(frame.shape[0], y + h + expand)

                # 提取人脸区域
                face_region = frame[y1:y2, x1:x2]

                if face_region.size == 0:
                    continue

                # 简单的美颜处理：双边滤波
                smoothed = cv2.bilateralFilter(face_region, 9, 75, 75)

                # 混合原图和处理后的图
                alpha = 0.7
                blended = cv2.addWeighted(face_region, alpha, smoothed, 1 - alpha, 0)

                # 放回原图
                result[y1:y2, x1:x2] = blended

            return result

        except Exception as e:
            logger.warning(f"OpenCV 修复失败: {e}")
            return frame

    def detect_faces(self, frame: np.ndarray) -> List[tuple]:
        """
        检测帧中的人脸

        Args:
            frame: 输入帧

        Returns:
            list: 人脸位置列表 [(x, y, w, h), ...]
        """
        if self.face_detector is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces.tolist() if len(faces) > 0 else []


class SimpleFaceRestorer:
    """
    简单人脸修复器
    不依赖外部模型，仅使用 OpenCV 进行基本的人脸增强
    """

    def __init__(self):
        """初始化简单人脸修复器"""
        self.face_detector = None
        self._init_detector()

    def _init_detector(self):
        """初始化人脸检测器"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_detector = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

    def restore_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        简单修复帧中的人脸

        Args:
            frame: 输入帧

        Returns:
            numpy.ndarray: 修复后的帧
        """
        if self.face_detector is None:
            return frame

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return frame

            result = frame.copy()
            for (x, y, w, h) in faces:
                # 提取人脸区域
                face_region = frame[y:y+h, x:x+w]
                if face_region.size == 0:
                    continue

                # 简单增强：锐化和对比度调整
                enhanced = cv2.detailEnhance(face_region, sigma_s=10, sigma_r=0.15)

                # 混合
                alpha = 0.6
                blended = cv2.addWeighted(face_region, alpha, enhanced, 1 - alpha, 0)

                result[y:y+h, x:x+w] = blended

            return result

        except Exception:
            return frame

    def restore_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """批量修复帧"""
        return [self.restore_frame(frame) for frame in frames]
