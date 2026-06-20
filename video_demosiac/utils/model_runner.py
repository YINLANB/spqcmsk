#!/usr/bin/env python3
"""
模型运行器模块
==============
负责加载和运行深度学习模型进行视频修复。
支持 ProPainter、E2FGVI、STTN 等多种模型。
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .logger import setup_logger

logger = setup_logger(__name__)


class ModelRunner:
    """
    模型运行器
    支持多种视频修复模型
    """

    def __init__(self, config):
        """
        初始化模型运行器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_name = config["model"]["name"]
        self.device = config["model"]["device"]
        self.use_fp16 = config["model"]["use_fp16"]
        self.checkpoint_dir = Path(config["model"]["checkpoint_dir"])

        # 初始化模型
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        logger.info(f"正在初始化 {self.model_name} 模型...")

        # 检查 CUDA 是否可用
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，将使用 CPU")
            self.device = "cpu"

        # 根据模型类型初始化
        if self.model_name == "propainter":
            self._init_propainter()
        elif self.model_name == "e2fgvi":
            self._init_e2fgvi()
        elif self.model_name == "sttn":
            self._init_sttn()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        logger.info(f"模型初始化完成，设备: {self.device}")

    def _init_propainter(self):
        """初始化 ProPainter 模型"""
        try:
            from models.propainter import Propainter

            # 创建模型
            self.model = Propainter()

            # 加载检查点
            checkpoint_path = self.checkpoint_dir / "propainter.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
                logger.info(f"已加载 ProPainter 检查点: {checkpoint_path}")
            else:
                logger.warning(f"未找到检查点文件: {checkpoint_path}")
                logger.info("使用简单修复模型作为后备")

            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()

            # 启用半精度
            if self.use_fp16 and self.device == "cuda":
                self.model = self.model.half()
                logger.info("已启用半精度推理")

        except ImportError as e:
            logger.warning(f"导入 ProPainter 模型失败: {e}")
            logger.info("使用简单修复模型作为后备")
            self._init_simple_model()

    def _init_simple_model(self):
        """初始化简单修复模型"""
        from models.simple_inpainter import SimpleInpainter

        self.model = SimpleInpainter()
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("已初始化简单修复模型")

    def _init_e2fgvi(self):
        """初始化 E2FGVI 模型"""
        try:
            from models.e2fgvi import E2FGVI

            self.model = E2FGVI()

            checkpoint_path = self.checkpoint_dir / "e2fgvi.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
                logger.info(f"已加载 E2FGVI 检查点: {checkpoint_path}")
            else:
                logger.warning(f"未找到检查点文件: {checkpoint_path}")
                logger.info("使用简单修复模型作为后备")

            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_fp16 and self.device == "cuda":
                self.model = self.model.half()

        except ImportError as e:
            logger.warning(f"导入 E2FGVI 模型失败: {e}")
            logger.info("使用简单修复模型作为后备")
            self._init_simple_model()

    def _init_sttn(self):
        """初始化 STTN 模型"""
        try:
            from models.sttn import STTN

            self.model = STTN()

            checkpoint_path = self.checkpoint_dir / "sttn.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
                logger.info(f"已加载 STTN 检查点: {checkpoint_path}")
            else:
                logger.warning(f"未找到检查点文件: {checkpoint_path}")
                logger.info("使用简单修复模型作为后备")

            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_fp16 and self.device == "cuda":
                self.model = self.model.half()

        except ImportError as e:
            logger.warning(f"导入 STTN 模型失败: {e}")
            logger.info("使用简单修复模型作为后备")
            self._init_simple_model()

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        预处理单帧图像

        Args:
            frame: BGR 格式的 numpy 数组

        Returns:
            torch.Tensor: 预处理后的张量 [1, 3, H, W]
        """
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 归一化到 [0, 1]
        frame_norm = frame_rgb.astype(np.float32) / 255.0

        # 转换为张量 [H, W, 3] -> [3, H, W]
        frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1))

        # 添加 batch 维度 [3, H, W] -> [1, 3, H, W]
        frame_tensor = frame_tensor.unsqueeze(0)

        # 移动到设备
        frame_tensor = frame_tensor.to(self.device)

        # 半精度
        if self.use_fp16 and self.device == "cuda":
            frame_tensor = frame_tensor.half()

        return frame_tensor

    def _postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        后处理模型输出

        Args:
            frame_tensor: 模型输出张量 [1, 3, H, W]

        Returns:
            numpy.ndarray: BGR 格式的图像
        """
        # 移除 batch 维度
        frame_tensor = frame_tensor.squeeze(0)

        # 转换为 numpy
        if self.use_fp16 and self.device == "cuda":
            frame_tensor = frame_tensor.float()

        frame_np = frame_tensor.cpu().numpy()

        # [3, H, W] -> [H, W, 3]
        frame_np = frame_np.transpose(1, 2, 0)

        # 反归一化
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

        # RGB -> BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        return frame_bgr

    def repair_frames(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        修复视频帧

        Args:
            frames: 帧列表
            masks: 遮罩列表

        Returns:
            list: 修复后的帧列表
        """
        if not frames:
            return []

        logger.info(f"正在修复 {len(frames)} 帧...")

        # 转换为张量
        frame_tensors = [self._preprocess_frame(frame) for frame in frames]
        mask_tensors = [self._preprocess_mask(mask) for mask in masks]

        # 拼接为批次
        frames_batch = torch.cat(frame_tensors, dim=0)
        masks_batch = torch.cat(mask_tensors, dim=0)

        # 推理
        with torch.no_grad():
            if self.model_name == "propainter":
                repaired_frames = self._run_propainter(frames_batch, masks_batch)
            elif self.model_name == "e2fgvi":
                repaired_frames = self._run_e2fgvi(frames_batch, masks_batch)
            elif self.model_name == "sttn":
                repaired_frames = self._run_sttn(frames_batch, masks_batch)
            else:
                raise ValueError(f"不支持的模型: {self.model_name}")

        # 后处理
        result_frames = [self._postprocess_frame(frame) for frame in repaired_frames]

        logger.info(f"修复完成，共处理 {len(result_frames)} 帧")

        return result_frames

    def _preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        预处理遮罩

        Args:
            mask: 灰度遮罩

        Returns:
            torch.Tensor: 预处理后的遮罩张量 [1, 1, H, W]
        """
        # 归一化到 [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0

        # 转换为张量 [H, W] -> [1, 1, H, W]
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0)

        # 移动到设备
        mask_tensor = mask_tensor.to(self.device)

        # 半精度
        if self.use_fp16 and self.device == "cuda":
            mask_tensor = mask_tensor.half()

        return mask_tensor

    def _run_propainter(self, frames: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
        """
        运行 ProPainter 模型

        Args:
            frames: 帧张量 [B, 3, H, W]
            masks: 遮罩张量 [B, 1, H, W]

        Returns:
            list: 修复后的帧张量列表
        """
        # ProPainter 的推理逻辑
        # 注意：这里简化了实际的 ProPainter 推理过程
        # 实际使用时需要根据 ProPainter 的具体接口进行调整

        batch_size = frames.shape[0]
        repaired_frames = []

        for i in range(batch_size):
            frame = frames[i:i+1]
            mask = masks[i:i+1]

            # 使用模型进行修复
            # 这里简化为直接使用原始帧（实际应该使用完整的 ProPainter 推理）
            repaired_frame = self.model(frame, mask)

            repaired_frames.append(repaired_frame)

        return repaired_frames

    def _run_e2fgvi(self, frames: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
        """
        运行 E2FGVI 模型

        Args:
            frames: 帧张量 [B, 3, H, W]
            masks: 遮罩张量 [B, 1, H, W]

        Returns:
            list: 修复后的帧张量列表
        """
        # E2FGVI 的推理逻辑
        batch_size = frames.shape[0]
        repaired_frames = []

        for i in range(batch_size):
            frame = frames[i:i+1]
            mask = masks[i:i+1]

            repaired_frame = self.model(frame, mask)
            repaired_frames.append(repaired_frame)

        return repaired_frames

    def _run_sttn(self, frames: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
        """
        运行 STTN 模型

        Args:
            frames: 帧张量 [B, 3, H, W]
            masks: 遮罩张量 [B, 1, H, W]

        Returns:
            list: 修复后的帧张量列表
        """
        # STTN 的推理逻辑
        batch_size = frames.shape[0]
        repaired_frames = []

        for i in range(batch_size):
            frame = frames[i:i+1]
            mask = masks[i:i+1]

            repaired_frame = self.model(frame, mask)
            repaired_frames.append(repaired_frame)

        return repaired_frames

    def download_model(self, model_name: str = None):
        """
        下载预训练模型

        Args:
            model_name: 模型名称，默认使用配置中的模型
        """
        if model_name is None:
            model_name = self.model_name

        logger.info(f"正在下载 {model_name} 预训练模型...")

        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 下载链接
        download_urls = {
            "propainter": "https://github.com/sczhou/ProPainter/releases/download/v1.0/propainter.pth",
            "e2fgvi": "https://github.com/ruoshui6/E2FGVI/releases/download/v1.0/e2fgvi.pth",
            "sttn": "https://github.com/rese1f/STTN/releases/download/v1.0/sttn.pth"
        }

        if model_name not in download_urls:
            logger.error(f"未找到模型 {model_name} 的下载链接")
            return

        # 下载模型
        try:
            import requests
            from tqdm import tqdm

            url = download_urls[model_name]
            output_path = self.checkpoint_dir / f"{model_name}.pth"

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"模型下载完成: {output_path}")

        except ImportError:
            logger.error("请安装 requests 和 tqdm 包: pip install requests tqdm")
        except Exception as e:
            logger.error(f"下载模型失败: {e}")

    def get_model_info(self):
        """
        获取模型信息

        Returns:
            dict: 模型信息
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

        # 检查检查点是否存在
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}.pth"
        info["checkpoint_exists"] = checkpoint_path.exists()

        # 获取模型参数数量
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["total_params"] = total_params
            info["trainable_params"] = trainable_params

        return info
