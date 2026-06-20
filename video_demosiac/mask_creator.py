#!/usr/bin/env python3
"""
遮罩创建工具
============
交互式创建视频马赛克区域的遮罩文件。
支持矩形、画笔、橡皮擦工具，以及真正的撤销/重做功能。
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MaskCreator:
    """
    交互式遮罩创建工具
    支持矩形、画笔、橡皮擦工具，以及真正的撤销/重做功能
    """

    def __init__(self, video_path: str):
        """
        初始化遮罩创建工具

        Args:
            video_path: 视频路径
        """
        self.video_path = video_path
        self.cap = None
        self.frame_idx = 0
        self.total_frames = 0
        self.fps = 0
        self.frame_height = 0
        self.frame_width = 0

        # 遮罩数据
        self.masks_data = {"frames": [], "global_masks": []}

        # 工具状态
        self.tool = "rectangle"  # rectangle, brush, eraser
        self.brush_size = 20
        self.min_brush_size = 5
        self.max_brush_size = 100

        # 绘制状态
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.current_rect = None
        self.current_mask = None  # 当前帧的累积遮罩

        # 模式
        self.mode = "frame"  # frame 或 global
        self.global_start = 0
        self.global_end = 0

        # 撤销/重做栈
        self.action_history: List[np.ndarray] = []  # 操作历史栈
        self.redo_stack: List[np.ndarray] = []  # 重做栈
        self.max_history = 50  # 最大历史记录数

        # 显示设置
        self.show_help = True
        self.show_mask_overlay = True
        self.mask_overlay_alpha = 0.4

    def _init_video(self):
        """初始化视频捕获"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化当前帧遮罩
        self.current_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        logger.info(f"视频信息:")
        logger.info(f"  路径: {self.video_path}")
        logger.info(f"  总帧数: {self.total_frames}")
        logger.info(f"  帧率: {self.fps} FPS")
        logger.info(f"  分辨率: {self.frame_width}x{self.frame_height}")
        logger.info(f"  时长: {self.total_frames / self.fps:.2f} 秒")

    def _save_action(self):
        """保存当前状态到历史栈"""
        if self.current_mask is not None:
            self.action_history.append(self.current_mask.copy())
            self.redo_stack.clear()  # 新操作清空重做栈

            # 限制历史记录大小
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)

    def _undo(self):
        """撤销操作"""
        if self.action_history:
            self.redo_stack.append(self.current_mask.copy())
            self.current_mask = self.action_history.pop()
            print("已撤销")
        else:
            print("没有可撤销的操作")

    def _redo(self):
        """重做操作"""
        if self.redo_stack:
            self.action_history.append(self.current_mask.copy())
            self.current_mask = self.redo_stack.pop()
            print("已重做")
        else:
            print("没有可重做的操作")

    def _apply_feather(self, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        应用羽化效果（边缘平滑）

        Args:
            mask: 输入遮罩
            radius: 羽化半径

        Returns:
            numpy.ndarray: 羽化后的遮罩
        """
        if radius <= 0:
            return mask

        # 使用高斯模糊实现羽化
        kernel_size = radius * 2 + 1
        feathered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return feathered

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if self.tool == "rectangle":
            self._rectangle_mouse_callback(event, x, y, flags, param)
        elif self.tool in ["brush", "eraser"]:
            self._brush_mouse_callback(event, x, y, flags, param)

    def _rectangle_mouse_callback(self, event, x, y, flags, param):
        """矩形工具的鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.start_x, self.start_y, x - self.start_x, y - self.start_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_rect:
                x, y, w, h = self.current_rect
                # 确保宽高为正
                if w < 0:
                    x += w
                    w = -w
                if h < 0:
                    y += h
                    h = -h
                self.current_rect = (x, y, w, h)

                # 保存操作并添加遮罩
                self._save_action()
                cv2.rectangle(self.current_mask, (x, y), (x + w, y + h), 255, -1)

    def _brush_mouse_callback(self, event, x, y, flags, param):
        """画笔/橡皮擦工具的鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._save_action()  # 保存操作前状态
            # 绘制第一个点
            value = 255 if self.tool == "brush" else 0
            cv2.circle(self.current_mask, (x, y), self.brush_size, value, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
                value = 255 if self.tool == "brush" else 0
                cv2.circle(self.current_mask, (x, y), self.brush_size, value, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _get_frame_mask(self, frame_idx: int) -> np.ndarray:
        """
        获取指定帧的遮罩

        Args:
            frame_idx: 帧索引

        Returns:
            numpy.ndarray: 遮罩
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        # 添加帧遮罩
        for fm in self.masks_data["frames"]:
            if fm["frame_idx"] == frame_idx:
                for m in fm["masks"]:
                    x, y, w, h = m["x"], m["y"], m["width"], m["height"]
                    # 处理不同类型的遮罩
                    if m.get("type") == "brush":
                        # 画笔遮罩（存储为 RLE 或直接像素值）
                        if "data" in m:
                            # 从压缩数据恢复
                            pass  # 简化处理
                    else:
                        # 矩形遮罩
                        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # 添加全局遮罩
        for gm in self.masks_data["global_masks"]:
            start = gm.get("start_frame", 0)
            end = gm.get("end_frame", self.total_frames - 1)
            if start <= frame_idx <= end:
                x, y, w, h = gm["x"], gm["y"], gm["width"], gm["height"]
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        return mask

    def _draw_masks(self, frame):
        """在帧上绘制遮罩"""
        display = frame.copy()

        # 获取已保存的遮罩
        saved_mask = self._get_frame_mask(self.frame_idx)

        # 叠加显示已保存的遮罩
        if self.show_mask_overlay and np.any(saved_mask > 0):
            # 创建红色遮罩层
            mask_overlay = np.zeros_like(display)
            mask_overlay[:, :, 2] = saved_mask  # 红色通道

            # 混合
            display = cv2.addWeighted(display, 1 - self.mask_overlay_alpha,
                                     mask_overlay, self.mask_overlay_alpha, 0)

        # 叠加显示当前正在编辑的遮罩
        if np.any(self.current_mask > 0):
            current_overlay = np.zeros_like(display)
            current_overlay[:, :, 1] = self.current_mask  # 绿色通道
            display = cv2.addWeighted(display, 1 - self.mask_overlay_alpha,
                                     current_overlay, self.mask_overlay_alpha, 0)

        # 绘制当前正在画的矩形
        if self.current_rect and self.tool == "rectangle":
            x, y, w, h = self.current_rect
            color = (0, 255, 0) if self.mode == "frame" else (255, 0, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # 绘制画笔光标
        if self.tool in ["brush", "eraser"] and hasattr(self, '_mouse_x'):
            color = (0, 255, 0) if self.tool == "brush" else (0, 0, 255)
            cv2.circle(display, (self._mouse_x, self._mouse_y),
                      self.brush_size, color, 2)

        return display

    def _show_help(self, display):
        """显示帮助信息"""
        height, width = display.shape[:2]

        # 创建帮助面板
        panel_height = 180
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        # 工具颜色
        tool_colors = {
            "rectangle": (0, 255, 0),
            "brush": (0, 200, 0),
            "eraser": (0, 0, 255)
        }
        tool_color = tool_colors.get(self.tool, (255, 255, 255))

        # 显示帮助信息
        help_texts = [
            f"Frame: {self.frame_idx}/{self.total_frames} | Mode: {self.mode.upper()} | Tool: {self.tool.upper()}",
            f"Brush Size: {self.brush_size} | History: {len(self.action_history)} | Redo: {len(self.redo_stack)}",
            "",
            "Tools: [1] Rectangle  [2] Brush  [3] Eraser  [+/-] Brush Size",
            "Edit:  [Ctrl+Z] Undo  [Ctrl+Y] Redo  [D] Delete Last  [R] Reset",
            "View:  [H] Help  [V] Toggle Overlay  [</>] Overlay Alpha",
            "Mode:  [G] Global  [F] Frame  [N] Next  [P] Previous",
            "File:  [S] Save  [Q] Quit"
        ]

        for i, text in enumerate(help_texts):
            color = tool_color if i == 0 else (200, 200, 200)
            cv2.putText(panel, text, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 合并面板
        return np.vstack([display, panel])

    def _update_frame(self):
        """更新当前帧的遮罩"""
        # 加载已保存的遮罩
        saved_mask = self._get_frame_mask(self.frame_idx)

        # 合并当前编辑的遮罩
        if np.any(self.current_mask > 0):
            combined = cv2.bitwise_or(saved_mask, self.current_mask)
        else:
            combined = saved_mask

        return combined

    def run(self):
        """运行遮罩创建工具"""
        try:
            self._init_video()

            print("\n" + "="*60)
            print("遮罩创建工具 v2.0")
            print("="*60)
            print("新增功能：")
            print("  - 画笔工具 (按 2)")
            print("  - 橡皮擦工具 (按 3)")
            print("  - 真正的撤销/重做 (Ctrl+Z / Ctrl+Y)")
            print("  - 遮罩羽化")
            print("="*60 + "\n")

            # 创建窗口
            cv2.namedWindow("Mask Creator", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Mask Creator", self._mouse_callback)

            # 为每一帧存储遮罩
            frame_masks = {}

            while True:
                # 定位到当前帧
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                ret, frame = self.cap.read()

                if not ret:
                    logger.error(f"无法读取帧 {self.frame_idx}")
                    break

                # 加载当前帧的遮罩
                if self.frame_idx not in frame_masks:
                    frame_masks[self.frame_idx] = self.current_mask.copy()
                else:
                    self.current_mask = frame_masks[self.frame_idx]

                # 绘制遮罩
                display = self._draw_masks(frame)

                # 显示帮助
                if self.show_help:
                    display = self._show_help(display)

                # 显示帧信息
                cv2.imshow("Mask Creator", display)

                # 等待按键
                key = cv2.waitKey(30) & 0xFF

                # 保存当前帧遮罩到帧字典
                frame_masks[self.frame_idx] = self.current_mask.copy()

                if key == ord('q'):
                    # 退出
                    response = input("\n是否保存遮罩？(y/n): ")
                    if response.lower() == 'y':
                        self._save_masks(frame_masks)
                    break

                elif key == ord('n'):
                    # 下一帧
                    if self.frame_idx < self.total_frames - 1:
                        self.frame_idx += 1
                        self.current_rect = None
                        self.action_history.clear()
                        self.redo_stack.clear()

                elif key == ord('p'):
                    # 上一帧
                    if self.frame_idx > 0:
                        self.frame_idx -= 1
                        self.current_rect = None
                        self.action_history.clear()
                        self.redo_stack.clear()

                elif key == ord('s'):
                    # 保存遮罩
                    self._save_masks(frame_masks)

                elif key == ord('g'):
                    # 切换到全局模式
                    self.mode = "global"
                    print(f"切换到全局遮罩模式 (当前帧: {self.frame_idx})")
                    self.global_start = self.frame_idx
                    response = input("输入结束帧号（或按 Enter 使用当前帧）: ")
                    if response.strip():
                        self.global_end = int(response)
                    else:
                        self.global_end = self.frame_idx

                elif key == ord('f'):
                    # 切换到帧模式
                    self.mode = "frame"
                    print(f"切换到帧遮罩模式 (当前帧: {self.frame_idx})")

                elif key == ord('1'):
                    # 矩形工具
                    self.tool = "rectangle"
                    print("切换到矩形工具")

                elif key == ord('2'):
                    # 画笔工具
                    self.tool = "brush"
                    print(f"切换到画笔工具 (大小: {self.brush_size})")

                elif key == ord('3'):
                    # 橡皮擦工具
                    self.tool = "eraser"
                    print(f"切换到橡皮擦工具 (大小: {self.brush_size})")

                elif key in [ord('+'), ord('=')]:
                    # 增大画笔
                    self.brush_size = min(self.max_brush_size, self.brush_size + 5)
                    print(f"画笔大小: {self.brush_size}")

                elif key in [ord('-'), ord('_')]:
                    # 减小画笔
                    self.brush_size = max(self.min_brush_size, self.brush_size - 5)
                    print(f"画笔大小: {self.brush_size}")

                elif key == ord('d'):
                    # 删除最后一个遮罩
                    self._delete_last_mask(frame_masks)

                elif key == 26:  # Ctrl+Z
                    # 撤销
                    self._undo()

                elif key == 25:  # Ctrl+Y
                    # 重做
                    self._redo()

                elif key == ord('r'):
                    # 重置
                    response = input("确定要重置当前帧的遮罩吗？(y/n): ")
                    if response.lower() == 'y':
                        self.current_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                        frame_masks[self.frame_idx] = self.current_mask.copy()
                        self.action_history.clear()
                        self.redo_stack.clear()
                        print("已重置当前帧遮罩")

                elif key == ord('h'):
                    # 显示/隐藏帮助
                    self.show_help = not self.show_help

                elif key == ord('v'):
                    # 切换遮罩叠加显示
                    self.show_mask_overlay = not self.show_mask_overlay
                    print(f"遮罩叠加显示: {'开启' if self.show_mask_overlay else '关闭'}")

                elif key == ord('.'):
                    # 增加叠加透明度
                    self.mask_overlay_alpha = min(0.8, self.mask_overlay_alpha + 0.1)
                    print(f"叠加透明度: {self.mask_overlay_alpha:.1f}")

                elif key == ord(','):
                    # 减少叠加透明度
                    self.mask_overlay_alpha = max(0.1, self.mask_overlay_alpha - 0.1)
                    print(f"叠加透明度: {self.mask_overlay_alpha:.1f}")

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

    def _delete_last_mask(self, frame_masks):
        """删除最后一个遮罩"""
        if self.mode == "frame":
            # 删除当前帧的最后一个遮罩
            for fm in self.masks_data["frames"]:
                if fm["frame_idx"] == self.frame_idx and fm["masks"]:
                    fm["masks"].pop()
                    # 重新生成当前帧遮罩
                    self.current_mask = self._get_frame_mask(self.frame_idx)
                    frame_masks[self.frame_idx] = self.current_mask.copy()
                    print(f"已删除帧 {self.frame_idx} 的最后一个遮罩")
                    return
            print("当前帧没有遮罩")

        elif self.mode == "global":
            # 删除最后一个全局遮罩
            if self.masks_data["global_masks"]:
                self.masks_data["global_masks"].pop()
                # 重新生成当前帧遮罩
                self.current_mask = self._get_frame_mask(self.frame_idx)
                frame_masks[self.frame_idx] = self.current_mask.copy()
                print("已删除最后一个全局遮罩")
            else:
                print("没有全局遮罩")

    def _save_masks(self, frame_masks):
        """保存遮罩"""
        output_file = Path(self.video_path).stem + "_masks.json"

        response = input(f"输入保存文件名 (默认: {output_file}): ")
        if response.strip():
            output_file = response

        # 将画笔遮罩转换为可保存的格式
        frames_data = []
        for frame_idx, mask in frame_masks.items():
            if np.any(mask > 0):
                # 找到遮罩区域
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                masks_list = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    masks_list.append({"x": x, "y": y, "width": w, "height": h, "type": "rectangle"})
                if masks_list:
                    frames_data.append({"frame_idx": frame_idx, "masks": masks_list})

        # 更新 masks_data
        self.masks_data["frames"] = frames_data

        # 保存到 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.masks_data, f, indent=2, ensure_ascii=False)

        print(f"遮罩已保存到: {output_file}")

        # 打印统计
        frame_masks_count = len(self.masks_data["frames"])
        global_masks = len(self.masks_data["global_masks"])
        print(f"统计: {frame_masks_count} 个帧遮罩, {global_masks} 个全局遮罩")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="交互式遮罩创建工具 v2.0")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入视频路径"
    )

    args = parser.parse_args()

    # 创建遮罩创建器
    creator = MaskCreator(args.input)
    creator.run()


if __name__ == "__main__":
    main()
