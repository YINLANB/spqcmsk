#!/usr/bin/env python3
"""
使用示例
========
展示如何使用视频马赛克去除工具的各种功能。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_processor import VideoProcessor
from utils.mask_generator import MaskGenerator
from utils.model_runner import ModelRunner
from utils.logger import setup_logger

logger = setup_logger(__name__)


def example_basic():
    """基本使用示例"""
    print("\n" + "="*60)
    print("示例 1: 基本使用")
    print("="*60)

    # 配置
    config = {
        "base": {
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cuda",
            "use_fp16": True
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 10,
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
            "show_progress": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()


def example_with_mask():
    """使用遮罩文件的示例"""
    print("\n" + "="*60)
    print("示例 2: 使用遮罩文件")
    print("="*60)

    # 配置
    config = {
        "base": {
            "input_video": "input.mp4",
            "output_video": "output_with_mask.mp4",
            "mask_file": "mask.json"  # 遮罩文件
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cuda",
            "use_fp16": True
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 10,
            "bidirectional": True,
            "dilate_kernel": 5
        },
        "detection": {
            "auto_detect": False,  # 不使用自动检测
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

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()


def example_cpu_mode():
    """CPU 模式示例"""
    print("\n" + "="*60)
    print("示例 3: CPU 模式")
    print("="*60)

    # 配置
    config = {
        "base": {
            "input_video": "input.mp4",
            "output_video": "output_cpu.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cpu",  # 使用 CPU
            "use_fp16": False  # CPU 不支持半精度
        },
        "process": {
            "chunk_size": 5,  # 减小块大小
            "neighbor_frames": 5,  # 减小邻近帧数
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
            "show_progress": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()


def example_save_frames():
    """保存处理帧的示例"""
    print("\n" + "="*60)
    print("示例 4: 保存处理后的帧")
    print("="*60)

    # 配置
    config = {
        "base": {
            "input_video": "input.mp4",
            "output_video": "output_frames.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cuda",
            "use_fp16": True
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 10,
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
            "save_frames": True,  # 保存帧
            "frames_dir": "./output_frames",
            "show_progress": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()


def example_different_models():
    """使用不同模型的示例"""
    print("\n" + "="*60)
    print("示例 5: 使用不同模型")
    print("="*60)

    # 测试不同模型
    models = ["propainter", "e2fgvi", "sttn"]

    for model_name in models:
        print(f"\n测试模型: {model_name}")

        # 配置
        config = {
            "base": {
                "input_video": "input.mp4",
                "output_video": f"output_{model_name}.mp4",
                "mask_file": ""
            },
            "model": {
                "name": model_name,
                "checkpoint_dir": "./checkpoints",
                "device": "cuda",
                "use_fp16": True
            },
            "process": {
                "chunk_size": 10,
                "neighbor_frames": 10,
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
                "show_progress": True
            }
        }

        # 创建组件
        mask_generator = MaskGenerator(config)
        model_runner = ModelRunner(config)
        video_processor = VideoProcessor(config, mask_generator, model_runner)

        # 处理视频
        try:
            video_processor.process()
            print(f"模型 {model_name} 处理完成")
        except Exception as e:
            print(f"模型 {model_name} 处理失败: {e}")


def example_custom_detection():
    """自定义检测参数的示例"""
    print("\n" + "="*60)
    print("示例 6: 自定义检测参数")
    print("="*60)

    # 配置
    config = {
        "base": {
            "input_video": "input.mp4",
            "output_video": "output_custom.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cuda",
            "use_fp16": True
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 10,
            "bidirectional": True,
            "dilate_kernel": 10  # 增大膨胀核
        },
        "detection": {
            "auto_detect": True,
            "sensitivity": 0.8,  # 提高灵敏度
            "min_area": 50,  # 减小最小面积
            "morphology": True,
            "dilate_iterations": 3,  # 增加膨胀次数
            "close_kernel": 7  # 增大闭运算核
        },
        "output": {
            "codec": "mp4v",
            "quality": 95,
            "save_frames": False,
            "show_progress": True
        }
    }

    # 创建组件
    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    # 处理视频
    video_processor.process()


def main():
    """运行所有示例"""
    print("="*60)
    print("视频马赛克去除工具 - 使用示例")
    print("="*60)

    print("\n注意：以下示例需要实际的视频文件才能运行。")
    print("请确保 'input.mp4' 文件存在于当前目录。")

    # 运行示例（取消注释要运行的示例）
    # example_basic()
    # example_with_mask()
    # example_cpu_mode()
    # example_save_frames()
    # example_different_models()
    # example_custom_detection()

    print("\n要运行示例，请取消注释 main() 函数中对应的函数调用。")


if __name__ == "__main__":
    main()
