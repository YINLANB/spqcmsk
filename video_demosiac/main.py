#!/usr/bin/env python3
"""
视频马赛克去除工具
==================
基于深度学习的视频马赛克去除系统，支持长视频处理和高精度修复。

主要功能：
1. 自动检测视频中的马赛克区域
2. 使用深度学习模型进行高质量修复
3. 支持长视频分块处理，避免显存溢出
4. 保持视频时序一致性
5. 实时预览修复效果
6. 断点续传，中断后可继续处理
7. 详细的处理进度和统计信息

使用方法：
    python main.py --input video.mp4 --output output.mp4

    # 使用自动检测
    python main.py --input video.mp4 --output output.mp4 --auto-detect

    # 指定马赛克区域（JSON 格式）
    python main.py --input video.mp4 --output output.mp4 --mask mask.json

    # 启用实时预览
    python main.py --input video.mp4 --output output.mp4 --preview

    # 使用配置文件
    python main.py --config configs/config.yaml

作者：Video Demosiac Team
版本：2.0.0
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

from utils.video_processor import VideoProcessor
from utils.mask_generator import MaskGenerator
from utils.model_runner import ModelRunner
from utils.logger import setup_logger, create_log_file

logger = setup_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频马赛克去除工具 - 基于深度学习的高质量修复",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 基本使用
    python main.py --input input.mp4 --output output.mp4

    # 自动检测马赛克
    python main.py --input input.mp4 --output output.mp4 --auto-detect

    # 启用实时预览
    python main.py --input input.mp4 --output output.mp4 --preview

    # 使用 CPU 处理
    python main.py --input input.mp4 --output output.mp4 --device cpu

    # 使用配置文件
    python main.py --config configs/config.yaml
        """
    )

    # 必需参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入视频路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出视频路径"
    )

    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)"
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        choices=["propainter", "e2fgvi", "sttn"],
        default="propainter",
        help="选择模型 (默认: propainter)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="运行设备 (默认: 自动检测)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用半精度推理"
    )

    # 处理参数
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="处理块大小 (默认: 10)"
    )
    parser.add_argument(
        "--neighbor-frames",
        type=int,
        default=10,
        help="时序邻近帧数 (默认: 10)"
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="启用双向传播"
    )

    # 马赛克检测参数
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="启用自动马赛克检测"
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="马赛克区域遮罩文件路径 (JSON 格式)"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="自动检测灵敏度 (0-1，默认: 0.7)"
    )

    # 输出参数
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="输出视频质量 (0-100，默认: 95)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="保存处理后的帧"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示处理进度"
    )

    # 预览参数
    parser.add_argument(
        "--preview",
        action="store_true",
        help="启用实时预览"
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="预览窗口缩放比例 (0.1-1.0，默认: 0.5)"
    )

    # 断点续传参数
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传"
    )

    # 日志参数
    parser.add_argument(
        "--log-file",
        type=str,
        help="日志文件路径（可选）"
    )

    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def validate_args(args):
    """验证参数"""
    # 验证输入文件
    if args.input and not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return False

    # 验证遮罩文件
    if args.mask and not os.path.exists(args.mask):
        logger.error(f"遮罩文件不存在: {args.mask}")
        return False

    # 验证检查点目录
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 验证输出路径
    if args.output:
        output_dir = Path(args.output).parent
        if output_dir and not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"无法创建输出目录: {output_dir}, 错误: {e}")
                return False

    return True


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确保配置字典存在
    if not config:
        config = {}

    # 初始化默认配置
    if "base" not in config:
        config["base"] = {}
    if "model" not in config:
        config["model"] = {}
    if "process" not in config:
        config["process"] = {}
    if "detection" not in config:
        config["detection"] = {}
    if "output" not in config:
        config["output"] = {}
    if "resume" not in config:
        config["resume"] = {}

    # 合并命令行参数和配置文件
    # 命令行参数优先级更高
    if args.input:
        config["base"]["input_video"] = args.input
    if args.output:
        config["base"]["output_video"] = args.output

    # 检查必需参数
    if not config.get("base", {}).get("input_video"):
        logger.error("请指定输入视频路径 (--input 或 --config)")
        sys.exit(1)

    if not config.get("base", {}).get("output_video"):
        # 自动生成输出文件名
        input_path = Path(config["base"]["input_video"])
        config["base"]["output_video"] = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")

    # 更新配置
    if args.model:
        config["model"]["name"] = args.model
    if args.device:
        config["model"]["device"] = args.device
    if args.fp16:
        config["model"]["use_fp16"] = True
    if args.chunk_size:
        config["process"]["chunk_size"] = args.chunk_size
    if args.neighbor_frames:
        config["process"]["neighbor_frames"] = args.neighbor_frames
    if args.bidirectional:
        config["process"]["bidirectional"] = True
    if args.auto_detect:
        config["detection"]["auto_detect"] = True
    if args.mask:
        config["base"]["mask_file"] = args.mask
    if args.sensitivity:
        config["detection"]["sensitivity"] = args.sensitivity
    if args.quality:
        config["output"]["quality"] = args.quality
    if args.save_frames:
        config["output"]["save_frames"] = True
    if args.no_progress:
        config["output"]["show_progress"] = False
    if args.preview:
        config["output"]["preview"] = True
    if args.preview_scale:
        config["output"]["preview_scale"] = args.preview_scale
    if args.no_resume:
        config["resume"]["enabled"] = False

    # 设置日志文件
    log_file = args.log_file
    if not log_file:
        log_file = create_log_file(config["base"]["output_video"])

    # 显示配置信息
    logger.info("=" * 60)
    logger.info("视频马赛克去除工具 v2.0.0")
    logger.info("=" * 60)
    logger.info(f"输入视频: {config['base']['input_video']}")
    logger.info(f"输出视频: {config['base']['output_video']}")
    logger.info(f"使用模型: {config['model']['name']}")
    logger.info(f"运行设备: {config['model']['device']}")
    logger.info(f"实时预览: {'启用' if config['output'].get('preview', False) else '禁用'}")
    logger.info(f"断点续传: {'启用' if config['resume'].get('enabled', True) else '禁用'}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)

    # 验证参数
    if not validate_args(args):
        sys.exit(1)

    try:
        # 初始化组件
        logger.info("正在初始化组件...")

        # 遮罩生成器
        mask_generator = MaskGenerator(config)

        # 模型运行器
        model_runner = ModelRunner(config)

        # 视频处理器
        video_processor = VideoProcessor(config, mask_generator, model_runner)

        # 开始处理
        logger.info("开始处理视频...")
        video_processor.process()

        logger.info("处理完成！")
        logger.info(f"输出视频已保存到: {config['base']['output_video']}")

    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(0)
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
