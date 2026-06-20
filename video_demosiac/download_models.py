#!/usr/bin/env python3
"""
模型下载脚本
============
下载视频修复所需的预训练模型。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_runner import ModelRunner
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument(
        "--model",
        type=str,
        choices=["propainter", "e2fgvi", "sttn", "all"],
        default="propainter",
        help="要下载的模型 (默认: propainter)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="模型保存目录 (默认: ./checkpoints)"
    )

    args = parser.parse_args()

    # 创建配置
    config = {
        "model": {
            "name": args.model if args.model != "all" else "propainter",
            "device": "cpu",
            "use_fp16": False,
            "checkpoint_dir": args.checkpoint_dir
        }
    }

    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 初始化模型运行器
    runner = ModelRunner(config)

    # 下载模型
    if args.model == "all":
        models = ["propainter", "e2fgvi", "sttn"]
    else:
        models = [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"下载模型: {model_name}")
        print(f"{'='*60}")

        config["model"]["name"] = model_name
        runner = ModelRunner(config)
        runner.download_model(model_name)

    print("\n" + "="*60)
    print("下载完成！")
    print("="*60)


if __name__ == "__main__":
    main()
