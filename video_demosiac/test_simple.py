#!/usr/bin/env python3
"""
简单测试脚本
============
测试基本功能是否正常
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("视频马赛克去除工具 - 功能测试")
print("=" * 60)

# 1. 测试 Python 版本
print("\n[1] 检查 Python 版本...")
print(f"    Python {sys.version}")

# 2. 测试核心依赖
print("\n[2] 检查核心依赖...")
try:
    import cv2
    print(f"    [OK] OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"    [FAIL] OpenCV 未安装: {e}")

try:
    import numpy as np
    print(f"    [OK] NumPy {np.__version__}")
except ImportError as e:
    print(f"    [FAIL] NumPy 未安装: {e}")

try:
    import yaml
    print(f"    [OK] PyYAML {yaml.__version__}")
except ImportError as e:
    print(f"    [FAIL] PyYAML 未安装: {e}")

try:
    import tqdm
    print(f"    [OK] tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"    [FAIL] tqdm 未安装: {e}")

# 3. 测试 Gradio
print("\n[3] 检查 Gradio...")
try:
    import gradio as gr
    print(f"    [OK] Gradio {gr.__version__}")
except ImportError as e:
    print(f"    [FAIL] Gradio 未安装，请运行: pip install gradio")

# 4. 测试项目模块
print("\n[4] 检查项目模块...")
try:
    from utils.logger import setup_logger
    print("    [OK] logger 模块")
except ImportError as e:
    print(f"    [FAIL] logger 模块: {e}")

try:
    from utils.mask_generator import MaskGenerator
    print("    [OK] mask_generator 模块")
except ImportError as e:
    print(f"    [FAIL] mask_generator 模块: {e}")

try:
    from utils.model_runner import ModelRunner
    print("    [OK] model_runner 模块")
except ImportError as e:
    print(f"    [FAIL] model_runner 模块: {e}")

try:
    from utils.video_processor import VideoProcessor
    print("    [OK] video_processor 模块")
except ImportError as e:
    print(f"    [FAIL] video_processor 模块: {e}")

try:
    from utils.face_restorer import FaceRestorer
    print("    [OK] face_restorer 模块")
except ImportError as e:
    print(f"    [FAIL] face_restorer 模块: {e}")

try:
    from utils.output_writer import create_output_writer
    print("    [OK] output_writer 模块")
except ImportError as e:
    print(f"    [FAIL] output_writer 模块: {e}")

try:
    from utils.history import ProcessingHistory
    print("    [OK] history 模块")
except ImportError as e:
    print(f"    [FAIL] history 模块: {e}")

# 5. 创建测试视频
print("\n[5] 创建测试视频...")
test_video = "test_input.mp4"
width, height = 320, 240
fps = 30
num_frames = 60  # 2 秒

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(test_video, fourcc, fps, (width, height))

for i in range(num_frames):
    # 创建彩色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color1 = int(128 + 127 * np.sin(2 * np.pi * i / num_frames))
    color2 = int(128 + 127 * np.cos(2 * np.pi * i / num_frames))
    frame[:] = (color1, color2, 200)

    # 添加移动的矩形
    x = int(50 + 100 * np.sin(2 * np.pi * i / num_frames))
    y = int(50 + 50 * np.cos(2 * np.pi * i / num_frames))
    cv2.rectangle(frame, (x, y), (x+80, y+60), (0, 255, 0), -1)

    # 添加马赛克区域
    for my in range(200, 240, 10):
        for mx in range(250, 320, 10):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv2.rectangle(frame, (mx, my), (mx+10, my+10), color, -1)

    # 添加文本
    cv2.putText(frame, f"Frame {i}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)

out.release()
print(f"    [OK] 测试视频已创建: {test_video}")

# 6. 测试视频处理
print("\n[6] 测试视频处理...")
try:
    from utils.video_processor import VideoProcessor
    from utils.mask_generator import MaskGenerator
    from utils.model_runner import ModelRunner

    config = {
        "base": {
            "input_video": test_video,
            "output_video": "test_output.mp4",
            "mask_file": ""
        },
        "model": {
            "name": "propainter",
            "checkpoint_dir": "./checkpoints",
            "device": "cpu",
            "use_fp16": False
        },
        "process": {
            "chunk_size": 10,
            "neighbor_frames": 3,
            "bidirectional": True,
            "dilate_kernel": 5
        },
        "detection": {
            "auto_detect": True,
            "sensitivity": 0.7,
            "min_area": 50,
            "morphology": True,
            "dilate_iterations": 1,
            "close_kernel": 3
        },
        "output": {
            "codec": "mp4v",
            "quality": 95,
            "save_frames": False,
            "show_progress": False,
            "preview": False
        },
        "resume": {
            "enabled": False
        }
    }

    mask_generator = MaskGenerator(config)
    model_runner = ModelRunner(config)
    video_processor = VideoProcessor(config, mask_generator, model_runner)

    print("    开始处理测试视频...")
    video_processor.process()

    if os.path.exists("test_output.mp4"):
        print("    [OK] 视频处理成功！输出文件: test_output.mp4")
    else:
        print("    [FAIL] 视频处理失败：未生成输出文件")

except Exception as e:
    print(f"    [FAIL] 视频处理失败: {e}")
    import traceback
    traceback.print_exc()

# 7. 测试历史记录
print("\n[7] 测试历史记录...")
try:
    from utils.history import ProcessingHistory

    history = ProcessingHistory("./test_history")
    record_id = history.add_record(
        input_video="test_input.mp4",
        output_video="test_output.mp4",
        model="propainter",
        device="cpu",
        duration=5.0,
        frames_processed=60
    )
    print(f"    [OK] 添加历史记录: {record_id}")

    records = history.list_records()
    print(f"    [OK] 获取历史记录: {len(records)} 条")

except Exception as e:
    print(f"    [FAIL] 历史记录测试失败: {e}")

# 清理测试文件
print("\n[8] 清理测试文件...")
for f in ["test_input.mp4", "test_output.mp4"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"    [OK] 删除: {f}")

# 清理测试历史目录
import shutil
if os.path.exists("./test_history"):
    shutil.rmtree("./test_history")
    print("    [OK] 删除测试历史目录")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
