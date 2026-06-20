#!/usr/bin/env python3
"""
视频马赛克去除工具 - Web 界面
==============================
基于 Gradio 的 Web 界面，提供直观的视频处理体验。
"""

import os
import sys
import json
import uuid
import time
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import gradio as gr
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_processor import VideoProcessor
from utils.mask_generator import MaskGenerator
from utils.model_runner import ModelRunner
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 全局变量
current_model = None
current_model_name = None
processing_lock = threading.Lock()
history_records = []


class ProcessingTask:
    """处理任务状态跟踪"""

    def __init__(self):
        self.task_id = str(uuid.uuid4())
        self.status = "idle"  # idle, processing, completed, failed, cancelled
        self.progress = 0.0
        self.message = ""
        self.start_time = None
        self.end_time = None
        self.output_file = None
        self.error = None

    def start(self):
        self.status = "processing"
        self.start_time = datetime.now()
        self.progress = 0.0
        self.message = "开始处理..."

    def update(self, progress: float, message: str):
        self.progress = progress
        self.message = message

    def complete(self, output_file: str):
        self.status = "completed"
        self.end_time = datetime.now()
        self.progress = 100.0
        self.message = "处理完成"
        self.output_file = output_file

    def fail(self, error: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error
        self.message = f"处理失败: {error}"

    def cancel(self):
        self.status = "cancelled"
        self.end_time = datetime.now()
        self.message = "处理已取消"


# 当前任务状态
current_task = ProcessingTask()


def init_model(model_name: str, device: str, use_fp16: bool) -> str:
    """
    初始化模型

    Args:
        model_name: 模型名称
        device: 设备
        use_fp16: 是否使用半精度

    Returns:
        str: 状态消息
    """
    global current_model, current_model_name

    try:
        config = {
            "model": {
                "name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "checkpoint_dir": "./checkpoints"
            }
        }

        current_model = ModelRunner(config)
        current_model_name = model_name

        return f"✅ 模型 {model_name} 已加载到 {device}"

    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


def process_video(
    input_file: str,
    model_name: str,
    device: str,
    use_fp16: bool,
    auto_detect: bool,
    sensitivity: float,
    chunk_size: int,
    neighbor_frames: int,
    output_format: str,
    face_restore: bool,
    face_model: str,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    """
    处理视频

    Args:
        input_file: 输入文件路径
        model_name: 模型名称
        device: 设备
        use_fp16: 是否使用半精度
        auto_detect: 是否自动检测
        sensitivity: 检测灵敏度
        chunk_size: 块大小
        neighbor_frames: 邻近帧数
        output_format: 输出格式
        face_restore: 是否启用人脸修复
        face_model: 人脸修复模型

    Returns:
        tuple: (输出文件路径, 处理日志)
    """
    global current_model, current_task

    if input_file is None:
        return None, "❌ 请上传视频文件"

    if not os.path.exists(input_file):
        return None, f"❌ 文件不存在: {input_file}"

    # 创建处理任务
    current_task = ProcessingTask()
    current_task.start()

    logs = []
    logs.append(f"开始处理: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logs.append(f"输入文件: {input_file}")
    logs.append(f"模型: {model_name}")
    logs.append(f"设备: {device}")

    try:
        # 初始化模型
        if current_model is None or current_model_name != model_name:
            logs.append("正在加载模型...")
            progress(0.01, desc="加载模型中...")
            init_status = init_model(model_name, device, use_fp16)
            logs.append(init_status)

        # 创建输出目录
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)

        # 生成输出文件名
        input_path = Path(input_file)
        output_ext = {
            "mp4": ".mp4",
            "avi": ".avi",
            "mkv": ".mkv",
            "webm": ".webm",
            "gif": ".gif",
            "image_sequence": ""
        }.get(output_format, ".mp4")

        if output_format == "image_sequence":
            output_file = str(output_dir / f"{input_path.stem}_fixed")
            os.makedirs(output_file, exist_ok=True)
        else:
            output_file = str(output_dir / f"{input_path.stem}_fixed{output_ext}")

        # 创建配置
        config = {
            "base": {
                "input_video": input_file,
                "output_video": output_file,
                "mask_file": ""
            },
            "model": {
                "name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "checkpoint_dir": "./checkpoints"
            },
            "process": {
                "chunk_size": chunk_size,
                "neighbor_frames": neighbor_frames,
                "bidirectional": True,
                "dilate_kernel": 5
            },
            "detection": {
                "auto_detect": auto_detect,
                "sensitivity": sensitivity,
                "min_area": 100,
                "morphology": True,
                "dilate_iterations": 2,
                "close_kernel": 5
            },
            "output": {
                "codec": "mp4v",
                "quality": 95,
                "save_frames": False,
                "show_progress": True,
                "preview": False
            },
            "resume": {
                "enabled": True
            }
        }

        # 创建组件
        logs.append("正在初始化处理器...")
        progress(0.05, desc="初始化处理器...")

        mask_generator = MaskGenerator(config)
        model_runner = ModelRunner(config)
        video_processor = VideoProcessor(config, mask_generator, model_runner)

        # 处理视频
        logs.append("开始处理视频...")
        progress(0.1, desc="处理视频中...")

        # 在后台线程中运行处理
        def run_processing():
            try:
                video_processor.process()
                current_task.complete(output_file)
            except Exception as e:
                current_task.fail(str(e))

        # 启动处理线程
        process_thread = threading.Thread(target=run_processing, daemon=True)
        process_thread.start()

        # 等待处理完成
        start_time = time.time()
        max_wait_time = 3600  # 最大等待时间 1 小时

        while process_thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                current_task.cancel()
                logs.append("❌ 处理超时")
                return None, "\n".join(logs)

            # 更新进度
            progress_pct = min(0.9, elapsed / 600)  # 假设 10 分钟完成
            progress(progress_pct, desc=f"处理中... ({elapsed:.0f}秒)")

            time.sleep(1)

        # 检查结果
        if current_task.status == "completed":
            logs.append(f"✅ 处理完成: {output_file}")
            progress(1.0, desc="处理完成")

            # 记录历史
            _add_history_record(
                input_file=input_file,
                output_file=output_file,
                model=model_name,
                device=device,
                duration=time.time() - start_time
            )

            return output_file, "\n".join(logs)

        elif current_task.status == "failed":
            logs.append(f"❌ 处理失败: {current_task.error}")
            return None, "\n".join(logs)

        else:
            logs.append("⚠️ 处理已取消")
            return None, "\n".join(logs)

    except Exception as e:
        logs.append(f"❌ 错误: {str(e)}")
        return None, "\n".join(logs)


def _add_history_record(input_file: str, output_file: str, model: str, device: str, duration: float):
    """添加历史记录"""
    global history_records

    record = {
        "id": str(uuid.uuid4()),
        "input_file": input_file,
        "output_file": output_file,
        "model": model,
        "device": device,
        "duration": duration,
        "timestamp": datetime.now().isoformat()
    }

    history_records.append(record)

    # 保存到文件
    history_file = Path("./history.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_records, f, indent=2, ensure_ascii=False)


def load_history() -> List[Dict]:
    """加载历史记录"""
    global history_records

    history_file = Path("./history.json")
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_records = json.load(f)
        except Exception:
            history_records = []

    return history_records


def delete_history_record(record_id: str) -> str:
    """删除历史记录"""
    global history_records

    history_records = [r for r in history_records if r["id"] != record_id]

    # 保存到文件
    history_file = Path("./history.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_records, f, indent=2, ensure_ascii=False)

    return "✅ 记录已删除"


def create_ui():
    """创建 Gradio 界面"""

    # 加载历史记录
    load_history()

    with gr.Blocks(
        title="视频马赛克去除工具 v2.0",
        theme=gr.themes.Soft()
    ) as demo:
        # 标题
        gr.Markdown("""
        # 🎬 视频马赛克去除工具 v2.0

        基于深度学习的视频马赛克去除系统，支持长视频处理、实时预览和断点续传。
        """)

        # 主标签页
        with gr.Tabs():
            # ========== 处理标签页 ==========
            with gr.TabItem("🎬 处理视频"):
                with gr.Row():
                    # 左侧：输入区
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 输入设置")

                        input_video = gr.Video(
                            label="上传视频",
                            type="filepath"
                        )

                        with gr.Row():
                            model_name = gr.Dropdown(
                                choices=["propainter", "e2fgvi", "sttn"],
                                value="propainter",
                                label="选择模型"
                            )
                            device = gr.Dropdown(
                                choices=["cuda", "cpu"],
                                value="cuda",
                                label="运行设备"
                            )

                        use_fp16 = gr.Checkbox(
                            label="使用半精度推理 (节省显存)",
                            value=True
                        )

                        gr.Markdown("### 🎯 检测设置")

                        auto_detect = gr.Checkbox(
                            label="自动检测马赛克",
                            value=True
                        )

                        sensitivity = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="检测灵敏度"
                        )

                        gr.Markdown("### ⚙️ 处理设置")

                        with gr.Row():
                            chunk_size = gr.Number(
                                value=10,
                                label="块大小",
                                minimum=1,
                                maximum=100
                            )
                            neighbor_frames = gr.Number(
                                value=10,
                                label="邻近帧数",
                                minimum=1,
                                maximum=50
                            )

                        gr.Markdown("### 📤 输出设置")

                        output_format = gr.Dropdown(
                            choices=["mp4", "avi", "mkv", "webm", "gif", "image_sequence"],
                            value="mp4",
                            label="输出格式"
                        )

                        face_restore = gr.Checkbox(
                            label="启用人脸修复",
                            value=False
                        )

                        face_model = gr.Dropdown(
                            choices=["gfpgan", "codeformer"],
                            value="gfpgan",
                            label="人脸修复模型",
                            visible=False
                        )

                        # 当勾选人脸修复时显示模型选择
                        face_restore.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[face_restore],
                            outputs=[face_model]
                        )

                        process_btn = gr.Button(
                            "🚀 开始处理",
                            variant="primary",
                            size="lg"
                        )

                    # 右侧：输出区
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 输出结果")

                        output_video = gr.Video(
                            label="处理结果",
                            interactive=False
                        )

                        process_log = gr.Textbox(
                            label="处理日志",
                            lines=15,
                            interactive=False
                        )

            # ========== 历史标签页 ==========
            with gr.TabItem("📋 处理历史"):
                history_refresh_btn = gr.Button("🔄 刷新历史")
                history_table = gr.Dataframe(
                    headers=["时间", "输入文件", "输出文件", "模型", "设备", "耗时"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False
                )

                with gr.Row():
                    history_delete_id = gr.Textbox(
                        label="输入记录 ID 以删除",
                        placeholder="输入记录 ID..."
                    )
                    history_delete_btn = gr.Button("🗑️ 删除记录", variant="stop")
                    history_status = gr.Textbox(label="状态", interactive=False)

            # ========== 帮助标签页 ==========
            with gr.TabItem("❓ 帮助"):
                gr.Markdown("""
                ## 使用说明

                ### 基本流程
                1. 上传视频文件
                2. 选择模型和参数
                3. 点击"开始处理"
                4. 等待处理完成
                5. 下载处理结果

                ### 模型选择
                - **ProPainter**: 效果最好，推荐首选
                - **E2FGVI**: 平衡效果和速度
                - **STTN**: 速度最快

                ### 参数说明
                - **块大小**: 影响显存占用和处理速度，越小越省显存
                - **邻近帧数**: 影响时序一致性，越大效果越好但更慢
                - **检测灵敏度**: 越高越容易检测到马赛克区域

                ### 常见问题
                - **显存不足**: 减小块大小或启用半精度推理
                - **处理太慢**: 使用 GPU 或选择更快的模型
                - **效果不理想**: 调整灵敏度或手动指定遮罩区域

                ### 快捷键
                - 预览窗口按 `Q` 键退出
                """)

        # ========== 事件绑定 ==========

        # 处理按钮点击
        process_btn.click(
            fn=process_video,
            inputs=[
                input_video,
                model_name,
                device,
                use_fp16,
                auto_detect,
                sensitivity,
                chunk_size,
                neighbor_frames,
                output_format,
                face_restore,
                face_model
            ],
            outputs=[output_video, process_log]
        )

        # 历史刷新按钮
        def refresh_history():
            records = load_history()
            table_data = []
            for r in records[-20:]:  # 显示最近 20 条
                table_data.append([
                    r.get("timestamp", ""),
                    os.path.basename(r.get("input_file", "")),
                    os.path.basename(r.get("output_file", "")),
                    r.get("model", ""),
                    r.get("device", ""),
                    f"{r.get('duration', 0):.1f}秒"
                ])
            return table_data

        history_refresh_btn.click(
            fn=refresh_history,
            outputs=[history_table]
        )

        # 删除历史记录
        history_delete_btn.click(
            fn=delete_history_record,
            inputs=[history_delete_id],
            outputs=[history_status]
        )

        # 页面加载时刷新历史
        demo.load(
            fn=refresh_history,
            outputs=[history_table]
        )

    return demo


def get_local_ip():
    """获取本机局域网 IP 地址"""
    import socket
    try:
        # 创建一个 UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    """主函数"""
    local_ip = get_local_ip()

    print("=" * 60)
    print("视频马赛克去除工具 v3.0 - Web 界面")
    print("=" * 60)
    print("正在启动 Web 服务器...")
    print()
    print("访问地址:")
    print(f"  电脑访问: http://localhost:7860")
    print(f"  手机访问: http://{local_ip}:7860")
    print()
    print("提示: 手机和电脑需要连接同一个 WiFi")
    print("=" * 60)
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
