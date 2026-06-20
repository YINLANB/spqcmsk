# 视频马赛克去除工具 v3.0

基于深度学习的视频马赛克去除系统，支持 Web 界面、人脸修复、多格式输出和处理历史。

## ✨ 功能特点

### 核心功能
- 🎯 **自动检测**：智能识别视频中的马赛克区域
- 🧠 **深度学习修复**：使用 ProPainter、E2FGVI、STTN 等先进模型
- 📹 **长视频支持**：分块处理，支持任意长度视频
- ⚡ **GPU 加速**：支持 CUDA 和半精度推理，大幅提升处理速度
- 🎨 **时序一致性**：使用双向传播保持视频时序连贯性

### v3.0 新增功能
- 🌐 **Web 界面**：基于 Gradio 的直观 Web 界面，无需命令行
- 👤 **人脸修复**：集成 GFPGAN/CodeFormer，提升人脸修复效果
- 📁 **多格式输出**：支持 MP4/AVI/MKV/WebM/GIF/图片序列
- 📋 **处理历史**：记录每次处理，方便回溯和重新处理

### v2.0 功能
- 👁️ **实时预览**：处理过程中实时查看修复效果对比
- 🔄 **断点续传**：处理中断后可从上次位置继续，无需重新开始
- 🎨 **遮罩编辑增强**：支持画笔、橡皮擦工具，真正的撤销/重做功能
- 📊 **详细进度**：显示处理速度、ETA、显存使用等统计信息

## 📋 系统要求

### 硬件要求
- **CPU**：Intel i5 或同等性能以上
- **内存**：8GB 以上（推荐 16GB）
- **显卡**：NVIDIA GPU（推荐 RTX 2060 以上，显存 6GB+）
  - 无 GPU 也可使用 CPU 处理（速度较慢）

### 软件要求
- Python 3.8+
- CUDA 11.0+（如使用 GPU）
- FFmpeg（可选，用于高级视频处理）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 下载 ProPainter 模型（推荐）
python download_models.py --model propainter

# 或下载所有模型
python download_models.py --model all
```

### 3. 处理视频

```bash
# 基本使用（自动检测马赛克）
python main.py --input input.mp4 --output output.mp4

# 启用实时预览
python main.py --input input.mp4 --output output.mp4 --preview

# 使用 CPU 处理
python main.py --input input.mp4 --output output.mp4 --device cpu

# 指定马赛克区域（JSON 格式）
python main.py --input input.mp4 --output output.mp4 --mask mask.json

# 使用配置文件
python main.py --config configs/config.yaml
```

### 4. 启动 Web 界面

```bash
# 启动 Web 界面
python web_app.py

# 访问 http://localhost:7860
```

## 📖 详细使用说明

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入视频路径 | 必需 |
| `--output, -o` | 输出视频路径 | 自动生成 |
| `--model` | 选择模型 | propainter |
| `--device` | 运行设备 | cuda (自动检测) |
| `--fp16` | 启用半精度推理 | false |
| `--auto-detect` | 自动检测马赛克 | false |
| `--mask` | 指定遮罩文件 | 无 |
| `--chunk-size` | 处理块大小 | 10 |
| `--neighbor-frames` | 邻近帧数 | 10 |
| `--bidirectional` | 双向传播 | false |
| `--quality` | 输出视频质量 | 95 |
| `--preview` | 启用实时预览 | false |
| `--preview-scale` | 预览窗口缩放比例 | 0.5 |
| `--no-resume` | 禁用断点续传 | false |

### 实时预览

启用实时预览可以在处理过程中查看修复效果：

```bash
# 启用预览
python main.py --input input.mp4 --output output.mp4 --preview

# 调整预览窗口大小
python main.py --input input.mp4 --output output.mp4 --preview --preview-scale 0.8
```

预览窗口显示：
- **左侧**：原始帧
- **中间**：遮罩 + 修复效果
- **右侧**：修复后的帧

按 `Q` 键可以退出预览。

### 断点续传

断点续传功能默认启用，处理中断后会自动保存进度：

```bash
# 处理视频（自动启用断点续传）
python main.py --input input.mp4 --output output.mp4

# 按 Ctrl+C 中断后，重新运行相同命令即可继续
python main.py --input input.mp4 --output output.mp4

# 禁用断点续传（从头开始）
python main.py --input input.mp4 --output output.mp4 --no-resume
```

### 遮罩编辑工具

v2.0 增强了遮罩创建工具，支持更多编辑功能：

```bash
# 启动遮罩创建工具
python mask_creator.py --input input.mp4
```

**快捷键说明**：

| 快捷键 | 功能 |
|--------|------|
| `1` | 切换到矩形工具 |
| `2` | 切换到画笔工具 |
| `3` | 切换到橡皮擦工具 |
| `+` / `-` | 调整画笔大小 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Y` | 重做 |
| `D` | 删除最后一个遮罩 |
| `R` | 重置当前帧遮罩 |
| `N` | 下一帧 |
| `P` | 上一帧 |
| `G` | 切换到全局遮罩模式 |
| `F` | 切换到帧遮罩模式 |
| `S` | 保存遮罩 |
| `H` | 显示/隐藏帮助 |
| `V` | 切换遮罩叠加显示 |
| `.` / `,` | 调整遮罩透明度 |
| `Q` | 退出 |

### 模型选择

| 模型 | 特点 | 推荐场景 |
|------|------|----------|
| **ProPainter** | 最先进，效果最好 | 通用场景，推荐首选 |
| **E2FGVI** | 平衡效果和速度 | 中等长度视频 |
| **STTN** | 速度快 | 长视频，实时处理 |

### 遮罩文件格式

如果需要手动指定马赛克区域，可以使用 JSON 格式的遮罩文件：

```json
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
```

- `frames`：指定特定帧的遮罩区域
- `global_masks`：指定跨多帧的全局遮罩区域

## ⚙️ 配置文件

详细配置请参考 `configs/config.yaml`：

```yaml
# 基础配置
base:
  input_video: ""
  output_video: ""
  mask_file: ""

# 模型配置
model:
  name: "propainter"  # propainter, e2fgvi, sttn
  checkpoint_dir: "./checkpoints"
  device: "cuda"
  use_fp16: true

# 处理配置
process:
  chunk_size: 10
  neighbor_frames: 10
  bidirectional: true
  dilate_kernel: 5

# 马赛克检测配置
detection:
  auto_detect: true
  sensitivity: 0.7
  min_area: 100

# 输出配置
output:
  codec: "mp4v"
  quality: 95
  save_frames: false
  show_progress: true
  preview: false
  preview_scale: 0.5

# 断点续传配置
resume:
  enabled: true
```

## 🔧 性能优化

### 显存不足时

1. **启用半精度推理**：
   ```bash
   python main.py --input input.mp4 --output output.mp4 --fp16
   ```

2. **减小处理块大小**：
   ```bash
   python main.py --input input.mp4 --output output.mp4 --chunk-size 5
   ```

3. **使用 CPU 处理**：
   ```bash
   python main.py --input input.mp4 --output output.mp4 --device cpu
   ```

### 加速处理

1. **使用 GPU**（推荐）：
   ```bash
   python main.py --input input.mp4 --output output.mp4 --device cuda
   ```

2. **减小邻近帧数**：
   ```bash
   python main.py --input input.mp4 --output output.mp4 --neighbor-frames 5
   ```

## 📁 项目结构

```
video_demosiac/
├── main.py                 # 主程序入口
├── web_app.py              # Web 界面（v3.0 新增）
├── mask_creator.py         # 遮罩创建工具
├── batch_process.py        # 批量处理脚本
├── download_models.py      # 模型下载脚本
├── requirements.txt        # 依赖文件
├── README.md               # 使用说明
├── configs/
│   └── config.yaml         # 配置文件
├── models/                 # 模型定义
│   └── simple_inpainter.py # 简单修复模型（后备）
├── checkpoints/            # 预训练模型存放目录
├── history/                # 处理历史记录
└── utils/
    ├── __init__.py
    ├── video_processor.py  # 视频处理器
    ├── mask_generator.py   # 遮罩生成器
    ├── model_runner.py     # 模型运行器
    ├── face_restorer.py    # 人脸修复模块（v3.0 新增）
    ├── output_writer.py    # 多格式输出写入器（v3.0 新增）
    ├── history.py          # 处理历史管理（v3.0 新增）
    └── logger.py           # 日志工具
```

## 🐛 常见问题

### 1. CUDA out of memory
- 减小 `--chunk-size` 和 `--neighbor-frames`
- 启用 `--fp16` 半精度推理
- 使用 CPU 处理

### 2. 处理速度太慢
- 使用 GPU（`--device cuda`）
- 选择更快的模型（如 STTN）
- 减小邻近帧数

### 3. 修复效果不理想
- 尝试不同的模型
- 调整检测灵敏度（`--sensitivity`）
- 手动指定遮罩区域

### 4. 无法打开视频
- 检查视频文件路径是否正确
- 确保安装了 opencv-python
- 尝试使用 FFmpeg 转换视频格式

### 5. 断点续传不工作
- 确保没有禁用断点续传（不使用 `--no-resume`）
- 检查输出目录是否可写
- 确保输入视频没有变化

### 6. 预览窗口不显示
- 确保在有 GUI 的环境中运行
- 检查 OpenCV 是否正确安装
- 尝试使用 `--preview-scale` 调整窗口大小

## 📚 参考资料

- [ProPainter](https://github.com/sczhou/ProPainter)
- [E2FGVI](https://github.com/ruoshui6/E2FGVI)
- [STTN](https://github.com/rese1f/STTN)
- [Video Inpainting Survey](https://arxiv.org/abs/2012.13147)

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系我们。
