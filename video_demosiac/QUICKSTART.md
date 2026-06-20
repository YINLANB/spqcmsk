# 快速开始指南

## 1. 环境准备

### 安装 Python
确保已安装 Python 3.8 或更高版本。

### 安装依赖
```bash
# 进入项目目录
cd video_demosiac

# 安装依赖
pip install -r requirements.txt
```

### 安装 CUDA（可选但推荐）
如果有 NVIDIA GPU，建议安装 CUDA 以加速处理：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. 下载模型（可选）

```bash
# 下载推荐的 ProPainter 模型
python download_models.py --model propainter

# 或下载所有模型
python download_models.py --model all
```

**注意**：如果不下载预训练模型，系统会自动使用简单修复模型作为后备。

## 3. 处理视频

### 基本使用
```bash
# 处理单个视频
python main.py --input input.mp4 --output output.mp4

# 自动检测马赛克区域
python main.py --input input.mp4 --output output.mp4 --auto-detect
```

### 使用 Windows 批处理脚本
```bash
# 双击 run.bat 或在命令行运行
run.bat --input input.mp4 --output output.mp4
```

### 使用 Shell 脚本（Linux/Mac）
```bash
chmod +x run.sh
./run.sh --input input.mp4 --output output.mp4
```

## 4. 高级用法

### 指定马赛克区域
如果自动检测效果不理想，可以手动指定马赛克区域：

```bash
# 使用遮罩创建工具
python mask_creator.py --input input.mp4

# 使用已有的遮罩文件
python main.py --input input.mp4 --output output.mp4 --mask mask.json
```

### 批量处理
```bash
# 批量处理目录中的所有视频
python batch_process.py --input-dir ./videos --output-dir ./output
```

### 使用不同模型
```bash
# 使用 ProPainter（推荐）
python main.py --input input.mp4 --output output.mp4 --model propainter

# 使用 E2FGVI
python main.py --input input.mp4 --output output.mp4 --model e2fgvi

# 使用 STTN（更快）
python main.py --input input.mp4 --output output.mp4 --model sttn
```

## 5. 性能优化

### 显存不足时
```bash
# 启用半精度推理
python main.py --input input.mp4 --output output.mp4 --fp16

# 减小处理块大小
python main.py --input input.mp4 --output output.mp4 --chunk-size 5

# 使用 CPU 处理
python main.py --input input.mp4 --output output.mp4 --device cpu
```

### 加速处理
```bash
# 使用 GPU（推荐）
python main.py --input input.mp4 --output output.mp4 --device cuda

# 减小邻近帧数
python main.py --input input.mp4 --output output.mp4 --neighbor-frames 5
```

## 6. 配置文件

详细配置请参考 `configs/config.yaml`。

## 7. 常见问题

### Q: 如何处理长视频？
A: 系统会自动将长视频分块处理，无需额外配置。可以通过 `--chunk-size` 调整块大小。

### Q: 处理速度太慢怎么办？
A: 1. 使用 GPU（`--device cuda`）
   2. 选择更快的模型（如 STTN）
   3. 减小邻近帧数

### Q: 修复效果不理想？
A: 1. 尝试不同的模型
   2. 调整检测灵敏度
   3. 手动指定遮罩区域

### Q: 无法打开视频？
A: 1. 检查视频文件路径
   2. 确保安装了 opencv-python
   3. 尝试使用 FFmpeg 转换视频格式

## 8. 更多信息

详见 [README.md](README.md)。
