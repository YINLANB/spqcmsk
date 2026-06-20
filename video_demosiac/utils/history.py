#!/usr/bin/env python3
"""
处理历史管理模块
================
记录和管理视频处理历史。
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .logger import setup_logger

logger = setup_logger(__name__)


class ProcessingHistory:
    """
    处理历史管理器
    记录每次视频处理的元数据
    """

    def __init__(self, history_dir: str = "./history"):
        """
        初始化处理历史管理器

        Args:
            history_dir: 历史记录目录
        """
        self.history_dir = Path(history_dir)
        self.history_file = self.history_dir / "history.json"
        self.max_records = 1000

        # 创建目录
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # 加载历史记录
        self.records = self._load_records()

    def _load_records(self) -> List[Dict]:
        """加载历史记录"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            return records if isinstance(records, list) else []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"加载历史记录失败: {e}")
            return []

    def _save_records(self):
        """保存历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")

    def add_record(
        self,
        input_video: str,
        output_video: str,
        model: str = "",
        device: str = "",
        duration: float = 0,
        frames_processed: int = 0,
        config: Optional[Dict] = None,
        status: str = "completed",
        error: Optional[str] = None
    ) -> str:
        """
        添加处理记录

        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            model: 使用的模型
            device: 运行设备
            duration: 处理时长（秒）
            frames_processed: 处理的帧数
            config: 处理配置
            status: 处理状态
            error: 错误信息

        Returns:
            str: 记录 ID
        """
        record_id = str(uuid.uuid4())

        record = {
            "id": record_id,
            "input_video": input_video,
            "output_video": output_video,
            "model": model,
            "device": device,
            "duration": duration,
            "frames_processed": frames_processed,
            "config": config or {},
            "status": status,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "input_filename": os.path.basename(input_video),
            "output_filename": os.path.basename(output_video)
        }

        # 添加到记录列表
        self.records.append(record)

        # 限制记录数量
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

        # 保存
        self._save_records()

        logger.info(f"已添加处理记录: {record_id}")
        return record_id

    def update_record(
        self,
        record_id: str,
        **kwargs
    ) -> bool:
        """
        更新处理记录

        Args:
            record_id: 记录 ID
            **kwargs: 要更新的字段

        Returns:
            bool: 是否成功
        """
        for record in self.records:
            if record["id"] == record_id:
                record.update(kwargs)
                record["updated_at"] = datetime.now().isoformat()
                self._save_records()
                return True

        return False

    def get_record(self, record_id: str) -> Optional[Dict]:
        """
        获取单条记录

        Args:
            record_id: 记录 ID

        Returns:
            dict: 记录信息，如果不存在则返回 None
        """
        for record in self.records:
            if record["id"] == record_id:
                return record

        return None

    def list_records(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        列出处理记录

        Args:
            limit: 返回数量限制
            offset: 偏移量
            status: 过滤状态

        Returns:
            list: 记录列表
        """
        records = self.records

        # 按状态过滤
        if status:
            records = [r for r in records if r.get("status") == status]

        # 按时间倒序排列
        records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)

        # 分页
        return records[offset:offset + limit]

    def delete_record(self, record_id: str, delete_files: bool = False) -> bool:
        """
        删除处理记录

        Args:
            record_id: 记录 ID
            delete_files: 是否删除输出文件

        Returns:
            bool: 是否成功
        """
        for i, record in enumerate(self.records):
            if record["id"] == record_id:
                # 删除输出文件（如果需要）
                if delete_files:
                    output_file = record.get("output_video")
                    if output_file and os.path.exists(output_file):
                        try:
                            os.remove(output_file)
                            logger.info(f"已删除输出文件: {output_file}")
                        except Exception as e:
                            logger.warning(f"删除输出文件失败: {e}")

                # 从列表中移除
                self.records.pop(i)
                self._save_records()
                return True

        return False

    def clear_history(self, delete_files: bool = False) -> int:
        """
        清空历史记录

        Args:
            delete_files: 是否删除所有输出文件

        Returns:
            int: 删除的记录数量
        """
        count = len(self.records)

        if delete_files:
            for record in self.records:
                output_file = record.get("output_video")
                if output_file and os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception:
                        pass

        self.records.clear()
        self._save_records()

        return count

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            dict: 统计信息
        """
        total = len(self.records)
        completed = len([r for r in self.records if r.get("status") == "completed"])
        failed = len([r for r in self.records if r.get("status") == "failed"])

        total_duration = sum(r.get("duration", 0) for r in self.records)
        total_frames = sum(r.get("frames_processed", 0) for r in self.records)

        # 模型使用统计
        model_counts = {}
        for record in self.records:
            model = record.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "total_records": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "total_duration": total_duration,
            "total_frames": total_frames,
            "average_duration": (total_duration / total) if total > 0 else 0,
            "model_usage": model_counts
        }

    def export_history(self, export_path: str) -> bool:
        """
        导出历史记录

        Args:
            export_path: 导出路径

        Returns:
            bool: 是否成功
        """
        try:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_records": len(self.records),
                "statistics": self.get_statistics(),
                "records": self.records
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"导出历史记录失败: {e}")
            return False

    def import_history(self, import_path: str) -> int:
        """
        导入历史记录

        Args:
            import_path: 导入路径

        Returns:
            int: 导入的记录数量
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            records = import_data.get("records", [])
            if not isinstance(records, list):
                return 0

            # 添加记录
            count = 0
            for record in records:
                if "id" in record and "timestamp" in record:
                    # 检查是否已存在
                    existing = self.get_record(record["id"])
                    if not existing:
                        self.records.append(record)
                        count += 1

            # 保存
            self._save_records()

            return count

        except Exception as e:
            logger.error(f"导入历史记录失败: {e}")
            return 0


# 全局历史管理器实例
_global_history: Optional[ProcessingHistory] = None


def get_history_manager(history_dir: str = "./history") -> ProcessingHistory:
    """
    获取全局历史管理器实例

    Args:
        history_dir: 历史记录目录

    Returns:
        ProcessingHistory: 历史管理器实例
    """
    global _global_history

    if _global_history is None:
        _global_history = ProcessingHistory(history_dir)

    return _global_history


def add_processing_record(
    input_video: str,
    output_video: str,
    model: str = "",
    device: str = "",
    duration: float = 0,
    frames_processed: int = 0,
    config: Optional[Dict] = None,
    status: str = "completed",
    error: Optional[str] = None
) -> str:
    """
    添加处理记录的便捷函数

    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        model: 使用的模型
        device: 运行设备
        duration: 处理时长（秒）
        frames_processed: 处理的帧数
        config: 处理配置
        status: 处理状态
        error: 错误信息

    Returns:
        str: 记录 ID
    """
    history = get_history_manager()
    return history.add_record(
        input_video=input_video,
        output_video=output_video,
        model=model,
        device=device,
        duration=duration,
        frames_processed=frames_processed,
        config=config,
        status=status,
        error=error
    )
