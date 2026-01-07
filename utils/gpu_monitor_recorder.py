"""
GPU监控数据记录模块，用于在采样过程中记录GPU显存、时间、FLOPs等信息并写入Excel。

每个记录包含：
    - 时间戳（精确到毫秒）
    - 采样相关信息（data_id, mode等）
    - GPU显存使用情况（allocated, max_allocated, peak）
    - 前向传播时间
    - FLOPs信息（如果启用）
    - 显存检查点摘要
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_MISSING_MSG = (
        'pandas is required to record GPU监控信息，运行 `pip install pandas` 以启用该功能。'
    )
else:
    _PANDAS_MISSING_MSG = ''

try:
    import openpyxl  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _OPENPYXL_MISSING_MSG = (
        'openpyxl 缺失，Excel 写入将被禁用。运行 `pip install openpyxl` 以启用。'
    )
else:
    _OPENPYXL_MISSING_MSG = ''

DEFAULT_RECORD_PATH = Path('outputs') / 'gpu_monitor_history.xlsx'


def _coerce_to_builtin(value: Any) -> Any:
    """Recursively convert EasyDict/sequence/numpy scalars to JSON-friendly types."""
    if isinstance(value, Mapping):
        return {str(k): _coerce_to_builtin(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_to_builtin(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):  # numpy / torch scalars
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Recursively flatten a nested dictionary into a single level with joined keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            # Convert value to a serializable type
            if v is None:
                items.append((new_key, ''))
            elif isinstance(v, (bool, int, float, str)):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
    return dict(items)


def log_gpu_monitor_record(
    memory_info: Optional[Dict[str, Any]] = None,
    forward_time: Optional[float] = None,
    flops_info: Optional[Dict[str, Any]] = None,
    memory_summary: Optional[Dict[str, Any]] = None,
    sampling_info: Optional[Dict[str, Any]] = None,
    record_path: Optional[str] = None,
    logger: Any = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    将GPU监控记录追加到Excel日志中。

    Args:
        memory_info: 显存信息字典，包含 'allocated', 'max_allocated' 等字段。
        forward_time: 前向传播时间（秒）。
        flops_info: FLOPs信息字典，包含 'flops_g', 'params' 等字段。
        memory_summary: 显存摘要字典，包含 'peak_memory_mb', 'checkpoints' 等字段。
        sampling_info: 采样相关信息，如 'data_id', 'mode', 'sample_idx' 等。
        record_path: 自定义Excel路径；默认为 outputs/gpu_monitor_history.xlsx。
        logger: 可选的日志记录器，用于输出成功/失败消息。
        extra_info: 任意键值元数据。

    Returns:
        bool: 如果行写入成功返回True；否则返回False。
    """
    if pd is None:
        message = _PANDAS_MISSING_MSG or 'pandas is required but not installed.'
        if logger:
            logger.warning(message)
        else:
            print(f'[gpu_monitor_recorder] {message}')
        return False

    log_path = Path(record_path) if record_path else DEFAULT_RECORD_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

    # 构建基础行数据
    row: Dict[str, Any] = {
        'timestamp': timestamp,
    }

    # 添加采样信息
    if sampling_info:
        for key, value in sampling_info.items():
            row[f'sampling_{key}'] = _coerce_to_builtin(value)

    # 添加显存信息
    if memory_info:
        row['memory_allocated_mb'] = memory_info.get('allocated', '')
        row['memory_max_allocated_mb'] = memory_info.get('max_allocated', '')
        row['memory_reserved_mb'] = memory_info.get('reserved', '')
        row['memory_max_reserved_mb'] = memory_info.get('max_reserved', '')
        if 'allocated_delta' in memory_info:
            row['memory_allocated_delta_mb'] = memory_info.get('allocated_delta', '')

    # 添加前向传播时间
    if forward_time is not None:
        row['forward_time_sec'] = forward_time

    # 添加FLOPs信息
    if flops_info:
        row['flops_g'] = flops_info.get('flops_g', '')
        row['flops_params'] = flops_info.get('params', '')
        if 'flops' in flops_info:
            row['flops_total'] = flops_info.get('flops', '')

    # 添加显存摘要
    if memory_summary:
        row['memory_peak_mb'] = memory_summary.get('peak_memory_mb', '')
        # 展平检查点信息
        checkpoints = memory_summary.get('checkpoints', [])
        if checkpoints:
            # 将检查点列表转换为字符串（JSON格式）或展开为多列
            checkpoint_names = [cp.get('name', '') for cp in checkpoints]
            checkpoint_memories = [cp.get('current_mb', '') for cp in checkpoints]
            checkpoint_peaks = [cp.get('peak_mb', '') for cp in checkpoints]
            row['checkpoint_names'] = ';'.join(str(n) for n in checkpoint_names)
            row['checkpoint_memories_mb'] = ';'.join(str(m) for m in checkpoint_memories)
            row['checkpoint_peaks_mb'] = ';'.join(str(p) for p in checkpoint_peaks)
            # 也可以记录检查点数量
            row['checkpoint_count'] = len(checkpoints)

    # 添加额外信息
    if extra_info:
        for key, value in extra_info.items():
            row[str(key)] = _coerce_to_builtin(value)

    try:
        new_row_df = pd.DataFrame([row])
        existing = None
        file_corrupted = False
        
        if log_path.exists():
            try:
                existing = pd.read_excel(log_path, engine='openpyxl')
            except Exception as read_exc:
                # Excel文件可能损坏，尝试备份并创建新文件
                error_str = str(read_exc).lower()
                if 'bad magic number' in error_str or 'corrupt' in error_str or 'invalid' in error_str:
                    file_corrupted = True
                    # 备份损坏的文件
                    backup_path = log_path.with_suffix('.xlsx.corrupted_backup')
                    try:
                        import shutil
                        shutil.move(str(log_path), str(backup_path))
                        if logger:
                            logger.warning(
                                f'Excel文件 {log_path} 已损坏，已备份到 {backup_path}，将创建新文件。'
                            )
                        else:
                            print(f'[gpu_monitor_recorder] Excel文件 {log_path} 已损坏，已备份到 {backup_path}，将创建新文件。')
                    except Exception as backup_exc:
                        if logger:
                            logger.warning(f'无法备份损坏的文件 {log_path}: {backup_exc}')
                        else:
                            print(f'[gpu_monitor_recorder] 无法备份损坏的文件 {log_path}: {backup_exc}')
                else:
                    # 其他读取错误，直接抛出
                    raise
        
        if existing is not None and not file_corrupted:
            # 确保所有列在两个DataFrame中都存在（缺失的用空字符串填充）
            all_columns = set(existing.columns) | set(new_row_df.columns)
            for col in all_columns:
                if col not in existing.columns:
                    existing[col] = ''
                if col not in new_row_df.columns:
                    new_row_df[col] = ''
            # 重新排序列（timestamp优先，然后按字母顺序）
            priority_cols = ['timestamp']
            other_cols = sorted([c for c in all_columns if c not in priority_cols])
            column_order = [c for c in priority_cols if c in all_columns] + other_cols
            existing = existing[column_order]
            new_row_df = new_row_df[column_order]
            combined = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            # 文件不存在或已损坏，创建新文件
            combined = new_row_df

        try:
            with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
                combined.to_excel(writer, index=False)
        except ImportError as e:
            error_msg = _OPENPYXL_MISSING_MSG or (
                'Missing optional dependency "openpyxl". '
                'Install it with: pip install openpyxl or conda install openpyxl'
            )
            if logger:
                logger.warning(f'Failed to write GPU monitor record to {log_path}: {error_msg}')
            else:
                print(f'[gpu_monitor_recorder] {error_msg}')
            return False
    except Exception as exc:  # pragma: no cover - defensive logging only
        error_msg = str(exc)
        if 'bad magic number' in error_msg.lower():
            error_msg = f'Excel文件损坏 (Bad magic number): {error_msg}'
        if logger:
            logger.warning(f'Failed to write GPU monitor record to {log_path}: {error_msg}')
        else:
            print(f'[gpu_monitor_recorder] Failed to write GPU monitor record: {error_msg}')
        return False

    if logger:
        logger.info(f'GPU monitor record appended to {log_path}')
    return True


def log_monitor_from_context(
    monitor_result: Dict[str, Any],
    sampling_info: Optional[Dict[str, Any]] = None,
    record_path: Optional[str] = None,
    logger: Any = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    从监控上下文结果中提取信息并记录。

    这是一个便捷函数，用于处理从 GPUMonitor.monitor_forward() 返回的结果。

    Args:
        monitor_result: 监控结果字典，通常包含 'time', 'memory', 'flops' 等字段。
        sampling_info: 采样相关信息。
        record_path: 自定义Excel路径。
        logger: 可选的日志记录器。
        extra_info: 任意键值元数据。

    Returns:
        bool: 如果记录成功返回True；否则返回False。
    """
    memory_info = monitor_result.get('memory', {})
    forward_time = monitor_result.get('time')
    flops_info = monitor_result.get('flops', {})

    return log_gpu_monitor_record(
        memory_info=memory_info,
        forward_time=forward_time,
        flops_info=flops_info if flops_info else None,
        sampling_info=sampling_info,
        record_path=record_path,
        logger=logger,
        extra_info=extra_info,
    )


__all__ = [
    'log_gpu_monitor_record',
    'log_monitor_from_context',
    'DEFAULT_RECORD_PATH',
]

