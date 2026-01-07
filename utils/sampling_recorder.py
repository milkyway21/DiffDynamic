"""
Utility helpers for persisting per-run sampling metadata to an Excel log.

Each record captures:
    - timestamp with millisecond precision
    - flattened sampling parameters (model + sample sections)
    - output directory and optional result file path
    - an arbitrary mode/extra info map for future filtering
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _PANDAS_MISSING_MSG = (
        'pandas is required to 记录采样信息，运行 `pip install pandas` 以启用该功能。'
    )
else:
    _PANDAS_MISSING_MSG = ''

try:
    import openpyxl  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _OPENPYXL_MISSING_MSG = (
        'openpyxl 缺失，无法写入 Excel。运行 `pip install openpyxl` 以启用。'
    )
else:
    _OPENPYXL_MISSING_MSG = ''

DEFAULT_RECORD_PATH = Path('outputs') / 'sampling_history.xlsx'


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


def _json_fallback(obj: Any) -> Any:
    """Ensure json.dumps always succeeds."""
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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


def extract_sampling_params(config: Any) -> Dict[str, Any]:
    """Pick the model/sample sub-configs and normalize them for serialization."""
    def _get_section(name: str) -> Any:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        return getattr(config, name, {})

    return {
        'model': _coerce_to_builtin(_get_section('model')),
        'sample': _coerce_to_builtin(_get_section('sample')),
    }


def log_sampling_record(
    params: Dict[str, Any],
    result_dir: str,
    sampling_mode: str,
    result_file: Optional[str] = None,
    record_path: Optional[str] = None,
    logger: Any = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Append a sampling record to the Excel log.

    Args:
        params: Dict containing serialized config sections.
        result_dir: Directory holding generated artifacts.
        sampling_mode: 'baseline' / 'dynamic' or other mode label.
        result_file: Optional file path for the serialized outputs.
        record_path: Custom Excel path; defaults to outputs/sampling_history.xlsx.
        logger: Optional logger for success/failure messages.
        extra_info: Arbitrary key-value metadata (e.g., data_id).

    Returns:
        bool: True if the row was written; False if the operation failed.
    """
    if pd is None:
        message = _PANDAS_MISSING_MSG or 'pandas is required but not installed.'
        if logger:
            logger.warning(message)
        else:
            print(f'[sampling_recorder] {message}')
        return False

    log_path = Path(record_path) if record_path else DEFAULT_RECORD_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    normalized_params = _coerce_to_builtin(params or {})

    # Start with basic fields
    row: Dict[str, Any] = {
        'timestamp': timestamp,
        'result_dir': str(Path(result_dir).resolve()),
        'result_file': str(Path(result_file).resolve()) if result_file else '',
        'mode': sampling_mode,
    }

    # Flatten and add all parameters as separate columns
    flattened_params = _flatten_dict(normalized_params)
    for key, value in flattened_params.items():
        row[key] = value

    # Add extra info fields
    if extra_info:
        for key, value in extra_info.items():
            row[str(key)] = _coerce_to_builtin(value)

    try:
        new_row_df = pd.DataFrame([row])
        if log_path.exists():
            existing = pd.read_excel(log_path)
            # Ensure all columns exist in both DataFrames (fill missing with empty string)
            all_columns = set(existing.columns) | set(new_row_df.columns)
            for col in all_columns:
                if col not in existing.columns:
                    existing[col] = ''
                if col not in new_row_df.columns:
                    new_row_df[col] = ''
            # Reorder columns consistently (timestamp first, then alphabetical)
            priority_cols = ['timestamp', 'result_dir', 'result_file', 'mode']
            other_cols = sorted([c for c in all_columns if c not in priority_cols])
            column_order = [c for c in priority_cols if c in all_columns] + other_cols
            existing = existing[column_order]
            new_row_df = new_row_df[column_order]
            combined = pd.concat([existing, new_row_df], ignore_index=True)
        else:
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
                logger.warning(f'Failed to write sampling record to {log_path}: {error_msg}')
            else:
                print(f'[sampling_recorder] {error_msg}')
            return False
    except Exception as exc:  # pragma: no cover - defensive logging only
        if logger:
            logger.warning(f'Failed to write sampling record to {log_path}: {exc}')
        else:
            print(f'[sampling_recorder] Failed to write sampling record: {exc}')
        return False

    if logger:
        logger.info(f'Sampling record appended to {log_path}')
    return True


__all__ = [
    'extract_sampling_params',
    'log_sampling_record',
    'DEFAULT_RECORD_PATH',
]

