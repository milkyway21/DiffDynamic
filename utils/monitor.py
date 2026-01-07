"""
GPU监控工具模块，用于在采样和训练过程中监控GPU显存使用情况。

提供简单的显存监控功能，不依赖额外的库。
"""

import torch
from typing import Dict, Any, Optional
from contextlib import contextmanager
import time


class GPUMonitor:
    """GPU显存和性能监控器。"""
    
    def __init__(self, device: str = 'cuda:0', enable_flops: bool = False):
        """
        初始化GPU监控器。
        
        Args:
            device: GPU设备字符串，如 'cuda:0'。
            enable_flops: 是否启用FLOPs计算（需要thop或fvcore）。
        """
        self.device = device
        self.enable_flops = enable_flops
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0
        self._last_info = {}
        
    def reset_peak_stats(self):
        """重置峰值统计。"""
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_allocated = 0.0
            self._peak_reserved = 0.0
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        获取当前显存使用信息。
        
        Returns:
            包含显存信息的字典：
            - allocated: 当前分配的显存（MB）
            - max_allocated: 峰值分配的显存（MB）
            - reserved: 当前保留的显存（MB）
            - max_reserved: 峰值保留的显存（MB）
        """
        if not torch.cuda.is_available() or 'cuda' not in self.device:
            return {
                'allocated': 0.0,
                'max_allocated': 0.0,
                'reserved': 0.0,
                'max_reserved': 0.0,
            }
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
        
        try:
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        except RuntimeError:
            max_allocated = allocated
        
        try:
            max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2  # MB
        except RuntimeError:
            max_reserved = reserved
        
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)
        
        return {
            'allocated': allocated,
            'max_allocated': max(max_allocated, self._peak_allocated),
            'reserved': reserved,
            'max_reserved': max(max_reserved, self._peak_reserved),
        }
    
    @contextmanager
    def monitor_forward(self, model, inputs, input_kwargs=None, log_fn=None):
        """
        监控模型前向传播的上下文管理器。
        
        Args:
            model: 要监控的模型。
            inputs: 模型输入（元组或单个张量）。
            input_kwargs: 模型输入的额外关键字参数。
            log_fn: 可选的日志函数，用于输出监控信息。
        
        Yields:
            包含监控结果的字典。
        """
        mem_before = self.get_memory_info()
        t_start = time.time()
        
        try:
            yield None
        finally:
            t_end = time.time()
            mem_after = self.get_memory_info()
            
            forward_time = t_end - t_start
            mem_delta = mem_after['allocated'] - mem_before['allocated']
            
            result = {
                'time': forward_time,
                'memory': {
                    'allocated': mem_after['allocated'],
                    'max_allocated': mem_after['max_allocated'],
                    'reserved': mem_after['reserved'],
                    'max_reserved': mem_after['max_reserved'],
                    'allocated_delta': mem_delta,
                },
                'flops': None,  # FLOPs计算需要额外库支持
            }
            
            self._last_info = result
            
            if log_fn:
                log_fn(
                    f'Forward pass: time={forward_time:.4f}s, '
                    f'mem_delta={mem_delta:.2f}MB, '
                    f'max_mem={mem_after["max_allocated"]:.2f}MB'
                )
    
    def get_last_info(self) -> Dict[str, Any]:
        """获取最后一次监控的信息。"""
        return self._last_info.copy()


class MemoryProfiler:
    """显存分析器，用于追踪显存使用峰值和检查点。"""
    
    def __init__(self, device: str = 'cuda:0'):
        """
        初始化显存分析器。
        
        Args:
            device: GPU设备字符串。
        """
        self.device = device
        self.checkpoints = []
        self.peak_memory = 0.0
    
    def checkpoint(self, name: str):
        """
        记录一个显存检查点。
        
        Args:
            name: 检查点名称。
        """
        if torch.cuda.is_available() and 'cuda' in self.device:
            current = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            try:
                peak = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            except RuntimeError:
                peak = current
        else:
            current = 0.0
            peak = 0.0
        
        self.peak_memory = max(self.peak_memory, peak)
        
        self.checkpoints.append({
            'name': name,
            'current_mb': current,
            'peak_mb': peak,
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取显存使用摘要。
        
        Returns:
            包含峰值显存和检查点列表的字典。
        """
        return {
            'peak_memory_mb': self.peak_memory,
            'checkpoints': self.checkpoints.copy(),
        }


__all__ = ['GPUMonitor', 'MemoryProfiler']

