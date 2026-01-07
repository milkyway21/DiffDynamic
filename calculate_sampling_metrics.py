#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算采样指标：总采样步数和实际长度

使用方法：
    # 使用默认配置文件
    python3 calculate_sampling_metrics.py
    
    # 指定配置文件
    python3 calculate_sampling_metrics.py --config configs/sampling.yml
    
    # 直接通过参数计算（不读取配置文件）
    python3 calculate_sampling_metrics.py --num_steps 1000 --time_boundary 750 \
        --ls_schedule lambda --ls_coeff_a 60.0 --ls_coeff_b 20.0 --ls_step_size 0.4 \
        --rf_schedule lambda --rf_coeff_a 40.0 --rf_coeff_b 5.0 --rf_step_size 0.4
"""

import sys
import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    print("错误：需要安装 yaml 模块")
    print("请运行: pip install pyyaml")
    sys.exit(1)

import numpy as np


def build_lambda_schedule(start_t, end_t, coeff_a, coeff_b, num_timesteps):
    """构建 lambda 调度序列"""
    start_t = int(max(0, min(start_t, num_timesteps - 1)))
    end_t = int(max(0, min(end_t, num_timesteps - 1)))
    if start_t == end_t:
        return [start_t], []
    
    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    t = start_t
    indices = []
    step_sizes = []
    
    while True:
        indices.append(t)
        if t == end_t:
            break
        lambda_t = float(t) / float(num_timesteps)
        lambda_t = max(min(lambda_t, 1.0), 0.0)
        n = coeff_a * lambda_t + coeff_b
        step = max(1, int(round(n)))
        step_sizes.append(step)
        t_next = t + step_sign * step
        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t
        if t_next == t:
            t_next = t + step_sign
        t = int(max(0, min(t_next, num_timesteps - 1)))
    
    return indices, step_sizes


def build_linear_schedule(start_t, end_t, step_upper, step_lower, num_timesteps):
    """构建线性调度序列"""
    if start_t == end_t:
        return [start_t], []
    
    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    t = start_t
    indices = []
    step_sizes = []
    
    initial_range = abs(start_t - end_t)
    
    while True:
        indices.append(t)
        if t == end_t:
            break
        
        remaining_range = abs(t - end_t)
        if initial_range > 0:
            progress = float(remaining_range) / float(initial_range)
        else:
            progress = 0.0
        progress = max(0.0, min(1.0, progress))
        
        coeff_a = step_upper - step_lower
        coeff_b = step_lower
        n = coeff_a * progress + coeff_b
        step = max(1, int(round(n)))
        step_sizes.append(step)
        
        t_next = t + step_sign * step
        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t
        if t_next == t:
            t_next = t + step_sign
        
        t = int(max(0, min(t_next, num_timesteps - 1)))
    
    return indices, step_sizes


def build_fixed_schedule(start_t, end_t, stride, num_timesteps):
    """构建固定步长调度序列"""
    start_t = int(max(0, min(start_t, num_timesteps - 1)))
    end_t = int(max(0, min(end_t, num_timesteps - 1)))
    if start_t == end_t:
        return [start_t], []
    
    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    stride = max(1, int(stride))
    
    indices = []
    step_sizes = []
    t = start_t
    
    while True:
        indices.append(t)
        if t == end_t:
            break
        
        t_next = t + step_sign * stride
        step_sizes.append(stride)
        
        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t
        
        if t_next == t:
            t_next = t + step_sign
        
        t = int(max(0, min(t_next, num_timesteps - 1)))
    
    return indices, step_sizes


def calculate_sampling_metrics(
    num_timesteps=1000,
    time_boundary=700,
    # Large Step 参数
    ls_schedule='lambda',
    ls_coeff_a=60.0,
    ls_coeff_b=20.0,
    ls_step_upper=20,
    ls_step_lower=20,
    ls_stride=25,
    ls_step_size=0.6,
    ls_noise=0.0,  # 不影响计算，仅用于记录
    # Refine 参数
    rf_schedule='lambda',
    rf_coeff_a=40.0,
    rf_coeff_b=5.0,
    rf_step_upper=20,
    rf_step_lower=5,
    rf_stride=10,
    rf_step_size=0.4,
    rf_noise=0.05,  # 不影响计算，仅用于记录
    rf_time_lower=0
):
    """
    计算总采样步数和实际长度
    
    Args:
        num_timesteps: 总扩散步数
        time_boundary: 阶段边界时间步
        ls_schedule: Large Step调度模式 ('lambda', 'linear', 'fixed')
        ls_coeff_a: Large Step lambda参数a
        ls_coeff_b: Large Step lambda参数b
        ls_step_upper: Large Step linear参数upper
        ls_step_lower: Large Step linear参数lower
        ls_stride: Large Step fixed参数stride
        ls_step_size: Large Step的step_size
        rf_schedule: Refine调度模式 ('lambda', 'linear', 'fixed')
        rf_coeff_a: Refine lambda参数a
        rf_coeff_b: Refine lambda参数b
        rf_step_upper: Refine linear参数upper
        rf_step_lower: Refine linear参数lower
        rf_stride: Refine fixed参数stride
        rf_step_size: Refine的step_size
        rf_time_lower: Refine阶段结束时间步
    
    Returns:
        tuple: (总采样步数, 实际长度)
    """
    # Large Step 阶段
    large_step_time_lower = time_boundary
    large_step_time_upper = num_timesteps - 1
    
    # Refine 阶段
    refine_time_upper = time_boundary
    refine_time_lower = rf_time_lower
    
    # 计算 Large Step 阶段时间步
    if ls_schedule == 'lambda':
        large_step_indices, _ = build_lambda_schedule(
            large_step_time_upper, large_step_time_lower,
            ls_coeff_a, ls_coeff_b, num_timesteps
        )
    elif ls_schedule == 'linear':
        large_step_indices, _ = build_linear_schedule(
            large_step_time_upper, large_step_time_lower,
            ls_step_upper, ls_step_lower, num_timesteps
        )
    else:  # fixed
        large_step_indices, _ = build_fixed_schedule(
            large_step_time_upper, large_step_time_lower,
            ls_stride, num_timesteps
        )
    
    # 计算 Refine 阶段时间步
    if rf_schedule == 'lambda':
        refine_indices, _ = build_lambda_schedule(
            refine_time_upper, refine_time_lower,
            rf_coeff_a, rf_coeff_b, num_timesteps
        )
    elif rf_schedule == 'linear':
        refine_indices, _ = build_linear_schedule(
            refine_time_upper, refine_time_lower,
            rf_step_upper, rf_step_lower, num_timesteps
        )
    else:  # fixed
        refine_indices, _ = build_fixed_schedule(
            refine_time_upper, refine_time_lower,
            rf_stride, num_timesteps
        )
    
    # 计算实际长度：ls的步数 × ls的step_size + rf的步数 × rf的step_size
    large_step_num_steps = len(large_step_indices) if large_step_indices else 0
    refine_num_steps = len(refine_indices) if refine_indices else 0
    actual_length = large_step_num_steps * ls_step_size + refine_num_steps * rf_step_size
    
    # 合并时间步序列（去掉重复的连接点）
    if large_step_indices and refine_indices:
        if large_step_indices[-1] == refine_indices[0]:
            all_indices = large_step_indices + refine_indices[1:]
        else:
            all_indices = large_step_indices + refine_indices
    elif large_step_indices:
        all_indices = large_step_indices
    elif refine_indices:
        all_indices = refine_indices
    else:
        return None, None
    
    total_steps = len(all_indices)
    
    return total_steps, actual_length


def load_config_from_file(config_file):
    """从配置文件加载参数"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"错误：配置文件不存在: {config_file}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print("错误：配置文件为空")
            return None
        
        sample_cfg = config.get('sample', {})
        num_timesteps = sample_cfg.get('num_steps', 1000)
        
        dynamic_cfg = sample_cfg.get('dynamic', {})
        time_boundary = dynamic_cfg.get('time_boundary', 700)
        
        large_step_cfg = dynamic_cfg.get('large_step', {})
        refine_cfg = dynamic_cfg.get('refine', {})
        
        return {
            'num_timesteps': num_timesteps,
            'time_boundary': time_boundary,
            'ls_schedule': large_step_cfg.get('schedule', 'lambda'),
            'ls_coeff_a': large_step_cfg.get('lambda_coeff_a', 60.0),
            'ls_coeff_b': large_step_cfg.get('lambda_coeff_b', 20.0),
            'ls_step_upper': large_step_cfg.get('linear_step_upper', 20),
            'ls_step_lower': large_step_cfg.get('linear_step_lower', 20),
            'ls_stride': large_step_cfg.get('stride', 25),
            'ls_step_size': large_step_cfg.get('step_size', 0.6),
            'ls_noise': large_step_cfg.get('noise_scale', 0.0),
            'rf_schedule': refine_cfg.get('schedule', 'lambda'),
            'rf_coeff_a': refine_cfg.get('lambda_coeff_a', 40.0),
            'rf_coeff_b': refine_cfg.get('lambda_coeff_b', 5.0),
            'rf_step_upper': refine_cfg.get('linear_step_upper', 20),
            'rf_step_lower': refine_cfg.get('linear_step_lower', 5),
            'rf_stride': refine_cfg.get('stride', 10),
            'rf_step_size': refine_cfg.get('step_size', 0.4),
            'rf_noise': refine_cfg.get('noise_scale', 0.05),
            'rf_time_lower': refine_cfg.get('time_lower', 0),
        }
    except Exception as e:
        print(f"错误：读取配置文件失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='计算采样指标：总采样步数和实际长度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认配置文件
  python3 calculate_sampling_metrics.py
  
  # 指定配置文件
  python3 calculate_sampling_metrics.py --config configs/sampling.yml
  
  # 直接通过参数计算
  python3 calculate_sampling_metrics.py \\
      --num_steps 1000 \\
      --time_boundary 700 \\
      --ls_schedule lambda --ls_coeff_a 60.0 --ls_coeff_b 20.0 --ls_step_size 0.6 \\
      --rf_schedule lambda --rf_coeff_a 40.0 --rf_coeff_b 5.0 --rf_step_size 0.4
        """
    )
    
    # 配置文件选项
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/sampling.yml）')
    
    # 基础参数
    parser.add_argument('--num_steps', type=int, default=None,
                       help='总扩散步数（默认: 1000）')
    parser.add_argument('--time_boundary', type=int, default=None,
                       help='阶段边界时间步（默认: 700）')
    
    # Large Step 参数
    parser.add_argument('--ls_schedule', type=str, choices=['lambda', 'linear', 'fixed'],
                       default=None, help='Large Step调度模式（默认: lambda）')
    parser.add_argument('--ls_coeff_a', type=float, default=None,
                       help='Large Step lambda参数a（默认: 60.0）')
    parser.add_argument('--ls_coeff_b', type=float, default=None,
                       help='Large Step lambda参数b（默认: 20.0）')
    parser.add_argument('--ls_step_upper', type=int, default=None,
                       help='Large Step linear参数upper（默认: 20）')
    parser.add_argument('--ls_step_lower', type=int, default=None,
                       help='Large Step linear参数lower（默认: 20）')
    parser.add_argument('--ls_stride', type=int, default=None,
                       help='Large Step fixed参数stride（默认: 25）')
    parser.add_argument('--ls_step_size', type=float, default=None,
                       help='Large Step的step_size（默认: 0.6）')
    parser.add_argument('--ls_noise', type=float, default=None,
                       help='Large Step的noise_scale（不影响计算，仅用于记录）')
    
    # Refine 参数
    parser.add_argument('--rf_schedule', type=str, choices=['lambda', 'linear', 'fixed'],
                       default=None, help='Refine调度模式（默认: lambda）')
    parser.add_argument('--rf_coeff_a', type=float, default=None,
                       help='Refine lambda参数a（默认: 40.0）')
    parser.add_argument('--rf_coeff_b', type=float, default=None,
                       help='Refine lambda参数b（默认: 5.0）')
    parser.add_argument('--rf_step_upper', type=int, default=None,
                       help='Refine linear参数upper（默认: 20）')
    parser.add_argument('--rf_step_lower', type=int, default=None,
                       help='Refine linear参数lower（默认: 5）')
    parser.add_argument('--rf_stride', type=int, default=None,
                       help='Refine fixed参数stride（默认: 10）')
    parser.add_argument('--rf_step_size', type=float, default=None,
                       help='Refine的step_size（默认: 0.4）')
    parser.add_argument('--rf_noise', type=float, default=None,
                       help='Refine的noise_scale（不影响计算，仅用于记录）')
    parser.add_argument('--rf_time_lower', type=int, default=None,
                       help='Refine阶段结束时间步（默认: 0）')
    
    args = parser.parse_args()
    
    # 确定参数来源：配置文件或命令行参数
    if args.config or (args.num_steps is None and args.time_boundary is None):
        # 使用配置文件
        script_dir = Path(__file__).parent
        if args.config:
            config_file = args.config
        else:
            # 尝试默认路径
            possible_paths = [
                script_dir / 'configs' / 'sampling.yml',
                Path('configs') / 'sampling.yml',
                Path('configs/sampling.yml'),
            ]
            config_file = None
            for path in possible_paths:
                if path.exists():
                    config_file = str(path)
                    break
            
            if not config_file:
                print("错误：找不到配置文件，请使用 --config 指定配置文件路径")
                return
        
        params = load_config_from_file(config_file)
        if params is None:
            return
        
        # 命令行参数覆盖配置文件参数
        if args.num_steps is not None:
            params['num_timesteps'] = args.num_steps
        if args.time_boundary is not None:
            params['time_boundary'] = args.time_boundary
        if args.ls_schedule is not None:
            params['ls_schedule'] = args.ls_schedule
        if args.ls_coeff_a is not None:
            params['ls_coeff_a'] = args.ls_coeff_a
        if args.ls_coeff_b is not None:
            params['ls_coeff_b'] = args.ls_coeff_b
        if args.ls_step_upper is not None:
            params['ls_step_upper'] = args.ls_step_upper
        if args.ls_step_lower is not None:
            params['ls_step_lower'] = args.ls_step_lower
        if args.ls_stride is not None:
            params['ls_stride'] = args.ls_stride
        if args.ls_step_size is not None:
            params['ls_step_size'] = args.ls_step_size
        if args.rf_schedule is not None:
            params['rf_schedule'] = args.rf_schedule
        if args.rf_coeff_a is not None:
            params['rf_coeff_a'] = args.rf_coeff_a
        if args.rf_coeff_b is not None:
            params['rf_coeff_b'] = args.rf_coeff_b
        if args.rf_step_upper is not None:
            params['rf_step_upper'] = args.rf_step_upper
        if args.rf_step_lower is not None:
            params['rf_step_lower'] = args.rf_step_lower
        if args.rf_stride is not None:
            params['rf_stride'] = args.rf_stride
        if args.rf_step_size is not None:
            params['rf_step_size'] = args.rf_step_size
        if args.rf_noise is not None:
            params['rf_noise'] = args.rf_noise
        if args.rf_time_lower is not None:
            params['rf_time_lower'] = args.rf_time_lower
    else:
        # 完全使用命令行参数
        params = {
            'num_timesteps': args.num_steps or 1000,
            'time_boundary': args.time_boundary or 700,
            'ls_schedule': args.ls_schedule or 'lambda',
            'ls_coeff_a': args.ls_coeff_a or 60.0,
            'ls_coeff_b': args.ls_coeff_b or 20.0,
            'ls_step_upper': args.ls_step_upper or 20,
            'ls_step_lower': args.ls_step_lower or 20,
            'ls_stride': args.ls_stride or 25,
            'ls_step_size': args.ls_step_size or 0.6,
            'ls_noise': args.ls_noise if args.ls_noise is not None else 0.0,
            'rf_schedule': args.rf_schedule or 'lambda',
            'rf_coeff_a': args.rf_coeff_a or 40.0,
            'rf_coeff_b': args.rf_coeff_b or 5.0,
            'rf_step_upper': args.rf_step_upper or 20,
            'rf_step_lower': args.rf_step_lower or 5,
            'rf_stride': args.rf_stride or 10,
            'rf_step_size': args.rf_step_size or 0.4,
            'rf_noise': args.rf_noise if args.rf_noise is not None else 0.05,
            'rf_time_lower': args.rf_time_lower or 0,
        }
    
    # 计算指标
    total_steps, actual_length = calculate_sampling_metrics(**params)
    
    if total_steps is None or actual_length is None:
        print("错误：计算失败")
        return
    
    # 输出结果
    print("=" * 60)
    print("采样指标计算结果")
    print("=" * 60)
    print(f"总采样步数: {total_steps}")
    print(f"实际长度: {actual_length:.6f}")
    print("=" * 60)
    print()
    print(f"({total_steps}, {actual_length:.6f})")


if __name__ == "__main__":
    main()

