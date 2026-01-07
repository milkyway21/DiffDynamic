#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动读取 sampling.yml 配置文件，输出跳步的时间步和梯度融合权重
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("错误：需要安装 yaml 模块")
    print("请运行: pip install pyyaml")
    sys.exit(1)

import numpy as np


def compute_grad_fusion_lambda(timestep, cfg, num_timesteps):
    """计算梯度融合lambda值"""
    if cfg is None:
        return 0.5
    
    if isinstance(cfg, (int, float)):
        return float(cfg)
    
    if isinstance(cfg, dict):
        mode = cfg.get('mode', 'linear')
        start = float(cfg.get('start', 0.8))
        end = float(cfg.get('end', 0.2))
        ratio = float(timestep) / max(float(num_timesteps - 1), 1.0)  # 归一化时间步 [0, 1]
        
        if mode == 'linear':
            # 线性插值：lambda = start * ratio + end * (1.0 - ratio)
            lambda_val = start * ratio + end * (1.0 - ratio)
            return max(min(lambda_val, 1.0), 0.0)
        
        elif mode == 'exponential':
            # 指数衰减
            if start > 0 and end > 0:
                decay_rate = (end / start) ** (1.0 - ratio)
                lambda_val = start * decay_rate
            else:
                lambda_val = start * ratio + end * (1.0 - ratio)
            return max(min(lambda_val, 1.0), 0.0)
        
        elif mode == 'quadratic':
            # 二次衰减：lambda = start - (start - end) * ratio^power
            power = float(cfg.get('power', 2.0))
            lambda_val = start - (start - end) * (ratio ** power)
            return max(min(lambda_val, 1.0), 0.0)
        
        elif mode == 'time':
            # 线性升权
            lambda_val = start + (end - start) * ratio
            return max(min(lambda_val, 1.0), 0.0)
        
        else:
            # 默认线性
            lambda_val = start * ratio + end * (1.0 - ratio)
            return max(min(lambda_val, 1.0), 0.0)
    
    return 0.5


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


def generate_sampling_steps_text(config: dict) -> str:
    """
    根据配置生成采样步骤信息的文本字符串
    
    Args:
        config: 配置字典（从 YAML 文件读取）
    
    Returns:
        str: 格式化的采样步骤信息文本
    """
    if config is None:
        return ""
    
    try:
        # 提取配置
        sample_cfg = config.get('sample', {})
        if not isinstance(sample_cfg, dict):
            sample_cfg = {}
        
        num_timesteps = sample_cfg.get('num_steps', 1000)
        grad_fusion_cfg = config.get('model', {}).get('grad_fusion_lambda', None) if isinstance(config.get('model'), dict) else None
        
        dynamic_cfg = sample_cfg.get('dynamic', {})
        if not isinstance(dynamic_cfg, dict):
            dynamic_cfg = {}
        
        large_step_cfg = dynamic_cfg.get('large_step', {})
        if not isinstance(large_step_cfg, dict):
            large_step_cfg = {}
        
        refine_cfg = dynamic_cfg.get('refine', {})
        if not isinstance(refine_cfg, dict):
            refine_cfg = {}
        
        # 获取 time_boundary（支持向后兼容）
        def get_time_boundary_from_cfg(cfg, default=600):
            """获取阶段边界时间步，支持向后兼容"""
            if not isinstance(cfg, dict):
                return default
            # 优先读取统一的 time_boundary
            if 'time_boundary' in cfg:
                return cfg.get('time_boundary', default)
            # 向后兼容：从 large_step.time_lower 获取
            large_step = cfg.get('large_step', {})
            if isinstance(large_step, dict) and 'time_lower' in large_step:
                return large_step.get('time_lower', default)
            # 向后兼容：从 refine.time_upper 获取
            refine = cfg.get('refine', {})
            if isinstance(refine, dict) and 'time_upper' in refine:
                return refine.get('time_upper', default)
            return default
        
        time_boundary = get_time_boundary_from_cfg(dynamic_cfg, 600)
        
        # Large Step 阶段
        large_step_schedule = large_step_cfg.get('schedule', 'lambda')
        large_step_time_lower = time_boundary  # 使用统一的 time_boundary
        large_step_time_upper = num_timesteps - 1  # 默认从最后一步开始
        
        # Refine 阶段
        refine_schedule = refine_cfg.get('schedule', 'lambda')
        refine_time_upper = time_boundary  # 使用统一的 time_boundary
        refine_time_lower = refine_cfg.get('time_lower', 0)
        
        # 计算 Large Step 阶段时间步
        if large_step_schedule == 'lambda':
            lambda_coeff_a = large_step_cfg.get('lambda_coeff_a', 80.0)
            lambda_coeff_b = large_step_cfg.get('lambda_coeff_b', 20.0)
            large_step_indices, large_step_steps = build_lambda_schedule(
                large_step_time_upper, large_step_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif large_step_schedule == 'linear':
            step_upper = large_step_cfg.get('linear_step_upper', 100.0)
            step_lower = large_step_cfg.get('linear_step_lower', 20.0)
            large_step_indices, large_step_steps = build_linear_schedule(
                large_step_time_upper, large_step_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = large_step_cfg.get('stride', 15)
            large_step_indices, large_step_steps = build_fixed_schedule(
                large_step_time_upper, large_step_time_lower,
                stride, num_timesteps
            )
        
        # 计算 Refine 阶段时间步
        if refine_schedule == 'lambda':
            lambda_coeff_a = refine_cfg.get('lambda_coeff_a', 40.0)
            lambda_coeff_b = refine_cfg.get('lambda_coeff_b', 5.0)
            refine_indices, refine_steps = build_lambda_schedule(
                refine_time_upper, refine_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif refine_schedule == 'linear':
            step_upper = refine_cfg.get('linear_step_upper', 40.0)
            step_lower = refine_cfg.get('linear_step_lower', 5.0)
            refine_indices, refine_steps = build_linear_schedule(
                refine_time_upper, refine_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = refine_cfg.get('stride', 8)
            refine_indices, refine_steps = build_fixed_schedule(
                refine_time_upper, refine_time_lower,
                stride, num_timesteps
            )
        
        # 合并时间步序列（去掉重复的连接点）
        if large_step_indices and refine_indices:
            if large_step_indices[-1] == refine_indices[0]:
                all_indices = large_step_indices + refine_indices[1:]
                all_steps = large_step_steps + refine_steps
            else:
                all_indices = large_step_indices + refine_indices
                all_steps = large_step_steps + refine_steps
        elif large_step_indices:
            all_indices = large_step_indices
            all_steps = large_step_steps
        elif refine_indices:
            all_indices = refine_indices
            all_steps = refine_steps
        else:
            # 如果没有配置任何阶段，返回空字符串
            return ""
        
        # 生成文本字符串
        lines = []
        lines.append("序号     当前时间步        下一步时间步         跳步步长       梯度融合权重             阶段")
        lines.append("-" * 120)
        
        step_num = 0
        for i in range(len(all_indices) - 1):
            current_t = all_indices[i]
            next_t = all_indices[i + 1]
            step_size = all_steps[i] if i < len(all_steps) else abs(current_t - next_t)
            
            # 计算当前时间步的梯度融合权重
            lambda_val = compute_grad_fusion_lambda(current_t, grad_fusion_cfg, num_timesteps)
            
            # 判断阶段
            if current_t > refine_time_upper:
                stage = "Large Step"
            else:
                stage = "Refine"
            
            step_num += 1
            lines.append(f"{step_num:<6} {current_t:<12} {next_t:<14} {step_size:<10} {lambda_val:<18.6f} {stage:<15}")
        
        # 最后一步
        step_num += 1
        final_t = all_indices[-1]
        final_lambda = compute_grad_fusion_lambda(final_t, grad_fusion_cfg, num_timesteps)
        lines.append(f"{step_num:<6} {final_t:<12} {'结束':<14} {'-':<10} {final_lambda:<18.6f} {'Refine':<15}")
        
        return "\n".join(lines)
    except Exception as e:
        # 如果生成过程中出现任何错误，返回错误信息
        return f"生成采样步骤信息时出错: {str(e)}"


def main():
    # 读取配置文件 - 使用相对路径，优先使用脚本所在目录，否则使用当前工作目录
    script_dir = Path(__file__).parent
    current_dir = Path.cwd()
    
    # 尝试多个可能的路径
    possible_paths = [
        script_dir / "configs" / "sampling.yml",  # 脚本所在目录下的 configs
        current_dir / "configs" / "sampling.yml",  # 当前工作目录下的 configs
        script_dir.parent / "configs" / "sampling.yml",  # 脚本父目录下的 configs
    ]
    
    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        print(f"错误：找不到配置文件 sampling.yml")
        print(f"已尝试的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"当前工作目录: {current_dir}")
        print(f"脚本所在目录: {script_dir}")
        return
    
    # 读取配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"错误：读取配置文件失败: {e}")
        return
    
    if config is None:
        print("错误：配置文件为空或格式错误")
        return
    
    # 提取配置
    num_timesteps = config.get('sample', {}).get('num_steps', 1000)
    grad_fusion_cfg = config.get('model', {}).get('grad_fusion_lambda', None)
    
    dynamic_cfg = config.get('sample', {}).get('dynamic', {})
    large_step_cfg = dynamic_cfg.get('large_step', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    
    # 获取 time_boundary（支持向后兼容）
    def get_time_boundary_from_cfg(cfg, default=600):
        """获取阶段边界时间步，支持向后兼容"""
        if not isinstance(cfg, dict):
            return default
        # 优先读取统一的 time_boundary
        if 'time_boundary' in cfg:
            return cfg.get('time_boundary', default)
        # 向后兼容：从 large_step.time_lower 获取
        large_step = cfg.get('large_step', {})
        if isinstance(large_step, dict) and 'time_lower' in large_step:
            return large_step.get('time_lower', default)
        # 向后兼容：从 refine.time_upper 获取
        refine = cfg.get('refine', {})
        if isinstance(refine, dict) and 'time_upper' in refine:
            return refine.get('time_upper', default)
        return default
    
    time_boundary = get_time_boundary_from_cfg(dynamic_cfg, 600)
    
    # Large Step 阶段
    large_step_schedule = large_step_cfg.get('schedule', 'lambda')
    large_step_time_lower = time_boundary  # 使用统一的 time_boundary
    large_step_time_upper = num_timesteps - 1  # 默认从最后一步开始
    
    # Refine 阶段
    refine_schedule = refine_cfg.get('schedule', 'lambda')
    refine_time_upper = time_boundary  # 使用统一的 time_boundary
    refine_time_lower = refine_cfg.get('time_lower', 0)
    
    print("=" * 120)
    print("采样配置信息")
    print("=" * 120)
    print(f"总扩散步数: {num_timesteps}")
    print(f"梯度融合配置: {grad_fusion_cfg}")
    print(f"Large Step 调度模式: {large_step_schedule}")
    print(f"Large Step 时间范围: {large_step_time_upper} → {large_step_time_lower}")
    print(f"Refine 调度模式: {refine_schedule}")
    print(f"Refine 时间范围: {refine_time_upper} → {refine_time_lower}")
    print()
    
    # 计算 Large Step 阶段时间步
    print("=" * 120)
    print("计算 Large Step 阶段时间步...")
    print("=" * 120)
    
    if large_step_schedule == 'lambda':
        lambda_coeff_a = large_step_cfg.get('lambda_coeff_a', 80.0)
        lambda_coeff_b = large_step_cfg.get('lambda_coeff_b', 20.0)
        large_step_indices, large_step_steps = build_lambda_schedule(
            large_step_time_upper, large_step_time_lower,
            lambda_coeff_a, lambda_coeff_b, num_timesteps
        )
        print(f"使用 Lambda 调度: coeff_a={lambda_coeff_a}, coeff_b={lambda_coeff_b}")
    
    elif large_step_schedule == 'linear':
        step_upper = large_step_cfg.get('linear_step_upper', 100.0)
        step_lower = large_step_cfg.get('linear_step_lower', 20.0)
        large_step_indices, large_step_steps = build_linear_schedule(
            large_step_time_upper, large_step_time_lower,
            step_upper, step_lower, num_timesteps
        )
        print(f"使用 Linear 调度: step_upper={step_upper}, step_lower={step_lower}")
    
    else:
        stride = large_step_cfg.get('stride', 15)
        large_step_indices, large_step_steps = build_fixed_schedule(
            large_step_time_upper, large_step_time_lower,
            stride, num_timesteps
        )
        print(f"使用固定步长调度: stride={stride}")
    
    print(f"Large Step 阶段共 {len(large_step_indices)} 步")
    print()
    
    # 计算 Refine 阶段时间步
    print("=" * 120)
    print("计算 Refine 阶段时间步...")
    print("=" * 120)
    
    if refine_schedule == 'lambda':
        lambda_coeff_a = refine_cfg.get('lambda_coeff_a', 40.0)
        lambda_coeff_b = refine_cfg.get('lambda_coeff_b', 5.0)
        refine_indices, refine_steps = build_lambda_schedule(
            refine_time_upper, refine_time_lower,
            lambda_coeff_a, lambda_coeff_b, num_timesteps
        )
        print(f"使用 Lambda 调度: coeff_a={lambda_coeff_a}, coeff_b={lambda_coeff_b}")
    
    elif refine_schedule == 'linear':
        step_upper = refine_cfg.get('linear_step_upper', 40.0)
        step_lower = refine_cfg.get('linear_step_lower', 5.0)
        refine_indices, refine_steps = build_linear_schedule(
            refine_time_upper, refine_time_lower,
            step_upper, step_lower, num_timesteps
        )
        print(f"使用 Linear 调度: step_upper={step_upper}, step_lower={step_lower}")
    
    else:
        stride = refine_cfg.get('stride', 8)
        refine_indices, refine_steps = build_fixed_schedule(
            refine_time_upper, refine_time_lower,
            stride, num_timesteps
        )
        print(f"使用固定步长调度: stride={stride}")
    
    print(f"Refine 阶段共 {len(refine_indices)} 步")
    print()
    
    # 合并时间步序列（去掉重复的连接点）
    if large_step_indices and refine_indices:
        if large_step_indices[-1] == refine_indices[0]:
            all_indices = large_step_indices + refine_indices[1:]
            all_steps = large_step_steps + refine_steps
        else:
            all_indices = large_step_indices + refine_indices
            all_steps = large_step_steps + refine_steps
    elif large_step_indices:
        all_indices = large_step_indices
        all_steps = large_step_steps
    else:
        all_indices = refine_indices
        all_steps = refine_steps
    
    # 输出结果
    print("=" * 120)
    print("完整采样序列：跳步时间步和梯度融合权重")
    print("=" * 120)
    print()
    
    # 使用 generate_sampling_steps_text 生成文本并打印
    steps_text = generate_sampling_steps_text(config)
    print(steps_text)
    
    print()
    print("=" * 120)
    print("统计信息")
    print("=" * 120)
    print(f"总采样步数: {len(all_indices)}")
    if large_step_indices:
        print(f"Large Step 阶段: {len(large_step_indices)}步 (时间步 {large_step_indices[0]} → {large_step_indices[-1]})")
    if refine_indices:
        print(f"Refine 阶段: {len(refine_indices)}步 (时间步 {refine_indices[0]} → {refine_indices[-1]})")
    print()
    
    # 梯度融合权重统计
    if grad_fusion_cfg:
        print("梯度融合权重范围:")
        lambda_start = compute_grad_fusion_lambda(num_timesteps - 1, grad_fusion_cfg, num_timesteps)
        lambda_mid = compute_grad_fusion_lambda(num_timesteps // 2, grad_fusion_cfg, num_timesteps)
        lambda_end = compute_grad_fusion_lambda(0, grad_fusion_cfg, num_timesteps)
        print(f"  - 初始值 (t={num_timesteps - 1}): {lambda_start:.6f}")
        print(f"  - 中间值 (t={num_timesteps // 2}): {lambda_mid:.6f}")
        print(f"  - 最终值 (t=0): {lambda_end:.6f}")
        
        if isinstance(grad_fusion_cfg, dict):
            mode = grad_fusion_cfg.get('mode', 'linear')
            start = grad_fusion_cfg.get('start', 0.8)
            end = grad_fusion_cfg.get('end', 0.2)
            print()
            print("梯度融合权重计算公式:")
            print(f"  模式: {mode}")
            print(f"  起始值: {start}")
            print(f"  结束值: {end}")
            if mode == 'quadratic':
                power = grad_fusion_cfg.get('power', 2.0)
                print(f"  衰减指数: {power}")
                print()
                print("  计算公式: lambda = start - (start - end) * (ratio ^ power)")
                print("  其中 ratio = timestep / (num_timesteps - 1)  # 归一化时间步 [0, 1]")
            elif mode == 'linear':
                print()
                print("  计算公式: lambda = start * ratio + end * (1.0 - ratio)")
                print("  其中 ratio = timestep / (num_timesteps - 1)  # 归一化时间步 [0, 1]")
    
    print()
    print("=" * 120)


if __name__ == "__main__":
    main()

