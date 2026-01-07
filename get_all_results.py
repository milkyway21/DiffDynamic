#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有采样指标计算命令并输出结果
"""

import subprocess
import sys

commands = [
    # 第1行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第2行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "40.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "20.0", "--rf_coeff_b", "2.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第3行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "20.0", "--ls_coeff_b", "10.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "2.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第4行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "400",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "10.0", "--ls_step_size", "0.5", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第5行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "300",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "10.0", "--ls_step_size", "0.5", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第6行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "300",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "20.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.3", "--rf_noise", "0.05"],
    
    # 第7行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "150",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "5.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.4", "--rf_noise", "0.05"],
    
    # 第8行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.6", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.4", "--rf_noise", "0.05"],
    
    # 第9行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.5", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第10行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "40.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第11行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.6", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第12行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第13行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第14行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第15行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "0.4", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.05"],
    
    # 第16行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "10.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.1", "--rf_noise", "0.1"],
    
    # 第17行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "500",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
    
    # 第18行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "550",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
    
    # 第19行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "600",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
    
    # 第20行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "650",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
    
    # 第21行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "700",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
    
    # 第22行数据
    ["python3", "calculate_sampling_metrics.py", "--num_steps", "1000", "--time_boundary", "750",
     "--ls_schedule", "lambda", "--ls_coeff_a", "80.0", "--ls_coeff_b", "20.0", "--ls_step_size", "1.0", "--ls_noise", "0.0",
     "--rf_schedule", "lambda", "--rf_coeff_a", "40.0", "--rf_coeff_b", "5.0", "--rf_step_size", "0.2", "--rf_noise", "0.1"],
]

print("=" * 80)
print("所有采样指标计算结果")
print("=" * 80)
print()

for i, cmd in enumerate(commands, 1):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # 提取最后一行（元组格式）
        output_lines = result.stdout.strip().split('\n')
        tuple_line = None
        for line in reversed(output_lines):
            if line.startswith('(') and ',' in line and line.endswith(')'):
                tuple_line = line
                break
        
        if tuple_line:
            print(f"# 第{i}行数据")
            # 移除第一个元素（脚本名），只保留参数
            args_only = cmd[2:] if len(cmd) > 2 and cmd[1] == 'calculate_sampling_metrics.py' else cmd[1:]
            print(f"python3 calculate_sampling_metrics.py {' '.join(args_only)}")
            print(f"结果: {tuple_line}")
            print()
        else:
            print(f"# 第{i}行数据 - 解析失败")
            print(f"python3 calculate_sampling_metrics.py {' '.join(cmd[1:])}")
            print()
    except subprocess.CalledProcessError as e:
        print(f"# 第{i}行数据 - 执行失败")
        print(f"python3 calculate_sampling_metrics.py {' '.join(cmd[1:])}")
        print(f"错误: {e.stderr}")
        print()

print("=" * 80)

