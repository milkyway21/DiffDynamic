#!/usr/bin/env python3
"""
批量采样和评估脚本：并行执行采样和评估

功能：
1. 对每个 data_id 执行采样：python3 scripts/sample_diffusion.py configs/sampling.yml --data_id {i} --device cuda:{gpu_id}
2. 找到生成的文件：outputs/result_{data_id}_{timestamp}.pt
3. 执行评估：python3 evaluate_pt_with_correct_reconstruct.py {pt_file} --protein_root ... --output_dir ... --atom_mode add_aromatic --exhaustiveness 8

并行配置：
- GPU: 使用GPU 2、3、4、5（4个GPU并行）
- CPU: 使用64个核心（通过multiprocessing限制）

使用方法：
    # 基本用法（0到99，使用默认GPU 2,3,4,5）
    python3 batch_sampleandeval_parallel.py
    
    # 指定范围
    python3 batch_sampleandeval_parallel.py --start 0 --end 99
    
    # 指定使用的GPU（多种格式）
    python3 batch_sampleandeval_parallel.py --gpus "0,1,2,3"
    python3 batch_sampleandeval_parallel.py --gpus "0-3"
    python3 batch_sampleandeval_parallel.py --gpus "0,2-4,6"
    python3 batch_sampleandeval_parallel.py --gpus "all"  # 使用所有可用GPU
    
    # 指定CPU核心数
    python3 batch_sampleandeval_parallel.py --num_cpu_cores 32
    
    # 组合使用
    python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus "0" --num_cpu_cores 20
     iii
    # 指定蛋白质数据根目录
    python3 batch_sampleandeval_parallel.py --protein_root /path/to/protein/data
    
    # 只生成模式（不执行评估）
    python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus "0-5" --sample-only
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import glob
import threading
import traceback
import re
import signal
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️  警告: pandas未安装，无法记录Excel。运行: pip install pandas openpyxl")
else:
    try:
        import openpyxl
    except ImportError:
        print("⚠️  警告: openpyxl未安装，无法写入Excel。运行: pip install openpyxl")

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None
    print("⚠️  警告: torch或numpy未安装，可能影响功能")
    if torch is None:
        print("   请安装: pip install torch")
    if np is None:
        print("   请安装: pip install numpy")

try:
    import yaml
except ImportError:
    yaml = None
    print("⚠️  警告: yaml未安装，无法读取配置文件参数。运行: pip install pyyaml")

# 并行配置（默认值，可通过命令行参数覆盖）
DEFAULT_GPU_IDS = [2, 3, 4, 5]  # 默认使用的GPU ID列表
DEFAULT_NUM_CPU_CORES = 64  # 默认使用的CPU核心数


def parse_gpu_ids(gpu_str):
    """
    解析GPU ID字符串，支持多种格式：
    - "0,1,2,3" -> [0, 1, 2, 3]
    - "0-3" -> [0, 1, 2, 3]
    - "0,2-4,6" -> [0, 2, 3, 4, 6]
    - "all" -> 自动检测所有可用GPU
    
    Args:
        gpu_str: GPU ID字符串
    
    Returns:
        list: GPU ID列表
    """
    if gpu_str.lower() == 'all':
        # 自动检测所有可用GPU
        if torch is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        else:
            print("⚠️  警告: CUDA不可用，无法自动检测GPU")
            return []
    
    gpu_ids = []
    parts = gpu_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 范围格式，如 "0-3"
            start, end = part.split('-')
            try:
                start_id = int(start.strip())
                end_id = int(end.strip())
                gpu_ids.extend(range(start_id, end_id + 1))
            except ValueError:
                raise ValueError(f"无效的GPU范围格式: {part}")
        else:
            # 单个ID
            try:
                gpu_ids.append(int(part))
            except ValueError:
                raise ValueError(f"无效的GPU ID: {part}")
    
    # 去重并排序
    gpu_ids = sorted(list(set(gpu_ids)))
    return gpu_ids


def get_available_gpus():
    """
    获取所有可用的GPU ID列表
    
    Returns:
        list: 可用GPU ID列表
    """
    if torch is not None and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

# Excel写入锁（将在main函数中通过Manager创建，用于进程间共享）
excel_write_lock = None

# 项目根目录
REPO_ROOT = Path(__file__).parent
SCRIPT = REPO_ROOT / 'scripts' / 'sample_diffusion.py'
CONFIG = REPO_ROOT / 'configs' / 'sampling.yml'
EVAL_SCRIPT = REPO_ROOT / 'evaluate_pt_with_correct_reconstruct.py'
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def format_float_for_filename(value):
    """
    将浮点数格式化为文件名格式，用p代替小数点
    例如：80.0 -> 80p0, 10.5 -> 10p5, 80 -> 80p0
    """
    if isinstance(value, (int, float)):
        # 转换为浮点数以确保统一格式
        float_value = float(value)
        # 转换为字符串，用p代替小数点
        str_value = str(float_value)
        return str_value.replace('.', 'p')
    return str(value)


def generate_config_params_string(config_file):
    """
    从sampling.yml配置文件中提取核心参数并生成参数字符串
    格式根据调度模式动态调整：
    - 梯度融合：gf{mode}_{start}_{end}
    - 时间边界：tl{time_boundary}
    - 大步探索阶段（根据schedule模式）：
      * lambda模式：lslambda_{ls_a}_{ls_b}_lsstep_{step_size}_lsnoise_{noise_scale}
      * linear模式：lslinear_{lower}_{upper}_lsstep_{step_size}_lsnoise_{noise_scale}
      * fixed模式：lsfixed_{stride}_lsstep_{step_size}_lsnoise_{noise_scale}
      * 其他模式：ls{schedule}_{stride}_lsstep_{step_size}_lsnoise_{noise_scale}
    - 精炼阶段（根据schedule模式）：
      * lambda模式：rflambda_{rf_a}_{rf_b}_rfstep_{step_size}_rfnoise_{noise_scale}
      * linear模式：rflinear_{lower}_{upper}_rfstep_{step_size}_rfnoise_{noise_scale}
      * fixed模式：rffixed_{stride}_rfstep_{step_size}_rfnoise_{noise_scale}
      * 其他模式：rf{schedule}_{stride}_rfstep_{step_size}_rfnoise_{noise_scale}
    
    例如：
    - lambda模式：gfquadratic_1_0_tl750_lslambda_80p0_20p0_lsstep_0p6_lsnoise_0p0_rflambda_10p0_5p0_rfstep_0p4_rfnoise_0p05
    - linear模式：gfquadratic_1_0_tl750_lslinear_20_20_lsstep_0p6_lsnoise_0p0_rflinear_5_20_rfstep_0p4_rfnoise_0p05
    - fixed模式：gfquadratic_1_0_tl750_lsfixed_25_lsstep_0p6_lsnoise_0p0_rffixed_10_rfstep_0p4_rfnoise_0p05
    
    Args:
        config_file: 配置文件路径
    
    Returns:
        str: 参数字符串，如果读取失败则返回空字符串
    """
    if yaml is None:
        return ""
    
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return ""
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return ""
        
        params = []
        
        # 梯度融合参数
        if 'model' in config and 'grad_fusion_lambda' in config['model']:
            gf_config = config['model']['grad_fusion_lambda']
            mode = gf_config.get('mode', 'unknown')
            start = gf_config.get('start', 0)
            end = gf_config.get('end', 0)
            params.append(f"gf{mode}_{start}_{end}")
        
        # 时间边界
        if 'sample' in config and 'dynamic' in config['sample']:
            dynamic_config = config['sample']['dynamic']
            time_boundary = dynamic_config.get('time_boundary', 0)
            params.append(f"tl{time_boundary}")
            
            # 大步探索阶段参数（根据schedule模式动态调整）
            if 'large_step' in dynamic_config:
                ls_config = dynamic_config['large_step']
                schedule = ls_config.get('schedule', 'unknown')
                
                # 获取step_size和noise_scale参数
                ls_step_size = ls_config.get('step_size', 1.0)
                ls_noise_scale = ls_config.get('noise_scale', 0.0)
                ls_step_size_str = format_float_for_filename(ls_step_size)
                ls_noise_scale_str = format_float_for_filename(ls_noise_scale)
                
                if schedule == 'lambda':
                    # Lambda调度模式
                    ls_a = ls_config.get('lambda_coeff_a', 0.0)
                    ls_b = ls_config.get('lambda_coeff_b', 0.0)
                    ls_a_str = format_float_for_filename(ls_a)
                    ls_b_str = format_float_for_filename(ls_b)
                    params.append(f"lslambda_{ls_a_str}_{ls_b_str}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                elif schedule == 'linear':
                    # 线性调度模式
                    ls_lower = ls_config.get('linear_step_lower', 0)
                    ls_upper = ls_config.get('linear_step_upper', 0)
                    params.append(f"lslinear_{ls_lower}_{ls_upper}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                elif schedule == 'fixed':
                    # 固定步长模式
                    ls_stride = ls_config.get('stride', 0)
                    params.append(f"lsfixed_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                else:
                    # 其他模式，使用stride作为标识
                    ls_stride = ls_config.get('stride', 0)
                    schedule_safe = str(schedule).replace('-', '_').replace(' ', '_')
                    params.append(f"ls{schedule_safe}_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
            
            # 精炼阶段参数（根据schedule模式动态调整）
            if 'refine' in dynamic_config:
                rf_config = dynamic_config['refine']
                schedule = rf_config.get('schedule', 'unknown')
                
                # 获取step_size和noise_scale参数
                rf_step_size = rf_config.get('step_size', 0.2)
                rf_noise_scale = rf_config.get('noise_scale', 0.05)
                rf_step_size_str = format_float_for_filename(rf_step_size)
                rf_noise_scale_str = format_float_for_filename(rf_noise_scale)
                
                if schedule == 'lambda':
                    # Lambda调度模式
                    rf_a = rf_config.get('lambda_coeff_a', 0.0)
                    rf_b = rf_config.get('lambda_coeff_b', 0.0)
                    rf_a_str = format_float_for_filename(rf_a)
                    rf_b_str = format_float_for_filename(rf_b)
                    params.append(f"rflambda_{rf_a_str}_{rf_b_str}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                elif schedule == 'linear':
                    # 线性调度模式
                    rf_lower = rf_config.get('linear_step_lower', 0)
                    rf_upper = rf_config.get('linear_step_upper', 0)
                    params.append(f"rflinear_{rf_lower}_{rf_upper}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                elif schedule == 'fixed':
                    # 固定步长模式
                    rf_stride = rf_config.get('stride', 0)
                    params.append(f"rffixed_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                else:
                    # 其他模式，使用stride作为标识
                    rf_stride = rf_config.get('stride', 0)
                    schedule_safe = str(schedule).replace('-', '_').replace(' ', '_')
                    params.append(f"rf{schedule_safe}_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
        
        return "_".join(params)
        
    except Exception as e:
        print(f"⚠️  读取配置文件参数失败: {e}")
        return ""


def find_latest_result_file(data_id, output_dir=None):
    """
    查找指定data_id最新生成的.pt文件
    
    Args:
        data_id: 数据ID
        output_dir: 输出目录（默认：outputs）
    
    Returns:
        Path对象或None
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # 查找所有匹配的.pt文件（格式：result_{data_id}_*.pt）
    pattern = str(output_dir / f'result_{data_id}_*.pt')
    pt_files = glob.glob(pattern)
    
    if not pt_files:
        return None
    
    # 按修改时间排序，返回最新的
    pt_files.sort(key=os.path.getmtime, reverse=True)
    
    # 返回最新的文件
    return Path(pt_files[0]) if pt_files else None


def read_evaluation_results(pt_file_path, data_id, wait_timeout=300):
    """
    读取评估结果文件中的统计数据
    
    Args:
        pt_file_path: 采样结果.pt文件路径
        data_id: 数据ID
        wait_timeout: 等待评估结果的最大时间（秒）
    
    Returns:
        tuple: (success, vina_mean, vina_median, num_scores, message, eval_output_dir)
    """
    if torch is None or np is None:
        return (False, None, None, 0, "torch或numpy未安装", None)
    
    pt_file_path = Path(pt_file_path).resolve()
    outputs_dir = pt_file_path.parent
    
    # 从.pt文件名提取口袋编号
    pt_filename = pt_file_path.stem
    if pt_filename.startswith('result_'):
        parts = pt_filename.split('_')
        if len(parts) >= 3:
            pocket_id = parts[1]
        else:
            pocket_id = str(data_id)
    else:
        pocket_id = str(data_id)
    
    # 查找评估目录
    eval_dirs = list(outputs_dir.glob(f'eval_{pocket_id}_*'))
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob('eval_*'))
    
    if not eval_dirs:
        return (False, None, None, 0, f"未找到评估输出目录（在 {outputs_dir} 中，查找模式: eval_{pocket_id}_*）", None)
    
    # 优先选择带时间戳的新格式目录
    timestamp_pattern = r'_\d{8}_\d{6}_'
    new_format_dirs = [d for d in eval_dirs if re.search(timestamp_pattern, d.name)]
    old_format_dirs = [d for d in eval_dirs if d not in new_format_dirs]
    
    if new_format_dirs:
        new_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = new_format_dirs[0]
    elif old_format_dirs:
        old_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = old_format_dirs[0]
    else:
        eval_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = eval_dirs[0]
    
    if not eval_output_dir.exists():
        return (False, None, None, 0, f"评估输出目录不存在: {eval_output_dir}", None)
    
    # 等待评估结果文件生成
    start_wait = time.time()
    eval_result_files = []
    while time.time() - start_wait < wait_timeout:
        eval_result_files = list(eval_output_dir.glob('eval_results_*.pt'))
        if eval_result_files:
            break
        time.sleep(2)
    
    if not eval_result_files:
        all_files = list(eval_output_dir.glob('*'))
        file_list = ', '.join([f.name for f in all_files[:10]])
        if len(all_files) > 10:
            file_list += f' ... (共{len(all_files)}个文件)'
        return (False, None, None, 0, 
                f"等待{wait_timeout}秒后仍未找到评估结果文件 (eval_results_*.pt)\n"
                f"   评估目录: {eval_output_dir}\n"
                f"   目录中的文件: {file_list if all_files else '空目录'}", 
                str(eval_output_dir))
    
    try:
        latest_eval_file = max(eval_result_files, key=os.path.getmtime)
        eval_data = torch.load(latest_eval_file, map_location='cpu')
        
        statistics = eval_data.get('statistics', {})
        vina_dock_scores = statistics.get('vina_dock_scores', [])
        vina_score_only_scores = statistics.get('vina_score_only_scores', [])
        vina_minimize_scores = statistics.get('vina_minimize_scores', [])
        vina_scores = statistics.get('vina_scores', [])
        
        if vina_dock_scores:
            vina_scores = vina_dock_scores
        elif vina_minimize_scores:
            vina_scores = vina_minimize_scores
        elif vina_score_only_scores:
            vina_scores = vina_score_only_scores
        
        n_reconstruct_success = eval_data.get('n_reconstruct_success', 0)
        n_eval_success = eval_data.get('n_eval_success', 0)
        
        if not vina_scores:
            diagnostic_msg = f"评估结果中无vina得分"
            if n_reconstruct_success > 0 and n_eval_success == 0:
                diagnostic_msg += f" (重建成功{n_reconstruct_success}个，但对接全部失败)"
            elif n_reconstruct_success == 0:
                diagnostic_msg += f" (重建失败，重建成功数: {n_reconstruct_success})"
            return (False, None, None, 0, diagnostic_msg, str(eval_output_dir))
        
        vina_mean = float(np.mean(vina_scores))
        vina_median = float(np.median(vina_scores))
        num_scores = len(vina_scores)
        
        return (True, vina_mean, vina_median, num_scores, 
                f"成功读取评估结果，得分数量: {num_scores}", str(eval_output_dir))
        
    except Exception as e:
        return (False, None, None, 0, f"读取评估结果异常: {str(e)}", str(eval_output_dir))


def run_single_sample(data_id, config_file, gpu_id, max_retries=3, retry_delay=5):
    """
    执行单个采样任务（带GPU指定和重试机制）
    
    Args:
        data_id: 数据ID
        config_file: 配置文件路径
        gpu_id: GPU ID
        max_retries: 最大重试次数（默认3次）
        retry_delay: 重试延迟（秒，默认5秒）
    
    Returns:
        tuple: (success, pt_file_path, message)
    """
    if config_file is None:
        config_file = CONFIG
    
    print(f"[GPU {gpu_id}] 开始采样 data_id={data_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    # 构建采样命令（指定GPU为cuda:0，因为CUDA_VISIBLE_DEVICES已经在process_sampling_task中设置）
    # 在这个进程中，cuda:0 对应物理GPU gpu_id
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        '--data_id', str(data_id),
        '--device', 'cuda:0'  # 使用cuda:0，因为CUDA_VISIBLE_DEVICES已经在进程级别设置了
    ]
    
    # 重试机制
    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"[GPU {gpu_id}] 重试采样 data_id={data_id} (尝试 {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)  # 等待一段时间再重试
        
        try:
            # 执行采样
            # 使用subprocess.run捕获输出，确保能获取完整错误信息
            # 注意：如果进程被系统杀死（如OOM），可能无法捕获完整输出
            # 环境变量已经在进程级别设置，子进程会自动继承
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,  # 捕获输出以避免混乱
                text=True,
                timeout=3600,  # 设置超时时间（1小时）
                encoding='utf-8',
                errors='replace'  # 处理编码错误，避免因编码问题导致输出截断
                # 不需要显式传递env，因为进程级别已经设置了CUDA_VISIBLE_DEVICES
            )
            
            # 等待一小段时间，确保文件已保存
            time.sleep(1)
            
            # 查找生成的文件
            pt_file = find_latest_result_file(data_id)
            
            if pt_file and pt_file.exists():
                print(f"[GPU {gpu_id}] ✅ 采样成功: {pt_file}")
                return True, str(pt_file), "采样成功"
            else:
                print(f"[GPU {gpu_id}] ⚠️  采样完成但未找到结果文件 (data_id={data_id})")
                # 输出stdout和stderr帮助调试
                if result.stdout:
                    print(f"[GPU {gpu_id}] stdout: {result.stdout[-500:]}")  # 最后500字符
                if result.stderr:
                    print(f"[GPU {gpu_id}] stderr: {result.stderr[-500:]}")
                return False, None, "未找到结果文件"
                
        except subprocess.TimeoutExpired as e:
            error_msg = f"采样超时 (超过1小时)"
            if e.stdout:
                error_msg += f"\nstdout (最后500字符): {e.stdout[-500:]}"
            if e.stderr:
                error_msg += f"\nstderr (最后500字符): {e.stderr[-500:]}"
            print(f"[GPU {gpu_id}] ❌ 采样超时 (data_id={data_id})")
            print(f"[GPU {gpu_id}] {error_msg}")
            # 保存完整超时日志
            error_log_file = OUTPUT_DIR / f"sampling_timeout_{data_id}_{int(time.time())}.log"
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"采样超时 (data_id={data_id}, GPU={gpu_id})\n")
                    f.write(f"命令: {' '.join(cmd)}\n")
                    if e.stdout:
                        f.write(f"\n完整stdout:\n{e.stdout}\n")
                    if e.stderr:
                        f.write(f"\n完整stderr:\n{e.stderr}\n")
                print(f"[GPU {gpu_id}] 完整超时日志已保存到: {error_log_file}")
            except Exception as log_err:
                print(f"[GPU {gpu_id}] 保存超时日志失败: {log_err}")
            last_error = error_msg
            # 超时不重试
            return False, None, error_msg
        except subprocess.CalledProcessError as e:
            # 组合stdout和stderr获取完整错误信息
            error_parts = []
            if e.stdout:
                error_parts.append(f"stdout: {e.stdout}")
            if e.stderr:
                error_parts.append(f"stderr: {e.stderr}")
            if not error_parts:
                error_parts.append(str(e))
            
            error_msg = "\n".join(error_parts)
            last_error = error_msg
            
            # 检查是否是CUDA初始化错误（可重试的错误）
            is_cuda_error = False
            if e.stderr and ('CUBLAS_STATUS_NOT_INITIALIZED' in e.stderr or 
                           'CUDA error' in e.stderr or
                           'cublasCreate' in e.stderr):
                is_cuda_error = True
                print(f"[GPU {gpu_id}] ⚠️  检测到CUDA初始化错误，将重试...")
            
            # 始终保存完整错误日志到文件（即使很短）
            error_log_file = OUTPUT_DIR / f"sampling_error_{data_id}_{int(time.time())}.log"
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"采样失败 (data_id={data_id}, GPU={gpu_id}, 尝试 {attempt + 1}/{max_retries})\n")
                    f.write(f"命令: {' '.join(cmd)}\n")
                    f.write(f"返回码: {e.returncode}\n")
                    f.write(f"\n完整stdout ({len(e.stdout) if e.stdout else 0} 字符):\n")
                    f.write(f"{e.stdout if e.stdout else '(无)'}\n")
                    f.write(f"\n完整stderr ({len(e.stderr) if e.stderr else 0} 字符):\n")
                    f.write(f"{e.stderr if e.stderr else '(无)'}\n")
                    f.write(f"\n组合错误信息:\n{error_msg}\n")
                if attempt == 0:  # 只在第一次失败时打印
                    print(f"[GPU {gpu_id}] ❌ 采样失败 (data_id={data_id}, 返回码={e.returncode})")
                    print(f"[GPU {gpu_id}] 完整错误日志已保存到: {error_log_file}")
            except Exception as log_err:
                print(f"[GPU {gpu_id}] ⚠️  保存错误日志失败: {log_err}")
            
            # 显示错误摘要（前500字符）
            if attempt == 0:  # 只在第一次失败时显示详细错误
                error_summary = error_msg[:500] if len(error_msg) > 500 else error_msg
                if len(error_msg) > 500:
                    error_summary += f"\n... (共{len(error_msg)}字符，完整信息已保存到日志文件)"
                print(f"[GPU {gpu_id}] 错误摘要:")
                for line in error_summary.split('\n')[:20]:  # 最多显示20行
                    if line.strip():
                        print(f"[GPU {gpu_id}]   {line}")
                
                # 诊断常见问题
                if not e.stdout and not e.stderr:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 没有捕获到任何输出，可能原因:")
                    print(f"[GPU {gpu_id}]   - 进程被系统杀死（OOM killer）")
                    print(f"[GPU {gpu_id}]   - GPU内存不足")
                    print(f"[GPU {gpu_id}]   - 进程启动失败")
                elif e.returncode == -9 or e.returncode == 137:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 进程被信号9（SIGKILL）杀死，通常是OOM killer")
                elif e.returncode == -11 or e.returncode == 139:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 进程段错误（SIGSEGV），可能是内存访问错误")
            
            # 如果是CUDA错误且还有重试机会，继续重试
            if is_cuda_error and attempt < max_retries - 1:
                continue
            else:
                # 重试次数用完或不是可重试的错误，返回失败
                return False, None, f"采样失败 (返回码={e.returncode}): {error_msg[:1000]}"
        except Exception as e:
            error_msg = f"采样出错: {str(e)}"
            last_error = error_msg
            print(f"[GPU {gpu_id}] ❌ {error_msg} (data_id={data_id})")
            traceback.print_exc()
            # 如果是最后一次尝试，返回失败
            if attempt >= max_retries - 1:
                return False, None, error_msg
            # 否则继续重试
            continue
    
    # 所有重试都失败
    return False, None, f"采样失败（已重试{max_retries}次）: {last_error[:1000] if last_error else '未知错误'}"


def run_single_evaluation(pt_file, protein_root, data_id, atom_mode='add_aromatic', exhaustiveness=8):
    """
    执行单个评估任务
    
    Args:
        pt_file: .pt文件路径
        protein_root: 蛋白质数据根目录
        data_id: 数据ID
        atom_mode: 原子模式
        exhaustiveness: Vina对接强度
    
    Returns:
        tuple: (success, message, eval_output_dir)
    """
    print(f"[评估] 开始评估: {Path(pt_file).name} (data_id={data_id})")
    
    pt_path = Path(pt_file)
    outputs_dir = pt_path.parent
    
    # 使用本地时区（CST，UTC+8）生成时间戳
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
    eval_output_dir = outputs_dir / f'eval_{data_id}_{eval_timestamp}'
    
    # 构建评估命令
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(pt_file),
        '--protein_root', str(protein_root),
        '--output_dir', str(eval_output_dir),
        '--atom_mode', atom_mode,
        '--exhaustiveness', str(exhaustiveness)
    ]
    
    try:
        # 执行评估
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,  # 捕获输出以避免混乱
            text=True,
            timeout=7200  # 设置超时时间（2小时，评估可能更慢）
        )
        
        print(f"[评估] ✅ 评估成功 (data_id={data_id})")
        return True, "评估成功", str(eval_output_dir)
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"评估超时 (超过2小时)"
        if e.stdout:
            error_msg += f"\nstdout: {e.stdout[-500:]}"
        if e.stderr:
            error_msg += f"\nstderr: {e.stderr[-500:]}"
        print(f"[评估] ❌ 评估超时 (data_id={data_id})")
        print(f"[评估] {error_msg}")
        return False, error_msg, None
    except subprocess.CalledProcessError as e:
        # 组合stdout和stderr获取完整错误信息
        error_parts = []
        if e.stdout:
            error_parts.append(f"stdout: {e.stdout}")
        if e.stderr:
            error_parts.append(f"stderr: {e.stderr}")
        if not error_parts:
            error_parts.append(str(e))
        
        error_msg = "\n".join(error_parts)
        # 显示完整错误信息（不截断）
        print(f"[评估] ❌ 评估失败 (data_id={data_id}):")
        print(f"[评估] {error_msg}")
        # 返回时保留完整错误信息，但限制长度避免过长
        return False, f"评估失败: {error_msg[:1000]}", None
    except Exception as e:
        error_msg = f"评估出错: {str(e)}"
        print(f"[评估] ❌ {error_msg} (data_id={data_id})")
        traceback.print_exc()
        return False, error_msg, None


def check_gpu_memory_available(gpu_id, min_free_memory_mb=5000):
    """
    检查GPU是否有足够的可用内存
    
    Args:
        gpu_id: GPU ID
        min_free_memory_mb: 最小可用内存（MB），默认5GB
    
    Returns:
        tuple: (has_enough_memory, free_memory_mb, total_memory_mb, message)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', 
             '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                free_memory_mb = int(parts[0].strip())
                total_memory_mb = int(parts[1].strip())
                has_enough = free_memory_mb >= min_free_memory_mb
                message = f"GPU {gpu_id}: 可用内存 {free_memory_mb}MB / 总计 {total_memory_mb}MB"
                return has_enough, free_memory_mb, total_memory_mb, message
    except Exception as e:
        pass
    
    return None, None, None, f"无法检查GPU {gpu_id}的内存状态"


def check_cuda_available_in_subprocess():
    """
    在子进程中检查 CUDA 是否可用
    在设置 CUDA_VISIBLE_DEVICES 后调用此函数
    
    Returns:
        tuple: (is_available, num_gpus)
    """
    # 方法1：使用 nvidia-smi 检测 GPU（不依赖 PyTorch，更可靠）
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi 可用，说明 GPU 驱动正常
            num_gpus = len(result.stdout.strip().split('\n'))
            if num_gpus > 0:
                return True, num_gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 方法2：检查 PyTorch CUDA（在设置 CUDA_VISIBLE_DEVICES 后应该能检测到）
    try:
        # 如果 torch 已经在主进程中导入，子进程会继承
        # 但 CUDA 的检测应该会使用新的 CUDA_VISIBLE_DEVICES 环境变量
        if torch is not None:
            # 尝试重新初始化 CUDA（如果可能）
            try:
                # 清除 CUDA 缓存（如果已初始化）
                if hasattr(torch.cuda, '_lazy_init'):
                    torch.cuda._lazy_init()
            except Exception:
                pass
            
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                return True, num_gpus
    except Exception as e:
        # 如果检测失败，记录但不抛出异常
        pass
    
    return False, 0


def process_sampling_task(args_tuple):
    """
    处理单个采样任务（仅采样，不评估）
    这个函数会在独立的进程中运行，限制为4个进程（每个GPU一个）
    
    Args:
        args_tuple: (data_id, gpu_id, config_file, skip_existing)
    
    Returns:
        tuple: (data_id, success, pt_file, message)
    """
    (data_id, gpu_id, config_file, skip_existing) = args_tuple
    
    # 在进程开始时立即设置CUDA_VISIBLE_DEVICES
    # 这必须在任何CUDA初始化之前完成
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 关键修复：如果torch已经在主进程中导入，需要清除CUDA缓存并重新初始化
    # 这是因为子进程可能继承了主进程的CUDA上下文状态
    if torch is not None:
        try:
            # 清除可能存在的CUDA缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # 尝试重置CUDA状态（如果可能）
            if hasattr(torch.cuda, '_lazy_init'):
                # 清除lazy init状态，强制重新初始化
                torch.cuda._lazy_init()
        except Exception as e:
            # 如果清除失败，继续执行（可能CUDA还未初始化）
            pass
    
    try:
        # 检查GPU内存是否足够（在设置CUDA_VISIBLE_DEVICES之前检查物理GPU）
        has_memory, free_mb, total_mb, mem_msg = check_gpu_memory_available(gpu_id, min_free_memory_mb=5000)
        if has_memory is False:
            error_msg = (
                f"GPU {gpu_id} 内存不足 (data_id={data_id})\n"
                f"   {mem_msg}\n"
                f"   需要至少 5000MB 可用内存，但只有 {free_mb}MB\n"
                f"   建议: 等待其他任务完成或清理GPU内存"
            )
            print(f"[GPU {gpu_id}] ⚠️  {error_msg}")
            # 内存不足时返回失败，但不抛出异常，让重试机制处理
            return (data_id, False, None, error_msg)
        elif has_memory is True:
            print(f"[GPU {gpu_id}] ✅ {mem_msg}")
        
        # 验证 CUDA 是否可用（在设置 CUDA_VISIBLE_DEVICES 后）
        cuda_available, num_gpus = check_cuda_available_in_subprocess()
        if not cuda_available:
            error_msg = (
                f"CUDA不可用 (GPU {gpu_id}, CUDA_VISIBLE_DEVICES={gpu_id})\n"
                f"   可能原因:\n"
                f"   1. Docker容器未正确配置GPU支持\n"
                f"   2. NVIDIA驱动未安装或版本不兼容\n"
                f"   3. PyTorch未正确编译CUDA支持\n"
                f"   4. CUDA运行时库未正确安装\n"
                f"   请检查: nvidia-smi 是否可用，以及容器是否正确配置了 --gpus all"
            )
            print(f"[GPU {gpu_id}] ❌ {error_msg}")
            return (data_id, False, None, error_msg)
        
        # 检查是否已存在
        if skip_existing:
            pt_file = find_latest_result_file(data_id)
            if pt_file and pt_file.exists():
                print(f"[GPU {gpu_id}] ⏭️  跳过已存在的文件: {pt_file} (data_id={data_id})")
                return (data_id, True, str(pt_file), "文件已存在")
        
        # 执行采样（带重试机制）
        sample_success, pt_file, sample_msg = run_single_sample(data_id, config_file, gpu_id, max_retries=3, retry_delay=5)
        
        if not sample_success or pt_file is None:
            return (data_id, False, None, sample_msg)
        
        return (data_id, True, pt_file, sample_msg)
            
    except Exception as e:
        error_msg = f"采样任务异常 (data_id={data_id}): {str(e)}"
        print(f"[GPU {gpu_id}] ❌ {error_msg}")
        traceback.print_exc()
        return (data_id, False, None, error_msg)


def process_evaluation_task(args_tuple):
    """
    处理单个评估任务（仅评估，不采样）
    这个函数会在独立的进程中运行，可以使用64个进程并行
    
    Args:
        args_tuple: (data_id, pt_file, protein_root, atom_mode, exhaustiveness, excel_file, batch_start_time, excel_lock)
    
    Returns:
        tuple: (data_id, success, message, log_file, pt_file, eval_output_dir)
    """
    (data_id, pt_file, protein_root, atom_mode, exhaustiveness, 
     excel_file, batch_start_time, excel_lock) = args_tuple
    
    # 设置全局锁（用于Excel写入）
    global excel_write_lock
    excel_write_lock = excel_lock
    
    task_start_time = time.time()
    
    try:
        if not pt_file:
            return (data_id, False, "无PT文件", None, None, None)
        
        # 执行评估
        eval_success, eval_msg, eval_output_dir = run_single_evaluation(
            pt_file, protein_root, data_id, atom_mode, exhaustiveness
        )
        
        task_time = time.time() - task_start_time
        # 使用本地时区（CST，UTC+8）生成时间戳
        cst = timezone(timedelta(hours=8))
        timestamp_str = datetime.now(cst).strftime('%Y-%m-%d %H:%M:%S')
        
        if eval_success:
            # 等待评估结果文件生成
            time.sleep(2)
            eval_success_read, vina_mean, vina_median, num_scores, eval_message, _ = read_evaluation_results(
                pt_file, data_id, wait_timeout=60
            )
            
            if eval_success_read:
                if excel_file:
                    append_to_excel(
                        excel_file, timestamp_str, task_time, data_id, pt_file,
                        vina_mean, vina_median, num_scores, '成功', eval_message
                    )
            else:
                if excel_file:
                    append_to_excel(
                        excel_file, timestamp_str, task_time, data_id, pt_file,
                        None, None, 0, '部分成功', f"评估完成但读取结果失败: {eval_message}"
                    )
            
            return (data_id, True, eval_msg, None, pt_file, str(eval_output_dir) if eval_output_dir else None)
        else:
            if excel_file:
                append_to_excel(
                    excel_file, timestamp_str, task_time, data_id, pt_file,
                    None, None, 0, '失败', eval_msg
                )
            
            return (data_id, False, eval_msg, None, pt_file, None)
            
    except Exception as e:
        error_msg = f"评估任务异常 (data_id={data_id}): {str(e)}"
        print(f"[评估] ❌ {error_msg}")
        traceback.print_exc()
        return (data_id, False, error_msg, None, pt_file, None)


def append_to_excel(excel_file, timestamp, execution_time, data_id, pt_file, vina_mean, vina_median, 
                    num_scores, status, message):
    """
    将评估结果追加到Excel文件（进程安全版本）
    使用临时文件+重命名机制避免并发写入损坏
    """
    if pd is None:
        return False
    
    # 使用锁确保进程安全
    with excel_write_lock:
        try:
            if not isinstance(excel_file, Path):
                excel_file = Path(excel_file)
            
            new_row = {
                '执行时间': timestamp,
                '执行耗时(秒)': execution_time,
                '数据ID': data_id,
                'PT文件': os.path.basename(str(pt_file)) if pt_file else '',
                'Vina平均得分': vina_mean if vina_mean is not None else '',
                'Vina中位数得分': vina_median if vina_median is not None else '',
                '得分数量': num_scores if num_scores else 0,
                '状态': status,
                '备注': message
            }
            
            # 定义必需的列
            required_columns = ['执行时间', '执行耗时(秒)', '数据ID', 'PT文件', 
                              'Vina平均得分', 'Vina中位数得分', '得分数量', '状态', '备注', '累计均值']
            
            df = pd.DataFrame()
            cumulative_mean = vina_mean
            
            if excel_file.exists():
                try:
                    # 尝试读取Excel文件
                    df = pd.read_excel(excel_file, engine='openpyxl', sheet_name='评估记录')
                    
                    # 检查是否有必需的列，如果没有或列不匹配，创建新的DataFrame
                    if df.empty or not all(col in df.columns for col in required_columns[:-1]):  # 累计均值可能不存在
                        print(f'⚠️  警告: Excel文件列不匹配，创建新的DataFrame.')
                        df = pd.DataFrame()
                    else:
                        # 如果缺少累计均值列，添加它
                        if '累计均值' not in df.columns:
                            df['累计均值'] = ''
                        
                        # 计算累计均值
                        if '状态' in df.columns:
                            successful_rows = df[df['状态'] == '成功']
                            if len(successful_rows) > 0:
                                all_means = successful_rows['Vina平均得分'].dropna().tolist()
                                if vina_mean is not None:
                                    all_means.append(vina_mean)
                                cumulative_mean = np.mean(all_means) if all_means else None
                except Exception as e:
                    # 文件损坏或读取失败，创建新的DataFrame
                    print(f'⚠️  警告: 读取Excel文件失败 {excel_file}: {e}. 创建新的DataFrame.')
                    df = pd.DataFrame()
            
            new_row['累计均值'] = cumulative_mean if cumulative_mean is not None else ''
            
            # 确保所有列都存在
            for col in required_columns:
                if col not in new_row:
                    new_row[col] = ''
            
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # 使用临时文件写入，然后重命名，避免并发写入损坏
            excel_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = excel_file.with_suffix('.tmp.xlsx')
            
            try:
                with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='评估记录')
                    
                    if len(df) > 0 and '状态' in df.columns:
                        successful_df = df[df['状态'] == '成功']
                        if len(successful_df) > 0:
                            stats = {
                                '统计项目': [
                                    '总评估次数',
                                    '成功次数',
                                    '失败次数',
                                    '当前累计均值',
                                    '当前累计中位数',
                                    '最佳得分',
                                    '最差得分'
                                ],
                                '数值': [
                                    len(df),
                                    len(successful_df),
                                    len(df) - len(successful_df),
                                    successful_df['Vina平均得分'].mean() if len(successful_df) > 0 else '',
                                    successful_df['Vina平均得分'].median() if len(successful_df) > 0 else '',
                                    successful_df['Vina平均得分'].min() if len(successful_df) > 0 else '',
                                    successful_df['Vina平均得分'].max() if len(successful_df) > 0 else ''
                                ]
                            }
                            stats_df = pd.DataFrame(stats)
                            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
                
                # 原子性重命名（在Unix系统上）
                temp_file.replace(excel_file)
                
            except Exception as e:
                # 如果写入失败，清理临时文件
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                raise e
            
            return True
                
        except Exception as e:
            print(f"⚠️  写入Excel失败 (data_id={data_id}): {e}")
            # 不打印完整traceback，避免输出过多
            return False


def collect_all_evaluation_results(results, batch_start_time):
    """
    从所有评估结果Excel文件中收集对接成功的分子数据
    扫描outputs目录下所有eval_*目录，读取其中的evaluation_results_*.xlsx文件并合并
    
    Returns:
        tuple: (molecule_records, summary_stats, pocket_stats_list)
            - molecule_records: 所有分子的记录列表
            - summary_stats: 汇总统计信息
            - pocket_stats_list: 每个口袋的统计信息列表（用于加权均值计算）
    """
    if pd is None:
        print('⚠️  pandas未安装，无法读取Excel文件')
        return [], {}, []
    
    molecule_records = []
    total_num_samples = 0
    total_n_reconstruct_success = 0
    total_n_eval_success = 0
    pocket_stats_list = []  # 存储每个口袋的统计信息
    
    batch_start_datetime = datetime.fromtimestamp(batch_start_time)
    
    # 扫描outputs目录下所有eval_*目录
    eval_dirs = list(OUTPUT_DIR.glob('eval_*'))
    
    if not eval_dirs:
        print(f"⚠️  未找到任何评估目录（在 {OUTPUT_DIR} 中查找 eval_*）")
        return [], {}, []
    
    print(f"找到 {len(eval_dirs)} 个评估目录，开始读取Excel文件...")
    
    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue
        
        # 查找该目录下的所有evaluation_results_*.xlsx文件
        excel_files = list(eval_dir.glob('evaluation_results_*.xlsx'))
        
        if not excel_files:
            continue
        
        # 只处理在batch_start_time之后生成的文件
        recent_excel_files = [
            f for f in excel_files
            if datetime.fromtimestamp(f.stat().st_mtime) >= batch_start_datetime
        ]
        
        if not recent_excel_files:
            continue
        
        # 选择最新的Excel文件（如果有多个）
        latest_excel_file = max(recent_excel_files, key=os.path.getmtime)
        
        try:
            # 读取Excel文件的"评估结果"工作表
            df = pd.read_excel(latest_excel_file, sheet_name='评估结果', engine='openpyxl')
            
            if df.empty:
                continue
            
            # 将DataFrame转换为记录列表
            for _, row in df.iterrows():
                record = row.to_dict()
                
                # 处理NaN值，将NaN转换为None（后续处理会统一处理None和'N/A'）
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                
                # 确保数据ID字段存在（从目录名或文件名中提取）
                data_id = record.get('数据ID')
                if data_id is None or (isinstance(data_id, float) and pd.isna(data_id)):
                    # 尝试从目录名中提取data_id
                    dir_name = eval_dir.name
                    if dir_name.startswith('eval_'):
                        parts = dir_name.split('_')
                        if len(parts) >= 2:
                            try:
                                record['数据ID'] = int(parts[1])
                            except ValueError:
                                record['数据ID'] = parts[1]
                
                molecule_records.append(record)
            
            # 获取数据ID
            data_id = None
            dir_name = eval_dir.name
            if dir_name.startswith('eval_'):
                parts = dir_name.split('_')
                if len(parts) >= 2:
                    try:
                        data_id = int(parts[1])
                    except ValueError:
                        data_id = parts[1]
            
            # 读取统计信息工作表以获取统计数据
            pocket_stats = {'数据ID': data_id, '评估成功数': len(df)}
            try:
                stats_df = pd.read_excel(latest_excel_file, sheet_name='统计信息', engine='openpyxl')
                if not stats_df.empty and '统计项目' in stats_df.columns and '数值' in stats_df.columns:
                    stats_dict = dict(zip(stats_df['统计项目'], stats_df['数值']))
                    
                    # 提取统计数据
                    if '总样本数' in stats_dict:
                        try:
                            total_num_samples += int(stats_dict['总样本数'])
                        except (ValueError, TypeError):
                            pass
                    
                    if '重建成功' in stats_dict:
                        try:
                            total_n_reconstruct_success += int(stats_dict['重建成功'])
                        except (ValueError, TypeError):
                            pass
                    
                    if '评估成功' in stats_dict:
                        try:
                            eval_success = int(stats_dict['评估成功'])
                            total_n_eval_success += eval_success
                            pocket_stats['评估成功数'] = eval_success
                        except (ValueError, TypeError):
                            pass
                    
                    # 提取各个参数的均值（用于加权平均计算）
                    # 查找所有包含"平均"的统计项目
                    for stat_name, stat_value in stats_dict.items():
                        if '平均' in stat_name and stat_value not in ('N/A', None, ''):
                            try:
                                # 尝试转换为数值
                                if isinstance(stat_value, str):
                                    # 尝试转换为浮点数
                                    stat_value = float(stat_value)
                                pocket_stats[stat_name] = stat_value
                            except (ValueError, TypeError):
                                pass
                    
                    # 也查找一些常见的统计指标（即使没有"平均"字样）
                    for stat_name in ['Vina_Dock_最佳亲和力', 'Vina_Dock_最差亲和力', 
                                     'Vina_ScoreOnly_最佳亲和力', 'Vina_Minimize_最佳亲和力']:
                        if stat_name in stats_dict:
                            try:
                                val = stats_dict[stat_name]
                                if val not in ('N/A', None, '') and not pd.isna(val):
                                    if isinstance(val, str):
                                        val = float(val)
                                    pocket_stats[stat_name] = val
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                # 如果读取统计信息失败，不影响主流程
                pass
            
            # 如果没有从统计信息中获取到评估成功数，使用分子记录数
            if pocket_stats['评估成功数'] == 0:
                pocket_stats['评估成功数'] = len(df)
            
            # 添加到口袋统计列表
            if pocket_stats['评估成功数'] > 0:
                pocket_stats_list.append(pocket_stats)
            
            print(f"  ✅ 已读取: {latest_excel_file.name} ({len(df)} 条记录)")
            
        except Exception as e:
            print(f"⚠️  读取Excel文件失败 {latest_excel_file}: {e}")
            continue
    
    # 计算汇总统计信息
    # 计算百分比
    reconstruct_success_rate = (total_n_reconstruct_success / total_num_samples * 100) if total_num_samples > 0 else 0.0
    docking_success_rate = (len(molecule_records) / total_num_samples * 100) if total_num_samples > 0 else 0.0
    
    # 计算有效分子数量（剔除vinadock>0、vinascore>0或vinamin>0的异常数据）
    # 注意：不删除任何能对接的分子，只是统计有效分子数量
    valid_molecule_count = 0
    for r in molecule_records:
        vina_dock = r.get('Vina_Dock_亲和力', 'N/A')
        vina_score = r.get('Vina_ScoreOnly_亲和力', 'N/A')
        vina_min = r.get('Vina_Minimize_亲和力', 'N/A')
        
        # 检查是否为异常数据（vinadock>0、vinascore>0或vinamin>0）
        is_abnormal = False
        try:
            if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                if float(vina_dock) > 0:
                    is_abnormal = True
            if vina_score not in ('N/A', None) and not pd.isna(vina_score):
                if float(vina_score) > 0:
                    is_abnormal = True
            if vina_min not in ('N/A', None) and not pd.isna(vina_min):
                if float(vina_min) > 0:
                    is_abnormal = True
        except (ValueError, TypeError):
            pass
        
        # 如果不是异常数据，则计入有效分子
        if not is_abnormal:
            valid_molecule_count += 1
    
    # 计算有效分子比例（有效分子数量 / 应生成分子数）
    valid_molecule_ratio = (valid_molecule_count / total_num_samples * 100) if total_num_samples > 0 else 0.0
    
    summary_stats = {
        'batch启动时间': datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        '应生成分子数': total_num_samples,
        '可重建分子数': total_n_reconstruct_success,
        '重建成功百分比(%)': f"{reconstruct_success_rate:.2f}",
        '对接成功分子数': len(molecule_records),
        '对接成功百分比(%)': f"{docking_success_rate:.2f}",
        '有效分子数量': valid_molecule_count,
        '有效分子比例(%)': f"{valid_molecule_ratio:.2f}",
    }
    
    if molecule_records:
        # 计算平均得分
        vina_dock_scores = []
        vina_score_only_scores = []
        vina_minimize_scores = []
        qed_values = []
        sa_values = []
        
        for r in molecule_records:
            # 处理Vina_Dock_亲和力
            vina_dock = r.get('Vina_Dock_亲和力', 'N/A')
            if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                try:
                    vina_dock_scores.append(float(vina_dock))
                except (ValueError, TypeError):
                    pass
            
            # 处理Vina_ScoreOnly_亲和力
            vina_score_only = r.get('Vina_ScoreOnly_亲和力', 'N/A')
            if vina_score_only not in ('N/A', None) and not pd.isna(vina_score_only):
                try:
                    vina_score_only_scores.append(float(vina_score_only))
                except (ValueError, TypeError):
                    pass
            
            # 处理Vina_Minimize_亲和力
            vina_minimize = r.get('Vina_Minimize_亲和力', 'N/A')
            if vina_minimize not in ('N/A', None) and not pd.isna(vina_minimize):
                try:
                    vina_minimize_scores.append(float(vina_minimize))
                except (ValueError, TypeError):
                    pass
            
            # 处理QED评分
            qed = r.get('QED评分', 'N/A')
            if qed not in ('N/A', None) and not pd.isna(qed):
                try:
                    qed_values.append(float(qed))
                except (ValueError, TypeError):
                    pass
            
            # 处理SA评分
            sa = r.get('SA评分', 'N/A')
            if sa not in ('N/A', None) and not pd.isna(sa):
                try:
                    sa_values.append(float(sa))
                except (ValueError, TypeError):
                    pass
        
        if vina_dock_scores:
            summary_stats['Vina_Dock_平均亲和力'] = np.mean(vina_dock_scores)
        if vina_score_only_scores:
            summary_stats['Vina_ScoreOnly_平均亲和力'] = np.mean(vina_score_only_scores)
        if vina_minimize_scores:
            summary_stats['Vina_Minimize_平均亲和力'] = np.mean(vina_minimize_scores)
        if qed_values:
            summary_stats['QED平均评分'] = np.mean(qed_values)
        if sa_values:
            summary_stats['SA平均评分'] = np.mean(sa_values)
    
    return molecule_records, summary_stats, pocket_stats_list


def flatten_config(config, parent_key='', sep='.'):
    """
    将嵌套的配置字典扁平化为键值对列表
    
    Args:
        config: 配置字典
        parent_key: 父键（用于构建完整的键路径）
        sep: 分隔符
    
    Returns:
        list: [(键路径, 值), ...] 的列表
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if value is None:
            # None值转换为字符串
            items.append((new_key, 'null'))
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            items.extend(flatten_config(value, new_key, sep=sep))
        elif isinstance(value, list):
            # 列表转换为字符串
            items.append((new_key, str(value)))
        else:
            # 普通值
            items.append((new_key, value))
    
    return items


def build_lambda_schedule(start_t, end_t, coeff_a, coeff_b, num_timesteps):
    """构建 lambda 调度序列（从show_sampling_steps.py复制）"""
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
    """构建线性调度序列（从show_sampling_steps.py复制）"""
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
    """构建固定步长调度序列（从show_sampling_steps.py复制）"""
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


def calculate_total_sampling_steps(config):
    """
    计算总采样步数（跳步总次数）和实际长度
    
    Args:
        config: 配置字典
    
    Returns:
        tuple: (总采样步数, 实际长度)，如果计算失败则返回(None, None)
    """
    try:
        sample_cfg = config.get('sample', {})
        if not isinstance(sample_cfg, dict):
            return None, None
        
        num_timesteps = sample_cfg.get('num_steps', 1000)
        dynamic_cfg = sample_cfg.get('dynamic', {})
        if not isinstance(dynamic_cfg, dict):
            return None, None
        
        large_step_cfg = dynamic_cfg.get('large_step', {})
        if not isinstance(large_step_cfg, dict):
            large_step_cfg = {}
        
        refine_cfg = dynamic_cfg.get('refine', {})
        if not isinstance(refine_cfg, dict):
            refine_cfg = {}
        
        # 获取 time_boundary
        time_boundary = dynamic_cfg.get('time_boundary', 600)
        
        # Large Step 阶段
        large_step_schedule = large_step_cfg.get('schedule', 'lambda')
        large_step_time_lower = time_boundary
        large_step_time_upper = num_timesteps - 1
        large_step_size = large_step_cfg.get('step_size', 1.0)  # 获取step_size
        
        # Refine 阶段
        refine_schedule = refine_cfg.get('schedule', 'lambda')
        refine_time_upper = time_boundary
        refine_time_lower = refine_cfg.get('time_lower', 0)
        refine_step_size = refine_cfg.get('step_size', 0.2)  # 获取step_size
        
        # 计算 Large Step 阶段时间步
        if large_step_schedule == 'lambda':
            lambda_coeff_a = large_step_cfg.get('lambda_coeff_a', 80.0)
            lambda_coeff_b = large_step_cfg.get('lambda_coeff_b', 20.0)
            large_step_indices, _ = build_lambda_schedule(
                large_step_time_upper, large_step_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif large_step_schedule == 'linear':
            step_upper = large_step_cfg.get('linear_step_upper', 100.0)
            step_lower = large_step_cfg.get('linear_step_lower', 20.0)
            large_step_indices, _ = build_linear_schedule(
                large_step_time_upper, large_step_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = large_step_cfg.get('stride', 15)
            large_step_indices, _ = build_fixed_schedule(
                large_step_time_upper, large_step_time_lower,
                stride, num_timesteps
            )
        
        # 计算 Refine 阶段时间步
        if refine_schedule == 'lambda':
            lambda_coeff_a = refine_cfg.get('lambda_coeff_a', 40.0)
            lambda_coeff_b = refine_cfg.get('lambda_coeff_b', 5.0)
            refine_indices, _ = build_lambda_schedule(
                refine_time_upper, refine_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif refine_schedule == 'linear':
            step_upper = refine_cfg.get('linear_step_upper', 40.0)
            step_lower = refine_cfg.get('linear_step_lower', 5.0)
            refine_indices, _ = build_linear_schedule(
                refine_time_upper, refine_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = refine_cfg.get('stride', 8)
            refine_indices, _ = build_fixed_schedule(
                refine_time_upper, refine_time_lower,
                stride, num_timesteps
            )
        
        # 计算实际长度：ls的步数 × ls的step_size + rf的步数 × rf的step_size
        # 步数是指每个阶段的采样点数量（即indices列表的长度）
        large_step_num_steps = len(large_step_indices) if large_step_indices else 0
        refine_num_steps = len(refine_indices) if refine_indices else 0
        
        # 计算实际长度
        actual_length = large_step_num_steps * large_step_size + refine_num_steps * refine_step_size
        
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
        
    except Exception as e:
        print(f"⚠️  计算总采样步数和实际长度失败: {e}")
        return None, None


def load_config_to_dataframe(config_file):
    """
    将配置文件加载并转换为DataFrame
    
    Args:
        config_file: 配置文件路径
    
    Returns:
        pd.DataFrame: 包含参数路径和值的DataFrame，如果失败则返回None
    """
    if yaml is None:
        return None
    
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return None
        
        # 扁平化配置
        flat_config = flatten_config(config)
        
        # 计算总采样步数和实际长度
        total_steps, actual_length = calculate_total_sampling_steps(config)
        if total_steps is not None:
            flat_config.append(('计算.跳步总次数', total_steps))
        if actual_length is not None:
            flat_config.append(('计算.实际长度', actual_length))
        
        # 转换为DataFrame
        df = pd.DataFrame(flat_config, columns=['参数路径', '参数值'])
        
        # 将值转换为字符串（便于在Excel中显示）
        df['参数值'] = df['参数值'].astype(str)
        
        return df
        
    except Exception as e:
        print(f"⚠️  读取配置文件失败 {config_file}: {e}")
        return None


def parse_filename_params(filename):
    """
    从文件名解析参数
    
    示例文件名: batch_evaluation_summary_20260105_002424_gfquadratic_1_0_tl800_lslambda_60p0_20p0_lsstep_0p6_lsnoise_0p0_rflambda_10p0_5p0_rfstep_0p25_rfnoise_0p05.xlsx
    
    返回参数字典
    """
    params = {}
    filename_str = str(filename) if isinstance(filename, Path) else filename
    
    # 提取权重策略 (gfquadratic -> quadratic)
    gf_match = re.search(r'gf(\w+)', filename_str)
    if gf_match:
        params['权重策略'] = gf_match.group(1)
    
    # 提取开始权重和结束权重 (gfquadratic_1_0 -> 开始权重=1, 结束权重=0)
    weight_match = re.search(r'gf\w+_(\d+)_(\d+)', filename_str)
    if weight_match:
        params['开始权重'] = float(weight_match.group(1))
        params['结束权重'] = float(weight_match.group(2))
    
    # 提取时间长度 (tl800 -> 800)
    tl_match = re.search(r'tl(\d+)', filename_str)
    if tl_match:
        params['时间长度 (TL)'] = int(tl_match.group(1))
    
    # 提取LS Lambda值 (lslambda_60p0_20p0 -> LSLambda1=60.0, LSLambda2=20.0)
    ls_lambda_match = re.search(r'lslambda_(\d+p\d+)_(\d+p\d+)', filename_str)
    if ls_lambda_match:
        params['LSLambda1'] = float(ls_lambda_match.group(1).replace('p', '.'))
        params['LSLambda2'] = float(ls_lambda_match.group(2).replace('p', '.'))
    
    # 提取LS step size (lsstep_0p6 -> 0.6)
    ls_step_match = re.search(r'lsstep_(\d+p\d+)', filename_str)
    if ls_step_match:
        params['LSstepsize'] = float(ls_step_match.group(1).replace('p', '.'))
    
    # 提取LS noise (lsnoise_0p0 -> 0.0)
    ls_noise_match = re.search(r'lsnoise_(\d+p\d+)', filename_str)
    if ls_noise_match:
        params['LSnosie'] = float(ls_noise_match.group(1).replace('p', '.'))
    
    # 提取RF Lambda值 (rflambda_10p0_5p0 -> RFLambda1=10.0, RFLambda2=5.0)
    rf_lambda_match = re.search(r'rflambda_(\d+p\d+)_(\d+p\d+)', filename_str)
    if rf_lambda_match:
        params['RFLambda1'] = float(rf_lambda_match.group(1).replace('p', '.'))
        params['RFLambda2'] = float(rf_lambda_match.group(2).replace('p', '.'))
    
    # 提取RF step size (rfstep_0p25 -> 0.25)
    rf_step_match = re.search(r'rfstep_(\d+p\d+)', filename_str)
    if rf_step_match:
        params['RFstepsize'] = float(rf_step_match.group(1).replace('p', '.'))
    
    # 提取RF noise (rfnoise_0p05 -> 0.05)
    rf_noise_match = re.search(r'rfnoise_(\d+p\d+)', filename_str)
    if rf_noise_match:
        params['RFnosie'] = float(rf_noise_match.group(1).replace('p', '.'))
    
    return params


def append_to_merged_summary(excel_file, summary_stats, config_file=None):
    """
    将当前批次的数据追加到汇总Excel文件中
    
    Args:
        excel_file: 刚保存的批次Excel文件路径
        summary_stats: 汇总统计信息
        config_file: 配置文件路径（用于提取配置参数）
    """
    if pd is None:
        return
    
    try:
        excel_file = Path(excel_file)
        if not excel_file.exists():
            return
        
        # 汇总Excel文件路径
        batchsummary_dir = excel_file.parent
        merged_summary_file = batchsummary_dir / 'merged_summary.xlsx'
        
        # 定义列的顺序
        columns_order = [
            '权重策略', '下降速率', '开始权重', '结束权重', '时间长度 (TL)',
            'LSstepsize', 'LSnosie', 'LSLambda1', 'LSLambda2',
            'RFstepsize', 'RFnosie', 'RFLambda1', 'RFLambda2',
            '步数', '取模步长', '可重建率 (%)', '对接成功率 (%)',
            '有效分子比例 (%)',
            'Vina_Dock 亲和力', 'Vina_ScoreOnly', 'Vina_Minimize',
            'QED 评分（均值）', 'SA 评分（均值）'
        ]
        
        # 从文件名解析参数
        filename_params = parse_filename_params(excel_file.name)
        
        # 从Excel文件提取统计数据
        try:
            df_stats = pd.read_excel(excel_file, sheet_name='统计信息', engine='openpyxl')
            stats_dict = {}
            for _, row in df_stats.iterrows():
                key = str(row['统计项目'])
                value = row['数值']
                # 尝试转换为数值类型
                if isinstance(value, str):
                    try:
                        # 尝试转换为float
                        value = float(value)
                    except ValueError:
                        pass
                stats_dict[key] = value
        except Exception:
            stats_dict = {}
        
        # 从配置参数sheet提取数据
        config_dict = {}
        try:
            df_config = pd.read_excel(excel_file, sheet_name='配置参数', engine='openpyxl')
            for _, row in df_config.iterrows():
                key = str(row['参数路径'])
                value = row['参数值']
                # 处理NaN值
                if pd.isna(value):
                    continue
                config_dict[key] = value
        except Exception:
            # 如果配置参数sheet不存在或读取失败，尝试从config_file读取
            if config_file:
                try:
                    config_df = load_config_to_dataframe(config_file)
                    if config_df is not None and len(config_df) > 0:
                        for _, row in config_df.iterrows():
                            key = str(row['参数路径'])
                            value = row['参数值']
                            if pd.isna(value):
                                continue
                            config_dict[key] = value
                except Exception:
                    pass
        
        # 构建新行数据
        row_data = {}
        
        # 从文件名参数获取（优先）
        row_data.update(filename_params)
        
        # 从配置参数中获取（如果文件名中没有）
        if '下降速率' not in row_data:
            power = config_dict.get('model.grad_fusion_lambda.power', None)
            if power is not None:
                row_data['下降速率'] = float(power)
        
        if '步数' not in row_data:
            steps = config_dict.get('计算.跳步总次数', None)
            if steps is not None:
                row_data['步数'] = int(steps)
        
        if '取模步长' not in row_data:
            mod_step = config_dict.get('计算.实际长度', None)
            if mod_step is not None:
                row_data['取模步长'] = float(mod_step)
        
        # 从配置参数补充缺失的参数
        if 'LSstepsize' not in row_data:
            ls_step = config_dict.get('sample.dynamic.large_step.step_size', None)
            if ls_step is not None:
                row_data['LSstepsize'] = float(ls_step)
        
        if 'LSnosie' not in row_data:
            ls_noise = config_dict.get('sample.dynamic.large_step.noise_scale', None)
            if ls_noise is not None:
                row_data['LSnosie'] = float(ls_noise)
        
        if 'LSLambda1' not in row_data:
            ls_lambda_a = config_dict.get('sample.dynamic.large_step.lambda_coeff_a', None)
            if ls_lambda_a is not None:
                row_data['LSLambda1'] = float(ls_lambda_a)
        
        if 'LSLambda2' not in row_data:
            ls_lambda_b = config_dict.get('sample.dynamic.large_step.lambda_coeff_b', None)
            if ls_lambda_b is not None:
                row_data['LSLambda2'] = float(ls_lambda_b)
        
        if 'RFstepsize' not in row_data:
            rf_step = config_dict.get('sample.dynamic.refine.step_size', None)
            if rf_step is not None:
                row_data['RFstepsize'] = float(rf_step)
        
        if 'RFnosie' not in row_data:
            rf_noise = config_dict.get('sample.dynamic.refine.noise_scale', None)
            if rf_noise is not None:
                row_data['RFnosie'] = float(rf_noise)
        
        if 'RFLambda1' not in row_data:
            rf_lambda_a = config_dict.get('sample.dynamic.refine.lambda_coeff_a', None)
            if rf_lambda_a is not None:
                row_data['RFLambda1'] = float(rf_lambda_a)
        
        if 'RFLambda2' not in row_data:
            rf_lambda_b = config_dict.get('sample.dynamic.refine.lambda_coeff_b', None)
            if rf_lambda_b is not None:
                row_data['RFLambda2'] = float(rf_lambda_b)
        
        if '时间长度 (TL)' not in row_data:
            time_boundary = config_dict.get('sample.dynamic.time_boundary', None)
            if time_boundary is not None:
                row_data['时间长度 (TL)'] = int(time_boundary)
        
        if '权重策略' not in row_data:
            mode = config_dict.get('model.grad_fusion_lambda.mode', None)
            if mode is not None:
                row_data['权重策略'] = str(mode)
        
        if '开始权重' not in row_data:
            start = config_dict.get('model.grad_fusion_lambda.start', None)
            if start is not None:
                row_data['开始权重'] = float(start)
        
        if '结束权重' not in row_data:
            end = config_dict.get('model.grad_fusion_lambda.end', None)
            if end is not None:
                row_data['结束权重'] = float(end)
        
        # 从统计信息中提取（安全转换）
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    return float(value)
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        row_data['可重建率 (%)'] = safe_float(stats_dict.get('重建成功百分比(%)', 0))
        row_data['对接成功率 (%)'] = safe_float(stats_dict.get('对接成功百分比(%)', 0))
        row_data['有效分子比例 (%)'] = safe_float(stats_dict.get('有效分子比例(%)', 0))
        row_data['Vina_Dock 亲和力'] = safe_float(stats_dict.get('Vina_Dock_平均亲和力', 0))
        row_data['Vina_ScoreOnly'] = safe_float(stats_dict.get('Vina_ScoreOnly_平均亲和力', 0))
        row_data['Vina_Minimize'] = safe_float(stats_dict.get('Vina_Minimize_平均亲和力', 0))
        row_data['QED 评分（均值）'] = safe_float(stats_dict.get('QED平均评分', 0))
        row_data['SA 评分（均值）'] = safe_float(stats_dict.get('SA平均评分', 0))
        
        # 确保所有列都存在
        new_row = {}
        for col in columns_order:
            new_row[col] = row_data.get(col, None)
        # 添加文件名列
        new_row['文件名'] = excel_file.name
        
        # 读取现有的汇总文件或创建新的
        df_combined = None
        if merged_summary_file.exists():
            try:
                # 尝试读取"批次汇总"表，如果不存在则读取第一个sheet
                try:
                    df_existing = pd.read_excel(merged_summary_file, sheet_name='批次汇总', engine='openpyxl')
                except Exception:
                    df_existing = pd.read_excel(merged_summary_file, engine='openpyxl')
                
                # 检查是否已存在相同的记录（通过文件名判断）
                if '文件名' in df_existing.columns:
                    if excel_file.name in df_existing['文件名'].values:
                        # 更新现有记录而不是追加
                        idx = df_existing[df_existing['文件名'] == excel_file.name].index[0]
                        for col in columns_order + ['文件名']:
                            if col in df_existing.columns:
                                df_existing.at[idx, col] = new_row.get(col)
                        df_combined = df_existing
                    else:
                        # 追加新行
                        all_columns = columns_order + ['文件名']
                        df_new = pd.DataFrame([new_row], columns=all_columns)
                        # 如果现有文件缺少某些列，补齐
                        for col in all_columns:
                            if col not in df_existing.columns:
                                df_existing[col] = None
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    # 如果没有文件名列，直接追加
                    all_columns = columns_order + ['文件名']
                    df_new = pd.DataFrame([new_row], columns=all_columns)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as e:
                print(f"⚠️  读取现有汇总文件失败，创建新文件: {e}")
                df_combined = pd.DataFrame([new_row], columns=columns_order + ['文件名'])
        else:
            df_combined = pd.DataFrame([new_row], columns=columns_order + ['文件名'])
        
        # 保存汇总文件（使用ExcelWriter以支持多个sheet）
        with pd.ExcelWriter(merged_summary_file, engine='openpyxl') as writer:
            df_combined.to_excel(writer, sheet_name='批次汇总', index=False)
            
            # 合并所有批次Excel文件中的"正常分子"表
            try:
                all_valid_molecules = []
                batchsummary_dir = excel_file.parent
                
                # 查找所有批次Excel文件
                batch_files = list(batchsummary_dir.glob('batch_evaluation_summary_*.xlsx'))
                
                for batch_file in batch_files:
                    try:
                        # 尝试读取"正常分子"表
                        df_valid = pd.read_excel(batch_file, sheet_name='正常分子', engine='openpyxl')
                        if df_valid is not None and len(df_valid) > 0:
                            # 添加来源文件列
                            df_valid['来源文件'] = batch_file.name
                            all_valid_molecules.append(df_valid)
                    except Exception:
                        # 如果该文件没有"正常分子"表，跳过
                        continue
                
                # 合并所有正常分子
                if all_valid_molecules:
                    df_all_valid = pd.concat(all_valid_molecules, ignore_index=True)
                    # 按Vina_Dock_亲和力排序
                    if 'Vina_Dock_亲和力' in df_all_valid.columns:
                        df_all_valid['Vina_Dock_亲和力_temp'] = df_all_valid['Vina_Dock_亲和力'].replace('N/A', np.nan)
                        df_all_valid = df_all_valid.sort_values('Vina_Dock_亲和力_temp', na_position='last')
                        df_all_valid = df_all_valid.drop(columns=['Vina_Dock_亲和力_temp'])
                    df_all_valid.to_excel(writer, sheet_name='合并正常分子', index=False)
                    print(f"✅ 已合并 {len(df_all_valid)} 个正常分子到汇总文件")
                else:
                    # 创建空的正常分子表
                    pd.DataFrame().to_excel(writer, sheet_name='合并正常分子', index=False)
            except Exception as e:
                print(f"⚠️  合并正常分子失败: {e}")
                import traceback
                traceback.print_exc()
                # 即使合并失败，也要保存批次汇总
                df_combined.to_excel(writer, sheet_name='批次汇总', index=False)
        
        print(f"✅ 已追加到汇总Excel文件: {merged_summary_file.name}")
        
    except Exception as e:
        print(f"⚠️  追加到汇总Excel失败: {e}")
        import traceback
        traceback.print_exc()


def save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time, pocket_stats_list=None, config_file=None):
    """
    将所有对接成功的分子数据保存到Excel
    
    Args:
        excel_file: Excel文件路径
        molecule_records: 分子记录列表
        summary_stats: 汇总统计信息
        batch_start_time: 批次启动时间
        pocket_stats_list: 每个口袋的统计信息列表（用于加权均值计算）
        config_file: 配置文件路径（用于保存配置参数）
    """
    if pd is None:
        print(f'⚠️  pandas未安装，无法保存Excel')
        return False
    
    try:
        if not isinstance(excel_file, Path):
            excel_file = Path(excel_file)
        
        excel_file.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            if molecule_records:
                df_molecules = pd.DataFrame(molecule_records)
                if 'Vina_Dock_亲和力' in df_molecules.columns:
                    df_molecules['Vina_Dock_亲和力_temp'] = df_molecules['Vina_Dock_亲和力'].replace('N/A', np.nan)
                    df_molecules = df_molecules.sort_values('Vina_Dock_亲和力_temp', na_position='last')
                    df_molecules = df_molecules.drop(columns=['Vina_Dock_亲和力_temp'])
                df_molecules.to_excel(writer, sheet_name='分子评估数据', index=False)
                
                # 筛选正常分子（剔除vina三个参数大于0的异常分子）
                def is_valid_molecule(row):
                    """判断分子是否正常（vina三个参数都不大于0）"""
                    vina_dock = row.get('Vina_Dock_亲和力', 'N/A')
                    vina_score = row.get('Vina_ScoreOnly_亲和力', 'N/A')
                    vina_min = row.get('Vina_Minimize_亲和力', 'N/A')
                    
                    # 检查是否为异常数据（vinadock>0、vinascore>0或vinamin>0）
                    try:
                        if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                            if float(vina_dock) > 0:
                                return False
                        if vina_score not in ('N/A', None) and not pd.isna(vina_score):
                            if float(vina_score) > 0:
                                return False
                        if vina_min not in ('N/A', None) and not pd.isna(vina_min):
                            if float(vina_min) > 0:
                                return False
                    except (ValueError, TypeError):
                        # 如果无法转换，保留该分子（可能是N/A）
                        pass
                    
                    return True
                
                # 筛选正常分子
                df_valid = df_molecules[df_molecules.apply(is_valid_molecule, axis=1)].copy()
                df_valid.to_excel(writer, sheet_name='正常分子', index=False)
            else:
                df_molecules = pd.DataFrame()
                df_molecules.to_excel(writer, sheet_name='分子评估数据', index=False)
                # 创建空的正常分子表
                pd.DataFrame().to_excel(writer, sheet_name='正常分子', index=False)
            
            stats_items = []
            stats_values = []
            
            for key, value in summary_stats.items():
                stats_items.append(key)
                if isinstance(value, float):
                    stats_values.append(f"{value:.3f}")
                else:
                    stats_values.append(str(value))
            
            stats_df = pd.DataFrame({
                '统计项目': stats_items,
                '数值': stats_values
            })
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
            
            # 计算加权均值（如果有口袋统计信息）
            if pocket_stats_list and len(pocket_stats_list) > 0:
                weighted_mean_results = calculate_weighted_means(pocket_stats_list)
                if weighted_mean_results is not None and len(weighted_mean_results) > 0:
                    weighted_mean_df = pd.DataFrame(weighted_mean_results)
                    weighted_mean_df.to_excel(writer, sheet_name='加权均值统计', index=False)
            
            # 保存配置文件参数（如果提供了配置文件路径）
            if config_file:
                config_df = load_config_to_dataframe(config_file)
                if config_df is not None and len(config_df) > 0:
                    config_df.to_excel(writer, sheet_name='配置参数', index=False)
        
        # 保存成功后，自动追加到汇总Excel文件
        try:
            append_to_merged_summary(excel_file, summary_stats, config_file)
        except Exception as e:
            print(f"⚠️  追加到汇总Excel失败: {e}")
            # 不影响主流程，只打印警告
        
        return True
        
    except Exception as e:
        print(f"⚠️  保存Excel失败: {e}")
        traceback.print_exc()
        return False


def cleanup_zombie_processes(interactive=True):
    """
    清理残留的采样和评估相关进程
    
    Args:
        interactive: 是否交互式询问用户（默认True）
    
    Returns:
        int: 清理的进程数量
    """
    # 定义要清理的进程模式
    PATTERNS = [
        "batch_sampleandeval_parallel.py",
        "sample_diffusion.py",
        "evaluate_pt_with_correct_reconstruct.py"
    ]
    
    # 定义要保护的进程模式（薛定谔相关）
    PROTECTED_PATTERNS = [
        "schrodinger",
        "gdesmond",
        "glide",
        "jmonitor"
    ]
    
    # 收集所有相关进程PID
    all_pids = []
    for pattern in PATTERNS:
        try:
            pids = subprocess.run(
                ['pgrep', '-f', pattern],
                capture_output=True,
                text=True,
                timeout=5
            )
            if pids.returncode == 0 and pids.stdout.strip():
                for pid_str in pids.stdout.strip().split('\n'):
                    try:
                        pid = int(pid_str.strip())
                        if pid > 0:
                            all_pids.append(pid)
                    except ValueError:
                        continue
        except Exception:
            continue
    
    # 去重
    all_pids = sorted(list(set(all_pids)))
    
    if not all_pids:
        return 0
    
    # 过滤掉受保护的进程和当前进程
    filtered_pids = []
    current_pid = os.getpid()
    
    for pid in all_pids:
        if pid == current_pid:
            continue
        
        try:
            # 获取进程命令
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'cmd='],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                cmd = result.stdout.strip()
                
                # 检查是否受保护
                is_protected = False
                for protected in PROTECTED_PATTERNS:
                    if protected.lower() in cmd.lower():
                        is_protected = True
                        break
                
                # 检查是否是我们要清理的进程
                if not is_protected:
                    for pattern in PATTERNS:
                        if pattern in cmd:
                            filtered_pids.append((pid, cmd))
                            break
        except Exception:
            continue
    
    if not filtered_pids:
        return 0
    
    # 显示找到的进程
    print(f"\n{'='*60}")
    print(f"检测到 {len(filtered_pids)} 个残留进程")
    print(f"{'='*60}")
    for pid, cmd in filtered_pids[:10]:  # 只显示前10个
        print(f"  PID {pid}: {cmd[:80]}")
    if len(filtered_pids) > 10:
        print(f"  ... 还有 {len(filtered_pids) - 10} 个进程")
    print(f"{'='*60}")
    
    # 询问用户是否清理（仅在交互模式下）
    if interactive:
        try:
            response = input(f"\n是否清理这些残留进程? (y/n，默认y): ").strip().lower()
            if response and response not in ['y', 'yes']:
                print("已跳过清理")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已跳过清理")
            return 0
    else:
        # 非交互模式：直接清理，不询问
        print(f"\n自动清理 {len(filtered_pids)} 个残留进程...")
    
    # 清理进程
    killed = 0
    failed = 0
    need_root = []
    
    print(f"\n开始清理...")
    for pid, cmd in filtered_pids:
        try:
            # 先尝试优雅终止
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            
            # 检查进程是否还在运行
            try:
                os.kill(pid, 0)  # 检查进程是否存在
                # 如果还在运行，强制终止
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.2)
            except ProcessLookupError:
                # 进程已不存在
                pass
            
            # 再次检查
            try:
                os.kill(pid, 0)
                # 进程还在运行，可能需要root权限
                need_root.append(pid)
                failed += 1
            except ProcessLookupError:
                killed += 1
        except PermissionError:
            # 需要root权限
            need_root.append(pid)
            failed += 1
        except Exception as e:
            # 其他错误，尝试使用kill命令
            need_root.append(pid)
            failed += 1
    
    # 如果有需要root权限的进程，尝试使用kill命令（可能需要sudo）
    if need_root:
        print(f"\n检测到 {len(need_root)} 个进程需要root权限，尝试使用kill命令清理...")
        for pid in need_root:
            try:
                # 尝试使用kill命令（如果当前用户有权限）
                result = subprocess.run(
                    ['kill', '-9', str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    time.sleep(0.2)
                    # 检查进程是否已终止
                    try:
                        os.kill(pid, 0)
                        # 进程还在，标记为失败
                    except ProcessLookupError:
                        # 进程已终止
                        killed += 1
                        failed -= 1
                        need_root.remove(pid)
            except Exception:
                pass
        
        # 如果还有需要root权限的进程，提示用户
        if need_root:
            print(f"⚠️  仍有 {len(need_root)} 个进程需要root权限才能清理")
            print(f"   这些进程的PID: {need_root[:10]}{'...' if len(need_root) > 10 else ''}")
            print(f"   建议:")
            print(f"   1. 使用root用户运行脚本")
            print(f"   2. 或手动清理: sudo kill -9 {' '.join(map(str, need_root[:10]))}")
            if len(need_root) > 10:
                print(f"   3. 或使用清理脚本: sudo bash cleanup_sampling_as_root.sh")
    
    if killed > 0:
        print(f"✓ 成功清理 {killed} 个进程")
        time.sleep(1)  # 等待资源释放
    if failed > 0:
        print(f"⚠️  {failed} 个进程清理失败（可能需要root权限）")
    
    return killed


def calculate_weighted_means(pocket_stats_list):
    """
    计算加权均值
    
    Args:
        pocket_stats_list: 每个口袋的统计信息列表
    
    Returns:
        list: 加权均值计算结果列表
    """
    if not pocket_stats_list or len(pocket_stats_list) == 0:
        return []
    
    try:
        # 转换为DataFrame
        df = pd.DataFrame(pocket_stats_list)
        
        # 计算总成功对接数
        total_success = df['评估成功数'].sum()
        if total_success == 0:
            return []
        
        # 找到所有包含"平均"的列（这些是需要计算加权均值的参数）
        valid_mean_columns = [col for col in df.columns 
                            if '平均' in col and col != '评估成功数' and col != '数据ID']
        
        if not valid_mean_columns:
            return []
        
        # 存储计算结果
        results = []
        
        for col in valid_mean_columns:
            # 创建临时数据框，只包含非空值的行
            temp_df = df[['评估成功数', col]].dropna()
            
            # 计算简单平均值（所有口袋的平均值，不管是否有数据）
            simple_mean = df[col].mean()
            
            if len(temp_df) == 0:
                real_mean = np.nan
                weighted_sum = 0.0
                used_pockets = 0
            else:
                # 计算加权和：评估成功数 × 参数均值
                temp_df['加权和'] = temp_df['评估成功数'] * temp_df[col]
                
                # 计算真实均值（加权平均）
                weighted_sum = temp_df['加权和'].sum()
                used_pockets = len(temp_df)
                real_mean = weighted_sum / total_success
            
            # 计算差异百分比
            if not pd.isna(simple_mean) and not pd.isna(real_mean) and simple_mean != 0:
                diff_percent = ((real_mean - simple_mean) / simple_mean * 100)
            else:
                diff_percent = np.nan
            
            results.append({
                '参数名称': col,
                '参与计算的口袋数': used_pockets,
                '加权和': weighted_sum,
                '简单平均值': simple_mean if not pd.isna(simple_mean) else np.nan,
                '真实均值（加权平均）': real_mean if not pd.isna(real_mean) else np.nan,
                '差异百分比': diff_percent if not pd.isna(diff_percent) else np.nan
            })
        
        return results
        
    except Exception as e:
        print(f"⚠️  计算加权均值失败: {e}")
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description='批量采样和评估脚本（并行执行）')
    
    parser.add_argument('--start', type=int, default=0,
                       help='起始 data_id（默认: 0）')
    parser.add_argument('--end', type=int, default=99,
                       help='结束 data_id（默认: 99）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/sampling.yml）')
    
    default_protein_root = os.environ.get('PROTEIN_ROOT', None)
    if default_protein_root is None:
        possible_paths = [
            REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
            Path('/mnt/e/DiffDynamic/data/crossdocked_v1.1_rmsd1.0_pocket10'),
            REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0',
        ]
        for path in possible_paths:
            if path.exists():
                default_protein_root = str(path)
                break
    
    parser.add_argument('--protein_root', type=str, default=default_protein_root,
                       help=f'蛋白质数据根目录（默认: {default_protein_root if default_protein_root else "未找到，请指定"}）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='评估输出目录（已废弃，评估结果自动保存在outputs目录下）')
    parser.add_argument('--atom_mode', type=str, default='add_aromatic',
                       help='原子模式（默认: add_aromatic）')
    parser.add_argument('--exhaustiveness', type=int, default=8,
                       help='AutoDock Vina对接强度（默认: 8）')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的.pt文件（不重新采样）')
    parser.add_argument('--excel_file', type=str, default=None,
                       help='Excel记录文件路径（默认: batch_evaluation_summary_{timestamp}.xlsx）')
    parser.add_argument('--sample-only', action='store_true',
                       help='只生成模式：只执行采样，不执行评估（默认: False）')
    parser.add_argument('--auto-cleanup', action='store_true',
                       help='自动清理残留进程（默认: False，启动时会询问）')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='不清理残留进程（默认: False）')
    
    # GPU配置参数
    parser.add_argument('--gpus', type=str, default=None,
                       help='指定使用的GPU ID，支持多种格式：\n'
                            '  - "0,1,2,3" 指定多个GPU\n'
                            '  - "0-3" 指定GPU范围\n'
                            '  - "0,2-4,6" 混合格式\n'
                            '  - "all" 使用所有可用GPU\n'
                            f'  （默认: {",".join(map(str, DEFAULT_GPU_IDS))}）')
    parser.add_argument('--num_cpu_cores', type=int, default=None,
                       help=f'用于评估的CPU核心数（默认: {DEFAULT_NUM_CPU_CORES}）')
    
    args = parser.parse_args()
    
    if args.config is None:
        args.config = CONFIG
    else:
        args.config = Path(args.config)
    
    # 只在非只生成模式下要求protein_root
    if not args.sample_only:
        if args.protein_root is None:
            print(f"❌ 错误: 未指定蛋白质数据根目录（--protein_root）")
            print(f"   请使用 --protein_root 参数指定蛋白质数据目录")
            sys.exit(1)
        
        args.protein_root = Path(args.protein_root)
        
        if not args.protein_root.exists():
            print(f"❌ 错误: 蛋白质数据根目录不存在: {args.protein_root}")
            sys.exit(1)
        
        if not EVAL_SCRIPT.exists():
            print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
            sys.exit(1)
    elif args.protein_root:
        # 如果只生成模式下提供了protein_root，也验证一下
        args.protein_root = Path(args.protein_root)
    
    if not args.config.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not SCRIPT.exists():
        print(f"❌ 错误: 采样脚本不存在: {SCRIPT}")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 自动清理残留进程（如果启用）
    if not args.no_cleanup:
        # 默认不询问，直接清理
        # --auto-cleanup: 自动清理（不询问，与默认行为相同）
        # 如果需要询问，可以设置 interactive=True
        interactive = False  # 默认不询问，直接清理
        cleaned_count = cleanup_zombie_processes(interactive=interactive)
        if cleaned_count > 0:
            print(f"✓ 已清理 {cleaned_count} 个残留进程，等待资源释放...")
            time.sleep(2)  # 等待资源释放
            print()
    
    # 解析GPU配置
    if args.gpus:
        try:
            gpu_ids = parse_gpu_ids(args.gpus)
            if not gpu_ids:
                # 如果指定了"all"但检测不到GPU，尝试使用nvidia-smi检测
                if args.gpus.lower() == 'all':
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--list-gpus'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            num_gpus = len(result.stdout.strip().split('\n'))
                            gpu_ids = list(range(num_gpus))
                            print(f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus} 个GPU")
                            print(f"   将使用GPU IDs: {gpu_ids}")
                            print(f"   子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA")
                        else:
                            print(f"⚠️  警告: 未找到可用的GPU（PyTorch和nvidia-smi都检测不到）")
                            print(f"   将继续运行，但任务可能会失败")
                            print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                            # 使用默认GPU列表，让程序继续运行
                            gpu_ids = DEFAULT_GPU_IDS
                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                        print(f"⚠️  警告: 未找到可用的GPU")
                        print(f"   将继续运行，但任务可能会失败")
                        print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                        # 使用默认GPU列表，让程序继续运行
                        gpu_ids = DEFAULT_GPU_IDS
                else:
                    print(f"❌ 错误: 未找到可用的GPU")
                    print(f"   指定的GPU: {args.gpus}")
                    sys.exit(1)
        except ValueError as e:
            print(f"❌ 错误: GPU ID格式无效: {e}")
            print(f"   支持的格式: '0,1,2,3', '0-3', '0,2-4,6', 'all'")
            sys.exit(1)
    else:
        gpu_ids = DEFAULT_GPU_IDS
    
    # 验证GPU是否可用
    # 注意：主进程可能检测不到CUDA（因为环境变量或驱动问题），
    # 但子进程在设置CUDA_VISIBLE_DEVICES后可能可以检测到
    # 所以这里只做警告，不阻止运行
    cuda_available_in_main = False
    if torch is not None and torch.cuda.is_available():
        cuda_available_in_main = True
        available_gpus = get_available_gpus()
        invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpus]
        if invalid_gpus:
            print(f"⚠️  警告: 以下GPU在主进程中不可用: {invalid_gpus}")
            print(f"   主进程检测到的可用GPU: {available_gpus}")
            print(f"   将继续尝试使用指定的GPU（子进程可能会检测到）")
    else:
        # 尝试使用 nvidia-smi 作为备用检测方法
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                num_gpus_detected = len(result.stdout.strip().split('\n'))
                print(f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus_detected} 个GPU")
                print(f"   将继续运行，子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA")
                print(f"   如果所有任务都失败，请检查:")
                print(f"   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
                print(f"   2. PyTorch是否正确安装了CUDA版本")
                print(f"   3. CUDA运行时库是否正确安装")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("⚠️  警告: CUDA不可用（PyTorch和nvidia-smi都检测不到）")
            print("   将继续运行，但任务可能会失败")
            print("   如果所有任务都失败，请检查:")
            print("   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
            print("   2. NVIDIA驱动是否正确安装")
            print("   3. PyTorch是否正确安装了CUDA版本")
    
    num_gpus = len(gpu_ids)
    num_cpu_cores = args.num_cpu_cores if args.num_cpu_cores is not None else DEFAULT_NUM_CPU_CORES
    
    # 确保batchsummary目录存在
    BATCHSUMMARY_DIR = REPO_ROOT / 'batchsummary'
    BATCHSUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    batch_start_time = time.time()
    # 使用本地时区（CST，UTC+8）生成时间戳
    cst = timezone(timedelta(hours=8))
    timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
    
    # 生成配置参数字符串
    config_params = generate_config_params_string(args.config)
    
    if args.excel_file:
        excel_file = Path(args.excel_file)
    else:
        # 原有文件名格式
        base_filename = f'batch_evaluation_summary_{timestamp}'
        # 如果成功提取了配置参数，则添加到文件名中
        if config_params:
            excel_file = BATCHSUMMARY_DIR / f'{base_filename}_{config_params}.xlsx'
        else:
            excel_file = BATCHSUMMARY_DIR / f'{base_filename}.xlsx'
    
    # 打印配置信息
    print(f"\n{'='*60}")
    if args.sample_only:
        print(f"批量采样配置（只生成模式）")
    else:
        print(f"批量采样和评估配置（并行版本）")
    print(f"{'='*60}")
    print(f"数据ID范围: {args.start} 到 {args.end}")
    print(f"并行配置:")
    print(f"  - GPU数量: {num_gpus} (GPU IDs: {gpu_ids})")
    if not args.sample_only:
        print(f"  - CPU核心数: {num_cpu_cores}")
    print(f"配置文件: {args.config}")
    if not args.sample_only:
        print(f"蛋白质数据根目录: {args.protein_root}")
        print(f"评估结果保存位置: {OUTPUT_DIR}")
        print(f"原子模式: {args.atom_mode}")
        print(f"对接强度: {args.exhaustiveness}")
    else:
        print(f"输出目录: {OUTPUT_DIR}")
    print(f"跳过已存在: {args.skip_existing}")
    if not args.sample_only:
        print(f"Excel记录文件: {excel_file}")
    print(f"{'='*60}\n")
    
    # 创建Manager用于进程间共享的锁
    manager = Manager()
    excel_lock = manager.Lock()
    
    total = args.end - args.start + 1
    start_time = time.time()
    
    # ========== 第一阶段：采样（4个GPU并行） ==========
    print(f"\n{'='*60}")
    print(f"第一阶段：并行采样")
    print(f"{'='*60}")
    print(f"数据ID范围: {args.start} 到 {args.end} (共 {total} 个)")
    print(f"使用 {num_gpus} 个GPU并行采样 (GPU IDs: {gpu_ids})")
    print(f"GPU列表详情: {gpu_ids}")
    print(f"{'='*60}\n")
    
    # 准备采样任务列表
    # 按GPU分组任务，确保每个GPU的任务均匀分布
    tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for i, data_id in enumerate(range(args.start, args.end + 1)):
        # 轮询分配GPU（确保正确轮询）
        gpu_id = gpu_ids[i % num_gpus]
        tasks_by_gpu[gpu_id].append((
            data_id, gpu_id, str(args.config), args.skip_existing
        ))
    
    # 交错排列任务，确保所有GPU的任务均匀分布
    # 这样pool.map()处理时，前几个任务会分配给不同的GPU
    sampling_tasks = []
    max_tasks_per_gpu = max(len(tasks) for tasks in tasks_by_gpu.values())
    for i in range(max_tasks_per_gpu):
        for gpu_id in gpu_ids:
            if i < len(tasks_by_gpu[gpu_id]):
                sampling_tasks.append(tasks_by_gpu[gpu_id][i])
    
    # 验证GPU分配（调试信息）
    gpu_distribution = {}
    for data_id, gpu_id, _, _ in sampling_tasks:
        if gpu_id not in gpu_distribution:
            gpu_distribution[gpu_id] = []
        gpu_distribution[gpu_id].append(data_id)
    
    print(f"GPU分配验证:")
    for gpu_id in sorted(gpu_distribution.keys()):
        count = len(gpu_distribution[gpu_id])
        print(f"  GPU {gpu_id}: {count} 个任务 (data_ids: {gpu_distribution[gpu_id][:10]}{'...' if count > 10 else ''})")
    print(f"任务列表前10个: {[(t[0], t[1]) for t in sampling_tasks[:10]]}")
    
    # 验证前N个任务是否分配给所有GPU（N=GPU数量）
    if len(sampling_tasks) >= num_gpus:
        first_n_tasks = [(t[0], t[1]) for t in sampling_tasks[:num_gpus]]
        first_n_gpus = [gpu_id for _, gpu_id in first_n_tasks]
        unique_gpus_in_first_n = set(first_n_gpus)
        print(f"前{num_gpus}个任务的GPU分配: {first_n_gpus}")
        print(f"前{num_gpus}个任务覆盖的GPU: {sorted(unique_gpus_in_first_n)} (期望: {sorted(gpu_ids)})")
        if len(unique_gpus_in_first_n) < num_gpus:
            print(f"⚠️  警告: 前{num_gpus}个任务没有覆盖所有GPU！")
            missing_gpus = set(gpu_ids) - unique_gpus_in_first_n
            print(f"   缺失的GPU: {sorted(missing_gpus)}")
    print()
    
    sampling_start_time = time.time()
    
    # 采样阶段：限制为GPU数量的进程（每个GPU一个）
    # 确保进程数等于GPU数量，以便充分利用所有GPU
    pool_processes = num_gpus
    print(f"创建进程池: {pool_processes} 个工作进程 (对应 {num_gpus} 个GPU)")
    
    # 使用按GPU分组的方式，确保每个GPU都有独立的工作进程
    # 方法：为每个GPU创建一个独立的进程池，每个进程池只处理该GPU的任务
    all_sampling_results = []
    
    try:
        # 为每个GPU创建独立的进程池
        pools = {}
        async_results = {}
        
        for gpu_id in gpu_ids:
            # 获取该GPU的所有任务
            gpu_tasks = [task for task in sampling_tasks if task[1] == gpu_id]
            if not gpu_tasks:
                continue
            
            print(f"GPU {gpu_id}: {len(gpu_tasks)} 个任务，创建独立进程池...")
            # 为每个GPU创建一个工作进程（因为每个GPU只需要一个进程）
            pool = Pool(processes=1)
            pools[gpu_id] = pool
            # 异步执行该GPU的所有任务
            async_results[gpu_id] = pool.map_async(process_sampling_task, gpu_tasks)
        
        # 等待所有GPU的任务完成
        print(f"等待所有GPU任务完成...")
        for gpu_id in gpu_ids:
            if gpu_id in async_results:
                try:
                    gpu_results = async_results[gpu_id].get()
                    all_sampling_results.extend(gpu_results)
                    print(f"GPU {gpu_id} 完成: {len(gpu_results)} 个任务")
                except Exception as e:
                    print(f"GPU {gpu_id} 执行出错: {e}")
        
        # 按data_id排序结果，保持原始顺序
        sampling_results = sorted(all_sampling_results, key=lambda x: x[0])
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在清理采样进程...")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()  # 强制终止所有工作进程
                pool.join()       # 等待进程完全退出
        raise
    except Exception as e:
        print(f"\n⚠️  采样阶段发生异常: {e}")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()
                pool.join()
        raise
    finally:
        # 清理所有进程池
        for gpu_id, pool in pools.items():
            if pool:
                pool.close()  # 关闭进程池，不再接受新任务
                pool.join()   # 等待所有工作进程完成
    
    sampling_time = time.time() - sampling_start_time
    
    # 统计采样结果
    sampling_success = [r for r in sampling_results if r[1]]
    sampling_fail = [r for r in sampling_results if not r[1]]
    
    print(f"\n{'='*60}")
    print(f"采样阶段完成")
    print(f"{'='*60}")
    print(f"成功: {len(sampling_success)}")
    print(f"失败: {len(sampling_fail)}")
    print(f"耗时: {sampling_time:.2f} 秒 ({sampling_time/60:.2f} 分钟)")
    print(f"{'='*60}\n")
    
    # 如果启用只生成模式，跳过评估阶段
    if args.sample_only:
        print(f"\n{'='*60}")
        print(f"只生成模式：跳过评估阶段")
        print(f"{'='*60}")
        print(f"采样成功的任务数: {len(sampling_success)}")
        print(f"生成的pt文件位置: {OUTPUT_DIR}")
        print(f"{'='*60}\n")
        
        # 只记录采样结果
        all_results = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, True, "采样成功", None, pt_file, None))
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))
        
        evaluation_time = 0
    else:
        # ========== 第二阶段：评估（64个CPU核心并行） ==========
        print(f"\n{'='*60}")
        print(f"第二阶段：并行评估")
        print(f"{'='*60}")
        print(f"待评估任务数: {len(sampling_success)}")
        print(f"使用最多 {num_cpu_cores} 个CPU核心并行评估")
        print(f"{'='*60}\n")
        
        # 准备评估任务列表（只评估采样成功的）
        evaluation_tasks = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            evaluation_tasks.append((
                data_id, pt_file, str(args.protein_root),
                args.atom_mode, args.exhaustiveness,
                str(excel_file), batch_start_time, excel_lock
            ))
        
        # 对于采样失败的任务，也记录到结果中
        all_results = []
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))
        
        evaluation_time = 0
        if evaluation_tasks:
            evaluation_start_time = time.time()
            
            # 评估阶段：使用指定数量的CPU核心并行
            max_eval_workers = min(num_cpu_cores, len(evaluation_tasks))
            
            eval_pool = None
            try:
                eval_pool = Pool(processes=max_eval_workers)
                evaluation_results = eval_pool.map(process_evaluation_task, evaluation_tasks)
            except KeyboardInterrupt:
                print("\n⚠️  收到中断信号，正在清理评估进程...")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            except Exception as e:
                print(f"\n⚠️  评估阶段发生异常: {e}")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            finally:
                if eval_pool:
                    eval_pool.close()
                    eval_pool.join()
            
            evaluation_time = time.time() - evaluation_start_time
            
            # 合并评估结果
            all_results.extend(evaluation_results)
            
            print(f"\n{'='*60}")
            print(f"评估阶段完成")
            print(f"{'='*60}")
            print(f"成功: {sum(1 for r in evaluation_results if r[1])}")
            print(f"失败: {sum(1 for r in evaluation_results if not r[1])}")
            print(f"耗时: {evaluation_time:.2f} 秒 ({evaluation_time/60:.2f} 分钟)")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"评估阶段：无任务（所有采样均失败）")
            print(f"{'='*60}\n")
    
    # 统计最终结果
    success_count = sum(1 for r in all_results if r[1])
    fail_count = sum(1 for r in all_results if not r[1])
    skip_count = 0
    
    # 批量保存所有结果到Excel（只生成模式下跳过）
    if excel_file and not args.sample_only:
        print(f"\n{'='*70}")
        print(f"收集并保存评估结果到Excel...")
        print(f"{'='*70}")
        
        molecule_records, summary_stats, pocket_stats_list = collect_all_evaluation_results(all_results, batch_start_time)
        
        print(f"收集到的分子记录数: {len(molecule_records)}")
        print(f"收集到的口袋统计信息数: {len(pocket_stats_list)}")
        
        if save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time, pocket_stats_list, config_file=str(args.config)):
            print(f"✅ 成功保存 {len(molecule_records)} 个对接成功分子到Excel: {excel_file}")
            print(f"   统计信息:")
            print(f"     - 应生成分子数: {summary_stats.get('应生成分子数', 0)}")
            print(f"     - 可重建分子数: {summary_stats.get('可重建分子数', 0)}")
            print(f"     - 对接成功分子数: {summary_stats.get('对接成功分子数', 0)}")
            if 'Vina_Dock_平均亲和力' in summary_stats:
                print(f"     - Vina_Dock_平均亲和力: {summary_stats['Vina_Dock_平均亲和力']:.3f} kcal/mol")
            
            # 批量评估完成后，从 batchsummary 文件读取参数并填写到 evaall bestchoice.xlsx
            try:
                # 导入更新函数（sys已在文件顶部导入）
                eval_script_path = REPO_ROOT / 'evaluate_pt_with_correct_reconstruct.py'
                if eval_script_path.exists():
                    # 动态导入模块
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("evaluate_pt_module", eval_script_path)
                    evaluate_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(evaluate_module)
                    
                    # 调用更新函数
                    if hasattr(evaluate_module, 'update_bestchoice_excel_with_params'):
                        evaluate_module.update_bestchoice_excel_with_params(output_dir=None)
            except Exception as e:
                print(f"  ⚠️  更新 evaall bestchoice.xlsx 失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Excel保存失败")
        print(f"{'='*70}\n")
    
    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    if args.sample_only:
        print(f"批量采样完成（只生成模式）")
    else:
        print(f"批量处理完成（并行版本）")
    print(f"{'='*60}")
    print(f"总计: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"  - 采样耗时: {sampling_time:.2f} 秒 ({sampling_time/60:.2f} 分钟)")
    if evaluation_time > 0:
        print(f"  - 评估耗时: {evaluation_time:.2f} 秒 ({evaluation_time/60:.2f} 分钟)")
    print(f"平均每个任务: {elapsed_time/total:.2f} 秒")
    if args.sample_only:
        print(f"📁 生成的pt文件保存在: {OUTPUT_DIR}")
        print(f"   成功生成 {len(sampling_success)} 个pt文件")
    elif excel_file:
        print(f"📊 详细记录已保存至: {excel_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
