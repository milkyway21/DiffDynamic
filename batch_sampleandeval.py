#!/usr/bin/env python3
"""
批量采样和评估脚本：串行执行采样和评估

功能：
1. 对每个 data_id 执行采样：python3 scripts/sample_diffusion.py configs/sampling.yml --data_id {i}
2. 找到生成的文件：outputs/result_{data_id}_{timestamp}.pt
3. 执行评估：python3 evaluate_pt_with_correct_reconstruct.py {pt_file} --protein_root ... --output_dir ... --atom_mode add_aromatic --exhaustiveness 8

使用方法：
    # 基本用法（0到99）
    python3 batch_sampleandeval.py
    
    # 指定范围
    python3 batch_sampleandeval.py --start 0 --end 99
    
    # 直接运行评估脚本的示例：
    python3 evaluate_pt_with_correct_reconstruct.py \
    /home/user/Desktop/Ye/DiffDynamic/outputs/result_55_20251211_153937.pt \
    --protein_root /home/user/Desktop/Ye/DiffDynamic/data/crossdocked_v1.1_rmsd1.0_pocket10 \
    --output_dir /home/user/Desktop/Ye/DiffDynamic/outputs/eval_results \
    --atom_mode add_aromatic \
    --exhaustiveness 8
    
    # 指定蛋白质数据根目录
    python3 batch_sampleandeval.py --protein_root /path/to/protein/data
    
    # 指定评估输出目录
    python3 batch_sampleandeval.py --output_dir /path/to/eval/results
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
import glob
import threading
import traceback
import re

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

# Excel写入锁（用于线程安全写入，虽然串行执行但保持一致性）
excel_write_lock = threading.Lock()

# 项目根目录
REPO_ROOT = Path(__file__).parent
SCRIPT = REPO_ROOT / 'scripts' / 'sample_diffusion.py'
CONFIG = REPO_ROOT / 'configs' / 'sampling.yml'
EVAL_SCRIPT = REPO_ROOT / 'evaluate_pt_with_correct_reconstruct.py'
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


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
    
    # 查找所有匹配的.pt文件（格式：result_{data_id}_{timestamp}.pt）
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
    读取评估结果文件中的统计数据（照抄自 batch_sample_all.py）
    
    Args:
        pt_file_path: 采样结果.pt文件路径（outputs/result_YYYYMMDD_HHMMSS.pt）
        data_id: 数据ID
        wait_timeout: 等待评估结果的最大时间（秒）
    
    Returns:
        tuple: (success, vina_mean, vina_median, num_scores, message, eval_output_dir)
    """
    if torch is None or np is None:
        return (False, None, None, 0, "torch或numpy未安装", None)
    
    pt_file_path = Path(pt_file_path).resolve()  # 转换为绝对路径
    outputs_dir = pt_file_path.parent  # outputs目录
    
    # 从.pt文件名提取口袋编号（result_idx_时间.pt）
    pt_filename = pt_file_path.stem  # result_idx_时间
    if pt_filename.startswith('result_'):
        parts = pt_filename.split('_')
        if len(parts) >= 3:
            pocket_id = parts[1]  # 第二部分是口袋编号（idx）
        else:
            # 兼容旧格式：result_YYYYMMDD_HHMMSS，使用data_id作为pocket_id
            pocket_id = str(data_id)
    else:
        pocket_id = str(data_id)
    
    # 使用glob模式匹配新的命名格式 eval_{data_id}_* 或旧的命名格式
    # 优先匹配以eval_{pocket_id}_开头的目录
    eval_dirs = list(outputs_dir.glob(f'eval_{pocket_id}_*'))
    
    # 如果没有找到，尝试匹配所有eval_*目录（兼容旧格式）
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob('eval_*'))
    
    if not eval_dirs:
        return (False, None, None, 0, f"未找到评估输出目录（在 {outputs_dir} 中，查找模式: eval_{pocket_id}_*）", None)
    
    # 优先选择带时间戳的新格式目录（格式：eval_{pocket_id}_gf*_{start}_{end}_{timestamp}_...）
    # 时间戳格式：YYYYMMDD_HHMMSS，可以通过检查是否包含类似 "20251208_011104" 的模式来判断
    timestamp_pattern = r'_\d{8}_\d{6}_'  # 匹配 _YYYYMMDD_HHMMSS_ 格式
    
    # 分离新格式目录（带时间戳）和旧格式目录（不带时间戳）
    new_format_dirs = [d for d in eval_dirs if re.search(timestamp_pattern, d.name)]
    old_format_dirs = [d for d in eval_dirs if d not in new_format_dirs]
    
    # 优先使用新格式目录，如果存在的话
    if new_format_dirs:
        # 在新格式目录中，选择最新的（按修改时间）
        new_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = new_format_dirs[0]
    elif old_format_dirs:
        # 如果没有新格式目录，使用旧格式目录（按修改时间）
        old_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = old_format_dirs[0]
    else:
        # 如果都没有，使用所有目录（按修改时间）
        eval_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = eval_dirs[0]
    
    # 检查eval目录是否存在
    if not eval_output_dir.exists():
        return (False, None, None, 0, f"评估输出目录不存在: {eval_output_dir}", None)
    
    # 等待评估结果文件生成（最多等待wait_timeout秒）
    start_wait = time.time()
    eval_result_files = []
    while time.time() - start_wait < wait_timeout:
        eval_result_files = list(eval_output_dir.glob('eval_results_*.pt'))
        if eval_result_files:
            break
        time.sleep(2)  # 每2秒检查一次
    
    if not eval_result_files:
        # 列出eval目录中的所有文件，帮助调试
        all_files = list(eval_output_dir.glob('*'))
        file_list = ', '.join([f.name for f in all_files[:10]])  # 只显示前10个
        if len(all_files) > 10:
            file_list += f' ... (共{len(all_files)}个文件)'
        return (False, None, None, 0, 
                f"等待{wait_timeout}秒后仍未找到评估结果文件 (eval_results_*.pt)\n"
                f"   评估目录: {eval_output_dir}\n"
                f"   目录中的文件: {file_list if all_files else '空目录'}", 
                str(eval_output_dir))
    
    try:
        # 读取最新的评估结果文件
        latest_eval_file = max(eval_result_files, key=os.path.getmtime)
        eval_data = torch.load(latest_eval_file, map_location='cpu')
        
        # 提取vina得分（兼容新旧格式）
        statistics = eval_data.get('statistics', {})
        # 优先读取新格式的vina得分（三种模式）
        vina_dock_scores = statistics.get('vina_dock_scores', [])
        vina_score_only_scores = statistics.get('vina_score_only_scores', [])
        vina_minimize_scores = statistics.get('vina_minimize_scores', [])
        # 兼容旧格式
        vina_scores = statistics.get('vina_scores', [])
        
        # 优先使用dock模式得分，如果没有则尝试其他模式
        if vina_dock_scores:
            vina_scores = vina_dock_scores
        elif vina_minimize_scores:
            vina_scores = vina_minimize_scores
        elif vina_score_only_scores:
            vina_scores = vina_score_only_scores
        
        # 获取诊断信息
        n_reconstruct_success = eval_data.get('n_reconstruct_success', 0)
        n_eval_success = eval_data.get('n_eval_success', 0)
        ligand_filename = eval_data.get('ligand_filename', 'N/A')
        protein_root = eval_data.get('protein_root', 'N/A')
        
        if not vina_scores:
            # 提供详细的诊断信息
            diagnostic_msg = f"评估结果中无vina得分"
            if n_reconstruct_success > 0 and n_eval_success == 0:
                diagnostic_msg += f" (重建成功{n_reconstruct_success}个，但对接全部失败)"
                diagnostic_msg += f"\n   配体文件: {ligand_filename}"
                diagnostic_msg += f"\n   蛋白根目录: {protein_root}"
                # 检查第一个失败的错误信息
                results = eval_data.get('results', [])
                for r in results[:5]:  # 检查前5个
                    if r.get('mol') and 'error' in r:
                        error_msg = r['error'][:200]  # 截取前200字符
                        diagnostic_msg += f"\n   错误示例: {error_msg}"
                        break
                    # 检查是否有对接结果但失败的情况
                    if r.get('mol') and not r.get('success'):
                        # 检查是否有vina结果但都失败
                        has_vina_dock = r.get('vina_dock') and len(r.get('vina_dock', [])) > 0
                        has_vina_score_only = r.get('vina_score_only') is not None
                        has_vina_minimize = r.get('vina_minimize') is not None
                        if not (has_vina_dock or has_vina_score_only or has_vina_minimize):
                            diagnostic_msg += f"\n   分子{r.get('mol_idx', 'N/A')}: 所有对接模式均失败"
                            break
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


def run_single_sample(data_id, config_file=None):
    """
    执行单个采样任务
    
    Args:
        data_id: 数据ID
        config_file: 配置文件路径（默认：configs/sampling.yml）
    
    Returns:
        tuple: (success, pt_file_path, message)
    """
    if config_file is None:
        config_file = CONFIG
    
    print(f"\n{'='*60}")
    print(f"开始采样 data_id={data_id}")
    print(f"{'='*60}")
    
    # 构建采样命令
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        '--data_id', str(data_id)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 执行采样
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # 实时显示输出
            text=True
        )
        
        # 等待一小段时间，确保文件已保存
        time.sleep(1)
        
        # 查找生成的文件
        pt_file = find_latest_result_file(data_id)
        
        if pt_file and pt_file.exists():
            print(f"✅ 采样成功: {pt_file}")
            return True, pt_file, "采样成功"
        else:
            print(f"⚠️  采样完成但未找到结果文件")
            return False, None, "未找到结果文件"
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 采样失败: {e}")
        return False, None, f"采样失败: {e}"
    except Exception as e:
        print(f"❌ 采样出错: {e}")
        return False, None, f"采样出错: {e}"


def append_to_excel(excel_file, timestamp, execution_time, data_id, pt_file, vina_mean, vina_median, 
                    num_scores, status, message):
    """
    将评估结果追加到Excel文件（线程安全版本，照抄自 batch_sample_all.py）
    
    Args:
        excel_file: Excel文件路径
        timestamp: 执行时间戳
        execution_time: 执行耗时（秒）
        data_id: 数据ID
        pt_file: .pt文件路径
        vina_mean: Vina平均得分
        vina_median: Vina中位数得分
        num_scores: 得分数量
        status: 状态（成功/失败）
        message: 备注信息
    """
    if pd is None:
        return False
    
    # 使用锁确保线程安全
    with excel_write_lock:
        try:
            # 确保excel_file是Path对象
            if not isinstance(excel_file, Path):
                excel_file = Path(excel_file)
            
            # 准备新行数据
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
            
            # 读取现有数据或创建新DataFrame
            if excel_file.exists():
                try:
                    df = pd.read_excel(excel_file, engine='openpyxl')
                except Exception as e:
                    print(f'⚠️  警告: 读取Excel文件失败 {excel_file}: {e}. 创建新的DataFrame.')
                    df = pd.DataFrame()
                # 计算累计均值（所有成功评估的）
                successful_rows = df[df['状态'] == '成功']
                if len(successful_rows) > 0:
                    all_means = successful_rows['Vina平均得分'].dropna().tolist()
                    if vina_mean is not None:
                        all_means.append(vina_mean)
                    cumulative_mean = np.mean(all_means) if all_means else None
                else:
                    cumulative_mean = vina_mean
            else:
                df = pd.DataFrame()
                cumulative_mean = vina_mean
            
            # 添加累计均值列
            new_row['累计均值'] = cumulative_mean if cumulative_mean is not None else ''
            
            # 添加新行
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # 保存到Excel
            excel_file.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='评估记录')
                
                # 添加统计信息工作表
                if len(df) > 0:
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
            
            return True
                
        except Exception as e:
            print(f"⚠️  写入Excel失败: {e}")
            traceback.print_exc()
            return False


def collect_all_evaluation_results(results, batch_start_time):
    """
    从所有评估结果文件中收集对接成功的分子数据（改进版：只读取本次运行的评估结果）
    
    Args:
        results: 批量采样结果列表，格式为 [(data_id, success, message, log_file, pt_file, eval_output_dir), ...]
        batch_start_time: batch启动时间（time.time()返回的时间戳）
    
    Returns:
        tuple: (molecule_records, summary_stats)
            molecule_records: 所有对接成功分子的记录列表
            summary_stats: 统计信息字典
    """
    if torch is None or np is None:
        return [], {}
    
    molecule_records = []
    total_num_samples = 0
    total_n_reconstruct_success = 0
    total_n_eval_success = 0
    
    for r in results:
        if len(r) < 5:
            continue
        
        # 支持新旧格式：新格式包含 eval_output_dir
        if len(r) >= 6:
            data_id, success, message, log_file, pt_file, eval_output_dir = r[:6]
        else:
            data_id, success, message, log_file, pt_file = r[:5]
            eval_output_dir = None
        
        if not pt_file:
            continue
        
        pt_path = Path(pt_file)
        outputs_dir = pt_path.parent
        
        # 如果已经提供了评估目录，验证它是在本次运行期间创建的；否则查找
        batch_start_datetime = datetime.fromtimestamp(batch_start_time)
        
        if eval_output_dir is None:
            # 查找对应的评估目录
            pt_filename = pt_path.stem
            if pt_filename.startswith('result_'):
                parts = pt_filename.split('_')
                if len(parts) >= 3:
                    pocket_id = parts[1]
                else:
                    pocket_id = str(data_id)
            else:
                pocket_id = str(data_id)
            
            # 查找评估目录（只查找在batch_start_time之后创建的）
            eval_dirs = list(outputs_dir.glob(f'eval_{pocket_id}_*'))
            if not eval_dirs:
                eval_dirs = list(outputs_dir.glob('eval_*'))
            
            if not eval_dirs:
                continue
            
            # 不按目录时间过滤，而是检查目录中的文件时间
            # 选择最新的评估目录（文件时间会在后面检查）
            eval_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            eval_output_dir = eval_dirs[0]
        else:
            # 使用提供的评估目录（不需要验证目录时间，因为文件时间才是关键）
            # eval_output_dir 可能是字符串或 Path 对象
            if isinstance(eval_output_dir, str):
                eval_output_dir = Path(eval_output_dir)
            elif not isinstance(eval_output_dir, Path):
                eval_output_dir = Path(str(eval_output_dir))
            
            if not eval_output_dir.exists():
                print(f"  ⚠️  评估目录不存在 (data_id={data_id}): {eval_output_dir}")
                continue
        
        # 查找评估结果文件（只查找在batch_start_time之后创建的）
        eval_result_files = list(eval_output_dir.glob('eval_results_*.pt'))
        if not eval_result_files:
            print(f"  ⚠️  评估目录中没有结果文件 (data_id={data_id}, 目录: {eval_output_dir})")
            continue
        
        # 只读取在batch_start_time之后创建的评估结果文件
        recent_eval_files = [
            f for f in eval_result_files
            if datetime.fromtimestamp(f.stat().st_mtime) >= batch_start_datetime
        ]
        
        if not recent_eval_files:
            # 如果没有找到本次运行的文件，跳过（不读取旧结果）
            print(f"  ⚠️  未找到本次运行的评估结果文件 (data_id={data_id})")
            print(f"     评估目录: {eval_output_dir}")
            print(f"     batch_start_time: {batch_start_datetime}")
            if eval_result_files:
                print(f"     找到的文件: {[str(f) for f in eval_result_files[:3]]}")
                for f in eval_result_files[:3]:
                    file_time = datetime.fromtimestamp(f.stat().st_mtime)
                    print(f"       {f.name}: {file_time} (早于启动时间: {file_time < batch_start_datetime})")
            continue
        
        # 读取本次运行中最新的评估结果文件
        latest_eval_file = max(recent_eval_files, key=os.path.getmtime)
        
        file_time = datetime.fromtimestamp(latest_eval_file.stat().st_mtime)
        print(f"  ✅ 读取评估结果文件 (data_id={data_id}): {latest_eval_file.name}")
        print(f"     文件时间: {file_time}, 启动时间: {batch_start_datetime}")
        
        try:
            eval_data = torch.load(latest_eval_file, map_location='cpu')
            
            # 提取统计信息
            num_samples = eval_data.get('num_samples', 0)
            n_reconstruct_success = eval_data.get('n_reconstruct_success', 0)
            n_eval_success = eval_data.get('n_eval_success', 0)
            
            total_num_samples += num_samples
            total_n_reconstruct_success += n_reconstruct_success
            total_n_eval_success += n_eval_success
            
            # 提取每个对接成功的分子数据
            results_list = eval_data.get('results', [])
            for result in results_list:
                # 只记录对接成功的分子
                if result.get('mol') is None:
                    continue
                
                if not result.get('success'):
                    continue
                
                # 检查是否至少有一种vina模式成功
                has_vina_result = (result.get('vina_dock') and len(result['vina_dock']) > 0) or \
                                 (result.get('vina_score_only') is not None) or \
                                 (result.get('vina_minimize') is not None)
                if not has_vina_result:
                    continue
                
                # 获取对接信息（三种模式）
                vina_dock_affinity = 'N/A'
                vina_dock_rmsd_lb = 'N/A'
                vina_dock_rmsd_ub = 'N/A'
                if result.get('vina_dock') and len(result['vina_dock']) > 0:
                    vina_dock_result = result['vina_dock'][0]
                    vina_dock_affinity = vina_dock_result['affinity']
                    vina_dock_rmsd_lb = vina_dock_result.get('rmsd_lb', 'N/A')
                    vina_dock_rmsd_ub = vina_dock_result.get('rmsd_ub', 'N/A')
                
                vina_score_only_affinity = result.get('vina_score_only', 'N/A')
                vina_minimize_affinity = result.get('vina_minimize', 'N/A')
                
                # 获取分子信息
                smiles = result.get('smiles', 'N/A')
                mol_idx = result.get('mol_idx', 'N/A')
                
                # 获取化学性质指标
                chem = result.get('chem', {})
                qed = chem.get('qed', 'N/A') if chem else 'N/A'
                sa = chem.get('sa', 'N/A') if chem else 'N/A'
                
                # 获取分子结构指标
                atom_type_jsd = result.get('atom_type_jsd', 'N/A')
                
                # 构建记录
                record = {
                    '数据ID': data_id,
                    '分子ID': mol_idx,
                    'SMILES': smiles,
                    'Vina_Dock_亲和力': vina_dock_affinity,
                    'Vina_Dock_RMSD下界': vina_dock_rmsd_lb,
                    'Vina_Dock_RMSD上界': vina_dock_rmsd_ub,
                    'Vina_ScoreOnly_亲和力': vina_score_only_affinity,
                    'Vina_Minimize_亲和力': vina_minimize_affinity,
                    'QED评分': qed,
                    'SA评分': sa,
                    '原子类型分布JSD': atom_type_jsd,
                    '原始PT文件': os.path.basename(str(pt_file)),
                    '配体文件': eval_data.get('ligand_filename', 'N/A'),
                    '原子编码模式': eval_data.get('atom_mode', 'N/A'),
                    '对接强度': eval_data.get('exhaustiveness', 'N/A'),
                }
                
                # 添加键长分布JSD
                bond_length_jsd = result.get('bond_length_jsd', {})
                if bond_length_jsd:
                    for key, value in bond_length_jsd.items():
                        record[f'键长JSD_{key}'] = value if value is not None else 'N/A'
                
                # 添加原子对距离分布JSD
                pair_length_jsd = result.get('pair_length_jsd', {})
                if pair_length_jsd:
                    for key, value in pair_length_jsd.items():
                        record[f'原子对距离JSD_{key}'] = value if value is not None else 'N/A'
                
                molecule_records.append(record)
                
        except Exception as e:
            print(f"⚠️  读取评估结果文件失败 {latest_eval_file}: {e}")
            continue
    
    # 计算统计信息
    summary_stats = {
        'batch启动时间': datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        '应生成分子数': total_num_samples,
        '可重建分子数': total_n_reconstruct_success,
        '对接成功分子数': len(molecule_records),  # 实际记录到Excel的对接成功分子数
    }
    
    # 计算各数据均值（只统计数值型数据）
    if molecule_records:
        # Vina得分均值
        vina_dock_scores = [r['Vina_Dock_亲和力'] for r in molecule_records 
                           if r['Vina_Dock_亲和力'] != 'N/A' and isinstance(r['Vina_Dock_亲和力'], (int, float))]
        vina_score_only_scores = [r['Vina_ScoreOnly_亲和力'] for r in molecule_records 
                                 if r['Vina_ScoreOnly_亲和力'] != 'N/A' and isinstance(r['Vina_ScoreOnly_亲和力'], (int, float))]
        vina_minimize_scores = [r['Vina_Minimize_亲和力'] for r in molecule_records 
                               if r['Vina_Minimize_亲和力'] != 'N/A' and isinstance(r['Vina_Minimize_亲和力'], (int, float))]
        
        if vina_dock_scores:
            summary_stats['Vina_Dock_平均亲和力'] = np.mean(vina_dock_scores)
        if vina_score_only_scores:
            summary_stats['Vina_ScoreOnly_平均亲和力'] = np.mean(vina_score_only_scores)
        if vina_minimize_scores:
            summary_stats['Vina_Minimize_平均亲和力'] = np.mean(vina_minimize_scores)
        
        # QED和SA均值
        qed_values = [r['QED评分'] for r in molecule_records 
                     if r['QED评分'] != 'N/A' and isinstance(r['QED评分'], (int, float))]
        sa_values = [r['SA评分'] for r in molecule_records 
                    if r['SA评分'] != 'N/A' and isinstance(r['SA评分'], (int, float))]
        
        if qed_values:
            summary_stats['QED平均评分'] = np.mean(qed_values)
        if sa_values:
            summary_stats['SA平均评分'] = np.mean(sa_values)
    
    return molecule_records, summary_stats


def save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time):
    """
    将所有对接成功的分子数据保存到Excel（照抄自 batch_sample_all.py）
    
    Args:
        excel_file: Excel文件路径
        molecule_records: 分子记录列表
        summary_stats: 统计信息字典
        batch_start_time: batch启动时间
    """
    if pd is None:
        print(f'⚠️  pandas未安装，无法保存Excel')
        return False
    
    try:
        # 确保excel_file是Path对象
        if not isinstance(excel_file, Path):
            excel_file = Path(excel_file)
        
        excel_file.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 保存每个分子的详细数据
            if molecule_records:
                df_molecules = pd.DataFrame(molecule_records)
                # 按Vina_Dock_亲和力排序（从小到大）
                if 'Vina_Dock_亲和力' in df_molecules.columns:
                    df_molecules['Vina_Dock_亲和力_temp'] = df_molecules['Vina_Dock_亲和力'].replace('N/A', np.nan)
                    df_molecules = df_molecules.sort_values('Vina_Dock_亲和力_temp', na_position='last')
                    df_molecules = df_molecules.drop(columns=['Vina_Dock_亲和力_temp'])
                df_molecules.to_excel(writer, sheet_name='分子评估数据', index=False)
            else:
                # 如果没有数据，创建空DataFrame
                df_molecules = pd.DataFrame()
                df_molecules.to_excel(writer, sheet_name='分子评估数据', index=False)
            
            # 保存统计信息
            stats_items = []
            stats_values = []
            
            for key, value in summary_stats.items():
                stats_items.append(key)
                if isinstance(value, float):
                    stats_values.append(f"{value:.3f}")
                else:
                    stats_values.append(str(value))
            
            # 注意：对接成功分子数已经在summary_stats中，不需要重复添加
            
            stats_df = pd.DataFrame({
                '统计项目': stats_items,
                '数值': stats_values
            })
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
        
        return True
        
    except Exception as e:
        print(f"⚠️  保存Excel失败: {e}")
        traceback.print_exc()
        return False


def run_single_evaluation(pt_file, protein_root, data_id, atom_mode='add_aromatic', exhaustiveness=8):
    """
    执行单个评估任务
    
    Args:
        pt_file: .pt文件路径
        protein_root: 蛋白质数据根目录
        data_id: 数据ID（用于生成评估输出目录名）
        atom_mode: 原子模式（默认：add_aromatic）
        exhaustiveness: Vina对接强度（默认：8）
    
    Returns:
        tuple: (success, message)
    """
    print(f"\n{'='*60}")
    print(f"开始评估: {pt_file.name}")
    print(f"{'='*60}")
    
    # 生成评估输出目录（放在outputs目录下，与batch_sample_all.py一致）
    pt_path = Path(pt_file)
    outputs_dir = pt_path.parent  # outputs目录
    
    # 生成评估目录名（简化版本，不使用配置文件）
    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 执行评估
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # 实时显示输出
            text=True
        )
        
        print(f"✅ 评估成功")
        return True, "评估成功"
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")
        return False, f"评估失败: {e}"
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        return False, f"评估出错: {e}"


def main():
    parser = argparse.ArgumentParser(description='批量采样和评估脚本（串行执行）')
    
    # 采样参数
    parser.add_argument('--start', type=int, default=0,
                       help='起始 data_id（默认: 0）')
    parser.add_argument('--end', type=int, default=99,
                       help='结束 data_id（默认: 99）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/sampling.yml）')
    
    # 评估参数
    # 尝试从环境变量或默认路径获取
    default_protein_root = os.environ.get('PROTEIN_ROOT', None)
    if default_protein_root is None:
        # 尝试常见的默认路径
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
    
    # 其他参数
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的.pt文件（不重新采样）')
    parser.add_argument('--excel_file', type=str, default=None,
                       help='Excel记录文件路径（默认: batch_evaluation_summary_{timestamp}.xlsx）')
    
    args = parser.parse_args()
    
    # 设置默认值
    if args.config is None:
        args.config = CONFIG
    else:
        args.config = Path(args.config)
    
    # 验证protein_root参数
    if args.protein_root is None:
        print(f"❌ 错误: 未指定蛋白质数据根目录（--protein_root）")
        print(f"   请使用 --protein_root 参数指定蛋白质数据目录")
        print(f"   示例: --protein_root /path/to/data/crossdocked_v1.1_rmsd1.0_pocket10")
        sys.exit(1)
    
    args.protein_root = Path(args.protein_root)
    
    # 验证路径
    if not args.config.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not args.protein_root.exists():
        print(f"❌ 错误: 蛋白质数据根目录不存在: {args.protein_root}")
        print(f"   请检查路径是否正确，或使用 --protein_root 指定正确的路径")
        sys.exit(1)
    
    if not SCRIPT.exists():
        print(f"❌ 错误: 采样脚本不存在: {SCRIPT}")
        sys.exit(1)
    
    if not EVAL_SCRIPT.exists():
        print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
        sys.exit(1)
    
    # 确保outputs目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 确保batchsummary目录存在
    BATCHSUMMARY_DIR = REPO_ROOT / 'batchsummary'
    BATCHSUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    # 设置Excel文件路径（如果启用评估，添加时间戳）
    batch_start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.excel_file:
        excel_file = Path(args.excel_file)
    else:
        excel_file = BATCHSUMMARY_DIR / f'batch_evaluation_summary_{timestamp}.xlsx'
    
    # 打印配置信息
    print(f"\n{'='*60}")
    print(f"批量采样和评估配置")
    print(f"{'='*60}")
    print(f"数据ID范围: {args.start} 到 {args.end}")
    print(f"配置文件: {args.config}")
    print(f"蛋白质数据根目录: {args.protein_root}")
    print(f"评估结果保存位置: {OUTPUT_DIR} (自动生成)")
    print(f"原子模式: {args.atom_mode}")
    print(f"对接强度: {args.exhaustiveness}")
    print(f"跳过已存在: {args.skip_existing}")
    print(f"Excel记录文件: {excel_file}")
    print(f"{'='*60}\n")
    
    # 统计信息
    total = args.end - args.start + 1
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 存储所有结果用于最后收集分子数据
    all_results = []
    
    start_time = time.time()
    
    # 串行执行每个 data_id
    for data_id in range(args.start, args.end + 1):
        print(f"\n{'#'*60}")
        print(f"处理 data_id={data_id} ({data_id - args.start + 1}/{total})")
        print(f"{'#'*60}")
        
        # 记录开始时间
        task_start_time = time.time()
        
        # 检查是否已存在
        if args.skip_existing:
            pt_file = find_latest_result_file(data_id)
            if pt_file and pt_file.exists():
                print(f"⏭️  跳过已存在的文件: {pt_file}")
                skip_count += 1
                # 直接进行评估
                eval_success, eval_msg = run_single_evaluation(
                    pt_file, args.protein_root, data_id,
                    args.atom_mode, args.exhaustiveness
                )
                
                # 读取评估结果并记录到Excel
                if eval_success:
                    success_count += 1
                    # 等待评估结果文件生成
                    time.sleep(2)
                    eval_success_read, vina_mean, vina_median, num_scores, eval_message, _ = read_evaluation_results(
                        pt_file, data_id, wait_timeout=60
                    )
                    if eval_success_read:
                        task_time = time.time() - task_start_time
                        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        append_to_excel(
                            excel_file, timestamp_str, task_time, data_id, pt_file,
                            vina_mean, vina_median, num_scores, '成功', eval_message
                        )
                    # 获取评估输出目录路径
                    eval_output_dir = None
                    if eval_success_read:
                        _, _, _, _, _, eval_output_dir = read_evaluation_results(
                            pt_file, data_id, wait_timeout=5
                        )
                    # 确保 eval_output_dir 是字符串格式（如果是 Path 对象则转换）
                    eval_output_dir_str = str(eval_output_dir) if eval_output_dir else None
                    all_results.append((data_id, True, eval_msg, None, pt_file, eval_output_dir_str))
                else:
                    fail_count += 1
                    task_time = time.time() - task_start_time
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    append_to_excel(
                        excel_file, timestamp_str, task_time, data_id, pt_file,
                        None, None, 0, '失败', eval_msg
                    )
                    all_results.append((data_id, False, eval_msg, None, pt_file, None))
                continue
        
        # 执行采样
        sample_success, pt_file, sample_msg = run_single_sample(data_id, args.config)
        
        if not sample_success or pt_file is None:
            print(f"❌ 采样失败，跳过评估")
            fail_count += 1
            task_time = time.time() - task_start_time
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            append_to_excel(
                excel_file, timestamp_str, task_time, data_id, None,
                None, None, 0, '失败', f"采样失败: {sample_msg}"
            )
            all_results.append((data_id, False, sample_msg, None, None, None))
            continue
        
        # 执行评估
        eval_success, eval_msg = run_single_evaluation(
            pt_file, args.protein_root, data_id,
            args.atom_mode, args.exhaustiveness
        )
        
        # 读取评估结果并记录到Excel
        task_time = time.time() - task_start_time
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if eval_success:
            success_count += 1
            # 等待评估结果文件生成
            time.sleep(2)
            eval_success_read, vina_mean, vina_median, num_scores, eval_message, eval_output_dir = read_evaluation_results(
                pt_file, data_id, wait_timeout=60
            )
            if eval_success_read:
                append_to_excel(
                    excel_file, timestamp_str, task_time, data_id, pt_file,
                    vina_mean, vina_median, num_scores, '成功', eval_message
                )
            else:
                append_to_excel(
                    excel_file, timestamp_str, task_time, data_id, pt_file,
                    None, None, 0, '部分成功', f"评估完成但读取结果失败: {eval_message}"
                )
            # 确保 eval_output_dir 是字符串格式（如果是 Path 对象则转换）
            eval_output_dir_str = str(eval_output_dir) if eval_output_dir else None
            all_results.append((data_id, True, eval_msg, None, pt_file, eval_output_dir_str))
        else:
            fail_count += 1
            append_to_excel(
                excel_file, timestamp_str, task_time, data_id, pt_file,
                None, None, 0, '失败', eval_msg
            )
            all_results.append((data_id, False, eval_msg, None, pt_file, None))
    
    # 批量保存所有结果到Excel（读取evaluate_pt_with_correct_reconstruct.py的评估结果）
    if excel_file:
        print(f"\n{'='*70}")
        print(f"收集并保存评估结果到Excel...")
        print(f"{'='*70}")
        print(f"batch_start_time: {datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"all_results 数量: {len(all_results)}")
        
        # 收集所有对接成功的分子数据
        molecule_records, summary_stats = collect_all_evaluation_results(all_results, batch_start_time)
        
        print(f"收集到的分子记录数: {len(molecule_records)}")
        
        # 如果收集到的记录数为0，打印详细信息帮助调试
        if len(molecule_records) == 0 and len(all_results) > 0:
            print(f"\n⚠️  警告: 未收集到任何分子记录，可能的原因：")
            print(f"   1. 评估结果文件的时间早于 batch_start_time")
            print(f"   2. 评估结果文件中没有对接成功的分子")
            print(f"   3. 评估目录路径不正确")
            print(f"\n   调试信息（前5个结果）：")
            for i, r in enumerate(all_results[:5]):
                if len(r) >= 6:
                    data_id, success, message, log_file, pt_file, eval_output_dir = r[:6]
                    print(f"     [{i+1}] data_id={data_id}, success={success}")
                    print(f"         pt_file={pt_file}")
                    print(f"         eval_output_dir={eval_output_dir}")
                elif len(r) >= 5:
                    data_id, success, message, log_file, pt_file = r[:5]
                    print(f"     [{i+1}] data_id={data_id}, success={success}, pt_file={pt_file}")
        
        # 保存到Excel
        if save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time):
            print(f"✅ 成功保存 {len(molecule_records)} 个对接成功分子到Excel: {excel_file}")
            print(f"   统计信息:")
            print(f"     - 应生成分子数: {summary_stats.get('应生成分子数', 0)}")
            print(f"     - 可重建分子数: {summary_stats.get('可重建分子数', 0)}")
            print(f"     - 对接成功分子数: {summary_stats.get('对接成功分子数', 0)}")
            if 'Vina_Dock_平均亲和力' in summary_stats:
                print(f"     - Vina_Dock_平均亲和力: {summary_stats['Vina_Dock_平均亲和力']:.3f} kcal/mol")
        else:
            print(f"⚠️  Excel保存失败")
        print(f"{'='*70}\n")
    
    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"总计: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skip_count}")
    print(f"耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    if excel_file:
        print(f"📊 详细记录已保存至: {excel_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

