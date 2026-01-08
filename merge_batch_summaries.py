#!/usr/bin/env python3
"""
合并batchsummary目录下的所有Excel文件，提取指定格式的数据

功能：
1. 扫描batchsummary目录下的所有Excel文件
2. 从文件名解析参数（权重策略、时间长度、Lambda值等）
3. 从Excel的"统计信息"和"配置参数"sheet中提取数据
4. 按照指定格式整理并保存到新的Excel文件
"""

import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def parse_filename_params(filename):
    """
    从文件名解析参数
    
    示例文件名: batch_evaluation_summary_20260105_002424_gfquadratic_1_0_tl800_lslambda_60p0_20p0_lsstep_0p6_lsnoise_0p0_rflambda_10p0_5p0_rfstep_0p25_rfnoise_0p05.xlsx
    
    返回参数字典
    """
    params = {}
    
    # 提取权重策略 (gfquadratic -> quadratic)
    gf_match = re.search(r'gf(\w+)', filename)
    if gf_match:
        params['权重策略'] = gf_match.group(1)
    
    # 提取开始权重和结束权重 (gfquadratic_1_0 -> 开始权重=1, 结束权重=0)
    weight_match = re.search(r'gf\w+_(\d+)_(\d+)', filename)
    if weight_match:
        params['开始权重'] = float(weight_match.group(1))
        params['结束权重'] = float(weight_match.group(2))
    
    # 提取下降速率 (从配置参数中获取，如果文件名中没有)
    # 默认从配置参数中获取
    
    # 提取时间长度 (tl800 -> 800)
    tl_match = re.search(r'tl(\d+)', filename)
    if tl_match:
        params['时间长度 (TL)'] = int(tl_match.group(1))
    
    # 提取LS Lambda值 (lslambda_60p0_20p0 -> LSLambda1=60.0, LSLambda2=20.0)
    ls_lambda_match = re.search(r'lslambda_(\d+p\d+)_(\d+p\d+)', filename)
    if ls_lambda_match:
        params['LSLambda1'] = float(ls_lambda_match.group(1).replace('p', '.'))
        params['LSLambda2'] = float(ls_lambda_match.group(2).replace('p', '.'))
    
    # 提取LS step size (lsstep_0p6 -> 0.6)
    ls_step_match = re.search(r'lsstep_(\d+p\d+)', filename)
    if ls_step_match:
        params['LSstepsize'] = float(ls_step_match.group(1).replace('p', '.'))
    
    # 提取LS noise (lsnoise_0p0 -> 0.0)
    ls_noise_match = re.search(r'lsnoise_(\d+p\d+)', filename)
    if ls_noise_match:
        params['LSnosie'] = float(ls_noise_match.group(1).replace('p', '.'))
    
    # 提取RF Lambda值 (rflambda_10p0_5p0 -> RFLambda1=10.0, RFLambda2=5.0)
    rf_lambda_match = re.search(r'rflambda_(\d+p\d+)_(\d+p\d+)', filename)
    if rf_lambda_match:
        params['RFLambda1'] = float(rf_lambda_match.group(1).replace('p', '.'))
        params['RFLambda2'] = float(rf_lambda_match.group(2).replace('p', '.'))
    
    # 提取RF step size (rfstep_0p25 -> 0.25)
    rf_step_match = re.search(r'rfstep_(\d+p\d+)', filename)
    if rf_step_match:
        params['RFstepsize'] = float(rf_step_match.group(1).replace('p', '.'))
    
    # 提取RF noise (rfnoise_0p05 -> 0.05)
    rf_noise_match = re.search(r'rfnoise_(\d+p\d+)', filename)
    if rf_noise_match:
        params['RFnosie'] = float(rf_noise_match.group(1).replace('p', '.'))
    
    return params

def extract_stats_from_excel(excel_path):
    """
    从Excel文件中提取统计信息和配置参数
    """
    try:
        # 读取统计信息sheet
        df_stats = pd.read_excel(excel_path, sheet_name='统计信息', engine='openpyxl')
        stats_dict = dict(zip(df_stats['统计项目'], df_stats['数值']))
        
        # 尝试读取配置参数sheet（可能不存在）
        config_dict = {}
        try:
            df_config = pd.read_excel(excel_path, sheet_name='配置参数', engine='openpyxl')
            config_dict = dict(zip(df_config['参数路径'], df_config['参数值']))
        except Exception:
            # 旧版本文件可能没有配置参数sheet，跳过
            pass
        
        # 提取所需的数据
        result = {}
        
        # 从统计信息中提取
        result['可重建率 (%)'] = float(stats_dict.get('重建成功百分比(%)', 0))
        result['对接成功率 (%)'] = float(stats_dict.get('对接成功百分比(%)', 0))
        result['Vina_Dock 亲和力'] = float(stats_dict.get('Vina_Dock_平均亲和力', 0))
        result['Vina_ScoreOnly'] = float(stats_dict.get('Vina_ScoreOnly_平均亲和力', 0))
        result['Vina_Minimize'] = float(stats_dict.get('Vina_Minimize_平均亲和力', 0))
        result['QED 评分（均值）'] = float(stats_dict.get('QED平均评分', 0))
        result['SA 评分（均值）'] = float(stats_dict.get('SA平均评分', 0))
        
        # 从配置参数中提取
        # 下降速率 (power)
        power = config_dict.get('model.grad_fusion_lambda.power', None)
        if power is not None:
            result['下降速率'] = float(power)
        
        # 步数 (计算.跳步总次数)
        steps = config_dict.get('计算.跳步总次数', None)
        if steps is not None:
            result['步数'] = int(steps)
        
        # 取模步长 (计算.实际长度)
        mod_step = config_dict.get('计算.实际长度', None)
        if mod_step is not None:
            result['取模步长'] = float(mod_step)
        
        # 如果文件名中没有提取到某些参数，尝试从配置参数中获取
        if 'LSstepsize' not in result:
            ls_step = config_dict.get('sample.dynamic.large_step.step_size', None)
            if ls_step is not None:
                result['LSstepsize'] = float(ls_step)
        
        if 'LSnosie' not in result:
            ls_noise = config_dict.get('sample.dynamic.large_step.noise_scale', None)
            if ls_noise is not None:
                result['LSnosie'] = float(ls_noise)
        
        if 'LSLambda1' not in result:
            ls_lambda_a = config_dict.get('sample.dynamic.large_step.lambda_coeff_a', None)
            if ls_lambda_a is not None:
                result['LSLambda1'] = float(ls_lambda_a)
        
        if 'LSLambda2' not in result:
            ls_lambda_b = config_dict.get('sample.dynamic.large_step.lambda_coeff_b', None)
            if ls_lambda_b is not None:
                result['LSLambda2'] = float(ls_lambda_b)
        
        if 'RFstepsize' not in result:
            rf_step = config_dict.get('sample.dynamic.refine.step_size', None)
            if rf_step is not None:
                result['RFstepsize'] = float(rf_step)
        
        if 'RFnosie' not in result:
            rf_noise = config_dict.get('sample.dynamic.refine.noise_scale', None)
            if rf_noise is not None:
                result['RFnosie'] = float(rf_noise)
        
        if 'RFLambda1' not in result:
            rf_lambda_a = config_dict.get('sample.dynamic.refine.lambda_coeff_a', None)
            if rf_lambda_a is not None:
                result['RFLambda1'] = float(rf_lambda_a)
        
        if 'RFLambda2' not in result:
            rf_lambda_b = config_dict.get('sample.dynamic.refine.lambda_coeff_b', None)
            if rf_lambda_b is not None:
                result['RFLambda2'] = float(rf_lambda_b)
        
        if '时间长度 (TL)' not in result:
            time_boundary = config_dict.get('sample.dynamic.time_boundary', None)
            if time_boundary is not None:
                result['时间长度 (TL)'] = int(time_boundary)
        
        if '权重策略' not in result:
            mode = config_dict.get('model.grad_fusion_lambda.mode', None)
            if mode is not None:
                result['权重策略'] = str(mode)
        
        if '开始权重' not in result:
            start = config_dict.get('model.grad_fusion_lambda.start', None)
            if start is not None:
                result['开始权重'] = float(start)
        
        if '结束权重' not in result:
            end = config_dict.get('model.grad_fusion_lambda.end', None)
            if end is not None:
                result['结束权重'] = float(end)
        
        return result
        
    except Exception as e:
        print(f"⚠️  读取文件 {excel_path} 时出错: {e}")
        return None

def merge_all_summaries(batchsummary_dir='batchsummary', output_file=None):
    """
    合并所有Excel文件的数据
    """
    batchsummary_path = Path(batchsummary_dir)
    if not batchsummary_path.exists():
        print(f"❌ 目录 {batchsummary_dir} 不存在")
        return
    
    # 查找所有Excel文件
    excel_files = list(batchsummary_path.glob('*.xlsx'))
    excel_files.sort()  # 按文件名排序
    
    print(f"📁 找到 {len(excel_files)} 个Excel文件")
    
    # 定义列的顺序
    columns_order = [
        '权重策略', '下降速率', '开始权重', '结束权重', '时间长度 (TL)',
        'LSstepsize', 'LSnosie', 'LSLambda1', 'LSLambda2',
        'RFstepsize', 'RFnosie', 'RFLambda1', 'RFLambda2',
        '步数', '取模步长', '可重建率 (%)', '对接成功率 (%)',
        'Vina_Dock 亲和力', 'Vina_ScoreOnly', 'Vina_Minimize',
        'QED 评分（均值）', 'SA 评分（均值）'
    ]
    
    all_data = []
    
    for excel_file in excel_files:
        print(f"📖 处理文件: {excel_file.name}")
        
        # 从文件名解析参数
        filename_params = parse_filename_params(excel_file.name)
        
        # 从Excel文件提取统计数据
        excel_data = extract_stats_from_excel(excel_file)
        
        if excel_data is None:
            continue
        
        # 合并文件名参数和Excel数据（文件名参数优先）
        combined_data = {**excel_data, **filename_params}
        
        # 确保所有列都存在
        row_data = {}
        for col in columns_order:
            row_data[col] = combined_data.get(col, None)
        
        all_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(all_data, columns=columns_order)
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = batchsummary_path / f'merged_summary_{timestamp}.xlsx'
    else:
        output_file = Path(output_file)
    
    # 保存到Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n✅ 合并完成！共 {len(all_data)} 条记录")
    print(f"📄 输出文件: {output_file}")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='合并batchsummary目录下的所有Excel文件')
    parser.add_argument('--input_dir', type=str, default='batchsummary',
                        help='输入目录路径（默认: batchsummary）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（默认: batchsummary/merged_summary_YYYYMMDD_HHMMSS.xlsx）')
    
    args = parser.parse_args()
    
    merge_all_summaries(args.input_dir, args.output)

