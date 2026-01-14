#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗并合并batch evaluation summary Excel文件

功能：
1. 扫描指定文件夹内的所有Excel文件
2. 清洗数据：剔除 vinadock>0 或 vinascore>0 或 vinamin>0 的异常数据
3. 重新统计清洗后的数据
4. 合并所有文件的结果到 merged_summary.xlsx
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
    
    示例文件名: batch_evaluation_summary_20260108_001301_gfquadratic_1.0_0.0_tl750_lslambda_80p0_20p0_lsstep_1p0_lsnoise_0p0_rflambda_20p0_2p0_rfstep_0p2_rfnoise_0p08.xlsx
    """
    params = {}
    
    # 提取权重策略 (gfquadratic -> quadratic)
    gf_match = re.search(r'gf(\w+)', filename)
    if gf_match:
        params['权重策略'] = gf_match.group(1)
    
    # 提取开始权重和结束权重 (gfquadratic_1.0_0.0 -> 开始权重=1.0, 结束权重=0.0)
    weight_match = re.search(r'gf\w+_(\d+p?\d*)_(\d+p?\d*)', filename)
    if weight_match:
        start_w = weight_match.group(1).replace('p', '.')
        end_w = weight_match.group(2).replace('p', '.')
        params['开始权重'] = float(start_w)
        params['结束权重'] = float(end_w)
    
    # 提取时间长度 (tl750 -> 750)
    tl_match = re.search(r'tl(\d+)', filename)
    if tl_match:
        params['时间长度 (TL)'] = int(tl_match.group(1))
    
    # 提取LS Lambda值 (lslambda_80p0_20p0 -> LSLambda1=80.0, LSLambda2=20.0)
    ls_lambda_match = re.search(r'lslambda_(\d+p\d+)_(\d+p\d+)', filename)
    if ls_lambda_match:
        params['LSLambda1'] = float(ls_lambda_match.group(1).replace('p', '.'))
        params['LSLambda2'] = float(ls_lambda_match.group(2).replace('p', '.'))
    
    # 提取LS step size (lsstep_1p0 -> 1.0)
    ls_step_match = re.search(r'lsstep_(\d+p\d+)', filename)
    if ls_step_match:
        params['LSstepsize'] = float(ls_step_match.group(1).replace('p', '.'))
    
    # 提取LS noise (lsnoise_0p0 -> 0.0)
    ls_noise_match = re.search(r'lsnoise_(\d+p\d+)', filename)
    if ls_noise_match:
        params['LSnosie'] = float(ls_noise_match.group(1).replace('p', '.'))
    
    # 提取RF Lambda值 (rflambda_20p0_2p0 -> RFLambda1=20.0, RFLambda2=2.0)
    rf_lambda_match = re.search(r'rflambda_(\d+p\d+)_(\d+p\d+)', filename)
    if rf_lambda_match:
        params['RFLambda1'] = float(rf_lambda_match.group(1).replace('p', '.'))
        params['RFLambda2'] = float(rf_lambda_match.group(2).replace('p', '.'))
    
    # 提取RF step size (rfstep_0p2 -> 0.2)
    rf_step_match = re.search(r'rfstep_(\d+p\d+)', filename)
    if rf_step_match:
        params['RFstepsize'] = float(rf_step_match.group(1).replace('p', '.'))
    
    # 提取RF noise (rfnoise_0p08 -> 0.08)
    rf_noise_match = re.search(r'rfnoise_(\d+p\d+)', filename)
    if rf_noise_match:
        params['RFnosie'] = float(rf_noise_match.group(1).replace('p', '.'))
    
    return params

def find_vina_columns(df):
    """
    查找Vina相关的列名（处理可能的编码问题）
    """
    vina_dock_col = None
    vina_score_col = None
    vina_min_col = None
    
    for col in df.columns:
        col_str = str(col)
        if 'Vina_Dock' in col_str and ('亲和' in col_str or 'affinity' in col_str.lower()):
            vina_dock_col = col
        elif 'Vina_ScoreOnly' in col_str and ('亲和' in col_str or 'affinity' in col_str.lower()):
            vina_score_col = col
        elif 'Vina_Minimize' in col_str and ('亲和' in col_str or 'affinity' in col_str.lower()):
            vina_min_col = col
    
    return vina_dock_col, vina_score_col, vina_min_col

def clean_and_recalculate_stats(excel_path):
    """
    清洗Excel文件中的数据并重新计算统计信息
    """
    try:
        # 读取详细数据sheet（第一个sheet）
        xl_file = pd.ExcelFile(excel_path, engine='openpyxl')
        sheet_names = xl_file.sheet_names
        
        # 第一个sheet通常是详细数据
        df = pd.read_excel(excel_path, sheet_name=sheet_names[0], engine='openpyxl')
        
        if df.empty:
            print(f"  文件 {excel_path.name} 的详细数据为空")
            return None
        
        # 查找Vina相关列
        vina_dock_col, vina_score_col, vina_min_col = find_vina_columns(df)
        
        if not vina_dock_col or not vina_score_col or not vina_min_col:
            print(f"  文件 {excel_path.name} 中未找到Vina相关列")
            print(f"    找到的列: dock={vina_dock_col}, score={vina_score_col}, min={vina_min_col}")
            return None
        
        # 记录清洗前的数据量
        original_count = len(df)
        
        # 清洗数据：剔除 vinadock>0 或 vinascore>0 或 vinamin>0 的数据
        # 将NaN值视为正常数据（不剔除）
        mask = (
            (pd.isna(df[vina_dock_col]) | (df[vina_dock_col] <= 0)) &
            (pd.isna(df[vina_score_col]) | (df[vina_score_col] <= 0)) &
            (pd.isna(df[vina_min_col]) | (df[vina_min_col] <= 0))
        )
        
        df_cleaned = df[mask].copy()
        removed_count = original_count - len(df_cleaned)
        valid_molecule_count = len(df_cleaned)  # 有效分子数量
        
        if removed_count > 0:
            print(f"  清洗: 原始 {original_count} 条，剔除 {removed_count} 条异常数据，剩余 {len(df_cleaned)} 条")
        
        # 从原始Excel的"统计信息"sheet中读取原始的重建成功率和对接成功率
        original_reconstruct_rate = None
        original_dock_rate = None
        expected_molecule_count = None  # 应生成分子数
        try:
            # 查找统计信息sheet（通常是第二个sheet，索引为1）
            stats_sheet_idx = None
            for idx, name in enumerate(sheet_names):
                name_str = str(name)
                if '统计' in name_str or 'stats' in name_str.lower() or '统计信息' in name_str:
                    stats_sheet_idx = idx
                    break
            
            if stats_sheet_idx is not None:
                df_stats = pd.read_excel(excel_path, sheet_name=stats_sheet_idx, engine='openpyxl')
                if not df_stats.empty and len(df_stats.columns) >= 2:
                    # 尝试构建字典，第一列是统计项目，第二列是数值
                    stats_dict = {}
                    for _, row in df_stats.iterrows():
                        key = str(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                        value = row.iloc[1] if len(row) > 1 else None
                        if key:
                            stats_dict[key] = value
                    
                    # 查找重建成功率（优先查找包含百分比的项）
                    for key in stats_dict.keys():
                        key_str = str(key)
                        if '重建' in key_str and ('成功' in key_str or '百分比' in key_str or '%' in key_str):
                            try:
                                val = stats_dict[key]
                                # 如果是字符串且包含%，提取数字
                                if isinstance(val, str) and '%' in val:
                                    val = val.replace('%', '').strip()
                                val_float = float(val)
                                # 如果值在0-100之间，认为是百分比；如果>100，可能是总数，需要转换
                                if val_float > 100 and original_count > 0:
                                    val_float = (val_float / original_count) * 100
                                original_reconstruct_rate = val_float
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    # 查找对接成功率 - 明确查找"对接成功百分比(%)"
                    for key in stats_dict.keys():
                        key_str = str(key)
                        # 优先查找完全匹配"对接成功百分比(%)"
                        if '对接成功百分比' in key_str or '对接成功百分比(%)' in key_str:
                            try:
                                val = stats_dict[key]
                                # 如果是字符串且包含%，提取数字
                                if isinstance(val, str) and '%' in val:
                                    val = val.replace('%', '').strip()
                                val_float = float(val)
                                # 如果值在0-100之间，认为是百分比；如果>100，可能是总数，需要转换
                                if val_float > 100 and original_count > 0:
                                    val_float = (val_float / original_count) * 100
                                original_dock_rate = val_float
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    # 如果没找到完全匹配的，再查找包含"对接"和"成功"和"百分比"的项
                    if original_dock_rate is None:
                        for key in stats_dict.keys():
                            key_str = str(key)
                            if '对接' in key_str and '成功' in key_str and ('百分比' in key_str or '%' in key_str):
                                try:
                                    val = stats_dict[key]
                                    # 如果是字符串且包含%，提取数字
                                    if isinstance(val, str) and '%' in val:
                                        val = val.replace('%', '').strip()
                                    val_float = float(val)
                                    # 如果值在0-100之间，认为是百分比；如果>100，可能是总数，需要转换
                                    if val_float > 100 and original_count > 0:
                                        val_float = (val_float / original_count) * 100
                                    original_dock_rate = val_float
                                    break
                                except (ValueError, TypeError):
                                    pass
                    
                    # 查找应生成分子数
                    for key in stats_dict.keys():
                        key_str = str(key)
                        if '应生成' in key_str and ('分子' in key_str or '数量' in key_str or '数' in key_str):
                            try:
                                val = stats_dict[key]
                                if isinstance(val, (int, float)):
                                    expected_molecule_count = float(val)
                                    break
                                elif isinstance(val, str):
                                    # 尝试提取数字
                                    import re
                                    numbers = re.findall(r'\d+', val)
                                    if numbers:
                                        expected_molecule_count = float(numbers[0])
                                        break
                            except (ValueError, TypeError):
                                pass
        except Exception as e:
            print(f"  读取原始统计信息时出错: {e}")
        
        # 重新计算统计信息
        stats = {}
        
        # 基本统计
        stats['总样本数'] = len(df_cleaned)
        stats['有效分子数量'] = valid_molecule_count
        
        # 计算有效分子比例：有效分子数量 / 应生成分子数
        if expected_molecule_count is not None and expected_molecule_count > 0:
            valid_molecule_ratio = (valid_molecule_count / expected_molecule_count * 100)
        else:
            # 如果没有找到应生成分子数，则使用原始数据量作为分母
            valid_molecule_ratio = (valid_molecule_count / original_count * 100) if original_count > 0 else 0
        
        stats['有效分子比例 (%)'] = valid_molecule_ratio
        stats['原始重建成功率 (%)'] = original_reconstruct_rate if original_reconstruct_rate is not None else None
        stats['原始对接成功率 (%)'] = original_dock_rate if original_dock_rate is not None else None
        
        # 重建成功率（需要找到重建相关的列）
        reconstruct_col = None
        for col in df_cleaned.columns:
            if '重建' in str(col) or 'reconstruct' in str(col).lower():
                reconstruct_col = col
                break
        
        if reconstruct_col:
            reconstruct_success = df_cleaned[reconstruct_col].notna().sum() if reconstruct_col else 0
            stats['重建成功数'] = reconstruct_success
            stats['可重建率 (%)'] = (reconstruct_success / len(df_cleaned) * 100) if len(df_cleaned) > 0 else 0
        else:
            stats['重建成功数'] = 0
            stats['可重建率 (%)'] = 0
        
        # 对接成功率（Vina_Dock有值且<=0视为成功）
        dock_success = df_cleaned[vina_dock_col].notna().sum()
        stats['对接成功数'] = dock_success
        stats['对接成功率 (%)'] = (dock_success / len(df_cleaned) * 100) if len(df_cleaned) > 0 else 0
        
        # Vina评分平均值（只计算有效值，即<=0的值）
        stats['Vina_Dock 亲和力'] = df_cleaned[vina_dock_col][df_cleaned[vina_dock_col] <= 0].mean() if dock_success > 0 else 0
        stats['Vina_ScoreOnly'] = df_cleaned[vina_score_col][df_cleaned[vina_score_col] <= 0].mean() if df_cleaned[vina_score_col].notna().sum() > 0 else 0
        stats['Vina_Minimize'] = df_cleaned[vina_min_col][df_cleaned[vina_min_col] <= 0].mean() if df_cleaned[vina_min_col].notna().sum() > 0 else 0
        
        # QED和SA评分（需要找到相关列）
        qed_col = None
        sa_col = None
        for col in df_cleaned.columns:
            if 'QED' in str(col) and '评分' in str(col):
                qed_col = col
            elif 'SA' in str(col) and '评分' in str(col):
                sa_col = col
        
        if qed_col:
            stats['QED 评分（均值）'] = df_cleaned[qed_col].mean()
        else:
            stats['QED 评分（均值）'] = 0
        
        if sa_col:
            stats['SA 评分（均值）'] = df_cleaned[sa_col].mean()
        else:
            stats['SA 评分（均值）'] = 0
        
        # 从配置参数sheet中提取其他参数
        config_dict = {}
        try:
            config_sheet_name = None
            for name in sheet_names:
                if '配置' in str(name) or 'config' in str(name).lower():
                    config_sheet_name = name
                    break
            
            if config_sheet_name:
                df_config = pd.read_excel(excel_path, sheet_name=config_sheet_name, engine='openpyxl')
                if not df_config.empty:
                    # 尝试不同的列名格式
                    if len(df_config.columns) >= 2:
                        config_dict = dict(zip(df_config.iloc[:, 0], df_config.iloc[:, 1]))
        except Exception as e:
            pass
        
        # 提取配置参数
        stats['下降速率'] = config_dict.get('model.grad_fusion_lambda.power', None)
        stats['步数'] = config_dict.get('计算.跳步总次数', None)
        stats['取模步长'] = config_dict.get('计算.实际长度', None)
        
        return stats
        
    except Exception as e:
        print(f"  处理文件 {excel_path.name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_and_merge_summaries(input_dir, output_file=None):
    """
    清洗并合并所有Excel文件的数据
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"目录 {input_dir} 不存在")
        return
    
    # 查找所有Excel文件（排除临时文件和输出文件）
    excel_files = [f for f in input_path.glob('*.xlsx') 
                   if not f.name.startswith('~$') and f.name != 'merged_summary.xlsx']
    excel_files.sort()
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    # 定义列的顺序
    columns_order = [
        '文件名', '权重策略', '下降速率', '开始权重', '结束权重', '时间长度 (TL)',
        'LSstepsize', 'LSnosie', 'LSLambda1', 'LSLambda2',
        'RFstepsize', 'RFnosie', 'RFLambda1', 'RFLambda2',
        '步数', '取模步长', '有效分子数量', '有效分子比例 (%)',
        '原始重建成功率 (%)', '原始对接成功率 (%)',
        '可重建率 (%)', '对接成功率 (%)',
        'Vina_Dock 亲和力', 'Vina_ScoreOnly', 'Vina_Minimize',
        'QED 评分（均值）', 'SA 评分（均值）'
    ]
    
    all_data = []
    
    for excel_file in excel_files:
        print(f"\n处理文件: {excel_file.name}")
        
        # 从文件名解析参数
        filename_params = parse_filename_params(excel_file.name)
        
        # 清洗数据并重新计算统计信息
        cleaned_stats = clean_and_recalculate_stats(excel_file)
        
        if cleaned_stats is None:
            continue
        
        # 合并文件名参数和清洗后的统计数据（文件名参数优先）
        combined_data = {**cleaned_stats, **filename_params}
        combined_data['文件名'] = excel_file.name
        
        # 确保所有列都存在
        row_data = {}
        for col in columns_order:
            row_data[col] = combined_data.get(col, None)
        
        all_data.append(row_data)
    
    if not all_data:
        print("\n没有成功处理任何文件")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(all_data, columns=columns_order)
    
    # 生成输出文件名
    if output_file is None:
        output_file = input_path / 'merged_summary.xlsx'
    else:
        output_file = Path(output_file)
    
    # 保存到Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n清洗和合并完成！共 {len(all_data)} 条记录")
    print(f"输出文件: {output_file}")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='清洗并合并batch evaluation summary Excel文件')
    parser.add_argument('--input_dir', type=str, 
                        default=r'F:\DiffDynamic\DiffDynamic\20250113',
                        help='输入目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（默认: input_dir/merged_summary.xlsx）')
    
    args = parser.parse_args()
    
    clean_and_merge_summaries(args.input_dir, args.output)

