#!/usr/bin/env python3
"""
批量运行不同采样策略的脚本

功能：
1. 读取策略参数列表（CSV格式）
2. 对每个策略，更新 sampling.yml 配置文件
3. 运行 batch_sampleandeval_parallel.py 命令

使用方法：

1. 运行所有策略（默认行为）：
   python3 run_sampling_strategies.py --start 0 --end 99 --gpus "0" --num_cpu_cores 20

2. 运行单个策略（指定策略索引，从0开始）：
   python3 run_sampling_strategies.py --strategy-index 0 --start 0 --end 9 --gpus "0" --num_cpu_cores 20
   python3 run_sampling_strategies.py --strategy-index 5 --start 0 --end 99 --gpus "0" --num_cpu_cores 20

3. 运行策略范围（指定起始和结束索引）：
   python3 run_sampling_strategies.py --strategy-range 0-39 --start 0 --end 99 --gpus "0" --num_cpu_cores 20
   python3 run_sampling_strategies.py --strategy-range 10-20 --start 0 --end 99 --gpus "0,1" --num_cpu_cores 20

4. 预览模式（不实际运行，只显示将要执行的命令）：
   python3 run_sampling_strategies.py --dry-run
   python3 run_sampling_strategies.py --strategy-range 0-2 --dry-run

5. 指定配置文件路径：
   python3 run_sampling_strategies.py --config configs/sampling.yml --start 0 --end 9

参数说明：
  --start: 数据ID起始索引（传递给 batch_sampleandeval_parallel.py）
  --end: 数据ID结束索引（传递给 batch_sampleandeval_parallel.py）
  --gpus: GPU ID，例如 "0" 或 "0,1,2" 或 "0-3"
  --num_cpu_cores: CPU核心数（用于并行评估）
  --config: 配置文件路径（默认：configs/sampling.yml）
  --strategy-index: 只运行指定索引的策略（从0开始，共100个策略）
  --strategy-range: 运行策略范围，格式：start-end（例如：0-5）
  --dry-run: 预览模式，只显示将要执行的命令，不实际运行

注意：
- 如果不指定 --strategy-index 或 --strategy-range，将运行所有100个策略
- 每个策略会依次更新配置文件并执行批量采样和评估
- 子进程的输出会实时显示在终端中
"""

import os
import sys
import yaml
import subprocess
import csv
import io
from pathlib import Path


# 策略参数数据（CSV格式）
STRATEGIES_DATA = """Strategy_Type,TL,LSstep,RFstep,LSL1,LSL2,RFL1,RFL2,Pred_Step,Start,End,Mode,Power,Description
Top1_NoiseTuning,750,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"降低噪声寻找平衡点 (Noise 0.08), RFnoise=0.08"
Top1_NoiseTuning,750,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"微降噪声 (Noise 0.09), RFnoise=0.09"
Top1_NoiseTuning,750,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"微增噪声 (Noise 0.11), RFnoise=0.11"
Top1_NoiseTuning,750,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"增加随机性 (Noise 0.12), RFnoise=0.12"
Top1_StepTuning,750,0.95,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"收缩步长 (Step 0.95), RFnoise=0.1"
Top1_StepTuning,750,0.98,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"微缩步长 (Step 0.98), RFnoise=0.1"
Top1_StepTuning,750,1.02,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"微放步长 (Step 1.02), RFnoise=0.1"
Top1_StepTuning,750,1.05,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"放宽步长 (Step 1.05), RFnoise=0.1"
Top1_GuideStrength,750,1.0,0.2,75,20,20,2,96,1.0,0.0,quadratic,2.0,"减弱引导 (Lambda 75), RFnoise=0.1"
Top1_GuideStrength,750,1.0,0.2,78,20,20,2,96,1.0,0.0,quadratic,2.0,"微弱引导 (Lambda 78), RFnoise=0.1"
Top1_GuideStrength,750,1.0,0.2,82,20,20,2,96,1.0,0.0,quadratic,2.0,"微强引导 (Lambda 82), RFnoise=0.1"
Top1_GuideStrength,750,1.0,0.2,85,20,20,2,96,1.0,0.0,quadratic,2.0,"增强引导 (Lambda 85), RFnoise=0.1"
Top1_StepCount,750,1.0,0.2,80,20,20,2,90,1.0,0.0,quadratic,2.0,"减少采样步数 (Steps 90), RFnoise=0.1"
Top1_StepCount,750,1.0,0.2,80,20,20,2,100,1.0,0.0,quadratic,2.0,"增加采样步数 (Steps 100), RFnoise=0.1"
Top1_StepCount,750,1.0,0.2,80,20,20,2,110,1.0,0.0,quadratic,2.0,"深度采样 (Steps 110), RFnoise=0.1"
Top1_Hybrid,750,0.95,0.2,85,20,20,2,100,1.0,0.0,quadratic,2.0,"高引导+小步长+多步数, RFnoise=0.1"
Top1_Hybrid,750,1.05,0.2,75,20,20,2,100,1.0,0.0,quadratic,2.0,"低引导+大步长+多步数, RFnoise=0.1"
Top1_Hybrid,750,1.0,0.2,80,20,20,2,100,1.0,0.0,quadratic,2.0,"低噪声+多步数 (稳健化), RFnoise=0.08"
Top1_TLExplore,780,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"轻微增加TL (780), RFnoise=0.1"
Top1_TLExplore,720,1.0,0.2,80,20,20,2,96,1.0,0.0,quadratic,2.0,"轻微减少TL (720), RFnoise=0.1"
Top2_GuideStrength,750,1.0,0.2,70,20,10,2,96,1.0,0.0,quadratic,2.0,"大幅减弱引导测试, RFnoise=0.05"
Top2_GuideStrength,750,1.0,0.2,75,20,10,2,96,1.0,0.0,quadratic,2.0,"中等减弱引导, RFnoise=0.05"
Top2_GuideStrength,750,1.0,0.2,85,20,10,2,96,1.0,0.0,quadratic,2.0,"增强引导测试, RFnoise=0.05"
Top2_GuideStrength,750,1.0,0.2,90,20,10,2,96,1.0,0.0,quadratic,2.0,"超强引导测试, RFnoise=0.05"
Top2_StepTuning,750,0.9,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"Step 0.9 寻找更稳点, RFnoise=0.05"
Top2_StepTuning,750,0.95,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"Step 0.95, RFnoise=0.05"
Top2_StepTuning,750,1.05,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"Step 1.05, RFnoise=0.05"
Top2_StepTuning,750,1.1,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"Step 1.1 激进测试, RFnoise=0.05"
Top2_StepCount,750,1.0,0.2,80,20,10,2,100,1.0,0.0,quadratic,2.0,"Steps 100, RFnoise=0.05"
Top2_StepCount,750,1.0,0.2,80,20,10,2,110,1.0,0.0,quadratic,2.0,"Steps 110 (低噪声适合多步), RFnoise=0.05"
Top2_StepCount,750,1.0,0.2,80,20,10,2,120,1.0,0.0,quadratic,2.0,"Steps 120 (极限测试), RFnoise=0.05"
Top2_StepCount,750,1.0,0.2,80,20,10,2,128,1.0,0.0,quadratic,2.0,"Steps 128 (2的幂次), RFnoise=0.05"
Top2_HybridDeep,750,1.0,0.2,90,20,10,2,110,1.0,0.0,quadratic,2.0,"强引导+深采样, RFnoise=0.05"
Top2_HybridDeep,750,1.0,0.2,70,20,10,2,110,1.0,0.0,quadratic,2.0,"弱引导+深采样, RFnoise=0.05"
Top2_TLTuning,740,1.0,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"TL 740, RFnoise=0.05"
Top2_TLTuning,760,1.0,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"TL 760, RFnoise=0.05"
Top2_TLTuning,750,1.0,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"极低噪声 0.04, RFnoise=0.04"
Top2_TLTuning,750,1.0,0.2,80,20,10,2,96,1.0,0.0,quadratic,2.0,"噪声 0.06, RFnoise=0.06"
Top2_OptimalGuess,750,0.98,0.2,82,20,10,2,100,1.0,0.0,quadratic,2.0,"参数微调组合A, RFnoise=0.05"
Top2_OptimalGuess,750,1.02,0.2,78,20,10,2,100,1.0,0.0,quadratic,2.0,"参数微调组合B, RFnoise=0.05"
Top3_TLExtend,800,0.6,0.2,80,20,10,5,70,1.0,0.0,quadratic,2.0,"基准Steps增加 (62->70), RFnoise=0.05"
Top3_TLExtend,800,0.6,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"Steps增加至80, RFnoise=0.05"
Top3_TLExtend,800,0.6,0.2,80,20,10,5,90,1.0,0.0,quadratic,2.0,"Steps增加至90, RFnoise=0.05"
Top3_TLExtend,800,0.6,0.2,80,20,10,5,100,1.0,0.0,quadratic,2.0,"Steps增加至100, RFnoise=0.05"
Top3_TLExtend,850,0.6,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL850 + Steps80, RFnoise=0.05"
Top3_TLExtend,850,0.6,0.2,80,20,10,5,100,1.0,0.0,quadratic,2.0,"TL850 + Steps100, RFnoise=0.05"
Top3_TLExtend,900,0.6,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL900 + Steps80, RFnoise=0.05"
Top3_TLExtend,900,0.6,0.2,80,20,10,5,100,1.0,0.0,quadratic,2.0,"TL900 + Steps100, RFnoise=0.05"
Top3_StepTuning,800,0.5,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的Step 0.5, RFnoise=0.05"
Top3_StepTuning,800,0.55,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的Step 0.55, RFnoise=0.05"
Top3_StepTuning,800,0.65,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的Step 0.65, RFnoise=0.05"
Top3_StepTuning,800,0.7,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的Step 0.7, RFnoise=0.05"
Top3_Lambda,800,0.6,0.2,70,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的弱引导, RFnoise=0.05"
Top3_Lambda,800,0.6,0.2,90,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的强引导, RFnoise=0.05"
Top3_Noise,800,0.6,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的中噪声, RFnoise=0.08"
Top3_Noise,800,0.6,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL800下的高噪声, RFnoise=0.1"
Top3_Hybrid,850,0.55,0.2,85,20,10,5,100,1.0,0.0,quadratic,2.0,"更长TL+更细步长+强引导, RFnoise=0.05"
Top3_Hybrid,850,0.65,0.2,75,20,10,5,100,1.0,0.0,quadratic,2.0,"更长TL+略大步长+弱引导, RFnoise=0.05"
Top3_Extreme,1000,0.6,0.2,80,20,10,5,120,1.0,0.0,quadratic,2.0,"TL1000 + Steps120, RFnoise=0.05"
Top3_Conservative,800,0.6,0.2,80,20,10,5,75,1.0,0.0,quadratic,2.0,"仅增加少量步数, RFnoise=0.05"
Top4_FineStep,750,0.38,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.0,"Step 0.38, RFnoise=0.05"
Top4_FineStep,750,0.39,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.0,"Step 0.39, RFnoise=0.05"
Top4_FineStep,750,0.41,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.0,"Step 0.41, RFnoise=0.05"
Top4_FineStep,750,0.42,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.0,"Step 0.42, RFnoise=0.05"
Top4_StepBurst,750,0.4,0.2,80,20,10,5,64,1.0,0.0,quadratic,2.0,"Steps 64 (基准是52), RFnoise=0.05"
Top4_StepBurst,750,0.4,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"Steps 80, RFnoise=0.05"
Top4_StepBurst,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.0,"Steps 96 (对齐其他组), RFnoise=0.05"
Top4_StepBurst,750,0.4,0.2,80,20,10,5,112,1.0,0.0,quadratic,2.0,"Steps 112 (深度采样), RFnoise=0.05"
Top4_LambdaMatch,750,0.4,0.2,70,20,10,5,80,1.0,0.0,quadratic,2.0,"低Step配合低Lambda, RFnoise=0.05"
Top4_LambdaMatch,750,0.4,0.2,90,20,10,5,80,1.0,0.0,quadratic,2.0,"低Step配合高Lambda, RFnoise=0.05"
Top4_LambdaMatch,750,0.4,0.2,100,20,10,5,80,1.0,0.0,quadratic,2.0,"低Step配合极高Lambda, RFnoise=0.05"
Top4_TLTuning,700,0.4,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL 700 + 多步, RFnoise=0.05"
Top4_TLTuning,800,0.4,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"TL 800 + 多步, RFnoise=0.05"
Top4_Hybrid,750,0.35,0.2,85,20,10,5,100,1.0,0.0,quadratic,2.0,"极低Step+高Lambda+多步, RFnoise=0.05"
Top4_Hybrid,750,0.45,0.2,75,20,10,5,80,1.0,0.0,quadratic,2.0,"稍高Step+低Lambda, RFnoise=0.05"
Top4_NoiseExplore,750,0.4,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"极低噪声 0.03, RFnoise=0.03"
Top4_NoiseExplore,750,0.4,0.2,80,20,10,5,80,1.0,0.0,quadratic,2.0,"稍高噪声 0.08, RFnoise=0.08"
Top4_OptimalGuess,750,0.4,0.2,85,20,10,5,90,1.0,0.0,quadratic,2.0,"低噪高引稳健版, RFnoise=0.04"
Top4_FastMode,750,0.5,0.2,80,20,10,5,60,1.0,0.0,quadratic,2.0,"Step 0.5 + 少步数, RFnoise=0.05"
Top4_SlowMode,750,0.3,0.2,80,20,10,5,120,1.0,0.0,quadratic,2.0,"Step 0.3 + 极多步数, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,0.5,"极缓衰减 Rate=0.5, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,1.0,"线性衰减 Rate=1.0, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,1.5,"中间态 Rate=1.5, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,2.5,"稍快衰减 Rate=2.5, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,3.0,"三次衰减 Rate=3.0, RFnoise=0.05"
Weight_Rate,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,4.0,"极快衰减 Rate=4.0, RFnoise=0.05"
Weight_Start,750,0.4,0.2,80,20,10,5,96,0.5,0.0,quadratic,2.0,"极弱开局 Start=0.5, RFnoise=0.05"
Weight_Start,750,0.4,0.2,80,20,10,5,96,0.8,0.0,quadratic,2.0,"弱开局 Start=0.8, RFnoise=0.05"
Weight_Start,750,0.4,0.2,80,20,10,5,96,1.2,0.0,quadratic,2.0,"强开局 Start=1.2, RFnoise=0.05"
Weight_Start,750,0.4,0.2,80,20,10,5,96,1.5,0.0,quadratic,2.0,"极强开局 Start=1.5, RFnoise=0.05"
Weight_Start,750,0.4,0.2,80,20,10,5,96,2.0,0.0,quadratic,2.0,"超强开局 Start=2.0, RFnoise=0.05"
Weight_End,750,0.4,0.2,80,20,10,5,96,1.0,0.05,quadratic,2.0,"末端微留 End=0.05, RFnoise=0.05"
Weight_End,750,0.4,0.2,80,20,10,5,96,1.0,0.1,quadratic,2.0,"末端少留 End=0.1, RFnoise=0.05"
Weight_End,750,0.4,0.2,80,20,10,5,96,1.0,0.15,quadratic,2.0,"末端中留 End=0.15, RFnoise=0.05"
Weight_End,750,0.4,0.2,80,20,10,5,96,1.0,0.2,quadratic,2.0,"末端强留 End=0.2, RFnoise=0.05"
Weight_Combo,750,0.4,0.2,80,20,10,5,96,1.5,0.1,quadratic,3.0,"高开快走留尾巴, RFnoise=0.05"
Weight_Combo,750,0.4,0.2,80,20,10,5,96,0.8,0.2,quadratic,1.0,"低开慢走留强尾, RFnoise=0.05"
Weight_Combo,750,0.4,0.2,80,20,10,5,96,2.0,0.05,quadratic,4.0,"爆发开局极速衰减, RFnoise=0.05"
Weight_Combo,750,0.4,0.2,80,20,10,5,96,1.2,0.1,quadratic,2.5,"中强开局稍快衰减, RFnoise=0.05"
Weight_Combo,750,0.4,0.2,80,20,10,5,96,1.0,0.0,quadratic,1.8,"Rate 1.8 探索微调, RFnoise=0.05"
"""


def parse_strategies():
    """解析策略数据"""
    import re
    reader = csv.DictReader(io.StringIO(STRATEGIES_DATA))
    strategies = []
    for row in reader:
        # 从Description中提取RFnoise值
        description = row['Description']
        rfnoise_match = re.search(r'RFnoise=([\d.]+)', description)
        rfnoise = float(rfnoise_match.group(1)) if rfnoise_match else 0.05  # 默认值0.05
        
        strategies.append({
            'strategy_type': row['Strategy_Type'],
            'TL': int(row['TL']),
            'LSstep': float(row['LSstep']),
            'RFstep': float(row['RFstep']),
            'LSL1': float(row['LSL1']),
            'LSL2': float(row['LSL2']),
            'RFL1': float(row['RFL1']),
            'RFL2': float(row['RFL2']),
            'pred_step': float(row['Pred_Step']),
            'start_weight': float(row['Start']),
            'end_weight': float(row['End']),
            'mode': row['Mode'],
            'power': float(row['Power']),
            'rfnoise': rfnoise,
            'description': description
        })
    return strategies


def update_sampling_config(config_path, strategy):
    """更新 sampling.yml 配置文件"""
    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新采样参数
    config['sample']['dynamic']['time_boundary'] = strategy['TL']
    config['sample']['dynamic']['large_step']['step_size'] = strategy['LSstep']
    config['sample']['dynamic']['refine']['step_size'] = strategy['RFstep']
    config['sample']['dynamic']['large_step']['lambda_coeff_a'] = strategy['LSL1']
    config['sample']['dynamic']['large_step']['lambda_coeff_b'] = strategy['LSL2']
    config['sample']['dynamic']['refine']['lambda_coeff_a'] = strategy['RFL1']
    config['sample']['dynamic']['refine']['lambda_coeff_b'] = strategy['RFL2']
    config['sample']['dynamic']['refine']['noise_scale'] = strategy['rfnoise']
    
    # 更新梯度融合参数
    config['model']['grad_fusion_lambda']['start'] = strategy['start_weight']
    config['model']['grad_fusion_lambda']['end'] = strategy['end_weight']
    config['model']['grad_fusion_lambda']['mode'] = strategy['mode']
    config['model']['grad_fusion_lambda']['power'] = strategy['power']
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"✓ 已更新配置: {strategy['strategy_type']}")
    print(f"  TL={strategy['TL']}, LSstep={strategy['LSstep']}, RFstep={strategy['RFstep']}, RFnoise={strategy['rfnoise']}")
    print(f"  LSL1={strategy['LSL1']}, LSL2={strategy['LSL2']}, RFL1={strategy['RFL1']}, RFL2={strategy['RFL2']}")
    print(f"  GradFusion: Start={strategy['start_weight']}, End={strategy['end_weight']}, Mode={strategy['mode']}, Power={strategy['power']}")


def run_batch_command(start=0, end=99, gpus="0", num_cpu_cores=20):
    """运行批量采样和评估命令，实时显示输出"""
    cmd = [
        'python3',
        '-u',  # 无缓冲模式，确保输出实时显示
        'batch_sampleandeval_parallel.py',
        '--start', str(start),
        '--end', str(end),
        '--gpus', gpus,
        '--num_cpu_cores', str(num_cpu_cores)
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("-" * 80)
    
    # 直接运行，输出会实时显示在终端中
    # stdout=None 和 stderr=None 表示继承父进程的终端
    # 环境变量 PYTHONUNBUFFERED=1 确保 Python 输出不被缓冲
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    result = subprocess.run(
        cmd,
        stdout=None,  # 直接输出到终端
        stderr=None,  # 直接输出到终端
        env=env,
        check=False
    )
    
    print("-" * 80)
    if result.returncode == 0:
        print("✓ 命令执行成功")
    else:
        print(f"✗ 命令执行失败，返回码: {result.returncode}")
    
    return result.returncode == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量运行不同采样策略',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  运行所有策略：
    %(prog)s --start 0 --end 99 --gpus "0" --num_cpu_cores 20
  
  运行单个策略（索引0）：
    %(prog)s --strategy-index 0 --start 0 --end 9 --gpus "0" --num_cpu_cores 20
  
  运行策略范围（索引0到5）：
    %(prog)s --strategy-range 0-5 --start 0 --end 9 --gpus "0" --num_cpu_cores 20
  
  预览模式：
    %(prog)s --strategy-range 0-2 --dry-run
        """
    )
    parser.add_argument('--start', type=int, default=0, help='数据ID起始索引（传递给 batch_sampleandeval_parallel.py）')
    parser.add_argument('--end', type=int, default=99, help='数据ID结束索引（传递给 batch_sampleandeval_parallel.py）')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID，例如 "0" 或 "0,1,2" 或 "0-3"')
    parser.add_argument('--num_cpu_cores', type=int, default=20, help='CPU核心数（用于并行评估）')
    parser.add_argument('--config', type=str, default='configs/sampling.yml', help='配置文件路径（默认：configs/sampling.yml）')
    parser.add_argument('--strategy-index', type=int, default=None, help='只运行指定索引的策略（从0开始，共40个策略）')
    parser.add_argument('--strategy-range', type=str, default=None, help='运行策略范围，格式：start-end（例如：0-5 表示运行索引0到5的策略）')
    parser.add_argument('--dry-run', action='store_true', help='预览模式：只显示将要执行的命令，不实际运行')
    
    args = parser.parse_args()
    
    # 解析策略列表
    strategies = parse_strategies()
    
    print(f"找到 {len(strategies)} 个策略")
    print("=" * 80)
    
    # 确定要运行的策略范围
    if args.strategy_index is not None:
        if 0 <= args.strategy_index < len(strategies):
            strategies_to_run = [strategies[args.strategy_index]]
            print(f"运行单个策略: 索引 {args.strategy_index}")
        else:
            print(f"错误: 策略索引 {args.strategy_index} 超出范围 [0, {len(strategies)-1}]")
            return 1
    elif args.strategy_range is not None:
        # 解析范围，格式：start-end
        try:
            start_idx, end_idx = map(int, args.strategy_range.split('-'))
            if start_idx < 0 or end_idx >= len(strategies) or start_idx > end_idx:
                print(f"错误: 策略范围 {args.strategy_range} 无效。有效范围: 0-{len(strategies)-1}")
                return 1
            strategies_to_run = strategies[start_idx:end_idx+1]
            print(f"运行策略范围: 索引 {start_idx} 到 {end_idx} (共 {len(strategies_to_run)} 个策略)")
        except ValueError:
            print(f"错误: 策略范围格式无效。请使用格式：start-end（例如：0-5）")
            return 1
    else:
        strategies_to_run = strategies
        print(f"运行所有策略: 共 {len(strategies_to_run)} 个")
    
    # 遍历每个策略
    for idx, strategy in enumerate(strategies_to_run):
        print(f"\n[{idx+1}/{len(strategies_to_run)}] 策略: {strategy['strategy_type']}")
        print(f"描述: {strategy['description']}")
        print(f"预计步长: {strategy['pred_step']}")
        
        # 更新配置文件
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误: 配置文件不存在: {config_path}")
            return 1
        
        if not args.dry_run:
            update_sampling_config(config_path, strategy)
        
        # 运行命令
        if args.dry_run:
            print(f"\n[DRY RUN] 将执行命令:")
            print(f"  python3 batch_sampleandeval_parallel.py --start {args.start} --end {args.end} --gpus \"{args.gpus}\" --num_cpu_cores {args.num_cpu_cores}")
        else:
            success = run_batch_command(
                start=args.start,
                end=args.end,
                gpus=args.gpus,
                num_cpu_cores=args.num_cpu_cores
            )
            
            if not success:
                print(f"\n警告: 策略 {strategy['strategy_type']} 执行失败")
                response = input("是否继续执行下一个策略? (y/n): ")
                if response.lower() != 'y':
                    print("已取消")
                    return 1
    
    print("\n" + "=" * 80)
    print("所有策略执行完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

