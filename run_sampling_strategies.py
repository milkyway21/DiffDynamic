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
  --strategy-index: 只运行指定索引的策略（从0开始，共40个策略）
  --strategy-range: 运行策略范围，格式：start-end（例如：0-5）
  --dry-run: 预览模式，只显示将要执行的命令，不实际运行

注意：
- 如果不指定 --strategy-index 或 --strategy-range，将运行所有40个策略
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
STRATEGIES_DATA = """Strategy_Type,TL,LSstep,RFstep,LSL1,LSL2,RFL1,RFL2,Pred_Step,Description
TL500-高精细 (High Precision),500,0.4,0.2,40,20,10,5,15.26,"TL=500, LS/RF步长=0.4/0.2, 预计步长=15.3"
TL500-抗噪测试 (Noise Test),500,0.6,0.4,40,20,30,5,15.28,"TL=500, LS/RF步长=0.6/0.4, 预计步长=15.3"
TL500-快速生成 (Fast Gen),500,0.6,0.25,40,20,10,5,19.93,"TL=500, LS/RF步长=0.6/0.25, 预计步长=19.9"
TL500-均衡探索 (Balanced),500,0.5,0.3,60,20,10,5,20.97,"TL=500, LS/RF步长=0.5/0.3, 预计步长=21.0"
TL500-高精细 (High Precision),500,0.4,0.4,40,20,10,5,27.10,"TL=500, LS/RF步长=0.4/0.4, 预计步长=27.1"
TL500-大步长 (Large Step),500,0.6,0.4,40,20,10,5,28.81,"TL=500, LS/RF步长=0.6/0.4, 预计步长=28.8"
TL550-高精细 (High Precision),550,0.3,0.2,60,20,10,5,15.14,"TL=550, LS/RF步长=0.3/0.2, 预计步长=15.1"
TL550-抗噪测试 (Noise Test),550,0.6,0.4,60,20,30,5,15.40,"TL=550, LS/RF步长=0.6/0.4, 预计步长=15.4"
TL550-快速生成 (Fast Gen),550,0.5,0.25,80,20,10,5,19.10,"TL=550, LS/RF步长=0.5/0.25, 预计步长=19.1"
TL550-高精细 (High Precision),550,0.4,0.25,40,20,10,5,20.04,"TL=550, LS/RF步长=0.4/0.25, 预计步长=20.0"
TL550-高精细 (High Precision),550,0.4,0.4,40,20,10,5,29.81,"TL=550, LS/RF步长=0.4/0.4, 预计步长=29.8"
TL600-抗噪测试 (Noise Test),600,0.3,0.4,40,20,30,5,15.26,"TL=600, LS/RF步长=0.3/0.4, 预计步长=15.3"
TL600-高精细 (High Precision),600,0.3,0.2,80,20,10,5,16.05,"TL=600, LS/RF步长=0.3/0.2, 预计步长=16.1"
TL600-快速生成 (Fast Gen),600,0.3,0.4,60,20,20,5,19.36,"TL=600, LS/RF步长=0.3/0.4, 预计步长=19.4"
TL600-均衡探索 (Balanced),600,0.5,0.35,40,20,20,5,20.05,"TL=600, LS/RF步长=0.5/0.35, 预计步长=20.1"
TL600-高精细 (High Precision),600,0.4,0.35,40,20,10,5,28.97,"TL=600, LS/RF步长=0.4/0.35, 预计步长=29.0"
TL600-大步长 (Large Step),600,0.5,0.35,40,20,10,5,29.99,"TL=600, LS/RF步长=0.5/0.35, 预计步长=30.0"
TL650-抗噪测试 (Noise Test),650,0.3,0.4,80,20,30,5,15.19,"TL=650, LS/RF步长=0.3/0.4, 预计步长=15.2"
TL650-高精细 (High Precision),650,0.4,0.15,40,20,10,5,15.99,"TL=650, LS/RF步长=0.4/0.15, 预计步长=16.0"
TL650-抗噪测试 (Noise Test),650,0.6,0.4,40,20,30,5,19.87,"TL=650, LS/RF步长=0.6/0.4, 预计步长=19.9"
TL650-均衡探索 (Balanced),650,0.6,0.3,40,20,20,5,20.53,"TL=650, LS/RF步长=0.6/0.3, 预计步长=20.5"
TL650-高精细 (High Precision),650,0.4,0.35,80,20,10,5,29.60,"TL=650, LS/RF步长=0.4/0.35, 预计步长=29.6"
TL650-大步长 (Large Step),650,0.6,0.3,40,20,10,5,29.76,"TL=650, LS/RF步长=0.6/0.3, 预计步长=29.8"
TL700-抗噪测试 (Noise Test),700,0.6,0.35,60,20,40,5,15.06,"TL=700, LS/RF步长=0.6/0.35, 预计步长=15.1"
TL700-高精细 (High Precision),700,0.3,0.15,60,20,10,5,15.13,"TL=700, LS/RF步长=0.3/0.15, 预计步长=15.1"
TL700-抗噪测试 (Noise Test),700,0.6,0.35,40,20,30,5,19.62,"TL=700, LS/RF步长=0.6/0.35, 预计步长=19.6"
TL700-均衡探索 (Balanced),700,0.3,0.35,60,20,20,5,20.10,"TL=700, LS/RF步长=0.3/0.35, 预计步长=20.1"
TL700-高精细 (High Precision),700,0.4,0.3,40,20,10,5,29.65,"TL=700, LS/RF步长=0.4/0.3, 预计步长=29.7"
TL750-抗噪测试 (Noise Test),750,0.6,0.25,40,20,40,5,15.10,"TL=750, LS/RF步长=0.6/0.25, 预计步长=15.1"
TL750-高精细 (High Precision),750,0.3,0.15,80,20,10,5,15.63,"TL=750, LS/RF步长=0.3/0.15, 预计步长=15.6"
TL750-抗噪测试 (Noise Test),750,0.6,0.4,40,20,40,5,19.54,"TL=750, LS/RF步长=0.6/0.4, 预计步长=19.5"
TL750-抗噪测试 (Noise Test),750,0.5,0.4,60,20,30,5,20.03,"TL=750, LS/RF步长=0.5/0.4, 预计步长=20.0"
TL750-高精细 (High Precision),750,0.4,0.3,80,20,10,5,29.71,"TL=750, LS/RF步长=0.4/0.3, 预计步长=29.7"
TL750-大步长 (Large Step),750,0.6,0.25,40,20,10,5,29.90,"TL=750, LS/RF步长=0.6/0.25, 预计步长=29.9"
TL800-抗噪测试 (Noise Test),800,0.6,0.25,80,20,30,5,15.08,"TL=800, LS/RF步长=0.6/0.25, 预计步长=15.1"
TL800-高精细 (High Precision),800,0.3,0.15,80,20,10,5,16.67,"TL=800, LS/RF步长=0.3/0.15, 预计步长=16.7"
TL800-推荐基准 (Recommended),800,0.3,0.3,60,20,20,5,20.13,"TL=800, LS/RF步长=0.3/0.3, 预计步长=20.1"
TL800-推荐基准 (Recommended),800,0.6,0.15,60,20,10,5,20.37,"TL=800, LS/RF步长=0.6/0.15, 预计步长=20.4"
TL800-高精细 (High Precision),800,0.4,0.25,40,20,10,5,29.15,"TL=800, LS/RF步长=0.4/0.25, 预计步长=29.2"
TL800-大步长 (Large Step),800,0.6,0.25,60,20,10,5,29.84,"TL=800, LS/RF步长=0.6/0.25, 预计步长=29.8"
"""


def parse_strategies():
    """解析策略数据"""
    reader = csv.DictReader(io.StringIO(STRATEGIES_DATA))
    strategies = []
    for row in reader:
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
            'description': row['Description']
        })
    return strategies


def update_sampling_config(config_path, strategy):
    """更新 sampling.yml 配置文件"""
    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新参数
    config['sample']['dynamic']['time_boundary'] = strategy['TL']
    config['sample']['dynamic']['large_step']['step_size'] = strategy['LSstep']
    config['sample']['dynamic']['refine']['step_size'] = strategy['RFstep']
    config['sample']['dynamic']['large_step']['lambda_coeff_a'] = strategy['LSL1']
    config['sample']['dynamic']['large_step']['lambda_coeff_b'] = strategy['LSL2']
    config['sample']['dynamic']['refine']['lambda_coeff_a'] = strategy['RFL1']
    config['sample']['dynamic']['refine']['lambda_coeff_b'] = strategy['RFL2']
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"✓ 已更新配置: {strategy['strategy_type']}")
    print(f"  TL={strategy['TL']}, LSstep={strategy['LSstep']}, RFstep={strategy['RFstep']}")
    print(f"  LSL1={strategy['LSL1']}, LSL2={strategy['LSL2']}, RFL1={strategy['RFL1']}, RFL2={strategy['RFL2']}")


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

