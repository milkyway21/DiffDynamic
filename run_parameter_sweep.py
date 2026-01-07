#!/usr/bin/env python3
"""
参数扫描脚本：自动修改 sampling.yml 参数并顺序执行多次 batch_sampleandeval_parallel.py

功能：
1. 修改 configs/sampling.yml 中的指定参数
2. 按顺序执行 batch_sampleandeval_parallel.py
3. 支持多个参数同时扫描

使用方法：
    # 扫描 time_lower 和 time_upper 从 700 到 500，每次减少 50（两个参数必须相等）
    # 方式1：使用 --sync 参数（推荐）
    python3 run_parameter_sweep.py \
        --param "sample.dynamic.large_step.time_lower:700:500:-50" \
        --param "sample.dynamic.refine.time_upper:700:500:-50" \
        --sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper" \
        --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus '0-5' --num_cpu_cores 100"
    
    # 方式2：使用 --param-sync 参数（只需指定一次值，自动应用到多个参数）
    python3 run_parameter_sweep.py \
        --param-sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper:700:500:-50" \
        --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus '0-5' --num_cpu_cores 100"
    
    # 只扫描一个参数
    python3 run_parameter_sweep.py \
        --param "sample.dynamic.large_step.time_lower:700:500:-50" \
        --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99"
"""

import os
import sys
import argparse
import subprocess
import shutil
import yaml
from pathlib import Path
from datetime import datetime
import time

# 项目根目录
REPO_ROOT = Path(__file__).parent
CONFIG_FILE = REPO_ROOT / 'configs' / 'sampling.yml'
CONFIG_BACKUP_DIR = REPO_ROOT / 'configs' / 'backups'


def parse_param_spec(param_spec):
    """
    解析参数规格字符串
    
    格式: 
        "path.to.param:start:end:step"      # 范围格式
        "path.to.param:value1,value2,value3" # 逗号分隔的值列表
        "path.to.param:value"                # 单个值（固定值）
    
    示例:
        "sample.dynamic.large_step.time_lower:650:500:-50"  # 从650到500，每次减50
        "sample.dynamic.refine.time_upper:650,600,550,500"  # 指定具体值列表
        "sample.dynamic.refine.time_upper:500"              # 固定值500
    
    Returns:
        tuple: (param_path, values_list)
    """
    if ':' not in param_spec:
        raise ValueError(f"参数规格格式错误: {param_spec}，应为 'path:start:end:step' 或 'path:val1,val2,val3'")
    
    parts = param_spec.split(':', 1)
    param_path = parts[0]
    value_spec = parts[1]
    
    # 检查是否是范围格式 (start:end:step)
    if ',' in value_spec:
        # 逗号分隔的列表
        values = [float(v.strip()) for v in value_spec.split(',')]
    else:
        # 范围格式 start:end:step 或单个值
        range_parts = value_spec.split(':')
        if len(range_parts) == 3:
            # 范围格式 start:end:step
            start = float(range_parts[0])
            end = float(range_parts[1])
            step = float(range_parts[2])
            
            # 生成值列表
            values = []
            current = start
            if step > 0:
                while current <= end:
                    values.append(current)
                    current += step
            else:
                while current >= end:
                    values.append(current)
                    current += step
        elif len(range_parts) == 1:
            # 单个值
            values = [float(value_spec.strip())]
        else:
            raise ValueError(f"范围格式错误: {value_spec}，应为 'start:end:step' 或单个值")
    
    # 转换为整数（如果所有值都是整数）
    if all(v.is_integer() for v in values):
        values = [int(v) for v in values]
    else:
        values = [float(v) for v in values]
    
    return param_path, values


def set_nested_value(config, path, value):
    """
    在嵌套字典中设置值
    
    Args:
        config: 配置字典
        path: 点分隔的路径，如 "sample.dynamic.large_step.time_lower"
        value: 要设置的值
    """
    keys = path.split('.')
    current = config
    
    # 遍历到倒数第二层
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # 设置最后一层的值
    current[keys[-1]] = value


def get_nested_value(config, path):
    """
    从嵌套字典中获取值
    
    Args:
        config: 配置字典
        path: 点分隔的路径
    
    Returns:
        值或None
    """
    keys = path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    
    return current


def load_config(config_file):
    """加载YAML配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config, config_file):
    """保存YAML配置文件"""
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def backup_config(config_file, backup_dir):
    """备份配置文件"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f'sampling_{timestamp}.yml'
    shutil.copy2(config_file, backup_file)
    return backup_file


def generate_combinations(param_specs, sync_groups=None):
    """
    生成所有参数组合
    
    Args:
        param_specs: 参数规格列表
        sync_groups: 同步组列表，每个组是一个参数路径列表，组内的参数必须保持相等
    
    Returns:
        list: 每个元素是一个字典，包含所有参数的组合
    """
    param_paths = []
    param_values_list = []
    
    for spec in param_specs:
        path, values = parse_param_spec(spec)
        param_paths.append(path)
        param_values_list.append(values)
    
    # 如果有同步组，检查并处理
    if sync_groups:
        # 验证同步组中的参数都在 param_paths 中
        all_sync_params = []
        for group in sync_groups:
            all_sync_params.extend(group)
        
        for sync_param in all_sync_params:
            if sync_param not in param_paths:
                raise ValueError(f"同步参数 {sync_param} 不在参数列表中")
        
        # 对于每个同步组，确保所有参数的值列表相同
        for group in sync_groups:
            if len(group) < 2:
                continue
            
            # 找到第一个参数的值列表
            first_param_idx = param_paths.index(group[0])
            first_values = param_values_list[first_param_idx]
            
            # 确保组内所有参数的值列表相同
            for param in group[1:]:
                param_idx = param_paths.index(param)
                if param_values_list[param_idx] != first_values:
                    raise ValueError(f"同步组 {group} 中的参数值列表不一致")
    
    # 生成组合
    import itertools
    
    if sync_groups:
        # 如果有同步组，需要特殊处理
        # 构建参数到同步组的映射
        param_to_group = {}
        for group in sync_groups:
            for param in group:
                param_to_group[param] = group
        
        # 找出每个同步组的代表参数（每个组选第一个）
        group_reps = {}
        for group in sync_groups:
            rep_param = group[0]
            group_reps[rep_param] = group
        
        # 找出需要参与组合生成的参数（非同步参数 + 每个同步组的代表）
        active_params = []
        active_indices = []
        for i, path in enumerate(param_paths):
            if path not in param_to_group:
                # 非同步参数
                active_params.append(path)
                active_indices.append(i)
            elif path in group_reps:
                # 同步组的代表参数
                active_params.append(path)
                active_indices.append(i)
        
        # 生成组合
        combinations = []
        active_values_list = [param_values_list[i] for i in active_indices]
        
        for combo in itertools.product(*active_values_list):
            combo_dict = {}
            # 填充所有参数
            for i, path in enumerate(param_paths):
                if path not in param_to_group:
                    # 非同步参数，直接使用对应的值
                    active_idx = active_params.index(path)
                    combo_dict[path] = combo[active_idx]
                else:
                    # 同步参数，使用组内代表参数的值
                    group = param_to_group[path]
                    rep_param = group[0]
                    rep_active_idx = active_params.index(rep_param)
                    combo_dict[path] = combo[rep_active_idx]
            
            combinations.append(combo_dict)
    else:
        # 没有同步组，生成所有组合
        combinations = []
        for combo in itertools.product(*param_values_list):
            combo_dict = {}
            for i, path in enumerate(param_paths):
                combo_dict[path] = combo[i]
            combinations.append(combo_dict)
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description='参数扫描脚本：自动修改 sampling.yml 参数并顺序执行多次 batch_sampleandeval_parallel.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 方式1：使用 --param-sync（推荐，只需指定一次值）
  python3 run_parameter_sweep.py \\
      --param-sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper:700:500:-50" \\
      --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus '0-5' --num_cpu_cores 25"
  
  # 方式2：使用 --param 和 --sync（需要指定两次相同的值）
  python3 run_parameter_sweep.py \\
      --param "sample.dynamic.large_step.time_lower:700:500:-50" \\
      --param "sample.dynamic.refine.time_upper:700:500:-50" \\
      --sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper" \\
      --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus '0-5' --num_cpu_cores 100"
  
  # 使用逗号分隔的指定值（--param-sync）
  python3 run_parameter_sweep.py \\
      --param-sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper:700,650,600,550,500" \\
      --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99"
  
  # 不同步（生成所有组合）
  python3 run_parameter_sweep.py \\
      --param "sample.dynamic.large_step.time_lower:700,650" \\
      --param "sample.dynamic.refine.time_upper:600,500" \\
      --cmd "python3 batch_sampleandeval_parallel.py --start 0 --end 99"
        """
    )
    
    parser.add_argument('--param', type=str, action='append', required=False,
                       help='参数规格，格式: "path.to.param:start:end:step" 或 "path.to.param:val1,val2,val3"\n'
                            '可以多次使用 --param 来扫描多个参数（会生成所有组合）')
    parser.add_argument('--cmd', type=str, required=True,
                       help='要执行的命令（包含所有参数）')
    parser.add_argument('--config', type=str, default=str(CONFIG_FILE),
                       help=f'配置文件路径（默认: {CONFIG_FILE}）')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='是否备份原始配置文件（默认: True）')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要执行的参数组合，不实际执行')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='如果某个组合执行失败，继续执行下一个组合')
    parser.add_argument('--sync', type=str, action='append',
                       help='指定必须同步的参数组（组内参数必须保持相等）\n'
                            '格式: "param1,param2,param3"（用逗号分隔）\n'
                            '可以多次使用 --sync 来指定多个同步组\n'
                            '示例: --sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper"')
    parser.add_argument('--param-sync', type=str, action='append',
                       help='指定同步参数组及其值（只需指定一次值，自动应用到组内所有参数）\n'
                            '格式: "param1,param2:start:end:step" 或 "param1,param2:val1,val2,val3"\n'
                            '示例: --param-sync "sample.dynamic.large_step.time_lower,sample.dynamic.refine.time_upper:700:500:-50"')
    
    args = parser.parse_args()
    
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ 错误: 配置文件不存在: {config_file}")
        sys.exit(1)
    
    # 备份原始配置
    if args.backup:
        backup_file = backup_config(config_file, CONFIG_BACKUP_DIR)
        print(f"✅ 已备份原始配置到: {backup_file}")
    
    # 检查是否至少指定了 --param 或 --param-sync
    if not args.param and not args.param_sync:
        print("❌ 错误: 必须指定 --param 或 --param-sync")
        parser.print_help()
        sys.exit(1)
    
    # 初始化 args.param（如果未指定）
    if not args.param:
        args.param = []
    
    # 处理 --param-sync 参数（简化方式：只需指定一次值）
    if args.param_sync:
        # 将 --param-sync 转换为 --param 和 --sync
        for param_sync_str in args.param_sync:
            if ':' not in param_sync_str:
                print(f"❌ 错误: --param-sync 格式错误: {param_sync_str}")
                print(f"   应为: 'param1,param2:start:end:step' 或 'param1,param2:val1,val2,val3'")
                sys.exit(1)
            
            # 分离参数列表和值规格
            parts = param_sync_str.split(':', 1)
            param_list_str = parts[0]
            value_spec = parts[1]
            
            # 解析参数列表
            param_list = [p.strip() for p in param_list_str.split(',')]
            if len(param_list) < 2:
                print(f"❌ 错误: --param-sync 至少需要2个参数: {param_sync_str}")
                sys.exit(1)
            
            # 为每个参数添加 --param（使用相同的值规格）
            for param in param_list:
                args.param.append(f"{param}:{value_spec}")
            
            # 添加 --sync
            if not args.sync:
                args.sync = []
            args.sync.append(param_list_str)
            print(f"✓ 同步参数组: {param_list} (将使用相同的值: {value_spec})")
    
    # 解析同步组
    sync_groups = None
    if args.sync:
        sync_groups = []
        for sync_str in args.sync:
            group = [p.strip() for p in sync_str.split(',')]
            if len(group) < 2:
                print(f"⚠️  警告: 同步组至少需要2个参数，忽略: {sync_str}")
                continue
            sync_groups.append(group)
            print(f"✓ 同步组: {group} (这些参数将保持相等)")
    
    # 生成所有参数组合
    try:
        combinations = generate_combinations(args.param, sync_groups)
    except Exception as e:
        print(f"❌ 错误: 解析参数规格失败: {e}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"参数扫描配置")
    print(f"{'='*80}")
    print(f"配置文件: {config_file}")
    print(f"参数组合数: {len(combinations)}")
    print(f"\n参数组合列表:")
    for i, combo in enumerate(combinations, 1):
        print(f"  [{i}] {combo}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("🔍 干运行模式：只显示参数组合，不实际执行")
        return
    
    # 加载原始配置
    original_config = load_config(config_file)
    
    # 执行每个组合
    results = []
    for i, combo in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"执行组合 [{i}/{len(combinations)}]")
        print(f"{'='*80}")
        print(f"参数: {combo}")
        print(f"{'='*80}\n")
        
        try:
            # 加载配置（每次都从原始配置开始）
            config = yaml.safe_load(yaml.dump(original_config))
            
            # 设置参数值
            for param_path, value in combo.items():
                set_nested_value(config, param_path, value)
                current_value = get_nested_value(config, param_path)
                print(f"  ✓ {param_path} = {current_value}")
            
            # 保存配置
            save_config(config, config_file)
            print(f"\n✅ 已更新配置文件: {config_file}")
            
            # 执行命令
            print(f"\n执行命令: {args.cmd}")
            print(f"{'-'*80}\n")
            
            start_time = time.time()
            result = subprocess.run(
                args.cmd,
                shell=True,
                check=False  # 不自动抛出异常，我们自己处理
            )
            elapsed_time = time.time() - start_time
            
            success = (result.returncode == 0)
            status = "✅ 成功" if success else "❌ 失败"
            
            print(f"\n{'-'*80}")
            print(f"{status} (返回码: {result.returncode}, 耗时: {elapsed_time:.2f}秒)")
            print(f"{'-'*80}\n")
            
            results.append({
                'combo': combo,
                'success': success,
                'returncode': result.returncode,
                'elapsed_time': elapsed_time
            })
            
            if not success and not args.continue_on_error:
                print(f"❌ 执行失败，停止扫描")
                break
                
        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断")
            break
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()
            results.append({
                'combo': combo,
                'success': False,
                'returncode': -1,
                'elapsed_time': 0,
                'error': str(e)
            })
            if not args.continue_on_error:
                break
    
    # 恢复原始配置
    try:
        save_config(original_config, config_file)
        print(f"\n✅ 已恢复原始配置文件: {config_file}")
    except Exception as e:
        print(f"\n⚠️  恢复原始配置失败: {e}")
        print(f"   请手动从备份恢复: {backup_file if args.backup else 'N/A'}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"扫描总结")
    print(f"{'='*80}")
    print(f"总组合数: {len(combinations)}")
    print(f"成功: {sum(1 for r in results if r['success'])}")
    print(f"失败: {sum(1 for r in results if not r['success'])}")
    print(f"总耗时: {sum(r['elapsed_time'] for r in results):.2f}秒")
    print(f"\n详细结果:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        print(f"  [{i}] {status} {result['combo']} "
              f"(返回码: {result['returncode']}, 耗时: {result['elapsed_time']:.2f}秒)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import traceback
    main()
