#!/usr/bin/env python3
"""
查找蛋白质ID对应的data_id

使用方法:
    python3 find_protein_data_id.py 7ew4
    python3 find_protein_data_id.py --protein_id 7ew4
    python3 find_protein_data_id.py --list-all  # 列出所有蛋白质ID
"""

import argparse
import pickle
import os
from pathlib import Path
import sys

# 将仓库根目录加入 sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import get_dataset
import utils.misc as misc


def find_data_id_by_protein_id(protein_id, dataset_path=None, config_path=None):
    """
    根据蛋白质ID查找对应的data_id
    
    Args:
        protein_id: 蛋白质ID（如 '7ew4'）
        dataset_path: 数据集路径（可选，会从配置文件读取）
        config_path: 配置文件路径（默认: configs/sampling.yml）
    
    Returns:
        list: 匹配的data_id列表
    """
    if config_path is None:
        config_path = REPO_ROOT / 'configs' / 'sampling.yml'
    
    # 加载配置
    config = misc.load_config(config_path)
    
    # 获取测试集
    _, test_set = get_dataset(config.data, transform=None)
    
    # 搜索匹配的data_id
    matches = []
    protein_id_lower = protein_id.lower()
    
    print(f"正在搜索蛋白质ID: {protein_id_lower}")
    print(f"测试集大小: {len(test_set)}")
    print(f"{'='*60}")
    
    for data_id in range(len(test_set)):
        try:
            data = test_set[data_id]
            # 检查protein_filename属性
            protein_filename = getattr(data, 'protein_filename', None)
            if protein_filename:
                # 检查文件名中是否包含蛋白质ID
                if protein_id_lower in protein_filename.lower():
                    matches.append({
                        'data_id': data_id,
                        'protein_filename': protein_filename,
                        'ligand_filename': getattr(data, 'ligand_filename', 'N/A')
                    })
                    print(f"✅ 找到匹配: data_id={data_id}")
                    print(f"   蛋白质文件: {protein_filename}")
                    print(f"   配体文件: {getattr(data, 'ligand_filename', 'N/A')}")
                    print()
        except Exception as e:
            print(f"⚠️  读取 data_id={data_id} 时出错: {e}")
            continue
    
    return matches


def list_all_proteins(test_set, max_display=100):
    """
    列出所有蛋白质ID和对应的data_id
    
    Args:
        test_set: 测试数据集
        max_display: 最多显示的数量
    """
    print(f"{'='*80}")
    print(f"蛋白质ID列表（最多显示 {max_display} 个）")
    print(f"{'='*80}")
    print(f"{'data_id':<10} {'蛋白质文件名':<50} {'配体文件名':<30}")
    print(f"{'-'*80}")
    
    count = 0
    for data_id in range(min(len(test_set), max_display)):
        try:
            data = test_set[data_id]
            protein_filename = getattr(data, 'protein_filename', 'N/A')
            ligand_filename = getattr(data, 'ligand_filename', 'N/A')
            
            # 提取蛋白质ID（从文件名中提取，通常是第一个下划线前的部分）
            if protein_filename != 'N/A':
                protein_id = protein_filename.split('/')[-1].split('_')[0].upper()
            else:
                protein_id = 'N/A'
            
            print(f"{data_id:<10} {protein_filename[:48]:<50} {ligand_filename[:28]:<30}")
            count += 1
        except Exception as e:
            print(f"{data_id:<10} 错误: {e}")
            continue
    
    if len(test_set) > max_display:
        print(f"\n... (总共 {len(test_set)} 个样本，仅显示前 {max_display} 个)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='查找蛋白质ID对应的data_id')
    parser.add_argument('protein_id', type=str, nargs='?', default=None,
                       help='蛋白质ID（如: 7ew4）')
    parser.add_argument('--protein_id', type=str, dest='protein_id_arg',
                       help='蛋白质ID（与位置参数相同）')
    parser.add_argument('--list-all', action='store_true',
                       help='列出所有蛋白质ID和对应的data_id')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/sampling.yml）')
    parser.add_argument('--max-display', type=int, default=100,
                       help='列出模式下的最大显示数量（默认: 100）')
    
    args = parser.parse_args()
    
    # 确定蛋白质ID
    protein_id = args.protein_id or args.protein_id_arg
    
    # 加载配置
    if args.config is None:
        config_path = REPO_ROOT / 'configs' / 'sampling.yml'
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    config = misc.load_config(config_path)
    
    # 获取测试集
    try:
        _, test_set = get_dataset(config.data, transform=None)
    except Exception as e:
        print(f"❌ 错误: 无法加载数据集: {e}")
        sys.exit(1)
    
    # 如果指定了列出所有，则列出所有蛋白质
    if args.list_all:
        list_all_proteins(test_set, max_display=args.max_display)
        return
    
    # 如果没有指定蛋白质ID，提示用户
    if protein_id is None:
        print("❌ 错误: 请指定蛋白质ID")
        print("\n使用方法:")
        print("  python3 find_protein_data_id.py <蛋白质ID>")
        print("  例如: python3 find_protein_data_id.py 7ew4")
        print("\n或者列出所有蛋白质:")
        print("  python3 find_protein_data_id.py --list-all")
        sys.exit(1)
    
    # 搜索匹配的data_id
    matches = find_data_id_by_protein_id(protein_id, config_path=config_path)
    
    # 输出结果
    print(f"\n{'='*60}")
    if matches:
        print(f"✅ 找到 {len(matches)} 个匹配项:")
        print(f"{'='*60}")
        for match in matches:
            print(f"data_id: {match['data_id']}")
            print(f"  蛋白质文件: {match['protein_filename']}")
            print(f"  配体文件: {match['ligand_filename']}")
            print()
        
        print(f"\n💡 生成分子的命令:")
        print(f"{'='*60}")
        data_ids = [m['data_id'] for m in matches]
        if len(data_ids) == 1:
            print(f"python3 batch_sampleandeval.py --start {data_ids[0]} --end {data_ids[0]}")
        else:
            print(f"# 单个生成:")
            for data_id in data_ids:
                print(f"python3 batch_sampleandeval.py --start {data_id} --end {data_id}")
            print(f"\n# 批量生成:")
            print(f"python3 batch_sampleandeval.py --start {min(data_ids)} --end {max(data_ids)}")
    else:
        print(f"❌ 未找到匹配的蛋白质ID: {protein_id}")
        print(f"\n💡 提示:")
        print(f"  - 检查蛋白质ID是否正确（不区分大小写）")
        print(f"  - 使用 --list-all 查看所有可用的蛋白质ID")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

