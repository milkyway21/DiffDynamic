#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 split.pt 索引快速定位 data_id 对应的蛋白质口袋文件。

示例：
    python scripts/find_pocket_from_split.py \
        --split data/crossdocked_pocket10_pose_split.pt \
        --dataset data/crossdocked_v1.1_rmsd1.0_pocket10 \
        --data_id 5
"""

import argparse
import os
import pickle
import sys
from typing import Any, Sequence

import torch


def _load_split_indices(split_path: str, subset: str) -> Sequence[int]:
    """读取 split.pt 并返回指定子集的索引列表。"""
    split_obj = torch.load(split_path, map_location='cpu')
    if subset not in split_obj:
        raise KeyError(f'在 {split_path} 中找不到 {subset} 子集，'
                       f'可用键为: {list(split_obj.keys())}')
    return split_obj[subset]


def _load_index_entries(index_path: str) -> Sequence[Any]:
    """读取 index.pkl，返回 (pocket_fn, ligand_fn, ...) 列表。"""
    with open(index_path, 'rb') as f:
        entries = pickle.load(f)
    if not isinstance(entries, (list, tuple)):
        raise TypeError(f'{index_path} 的内容格式异常，期望 list/tuple。')
    return entries


def _format_path(root: str, relative_path: str) -> str:
    """拼接根目录与相对路径，并返回规范化绝对路径。"""
    combined = os.path.abspath(os.path.join(root, relative_path))
    return os.path.normpath(combined)


def main():
    parser = argparse.ArgumentParser(
        description='根据 split 索引查找 data_id 对应的蛋白质口袋文件'
    )
    parser.add_argument('--split', type=str, required=True,
                        help='split_pl_dataset.py 生成的 split.pt 路径')
    parser.add_argument('--dataset', type=str, required=True,
                        help='extract_pockets.py --dest 指向的口袋数据根目录')
    parser.add_argument('--data_id', type=int, required=True,
                        help='测试集 data_id (0-99)')
    parser.add_argument('--subset', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='在 split.pt 中查询的子集，默认 test')
    parser.add_argument('--index-file', type=str, default=None,
                        help='index.pkl 路径，默认使用 <dataset>/index.pkl')
    args = parser.parse_args()

    split_path = os.path.abspath(args.split)
    dataset_root = os.path.abspath(args.dataset)
    index_path = os.path.abspath(args.index_file) if args.index_file else os.path.join(dataset_root, 'index.pkl')

    if not os.path.exists(split_path):
        sys.exit(f'找不到 split 文件：{split_path}')
    if not os.path.exists(dataset_root):
        sys.exit(f'找不到数据目录：{dataset_root}')
    if not os.path.exists(index_path):
        sys.exit(f'找不到 index.pkl：{index_path}')

    subset_indices = _load_split_indices(split_path, args.subset)
    if not subset_indices:
        sys.exit(f'{args.subset} 子集中没有记录，请检查 split 文件。')
    if args.data_id < 0 or args.data_id >= len(subset_indices):
        sys.exit(f'data_id 必须在 0~{len(subset_indices)-1} 范围内，当前为 {args.data_id}。')

    dataset_idx = int(subset_indices[args.data_id])
    index_entries = _load_index_entries(index_path)
    if dataset_idx < 0 or dataset_idx >= len(index_entries):
        sys.exit(f'索引 {dataset_idx} 超出 index.pkl 范围 (0~{len(index_entries)-1})。')

    entry = index_entries[dataset_idx]
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        protein_rel = entry[0]
        ligand_rel = entry[1]
    else:
        sys.exit('index.pkl 中的条目格式不包含 (protein, ligand) 两个字段，无法继续。')

    protein_path = _format_path(dataset_root, protein_rel)
    ligand_path = _format_path(dataset_root, ligand_rel)

    protein_exists = os.path.exists(protein_path)
    ligand_exists = os.path.exists(ligand_path)

    print('=' * 80)
    print(f'split 文件: {split_path}')
    print(f'数据目录: {dataset_root}')
    print(f'子集: {args.subset} | data_id: {args.data_id}')
    print(f'在原始索引中的条目编号: {dataset_idx}')
    print('-' * 80)
    print('蛋白质口袋文件:')
    print(f'  相对路径: {protein_rel}')
    print(f'  绝对路径: {protein_path}')
    print(f'  文件存在: {protein_exists}')
    if not protein_exists:
        print('  ⚠️  未找到该 .pdb，请确认 --dataset 是否指向 extract_pockets 输出目录。')
    print('-' * 80)
    print('配体文件:')
    print(f'  相对路径: {ligand_rel}')
    print(f'  绝对路径: {ligand_path}')
    print(f'  文件存在: {ligand_exists}')
    print('-' * 80)
    print('采样命令示例:')
    print(f'  python scripts/sample_diffusion.py configs/sampling.yml --data_id {args.data_id}')
    print('=' * 80)


if __name__ == '__main__':
    main()


