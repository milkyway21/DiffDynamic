#!/usr/bin/env python3
"""
对接生成分子的工具脚本

这个脚本用于在 sample 运行结束后，将生成的分子自动对接到其来源蛋白口袋：
1. 复用保存于 result.pt 中的 `ProteinLigandData` 元信息，定位原始 ligand/protein 文件（与 TargetDiff 官方实现一致）。
2. 将每个生成分子转换为 RDKit 分子并执行 QVina 对接。
3. 输出逐分子亲和力、统计结果以及可选的 Excel 报表。

示例：
    python -m utils.dock_generated_molecules \
        --pt_file outputs/result_20231201_101500.pt \
        --dataset ./data/crossdocked_v1.1_rmsd1.0_pocket10 \
        --protein_root ./data/crossdocked_v1.1_rmsd1.0 \
        --data_id 5
"""

import os
import re
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from easydict import EasyDict
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    print("Successfully imported RDKit")
except ImportError as e:
    print(f"RDKit import error: {e}")
    print("Please install RDKit: conda install -c conda-forge rdkit")
    sys.exit(1)

from utils import misc
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina_python import PythonVinaDockingTask
from utils.evaluation import scoring_func


# TargetDiff原子类型映射（从targetdiff_pt_to_sdf_converter.py复制）
MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,   # H
    (6, False): 1,   # C
    (6, True): 2,    # C aromatic
    (7, False): 3,   # N
    (7, True): 4,    # N aromatic
    (8, False): 5,   # O
    (8, True): 6,    # O aromatic
    (9, False): 7,   # F
    (15, False): 8,  # P
    (15, True): 9,   # P aromatic
    (16, False): 10,  # S
    (16, True): 11,   # S aromatic
    (17, False): 12   # Cl
}

MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

# 共价半径（Angstrom）
COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 0.99,  # Cl
    35: 1.20,  # Br
}

# 典型键长度（Angstrom，用于更精确的键推断）
BOND_LENGTHS = {
    (1, 1): 0.74,   # H-H
    (1, 6): 1.09,   # H-C
    (1, 7): 1.01,   # H-N
    (1, 8): 0.96,   # H-O
    (1, 9): 0.92,   # H-F
    (6, 6): 1.54,   # C-C (单键)
    (6, 7): 1.47,   # C-N (单键)
    (6, 8): 1.43,   # C-O (单键)
    (6, 9): 1.35,   # C-F
    (6, 15): 1.84,  # C-P
    (6, 16): 1.82,  # C-S
    (6, 17): 1.77,  # C-Cl
    (7, 7): 1.45,   # N-N
    (7, 8): 1.40,   # N-O
    (8, 8): 1.48,   # O-O
    (15, 15): 2.21, # P-P
    (16, 16): 2.04, # S-S
    (17, 17): 1.99, # Cl-Cl
}

# 原子最大价态（用于验证）
MAX_VALENCE = {
    1: 1,   # H
    6: 4,   # C
    7: 3,   # N (通常)
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
}

CONNECT_THE_DOTS_COVALENT_FACTOR = 1.3  # 与reconstruct.connect_the_dots保持一致


class GeneratedMoleculeConverter:
    """生成分子转换器（简化版，从targetdiff_pt_to_sdf_converter.py复制核心逻辑）"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.mode = 'add_aromatic'  # 固定使用add_aromatic模式

    def get_atomic_number_from_index(self, index: np.ndarray) -> List[int]:
        """将原子类型索引转换为原子序数"""
        return [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]

    def _to_numpy(self, data):
        """将torch tensor或numpy数组转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _load_pt_file(self, pt_path: str):
        """加载.pt文件，兼容不同PyTorch版本"""
        try:
            data = torch.load(pt_path, weights_only=False)
        except TypeError:
            try:
                data = torch.load(pt_path, map_location='cpu')
            except Exception as e:
                if self.debug:
                    print(f"Fallback load failed: {e}")
                data = torch.load(pt_path)
        return data

    def _get_bond_candidates(self, positions: np.ndarray, atomic_numbers: List[int]) -> List[Tuple[int, int, float]]:
        """获取所有可能的键候选（基于距离）"""
        candidates = []
        n_atoms = len(atomic_numbers)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                covalent_sum = (COVALENT_RADII.get(atomic_numbers[i], 1.0) +
                               COVALENT_RADII.get(atomic_numbers[j], 1.0))
                max_dist = covalent_sum * CONNECT_THE_DOTS_COVALENT_FACTOR

                if dist <= max_dist:
                    candidates.append((i, j, dist))

        candidates.sort(key=lambda x: x[2])
        return candidates

    def _check_valence(self, mol: Chem.RWMol, atom_idx: int, atomic_num: int) -> bool:
        """检查添加键后是否超出原子价态限制"""
        atom = mol.GetAtomWithIdx(atom_idx)
        current_degree = atom.GetDegree()
        max_valence = MAX_VALENCE.get(atomic_num, 4)

        if atomic_num == 6:  # C
            max_valence = 4
        elif atomic_num == 7:  # N
            max_valence = 4  # 允许季铵盐等
        elif atomic_num == 16:  # S
            max_valence = 6  # 允许硫酸根等

        return current_degree < max_valence

    def create_molecule_from_coords(self, positions: np.ndarray,
                                  atomic_numbers: List[int]) -> Tuple[Optional[Chem.Mol], int]:
        """从坐标和原子类型创建分子"""
        try:
            mol = Chem.RWMol()
            atom_indices = []

            for i, atomic_num in enumerate(atomic_numbers):
                atom = Chem.Atom(int(atomic_num))
                atom_idx = mol.AddAtom(atom)
                atom_indices.append(atom_idx)

            bond_candidates = self._get_bond_candidates(positions, atomic_numbers)
            bonds_added = 0

            for i, j, dist in bond_candidates:
                if mol.GetBondBetweenAtoms(atom_indices[i], atom_indices[j]) is not None:
                    continue

                if not self._check_valence(mol, atom_indices[i], atomic_numbers[i]):
                    continue
                if not self._check_valence(mol, atom_indices[j], atomic_numbers[j]):
                    continue

                mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
                bonds_added += 1

            mol = mol.GetMol()
            conf = Chem.Conformer(len(atomic_numbers))
            for i, pos in enumerate(positions):
                conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
            mol.AddConformer(conf)

            try:
                Chem.SanitizeMol(mol)
            except:
                pass

            return mol, bonds_added

        except Exception as e:
            if self.debug:
                print(f"Molecule creation error: {e}")
            return None, 0

    def convert_single_molecule(self, positions, atom_indices, molecule_idx: int) -> Tuple[Optional[Chem.Mol], bool]:
        """转换单个分子"""
        try:
            positions = self._to_numpy(positions)
            atom_indices = self._to_numpy(atom_indices)

            if atom_indices.ndim > 1:
                atom_indices = atom_indices.flatten()

            if positions.ndim == 1:
                positions = positions.reshape(-1, 3)

            if len(positions) != len(atom_indices):
                min_len = min(len(positions), len(atom_indices))
                positions = positions[:min_len]
                atom_indices = atom_indices[:min_len]

            atomic_numbers = self.get_atomic_number_from_index(atom_indices)
            mol, bonds_added = self.create_molecule_from_coords(positions, atomic_numbers)

            return mol, mol is not None

        except Exception as e:
            if self.debug:
                print(f"Error processing molecule {molecule_idx}: {e}")
            return None, False

    def convert_pt_to_molecules(self, pt_path: str, payload: Optional[Dict[str, Any]] = None) -> List[Chem.Mol]:
        """将.pt文件转换为分子列表"""
        data = payload if payload is not None else self._load_pt_file(pt_path)
        if 'pred_ligand_pos' not in data or 'pred_ligand_v' not in data:
            raise ValueError("Required keys 'pred_ligand_pos' and 'pred_ligand_v' not found")

        pred_positions = data['pred_ligand_pos']
        pred_atom_types = data['pred_ligand_v']

        molecules = []
        for i, (positions, atom_indices) in enumerate(zip(pred_positions, pred_atom_types)):
            mol, success = self.convert_single_molecule(positions, atom_indices, i)
            if success and mol is not None:
                molecules.append(mol)

        return molecules


def _sdf_sort_key(path: Path) -> Tuple[int, int, str]:
    """用于稳定排序 SDF 文件，优先按名称中的数字索引。"""
    match = re.search(r'(\d+)', path.stem)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def _load_molecules_from_sdf_dir(sdf_dir: str, logger=None) -> List[Chem.Mol]:
    """从目录中批量读取 SDF 分子，保持与 converter 输出一致的顺序。"""
    sdf_dir_path = Path(sdf_dir)
    if not sdf_dir_path.exists():
        raise FileNotFoundError(f"SDF 目录不存在：{sdf_dir}")

    sdf_files = sorted(
        (p for p in sdf_dir_path.glob('*.sdf') if p.is_file()),
        key=_sdf_sort_key
    )
    if not sdf_files:
        raise FileNotFoundError(f"SDF 目录中未找到 .sdf 文件：{sdf_dir}")

    molecules: List[Chem.Mol] = []
    for sdf_file in sdf_files:
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
        except Exception as exc:
            if logger:
                logger.warning(f"读取 SDF 失败（{sdf_file.name}）：{exc}")
            continue

        for pose_idx, mol in enumerate(supplier):
            if mol is None:
                if logger:
                    logger.warning(f"SDF {sdf_file.name} 的第 {pose_idx} 个分子解析失败，已跳过")
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:
                if logger:
                    logger.warning(f"SDF {sdf_file.name} 的第 {pose_idx} 个分子清理失败：{exc}")
            molecules.append(mol)

    if not molecules:
        raise ValueError(f"SDF 目录 {sdf_dir} 未能加载任何有效分子")

    return molecules


def _dedup(seq: List[str]) -> List[str]:
    """保持顺序地去重字符串列表。"""
    seen = set()
    result = []
    for item in seq:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _expand_root_candidates(*roots: Optional[str]) -> List[str]:
    """根据给定根目录生成额外的候选目录（处理 `_pocket10` 等后缀）。"""
    candidates: List[str] = []
    for root in roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        candidates.append(abs_root)
        path_obj = Path(abs_root)
        name = path_obj.name
        if name.endswith('_pocket10'):
            candidates.append(str(path_obj.with_name(name.replace('_pocket10', ''))))
        if name.endswith('_processed_final'):
            candidates.append(str(path_obj.with_name(name.replace('_processed_final', ''))))
    return _dedup(candidates)


def _normalize_token(path_token: str) -> str:
    """统一分隔符并移除冗余的 ./ 前缀。"""
    token = path_token.replace('\\', '/')
    while token.startswith('./'):
        token = token[2:]
    return token


def _generate_candidate_tokens(path_str: str, search_roots: List[str]) -> List[str]:
    """生成一系列可能的相对路径 token，兼容 Windows/Unix 路径与包含根目录的情况。"""
    base = _normalize_token(path_str)
    tokens = [base]

    def _maybe_strip_suffix(token: str) -> List[str]:
        variants = []
        suffixes = ['_pocket10.pdb', '_pocket.pdb']
        for suffix in suffixes:
            if token.lower().endswith(suffix):
                variants.append(token[:-len(suffix)] + '.pdb')
        return variants

    tokens.extend(_maybe_strip_suffix(base))

    root_variants = []
    for root in search_roots:
        abs_root = os.path.abspath(root)
        root_variants.append(Path(abs_root).as_posix())
        root_variants.append(Path(root).as_posix())
        root_variants.append(Path(abs_root).name)

    for token in list(tokens):
        lower_token = token.lower()
        for prefix in root_variants:
            if not prefix:
                continue
            norm_prefix = _normalize_token(prefix)
            lower_prefix = norm_prefix.lower()
            if lower_prefix and lower_token.startswith(lower_prefix):
                trimmed = token[len(norm_prefix):].lstrip('/\\')
                if trimmed:
                    tokens.append(trimmed)
            else:
                idx = lower_token.find(lower_prefix)
                if idx != -1:
                    trimmed = token[idx + len(norm_prefix):].lstrip('/\\')
                    if trimmed:
                        tokens.append(trimmed)
    return _dedup(tokens)


def _resolve_existing_path(path_str: str, search_roots: List[str], descriptor: str) -> str:
    """尝试在多个根目录下解析路径，若全部失败则抛出详细错误。"""
    tokens = _generate_candidate_tokens(path_str, search_roots)
    candidates: List[str] = []
    for token in tokens:
        if os.path.isabs(token):
            candidates.append(os.path.abspath(token))
        else:
            for root in search_roots:
                candidates.append(os.path.abspath(os.path.join(root, token)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    attempted = "\n    ".join(_dedup(candidates))
    raise FileNotFoundError(
        f'{descriptor} 不存在: {path_str}\n已尝试路径：\n    {attempted}'
    )


def _extract_reference_metadata(pt_payload: Dict[str, Any],
                                dataset_root: str,
                                protein_root: str) -> Tuple[str, str, str, str]:
    """
    从保存的 result.pt 中提取 ligand/protein 文件名并归一化路径。

    Returns:
        (ligand_filename, ligand_path_abs, protein_filename, protein_path_abs)
    """
    data_obj = pt_payload.get('data')
    if data_obj is None:
        raise ValueError('result.pt 中缺少原始数据对象 (key="data")，无法定位口袋文件。')

    ligand_filename = getattr(data_obj, 'ligand_filename', None)
    protein_filename = getattr(data_obj, 'protein_filename', None)
    if ligand_filename is None:
        raise ValueError('未在 result.pt 的 data 对象中找到 ligand_filename 属性。')

    ligand_filename = str(ligand_filename)
    ligand_dir = os.path.dirname(ligand_filename)
    ligand_basename = os.path.basename(ligand_filename)
    ligand_prefix = ligand_basename[:10]

    protein_candidates: List[str] = []
    if protein_filename:
        protein_candidates.append(str(protein_filename))
    if ligand_dir:
        protein_candidates.append(os.path.join(ligand_dir, f'{ligand_prefix}.pdb'))
        if '_rec' not in ligand_prefix.lower():
            protein_candidates.append(os.path.join(ligand_dir, f'{ligand_prefix}_rec.pdb'))
    else:
        protein_candidates.append(f'{ligand_prefix}.pdb')

    protein_candidates = _dedup(protein_candidates)

    ligand_path = _resolve_existing_path(
        ligand_filename,
        search_roots=_expand_root_candidates(dataset_root),
        descriptor='配体文件'
    )
    protein_roots = _expand_root_candidates(protein_root, dataset_root, str(Path(dataset_root).parent))

    protein_path = None
    last_error = None
    for candidate in protein_candidates:
        try:
            protein_path = _resolve_existing_path(
                candidate,
                search_roots=protein_roots,
                descriptor='蛋白质口袋文件'
            )
            protein_filename = candidate
            break
        except FileNotFoundError as exc:
            last_error = exc
    if protein_path is None:
        assert last_error is not None
        raise last_error

    return ligand_filename, ligand_path, protein_filename, protein_path


def dock_generated_molecules(pt_file: str,
                             dataset_root: str,
                             data_id: Optional[int] = None,
                             output_dir: str = "docking_results",
                             protein_root: Optional[str] = None,
                             use_uff: bool = True,
                             size_factor: float = 1.2,
                             debug: bool = False,
                             sdf_dir: Optional[str] = None,
                             dock_backend: str = 'vina_python',
                             tmp_dir: str = './tmp',
                             vina_exhaustiveness: int = 16,
                             vina_poses: int = 9):
    """
    对生成的分子进行对接的主函数

    Args:
        pt_file: 生成的.pt文件路径
        dataset_root: 数据集根目录
        data_id: 数据ID
        output_dir: 输出目录
        protein_root: 蛋白质根目录（默认与dataset_root相同，可指向未裁剪的pocket路径）
        use_uff: 是否使用UFF优化
        size_factor: 对接盒尺寸因子
        debug: 是否启用调试模式
        sdf_dir: 已预转换 SDF 文件所在目录（可选，优先使用）
        dock_backend: 对接后端（'vina_python' 或 'qvina_cli'）
        tmp_dir: 临时文件目录
        vina_exhaustiveness: Vina 搜索强度（仅对 vina_python 有效）
        vina_poses: 保存的姿势数量（仅对 vina_python 有效）
    """
    logger = misc.get_logger('dock_generated')

    dataset_root = os.path.abspath(dataset_root)
    protein_root = os.path.abspath(protein_root or dataset_root)
    tmp_dir = os.path.abspath(tmp_dir)

    if not os.path.exists(pt_file):
        raise FileNotFoundError(f'找不到 pt 文件：{pt_file}')

    # 1. 加载.pt 文件并转换为分子
    logger.info("Loading sampling payload from pt file...")
    pt_payload = torch.load(pt_file, map_location='cpu')
    molecules: List[Chem.Mol] = []
    if sdf_dir:
        try:
            molecules = _load_molecules_from_sdf_dir(sdf_dir, logger=logger)
            logger.info(f"Loaded {len(molecules)} molecules from SDF directory: {os.path.abspath(sdf_dir)}")
            expected = len(pt_payload.get('pred_ligand_pos', []))
            if expected and len(molecules) != expected:
                logger.warning(f"SDF 数量（{len(molecules)}）与 pt 预测数量（{expected}）不匹配，将以 SDF 为准继续。")
        except Exception as exc:
            logger.warning(f"加载预转换 SDF 失败，将回退至内置重建逻辑：{exc}")
            molecules = []

    if not molecules:
        converter = GeneratedMoleculeConverter(debug=debug)
        molecules = converter.convert_pt_to_molecules(pt_path=pt_file, payload=pt_payload)
        logger.info(f"Successfully converted {len(molecules)} molecules")

    if len(molecules) == 0:
        logger.error("No molecules were successfully converted")
        return

    # 2. 根据 result.pt 中保存的 meta 信息定位蛋白质口袋
    ligand_filename, ligand_path, protein_filename, protein_path = _extract_reference_metadata(
        pt_payload, dataset_root=dataset_root, protein_root=protein_root
    )
    logger.info(f"Ligand reference: {ligand_filename} -> {ligand_path}")
    logger.info(f"Protein reference: {protein_filename} -> {protein_path}")

    if not os.path.exists(protein_path):
        raise FileNotFoundError(f'找不到蛋白质口袋文件：{protein_path}')

    with open(protein_path, 'r') as f:
        protein_pdb_block = f.read()

    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 4. 对每个分子进行对接
    logger.info(f"Starting docking of {len(molecules)} molecules...")
    results = []

    for i, mol in enumerate(molecules):
        try:
            logger.info(f"Docking molecule {i+1}/{len(molecules)}...")

            # 计算化学性质
            chem_results = scoring_func.get_chem(mol)

            backend = (dock_backend or 'vina_python').lower()
            if backend == 'vina_python':
                task_kwargs = dict(
                    pdb_block=protein_pdb_block,
                    ligand_rdmol=mol,
                    tmp_dir=tmp_dir,
                    use_uff=use_uff,
                    size_factor=size_factor,
                    exhaustiveness=vina_exhaustiveness,
                    n_poses=vina_poses
                )
                docking_cls = PythonVinaDockingTask
            elif backend == 'qvina_cli':
                task_kwargs = dict(
                    pdb_block=protein_pdb_block,
                    ligand_rdmol=mol,
                    tmp_dir=tmp_dir,
                    use_uff=use_uff,
                    size_factor=size_factor
                )
                docking_cls = QVinaDockingTask
            else:
                raise ValueError(f'Unsupported docking backend: {dock_backend}')

            try:
                vina_task = docking_cls(**task_kwargs)
                vina_results = vina_task.run_sync()
            except ImportError as exc:
                logger.error(f"初始化对接后端 {dock_backend} 失败：{exc}")
                raise

            results.append({
                'molecule_id': i,
                'mol': mol,
                'smiles': Chem.MolToSmiles(mol),
                'chem_results': chem_results,
                'vina': vina_results,
                'pocket_path': protein_path,
                'ligand_filename': ligand_filename,
                'protein_filename': protein_filename
            })

            logger.info(f"Molecule {i+1} docked successfully. Best affinity: {vina_results[0]['affinity']:.3f}")

        except Exception as e:
            logger.error(f"Error docking molecule {i+1}: {e}")
            continue

    # 5. 保存结果并记录到Excel
    if results:
        suffix = f"data_{data_id}" if data_id is not None else "data_unknown"
        output_file = os.path.join(output_dir, f"docked_results_{suffix}.pt")
        torch.save(results, output_file)
        logger.info(f"Saved {len(results)} docking results to {output_file}")

        # 输出统计信息
        affinities = [r['vina'][0]['affinity'] for r in results]
        best_affinity = min(affinities)  # 更小的亲和力值表示更好的结合
        avg_affinity = sum(affinities) / len(affinities)

        logger.info(f"Best affinity: {best_affinity:.3f}")
        logger.info(f"Average affinity: {avg_affinity:.3f}")

        # 记录到Excel表格
        try:
            record_docking_results_to_excel(
                results=results,
                output_dir=output_dir,
                data_id=data_id,
                pt_file=pt_file,
                pocket_path=protein_path,
                logger=logger
            )
            logger.info("Docking results recorded to Excel successfully")
        except Exception as e:
            logger.error(f"Failed to record results to Excel: {e}")
    else:
        logger.error("No docking results to save")


def record_docking_results_to_excel(results: List[Dict], output_dir: str, data_id: Optional[int],
                                   pt_file: str, pocket_path: str, logger=None):
    """
    将对接结果记录到Excel表格中

    Args:
        results: 对接结果列表
        output_dir: 输出目录
        data_id: 数据ID（可为 None）
        pt_file: 原始pt文件路径
        pocket_path: 口袋文件路径
        logger: 日志记录器
    """
    data_suffix = data_id if data_id is not None else 'unknown'
    excel_file = os.path.join(output_dir, f"docking_results_data_{data_suffix}.xlsx")

    # 准备数据
    records = []
    for i, result in enumerate(results):
        # 获取对接信息
        vina_result = result['vina'][0]  # 取最佳姿势
        affinity = vina_result['affinity']
        rmsd_lb = vina_result['rmsd_lb']
        rmsd_ub = vina_result['rmsd_ub']

        # 获取分子信息
        smiles = result.get('smiles', 'N/A')

        # 获取化学性质指标
        chem_results = result.get('chem_results', {})
        qed = chem_results.get('qed', 'N/A')
        sa = chem_results.get('sa', 'N/A')

        # 构建记录
        record = {
            '分子ID': i,
            'SMILES': smiles,
            '对接亲和力': affinity,
            'RMSD下界': rmsd_lb,
            'RMSD上界': rmsd_ub,
            'QED评分': qed,
            'SA评分': sa,
            '原始PT文件': os.path.basename(pt_file),
            '口袋文件': os.path.basename(pocket_path),
            '数据ID': data_id,
            '对接时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        records.append(record)

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(records)

    # 按亲和力排序（从小到大，更小的亲和力表示更好的结合）
    df = df.sort_values('对接亲和力')

    # 保存到Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='对接结果', index=False)

        # 添加统计信息工作表
        if len(records) > 0:
            stats = {
                '统计项目': [
                    '总分子数',
                    '最佳亲和力',
                    '平均亲和力',
                    '最差亲和力',
                    '亲和力标准差'
                ],
                '数值': [
                    len(records),
                    df['对接亲和力'].min(),
                    df['对接亲和力'].mean(),
                    df['对接亲和力'].max(),
                    df['对接亲和力'].std()
                ]
            }
            stats_df = pd.DataFrame(stats)
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)

    if logger:
        logger.info(f"Docking results saved to Excel: {excel_file}")
        logger.info(f"Total molecules: {len(records)}")
        logger.info(f"Best affinity: {df['对接亲和力'].min():.3f}")
        logger.info(f"Average affinity: {df['对接亲和力'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='对接生成分子的工具脚本（复用了 TargetDiff 中的口袋解析逻辑）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m utils.dock_generated_molecules --pt_file result_5.pt \
      --dataset data/crossdocked_v1.1_rmsd1.0_pocket10 --protein_root data/crossdocked_v1.1_rmsd1.0 --data_id 5

参数说明：
  --pt_file: sample生成的.pt文件路径
  --dataset: 数据集根目录（含参考 ligand/pocket 文件）
  --protein_root: 蛋白质文件根目录（默认与dataset相同，可指向未裁剪的pocket目录）
  --data_id: 数据ID（可选，仅用于日志和文件命名）
  --output_dir: 输出目录（默认: docking_results）
  --use_uff: 是否使用UFF优化（默认: True）
  --size_factor: 对接盒尺寸因子（默认: 1.2）
  --debug: 启用调试模式
        """
    )

    parser.add_argument('--pt_file', type=str, required=True,
                       help='sample生成的.pt文件路径')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集根目录（包含参考 ligand/pocket 文件）')
    parser.add_argument('--data_id', type=int, default=None,
                       help='数据ID（对应sample时的--data_id参数，可选）')
    parser.add_argument('--output_dir', type=str, default='docking_results',
                       help='输出目录（默认: docking_results）')
    parser.add_argument('--protein_root', type=str, default=None,
                       help='蛋白质文件根目录（默认与dataset相同）')
    parser.add_argument('--use_uff', type=eval, default=True,
                       help='是否使用UFF优化（默认: True）')
    parser.add_argument('--size_factor', type=float, default=1.2,
                       help='对接盒尺寸因子（默认: 1.2）')
    parser.add_argument('--sdf_dir', type=str, default=None,
                       help='若已使用 targetdiff_pt_to_sdf_converter 预转换 SDF，可在此指定目录')
    parser.add_argument('--dock_backend', type=str, default='vina_python',
                       choices=['vina_python', 'qvina_cli'],
                       help='选择对接后端（默认: vina_python）')
    parser.add_argument('--tmp_dir', type=str, default='./tmp',
                       help='对接过程中使用的临时文件目录')
    parser.add_argument('--vina_exhaustiveness', type=int, default=16,
                       help='Vina 搜索强度（仅当后端为 vina_python 时生效）')
    parser.add_argument('--vina_poses', type=int, default=9,
                       help='Vina 输出姿势数量（仅当后端为 vina_python 时生效）')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.pt_file):
        print(f"错误：找不到.pt文件：{args.pt_file}")
        return 1

    if not os.path.exists(args.dataset):
        print(f"错误：找不到数据集目录：{args.dataset}")
        return 1

    if args.protein_root and not os.path.exists(args.protein_root):
        print(f"错误：找不到蛋白质根目录：{args.protein_root}")
        return 1

    # 运行对接
    try:
        dock_generated_molecules(
            pt_file=args.pt_file,
            dataset_root=args.dataset,
            data_id=args.data_id,
            output_dir=args.output_dir,
            protein_root=args.protein_root,
            use_uff=args.use_uff,
            size_factor=args.size_factor,
            debug=args.debug,
            sdf_dir=args.sdf_dir,
            dock_backend=args.dock_backend,
            tmp_dir=args.tmp_dir,
            vina_exhaustiveness=args.vina_exhaustiveness,
            vina_poses=args.vina_poses
        )
        print("对接完成！")
        return 0
    except Exception as e:
        print(f"对接过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
