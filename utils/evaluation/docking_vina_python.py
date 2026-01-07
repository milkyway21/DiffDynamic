"""
基于 Vina Python API + Meeko 的纯 Python 对接任务。

特点：
- 不依赖 /bin/bash、prepare_receptor4.py 或 obabel，可在纯 Windows/conda 环境运行。
- 直接使用 RDKit 分子与 PDB 文本，通过 Meeko 转换为 PDBQT，再调用 vina 包完成搜索。
- 输出格式与 QVinaDockingTask 保持一致，便于复用现有统计/报表逻辑。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from utils.evaluation.docking_qvina import BaseDockingTask, get_random_id


def _extract_vina_scores(line: str) -> List[float]:
    tokens = line.strip().split()
    values: List[float] = []
    for token in tokens[::-1]:
        try:
            values.append(float(token))
        except ValueError:
            continue
        if len(values) == 3:
            break
    values = list(reversed(values))
    while len(values) < 3:
        values.append(float('nan'))
    return values


def _parse_pdbqt_models(pdbqt_path: Path, template_mol: Chem.Mol) -> List[EasyDict]:
    """解析 Vina 生成的 PDBQT 文件，构建 RDKit 分子姿势。"""
    results: List[EasyDict] = []
    n_atoms = template_mol.GetNumAtoms()
    if n_atoms == 0:
        return results

    current_coords: List[Tuple[float, float, float]] = []
    current_scores: Tuple[float, float, float] = (float('nan'),) * 3
    mode_id = -1

    with open(pdbqt_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if line.startswith('MODEL'):
                current_coords = []
                current_scores = (float('nan'),) * 3
                mode_id += 1
            elif line.startswith('REMARK') and 'VINA RESULT' in line.upper():
                vals = _extract_vina_scores(line)
                current_scores = (vals[0], vals[1], vals[2])
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                current_coords.append((x, y, z))
            elif line.startswith('ENDMDL'):
                if len(current_coords) != n_atoms:
                    continue
                conf = Chem.Conformer(n_atoms)
                for idx, (x, y, z) in enumerate(current_coords):
                    conf.SetAtomPosition(idx, (x, y, z))

                pose_mol = Chem.Mol(template_mol)
                pose_mol.RemoveAllConformers()
                pose_mol.AddConformer(conf, assignId=True)
                try:
                    Chem.SanitizeMol(pose_mol)
                except Exception:
                    pass

                results.append(EasyDict({
                    'rdmol': pose_mol,
                    'mode_id': mode_id,
                    'affinity': current_scores[0],
                    'rmsd_lb': current_scores[1],
                    'rmsd_ub': current_scores[2],
                }))

    return results

try:
    from vina import Vina  # type: ignore
except ImportError:  # pragma: no cover - 依赖缺失时在运行期提示
    Vina = None

try:
    from meeko import (  # type: ignore
        MoleculePreparation,
        ReceptorPreparation,
        PDBQTWriterLegacy,
    )
except ImportError:  # pragma: no cover
    MoleculePreparation = None
    ReceptorPreparation = None
    PDBQTWriterLegacy = None


class PythonVinaDockingTask(BaseDockingTask):
    """使用 Python Vina & Meeko 完成对接的任务封装。"""

    def __init__(self,
                 pdb_block: str,
                 ligand_rdmol: Chem.Mol,
                 tmp_dir: str = './tmp',
                 use_uff: bool = True,
                 center: Optional[np.ndarray] = None,
                 size_factor: float = 1.2,
                 exhaustiveness: int = 16,
                 n_poses: int = 9):
        super().__init__(pdb_block, ligand_rdmol)

        if Vina is None:
            raise ImportError("缺少 vina 库，请执行 `pip install vina` 后重试。")
        missing_meeko = any(obj is None for obj in [
            MoleculePreparation, ReceptorPreparation, PDBQTWriterLegacy
        ])
        if missing_meeko:
            raise ImportError("缺少 meeko 依赖，请执行 `pip install meeko rdkit-pypi` 后重试。")

        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = get_random_id()
        self.receptor_pdb_path = self.tmp_dir / f'{self.task_id}_receptor.pdb'
        self.receptor_pdb_path.write_text(self.pdb_block)
        self.receptor_pdbqt_path = self.tmp_dir / f'{self.task_id}_receptor.pdbqt'
        self.ligand_pdbqt_path = self.tmp_dir / f'{self.task_id}_ligand.pdbqt'
        self.docked_pdbqt_path = self.tmp_dir / f'{self.task_id}_out.pdbqt'
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses

        self.ligand_rdmol = Chem.AddHs(self.ligand_rdmol, addCoords=True)
        if use_uff:
            UFFOptimizeMolecule(self.ligand_rdmol, maxIters=200)

        pos = np.array(self.ligand_rdmol.GetConformer(0).GetPositions())
        if center is None:
            self.center = (pos.max(axis=0) + pos.min(axis=0)) / 2
        else:
            self.center = np.asarray(center, dtype=float)

        extent = pos.max(axis=0) - pos.min(axis=0)
        extent = np.maximum(extent, np.array([5.0, 5.0, 5.0]))  # 盒子不得太小
        self.size_x, self.size_y, self.size_z = (extent * size_factor).tolist()

        self._prepare_receptor_pdbqt()
        self._prepare_ligand_pdbqt()

    def _prepare_receptor_pdbqt(self):
        rec_prep = ReceptorPreparation()
        rec_setup = rec_prep.prepare(str(self.receptor_pdb_path))
        writer = PDBQTWriterLegacy()
        pdbqt_str = writer.write_string(rec_setup)
        self.receptor_pdbqt_path.write_text(pdbqt_str)

    def _prepare_ligand_pdbqt(self):
        lig_prep = MoleculePreparation()
        lig_setup = lig_prep.prepare(self.ligand_rdmol)
        writer = PDBQTWriterLegacy()
        pdbqt_str = writer.write_string(lig_setup)
        self.ligand_pdbqt_path.write_text(pdbqt_str)

    def run_sync(self) -> List[EasyDict]:
        v = Vina(sf_name='vina', seed=random.randint(1, 10**7))
        v.set_receptor(rigid_pdbqt_filename=str(self.receptor_pdbqt_path))
        v.set_ligand_from_file(str(self.ligand_pdbqt_path))
        v.compute_vina_maps(
            center=self.center.tolist(),
            box_size=[self.size_x, self.size_y, self.size_z]
        )

        v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
        v.write_poses(
            str(self.docked_pdbqt_path),
            n_poses=self.n_poses,
            overwrite=True
        )

        poses = _parse_pdbqt_models(self.docked_pdbqt_path, self.ligand_rdmol)
        if not poses:
            raise RuntimeError('Python Vina 对接未返回任何姿势结果。')

        for pose in poses:
            if pose.rdmol.GetNumAtoms() == self.ligand_rdmol.GetNumAtoms():
                try:
                    rdMolAlign.AlignMol(pose.rdmol, self.ligand_rdmol)
                except Exception:
                    pass

        return poses

