"""生成分子属性与规则性打分的工具函数集合。"""

# 总结：
# - 提供 PAINS 过滤、Lipinski 规则打分、基本结构信息、QED/SA/logP 等化学属性评估。
# - 包含力场构建和构象能量计算，用于辅助稳定性分析。

from collections import Counter  # 导入 Counter，用于计数。
from copy import deepcopy  # 导入深拷贝，避免修改原分子。

import numpy as np  # 导入 NumPy。
from rdkit import Chem  # 导入 RDKit 主模块。
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski  # 导入常用子模块。
from rdkit.Chem.FilterCatalog import *  # noqa: F401,F403 保留通配导入以使用 FilterCatalog API。
from rdkit.Chem.QED import qed  # 导入 QED 计算函数。

from utils.evaluation.sascorer import compute_sa_score  # 导入合成可行性评分。


def is_pains(mol):
    """判断分子是否命中 PAINS (Pan-Assay Interference Compounds) 规则。"""
    params_pain = FilterCatalogParams()  # 构建过滤器参数。
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)  # 创建过滤器。
    mol = deepcopy(mol)  # 深拷贝避免副作用。
    Chem.SanitizeMol(mol)  # 标准化分子。
    entry = catalog_pain.GetFirstMatch(mol)  # 匹配 PAINS 子结构。
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    """统计分子满足多少条 Lipinski 五规则。"""
    mol = deepcopy(mol)  # 深拷贝用于处理。
    Chem.SanitizeMol(mol)  # 确保分子合法。
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_basic(mol):
    """返回分子的基础统计信息。"""
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        # mol3d = Chem.RemoveHs(mol3d)
        rmsd_list = np.array(rmsd_list)
        return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    except:
        return [np.nan, np.nan, np.nan]


def get_logp(mol):
    """计算 Crippen logP（疏水性）。"""
    return Crippen.MolLogP(mol)


def get_chem(mol):
    """汇总分子化学性质指标。"""
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies
