"""分子相似度与批量环数量评估工具。"""

# 总结：
# - 提供 Tanimoto 指纹相似度计算（单分子与批量），用于评估生成分子与参考分子的结构相似性。
# - 支持批量计算分子环数量，便于统计生成分子拓扑结构。

import numpy as np  # 导入 NumPy。
from rdkit import Chem, DataStructs  # 导入 RDKit 相关模块。


def tanimoto_sim(mol, ref):
    """计算分子与参考分子的 Tanimoto 相似度。"""
    fp1 = Chem.RDKFingerprint(ref)  # 生成参考分子的 RDK 指纹。
    fp2 = Chem.RDKFingerprint(mol)  # 生成目标分子的指纹。
    return DataStructs.TanimotoSimilarity(fp1, fp2)  # 返回 Tanimoto 相似度。


def tanimoto_sim_N_to_1(mols, ref):
    """计算多个分子与同一参考分子的相似度列表。"""
    sim = [tanimoto_sim(m, ref) for m in mols]  # 对每个分子调用单分子相似度。
    return sim


def batched_number_of_rings(mols):
    """批量计算分子的环数量。"""
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))  # 统计分子环数量。
    return np.array(n)  # 返回 NumPy 数组。
