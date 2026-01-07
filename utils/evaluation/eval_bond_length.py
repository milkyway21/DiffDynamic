"""Utils for evaluating bond length."""

# 总结：
# - 为生成分子的键长与原子对距离提供统计评估工具，并与经验分布进行 Jensen-Shannon 距离比较。
# - 包含键长分布获取、格式化、对比、绘图等函数，以及辅助计算原子对距离。

import collections  # 导入集合工具。
from typing import Tuple, Sequence, Dict, Optional  # 导入类型注解。

import numpy as np  # 导入 NumPy。
from scipy import spatial as sci_spatial  # 导入 SciPy 空间距离模块。
try:
    import matplotlib.pyplot as plt  # type: ignore  # 导入 Matplotlib。
except ImportError:  # pragma: no cover
    plt = None  # 在缺失 Matplotlib 时允许跳过绘图。

from utils.evaluation import eval_bond_length_config  # 导入经验分布配置。
import utils.data as utils_data  # 导入数据工具，其中包含键类型映射。

BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution


def get_distribution(distances: Sequence[float], bins=eval_bond_length_config.DISTANCE_BINS) -> np.ndarray:
    """Get the distribution of distances.

    Args:
        distances (list): List of distances.
        bins (list): bins of distances
    Returns:
        np.array: empirical distribution of distances with length equals to DISTANCE_BINS.
    """
    bin_counts = collections.Counter(np.searchsorted(bins, distances))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts)
    sum_counts = np.sum(bin_counts)
    if sum_counts == 0:  # 如果没有距离数据，返回全零数组。
        return bin_counts
    bin_counts = bin_counts / sum_counts
    return bin_counts


def _format_bond_type(bond_type: BondType) -> BondType:
    atom1, atom2, bond_category = bond_type
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1
    return atom1, atom2, bond_category


def get_bond_length_profile(bond_lengths: Sequence[BondLengthData]) -> BondLengthProfile:
    bond_length_profile = collections.defaultdict(list)
    for bond_type, bond_length in bond_lengths:
        bond_type = _format_bond_type(bond_type)
        bond_length_profile[bond_type].append(bond_length)
    bond_length_profile = {k: get_distribution(v) for k, v in bond_length_profile.items()}
    return bond_length_profile


def _bond_type_str(bond_type: BondType) -> str:
    atom1, atom2, bond_category = bond_type
    return f'{atom1}-{atom2}|{bond_category}'


def eval_bond_length_profile(bond_length_profile: BondLengthProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for bond_type, gt_distribution in eval_bond_length_config.EMPIRICAL_DISTRIBUTIONS.items():
        if bond_type not in bond_length_profile:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = None
        else:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(gt_distribution,
                                                                                             bond_length_profile[
                                                                                                 bond_type])

    return metrics


def get_pair_length_profile(pair_lengths):
    cc_dist = [d[1] for d in pair_lengths if d[0] == (6, 6) and d[1] < 2]
    all_dist = [d[1] for d in pair_lengths if d[1] < 12]
    pair_length_profile = {
        'CC_2A': get_distribution(cc_dist, bins=np.linspace(0, 2, 100)),
        'All_12A': get_distribution(all_dist, bins=np.linspace(0, 12, 100))
    }
    return pair_length_profile


def eval_pair_length_profile(pair_length_profile):
    metrics = {}
    for k, gt_distribution in eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS.items():
        if k not in pair_length_profile:
            metrics[f'JSD_{k}'] = None
        else:
            metrics[f'JSD_{k}'] = sci_spatial.distance.jensenshannon(gt_distribution, pair_length_profile[k])
    return metrics


def plot_distance_hist(pair_length_profile, metrics=None, save_path=None):
    """绘制原子对距离直方图，Matplotlib 不可用时抛出错误。"""
    if plt is None:
        raise ImportError('matplotlib is required for plotting distance histogram.')
    gt_profile = eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS
    plt.figure(figsize=(6 * len(gt_profile), 4))

    for idx, (k, gt_distribution) in enumerate(eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS.items()):
        plt.subplot(1, len(gt_profile), idx + 1)
        x = eval_bond_length_config.PAIR_EMPIRICAL_BINS[k]
        plt.step(x, gt_profile[k][1:])
        plt.step(x, pair_length_profile[k][1:])
        plt.legend(['True', 'Learned'])
        if metrics is not None:
            plt.title(f'{k} JS div: {metrics["JSD_" + k]:.4f}')
        else:
            plt.title(k)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def pair_distance_from_pos_v(pos, elements):
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    dist_list = []
    for s in range(len(pos)):
        for e in range(s + 1, len(pos)):
            s_sym = elements[s]
            e_sym = elements[e]
            d = pdist[s, e]
            dist_list.append(((s_sym, e_sym), d))
    return dist_list


def bond_distance_from_mol(mol):
    """从 RDKit 分子对象中提取键长信息。"""
    pos = mol.GetConformer().GetPositions()  # 获取三维坐标。
    pdist = pos[None, :] - pos[:, None]  # 计算差向量矩阵。
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))  # 转换为距离矩阵。
    all_distances = []  # 保存键长数据。
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()  # 起点原子序号。
        e_sym = bond.GetEndAtom().GetAtomicNum()  # 终点原子序号。
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # 对应索引。
        bond_type = utils_data.BOND_TYPES[bond.GetBondType()]  # 映射到整数键类型。
        distance = pdist[s_idx, e_idx]  # 查距阵得到距离。
        all_distances.append(((s_sym, e_sym, bond_type), distance))  # 记录键类型与距离。
    return all_distances
