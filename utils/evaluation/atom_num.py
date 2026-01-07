"""Utils for sampling size of a molecule of a given protein pocket."""

# 总结：
# - 根据蛋白口袋的三维尺寸估计，选择对应的原子数量概率分布并采样生成分子原子数。
# - 使用预先统计好的 `CONFIG` 数据，将空间尺度映射到分段索引并执行加权随机抽样。

import numpy as np  # 导入 NumPy，用于数组和统计计算。
from scipy import spatial as sc_spatial  # 导入 SciPy 空间距离模块，别名为 sc_spatial。

from utils.evaluation.atom_num_config import CONFIG  # 引入预生成的原子数量分布配置。


def get_space_size(pocket_3d_pos):
    """根据口袋中氨基酸原子坐标估计空间尺寸。"""
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')  # 计算所有两两距离。
    aa_dist = np.sort(aa_dist)[::-1]  # 将距离降序排列。
    return np.median(aa_dist[:10])  # 取前 10 个最大距离的中位数作为空间尺度。


def _get_bin_idx(space_size):
    """将空间尺度映射到概率分布区间索引。"""
    bounds = CONFIG['bounds']  # 读取分段阈值。
    for i in range(len(bounds)):
        if bounds[i] > space_size:  # 第一个大于当前尺度的阈值即对应区间。
            return i
    return len(bounds)  # 若没有阈值更大，则落入最后一个区间。


def sample_atom_num(space_size, min_atoms=10, max_atoms=50):
    """按空间尺度采样原子数量。
    
    Args:
        space_size: 蛋白口袋的空间尺度。
        min_atoms: 最小原子数，默认10（药物分子的合理下限）。
        max_atoms: 最大原子数，默认50（药物分子的合理上限）。
    
    Returns:
        int: 采样得到的原子数量，保证在 [min_atoms, max_atoms] 范围内的正整数。
    """
    bin_idx = _get_bin_idx(space_size)  # 获取区间索引。
    num_atom_list, prob_list = CONFIG['bins'][bin_idx]  # 读取该区间的候选原子数量及概率。
    sampled = np.random.choice(num_atom_list, p=prob_list)  # 按概率随机抽样一个原子数。
    # 确保返回值为正整数，限制在合理范围内（10-50，优先15-30）
    sampled_int = max(min_atoms, min(max_atoms, int(abs(sampled))))
    return sampled_int
