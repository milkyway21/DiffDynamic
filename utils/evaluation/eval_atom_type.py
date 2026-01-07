"""评估生成分子原子类型分布与参考分布的差异。"""

# 总结：
# - 统计预测分子中出现的原子类型频率，与预先统计的参考分布进行 Jensen-Shannon 距离比较。
# - 参考分布来源于训练数据集的归一化统计（概率形式）。

from collections import Counter  # 导入 Counter，用于计数原子类型出现次数。
from scipy import spatial as sci_spatial  # 导入 SciPy 空间距离模块。
import numpy as np  # 导入 NumPy，用于数组操作。

# 历史版本保留了未归一化的计数，可作为参考。
# ATOM_TYPE_DISTRIBUTION = {
#     6: 1585004,
#     7: 276248,
#     8: 400236,
#     9: 30871,
#     15: 26288,
#     16: 26529,
#     17: 15210,
# }

ATOM_TYPE_DISTRIBUTION = {  # 归一化后的参考概率分布。
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}


def eval_atom_type_distribution(pred_counter: Counter):
    """计算预测分子中原子类型分布与参考分布的 Jensen-Shannon 距离。"""
    total_num_atoms = sum(pred_counter.values())  # 统计总原子数量。
    if total_num_atoms == 0:  # 如果没有原子数据，返回 None。
        return None
    pred_atom_distribution = {}  # 保存预测分布。
    for k in ATOM_TYPE_DISTRIBUTION:
        pred_atom_distribution[k] = pred_counter[k] / total_num_atoms  # 归一化到概率。
    js = sci_spatial.distance.jensenshannon(
        np.array(list(ATOM_TYPE_DISTRIBUTION.values())),  # 参考分布。
        np.array(list(pred_atom_distribution.values()))  # 预测分布。
    )
    return js  # 返回 Jensen-Shannon 距离（数值越小越接近）。
