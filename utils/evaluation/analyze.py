"""分子稳定性与分布评估工具集合。"""

# 总结：
# - 提供多种统计距离（EMD、KL、JS）与键长度判断，用于分析生成分子的稳定性。
# - `check_stability` 根据原子坐标和类型判断键稳定性；`analyze_stability_for_molecules` 汇总批量结果。
# - 辅助函数支持直方图归一化、坐标距离转换以及原子类型与键级的映射。

import torch  # 导入 PyTorch，用于张量运算。
import numpy as np  # 导入 NumPy，用于数值计算。
import scipy.stats as sp_stats  # 导入 SciPy 统计模块，提供统计距离函数。

atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}  # 定义元素到原子序号的映射。
atom_decoder = {v: k for k, v in atom_encoder.items()}  # 构建反向映射，便于通过原子序号取元素符号。

# Bond lengths from http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html  # 保留注释：键长数据来源。
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92, 'P': 144, 'S': 134, 'Cl': 127},  # 单键经验长度。
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135, 'P': 184, 'S': 182, 'Cl': 177},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136, 'P': 177, 'S': 168, 'Cl': 175},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142, 'P': 163, 'S': 151, 'Cl': 164},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142, 'P': 156, 'S': 158, 'Cl': 166},
          'P': {'H': 144, 'C': 184, 'N': 177, 'O': 163, 'F': 156, 'P': 221, 'S': 210, 'Cl': 203},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'F': 158, 'P': 210, 'S': 204, 'Cl': 207},
          'Cl': {'H': 127, 'C': 177, 'N': 175, 'O': 164, 'F': 166, 'P': 203, 'S': 207, 'Cl': 199}
          }

bonds2 = {'H': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},  # 双键经验长度。
          'C': {'H': -1, 'C': 134, 'N': 129, 'O': 120, 'F': -1, 'P': -1, 'S': 160, 'Cl': -1},
          'N': {'H': -1, 'C': 129, 'N': 125, 'O': 121, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'O': {'H': -1, 'C': 120, 'N': 121, 'O': 121, 'F': -1, 'P': 150, 'S': -1, 'Cl': -1},
          'F': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'P': {'H': -1, 'C': -1, 'N': -1, 'O': 150, 'F': -1, 'P': -1, 'S': 186, 'Cl': -1},
          'S': {'H': -1, 'C': 160, 'N': -1, 'O': -1, 'F': -1, 'P': 186, 'S': -1, 'Cl': -1},
          'Cl': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          }

bonds3 = {'H': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},  # 三键经验长度。
          'C': {'H': -1, 'C': 120, 'N': 116, 'O': 113, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'N': {'H': -1, 'C': 116, 'N': 110, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'O': {'H': -1, 'C': 113, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'F': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'P': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'S': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          'Cl': {'H': -1, 'C': -1, 'N': -1, 'O': -1, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1},
          }
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}  # 定义不同元素对应的标准差，用于键长容差。
margin1, margin2, margin3 = 10, 5, 3  # 不同键级的额外容差参数。

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 4, 'Cl': 1}  # 各元素允许的最大价键数。


def normalize_histogram(hist):  # 将直方图归一化为概率分布。
    hist = np.array(hist)  # 转换为 NumPy 数组。
    prob = hist / np.sum(hist)  # 归一化到总和为 1。
    return prob  # 返回概率分布。


def coord2distances(x):  # 将坐标张量转换为距离向量。
    x = x.unsqueeze(2)  # 在末尾增加一个维度，便于广播。
    x_t = x.transpose(1, 2)  # 交换维度以获取转置矩阵。
    dist = (x - x_t) ** 2  # 计算元素平方差。
    dist = torch.sqrt(torch.sum(dist, 3))  # 对平方差求和并开方得到距离矩阵。
    dist = dist.flatten()  # 展平为一维向量。
    return dist  # 返回距离向量。


def earth_mover_distance(h1, h2):  # 计算两个直方图的 Earth Mover's Distance。
    p1 = normalize_histogram(h1)  # 将第一个直方图归一化。
    p2 = normalize_histogram(h2)  # 将第二个直方图归一化。

    distance = sp_stats.wasserstein_distance(p1, p2)  # 使用 SciPy 计算 Wasserstein 距离。
    return distance  # 返回距离结果。


def kl_divergence(p1, p2):  # 计算 KL 散度 D(p1||p2)。
    return np.sum(p1 * np.log(p1 / p2))  # 根据定义逐元素计算。


def kl_divergence_sym(h1, h2):  # 计算对称 KL 散度。
    p1 = normalize_histogram(h1) + 1e-10  # 归一化并避免 log(0)。
    p2 = normalize_histogram(h2) + 1e-10  # 同理处理第二个直方图。

    kl = kl_divergence(p1, p2)  # 计算 D(p1||p2)。
    kl_flipped = kl_divergence(p2, p1)  # 计算 D(p2||p1)。

    return (kl + kl_flipped) / 2.  # 返回平均值作为对称 KL。


def js_divergence(h1, h2):  # 计算 Jensen-Shannon 散度。
    p1 = normalize_histogram(h1) + 1e-10  # 归一化第一个直方图。
    p2 = normalize_histogram(h2) + 1e-10  # 归一化第二个直方图。

    M = (p1 + p2) / 2  # 计算中间分布 M。
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2  # 根据定义求平均 KL。
    return js  # 返回 JS 散度。


def get_bond_order(atom1, atom2, distance):  # 根据距离推断键级。
    distance = 100 * distance  # We change the metric  # 原注释：将单位转换为皮米。

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:  # 首先检查是否落在单键范围内。
        thr_bond2 = bonds2[atom1][atom2] + margin2  # 计算双键阈值。
        if distance < thr_bond2:  # 若小于双键阈值。
            thr_bond3 = bonds3[atom1][atom2] + margin3  # 计算三键阈值。
            if distance < thr_bond3:  # 若小于三键阈值。
                return 3  # 判定为三键。
            return 2  # 否则为双键。
        return 1  # 大于双键阈值则视为单键。
    return 0  # 超出单键范围则视为无键连接。


def check_stability(positions, atom_type, debug=False, hs=False, return_nr_bonds=False):  # 检查分子稳定性。
    assert len(positions.shape) == 2  # 确保坐标矩阵为二维。
    assert positions.shape[1] == 3  # 保证每个坐标包含 x/y/z 三维。

    x = positions[:, 0]  # 提取 x 坐标。
    y = positions[:, 1]  # 提取 y 坐标。
    z = positions[:, 2]  # 提取 z 坐标。

    nr_bonds = np.zeros(len(x), dtype='int')  # 初始化每个原子的键计数。

    for i in range(len(x)):  # 遍历所有原子。
        for j in range(i + 1, len(x)):  # 遍历上三角避免重复。
            p1 = np.array([x[i], y[i], z[i]])  # 构建原子 i 的坐标。
            p2 = np.array([x[j], y[j], z[j]])  # 构建原子 j 的坐标。
            dist = np.sqrt(np.sum((p1 - p2) ** 2))  # 计算两原子之间的欧氏距离。
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]  # 将索引转换为元素符号。
            order = get_bond_order(atom1, atom2, dist)  # 根据距离推断键级。
            # if i == 0:
            #     print(j, order)
            nr_bonds[i] += order  # 将键级累加到原子 i。
            nr_bonds[j] += order  # 将键级累加到原子 j。

    nr_stable_bonds = 0  # 初始化稳定原子计数。
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):  # 遍历每个原子与其键数。
        if hs:  # 若启用严格模式（精确匹配价键）。
            is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i  # 判断是否完全匹配。
        else:
            is_stable = (allowed_bonds[atom_decoder[atom_type_i]] >= nr_bonds_i > 0)  # 判断是否在允许范围内且大于 0。
        if is_stable == False and debug:  # 若不稳定且开启调试。
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))  # 输出错误信息。
        nr_stable_bonds += int(is_stable)  # 根据布尔值累加稳定原子数量。

    molecule_stable = nr_stable_bonds == len(x)  # 若所有原子均稳定则整体稳定。
    if return_nr_bonds:  # 若需要返回每个原子的键计数。
        return molecule_stable, nr_stable_bonds, len(x), nr_bonds  # 返回整体是否稳定、稳定原子数量、总原子数以及各原子键数。
    else:
        return molecule_stable, nr_stable_bonds, len(x)  # 默认返回整体稳定性与统计信息。


def analyze_stability_for_molecules(molecule_list):  # 批量分析分子稳定性。
    n_samples = len(molecule_list)  # 分子总数。
    molecule_stable_list = []  # 用于存储稳定分子。

    molecule_stable = 0  # 统计稳定分子数量。
    nr_stable_bonds = 0  # 统计稳定原子数量。
    n_atoms = 0  # 统计所有分子的原子总数。

    for one_hot, x in molecule_list:  # 遍历每个分子的输入（one-hot 原子类型与坐标）。
        atom_type = one_hot.argmax(2).squeeze(0).cpu().detach().numpy()  # 将 one-hot 转为原子序号。
        x = x.squeeze(0).cpu().detach().numpy()  # 转换坐标为 NumPy 数组。

        validity_results = check_stability(x, atom_type)  # 检查当前分子的稳定性。

        molecule_stable += int(validity_results[0])  # 若稳定则增加稳定分子计数。
        nr_stable_bonds += int(validity_results[1])  # 累加稳定原子数量。
        n_atoms += int(validity_results[2])  # 累加原子总数。

        if validity_results[0]:  # 若分子整体稳定。
            molecule_stable_list.append((x, atom_type))  # 将其添加到稳定列表。

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)  # 计算稳定分子比例。
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)  # 计算稳定原子比例。
    validity_dict = {
        'mol_stable': fraction_mol_stable,  # 分子层面稳定比例。
        'atm_stable': fraction_atm_stable,  # 原子层面稳定比例。
    }

    # print('Validity:', validity_dict)

