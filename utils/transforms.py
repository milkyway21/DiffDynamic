import torch  # 导入 PyTorch。
import torch.nn.functional as F  # 导入函数式接口。
import numpy as np  # 导入 NumPy。

from datasets.pl_data import ProteinLigandData  # 导入蛋白-配体数据结构。
from utils import data as utils_data  # 导入数据工具模块。

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']  # 获取芳香性特征在 RDKit 特征中的索引。

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist  # 保留原注释：限制元素种类。
MAP_ATOM_TYPE_FULL_TO_INDEX = {  # 将 (原子序号, 杂化, 芳香性) 映射到离散索引。
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {  # 仅依据原子序号的映射。
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {  # 同时考虑原子序号与芳香性标记的映射。
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}  # 反向映射：索引 -> 原子序号。
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}  # 反向映射：索引 -> (原子序号, 芳香性)。
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}  # 反向映射：索引 -> (原子序号, 杂化, 芳香性)。


def get_atomic_number_from_index(index, mode):  # 根据不同编码模式将索引还原为原子序号。
    """根据编码模式将类别索引转换为原子序号序列。"""
    if mode == 'basic':  # 基础模式：直接取序号。
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':  # 芳香模式：只取元组中的原子序号。
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'full':  # 完整模式：元组第一项为原子序号。
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError  # 未知模式抛出异常。
    return atomic_number  # 返回原子序号列表。


def is_aromatic_from_index(index, mode):  # 根据索引判断是否芳香。
    """根据编码模式还原是否芳香的布尔列表。"""
    if mode == 'add_aromatic':  # 芳香模式：取元组第二项。
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'full':  # 完整模式：取元组第三项。
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':  # 基础模式未编码芳香信息。
        is_aromatic = None
    else:
        raise ValueError  # 未知模式。
    return is_aromatic  # 返回布尔列表或 None。


def get_hybridization_from_index(index, mode):  # 根据索引还原杂化类型。
    """在 `full` 模式下将索引映射回杂化类型字符串。"""
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError  # 仅在完整模式下可用。
    return hybridization  # 返回杂化类型列表。


def get_index(atom_num, hybridization, is_aromatic, mode):  # 根据原子属性获取离散索引。
    """根据原子属性与编码模式生成离散类别索引。"""
    if mode == 'basic':  # 基础模式仅使用原子序号。
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':  # 芳香模式考虑原子序号与芳香标记。
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:  # 处理未定义组合，输出提示并回退到 (H, 非芳香)。
            print(int(atom_num), bool(is_aromatic))
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    else:  # 完整模式同时使用序号、杂化与芳香信息。
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]


class FeaturizeProteinAtom(object):  # 构造蛋白原子的一热特征。
    """为蛋白质节点构造元素/氨基酸/主链的组合特征。"""

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se 允许的元素序号。
        self.max_num_aa = 20  # 氨基酸种类数量。

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1  # 特征维度 = 元素数量 + 氨基酸数量 + 主链标记。

    def __call__(self, data: ProteinLigandData):
        """生成蛋白原子特征并写入 `protein_atom_feature`。"""
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)  # 元素指示矩阵。
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)  # 氨基酸一热编码。
        is_backbone = data.protein_is_backbone.view(-1, 1).long()  # 主链标记。
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)  # 拼接所有分量。
        data.protein_atom_feature = x  # 写回数据对象。
        return data  # 返回修改后的数据。


class FeaturizeLigandAtom(object):  # 构造配体原子离散特征索引。
    """将配体原子映射为可训练的离散类别索引。"""

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']  # 校验模式合法。
        self.mode = mode  # 保存编码模式。

    @property
    def feature_dim(self):
        if self.mode == 'basic':  # 基础模式特征数。
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':  # 芳香模式特征数。
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)  # 完整模式特征数。

    def __call__(self, data: ProteinLigandData):
        """根据模式编码配体原子并写入 `ligand_atom_feature_full`。"""
        element_list = data.ligand_element  # 配体原子元素序号列表。
        
        # 处理空配体情况（从 PDB 文件采样时）
        if len(element_list) == 0:
            data.ligand_atom_feature_full = torch.empty(0, dtype=torch.long)
            return data
        
        # 检查是否有必要的属性（空配体可能没有这些属性）
        if not hasattr(data, 'ligand_hybridization') or len(data.ligand_hybridization) == 0:
            # 如果没有杂化信息，创建默认值
            hybridization_list = [0] * len(element_list)
        else:
            hybridization_list = data.ligand_hybridization  # 配体杂化类型列表。
        
        if not hasattr(data, 'ligand_atom_feature') or len(data.ligand_atom_feature) == 0:
            # 如果没有原子特征，创建默认值
            aromatic_list = [0] * len(element_list)
        else:
            aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]  # 通过 RDKit 特征判定芳香性。

        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]  # 映射为索引。
        x = torch.tensor(x)  # 转换为张量。
        data.ligand_atom_feature_full = x  # 写回数据对象。
        return data  # 返回修改后的数据。


class FeaturizeLigandBond(object):  # 构造配体键的 one-hot 特征。
    """将配体键类型转换为 one-hot 特征向量。"""

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        """写入 `ligand_bond_feature`，作为键类型 one-hot。"""
        # 处理空配体情况（从 PDB 文件采样时）
        if not hasattr(data, 'ligand_bond_type') or len(data.ligand_bond_type) == 0:
            data.ligand_bond_feature = torch.empty(0, len(utils_data.BOND_TYPES), dtype=torch.long)
        else:
            data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.BOND_TYPES))  # 键类型 one-hot。
        return data  # 返回数据。


class RandomRotation(object):  # 对配体与蛋白坐标应用随机旋转。
    """采样随机正交矩阵，同步旋转蛋白/配体坐标。"""

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)  # 随机生成 3x3 矩阵。
        Q, __ = np.linalg.qr(M)  # 进行 QR 分解得到正交矩阵。
        Q = torch.from_numpy(Q.astype(np.float32))  # 转换为 torch 张量。
        data.ligand_pos = data.ligand_pos @ Q  # 旋转配体坐标。
        data.protein_pos = data.protein_pos @ Q  # 旋转蛋白坐标。
        return data  # 返回旋转后的数据。
