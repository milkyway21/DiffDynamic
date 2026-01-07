import torch  # 导入 PyTorch。
import torch.nn.functional as F  # 导入函数式接口。
from torch_geometric.nn import knn_graph  # 导入 kNN 图构建工具。
from torch_geometric.utils.num_nodes import maybe_num_nodes  # 导入节点计数工具。
from torch_scatter import scatter_add  # 导入加法散射函数。

from datasets.pl_data import ProteinLigandData  # 导入蛋白-配体数据结构。
from datasets.protein_ligand import ATOM_FEATS  # 导入原子特征配置。


class FeaturizeProteinAtom(object):  # 构造蛋白原子的独热特征。
    """为性质预测任务构造蛋白原子特征。"""

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        """生成蛋白原子特征并写入 `protein_atom_feature`。"""
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):  # 构造配体原子的特征向量。
    """按 RDKit 导出特征构造配体原子输入向量。"""
    
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        # self.n_degree = torch.LongTensor([0, 1, 2, 3, 4, 5])  # 0 - 5
        # self.n_num_hs = 6  # 0 - 5

    @property
    def num_properties(self):
        return sum(ATOM_FEATS.values())

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.num_properties

    def __call__(self, data: ProteinLigandData):
        """拼接元素 one-hot 与原子级属性，写入 `ligand_atom_feature_full`。"""
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # convert some features to one-hot vectors
        atom_feature = []
        for i, (k, v) in enumerate(ATOM_FEATS.items()):
            feat = data.ligand_atom_feature[:, i:i+1]
            if v > 1:
                feat = (feat == torch.LongTensor(list(range(v))).view(1, -1))
            else:
                if k == 'AtomicNumber':
                    feat = feat / 100.
            atom_feature.append(feat)

        # atomic_number = data.ligand_atom_feature[:, 0:1]
        # aromatic = data.ligand_atom_feature[:, 1:2]
        # degree = data.ligand_atom_feature[:, 2:3] == self.n_degree.view(1, -1)
        # num_hs = F.one_hot(data.ligand_atom_feature[:, 3], num_classes=self.n_num_hs)
        # data.ligand_atom_feature_full = torch.cat([element, atomic_number, aromatic, degree, num_hs], dim=-1)

        atom_feature = torch.cat(atom_feature, dim=-1)
        data.ligand_atom_feature_full = torch.cat([element, atom_feature], dim=-1)
        return data


class FeaturizeLigandBond(object):  # 配体键类型 one-hot 编码。
    """将配体键类型映射为固定长度的 one-hot 表示。"""

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        """写入 `ligand_bond_feature`。"""
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=4)    # (1,2,3) to (0,1,2)-onehot
        return data


class LigandCountNeighbors(object):  # 统计配体原子的邻居数量和价数。
    """统计配体原子的邻居数量与价数。"""

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        """计算邻居计数与价数并写入数据对象。"""
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        return data


class EdgeConnection(object):  # 构建蛋白-配体的边连接。
    """根据指定策略（目前为 kNN）构造蛋白-配体边。"""
    def __init__(self, kind, k):
        super(EdgeConnection, self).__init__()
        self.kind = kind
        self.k = k

    def __call__(self, data):
        """在数据对象上写入 `edge_index`。"""
        pos = torch.cat([data.protein_pos, data.ligand_pos], dim=0)
        if self.kind == 'knn':
            data.edge_index = knn_graph(pos, k=self.k, flow='target_to_source')
        return data


def convert_to_single_emb(x, offset=128):  # 将多维离散特征映射为单通道嵌入索引。
    """将多列离散特征平移到互不重叠的区间，方便共享嵌入。"""
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
