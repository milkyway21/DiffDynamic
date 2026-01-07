# 总结：
# - 定义蛋白-配体图数据结构以及对应的批量装载器。
# - 提供工具函数将原始字典转换为张量形式。
# - 提供根据批次生成连接矩阵的便捷函数。

import torch  # 导入 PyTorch，用于张量运算。
import torch_scatter  # 导入 torch_scatter，用于分段聚合操作。
import numpy as np  # 导入 NumPy，用于数组处理。
from torch_geometric.data import Data  # 从 PyG 导入 Data 基类定义图数据。
from torch_geometric.loader import DataLoader  # 从 PyG 导入 DataLoader 处理批量加载。

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',)  # 定义需要跟踪批次偏移的字段。


class ProteinLigandData(Data):  # 定义蛋白-配体图数据类，继承自 PyG Data。
    """蛋白-配体联合图数据结构，约定属性以 `protein_` / `ligand_` 为前缀。"""

    def __init__(self, *args, **kwargs):  # 初始化方法，接受任意参数传递给父类。
        super().__init__(*args, **kwargs)  # 调用父类构造函数完成基础初始化。

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):  # 静态方法：从蛋白和配体字典构造实例。
        """将蛋白/配体属性字典打包为 `ProteinLigandData` 实例。

        Args:
            protein_dict: 包含蛋白节点属性的字典（键为属性名、值为张量或数组）。
            ligand_dict: 包含配体节点/键属性的字典。
            **kwargs: 额外传递给 `ProteinLigandData` 构造函数的字段（例如标签）。

        Returns:
            ProteinLigandData: 带有标准化前缀及邻接列表的图对象。
        """
        instance = ProteinLigandData(**kwargs)  # 构造新的 ProteinLigandData 实例。

        if protein_dict is not None:  # 若提供蛋白字典。
            for key, item in protein_dict.items():  # 遍历蛋白字典的键值对。
                instance['protein_' + key] = item  # 为每一项添加前缀后存入实例。

        if ligand_dict is not None:  # 若提供配体字典。
            for key, item in ligand_dict.items():  # 遍历配体字典的键值对。
                instance['ligand_' + key] = item  # 为每一项添加前缀后存入实例。

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])  # 构造配体邻接列表字典。
                                                  if instance.ligand_bond_index[0, k].item() == i]  # 保留满足起点匹配的邻居索引。
                                       for i in instance.ligand_bond_index[0]}  # 遍历所有配体边的起点索引。
        return instance  # 返回构造完成的实例。

    def __inc__(self, key, value, *args, **kwargs):  # 重写 PyG 的增量函数以控制批处理偏移。
        if key == 'ligand_bond_index':  # 针对配体键索引需要特殊处理。
            return self['ligand_element'].size(0)  # 返回配体原子数量作为偏移量。
        else:  # 其他字段保持默认行为。
            return super().__inc__(key, value)  # 调用父类实现。


class ProteinLigandDataLoader(DataLoader):  # 定义蛋白-配体数据加载器，继承自 PyG DataLoader。
    """针对 `ProteinLigandData` 预设 `follow_batch` 的便捷 DataLoader。"""

    def __init__(  # 初始化加载器设置批量参数。
            self,  # self 引用实例本身。
            dataset,  # dataset 指定待加载的数据集。
            batch_size=1,  # batch_size 设置一次加载的样本数。
            shuffle=False,  # shuffle 控制是否打乱样本。
            follow_batch=FOLLOW_BATCH,  # follow_batch 设置需要跟踪批次信息的字段列表。
            **kwargs  # kwargs 允许传入额外的 DataLoader 配置。
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)  # 调用父类构造函数完成初始化。


def torchify_dict(data):  # 定义工具函数，将字典中的数组转换为张量。
    """将字典中的 NumPy 数组就地转换为 PyTorch 张量。

    Args:
        data: 字典，值可包含 ndarray 或原生类型。

    Returns:
        dict: 新字典，保持原键，数组类型被替换为张量。
    """
    output = {}  # 初始化输出字典。
    for k, v in data.items():  # 遍历输入字典的键值对。
        if isinstance(v, np.ndarray):  # 若值为 NumPy 数组。
            output[k] = torch.from_numpy(v)  # 转换为 PyTorch 张量。
        else:  # 对于其它类型。
            output[k] = v  # 保持原样复制到输出字典。
    return output  # 返回转换后的字典。


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):  # 定义函数生成批次连接矩阵。
    """根据批次信息构造配体分子的键连接矩阵列表。

    Args:
        ligand_batch: 每个配体原子所属的批次索引。
        ligand_bond_index: 形状 `[2, num_bonds]` 的边索引。
        ligand_bond_type: 每条边对应的键类型（1/2/3/4 等）。
        ligand_bond_batch: 每条边所属批次索引。

    Returns:
        list[torch.Tensor]: `len == batch_size` 的方阵列表，每个方阵表示对应样本的键连接关系。
    """
    batch_ligand_size = torch_scatter.segment_coo(  # 计算每个批次的配体原子数量。
        torch.ones_like(ligand_batch),  # 为每个原子创建值为 1 的张量。
        ligand_batch,  # 指定批次索引以聚合。
        reduce='sum',  # 使用求和聚合得到每批原子数量。
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size  # 计算每个批次在全局索引中的偏移。
    batch_size = len(batch_index_offset)  # 获取批次数量。
    batch_connectivity_matrix = []  # 初始化存储所有批次连通矩阵的列表。
    for batch_index in range(batch_size):  # 遍历每个批次。
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]  # 筛选当前批次的键索引。
        start_index -= batch_index_offset[batch_index]  # 调整起点索引到局部范围。
        end_index -= batch_index_offset[batch_index]  # 调整终点索引到局部范围。
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]  # 取得当前批次的键类型。
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.  # 保留注释：说明矩阵含义。
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],  # 创建方阵存储连接关系。
                                          dtype=torch.int)  # 设置矩阵元素类型为整数。
        for s, e, t in zip(start_index, end_index, bond_type):  # 遍历每条键及其类型。
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t  # 在矩阵中标记双向连接和键类型。
        batch_connectivity_matrix.append(connectivity_matrix)  # 将当前批次矩阵添加到列表。
    return batch_connectivity_matrix  # 返回所有批次的连接矩阵列表。
