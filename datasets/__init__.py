# 总结：
# - 提供数据集工厂函数以加载多种数据集类型。
# - 支持通过拆分文件创建可复现的子集划分。
# - 根据配置返回原始数据集或连同子集一起返回。

import torch  # 导入 PyTorch，用于张量相关操作和序列化加载。
from torch.utils.data import Subset  # 导入 Subset，用于基于索引列表构造数据集视图。
from .pl_pair_dataset import PocketLigandPairDataset  # 导入口袋-配体配对数据集实现。
from .pdbbind import PDBBindDataset  # 导入 PDBBind 数据集实现。


def get_dataset(config, *args, **kwargs):  # 定义数据集工厂函数，接收配置和可选参数。
    """根据配置名称实例化数据集，并在需要时返回拆分子集。

    Args:
        config: 含 `name`、`path` 以及可选 `split` 字段的配置对象（`EasyDict` 或命名空间）。
        *args: 透传给底层数据集构造器的额外位置参数。
        **kwargs: 透传给底层数据集构造器的额外关键字参数。

    Returns:
        Dataset 或 `(Dataset, dict[str, Subset])`:
            - 未提供 `split` 时直接返回完整数据集实例。
            - 提供拆分文件时返回 `(dataset, subsets)`，其中 `subsets` 为包含 `train/val/test` 等键的字典。
    """
    name = config.name  # 从配置中读取数据集名称。
    root = config.path  # 从配置中读取数据集根路径。
    if name == 'pl':  # 判断是否请求口袋-配体数据集。
        dataset = PocketLigandPairDataset(root, *args, **kwargs)  # 根据根路径和额外参数实例化口袋-配体数据集。
    elif name == 'pdbbind':  # 判断是否请求 PDBBind 数据集。
        dataset = PDBBindDataset(root, *args, **kwargs)  # 根据根路径和额外参数实例化 PDBBind 数据集。
    else:  # 如果名称不在支持列表中。
        raise NotImplementedError('Unknown dataset: %s' % name)  # 抛出未实现错误并指明非法数据集名称。

    if 'split' in config:  # 判断配置中是否提供数据集拆分信息。
        split = torch.load(config.split)  # 从磁盘加载序列化的拆分索引字典。
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}  # 使用索引字典构造命名子集。
        return dataset, subsets  # 返回原始数据集以及对应的子集映射。
    else:  # 如果未指定拆分信息。
        return dataset  # 仅返回原始数据集。
