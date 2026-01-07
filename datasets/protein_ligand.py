# 总结：
# - 提供解析蛋白-配体分子文件的工具函数，构建统一的数据字典。
# - 实现配体原子特征提取、SDF 文本解析与容错读取流程。
# - 支撑 PDBBind 等数据集的预处理与张量化操作。

import sys  # 导入系统模块以访问标准错误流等系统级功能。
from io import StringIO  # 导入 StringIO 以缓存文本流。

import numpy as np  # 导入 NumPy，用于数组和数值运算。
import torch  # 导入 PyTorch，用于构建张量。
from rdkit import Chem  # 从 RDKit 导入化学信息学核心接口。
from rdkit.Chem.rdchem import BondType, HybridizationType  # 导入键类型和杂化类型定义。
from torch_scatter import scatter  # 导入 scatter 函数以便进行分段聚合。

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']  # 定义原子功能家族列表。
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}  # 构建原子家族名称到索引的映射。
ATOM_FEATS = {'AtomicNumber': 1, 'Aromatic': 1, 'Degree': 6, 'NumHs': 6, 'Hybridization': len(HybridizationType.values)}  # 定义原子特征维度配置。
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}  # 将键类型名称映射至索引。
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}  # 将索引映射回键类型名称。
KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}  # 定义测量值类型与编码映射。


def get_ligand_atom_features(rdmol):  # 定义函数提取配体原子特征矩阵。
    """提取 RDKit 分子的基础原子特征矩阵。

    Args:
        rdmol: RDKit `Chem.Mol` 对象。

    Returns:
        np.ndarray: 形状为 `[num_atoms, 5]` 的整数矩阵，列依次为
        原子序号、芳香性、度数、氢原子计数与杂化类型索引。
    """
    num_atoms = rdmol.GetNumAtoms()  # 获取分子中原子数量。
    atomic_number = []  # 初始化原子序数组。
    aromatic = []  # 初始化芳香性标记数组。
    # sp, sp2, sp3 = [], [], []  # 保留注释：先前的杂化类型特征。
    hybrid = []  # 初始化杂化类型索引数组。
    degree = []  # 初始化键数度数组。
    for atom_idx in range(num_atoms):  # 遍历每个原子索引。
        atom = rdmol.GetAtomWithIdx(atom_idx)  # 按索引获取原子对象。
        atomic_number.append(atom.GetAtomicNum())  # 记录原子序号。
        aromatic.append(1 if atom.GetIsAromatic() else 0)  # 记录是否芳香。
        hybridization = atom.GetHybridization()  # 获取原子杂化类型。
        HYBRID_TYPES = {t: i for i, t in enumerate(HybridizationType.names.values())}  # 构造杂化类型映射。
        hybrid.append(HYBRID_TYPES[hybridization])  # 记录杂化类型索引。
        # sp.append(1 if hybridization == HybridizationType.SP else 0)  # 保留注释：原始特征构造。
        # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)  # 保留注释：原始特征构造。
        # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)  # 保留注释：原始特征构造。
        degree.append(atom.GetDegree())  # 记录原子度数。
    node_type = torch.tensor(atomic_number, dtype=torch.long)  # 将原子序号转换为长整型张量。

    row, col = [], []  # 初始化键连接的起点和终点列表。
    for bond in rdmol.GetBonds():  # 遍历分子中的所有键。
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # 获取键两端原子索引。
        row += [start, end]  # 在起点列表中追加双向索引。
        col += [end, start]  # 在终点列表中追加双向索引。
    row = torch.tensor(row, dtype=torch.long)  # 将起点列表转换为张量。
    col = torch.tensor(col, dtype=torch.long)  # 将终点列表转换为张量。
    hs = (node_type == 1).to(torch.float)  # 判断是否为氢原子并转换为浮点张量。
    num_hs = scatter(hs[row], col, dim_size=num_atoms).numpy()  # 聚合每个原子的氢原子数量并转换为 NumPy。
    # need to change ATOM_FEATS accordingly  # 保留注释：提示特征定义应同步更新。
    feat_mat = np.array([atomic_number, aromatic, degree, num_hs, hybrid], dtype=np.int64).transpose()  # 构造特征矩阵并转置为 (N, F)。
    return feat_mat  # 返回原子特征矩阵。


# used for fixing some errors in sdf file  # 保留注释：说明函数用途为修复 SDF 文件错误。
def parse_sdf_file_text(path):  # 定义函数以文本方式解析 SDF 文件。
    """以纯文本方式解析 SDF 文件，提取元素、坐标和键信息。

    Args:
        path: SDF 文件路径。

    Returns:
        dict: 包含 `element`、`pos`、`bond_index`、`bond_type`、`center_of_mass` 的字典。
    """
    with open(path, 'r') as f:  # 打开 SDF 文件读取文本。
        sdf = f.read()  # 将文件内容读入字符串。

    sdf = sdf.splitlines()  # 将文本按行切分为列表。
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])  # 从头部信息解析原子数和键数。
    ptable = Chem.GetPeriodicTable()  # 获取元素周期表以查询原子属性。

    element, pos = [], []  # 初始化元素列表和坐标列表。
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 初始化质心累积坐标。
    accum_mass = 0.0  # 初始化质量累积。
    for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):  # 遍历每一行原子记录并拆分字段。
        x, y, z = map(float, atom_line[:3])  # 读取原子坐标。
        symb = atom_line[3]  # 读取元素符号。
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())  # 通过周期表查询原子序号。
        element.append(atomic_number)  # 记录原子序号。
        pos.append([x, y, z])  # 记录原子坐标。

        atomic_weight = ptable.GetAtomicWeight(atomic_number)  # 查询原子量。
        accum_pos += np.array([x, y, z]) * atomic_weight  # 根据原子量加权累积坐标。
        accum_mass += atomic_weight  # 更新总质量。

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)  # 计算质心坐标。

    element = np.array(element, dtype=np.int64)  # 将元素列表转换为整数数组。
    pos = np.array(pos, dtype=np.float32)  # 将坐标列表转换为浮点数组。
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}  # 构造键类型映射。
    bond_type_map = {  # 定义 SDF 键类型编码到内部索引的映射。
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
        8: BOND_TYPES[BondType.UNSPECIFIED]
    }
    row, col, edge_type = [], [], []  # 初始化边的起点、终点与类型列表。
    for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:  # 遍历键记录行。
        start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1  # 解析键两端原子索引并调整为从零开始。
        row += [start, end]  # 添加起点索引。
        col += [end, start]  # 添加终点索引。
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]  # 按 SDF 编码查表添加边类型，双向写入。

    edge_index = np.array([row, col], dtype=np.int64)  # 将索引列表组合成 NumPy 数组。
    edge_type = np.array(edge_type, dtype=np.int64)  # 将边类型列表转换为数组。

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()  # 计算边的排序索引以保证稳定顺序。
    edge_index = edge_index[:, perm]  # 按排序结果重排边索引。
    edge_type = edge_type[perm]  # 按排序结果重排边类型。

    data = {  # 构造返回数据字典。
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass
    }
    return data  # 返回解析后的数据。


# used for preparing the dataset  # 保留注释：用于准备数据集。
def read_mol(sdf_fileName, mol2_fileName, verbose=False):  # 定义函数尝试读取分子文件并进行容错。
    """尝试读取 SDF / MOL2 文件并净化分子，必要时回退到备选格式。

    Args:
        sdf_fileName: 主 SDF 文件路径。
        mol2_fileName: 备选 MOL2 文件路径。
        verbose: 是否打印 RDKit 产生的日志。

    Returns:
        tuple: `(mol, problem, ligand_path)`，分别为 RDKit 分子对象、
        是否仍存在问题、成功解析的文件路径。
    """
    Chem.WrapLogs()  # 包装 RDKit 日志以便捕获输出。
    stderr = sys.stderr  # 备份当前标准错误流。
    sio = sys.stderr = StringIO()  # 将标准错误流重定向到字符串缓冲区。
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)  # 尝试从 SDF 文件加载分子并暂不净化。
    problem = False  # 初始化问题标记为否。
    ligand_path = None  # 初始化配体路径为空。
    try:  # 尝试对分子进行净化处理。
        Chem.SanitizeMol(mol)  # 净化分子结构。
        mol = Chem.RemoveHs(mol)  # 移除氢原子。
        sm = Chem.MolToSmiles(mol)  # 生成 SMILES 用于验证。
        ligand_path = sdf_fileName  # 标记成功读取的文件路径。
    except Exception as e:  # 捕获可能的异常。
        sm = str(e)  # 记录错误信息。
        problem = True  # 标记出现问题。
    if problem:  # 若 SDF 读取失败则尝试 mol2 文件。
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)  # 从 mol2 文件加载分子。
        problem = False  # 重置问题标记。
        try:  # 再次执行净化步骤。
            Chem.SanitizeMol(mol)  # 净化分子结构。
            mol = Chem.RemoveHs(mol)  # 移除氢原子。
            sm = Chem.MolToSmiles(mol)  # 生成 SMILES 验证。
            problem = False  # 标记成功。
            ligand_path = mol2_fileName  # 记录成功读取的 mol2 路径。
        except Exception as e:  # 如果再次失败。
            sm = str(e)  # 记录错误信息。
            problem = True  # 标记依旧存在问题。

    if verbose:  # 若需要输出详细日志。
        print(sio.getvalue())  # 打印捕获的 RDKit 日志。
    sys.stderr = stderr  # 恢复系统原始标准错误流。
    return mol, problem, ligand_path  # 返回分子对象、问题标记和成功读取的路径。


def parse_sdf_file_mol(path, heavy_only=True, mol=None):  # 定义函数将 SDF/mol2 文件解析为统一数据字典。
    """解析分子文件并产出统一的配体特征字典。

    Args:
        path: SDF 或 MOL2 文件路径。
        heavy_only: 若为 True，则移除氢原子。
        mol: 已经加载好的 `Chem.Mol`，可跳过重复读取。

    Returns:
        dict: 包含坐标、原子序号、键索引/类型及质心等信息的字典。
    """
    if mol is None:  # 若未传入预载分子对象则从文件读取。
        if path.endswith('.sdf'):  # 判断文件后缀为 SDF。
            mol = Chem.MolFromMolFile(path, sanitize=False)  # 从 SDF 文件读取分子。
        elif path.endswith('.mol2'):  # 判断文件后缀为 mol2。
            mol = Chem.MolFromMol2File(path, sanitize=False)  # 从 mol2 文件读取分子。
        else:  # 对于不支持的格式。
            raise ValueError  # 抛出异常提示。
        Chem.SanitizeMol(mol)  # 对读取的分子进行净化。
        if heavy_only:  # 如果只保留重原子。
            mol = Chem.RemoveHs(mol)  # 移除氢原子。
    # mol = next(iter(Chem.SDMolSupplier(path, removeHs=heavy_only)))  # 保留注释：替代的供应器读取方式。
    feat_mat = get_ligand_atom_features(mol)  # 调用特征提取函数获得原子特征。

    # fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')  # 保留注释：特征工厂配置。
    # factory = ChemicalFeatures.BuildFeatureFactory(fdefName)  # 保留注释：构建化学特征工厂。
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=heavy_only)))  # 保留注释：替代读取方式。
    # rd_num_atoms = rdmol.GetNumAtoms()  # 保留注释：原子数量。
    # feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.long)  # 保留注释：初始化特征矩阵。
    # for feat in factory.GetFeaturesForMol(rdmol):  # 保留注释：遍历化学特征。
    #     feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1  # 保留注释：标记特征。

    ptable = Chem.GetPeriodicTable()  # 获取元素周期表对象。

    num_atoms = mol.GetNumAtoms()  # 获取分子中的原子数量。
    num_bonds = mol.GetNumBonds()  # 获取分子中的键数量。
    pos = mol.GetConformer().GetPositions()  # 获取当前构象的原子坐标。

    element = []  # 初始化元素列表。
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 初始化质心累加坐标。
    accum_mass = 0.0  # 初始化质量累加值。
    for atom_idx in range(num_atoms):  # 遍历每个原子索引。
        atom = mol.GetAtomWithIdx(atom_idx)  # 获取对应原子对象。
        atomic_number = atom.GetAtomicNum()  # 获取原子序号。
        element.append(atomic_number)  # 记录原子序号。
        x, y, z = pos[atom_idx]  # 获取原子坐标。
        atomic_weight = ptable.GetAtomicWeight(atomic_number)  # 查询原子量。
        accum_pos += np.array([x, y, z]) * atomic_weight  # 根据原子量更新加权坐标。
        accum_mass += atomic_weight  # 累加质量。
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)  # 计算分子质心。
    element = np.array(element, dtype=np.int64)  # 将元素列表转换为整数数组。
    pos = np.array(pos, dtype=np.float32)  # 将坐标转换为浮点数组。

    row, col, edge_type = [], [], []  # 初始化边索引与类型列表。
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}  # 构建键类型映射。
    for bond in mol.GetBonds():  # 遍历分子中所有键。
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # 获取键的起止原子索引。
        row += [start, end]  # 添加起点索引。
        col += [end, start]  # 添加终点索引。
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]  # 记录双向边的类型索引。
    edge_index = np.array([row, col], dtype=np.long)  # 将边索引组装为数组。
    edge_type = np.array(edge_type, dtype=np.long)  # 将边类型转换为数组。
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()  # 计算重排顺序以保证一致性。
    edge_index = edge_index[:, perm]  # 按排序结果重排边索引。
    edge_type = edge_type[perm]  # 按排序结果重排边类型。

    data = {  # 构造最终输出字典。
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat
    }
    return data  # 返回解析结果。
