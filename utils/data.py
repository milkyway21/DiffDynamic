# 总结：
# - 提供蛋白质结构解析、邻域查询与口袋重建等数据处理工具。
# - 定义常用的原子/键特征映射，并生成 RDKit 兼容的分子与特征格式。
# - 支持将配体和蛋白数据转换为图数据所需的字典结构。

import os  # 导入操作系统模块，用于路径处理。
import numpy as np  # 导入 NumPy，用于数值运算。
from rdkit import Chem  # 从 RDKit 导入化学核心接口。
from rdkit.Chem.rdchem import BondType  # 导入键类型枚举。
from rdkit.Chem import ChemicalFeatures  # 导入化学特征工厂。
from rdkit import RDConfig  # 导入 RDKit 配置对象以找到数据文件。

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']  # 定义 RDKit 化学特征家族列表。
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}  # 将家族名称映射到索引。
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}  # 将 RDKit 键类型映射为整数编码。
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}  # 提供编码到字符串名称的反向映射。
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']  # 定义常见杂化类型。
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}  # 杂化类型名称到索引的映射。


class PDBProtein(object):  # 解析 PDB 蛋白质文件并生成结构化信息的工具类。
    """PDB 解析器：提供原子/残基级数据访问及邻域裁剪工具。"""
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }  # 定义氨基酸三字母到单字母符号的映射。

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }  # 将氨基酸名称映射到整数索引。

    BACKBONE_NAMES = ["CA", "C", "N", "O"]  # 定义蛋白主链原子名称。

    def __init__(self, data, mode='auto'):
        """解析 PDB 文件路径或文本块并缓存结构信息。"""
        super().__init__()  # 调用基类构造函数。
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':  # 根据模式判断是路径还是原始文本。
            with open(data, 'r') as f:  # 从文件读取 PDB 文本。
                self.block = f.read()
        else:
            self.block = data  # 直接使用传入的文本块。

        self.ptable = Chem.GetPeriodicTable()  # 获取周期表工具以查询原子性质。

        # Molecule properties  # 保留注释：以下初始化分子级属性。
        self.title = None  # PDB 标题信息。
        # Atom properties  # 保留注释：以下初始化原子级属性容器。
        self.atoms = []  # 存储原子详细信息。
        self.element = []  # 存储原子序号。
        self.atomic_weight = []  # 存储原子质量。
        self.pos = []  # 存储原子坐标。
        self.atom_name = []  # 存储原子名称。
        self.is_backbone = []  # 标记是否为主链原子。
        self.atom_to_aa_type = []  # 存储原子对应的氨基酸类型索引。
        # Residue properties  # 保留注释：以下初始化残基级属性。
        self.residues = []  # 残基列表。
        self.amino_acid = []  # 残基的氨基酸索引列表。
        self.center_of_mass = []  # 残基质心坐标。
        self.pos_CA = []  # 残基 CA 原子坐标。
        self.pos_C = []  # 残基 C 原子坐标。
        self.pos_N = []  # 残基 N 原子坐标。
        self.pos_O = []  # 残基 O 原子坐标。

        self._parse()  # 解析 PDB 文本填充上述属性。

    def _enum_formatted_atom_lines(self):  # 枚举并标准化 PDB 文件中的原子与表头行。
        for line in self.block.splitlines():  # 遍历每一行。
            if line[0:6].strip() == 'ATOM':  # 处理原子记录。
                element_symb = line[76:78].strip().capitalize()  # 首选列 77-78 的元素符号。
                if len(element_symb) == 0:  # 若缺失则退回到原子名称首字母。
                    element_symb = line[13:14]
                yield {  # 生成标准化字典。
                    'line': line,  # 原始文本行。
                    'type': 'ATOM',  # 行类型。
                    'atom_id': int(line[6:11]),  # 原子编号。
                    'atom_name': line[12:16].strip(),  # 原子名称。
                    'res_name': line[17:20].strip(),  # 残基名称。
                    'chain': line[21:22].strip(),  # 链 ID。
                    'res_id': int(line[22:26]),  # 残基编号。
                    'res_insert_id': line[26:27].strip(),  # 插入标记。
                    'x': float(line[30:38]),  # X 坐标。
                    'y': float(line[38:46]),  # Y 坐标。
                    'z': float(line[46:54]),  # Z 坐标。
                    'occupancy': float(line[54:60]),  # 占有率。
                    'segment': line[72:76].strip(),  # 片段标签。
                    'element_symb': element_symb,  # 元素符号。
                    'charge': line[78:80].strip(),  # 电荷。
                }
            elif line[0:6].strip() == 'HEADER':  # 处理表头信息。
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':  # 多模型 PDB 时只读取第一个模型。
                break  # Some PDBs have more than 1 model.

    def _parse(self):  # 解析 PDB 文本填充原子与残基信息。
        # Process atoms  # 保留注释：先处理原子行。
        residues_tmp = {}  # 临时存储残基信息的字典。
        for atom in self._enum_formatted_atom_lines():  # 遍历标准化行。
            if atom['type'] == 'HEADER':  # 处理表头。
                self.title = atom['value'].lower()  # 记录标题。
                continue  # 跳过后续逻辑。
            self.atoms.append(atom)  # 存储原子字典。
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])  # 获取原子序号。
            next_ptr = len(self.element)  # 当前原子在列表中的索引。
            self.element.append(atomic_number)  # 记录原子序号。
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))  # 记录原子量。
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))  # 存储坐标。
            self.atom_name.append(atom['atom_name'])  # 存储原子名称。
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)  # 标记是否为主链原子。
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])  # 记录所属氨基酸类型。

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])  # 构造唯一残基 ID。
            if chain_res_id not in residues_tmp:  # 首次遇到该残基。
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:  # 已存在残基，追加原子索引。
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']  # 校验残基名称一致。
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']  # 校验链 ID 一致。
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)  # 追加原子索引。

        # Process residues  # 保留注释：再处理残基属性。
        self.residues = [r for _, r in residues_tmp.items()]  # 将临时字典转为列表。
        for residue in self.residues:  # 遍历每个残基。
            sum_pos = np.zeros([3], dtype=np.float32)  # 初始化质心累积。
            sum_mass = 0.0  # 初始化质量累积。
            for atom_idx in residue['atoms']:  # 遍历残基内原子。
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]  # 质量加权坐标。
                sum_mass += self.atomic_weight[atom_idx]  # 累加质量。
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:  # 若为主链原子。
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]  # 记录对应坐标。
            residue['center_of_mass'] = sum_pos / sum_mass  # 计算质心坐标。

        # Process backbone atoms of residues  # 保留注释：补全主链信息列表。
        for residue in self.residues:  # 遍历残基。
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])  # 记录氨基酸编号。
            self.center_of_mass.append(residue['center_of_mass'])  # 记录质心。
            for name in self.BACKBONE_NAMES:  # 遍历主链原子名称。
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:  # 若残基中存在该主链原子。
                    getattr(self, pos_key).append(residue[pos_key])  # 保存对应坐标。
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])  # 缺失时使用质心近似。

    def to_dict_atom(self):  # 导出原子级别的字典表示。
        """以字典形式导出原子属性，便于构建图数据。"""
        return {
            'element': np.array(self.element, dtype=np.int64),  # 原子序号数组。
            'molecule_name': self.title,  # 分子名称。
            'pos': np.array(self.pos, dtype=np.float32),  # 原子坐标。
            'is_backbone': np.array(self.is_backbone, dtype=bool),  # 主链标记。
            'atom_name': self.atom_name,  # 原子名称列表。
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)  # 原子对应氨基酸编号。
        }

    def to_dict_residue(self):  # 导出残基级别的字典表示。
        """导出残基级属性（质心和主链坐标）。"""
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),  # 残基氨基酸类型索引。
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),  # 残基质心。
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),  # CA 坐标。
            'pos_C': np.array(self.pos_C, dtype=np.float32),  # C 坐标。
            'pos_N': np.array(self.pos_N, dtype=np.float32),  # N 坐标。
            'pos_O': np.array(self.pos_O, dtype=np.float32),  # O 坐标。
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):  # 基于半径从残基列表中筛选邻域。
        """基于距离阈值筛选邻域残基。

        Args:
            center: 三维坐标（长度 3）。
            radius: 距离阈值。
            criterion: 使用残基的哪种位置属性进行距离计算。

        Returns:
            list: 满足条件的残基字典集合。
        """
        center = np.array(center).reshape(3)  # 将中心坐标转换为长度为 3 的数组。
        selected = []  # 初始化筛选结果。
        for residue in self.residues:  # 遍历所有残基。
            distance = np.linalg.norm(residue[criterion] - center, ord=2)  # 计算残基与中心的距离。
            print(residue[criterion], distance)  # 调试输出距离。
            if distance < radius:  # 若在半径范围内。
                selected.append(residue)  # 将残基加入结果。
        return selected  # 返回筛选残基。

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):  # 基于配体位置筛选邻域残基。
        """围绕配体原子筛选一定半径内的残基集合。"""
        selected = []  # 初始化结果列表。
        sel_idx = set()  # 记录已选残基索引，避免重复。
        # The time-complexity is O(mn).  # 保留注释：双重循环复杂度。
        for center in ligand['pos']:  # 遍历配体原子坐标。
            for i, residue in enumerate(self.residues):  # 遍历残基。
                distance = np.linalg.norm(residue[criterion] - center, ord=2)  # 计算距离。
                if distance < radius and i not in sel_idx:  # 若满足距离且未被选中过。
                    selected.append(residue)  # 添加残基。
                    sel_idx.add(i)  # 标记已选索引。
        return selected  # 返回结果列表。

    def residues_to_pdb_block(self, residues, name='POCKET'):  # 将残基集合导出为 PDB 文本块。
        """将给定残基集合导出为新的 PDB 文本片段。"""
        block = "HEADER    %s\n" % name  # 写入头信息。
        block += "COMPND    %s\n" % name  # 写入化合物信息。
        for residue in residues:  # 遍历残基。
            for atom_idx in residue['atoms']:  # 遍历残基原子索引。
                block += self.atoms[atom_idx]['line'] + "\n"  # 追加原始原子行。
        block += "END\n"  # 添加结尾标记。
        return block  # 返回生成的 PDB 字符串。


def parse_pdbbind_index_file(path):  # 读取 PDBBind 索引文件并返回 PDB ID 列表。
    """读取 PDBBind 索引文件，返回 ID 列表。"""
    pdb_id = []  # 初始化 ID 列表。
    with open(path, 'r') as f:  # 打开文件。
        lines = f.readlines()  # 读取所有行。
    for line in lines:  # 遍历每一行。
        if line.startswith('#'): continue  # 跳过注释行。
        pdb_id.append(line.split()[0])  # 提取第一列作为 PDB ID。
    return pdb_id  # 返回 ID 列表。


def parse_sdf_file(path):  # 解析 SDF/MOL2 文件并提取分子图相关特征。
    """解析配体文件并返回构建 `ProteinLigandData` 所需的字典。"""
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')  # 获取 RDKit 特征定义文件路径。
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)  # 构建化学特征工厂。
    # read mol  # 保留注释：读取分子。
    if path.endswith('.sdf'):  # 根据后缀选择读取方式。
        rdmol = Chem.MolFromMolFile(path, sanitize=False)  # 从 SDF 读取。
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)  # 从 MOL2 读取。
    else:
        raise ValueError  # 不支持的格式抛出异常。
    Chem.SanitizeMol(rdmol)  # 对分子进行净化。
    rdmol = Chem.RemoveHs(rdmol)  # 移除氢原子简化图结构。

    # Remove Hydrogens.  # 保留注释：说明已经去氢。
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()  # 获取原子总数。
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int64)  # 初始化特征矩阵。
    for feat in factory.GetFeaturesForMol(rdmol):  # 遍历化学特征。
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1  # 标记相应原子特征。

    # Get hybridization in the order of atom idx.  # 保留注释：收集杂化信息。
    hybridization = []
    for atom in rdmol.GetAtoms():  # 遍历原子。
        hybr = str(atom.GetHybridization())  # 提取杂化类型。
        idx = atom.GetIdx()  # 获取原子索引。
        hybridization.append((idx, hybr))  # 记录元组以便排序。
    hybridization = sorted(hybridization)  # 按索引排序。
    hybridization = [v[1] for v in hybridization]  # 取出杂化类型列表。

    ptable = Chem.GetPeriodicTable()  # 获取周期表查询工具。

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)  # 读取 3D 坐标。
    element = []  # 初始化元素列表。
    accum_pos = 0  # 初始化质心累加值。
    accum_mass = 0  # 初始化质量累加值。
    for atom_idx in range(rd_num_atoms):  # 遍历原子索引。
        atom = rdmol.GetAtomWithIdx(atom_idx)  # 获取原子对象。
        atom_num = atom.GetAtomicNum()  # 获取原子序号。
        element.append(atom_num)  # 记录原子序号。
        atom_weight = ptable.GetAtomicWeight(atom_num)  # 查询原子质量。
        accum_pos += pos[atom_idx] * atom_weight  # 质量加权坐标。
        accum_mass += atom_weight  # 累加质量。
    center_of_mass = accum_pos / accum_mass  # 计算分子质心。
    element = np.array(element, dtype=np.int64)  # 转换为数组。

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.  # 保留注释：边类型含义。
    row, col, edge_type = [], [], []  # 初始化边索引与类型列表。
    for bond in rdmol.GetBonds():  # 遍历分子中的键。
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]  # 添加起点索引。
        col += [end, start]  # 添加终点索引（双向）。
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]  # 为双向边记录类型编码。

    edge_index = np.array([row, col], dtype=np.int64)  # 组合成边索引矩阵。
    edge_type = np.array(edge_type, dtype=np.int64)  # 转换为数组。

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()  # 根据源-目标组合排序。
    edge_index = edge_index[:, perm]  # 重新排序边索引。
    edge_type = edge_type[perm]  # 按相同顺序排列边类型。

    data = {
        'smiles': Chem.MolToSmiles(rdmol),  # 分子的 SMILES 表示。
        'element': element,  # 原子序号数组。
        'pos': pos,  # 原子坐标。
        'bond_index': edge_index,  # 键索引矩阵。
        'bond_type': edge_type,  # 键类型编码。
        'center_of_mass': center_of_mass,  # 分子质心。
        'atom_feature': feat_mat,  # 原子化学特征矩阵。
        'hybridization': hybridization  # 杂化类型列表。
    }
    return data  # 返回解析结果。
