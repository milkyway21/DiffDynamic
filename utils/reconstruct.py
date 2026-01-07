"""
https://github.com/mattragoza/liGAN/blob/master/fitting.py

License: GNU General Public License v2.0
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""
import itertools  # 导入迭代工具，用于组合遍历。

import numpy as np  # 导入 NumPy。
from rdkit.Chem import AllChem as Chem  # 从 RDKit 导入化学模块。
from rdkit import Geometry  # 导入几何工具。
from openbabel import openbabel as ob  # 导入 OpenBabel 接口。
from scipy.spatial.distance import pdist  # 导入点对距离计算。
from scipy.spatial.distance import squareform  # 导入距离矩阵转换工具。


class MolReconsError(Exception):  # 定义分子重建错误类型。
    pass  # 保持空实现。


def reachable_r(a, b, seenbonds):  # 递归辅助函数，判断两原子在去掉特定键后是否连通。
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):  # 遍历原子 a 的邻居。
        bond = a.GetBond(nbr).GetIdx()  # 获取与邻居的键索引。
        if bond not in seenbonds:  # 若该键未被访问过。
            seenbonds.add(bond)  # 标记已访问。
            if nbr == b:  # 找到目标原子。
                return True
            elif reachable_r(nbr, b, seenbonds):  # 否则继续递归搜索。
                return True
    return False  # 未找到则返回 False。


def reachable(a, b):  # 检查去掉 a-b 之间的键后，a 是否仍能到达 b。
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:  # 若任一原子度为 1。
        return False  # this is the _only_ bond for one atom
    # otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])  # 初始化已访问集合，排除当前键。
    return reachable_r(a, b, seenbonds)  # 调用递归辅助函数。


def forms_small_angle(a, b, cutoff=60):  # 判断 a-b 所在的角是否过小。
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):  # 遍历 a 的邻居。
        if nbr != b:  # 排除 b 本身。
            degrees = b.GetAngle(a, nbr)  # 计算角度。
            if degrees < cutoff:  # 小于阈值则视为小角。
                return True
    return False  # 否则返回 False。


def make_obmol(xyz, atomic_numbers):  # 将坐标与原子序号构建为 OBMol 对象。
    mol = ob.OBMol()  # 创建空分子。
    mol.BeginModify()  # 进入修改状态。
    atoms = []  # 用于存储 OBAtom 引用。
    for xyz, t in zip(xyz, atomic_numbers):  # 遍历坐标与原子种类。
        x, y, z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()  # 创建新原子。
        atom.SetAtomicNum(t)  # 设置原子序号。
        atom.SetVector(x, y, z)  # 设置坐标。
        atoms.append(atom)  # 保存原子引用。
    return mol, atoms  # 返回分子及原子列表。


def connect_the_dots(mol, atoms, indicators, covalent_factor=1.3):  # 根据原子位置尝试恢复键连接。
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''

    """
    for now, indicators only include 'is_aromatic'
    """
    pt = Chem.GetPeriodicTable()  # 获取周期表用于查找价数。

    if len(atoms) == 0:  # 若没有原子直接返回。
        return

    mol.BeginModify()  # 开始修改分子。

    # just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])  # 提取所有原子坐标。
    dists = squareform(pdist(coords))  # 计算两两距离。
    # types = [struct.channels[t].name for t in struct.c]

    for i, j in itertools.combinations(range(len(atoms)), 2):  # 遍历所有原子对。
        a = atoms[i]
        b = atoms[j]
        a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor  # 原子 a 的共价半径（放大系数）。
        b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor  # 原子 b 的共价半径。
        if dists[i, j] < a_r + b_r:  # 若距离小于半径之和，认为可能有键连接。
            flag = 0
            if indicators and indicators[i] and indicators[j]:  # 若两端均被标记为芳香。
                flag = ob.OB_AROMATIC_BOND  # 设置芳香键标志。
            mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)  # 新增单键。

    atom_maxb = {}  # 存储每个原子允许的最大键数。
    for (i, a) in enumerate(atoms):
        # set max valance to the smallest max allowed by openbabel or rdkit
        # since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))  # 取 OpenBabel 与 RDKit 的最小允许值。

        if a.GetAtomicNum() == 16:  # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb  # 记录最大价键。

    # remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):  # 遍历现有键。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:  # 若两端都只允许 1 个键。
            mol.DeleteBond(bond)  # 删除不可能的卤素键。

    def get_bond_info(biter):  # 返回按拉伸比例排序的键列表。
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            # compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
            stretch = bdist / ideal  # 计算拉伸比。
            binfo.append((stretch, bond))
        binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
        return binfo

    binfo = get_bond_info(ob.OBMolBondIter(mol))  # 获取所有键的拉伸排序。
    # now eliminate geometrically poor bonds
    for stretch, bond in binfo:  # 遍历拉伸大的键。

        # can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        # as long as we aren't disconnecting, let's remove things
        # that are excessively far away (0.45 from ConnectTheDots)
        # get bonds to be less than max allowed
        # also remove tight angles, because that is what ConnectTheDots does
        if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):  # 拉伸过大或角度过小。
            # don't fragment the molecule
            if not reachable(a1, a2):  # 如果移除该键会断裂分子，则跳过。
                continue
            mol.DeleteBond(bond)  # 否则删除。

    # prioritize removing hypervalency causing bonds, do more valent
    # constrained atoms first since their bonds introduce the most problems
    # with reachability (e.g. oxygen)
    hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]  # 计算每个原子超价情况。
    hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))  # 优先处理允许价小且超价多的原子。
    for mb, diff, a in hypers:
        if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:  # 若未超价则跳过。
            continue
        binfo = get_bond_info(ob.OBAtomBondIter(a))  # 获取该原子相关键的拉伸信息。
        for stretch, bond in binfo:

            if stretch < 0.9:  # the two atoms are too closed to remove the bond
                continue
            # can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # get right valence
            if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                # don't fragment the molecule
                if not reachable(a1, a2):  # 确保移除后仍连通。
                    continue
                mol.DeleteBond(bond)  # 删除该键。
                if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:  # 若超价问题已解决。
                    break  # let nbr atoms choose what bonds to throw out

    mol.EndModify()  # 结束修改。


def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()  # 移除氢原子，简化结构。
    n_atoms = ob_mol.NumAtoms()  # 取得原子数。
    rd_mol = Chem.RWMol()  # 创建 RDKit 可编辑分子。
    rd_conf = Chem.Conformer(n_atoms)  # 创建构象用于存储坐标。

    for ob_atom in ob.OBMolAtomIter(ob_mol):  # 遍历 OpenBabel 原子。
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())  # 复制原子序号。
        # TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:  # 针对芳香环。
            # don't commit to being aromatic unless rdkit will be okay with the ring status
            # (this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)  # 将原子加入 RDKit 分子，获取索引。
        ob_coords = ob_atom.GetVector()  # 读取 OpenBabel 坐标。
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)  # 构造 RDKit 坐标。
        rd_conf.SetAtomPosition(i, rd_coords)  # 设置原子位置。

    rd_mol.AddConformer(rd_conf)  # 添加构象。

    for ob_bond in ob.OBMolBondIter(ob_mol):  # 遍历 OpenBabel 键。
        i = ob_bond.GetBeginAtomIdx() - 1  # 转换为 RDKit 索引（从 0 开始）。
        j = ob_bond.GetEndAtomIdx() - 1
        bond_order = ob_bond.GetBondOrder()  # 读取键级。
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():  # 如果 OpenBabel 键为芳香。
            bond = rd_mol.GetBondBetweenAtoms(i, j)
            bond.SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)  # 去除所有显式氢原子（暂不执行 sanitize）。

    pt = Chem.GetPeriodicTable()
    # if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()  # 获取坐标。
    nonsingles = []  # 收集非单键。
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:  # 找出双/三键。
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i] - positions[j])  # 计算键长。
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])  # 按键长从大到小排序。

    for (d, bond) in nonsingles:  # 遍历双/三键。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):  # 如果超出默认价。
            btype = Chem.BondType.SINGLE  # 降级为单键。
            if bond.GetBondType() == Chem.BondType.TRIPLE:  # 如果原为三键，则先降为双键。
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():  # 遍历所有原子。
        # set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)  # 对四价氮设置 +1 电荷。

    rd_mol = Chem.AddHs(rd_mol, addCoords=True)  # 重新加入氢原子并生成坐标。

    positions = rd_mol.GetConformer().GetPositions()  # 获取坐标矩阵。
    center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)  # 计算有限坐标点的中心。
    for atom in rd_mol.GetAtoms():  # 遍历原子。
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):  # 若坐标包含 NaN。
            # hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i, center)  # 将该原子坐标重置为中心位置。

    try:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)  # 执行除去 kekulize 的结构检查。
    except:
        raise MolReconsError()  # 若失败则抛出重建错误。
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    # but at some point stop trying to enforce our aromaticity -
    # openbabel and rdkit have different aromaticity models so they
    # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():  # 再次同步键的芳香性标记。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol  # 返回处理后的分子。


def calc_valence(rdatom):  # 计算 RDKit 原子的显式价。
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):  # 统计指定原子的邻居中具有特定原子序号的数量。
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def fixup(atoms, mol, indicators):  # 调整 OBMol 原子的属性以符合指定指示器。
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    """
    for now, indicators only include 'is_aromatic'
    """
    mol.SetAromaticPerceived(True)  # avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 

        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
            # this is a little iffy, ommitting until there is more evidence it is a net positive
            # we don't have aromatic types for nitrogen, but if it
            # is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def raw_obmol_from_generated(data):  # 将生成的配体上下文转为 OpenBabel 分子。
    xyz = data.ligand_context_pos.clone().cpu().tolist()  # 提取坐标。
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()  # 提取原子序号。
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)  # 构建 OBMol 与 OBAtom 列表。
    return mol, atoms  # 返回分子及原子引用。


UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.TRIPLE}


def postprocess_rd_mol_1(rdmol):  # 对 RDKit 分子进行第一阶段后处理。
    rdmol = Chem.RemoveHs(rdmol)  # 移除氢原子。

    # Construct bond nbh list  # 保留注释：构建键邻接表。
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in nbh_list:
            nbh_list[begin] = [end]
        else:
            nbh_list[begin].append(end)

        if end not in nbh_list:
            nbh_list[end] = [begin]
        else:
            nbh_list[end].append(begin)

    # Fix missing bond-order  # 保留注释：修复缺失的键级。
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    return rdmol


def postprocess_rd_mol_2(rdmol):  # 对 RDKit 分子进行第二阶段后处理。
    rdmol_edit = Chem.RWMol(rdmol)  # 创建可编辑副本。

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol


def reconstruct_from_generated(xyz, atomic_nums, aromatic=None, basic_mode=True):  # 从生成的坐标和元素重建 RDKit 分子。
    """
    will utilize data.ligand_pos, data.ligand_element, data.ligand_atom_feature_full to reconstruct mol
    """
    # xyz = data.ligand_pos.clone().cpu().tolist()
    # atomic_nums = data.ligand_element.clone().cpu().tolist()
    # indicators = data.ligand_atom_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    # indicators = None
    if basic_mode:
        indicators = None  # 基础模式下不使用芳香指示。
    else:
        indicators = aromatic  # 否则使用外部提供的芳香标记。

    mol, atoms = make_obmol(xyz, atomic_nums)  # 构建 OpenBabel 分子。
    fixup(atoms, mol, indicators)  # 根据指示器调整原子属性。

    connect_the_dots(mol, atoms, indicators, covalent_factor=1.3)  # 根据距离连接键。
    fixup(atoms, mol, indicators)  # 再次调整原子属性确保一致。

    mol.AddPolarHydrogens()  # 添加极性氢。
    mol.PerceiveBondOrders()  # 让 OpenBabel 感知键级。
    fixup(atoms, mol, indicators)  # 再次修正芳香标记。

    for (i, a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)  # 为原子分配典型隐式氢数。
    fixup(atoms, mol, indicators)  # 再次同步属性。

    mol.AddHydrogens()  # 添加全部氢原子。
    fixup(atoms, mol, indicators)  # 再次修正属性。

    # make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):  # 遍历所有环，处理芳香性。
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                # set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    # bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):  # 确保键的芳香标记与原子一致。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)

    mol.PerceiveBondOrders()
    rd_mol = convert_ob_mol_to_rd_mol(mol)  # 转换为 RDKit 分子。
    try:
        # Post-processing
        rd_mol = postprocess_rd_mol_1(rd_mol)
        rd_mol = postprocess_rd_mol_2(rd_mol)
    except:
        raise MolReconsError()

    return rd_mol
