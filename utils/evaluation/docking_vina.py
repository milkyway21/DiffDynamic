"""AutoDock Vina 对接流程封装，包含配体/蛋白准备与打分接口。"""

# 总结：
# - 利用 OpenBabel、Meeko 和 AutoDockTools 将配体、蛋白转换为 PDBQT，调用 Vina 进行对接或打分。
# - 提供 `PrepLig`、`PrepProt`、`VinaDock` 等工具类，以及 `VinaDockingTask` 任务封装。
# - 支持从生成数据或原始数据集构造任务，自动计算搜索盒参数并返回亲和力与姿势。

# NumPy 兼容性修复：NumPy 1.20+ 移除了 np.int，需要添加兼容性别名
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.complex = complex
    np.bool = bool
    np.unicode = str
    np.long = int

from openbabel import pybel  # 导入 Pybel，封装 OpenBabel 操作。
from meeko import MoleculePreparation  # 导入 Meeko 配体预处理类。
from meeko import obutils  # 导入 Meeko-OpenBabel 工具函数。
from vina import Vina  # 导入 Vina Python 接口。
import subprocess  # 导入子进程管理模块。
import rdkit.Chem as Chem  # 导入 RDKit 化学模块。
from rdkit.Chem import AllChem  # 导入 RDKit 构象生成功能。
import tempfile  # 导入临时文件工具。
import AutoDockTools  # 导入 AutoDockTools 库，用于受体准备脚本定位。
import os  # 导入操作系统接口。
import contextlib  # 导入上下文工具，用于重定向输出。

from utils.reconstruct import reconstruct_from_generated  # 导入重建工具，将预测数据转为 RDKit 分子。
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask  # 重用生成随机 ID 与基础任务类。


def supress_stdout(func):
    """装饰器：静默函数的标准输出。"""
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):  # 将 stdout 重定向至空设备。
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    """负责转换并准备配体分子的工具类。"""
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)  # 通过 SMILES 构建分子。
        elif mol_format == 'sdf':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))  # 从 SDF 文件读取分子。
        else:
            raise ValueError(f'mol_format {mol_format} not supported')

    def addH(self, polaronly=False, correctforph=True, PH=7):
        """向分子添加氢原子，并输出到临时文件。"""
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        """使用 RDKit 生成 3D 构象并写回 OpenBabel 对象。"""
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        """通过 Meeko 生成配体 PDBQT，可以返回字符串或写入文件。"""
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()


class PrepProt(object):
    """负责蛋白前处理的工具类。"""
    def __init__(self, pdb_file):
        self.prot = pdb_file  # 保存蛋白 PDB 路径。

    def del_water(self, dry_pdb_file):  # optional
        """可选：移除 PDB 中的水分子。"""
        with open(self.prot) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            dry_lines = [l for l in lines if 'HOH' not in l]

        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file  # 更新当前蛋白路径。

    def addH(self, prot_pqr):  # call pdb2pqr
        """调用 pdb2pqr 添加氢并生成 PQR。"""
        self.prot_pqr = prot_pqr
        proc = subprocess.Popen(['pdb2pqr30', '--ff=AMBER', self.prot, self.prot_pqr],
                                stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'
            raise RuntimeError(f'pdb2pqr30 failed: {error_msg}')
        if not os.path.exists(prot_pqr):
            raise RuntimeError(f'pdb2pqr30 did not create output file: {prot_pqr}')

    def get_pdbqt(self, prot_pdbqt):
        """使用 AutoDockTools 脚本生成受体 PDBQT。"""
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        proc = subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                                stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'
            raise RuntimeError(f'prepare_receptor4.py failed: {error_msg}')
        if not os.path.exists(prot_pdbqt):
            raise RuntimeError(f'prepare_receptor4.py did not create output file: {prot_pdbqt}')


class VinaDock(object):
    """封装 Vina 评分/对接接口。"""
    def __init__(self, lig_pdbqt, prot_pdbqt):
        self.lig_pdbqt = lig_pdbqt  # 配体 PDBQT 路径。
        self.prot_pdbqt = prot_pdbqt  # 受体 PDBQT 路径。

    def _max_min_pdb(self, pdb, buffer):
        """读取 PDB 计算最大/最小坐标并返回中心与盒子大小。"""
        with open(pdb, 'r') as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size

    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None:
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, n_poses=9, save_pose=False, **kwargs):  # seed=0 mean random seed
        """执行 Vina 打分/对接，可选择返回姿势。
        
        Args:
            n_poses: 生成的对接姿势数量（默认9）
        """
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only':
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            score = v.energies(n_poses=n_poses)[0][0]  # 返回最佳姿势的得分
        else:
            raise ValueError

        if not save_pose:
            return score
        else:
            if mode == 'score_only':
                pose = None
            elif mode == 'minimize':
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f:
                    v.write_pose(tmp.name, overwrite=True)
                with open(tmp.name, 'r') as f:
                    pose = f.read()

            elif mode == 'dock':
                pose = v.poses(n_poses=n_poses)  # 返回所有姿势
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        """从生成样本构建 Vina 对接任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())  # TODO: 这里依赖数据克隆方法。
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        """从原始数据集中读取配体与蛋白构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        """从生成分子及其文件路径构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, n_poses=9, **kwargs):
        """执行 Vina 对接流程并返回亲和力与姿势。
        
        Args:
            n_poses: 生成的对接姿势数量（默认9，与configs/sampling.yml中的vina_poses一致）
        """
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        else:
            # 如果文件已存在，也需要设置 self.prot_pqr，因为 get_pdbqt 需要它
            prot.prot_pqr = protein_pqr
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, n_poses=n_poses, save_pose=True, **kwargs)
        return [{'affinity': score, 'pose': pose}]


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    

