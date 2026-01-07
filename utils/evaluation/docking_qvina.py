"""QVina 对接任务封装，支持从生成分子或原始数据构建任务并解析结果。"""

# 总结：
# - 负责将生成或原始的配体结构与蛋白口袋一起交由 QVina 进行刚体对接。
# - 支持从多种数据源构造任务，并自动处理文件准备、对接执行与结果解析。
# - 可选择是否进行 UFF 优化以及自定义搜索盒中心、尺寸和计算精度。

import os  # 导入操作系统接口。
import subprocess  # 导入子进程管理工具。
import random  # 导入随机数模块。
import string  # 导入字符串常量。
from easydict import EasyDict  # 导入 EasyDict，便于属性访问。
from rdkit import Chem  # 导入 RDKit 化学库。
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule  # 导入 UFF 优化器。

from utils.reconstruct import reconstruct_from_generated  # 导入重建工具，将预测结果转为 RDKit 分子。


def get_random_id(length=30):
    """生成指定长度的随机字符串，用于任务临时文件命名。"""
    letters = string.ascii_lowercase  # 仅使用小写字母。
    return ''.join(random.choice(letters) for i in range(length))  # 逐字符随机拼接。


def load_pdb(path):
    """读取 PDB 文件并返回其文本内容。"""
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path):
    """解析 QVina 输出的 SDF 文件，提取每个姿势的能量与 RMSD。"""
    suppl = Chem.SDMolSupplier(docked_sdf_path)  # 创建 SDF 读取器。
    results = []  # 存储解析结果。
    for i, mol in enumerate(suppl):  # 遍历每个对接姿势。
        if mol is None:  # 过滤损坏的条目。
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]  # 从备注解析能量与 RMSD。
        results.append(EasyDict({
            'rdmol': mol,  # 对接后的 RDKit 分子。
            'mode_id': i,  # 模式编号。
            'affinity': float(line[0]),  # 结合亲和力。
            'rmsd_lb': float(line[1]),  # RMSD 下界。
            'rmsd_ub': float(line[2]),  # RMSD 上界。
        }))

    return results  # 返回所有成功解析的姿势。


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()  # 调用基类构造。
        self.pdb_block = pdb_block  # 保存蛋白结构文本。
        self.ligand_rdmol = ligand_rdmol  # 保存配体分子对象。

    def run(self):
        raise NotImplementedError()  # 需在子类实现运行逻辑。

    def get_results(self):
        raise NotImplementedError()  # 需在子类实现结果获取逻辑。


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        """从生成样本 `ProteinLigandData` 构建对接任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)  # 拼接蛋白路径。
        with open(protein_path, 'r') as f:
            pdb_block = f.read()  # 读取蛋白结构。
        xyz = data.ligand_pos.clone().cpu().tolist()  # 提取配体坐标。
        atomic_nums = data.ligand_element.clone().cpu().tolist()  # 提取配体原子序号。
        ligand_rdmol = reconstruct_from_generated(xyz, atomic_nums)  # 重建 RDKit 配体。
        return cls(pdb_block, ligand_rdmol, **kwargs)  # 返回任务实例。

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        """从生成的 RDKit 分子与其路径构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        """从原始数据集中加载配体与蛋白构建任务。"""
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)  # 读取原配体文件。
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))  # 取出 RDKit 分子。
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, conda_env=None, tmp_dir='./tmp', use_uff=True, center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(pdb_block, ligand_rdmol)  # 调用父类构造。
        # 优先使用传入参数，其次环境变量，再次当前激活的 conda 环境，最后 fallback 到 'adt'
        if conda_env:
            self.conda_env = conda_env.strip()
        elif os.environ.get('QVINA_CONDA_ENV'):
            self.conda_env = os.environ.get('QVINA_CONDA_ENV').strip()
        elif os.environ.get('CONDA_DEFAULT_ENV'):
            self.conda_env = os.environ.get('CONDA_DEFAULT_ENV').strip()
        else:
            self.conda_env = 'adt'
        self.tmp_dir = os.path.realpath(tmp_dir)  # 规范化临时目录。
        os.makedirs(tmp_dir, exist_ok=True)  # 确保目录存在。

        self.task_id = get_random_id()  # 生成任务 ID。
        self.receptor_id = self.task_id + '_receptor'  # 受体文件前缀。
        self.ligand_id = self.task_id + '_ligand'  # 配体文件前缀。

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')  # 受体路径。
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')  # 配体路径。

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)  # 写入受体 PDB。

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)  # 加氢并生成坐标。
        if use_uff:
            UFFOptimizeMolecule(ligand_rdmol)  # 可选地执行 UFF 优化。
        sdf_writer = Chem.SDWriter(self.ligand_path)  # 构造 SDF 写入器。
        sdf_writer.write(ligand_rdmol)  # 保存配体。
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol  # 保存优化后的分子。

        pos = ligand_rdmol.GetConformer(0).GetPositions()  # 获取配体坐标。
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2  # 自动居中。
        else:
            self.center = center  # 使用外部指定中心。

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20  # 默认盒子尺寸。
        else:
            # 基于分子尺寸缩放，并添加buffer以确保搜索空间足够大
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None  # 运行进程句柄。
        self.results = None  # 缓存解析结果。
        self.output = None  # 标准输出缓存。
        self.error_output = None  # 标准错误缓存。
        self.docked_sdf_path = None  # 对接结果路径。

    def run(self, exhaustiveness=16):
        """异步启动 QVina 进程执行对接。"""
        # 尝试使用 conda activate，如果失败则直接执行命令（假设命令已在 PATH 中）
        conda_activate_cmd = f'eval "$(conda shell.bash hook)" && conda activate {self.conda_env} && '
        # 如果环境变量显示当前已在目标环境，可以跳过 activate
        current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if current_env == self.conda_env:
            conda_activate_cmd = ''
        
        # 检测所有必需工具的位置（使用完整路径，避免 PATH 问题）
        import shutil
        prep_receptor_cmd = 'prepare_receptor4.py'
        prep_receptor_path = shutil.which('prepare_receptor4.py') or shutil.which('prepare_receptor4')
        if prep_receptor_path:
            # 使用找到的完整路径
            prep_receptor_cmd = prep_receptor_path
            # 检查脚本的第一行，看是否需要特定 Python 版本
            try:
                with open(prep_receptor_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#!') and 'python2' in first_line:
                        # 尝试用 python2 运行
                        prep_receptor_cmd = f'python2 {prep_receptor_path}'
                    elif first_line.startswith('#!') and 'python' in first_line and 'python3' not in first_line:
                        # 可能是 Python 2，尝试 python2
                        prep_receptor_cmd = f'python2 {prep_receptor_path}'
            except Exception:
                pass  # 如果读取失败，使用默认命令
        
        # 查找 obabel 和 qvina2 的完整路径
        obabel_path = shutil.which('obabel')
        qvina2_path = shutil.which('qvina2') or shutil.which('qvina')
        
        # 如果找不到，尝试在 conda 环境的 bin 目录中查找
        if not obabel_path:
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                conda_bin = os.path.join(conda_prefix, 'bin', 'obabel')
                if os.path.exists(conda_bin):
                    obabel_path = conda_bin
            # 如果还是找不到，使用默认命令
            if not obabel_path:
                obabel_path = 'obabel'
        
        if not qvina2_path:
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                conda_bin_qvina = os.path.join(conda_prefix, 'bin', 'qvina2')
                if os.path.exists(conda_bin_qvina):
                    qvina2_path = conda_bin_qvina
            # 如果还是找不到，使用默认命令
            if not qvina2_path:
                qvina2_path = 'qvina2'
        
        self.docked_sdf_path = os.path.join(self.tmp_dir, f'{self.ligand_id}_out.sdf')
        ligand_pdbqt_path = os.path.join(self.tmp_dir, f'{self.ligand_id}.pdbqt')
        
        # 使用 obabel CLI 进行多轮次转换，以最大化成功率
        ligand_prep_cmd = f"""
# Prepare ligand (SDF->PDBQT) using obabel (multi-step fallback)
echo "开始转换配体 SDF -> PDBQT..."
ligand_converted=false
# 方法1: 直接转换 SDF -> PDBQT (with -h)
if {obabel_path} {self.ligand_id}.sdf -O{self.ligand_id}.pdbqt -h 2>&1 && [ -f {self.ligand_id}.pdbqt ]; then
    ligand_converted=true
    echo "配体转换成功 (方法1: SDF->PDBQT with -h)"
# 方法2: 直接转换 SDF -> PDBQT (without -h)
elif {obabel_path} {self.ligand_id}.sdf -O{self.ligand_id}.pdbqt 2>&1 && [ -f {self.ligand_id}.pdbqt ]; then
    ligand_converted=true
    echo "配体转换成功 (方法2: SDF->PDBQT without -h)"
# 方法3: SDF -> MOL2 -> PDBQT
else
    echo "Warning: 直接转换失败，尝试 SDF->MOL2->PDBQT"
    # 先转换 SDF -> MOL2
    if {obabel_path} {self.ligand_id}.sdf -O{self.ligand_id}.mol2 -h 2>&1 || {obabel_path} {self.ligand_id}.sdf -O{self.ligand_id}.mol2 2>&1; then
        # 再转换 MOL2 -> PDBQT
        if ({obabel_path} {self.ligand_id}.mol2 -O{self.ligand_id}.pdbqt -h 2>&1 || {obabel_path} {self.ligand_id}.mol2 -O{self.ligand_id}.pdbqt 2>&1) && [ -f {self.ligand_id}.pdbqt ]; then
            ligand_converted=true
            echo "配体转换成功 (方法3: SDF->MOL2->PDBQT)"
        fi
    fi
fi
# 检查是否成功
if [ "$ligand_converted" = false ] || [ ! -f {self.ligand_id}.pdbqt ]; then
    echo "ERROR: 所有配体转换方法都失败了"
    exit 1
fi
echo "配体转换完成"
"""
        
        commands = f"""
set -e
{conda_activate_cmd}cd {self.tmp_dir}
echo "工作目录: $(pwd)"
echo "开始准备受体..."
# Prepare receptor (PDB->PDBQT)
if ! {prep_receptor_cmd} -r {self.receptor_id}.pdb 2>&1; then
    echo "ERROR: 受体准备失败"
    exit 1
fi
if [ ! -f {self.receptor_id}.pdbqt ]; then
    echo "ERROR: 受体PDBQT文件未生成"
    exit 1
fi
echo "受体准备完成"
{ligand_prep_cmd}
echo "开始运行QVina对接..."
# Run QVina docking
if ! {qvina2_path} \
    --receptor {self.receptor_id}.pdbqt \
    --ligand {self.ligand_id}.pdbqt \
    --center_x {self.center[0]:.4f} \
    --center_y {self.center[1]:.4f} \
    --center_z {self.center[2]:.4f} \
    --size_x {self.size_x} --size_y {self.size_y} --size_z {self.size_z} \
    --exhaustiveness {exhaustiveness} 2>&1; then
    echo "ERROR: QVina对接失败"
    exit 1
fi
if [ ! -f {self.ligand_id}_out.pdbqt ]; then
    echo "ERROR: QVina输出PDBQT文件未生成"
    exit 1
fi
echo "QVina对接完成"
echo "开始转换结果 PDBQT -> SDF..."
# Convert result back to SDF (PDBQT->SDF)
if ! {obabel_path} {self.ligand_id}_out.pdbqt -O{self.ligand_id}_out.sdf -h 2>&1; then
    echo "ERROR: 结果转换失败"
    exit 1
fi
if [ ! -f {self.ligand_id}_out.sdf ]; then
    echo "ERROR: 最终SDF文件未生成"
    exit 1
fi
echo "结果转换完成"
        """

        self.proc = subprocess.Popen(  # 启动 Bash 子进程。
            '/bin/bash',
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))  # 将命令写入子进程。
        self.proc.stdin.close()  # 关闭标准输入，触发执行。

        # return commands

    def run_sync(self, exhaustiveness=16):
        """同步执行对接并返回结果。"""
        self.run(exhaustiveness=exhaustiveness)  # 启动异步流程，传递exhaustiveness参数。
        while self.get_results() is None:  # 轮询直至完成。
            pass
        results = self.get_results()  # 获取最终结果。
        if not results:
            stderr_msg = b''.join(self.error_output or []).decode('utf-8', errors='ignore')
            raise RuntimeError(
                f'QVina docking failed: 无法解析 {self.docked_sdf_path}\n'
                f'STDERR: {stderr_msg.strip() or "无"}'
            )
        print('Best affinity:', results[0]['affinity'])  # 输出最佳亲和力。
        return results  # 返回姿势列表。

    def get_results(self):
        """检查对接进程状态并解析结果。"""
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:  # 仅首次解析。
                self.output = self.proc.stdout.readlines()  # 缓存标准输出。
                self.error_output = self.proc.stderr.readlines()  # 缓存错误输出。
                
                # 检查进程返回码
                return_code = self.proc.returncode
                
                # 检查中间文件是否存在
                receptor_pdbqt = os.path.join(self.tmp_dir, f'{self.receptor_id}.pdbqt')
                ligand_pdbqt = os.path.join(self.tmp_dir, f'{self.ligand_id}.pdbqt')
                ligand_out_pdbqt = os.path.join(self.tmp_dir, f'{self.ligand_id}_out.pdbqt')
                
                # 诊断信息
                diagnostic_info = []
                diagnostic_info.append(f"进程返回码: {return_code}")
                diagnostic_info.append(f"输出文件路径: {self.docked_sdf_path}")
                diagnostic_info.append(f"输出文件存在: {os.path.exists(self.docked_sdf_path)}")
                if os.path.exists(self.docked_sdf_path):
                    file_size = os.path.getsize(self.docked_sdf_path)
                    diagnostic_info.append(f"输出文件大小: {file_size} 字节")
                
                diagnostic_info.append(f"受体PDBQT存在: {os.path.exists(receptor_pdbqt)}")
                diagnostic_info.append(f"配体PDBQT存在: {os.path.exists(ligand_pdbqt)}")
                diagnostic_info.append(f"配体输出PDBQT存在: {os.path.exists(ligand_out_pdbqt)}")
                
                # 读取stdout和stderr
                stdout_msg = b''.join(self.output or []).decode('utf-8', errors='ignore')
                stderr_msg = b''.join(self.error_output or []).decode('utf-8', errors='ignore')
                
                if stdout_msg.strip():
                    diagnostic_info.append(f"STDOUT:\n{stdout_msg}")
                if stderr_msg.strip():
                    diagnostic_info.append(f"STDERR:\n{stderr_msg}")
                
                try:
                    # 检查输出文件是否存在且非空
                    if not os.path.exists(self.docked_sdf_path):
                        raise FileNotFoundError(
                            f"输出文件不存在: {self.docked_sdf_path}\n" + 
                            "\n".join(diagnostic_info)
                        )
                    
                    if os.path.getsize(self.docked_sdf_path) == 0:
                        raise ValueError(
                            f"输出文件为空: {self.docked_sdf_path}\n" + 
                            "\n".join(diagnostic_info)
                        )
                    
                    # 检查中间文件
                    if not os.path.exists(ligand_out_pdbqt):
                        raise FileNotFoundError(
                            f"QVina输出PDBQT文件不存在: {ligand_out_pdbqt}\n" +
                            "这可能表示QVina对接失败。\n" +
                            "\n".join(diagnostic_info)
                        )
                    
                    self.results = parse_qvina_outputs(self.docked_sdf_path)  # 解析 SDF。
                    
                    if not self.results:
                        raise ValueError(
                            f"成功解析SDF文件，但未找到有效对接姿势。\n" +
                            "\n".join(diagnostic_info)
                        )
                        
                except Exception as exc:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)  # 打印异常信息。
                    print(f'[Error] Parser exception: {exc}')
                    print('\n'.join(diagnostic_info))
                    self.results = []
                    return self.results
            return self.results  # 返回缓存结果。
