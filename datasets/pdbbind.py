# 总结：
# - 实现用于 PDBBind 数据集的自定义 PyTorch Dataset。
# - 基于 LMDB 存储机制缓存处理后的蛋白-配体数据，加速读取。
# - 支持可选的外部嵌入向量与数据变换流程以增强训练输入。

import os  # 导入操作系统路径处理模块。
import pickle  # 导入 pickle，用于对象的序列化与反序列化。
import lmdb  # 导入 LMDB 数据库，用于高效键值存储。
import torch  # 导入 PyTorch，用于张量操作和持久化加载。
from torch.utils.data import Dataset  # 导入 Dataset 基类，用于自定义数据集。
from tqdm.auto import tqdm  # 导入 tqdm 自动选择前端，用于进度条显示。

from utils.data import PDBProtein  # 从工具模块导入 PDBProtein 类用于解析蛋白。
from datasets.protein_ligand import parse_sdf_file_mol  # 导入 SDF 文件解析函数以读取配体。
from datasets.pl_data import ProteinLigandData, torchify_dict  # 导入 ProteinLigandData 和字典张量化工具。
from scipy import stats  # 导入 scipy.stats，用于统计量计算。


class PDBBindDataset(Dataset):  # 定义 PDBBind 数据集类，继承自 PyTorch Dataset。
    """PDBBind 数据集封装，提供 LMDB 缓存与可选嵌入特征读取。"""

    def __init__(self, raw_path, transform=None, emb_path=None, heavy_only=False):  # 初始化函数，配置路径与开关。
        """创建数据集，若缓存缺失则从原始索引构建。

        Args:
            raw_path: 包含 `index.pkl` 与实际 pdb/sdf 文件的目录。
            transform: 可选的样本级变换函数。
            emb_path: 预计算嵌入文件路径，加载后附加到样本上。
            heavy_only: 解析配体时是否只保留重原子。
        """
        super().__init__()  # 调用父类构造函数，初始化 Dataset 状态。
        self.raw_path = raw_path.rstrip('/')  # 保存数据原始路径，并去除末尾斜杠。
        self.index_path = os.path.join(self.raw_path, 'index.pkl')  # 计算索引文件路径。
        self.processed_path = os.path.join(self.raw_path, os.path.basename(self.raw_path) + '_processed.lmdb')  # 计算预处理 LMDB 路径。
        self.emb_path = emb_path  # 保存嵌入文件路径，若无则为 None。
        self.transform = transform  # 保存可选的数据变换函数。
        self.heavy_only = heavy_only  # 标记是否只处理重原子。
        self.db = None  # 初始化 LMDB 数据库句柄为空。

        self.keys = None  # 初始化键列表为空。

        if not os.path.exists(self.processed_path):  # 若预处理 LMDB 不存在则进行处理。
            self._process()  # 调用内部处理函数生成 LMDB。
        print('Load dataset from ', self.processed_path)  # 打印加载数据集的 LMDB 路径信息。
        if self.emb_path is not None:  # 如果配置了嵌入路径。
            print('Load embedding from ', self.emb_path)  # 打印加载嵌入信息。
            self.emb = torch.load(self.emb_path)  # 加载嵌入张量数据。

    def _connect_db(self):  # 定义内部方法用于建立数据库连接。
        """建立 LMDB 只读连接并缓存键列表。"""
        assert self.db is None, 'A connection has already been opened.'  # 断言当前尚未建立连接，避免重复打开。
        self.db = lmdb.open(  # 打开 LMDB 数据库建立读连接。
            self.processed_path,  # 指定 LMDB 文件路径。
            map_size=10*(1024*1024*1024),   # 10GB  # 设置最大映射大小为 10GB。
            create=False,  # 指定不创建新库。
            subdir=False,  # 表明路径指向单一文件，而非目录。
            readonly=True,  # 启用只读模式。
            lock=False,  # 关闭文件锁以便多进程读取。
            readahead=False,  # 禁用预读来减少内存使用。
            meminit=False,  # 禁止预初始化内存以加快启动。
        )
        with self.db.begin() as txn:  # 启动只读事务获取键列表。
            self.keys = list(txn.cursor().iternext(values=False))  # 迭代游标收集所有键名。

    def _close_db(self):  # 定义内部方法用于关闭数据库连接。
        """关闭连接并重置缓存。"""
        self.db.close()  # 关闭 LMDB 数据库连接。
        self.db = None  # 重置数据库句柄。
        self.keys = None  # 清空键列表缓存。
        
    def _process(self):  # 定义内部方法处理原始数据并写入 LMDB。
        """遍历索引，将解析后的蛋白/配体及标签存入 LMDB。"""
        db = lmdb.open(  # 打开 LMDB 数据库，准备写入。
            self.processed_path,  # 指定输出 LMDB 文件路径。
            map_size=10*(1024*1024*1024),   # 10GB  # 设定最大容量为 10GB。
            create=True,  # 若不存在则新建数据库。
            subdir=False,  # LMDB 直接使用文件。
            readonly=False,  # Writable  # 启用写模式以便插入数据。
        )
        with open(self.index_path, 'rb') as f:  # 以二进制方式读取索引文件。
            index = pickle.load(f)  # 通过 pickle 反序列化索引列表。

        # index = parse_pdbbind_index_file(self.index_path)  # 保留的注释代码，表示可替换的索引解析逻辑。

        num_skipped = 0  # 初始化跳过计数器，用于记录失败样本。
        with db.begin(write=True, buffers=True) as txn:  # 启动写事务，允许缓冲区写入。
            for i, (pocket_fn, ligand_fn, resolution, pka, kind) in enumerate(tqdm(index)):  # 遍历索引中的样本条目，并显示进度条。
                # try:  # 保留注释：曾尝试捕获异常。
                # pdb_path = os.path.join(self.raw_path, 'refined-set', pdb_idx)  # 保留注释：推测原始路径构造。
                # pocket_fn = os.path.join(pdb_path, f'{pdb_idx}_pocket.pdb')  # 保留注释：蛋白口袋文件路径示例。
                # ligand_fn = os.path.join(pdb_path, f'{pdb_idx}_ligand.sdf')  # 保留注释：配体文件路径示例。
                pocket_dict = PDBProtein(pocket_fn).to_dict_atom()  # 解析蛋白口袋文件并转换为原子级字典。
                ligand_dict = parse_sdf_file_mol(ligand_fn, heavy_only=self.heavy_only)  # 解析配体 SDF 文件，按照 heavy_only 选择原子。
                data = ProteinLigandData.from_protein_ligand_dicts(  # 将蛋白与配体字典转换为图数据对象。
                    protein_dict=torchify_dict(pocket_dict),  # 将蛋白字典转换为张量格式。
                    ligand_dict=torchify_dict(ligand_dict),  # 将配体字典转换为张量格式。
                )
                data.protein_filename = pocket_fn  # 记录蛋白文件路径，方便追溯。
                data.ligand_filename = ligand_fn  # 记录配体文件路径。
                data.y = torch.tensor(float(pka))  # 将 pKa 数值转换为张量存储为监督信号。
                data.kind = torch.tensor(kind)  # 将样本类型标识转换为张量。
                txn.put(  # 向 LMDB 写入当前样本。
                    key=f'{i:05d}'.encode(),  # 使用零填充的序号作为键名。
                    value=pickle.dumps(data)  # 序列化数据对象并作为值存储。
                )
                # except:  # 保留注释：异常处理逻辑。
                #     num_skipped += 1  # 保留注释：记录跳过次数。
                #     print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))  # 保留注释：输出跳过信息。
                #     continue  # 保留注释：继续处理下一个样本。
        print('num_skipped: ', num_skipped)  # 输出跳过样本总数。
    
    def __len__(self):  # 实现数据集长度方法。
        """返回样本总数（懒加载 LMDB 连接）。"""
        if self.db is None:  # 若数据库连接尚未建立。
            self._connect_db()  # 建立连接以加载键列表。
        return len(self.keys)  # 返回样本总数。

    def __getitem__(self, idx):  # 实现按索引访问样本的方法。
        """读取单个样本，应用 transform 并附加嵌入特征。"""
        if self.db is None:  # 若数据库连接尚未建立。
            self._connect_db()  # 建立连接以准备读取。
        key = self.keys[idx]  # 获取对应索引的键名。
        data = pickle.loads(self.db.begin().get(key))  # 启动事务读取数据并反序列化。
        data.id = idx  # 为数据对象记录当前索引。
        assert data.protein_pos.size(0) > 0  # 断言蛋白节点数量大于零以确保有效样本。
        if self.transform is not None:  # 若指定了变换函数。
            data = self.transform(data)  # 对数据应用变换。
        # add features extracted by molopt  # 保留注释：说明以下为 molopt 提取特征。
        if self.emb_path is not None:  # 若存在嵌入信息。
            emb = self.emb[idx]  # 获取对应索引的嵌入条目。
            data.nll = torch.cat([emb['kl_pos'][1:], emb['kl_v'][1:]]).view(1, -1)  # 拼接 KL 散度特征并重塑形状。
            data.nll_all = torch.cat([emb['kl_pos'], emb['kl_v']]).view(1, -1)  # 拼接完整 KL 散度特征向量。
            data.pred_ligand_v = torch.softmax(emb['pred_ligand_v'], dim=-1)  # 计算配体原子预测的 softmax 分布。
            data.final_h = emb['final_h']  # 保存最终隐藏表示。
            # data.final_ligand_h = emb['final_ligand_h']  # 保留注释：可能的额外特征。
            data.pred_v_entropy = torch.from_numpy(  # 计算预测分布的熵并转换为张量。
                stats.entropy(torch.softmax(emb['pred_ligand_v'], dim=-1).numpy(), axis=-1)).view(-1, 1)  # 使用 scipy 计算熵并调整形状。

        return data  # 返回处理后的数据对象。
        

if __name__ == '__main__':  # 当模块作为脚本运行时执行下列逻辑。
    import argparse  # 导入 argparse 用于命令行解析。
    parser = argparse.ArgumentParser()  # 创建参数解析器实例。
    parser.add_argument('path', type=str)  # 定义位置参数 path 用于指定数据路径。
    args = parser.parse_args()  # 解析命令行参数。

    PDBBindDataset(args.path)  # 根据传入路径初始化数据集，触发预处理。
