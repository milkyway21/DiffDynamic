# 总结：
# - 实现基于 LMDB 的口袋-配体配对数据集加载与缓存。
# - 支持按需预处理原始数据并应用可选数据增强 transform。
# - 提供基础 Dataset 接口以供训练和推理场景使用。

import os  # 导入操作系统路径工具模块。
import pickle  # 导入 pickle 以进行对象序列化和反序列化。
import lmdb  # 导入 LMDB 作为高性能键值数据库。
from torch.utils.data import Dataset  # 从 PyTorch 导入 Dataset 基类。
from tqdm.auto import tqdm  # 导入 tqdm 自动进度条显示。

from utils.data import PDBProtein, parse_sdf_file  # 导入蛋白解析类和配体 SDF 解析函数。
from .pl_data import ProteinLigandData, torchify_dict  # 导入蛋白-配体数据结构及张量化工具函数。


class PocketLigandPairDataset(Dataset):  # 定义口袋-配体配对数据集类，继承自 PyTorch Dataset。
    """基于 LMDB 缓存的蛋白口袋-配体配对数据集，支持按需构建与变换。"""

    def __init__(self, raw_path, transform=None, version='final'):  # 初始化函数，配置原始路径、变换和版本标识。
        """构建数据集，若缓存不存在则触发预处理。

        Args:
            raw_path: 原始蛋白-配体数据所在目录（需包含 `index.pkl`）。
            transform: 可选的样本级变换函数。
            version: 缓存后缀，用于区分不同预处理配置。
        """
        super().__init__()  # 调用父类构造函数完成基础初始化。
        self.raw_path = raw_path.rstrip('/')  # 保存原始数据路径并去除尾部斜杠。
        self.index_path = os.path.join(self.raw_path, 'index.pkl')  # 计算索引文件路径。
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),  # 计算 LMDB 存储路径。
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')  # 依据版本号拼接文件名。
        self.transform = transform  # 保存可选的数据变换函数。
        self.db = None  # 初始化 LMDB 数据库连接句柄。

        self.keys = None  # 初始化键名列表缓存。

        if not os.path.exists(self.processed_path):  # 若预处理数据库文件不存在。
            print(f'{self.processed_path} does not exist, begin processing data')  # 提示即将开始处理数据。
            self._process()  # 调用内部处理函数生成 LMDB。

    def _connect_db(self):  # 定义内部方法建立只读数据库连接。
        """建立到 LMDB 缓存的只读连接，并缓存所有键。"""
        assert self.db is None, 'A connection has already been opened.'  # 确保当前没有已打开的连接。
        self.db = lmdb.open(  # 打开 LMDB 数据库以便读取。
            self.processed_path,  # 指定数据库文件路径。
            map_size=10*(1024*1024*1024),   # 10GB  # 设置最大映射容量为 10GB。
            create=False,  # 禁止创建新库。
            subdir=False,  # 指定路径为单一文件。
            readonly=True,  # 使用只读模式。
            lock=False,  # 关闭文件锁以提高并发读取能力。
            readahead=False,  # 禁用预读取以降低内存占用。
            meminit=False,  # 禁止预初始化内存。
        )
        with self.db.begin() as txn:  # 开启只读事务获取键列表。
            self.keys = list(txn.cursor().iternext(values=False))  # 遍历所有键存储在内存列表中。

    def _close_db(self):  # 定义内部方法关闭数据库连接。
        """关闭当前 LMDB 连接并清空键缓存。"""
        self.db.close()  # 关闭 LMDB 连接。
        self.db = None  # 重置连接句柄。
        self.keys = None  # 清空键列表缓存。
        
    def _process(self):  # 定义内部方法预处理原始数据并写入 LMDB。
        """遍历原始 `index.pkl`，将解析后的样本写入 LMDB 缓存。"""
        db = lmdb.open(  # 打开 LMDB 数据库准备写入。
            self.processed_path,  # 指定输出文件路径。
            map_size=10*(1024*1024*1024),   # 10GB  # 设置最大容量为 10GB。
            create=True,  # 若不存在则创建数据库。
            subdir=False,  # 以文件形式存储。
            readonly=False,  # Writable  # 启用写模式以插入数据。
        )
        with open(self.index_path, 'rb') as f:  # 打开索引文件并以二进制方式读取。
            index = pickle.load(f)  # 反序列化索引列表。

        num_skipped = 0  # 初始化跳过计数器。
        with db.begin(write=True, buffers=True) as txn:  # 开启写事务以批量写入数据。
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):  # 遍历索引条目并显示进度条。
                if pocket_fn is None: continue  # 如果缺少口袋文件路径则跳过当前样本。
                try:  # 捕获数据解析过程中的潜在异常。
                    # data_prefix = '/data/work/jiaqi/binding_affinity'  # 保留注释：原始数据前缀示例。
                    data_prefix = self.raw_path  # 使用当前原始路径作为数据前缀。
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()  # 解析蛋白口袋并转换为原子级字典。
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))  # 解析配体 SDF 文件为字典格式。
                    data = ProteinLigandData.from_protein_ligand_dicts(  # 构造蛋白-配体数据对象。
                        protein_dict=torchify_dict(pocket_dict),  # 将蛋白字典转换为张量字段。
                        ligand_dict=torchify_dict(ligand_dict),  # 将配体字典转换为张量字段。
                    )
                    data.protein_filename = pocket_fn  # 记录蛋白文件名以追溯原始数据。
                    data.ligand_filename = ligand_fn  # 记录配体文件名。
                    data = data.to_dict()  # avoid torch_geometric version issue  # 将数据转换为字典以规避 PyG 版本兼容问题。
                    txn.put(  # 将样本写入 LMDB。
                        key=str(i).encode(),  # 使用样本索引字符串作为键。
                        value=pickle.dumps(data)  # 序列化数据字典作为值存储。
                    )
                except Exception:  # 捕获所有异常以避免流程中断。
                    num_skipped += 1  # 增加跳过计数。
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))  # 输出被跳过的样本信息。
                    continue  # 继续处理下一条样本。
        db.close()  # 关闭写数据库连接。
    
    def __len__(self):  # 实现数据集长度接口。
        """返回数据集中样本数量（懒加载 LMDB 连接）。"""
        if self.db is None:  # 若数据库尚未连接。
            self._connect_db()  # 建立连接以读取键列表。
        return len(self.keys)  # 返回样本总数量。

    def __getitem__(self, idx):  # 实现按索引访问样本的接口。
        """根据索引读取样本，应用 transform 后返回。"""
        data = self.get_ori_data(idx)  # 调用内部方法读取原始数据。
        if self.transform is not None:  # 若存在变换函数。
            data = self.transform(data)  # 对数据应用变换后返回。
        return data  # 返回最终数据对象。

    def get_ori_data(self, idx):  # 定义内部方法获取未经过 transform 的原始数据。
        """读取 LMDB 中的原始样本并包装为 `ProteinLigandData`。"""
        if self.db is None:  # 若当前无数据库连接。
            self._connect_db()  # 建立连接以便读取。
        key = self.keys[idx]  # 获取对应索引的键名。
        data = pickle.loads(self.db.begin().get(key))  # 读取并反序列化 LMDB 中的数据。
        data = ProteinLigandData(**data)  # 将字典转换为 ProteinLigandData 对象。
        data.id = idx  # 记录当前样本索引。
        assert data.protein_pos.size(0) > 0  # 确保蛋白节点数量有效。
        return data  # 返回原始数据对象。
        

if __name__ == '__main__':  # 当脚本直接运行时执行以下逻辑。
    import argparse  # 导入 argparse 以解析命令行参数。
    parser = argparse.ArgumentParser()  # 创建参数解析器实例。
    parser.add_argument('path', type=str)  # 添加位置参数以指定数据路径。
    args = parser.parse_args()  # 解析命令行参数。

    dataset = PocketLigandPairDataset(args.path)  # 根据传入路径构造数据集。
    print(len(dataset), dataset[0])  # 打印数据集大小与首个样本，用于快速验证。
import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
