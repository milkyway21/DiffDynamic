# 总结：
# - 加载指定拆分的口袋-配体测试集，为每个配体执行 QVina 对接获得亲和力分数。
# - 同时计算化学性质指标，最终将结果保存为 PyTorch 对象用于评估或可视化。
# - 可自定义受体/配体根目录、是否进行 UFF 优化以及对接盒尺寸等参数。

import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，用于路径操作。

from rdkit import Chem  # 导入 RDKit，读取配体分子。
import torch  # 导入 PyTorch，保存对接结果。
from tqdm.auto import tqdm  # 导入 tqdm，显示进度条。
from utils import misc  # 导入工具函数（日志等）。
from utils.evaluation import scoring_func  # 导入化学属性评估。
from utils.evaluation.docking_qvina import QVinaDockingTask  # 导入 QVina 对接任务。
from datasets import get_dataset  # 导入数据集工厂函数。
from easydict import EasyDict  # 导入 EasyDict，方便构造配置。


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 参数解析器。
    parser.add_argument('-d', '--dataset', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('-s', '--split', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('-o', '--out', type=str, default=None)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--ligand_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--use_uff', type=eval, default=True)
    parser.add_argument('--size_factor', type=float, default=1.2)
    parser.add_argument('--exhaustiveness', type=int, default=16)  # QVina 搜索强度。
    args = parser.parse_args()

    logger = misc.get_logger('docking')  # 创建日志器。
    logger.info(args)

    # Load dataset
    dataset, subsets = get_dataset(
        config=EasyDict({
            'name': 'pl',
            'path': args.dataset,
            'split': args.split
        })
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Dock
    logger.info('Start docking...')
    results = []
    for i, data in enumerate(tqdm(test_set)):  # 按顺序遍历测试集。
        mol = next(iter(Chem.SDMolSupplier(os.path.join(args.ligand_root, data.ligand_filename))))  # 读取原始配体。
        # try:
        chem_results = scoring_func.get_chem(mol)  # 计算配体化学属性（QED/SA 等）。
        vina_task = QVinaDockingTask.from_original_data(
            data,
            ligand_root=args.ligand_root,
            protein_root=args.protein_root,
            use_uff=args.use_uff,
            size_factor=args.size_factor
        )
        vina_results = vina_task.run_sync(exhaustiveness=args.exhaustiveness)  # 运行 QVina 并获取对接姿势与能量。
        # except:
        #     logger.warning('Error #%d' % i)
        #     continue

        results.append({
            'mol': mol,
            'smiles': data.ligand_smiles,
            'ligand_filename': data.ligand_filename,
            'chem_results': chem_results,  # 化学指标。
            'vina': vina_results  # QVina 对接结果。
        })

    # Save
    if args.out is None:
        split_name = os.path.basename(args.split)
        split_name = split_name[:split_name.rfind('.')]
        docked_name = f'{split_name}_test_docked_uff_{args.use_uff}_size_{args.size_factor}.pt'
        out_path = os.path.join(os.path.dirname(args.dataset), docked_name)
    else:
        out_path = args.out
    logger.info('Num docked: %d' % len(results))
    logger.info('Saving results to %s' % out_path)
    torch.save(results, out_path)
