import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，用于路径处理。

import numpy as np  # 导入 NumPy，做统计计算。
from rdkit import RDLogger  # 导入 RDKit 日志控制。
import torch  # 导入 PyTorch，加载与保存元数据。
from tqdm.auto import tqdm  # 导入 tqdm，显示评估进度。
from copy import deepcopy  # 导入 deepcopy，防止对原始分子修改。

from utils import misc  # 导入通用辅助函数。
from utils.evaluation import scoring_func  # 导入化学属性评估函数。
from utils.evaluation.docking_qvina import QVinaDockingTask  # 导入 QVina 对接封装。
from utils.evaluation.docking_vina import VinaDockingTask  # 导入 Vina 对接封装。
from multiprocessing import Pool  # 导入多进程池。
from functools import partial  # 导入 partial，构造并行任务。
from glob import glob  # 导入 glob，匹配结果文件。


def eval_single_datapoint(index, id, args):  # 评估单个口袋的所有重建样本。
    """对单个口袋的生成样本执行化学与对接评估，并缓存结果。"""
    if isinstance(index, dict):
        # reference set
        index = [index]

    ligand_filename = index[0]['ligand_filename']  # 口袋对应的配体文件名。
    num_samples = len(index[:100])  # 最多评估前 100 个样本。
    results = []  # 存储评估结果。
    n_eval_success = 0  # 评估成功次数。
    for sample_idx, sample_dict in enumerate(tqdm(index[:num_samples], desc='Eval', total=num_samples)):
        mol = sample_dict['mol']
        smiles = sample_dict['smiles']
        if '.' in smiles:
            continue  # 忽略多片段分子。
        # chemical and docking check
        try:
            chem_results = scoring_func.get_chem(mol)  # 计算化学指标。
            if 'vina' in sample_dict:
                vina_results = sample_dict['vina']
            else:
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, ligand_filename, protein_root=args.protein_root,
                        size_factor=args.size_factor, buffer=args.buffer)
                    vina_results = vina_task.run_sync(exhaustiveness=args.exhaustiveness)
                elif args.docking_mode == 'vina':
                    vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run(mode='dock')
                elif args.docking_mode in ['vina_full', 'vina_score']:
                    vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol),
                                                                   ligand_filename, protein_root=args.protein_root)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_full':
                        dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results.update({
                            'dock': dock_results,
                        })

                elif args.docking_mode == 'none':
                    vina_results = None
                else:
                    raise NotImplementedError
            n_eval_success += 1
        except Exception as e:
            logger.warning('Evaluation failed for %s' % f'{sample_idx}')
            print(str(e))
            continue

        results.append({
            **sample_dict,
            'chem_results': chem_results,
            'vina': vina_results
        })
    logger.info(f'Evaluate No {id} done! {num_samples} samples in total. {n_eval_success} eval success!')
    torch.save(results, os.path.join(args.result_path, f'eval_{id:03d}_{os.path.basename(ligand_filename[:-4])}.pt'))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 参数解析器。
    parser.add_argument('meta_file', type=str)  # 'sampling_results/targetdiff_vina_docked.pt'
    parser.add_argument('-n', '--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--size_factor', type=float, default=1.0)  # 搜索盒尺寸因子。
    parser.add_argument('--buffer', type=float, default=5.0)  # 搜索盒缓冲尺寸（Å）。
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--aggregate_meta', type=eval, default=False)
    args = parser.parse_args()

    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    if args.aggregate_meta:
        meta_file_list = sorted(glob(os.path.join(args.meta_file, '*/result.pt')))
        print(f'There are {len(meta_file_list)} files to aggregate')
        test_index = []
        for f in tqdm(meta_file_list, desc='Load meta files'):
            test_index.append(torch.load(f))
    else:
        test_index = torch.load(args.meta_file)
        if isinstance(test_index[0], dict):  # single datapoint sampling result
            test_index = [test_index]

    testset_results = []
    with Pool(args.num_workers) as p:
        for r in tqdm(p.starmap(partial(eval_single_datapoint, args=args),
                                zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples)))),
                      total=args.eval_num_examples, desc='Overall Eval'):
            testset_results.append(r)

    if args.result_path:
        torch.save(testset_results, os.path.join(args.result_path, f'eval_all.pt'))

    qed = [x['chem_results']['qed'] for r in testset_results for x in r]
    sa = [x['chem_results']['sa'] for r in testset_results for x in r]
    num_atoms = [len(x['pred_pos']) for r in testset_results for x in r]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    logger.info('Num atoms:   Mean: %.3f Median: %.3f' % (np.mean(num_atoms), np.median(num_atoms)))
    if args.docking_mode in ['vina', 'qvina']:
        vina = [x['vina'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_full', 'vina_score']:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for r in testset_results for x in r]
        vina_min = [x['vina']['minimize'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_full':
            vina_dock = [x['vina']['dock'][0]['affinity'] for r in testset_results for x in r]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
