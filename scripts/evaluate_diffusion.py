# 总结：
# - 加载扩散模型生成的配体样本，逐步检查稳定性、重建分子、计算化学属性与对接得分。
# - 统计原子类型分布、键长分布、环结构等指标，并将结果与图表保存到指定目录。
# - 支持选择不同的对接模式（QVina/Vina）和评估步数，便于比较不同训练阶段的样本质量。

import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，用于路径操作。

import numpy as np  # 导入 NumPy，处理数组与统计。
from rdkit import Chem  # 导入 RDKit 化学模块。
from rdkit import RDLogger  # 导入 RDKit 日志控制。
import torch  # 导入 PyTorch，加载保存张量数据。
from tqdm.auto import tqdm  # 导入 tqdm，用于进度显示。
from glob import glob  # 导入 glob，查找匹配文件。
from collections import Counter  # 导入 Counter，统计原子/环分布。

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length  # 导入评估工具。
from utils import misc, reconstruct, transforms  # 导入通用工具、重建与特征转换模块。
from utils.evaluation.docking_qvina import QVinaDockingTask  # 导入 QVina 对接封装。
from utils.evaluation.docking_vina import VinaDockingTask  # 导入 Vina 对接封装。


def print_dict(d, logger):  # 将字典中的指标打印到日志。
    """逐项打印指标字典，方便对齐日志输出。"""
    for k, v in d.items():
        if v is not None:  # 若值存在，格式化为 4 位小数。
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')  # 否则记录 None。


def print_ring_ratio(all_ring_sizes, logger):  # 打印不同环尺寸出现比例。
    """统计各类环尺寸出现频次并写入日志。"""
    for ring_size in range(3, 10):  # 针对常见的 3~9 元环。
        n_mol = 0
        for counter in all_ring_sizes:  # 遍历每个分子的环计数。
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')  # 输出比例。


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器。
    parser.add_argument('sample_path', type=str)  # 指定生成样本所在目录。
    parser.add_argument('--verbose', type=eval, default=False)  # 控制日志详情。
    parser.add_argument('--eval_step', type=int, default=-1)  # 使用扩散轨迹的哪一时间步。
    parser.add_argument('--eval_num_examples', type=int, default=None)  # 限制评估文件数量。
    parser.add_argument('--save', type=eval, default=True)  # 是否保存评估结果与图表。
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')  # 蛋白根目录。
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')  # 原子类型编码模式。
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])  # 对接模式。
    parser.add_argument('--exhaustiveness', type=int, default=16)  # Vina 搜索强度。
    parser.add_argument('--size_factor', type=float, default=1.0)  # 搜索盒尺寸因子。
    parser.add_argument('--buffer', type=float, default=5.0)  # 搜索盒缓冲尺寸（Å）。
    args = parser.parse_args()  # 解析参数。

    result_path = os.path.join(args.sample_path, 'eval_results')  # 结果输出目录。
    os.makedirs(result_path, exist_ok=True)  # 确保目录存在。
    logger = misc.get_logger('evaluate', log_dir=result_path)  # 创建日志记录器。
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')  # 关闭 RDKit 冗余输出。

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))  # 搜索所有结果文件。
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))  # 按步数排序。
    if args.eval_num_examples is not None:  # 若限制数量，则截断列表。
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)  # 统计样本文件数。
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0  # 累计样本数量。
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0  # 累计稳定性指标。
    n_recon_success, n_eval_success, n_complete = 0, 0, 0  # 重建/评估成功计数。
    results = []  # 存储成功分子的详细结果。
    all_pair_dist, all_bond_dist = [], []  # 累积所有样本的距离统计。
    all_atom_types = Counter()  # 累积所有样本的原子类型频率。
    success_pair_dist, success_atom_types = [], Counter()  # 对成功分子统计。
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):  # 遍历每个结果文件。
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']  # 预测原子类型轨迹。
        num_samples += len(all_pred_ligand_pos)  # 累加样本数。

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]  # 取指定时间步的结果。

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)  # one-hot -> 原子序号。
            all_atom_types += Counter(pred_atom_type)  # 累加原子类型统计。
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)  # 检查结构稳定性。
            all_mol_stable += r_stable[0]  # 累加稳定分子数量。
            all_atom_stable += r_stable[1]  # 累加稳定原子数量。
            all_n_atom += r_stable[2]  # 累加总原子数。

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)  # 计算原子对距离。
            all_pair_dist += pair_dist  # 累加所有样本的原子对距离。

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)  # 识别芳香性。
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)  # RDKit 重建。
                smiles = Chem.MolToSmiles(mol)  # 转为 SMILES。
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')  # 打印失败信息。
                continue  # 重建失败则跳过。
            n_recon_success += 1  # 重建成功计数。

            if '.' in smiles:  # 若 SMILES 包含多个分子碎片，则跳过。
                continue
            n_complete += 1  # 统计完整分子数量。

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)  # 计算化学性质（QED/SA 等）。
                if args.docking_mode == 'qvina':  # 运行 QVina 对接。
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root,
                        size_factor=args.size_factor, buffer=args.buffer)
                    vina_results = vina_task.run_sync(exhaustiveness=args.exhaustiveness)
                elif args.docking_mode in ['vina_score', 'vina_dock']:  # 运行 Vina 打分/最小化。
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':  # 如需完整对接则追加 dock。
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None  # 不运行对接时设为 None。

                n_eval_success += 1  # 统计评估成功数量。
            except Exception:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')  # 打印错误信息。
                continue  # 评估失败时跳过。

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)  # 计算键长分布。
            all_bond_dist += bond_dist  # 累计所有样本的键长。

            success_pair_dist += pair_dist  # 收集成功样本的原子对距离。
            success_atom_types += Counter(pred_atom_type)  # 收集成功样本的原子类型分布。

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results
            })  # 保存该样本的详细信息。
    logger.info(f'Evaluate done! {num_samples} samples in total.')  # 输出评估总结。

    fraction_mol_stable = all_mol_stable / num_samples  # 分子级稳定率。
    fraction_atm_stable = all_atom_stable / all_n_atom  # 原子级稳定率。
    fraction_recon = n_recon_success / num_samples  # 重建成功率。
    fraction_eval = n_eval_success / num_samples  # 评估成功率。
    fraction_complete = n_complete / num_samples  # 完整分子比例。
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }  # 汇总稳定性指标。
    print_dict(validity_dict, logger)  # 打印稳定性统计。

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)  # 统计键长分布。
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)  # 计算键长 JS。
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)  # 打印键长评估结果。

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)  # 原子对距离直方图。
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)  # 计算 JS 指标。
    print_dict(success_js_metrics, logger)  # 输出原子对距离评估结果。

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)  # 计算原子类型分布 JS。
    if atom_type_js is not None:
        logger.info('Atom type JS: %.4f' % atom_type_js)
    else:
        logger.info('Atom type JS: None (no successful samples)')

    if args.save:  # 如需保存图像，则绘制原子对距离直方图。
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))  # 打印重建与评估成功数量。

    qed = [r['chem_results']['qed'] for r in results]  # 收集 QED 分数。
    sa = [r['chem_results']['sa'] for r in results]  # 收集合成可行性分数。
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))  # 输出 QED 统计。
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))  # 输出 SA 统计。
    if args.docking_mode == 'qvina':  # 若运行 QVina，对接能量统计。
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:  # 统计 Vina score/minimize。
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':  # 若额外执行 dock。
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)  # 输出环结构比例。

    if args.save:  # 保存整体评估结果。
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))
