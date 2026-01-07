# 总结：
# - 从生成的配体样本中对每个蛋白口袋执行 QVina 或 Vina 对接打分。
# - 支持多进程并行，对每组 pocket 样本批量评估并保存结果。
# - 可指定输出路径、对接盒大小、Vina 搜索强度以及对接模式。

import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，处理路径。
import torch  # 导入 PyTorch，用于加载/保存样本。
from tqdm.auto import tqdm  # 导入 tqdm，用于显示进度条。
from utils.evaluation.docking_qvina import QVinaDockingTask  # 导入 QVina 对接任务封装。
from utils.evaluation.docking_vina import VinaDockingTask  # 导入 Vina 对接任务封装。
import multiprocessing as mp  # 导入多进程模块，用于并发处理。


def dock_pocket_samples(pocket_samples):  # 对单个口袋的所有样本进行对接。
    """执行单个口袋的对接流程，并返回带有 Vina 结果的样本列表。"""
    ligand_fn = pocket_samples[0]['ligand_filename']  # 读取当前口袋的配体文件名。
    print('Start docking pocket: %s' % ligand_fn)  # 打印开始信息。
    pocket_results = []  # 存储该口袋的对接结果。
    for idx, s in enumerate(tqdm(pocket_samples, desc='docking %d' % os.getpid())):  # 遍历每个样本。
        try:
            if args.docking_mode == 'qvina':  # 使用 QVina 对接模式。
                vina_task = QVinaDockingTask.from_generated_mol(
                    s['mol'], s['ligand_filename'], protein_root=args.protein_root, size_factor=args.dock_size_factor)
                vina_results = vina_task.run_sync(exhaustiveness=args.exhaustiveness)  # 同步运行对接并获取结果。
            elif args.docking_mode == 'vina_score':  # 使用 Vina 打分模式。
                vina_task = VinaDockingTask.from_generated_mol(
                    s['mol'], s['ligand_filename'], protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)  # 仅打分。
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)  # 最小化。
                vina_results = {
                    'score_only': score_only_results,  # 保存打分结果。
                    'minimize': minimize_results  # 保存最小化姿势及能量。
                }
            else:
                raise ValueError  # 未知模式抛出异常。
        except Exception:  # 捕获对接过程中的异常。
            print('Error at %d of %s' % (idx, ligand_fn))  # 输出错误信息。
            vina_results = None  # 将结果标记为 None。
        pocket_results.append({**s, 'vina': vina_results})  # 记录样本及其对接结果。
    return pocket_results  # 返回当前口袋的对接结果列表。


if __name__ == '__main__':  # 程序入口。
    parser = argparse.ArgumentParser()  # 创建参数解析器。
    parser.add_argument('sample_path', type=str)  # 训练样本路径，包含每个口袋的生成分子。
    parser.add_argument('-o', '--out', type=str, default=None)  # 输出路径，可选。
    parser.add_argument('-n', '--num_processes', type=int, default=10)  # 并行进程数。
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')  # 蛋白数据根目录。
    parser.add_argument('--dock_size_factor', type=float, default=None)  # 对接盒子尺寸缩放因子。
    parser.add_argument('--exhaustiveness', type=int, default=16)  # Vina 搜索强度。
    parser.add_argument('--docking_mode', type=str, default='vina_score',
                        choices=['none', 'qvina', 'vina_score'])  # 指定对接模式。
    args = parser.parse_args()  # 解析命令行参数。

    samples = torch.load(args.sample_path)  # 加载样本列表（按口袋分组）。
    with mp.Pool(args.num_processes) as p:  # 启动进程池并并行处理。
        docked_samples = p.map(dock_pocket_samples, samples)  # 对各口袋执行对接。
    if args.out is None:  # 若未指定输出路径，则自动生成。
        dir_name = os.path.dirname(args.sample_path)  # 获取原始目录。
        baseline_name = os.path.basename(args.sample_path).split('_')[0]  # 取基准名称。
        out_path = os.path.join(dir_name, baseline_name + '_test_docked.pt')  # 默认输出文件名。
    else:
        out_path = args.out  # 使用用户指定的输出路径。
    torch.save(docked_samples, out_path)  # 保存对接结果。
