# 总结：
# - 解析配置并构建蛋白-配体扩散模型训练流程，含日志、数据加载、模型与优化器初始化。
# - 定义 `train`、`validate` 两个核心循环，处理梯度累积、噪声扰动、评价指标与学习率调度。
# - 支持保存最佳权重与 TensorBoard 记录，便于监控训练过程。

import argparse  # 导入 argparse，用于解析命令行参数。
import os  # 导入 os，处理路径与目录。
import shutil  # 导入 shutil，用于复制文件/目录。

import numpy as np  # 导入 NumPy，执行数值运算。
import torch  # 导入 PyTorch，构建模型与训练。
import torch.utils.tensorboard  # 导入 TensorBoard 写入器模块。
from sklearn.metrics import roc_auc_score  # type: ignore  # 从 sklearn 导入 ROC AUC 指标。
from torch.nn.utils import clip_grad_norm_  # 导入梯度裁剪函数。
from torch_geometric.loader import DataLoader  # 导入 PyG 数据加载器。
from torch_geometric.transforms import Compose  # 导入转换组合工具。
from tqdm.auto import tqdm  # 导入 tqdm，用于进度条显示。

import utils.misc as misc  # 导入工具函数：日志、配置、随机种子等。
import utils.train as utils_train  # 导入自定义训练工具（优化器、调度器等）。
import utils.transforms as trans  # 导入特征转换工具。
from datasets import get_dataset  # 导入数据集工厂函数。
from datasets.pl_data import FOLLOW_BATCH  # 导入 PyG follow_batch 配置。
from models.molopt_score_model import ScorePosNet3D, DiffDynamic  # 导入模型类。


def get_auroc(y_true, y_pred, feat_mode):  # 计算多分类的加权 AUC。
    """按类别出现频率加权计算多分类 AUROC，并打印各类别指标。"""
    y_true = np.array(y_true)  # 转换标签为 NumPy 数组。
    y_pred = np.array(y_pred)  # 转换预测为 NumPy 数组。
    avg_auroc = 0.  # 初始化加权 AUROC。
    possible_classes = set(y_true)  # 收集出现过的类别。
    for c in possible_classes:  # 对每个类别单独计算二分类 AUROC。
        auroc = roc_auc_score(y_true == c, y_pred[:, c])  # 计算当前类别的 AUROC。
        avg_auroc += auroc * np.sum(y_true == c)  # 按类别样本数加权。
        mapping = {  # 定义不同特征模式下的类别到元素映射。
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')  # 输出每类 AUROC。
    return avg_auroc / len(y_true)  # 返回加权平均 AUROC。


if __name__ == '__main__':  # 仅当脚本直接运行时执行。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器。
    parser.add_argument('config', type=str)  # 添加配置文件路径参数。
    parser.add_argument('--device', type=str, default='cuda')  # 添加设备参数，默认 GPU。
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')  # 添加日志目录参数。
    parser.add_argument('--tag', type=str, default='')  # 添加标签参数，用于区分实验。
    parser.add_argument('--train_report_iter', type=int, default=200)  # 添加训练日志间隔。
    args = parser.parse_args()  # 解析命令行参数。

    # Load configs
    config = misc.load_config(args.config)  # 加载 YAML 配置。
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]  # 提取配置名称。
    misc.seed_all(config.train.seed)  # 设定随机种子，保证可复现。

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)  # 创建新的日志目录。
    ckpt_dir = os.path.join(log_dir, 'checkpoints')  # 定义检查点目录。
    os.makedirs(ckpt_dir, exist_ok=True)  # 确保检查点目录存在。
    vis_dir = os.path.join(log_dir, 'vis')  # 定义可视化输出目录。
    os.makedirs(vis_dir, exist_ok=True)  # 创建可视化目录。
    logger = misc.get_logger('train', log_dir)  # 初始化日志记录器。
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)  # 创建 TensorBoard 写入器。
    logger.info(args)  # 记录命令行参数。
    logger.info(config)  # 记录配置内容。
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))  # 备份配置文件。
    shutil.copytree('./models', os.path.join(log_dir, 'models'))  # 备份模型目录以便复现。

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()  # 实例化蛋白原子特征器。
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)  # 按模式构建配体特征器。
    transform_list = [  # 构建转换列表。
        protein_featurizer,  # 添加蛋白特征转换。
        ligand_featurizer,  # 添加配体特征转换。
        trans.FeaturizeLigandBond(),  # 添加配体键特征转换。
    ]
    if config.data.transform.random_rot:  # 若启用随机旋转。
        transform_list.append(trans.RandomRotation())  # 添加随机旋转转换。
    transform = Compose(transform_list)  # 将转换组合为单一变换。

    # Datasets and loaders
    logger.info('Loading dataset...')  # 打印数据加载提示。
    dataset, subsets = get_dataset(  # 根据配置构建数据集及切分。
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']  # 获取训练与验证子集。
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')  # 记录数据集规模。

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']  # collate 时排除的键，减小数据量。
    train_iterator = utils_train.inf_iterator(DataLoader(  # 构建无限循环的训练迭代器。
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)  # 构建验证集加载器。

    # Model
    logger.info('Building model...')  # 打印模型构建信息。
    model_name = getattr(config.model, 'name', 'score').lower()  # 读取模型名称，默认为 score。
    # 支持 glintdm 和 diffdynamic 两种配置值（向后兼容）
    model_cls = DiffDynamic if model_name in ('glintdm', 'diffdynamic') else ScorePosNet3D  # 根据名称选择模型类。
    model = model_cls(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)  # 实例化模型并移动到目标设备。
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')  # 输出特征维度。
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')  # 记录可训练参数数量。

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)  # 初始化优化器。
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)  # 初始化学习率调度器。


    def train(it):  # 定义单次训练步骤。
        """执行一次梯度更新（含累积），记录损失与学习率等信息。"""
        model.train()  # 切换模型到训练模式。
        optimizer.zero_grad()  # 清空优化器梯度。
        for _ in range(config.train.n_acc_batch):  # 支持梯度累积。
            batch = next(train_iterator).to(args.device)  # 获取一个批次并移动到设备。

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std  # 生成蛋白位置噪声。
            gt_protein_pos = batch.protein_pos + protein_noise  # 将噪声加到蛋白位置。
            results = model.get_diffusion_loss(  # 计算扩散损失。
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss = results['loss']  # 总损失。
            loss_pos = results['loss_pos']  # 位置损失。
            loss_v = results['loss_v']  # 原子类型损失。
            loss_v2 = results.get('loss_v2')  # 可选的额外损失。
            loss = loss / config.train.n_acc_batch  # 按累积次数平均。
            loss.backward()  # 反向传播累积梯度。
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)  # 梯度裁剪并返回裁剪前范数。
        optimizer.step()  # 更新参数。

        if it % args.train_report_iter == 0:  # 达到日志间隔时记录。
            log_msg = '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f' % (
                it, loss, loss_pos, loss_v
            )
            if loss_v2 is not None and torch.is_tensor(loss_v2):  # 若存在 loss_v2。
                log_msg += ' | v2 %.6f' % float(loss_v2)
            log_msg += ') | Lr: %.6f | Grad Norm: %.6f' % (
                optimizer.param_groups[0]['lr'], orig_grad_norm
            )
            logger.info(log_msg)  # 输出日志。
            for k, v in results.items():  # 遍历结果写入 TensorBoard。
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)  # 记录学习率。
            writer.add_scalar('train/grad', orig_grad_norm, it)  # 记录梯度范数。
            writer.flush()  # 刷新写入器。


    def validate(it):  # 定义验证流程。
        """在验证集上评估扩散损失与 AUROC，更新调度器并记录最优指标。"""
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0  # 初始化累计指标。
        sum_loss_bond, sum_loss_non_bond = 0, 0  # 预留键相关指标（当前未使用）。
        all_pred_v, all_true_v = [], []  # 收集原子类型预测与标签。
        all_pred_bond_type, all_gt_bond_type = [], []  # 预留键类型列表（当前未使用）。
        with torch.no_grad():  # 验证时关闭梯度。
            model.eval()  # 切换模型到评估模式。
            for batch in tqdm(val_loader, desc='Validate'):  # 遍历验证数据。
                batch = batch.to(args.device)  # 将数据移动到设备。
                batch_size = batch.num_graphs  # 获取批次大小（分子数）。
                t_loss, t_loss_pos, t_loss_v = [], [], []  # 保留每个时间步的损失（当前未使用）。
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):  # 在 10 个时间步上评估。
                    time_step = torch.tensor([t] * batch_size).to(args.device)  # 构建时间步张量。
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']  # 读取损失。

                    sum_loss += float(loss) * batch_size  # 累加总损失。
                    sum_loss_pos += float(loss_pos) * batch_size  # 累加位置损失。
                    sum_loss_v += float(loss_v) * batch_size  # 累加类型损失。
                    sum_n += batch_size  # 增加样本数量。
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())  # 收集预测。
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())  # 收集真实标签。

        avg_loss = sum_loss / sum_n  # 计算平均总损失。
        avg_loss_pos = sum_loss_pos / sum_n  # 平均位置损失。
        avg_loss_v = sum_loss_v / sum_n  # 平均类型损失。
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)  # 计算微观 AUROC。

        if config.train.scheduler.type == 'plateau':  # 根据调度类型更新学习率。
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )  # 打印验证指标。
        writer.add_scalar('val/loss', avg_loss, it)  # 记录验证总损失。
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)  # 记录位置损失。
        writer.add_scalar('val/loss_v', avg_loss_v, it)  # 记录类型损失。
        writer.flush()  # 刷新写入器。
        return avg_loss  # 返回验证损失。


    try:  # 捕获键盘中断。
        best_loss, best_iter = None, None  # 初始化最佳验证损失记录。
        for it in range(1, config.train.max_iters + 1):  # 主训练循环。
            # with torch.autograd.detect_anomaly():
            train(it)  # 执行一次训练。
            if it % config.train.val_freq == 0 or it == config.train.max_iters:  # 到达验证间隔或最后一次迭代。
                val_loss = validate(it)  # 运行验证。
                if best_loss is None or val_loss < best_loss:  # 更新最佳模型。
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it  # 记录最佳损失及迭代。
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)  # 构造检查点路径。
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)  # 保存检查点。
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')  # 未提升时提示。
    except KeyboardInterrupt:  # 捕获 Ctrl+C。
        logger.info('Terminating...')  # 打印终止信息。
