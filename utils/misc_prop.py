# 总结：
# - 提供性质预测任务常用的评估指标、数据加载器构造与模型工厂函数。
# - 支持无外部特征与携带外部编码的模型选择。
# - 统一抽象训练/验证/测试加载器的批处理参数。

import numpy as np  # 导入 NumPy。
from scipy.stats import pearsonr, spearmanr  # 导入相关系数计算。
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 导入回归评估指标。
from torch_geometric.loader import DataLoader  # 导入 PyG 数据加载器。

from models.property_pred.prop_model import PropPredNet, PropPredNetEnc  # 导入性质预测模型。


def get_eval_scores(ypred_arr, ytrue_arr, logger, prefix='All'):  # 根据预测与真实值计算并打印回归指标。
    """计算并日志输出常用回归指标。

    Args:
        ypred_arr: 预测值数组。
        ytrue_arr: 真实标签数组。
        logger: 日志器，用于输出结果。
        prefix: 指标打印前缀（如数据子集名称）。

    Returns:
        float | None: 返回 RMSE，若输入为空则返回 None。
    """
    if len(ypred_arr) == 0:  # 若无预测结果。
        return None  # 返回空值。
    rmse = np.sqrt(mean_squared_error(ytrue_arr, ypred_arr))  # 计算 RMSE。
    mae = mean_absolute_error(ytrue_arr, ypred_arr)  # 计算 MAE。
    r2 = r2_score(ytrue_arr, ypred_arr)  # 计算决定系数。
    pearson, ppval = pearsonr(ytrue_arr, ypred_arr)  # 计算 Pearson 相关。
    spearman, spval = spearmanr(ytrue_arr, ypred_arr)  # 计算 Spearman 相关。
    mean = np.mean(ypred_arr)  # 预测值均值。
    std = np.std(ypred_arr)  # 预测值标准差。
    logger.info("Evaluation Summary:")  # 打印摘要头。
    logger.info(  # 打印详细指标。
        "[%4s] num: %3d, RMSE: %.3f, MAE: %.3f, "
        "R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (
            prefix, len(ypred_arr), rmse, mae, r2, pearson, spearman, mean, std))
    return rmse  # 返回 RMSE 作为主要指标。


def get_dataloader(train_set, val_set, test_set, config):  # 构建训练/验证/测试数据加载器。
    """根据配置构造训练/验证/测试的数据加载器。

    Args:
        train_set: 训练数据集合。
        val_set: 验证数据集合。
        test_set: 测试数据集合。
        config: 包含 `train.batch_size` 与 `train.num_workers` 的配置。

    Returns:
        tuple: `(train_loader, val_loader, test_loader)`。
    """
    follow_batch = ['protein_element', 'ligand_element']  # 需要跟踪的批次字段。
    collate_exclude_keys = ['ligand_nbh_list']  # 聚合时排除的键。
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys
    )  # 训练集加载器，启用打乱。
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=collate_exclude_keys)  # 验证集加载器。
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=collate_exclude_keys)  # 测试集加载器。
    return train_loader, val_loader, test_loader  # 返回三个加载器。


def get_model(config, protein_atom_feat_dim, ligand_atom_feat_dim):  # 根据配置选择性质预测模型。
    """根据模型配置构建性质预测网络。

    Args:
        config: 模型配置对象，需包含 `model`、`model.encoder` 等字段。
        protein_atom_feat_dim: 蛋白原子特征维度。
        ligand_atom_feat_dim: 配体原子特征维度。

    Returns:
        torch.nn.Module: 已实例化的性质预测模型。
    """
    if config.model.encoder.name == 'egnn_enc':  # 若编码器为带外部特征版本。
        model = PropPredNetEnc(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            enc_ligand_dim=config.model.enc_ligand_dim,
            enc_node_dim=config.model.enc_node_dim,
            enc_graph_dim=config.model.enc_graph_dim,
            enc_feature_type=config.model.enc_feature_type,
            output_dim=1
        )
    else:  # 默认使用基础模型。
        model = PropPredNet(
            config.model,
            protein_atom_feature_dim=protein_atom_feat_dim,
            ligand_atom_feature_dim=ligand_atom_feat_dim,
            output_dim=3
        )
    return model  # 返回构建好的模型。
