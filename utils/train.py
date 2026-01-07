import copy  # 导入副本工具。
import warnings  # 导入警告模块。

import numpy as np  # 导入 NumPy。
import torch  # 导入 PyTorch。
from torch_geometric.data import Data, Batch  # 导入 PyG 数据与批处理类。

from utils.warmup import GradualWarmupScheduler  # 导入自定义热身调度器。


# customize exp lr scheduler with min lr  # 保留注释：自定义指数学习率调度器，带最小 lr。
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    """在 PyTorch 指数调度器基础上增加最小学习率下限。"""
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):  # 初始化调度器。
        self.gamma = gamma  # 保存衰减系数。
        self.min_lr = min_lr  # 保存最小学习率。
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)  # 调用父类构造。

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) -> Batch:
    """深拷贝单个样本若干次并打包为批次。"""
    datas = [copy.deepcopy(data) for i in range(num_repeat)]  # 重复复制数据对象。
    return Batch.from_data_list(datas)  # 转换为 PyG 批数据。


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    """将批次中所有样本重复指定次数并重新拼接为新批次。"""
    datas = batch.to_data_list()  # 转为 Data 列表。
    new_data = []  # 存放复制后的数据。
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)  # 返回新的批次。


def inf_iterator(iterable):
    """将可迭代对象包装为无限循环的生成器。"""
    iterator = iterable.__iter__()  # 获取迭代器。
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()  # 迭代完毕后重置实现无限循环。


def get_optimizer(cfg, model):
    """依据配置创建优化器，目前支持 Adam。"""
    if cfg.type == 'adam':  # 支持 Adam 优化器。
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    """依据配置创建学习率调度器，支持 plateau/warmup/exponential 等策略。"""
    if cfg.type == 'plateau':  # 使用 ReduceLROnPlateau 调度器。
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    elif cfg.type == 'warmup_plateau':  # 先热身再使用 Plateau 调度。
        return GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.multiplier,
            total_epoch=cfg.total_epoch,
            after_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr
            )
        )
    elif cfg.type == 'expmin':  # 指数衰减并设定最小学习率。
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':  # 指定里程碑后计算对应 gamma。
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)
