# 总结：
# - 提供配置加载、日志管理、随机种子设定等通用工具函数。
# - 定义空对象 `BlackHole` 用于吞掉任意属性/调用，简化空实现。
# - 包含 TensorBoard 超参记录与字符串解析等辅助工具。

import logging  # 导入日志模块。
import os  # 导入 OS 模块，用于文件路径操作。
import random  # 导入随机数模块。
import time  # 导入时间模块，用于时间戳。

import numpy as np  # 导入 NumPy。
import torch  # 导入 PyTorch。
import yaml  # 导入 YAML 解析器。
from easydict import EasyDict  # 导入 EasyDict，支持属性访问形式的字典。


class BlackHole(object):  # 定义一个吞噬所有属性与调用的空对象。
    """空对象实现：吞掉任意属性与调用，常用于替代空日志器。"""
    def __setattr__(self, name, value):  # 拦截属性赋值。
        pass  # 丢弃所有赋值操作。

    def __call__(self, *args, **kwargs):  # 拦截函数调用。
        return self  # 返回自身以便链式调用。

    def __getattr__(self, name):  # 拦截属性访问。
        return self  # 返回自身，从而继续吞噬后续操作。


def load_config(path):  # 从 YAML 文件加载配置并转换为 EasyDict。
    """读取 YAML 配置文件并转换为 `EasyDict`。"""
    with open(path, 'r', encoding='utf-8') as f:  # 打开配置文件，使用 UTF-8 编码。
        return EasyDict(yaml.safe_load(f))  # 解析 YAML 并转换为可属性访问的字典。


def get_logger(name, log_dir=None):  # 创建带有控制台与可选文件输出的日志器。
    """创建命名日志器，可选写入指定目录的 `log.txt`。

    Args:
        name: 日志器名称。
        log_dir: 若提供，则额外输出到该目录下的 `log.txt`。

    Returns:
        logging.Logger: 配置好的日志器实例。
    """
    logger = logging.getLogger(name)  # 获取或创建命名日志器。
    logger.setLevel(logging.DEBUG)  # 设定日志级别为 DEBUG。
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')  # 定义输出格式。

    stream_handler = logging.StreamHandler()  # 创建控制台处理器。
    stream_handler.setLevel(logging.DEBUG)  # 设置控制台日志级别。
    stream_handler.setFormatter(formatter)  # 应用格式器。
    logger.addHandler(stream_handler)  # 附加控制台处理器。

    if log_dir is not None:  # 若指定日志目录。
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))  # 创建文件处理器。
        file_handler.setLevel(logging.DEBUG)  # 设置文件日志级别。
        file_handler.setFormatter(formatter)  # 应用格式器。
        logger.addHandler(file_handler)  # 附加文件处理器。

    return logger  # 返回配置好的日志器。


def get_new_log_dir(root='./logs', prefix='', tag=''):  # 根据时间戳生成新的日志目录。
    """基于时间戳创建唯一日志目录，返回绝对路径。"""
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())  # 生成时间戳字符串。
    if prefix != '':  # 若提供前缀。
        fn = prefix + '_' + fn  # 在开头添加前缀。
    if tag != '':  # 若提供标签。
        fn = fn + '_' + tag  # 在末尾添加标签。
    log_dir = os.path.join(root, fn)  # 拼接目录路径。
    os.makedirs(log_dir)  # 创建目录。
    return log_dir  # 返回新目录路径。


def seed_all(seed):  # 设置 PyTorch、NumPy 和标准库的随机种子。
    """为 PyTorch/NumPy/`random` 统一设定随机种子。"""
    torch.manual_seed(seed)  # 设定 PyTorch 随机种子。
    np.random.seed(seed)  # 设定 NumPy 随机种子。
    random.seed(seed)  # 设定标准库随机种子。


def log_hyperparams(writer, args):  # 将超参数记录到 TensorBoard。
    """将命令行参数写入 TensorBoard 超参面板。"""
    from torch.utils.tensorboard.summary import hparams  # 延迟导入避免依赖。
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}  # 将参数值转换为字符串。
    exp, ssi, sei = hparams(vars_args, {})  # 构建超参数摘要。
    writer.file_writer.add_summary(exp)  # 写入实验超参数。
    writer.file_writer.add_summary(ssi)  # 写入输入摘要。
    writer.file_writer.add_summary(sei)  # 写入指标摘要。


def int_tuple(argstr):  # 将逗号分隔的字符串解析为整数元组。
    """将形如 `'1,2,3'` 的字符串解析成整数元组。"""
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):  # 将逗号分隔的字符串解析为字符串元组。
    """将逗号分隔字符串拆分为元组。"""
    return tuple(argstr.split(','))


def count_parameters(model):  # 统计模型可训练参数数量。
    """统计模型中可训练参数的总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
