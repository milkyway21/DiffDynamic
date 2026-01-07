# 总结：
# - 定义扩散式配体优化模型的核心组件，包括调度函数、概率工具与主模型结构。
# - 实现坐标与离散原子类型的联合扩散、采样与精炼流程。
# - 提供大步/精炼/动态采样等多种生成策略，支持可配置的训练与推理。

import numpy as np  # 导入 NumPy，用于数值计算与数组操作。
import torch  # 导入 PyTorch 主库，用于张量与自动求导。
import torch.nn as nn  # 导入神经网络模块，构建可学习组件。
import torch.nn.functional as F  # 导入函数式接口，提供常用函数。
from torch_scatter import scatter_sum, scatter_mean  # 导入散射聚合函数，用于批次化聚合。
from tqdm.auto import tqdm  # 导入 tqdm 自动模式，显示进度条。

from models.common import compose_context, ShiftedSoftplus  # 从公共模块导入上下文拼接与平移 Softplus。
from models.egnn import EGNN  # 导入 EGNN 模型，用于几何更新。
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral  # 导入通用 UniTransformer 模型。


def get_refine_net(refine_net_type, config):  # 根据类型与配置构建精炼网络。
    if refine_net_type == 'uni_o2':  # 当选择 UniTransformer 架构时。
        refine_net = UniTransformerO2TwoUpdateGeneral(  # 初始化 UniTransformer 精炼网络。
            num_blocks=config.num_blocks,  # 设置块数。
            num_layers=config.num_layers,  # 设置每块层数。
            hidden_dim=config.hidden_dim,  # 设置隐藏维度。
            n_heads=config.n_heads,  # 设置注意力头数。
            k=config.knn,  # 设置 kNN 邻居数。
            edge_feat_dim=config.edge_feat_dim,  # 设置边特征维度。
            num_r_gaussian=config.num_r_gaussian,  # 设置径向高斯数量。
            num_node_types=config.num_node_types,  # 设置节点类型数量。
            act_fn=config.act_fn,  # 设置激活函数。
            norm=config.norm,  # 控制是否开启归一化。
            cutoff_mode=config.cutoff_mode,  # 设置邻接截断模式。
            ew_net_type=config.ew_net_type,  # 设置边权网络类型。
            num_x2h=config.num_x2h,  # 设置坐标到特征的更新次数。
            num_h2x=config.num_h2x,  # 设置特征到坐标的更新次数。
            r_max=config.r_max,  # 设置最大距离阈值。
            x2h_out_fc=config.x2h_out_fc,  # 设置是否在 x2h 末尾使用全连接层。
            sync_twoup=config.sync_twoup  # 设置双更新同步开关。
        )
    elif refine_net_type == 'egnn':  # 当选择 EGNN 架构时。
        refine_net = EGNN(  # 初始化 EGNN 精炼网络。
            num_layers=config.num_layers,  # 设置层数。
            hidden_dim=config.hidden_dim,  # 设置隐藏维度。
            edge_feat_dim=config.edge_feat_dim,  # 设置边特征维度。
            num_r_gaussian=1,  # 固定使用单一径向尺度（EGNN 中默认）。
            k=config.knn,  # 设置 kNN 邻居数。
            cutoff_mode=config.cutoff_mode  # 设置边构建模式。
        )
    else:  # 当类型不被支持时。
        raise ValueError(refine_net_type)  # 抛出异常提示。
    return refine_net  # 返回构建好的精炼网络实例。


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):  # 根据策略生成 beta 调度序列。
    def sigmoid(x):  # 定义内部 sigmoid 函数。
        return 1 / (np.exp(-x) + 1)  # 返回标准 sigmoid 输出。

    if beta_schedule == "quad":  # 二次调度：beta 取平方递增。
        betas = (  # 计算 beta 序列。
                np.linspace(
                    beta_start ** 0.5,  # 起始值开方。
                    beta_end ** 0.5,  # 结束值开方。
                    num_diffusion_timesteps,  # 采样步数。
                    dtype=np.float64,  # 使用双精度浮点。
                )
                ** 2  # 再平方以得到二次分布。
        )
    elif beta_schedule == "linear":  # 线性调度。
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64  # 等间距生成序列。
        )
    elif beta_schedule == "const":  # 常数调度。
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)  # 使用固定值填充。
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1  # Jensen-Shannon 分布调度。
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64  # 生成倒数序列。
        )
    elif beta_schedule == "sigmoid":  # Sigmoid 调度。
        betas = np.linspace(-6, 6, num_diffusion_timesteps)  # 先在区间内均匀采样。
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start  # 通过 sigmoid 缩放到目标范围。
    else:  # 不支持的调度类型。
        raise NotImplementedError(beta_schedule)  # 抛出未实现错误。
    assert betas.shape == (num_diffusion_timesteps,)  # 断言序列长度正确。
    return betas  # 返回 beta 序列。


def cosine_beta_schedule(timesteps, s=0.008):  # 生成余弦调度的 alpha 序列。
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1  # 计算步数边界。
    x = np.linspace(0, steps, steps)  # 在步数区间均匀采样。
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 根据论文公式计算累计 alpha。
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化使得起始项为 1。
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 通过相邻比值得到每步 alpha。

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)  # 限制范围避免过小或超过 1。

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.  # 保留原注释：说明取平方根的原因。
    alphas = np.sqrt(alphas)  # 对 alpha 取平方根与参考实现保持一致。
    return alphas  # 返回余弦调度结果。


def get_distance(pos, edge_index):  # 根据边索引计算节点间距离。
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)  # 取差向量并计算 L2 范数。


def to_torch_const(x):  # 将 NumPy 数组转换为常量形式的可注册参数。
    x = torch.from_numpy(x).float()  # 转换为浮点张量。
    x = nn.Parameter(x, requires_grad=False)  # 包装为不可训练参数便于保存到模块。
    return x  # 返回常量参数。


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):  # 根据模式对坐标进行中心化。
    if mode == 'none':  # 不进行中心化。
        offset = 0.  # 偏移设置为 0。
        pass  # 保留位置，确保结构一致。
    elif mode == 'protein':  # 以蛋白几何中心进行平移。
        offset = scatter_mean(protein_pos, batch_protein, dim=0)  # 按批次计算蛋白坐标的均值。
        protein_pos = protein_pos - offset[batch_protein]  # 蛋白坐标减去偏移。
        ligand_pos = ligand_pos - offset[batch_ligand]  # 配体坐标同步减去偏移。
    else:  # 不支持的模式。
        raise NotImplementedError  # 抛出异常提示。
    return protein_pos, ligand_pos, offset  # 返回中心化后的坐标与偏移量。


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):  # 将分类索引转换为 log one-hot 表示。
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'  # 确保索引未越界。
    x_onehot = F.one_hot(x, num_classes)  # 生成 one-hot 向量。
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))  # 保留原注释：可选维度重排。
    # x_onehot = x_onehot.permute(permute_order)  # 保留原注释：示例重排操作。
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))  # 对 one-hot 取 log，避免对数零。
    return log_x  # 返回 log one-hot 表示。


def log_onehot_to_index(log_x):  # 将 log one-hot 表示转换回类别索引。
    return log_x.argmax(1)  # 沿类别维度取最大值所在索引。


def categorical_kl(log_prob1, log_prob2):  # 计算两个分类分布的 KL 散度。
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)  # 使用 log 概率公式计算 KL。
    return kl  # 返回 KL 值。


def log_categorical(log_x_start, log_prob):  # 计算分类对数似然。
    return (log_x_start.exp() * log_prob).sum(dim=1)  # 对应 one-hot 权重乘 log 概率累加。


def normal_kl(mean1, logvar1, mean2, logvar2):  # 计算两个正态分布之间的 KL 散度。
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))  # 按闭式公式计算 KL。
    return kl.sum(-1)  # 对最后一维求和得到标量。


def log_normal(values, means, log_scales):  # 计算正态分布的对数概率密度。
    var = torch.exp(log_scales * 2)  # 根据 log σ 还原方差。
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))  # 根据正态分布公式计算 logpdf。
    return log_prob.sum(-1)  # 对最后一维求和得到标量。


def log_sample_categorical(logits):  # 在 logit 空间对分类分布进行采样。
    uniform = torch.rand_like(logits)  # 采样均匀噪声以构造 Gumbel 噪声。
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)  # 根据公式生成 Gumbel 噪声。
    sample_index = (gumbel_noise + logits).argmax(dim=-1)  # 将噪声加到 logit 后取最大值。
    # sample_onehot = F.one_hot(sample, self.num_classes)  # 保留注释：生成 one-hot 示例。
    # log_sample = index_to_log_onehot(sample, self.num_classes)  # 保留注释：转换为 log one-hot。
    return sample_index  # 返回采样结果索引。


def log_1_min_a(a):  # 计算 log(1 - exp(a))，避免数值不稳定。
    return np.log(1 - np.exp(a) + 1e-40)  # 添加微小常数避免对数负数。


def log_add_exp(a, b):  # 计算 log(exp(a) + exp(b))，使用稳定公式。
    maximum = torch.max(a, b)  # 取较大值作为参考，避免指数溢出。
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))  # 返回稳定求和值。


def ensure_log_ligand(value, num_classes, mode='auto'):  # 根据指定模式将输入转换为 log 概率。
    if torch.is_floating_point(value):  # 若输入为浮点张量。
        if mode == 'log_prob':  # 已经是 log 概率。
            return value  # 直接返回。
        if mode == 'logits':  # 输入为未归一化 logits。
            return F.log_softmax(value, dim=-1)  # 转换为 log 概率。
        if mode == 'prob':  # 输入为概率分布。
            return torch.log(value.clamp(min=1e-30))  # 对概率取对数。
        if mode == 'auto':  # 自动模式尝试推断类型。
            if value.dim() >= 2 and torch.allclose(value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)), atol=1e-4):  # 若各行近似归一化。
                return torch.log(value.clamp(min=1e-30))  # 视为概率。
            return F.log_softmax(value, dim=-1)  # 否则视为 logits。
        raise ValueError(f'Unsupported ligand_v_input mode: {mode}')  # 未知模式时抛出异常。
    one_hot = F.one_hot(value.long(), num_classes=num_classes).float()  # 将离散索引转换为 one-hot。
    return torch.log(one_hot.clamp(min=1e-30))  # 返回 log one-hot。


def clamp_by_norm(tensor, max_norm, clip_strategy='norm'):  # 按指定策略裁剪张量，支持多种裁剪方法。
    if max_norm is None or max_norm <= 0:  # 无需裁剪时直接返回。
        return tensor
    
    if clip_strategy == 'norm':  # 基于范数的裁剪（原有方法）。
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        return tensor * scale  # 返回裁剪后的张量。
    
    elif clip_strategy == 'value':  # 基于值的裁剪（逐元素裁剪）。
        return torch.clamp(tensor, min=-max_norm, max=max_norm)  # 将值限制在 [-max_norm, max_norm] 范围内。
    
    elif clip_strategy == 'adaptive':  # 自适应裁剪（结合范数和值）。
        # 先进行范数裁剪
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        tensor_clipped = tensor * scale  # 范数裁剪后的张量。
        # 再进行值裁剪
        value_clip = max_norm * 0.5  # 值裁剪阈值设为范数阈值的一半。
        tensor_clipped = torch.clamp(tensor_clipped, min=-value_clip, max=value_clip)  # 值裁剪。
        return tensor_clipped  # 返回双重裁剪后的张量。
    
    else:  # 未知策略，回退到范数裁剪。
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        return tensor * scale  # 返回裁剪后的张量。


def adaptive_step_size(current_error, base_step_size, min_step=0.1, max_step=2.0):  # 根据当前误差自适应调整步长。
    """
    根据当前误差自适应调整步长。
    
    Args:
        current_error: 当前误差值（可以是梯度范数、位置误差等）。
        base_step_size: 基础步长。
        min_step: 最小步长限制。
        max_step: 最大步长限制。
    
    Returns:
        float: 调整后的步长。
    """
    if current_error is None or current_error <= 0:  # 若误差无效或为 0。
        return base_step_size  # 返回基础步长。
    # 误差小则增大步长，误差大则减小步长。
    factor = min(max(0.5 / (current_error + 1e-8), 0.5), 2.0)  # 限制因子在 [0.5, 2.0] 范围内。
    adjusted_step = base_step_size * factor  # 计算调整后的步长。
    return min(max(adjusted_step, min_step), max_step)  # 限制在 [min_step, max_step] 范围内。


def cal_kl_gradient(log_target, log_current):  # 计算将当前分布朝目标分布移动的梯度方向。
    target_prob = log_target.exp()  # 将目标 log 概率转换为概率。
    current_prob = log_current.exp()  # 将当前 log 概率转换为概率。
    grad = target_prob - current_prob  # 使用概率差作为梯度方向。
    return grad  # 返回梯度张量。


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):  # 定义正弦位置编码模块。
    def __init__(self, dim):  # 初始化函数，指定维度。
        super().__init__()  # 调用父类构造函数。
        self.dim = dim  # 保存嵌入维度。

    def forward(self, x):  # 前向传播，根据输入生成时间嵌入。
        device = x.device  # 获取输入张量所在设备。
        half_dim = self.dim // 2  # 计算一半维度用于正弦/余弦配对。
        emb = np.log(10000) / (half_dim - 1)  # 计算指数步长。
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成频率序列。
        emb = x[:, None] * emb[None, :]  # 计算时间与频率的乘积。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接正弦与余弦成完整编码。
        return emb  # 返回位置嵌入。


# Model
class ScorePosNet3D(nn.Module):  # 定义三维位置-类别扩散模型。

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):  # 初始化模型结构。
        super().__init__()  # 调用父类构造函数。
        self.config = config  # 保存配置对象。

        # variance schedule  # 保留原注释：说明下方配置与方差调度相关。
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']  # 指示模型输出类型（噪声或均值）。
        self.loss_v_weight = config.loss_v_weight  # 设置原子类型损失权重。
        self.ligand_v_input = getattr(config, 'ligand_v_input', 'onehot')  # 获取配体类型输入模式。
        self.use_grad_fusion = getattr(config, 'use_grad_fusion', False)  # 是否启用梯度融合策略。
        self.grad_fusion_lambda = getattr(config, 'grad_fusion_lambda', 0.5)  # 梯度融合系数或调度策略。
        self.dynamic_pos_step_clip = getattr(config, 'dynamic_pos_step_clip', None)  # 动态位置梯度裁剪阈值。
        self.dynamic_v_step_clip = getattr(config, 'dynamic_v_step_clip', None)  # 动态类别梯度裁剪阈值。
        self.loss_v2_weight = getattr(config, 'loss_v2_weight', 0.0)  # 额外类别损失权重。
        self.dynamic_large_step_defaults = getattr(config, 'dynamic_large_step', {})  # 大步采样的默认配置。
        self.dynamic_refine_defaults = getattr(config, 'dynamic_refine', {})  # 精炼采样的默认配置。
        # self.v_mode = config.v_mode  # 保留注释：遗留配置。
        # assert self.v_mode == 'categorical'  # 保留注释：对应断言。
        # self.v_net_type = getattr(config, 'v_net_type', 'mlp')  # 保留注释：额外网络类型。
        # self.bond_loss = getattr(config, 'bond_loss', False)  # 保留注释：键损失开关。
        # self.bond_net_type = getattr(config, 'bond_net_type', 'pre_att')  # 保留注释：键预测网络类型。
        # self.loss_bond_weight = getattr(config, 'loss_bond_weight', 0.)  # 保留注释：键损失权重。
        # self.loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.)  # 保留注释：非键损失权重。

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']  # 指定采样时间策略。
        # self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']  # 保留注释：位置损失形式。
        # print(f'Loss pos mode {self.loss_pos_type} applied!')  # 保留调试输出。
        # print(f'Loss bond net type: {self.bond_net_type} '  # 保留调试输出。
        #       f'bond weight: {self.loss_bond_weight} non bond weight: {self.loss_non_bond_weight}')

        if config.beta_schedule == 'cosine':  # 根据配置选择余弦调度。
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2  # 获取余弦调度并平方。
            # print('cosine pos alpha schedule applied!')  # 保留注释：调试输出。
            betas = 1. - alphas  # 将 alpha 转换为 beta。
        else:  # 使用通用调度生成函数。
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,  # 指定调度名称。
                beta_start=config.beta_start,  # 起始 beta。
                beta_end=config.beta_end,  # 结束 beta。
                num_diffusion_timesteps=config.num_diffusion_timesteps,  # 扩散步数。
            )
            alphas = 1. - betas  # 通过 beta 推导 alpha。
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算 alpha 的累计乘积。
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 构造前一项累计乘积，首项补 1。

        self.betas = to_torch_const(betas)  # 将 beta 转换为常量张量并注册。
        self.num_timesteps = self.betas.size(0)  # 记录扩散步数。
        self.alphas_cumprod = to_torch_const(alphas_cumprod)  # 注册 alpha 累乘作为常量。
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)  # 注册前置累乘。

        # calculations for diffusion q(x_t | x_{t-1}) and others  # 保留注释：说明以下常量用于前向扩散。
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))  # 注册 sqrt(ᾱ_t)。
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))  # 注册 sqrt(1-ᾱ_t)。
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))  # 注册 sqrt(1/ᾱ_t)。
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))  # 注册 sqrt(1/ᾱ_t -1)。

        # calculations for posterior q(x_{t-1} | x_t, x_0)  # 保留注释：以下参数用于后验计算。
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 计算后验方差。
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # 注册均值系数 (x0 部分)。
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))  # 注册均值系数 (xt 部分)。
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain  # 保留注释：解释对数截断。
        self.posterior_var = to_torch_const(posterior_variance)  # 注册后验方差。
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))  # 注册后验对数方差，首项重复第二项。

        # atom type diffusion schedule in log space  # 保留注释：以下针对离散类型扩散。
        if config.v_beta_schedule == 'cosine':  # 仅支持余弦调度。
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)  # 计算类型扩散的 alpha。
            # print('cosine v alpha schedule applied!')  # 保留注释：调试输出。
        else:  # 不支持其他调度。
            raise NotImplementedError  # 抛出未实现错误。
        log_alphas_v = np.log(alphas_v)  # 计算 log α。
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)  # 计算 log α 的累积和。
        self.log_alphas_v = to_torch_const(log_alphas_v)  # 注册 log α。
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))  # 注册 log(1-α)。
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)  # 注册累计 log α。
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))  # 注册 log(1-ᾱ)。

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))  # 记录不同时间步的损失历史。
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))  # 记录损失统计次数。

        # model definition  # 保留注释：开始定义模型结构。
        self.hidden_dim = config.hidden_dim  # 设置模型隐藏维度。
        self.num_classes = ligand_atom_feature_dim  # 设置配体类别数量。
        if self.config.node_indicator:  # 若启用节点类型指示器。
            emb_dim = self.hidden_dim - 1  # 预留一维用于节点类别标记。
        else:  # 未启用指示器。
            emb_dim = self.hidden_dim  # 嵌入维度等于隐藏维度。

        # atom embedding  # 保留注释：原子特征嵌入。
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)  # 映射蛋白原子特征到嵌入空间。

        # center pos  # 保留注释：中心化模式。
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']  # 保存中心化策略。

        # time embedding  # 保留注释：时间嵌入。
        self.time_emb_dim = config.time_emb_dim  # 时间嵌入维度。
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']  # 时间嵌入模式。
        if self.time_emb_dim > 0:  # 如果启用时间嵌入。
            if self.time_emb_mode == 'simple':  # 简单模式：直接拼接时间标量。
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)  # 线性层接受多一维时间标量。
            elif self.time_emb_mode == 'sin':  # 正弦模式：使用位置编码。
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),  # 生成正弦时间编码。
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),  # 提升维度。
                    nn.GELU(),  # 使用 GELU 激活。
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)  # 映射回时间嵌入维度。
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)  # 输入拼接时间嵌入。
            else:  # 未支持的时间模式。
                raise NotImplementedError  # 抛出异常提示。
        else:  # 不使用时间嵌入。
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)  # 仅基于原子特征映射。

        self.refine_net_type = config.model_type  # 记录精炼网络类型。
        self.refine_net = get_refine_net(self.refine_net_type, config)  # 根据配置实例化精炼网络。
        self.v_inference = nn.Sequential(  # 构建原子类型预测头。
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 第一层线性映射。
            ShiftedSoftplus(),  # 使用平移 Softplus 激活。
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),  # 输出每个类别的 logits。
        )

    def _prepare_ligand_inputs(self, init_ligand_v):  # 预处理配体类别输入，统一为张量形式。
        # 验证输入不为空
        if isinstance(init_ligand_v, torch.Tensor) and init_ligand_v.numel() == 0:
            raise ValueError(
                f"_prepare_ligand_inputs: Empty input detected. init_ligand_v.shape={init_ligand_v.shape}, "
                f"num_classes={self.num_classes}"
            )
        
        if torch.is_floating_point(init_ligand_v):  # 若输入已是浮点表示。
            if init_ligand_v.dim() != 2:  # 检查维度是否符合 [N, num_classes]。
                raise ValueError(
                    f'Expect ligand features of shape [N, num_classes] when providing floating tensors. '
                    f'Got shape {init_ligand_v.shape}, num_classes={self.num_classes}'
                )  # 抛出异常提示。
            # 验证特征维度
            if init_ligand_v.shape[1] != self.num_classes:
                raise ValueError(
                    f"_prepare_ligand_inputs: Feature dimension mismatch. init_ligand_v.shape={init_ligand_v.shape}, "
                    f"expected num_classes={self.num_classes}"
                )
            log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=self.ligand_v_input)  # 根据模式转为 log 概率。
            if self.ligand_v_input == 'log_prob':  # 如果配置期望 log 概率。
                ligand_emb_input = init_ligand_v  # 直接使用原始输入。
            elif self.ligand_v_input == 'logits':  # 如果配置期望 logits。
                ligand_emb_input = init_ligand_v  # 按原值返回。
            elif self.ligand_v_input == 'prob':  # 如果配置期望概率分布。
                ligand_emb_input = init_ligand_v  # 直接作为概率。
            else:  # 默认转换为概率。
                ligand_emb_input = log_ligand_v.exp()  # 对 log 概率取指数得到概率。
            return ligand_emb_input, log_ligand_v, init_ligand_v  # 返回嵌入输入、log 概率和原始张量。

        ligand_emb_input = F.one_hot(init_ligand_v.long(), self.num_classes).float()  # 将离散索引转换为 one-hot。
        log_ligand_v = torch.log(ligand_emb_input.clamp(min=1e-30))  # 转化为 log 概率表示。
        return ligand_emb_input, log_ligand_v, init_ligand_v  # 返回嵌入输入、log 概率和索引。

    def _format_forward_ligand_input(self, log_ligand_v):  # 根据配置将 log 概率转换为前向所需格式。
        mode = self.ligand_v_input  # 获取输入模式。
        if mode == 'log_prob':  # 若模型期望 log 概率。
            return log_ligand_v  # 直接返回。
        if mode == 'prob':  # 若期望概率分布。
            return log_ligand_v.exp()  # 对 log 概率取指数。
        if mode == 'logits':  # 若期望 logits。
            return log_ligand_v  # 此处同样返回 log 概率作为 logits。
        # default to categorical indices  # 保留注释：默认转换为类别索引。
        return log_ligand_v.argmax(dim=-1)  # 取最大值索引作为类别标签。

    def _compute_grad_lambda(self, timestep_value, mode=None, grad_norm_ratio=None):  # 根据时间步计算梯度融合系数，支持多种融合策略。
        if not self.use_grad_fusion:  # 若未开启梯度融合。
            return 0.0  # 返回零表示不融合。
        cfg = self.grad_fusion_lambda  # 读取融合配置。
        
        # 常数模式
        if isinstance(cfg, (float, int)):  # 常数模式。
            return float(cfg)  # 直接返回常数。
        
        # 字符串模式表示预设调度
        if isinstance(cfg, str):  # 字符串模式表示预设调度。
            alpha = float(timestep_value) / max(float(self.num_timesteps - 1), 1.0)  # 归一化时间步。
            if cfg == 'linear':  # 线性递减。
                return max(min(1.0 - alpha, 1.0), 0.0)  # 限制在 [0,1]。
            if cfg == 'time':  # 随时间递增。
                return max(min(alpha, 1.0), 0.0)  # 限制在 [0,1]。
            if cfg == 'auto':  # 自动模式。
                return 0.5 * (1.0 - alpha)  # 线性插值。
            return 0.5  # 默认返回 0.5。
        
        # 字典模式支持自定义调度
        if isinstance(cfg, dict):  # 字典模式支持自定义调度。
            strategy_mode = mode if mode is not None else cfg.get('mode', 'linear')  # 获取融合策略模式。
            start = float(cfg.get('start', 0.8))  # 起始值，默认为 0.8。
            end = float(cfg.get('end', 0.2))  # 结束值，默认为 0.2。
            ratio = float(timestep_value) / max(float(self.num_timesteps - 1), 1.0)  # 归一化时间步。
            
            if strategy_mode == 'linear':  # 线性衰减策略。
                lambda_val = start * ratio + end * (1.0 - ratio)  # 线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'exponential':  # 指数衰减策略。
                if start > 0 and end > 0:  # 确保起始和结束值都大于 0。
                    decay_rate = (end / start) ** (1.0 - ratio)  # 计算衰减率。
                    lambda_val = start * decay_rate  # 指数衰减。
                else:  # 若起始或结束值为 0，回退到线性衰减。
                    lambda_val = start * ratio + end * (1.0 - ratio)  # 线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'adaptive':  # 基于梯度大小的自适应策略。
                base_lambda = start * ratio + end * (1.0 - ratio)  # 基础线性插值。
                if grad_norm_ratio is not None and grad_norm_ratio > 0:  # 若提供了梯度范数比率。
                    # 当全局梯度范数较大时，增加其权重；否则减少其权重。
                    adaptive_factor = min(max(1.0 / (grad_norm_ratio + 1e-8), 0.5), 2.0)  # 自适应因子限制在 [0.5, 2.0]。
                    lambda_val = base_lambda * adaptive_factor  # 应用自适应因子。
                else:  # 若未提供梯度信息，回退到线性策略。
                    lambda_val = base_lambda  # 使用基础线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'time':  # 线性升权。
                lambda_val = start + (end - start) * ratio  # 插值计算。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'quadratic':  # 二次衰减策略（-x^power 模式）。
                # lambda = start - (start - end) * ratio^power
                # 前期（ratio接近0）变化慢，主要依赖全局梯度
                # 后期（ratio接近1）变化快，快速过渡到自由发展
                power = float(cfg.get('power', 2.0))  # 衰减指数，默认 2.0（二次）
                lambda_val = start - (start - end) * (ratio ** power)  # 幂次衰减。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            else:  # 未知策略模式，使用默认值。
                return 0.5  # 默认返回 0.5。
        
        return 0.5  # 默认返回 0.5。


    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):  # 前向传播，输出配体位置与类别预测。

        batch_size = batch_protein.max().item() + 1  # 计算批次数量。
        ligand_emb_input, _, _ = self._prepare_ligand_inputs(init_ligand_v)  # 准备配体输入嵌入与 log 概率。
        # time embedding  # 保留注释：以下处理时间嵌入。
        if self.time_emb_dim > 0:  # 若启用时间嵌入。
            if self.time_emb_mode == 'simple':  # 简单模式。
                input_ligand_feat = torch.cat([
                    ligand_emb_input,  # 拼接配体初始特征。
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)  # 添加归一化时间步。
                ], -1)  # 沿特征维度拼接。
            elif self.time_emb_mode == 'sin':  # 正弦位置编码模式。
                time_feat = self.time_emb(time_step)  # 计算时间嵌入向量。
                input_ligand_feat = torch.cat([ligand_emb_input, time_feat], -1)  # 拼接原始特征与时间嵌入。
            else:  # 未实现的时间模式。
                raise NotImplementedError  # 抛出异常提示。
        else:  # 未启用时间嵌入。
            input_ligand_feat = ligand_emb_input  # 直接使用初始特征。

        h_protein = self.protein_atom_emb(protein_v)  # 将蛋白原子特征映射到隐藏空间。
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # 将配体特征映射到隐藏空间。

        if self.config.node_indicator:  # 若使用节点类型指示器。
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)  # 为蛋白节点拼接 0 指示位。
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)  # 为配体节点拼接 1 指示位。

        h_all, pos_all, batch_all, mask_ligand = compose_context(  # 拼接蛋白与配体节点上下文。
            h_protein=h_protein,  # 传入蛋白隐藏特征。
            h_ligand=init_ligand_h,  # 传入配体隐藏特征。
            pos_protein=protein_pos,  # 传入蛋白坐标。
            pos_ligand=init_ligand_pos,  # 传入配体坐标。
            batch_protein=batch_protein,  # 蛋白批次索引。
            batch_ligand=batch_ligand,  # 配体批次索引。
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)  # 调用精炼网络更新几何与特征。
        final_pos, final_h = outputs['x'], outputs['h']  # 提取全局位置和隐藏特征结果。
        
        # 验证 refine_net 输出
        if final_pos.numel() == 0 or final_h.numel() == 0:
            raise ValueError(
                f"forward: refine_net returned empty outputs. final_pos.shape={final_pos.shape}, "
                f"final_h.shape={final_h.shape}, h_all.shape={h_all.shape}, pos_all.shape={pos_all.shape}, "
                f"mask_ligand.sum()={mask_ligand.sum().item()}, batch_all.shape={batch_all.shape}"
            )
        
        # 验证 mask_ligand 的有效性
        if mask_ligand.sum().item() == 0:
            raise ValueError(
                f"forward: mask_ligand contains no True values. mask_ligand.shape={mask_ligand.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}, batch_ligand.numel()={batch_ligand.numel()}"
            )
        
        if final_pos.shape[0] != mask_ligand.shape[0] or final_h.shape[0] != mask_ligand.shape[0]:
            raise ValueError(
                f"forward: refine_net output shape mismatch. final_pos.shape={final_pos.shape}, "
                f"final_h.shape={final_h.shape}, mask_ligand.shape={mask_ligand.shape}"
            )
        
        final_ligand_pos = final_pos[mask_ligand]  # 选择配体对应的结果。
        final_ligand_h = final_h[mask_ligand]
        
        # 验证索引后的结果
        if final_ligand_pos.numel() == 0 or final_ligand_h.numel() == 0:
            raise ValueError(
                f"forward: Empty result after mask_ligand indexing. final_ligand_pos.shape={final_ligand_pos.shape}, "
                f"final_ligand_h.shape={final_ligand_h.shape}, mask_ligand.sum()={mask_ligand.sum().item()}, "
                f"batch_ligand.shape={batch_ligand.shape}, expected ligand atoms={batch_ligand.numel()}"
            )
        
        if final_ligand_pos.shape[0] != batch_ligand.shape[0]:
            raise ValueError(
                f"forward: Ligand position count mismatch. final_ligand_pos.shape={final_ligand_pos.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}"
            )
        
        final_ligand_v = self.v_inference(final_ligand_h)  # 通过分类头预测配体原子类型 logits。

        preds = {  # 构造输出字典。
            'pred_ligand_pos': final_ligand_pos,  # 配体位置预测。
            'pred_ligand_v': final_ligand_v,  # 配体类别预测。
            'final_h': final_h,  # 所有节点的最终隐藏特征。
            'final_ligand_h': final_ligand_h  # 配体节点的最终隐藏特征。
        }
        if return_all:  # 若需要返回所有层的中间结果。
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']  # 从精炼网络获取全层列表。
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]  # 提取每层的配体位置。
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]  # 对每层配体特征进行推断。
            preds.update({  # 将分层结果添加到输出。
                'layer_pred_ligand_pos': final_all_ligand_pos,  # 每层配体位置。
                'layer_pred_ligand_v': final_all_ligand_v  # 每层配体类别 logits。
            })
        return preds  # 返回预测结果。

    # atom type diffusion process  # 保留注释：以下函数与离散类型扩散相关。
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):  # 计算一步前向扩散的 log 概率。
        # q(vt | vt-1)  # 保留注释：说明公式来源。
        log_alpha_t = extract(self.log_alphas_v, t, batch)  # 提取对应时间步的 log α。
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)  # 提取 log(1-α)。

        # alpha_t * vt + (1 - alpha_t) 1 / K  # 保留注释：公式描述。
        log_probs = log_add_exp(  # 使用 log-sum-exp 合并两项概率。
            log_vt_1 + log_alpha_t,  # 前一时刻概率乘 α。
            log_1_min_alpha_t - np.log(self.num_classes)  # 均匀噪声项 (1-α)/K。
        )
        return log_probs  # 返回当前时刻的 log 概率。

    def q_v_pred(self, log_v0, t, batch):  # 计算 q(v_t | v_0) 的 log 概率。
        # compute q(vt | v0)  # 保留注释：说明操作。
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)  # 提取 log ᾱ。
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)  # 提取 log(1-ᾱ)。

        log_probs = log_add_exp(  # 合并 v0 贡献与均匀噪声。
            log_v0 + log_cumprod_alpha_t,  # v0 对应的缩放项。
            log_1_min_cumprod_alpha - np.log(self.num_classes)  # 均匀噪声项。
        )
        return log_probs  # 返回 log 概率。

    def q_v_sample(self, log_v0, t, batch):  # 从 q(v_t | v_0) 中采样离散原子类型。
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)  # 计算条件分布。
        sample_index = log_sample_categorical(log_qvt_v0)  # 采样类别索引。
        log_sample = index_to_log_onehot(sample_index, self.num_classes)  # 转换为 log one-hot。
        return sample_index, log_sample  # 返回索引与 log one-hot。

    def q_v_pred_with_noise(self, log_v0, t, batch):  # 计算带噪声的 q(v_t | v_0) 的 log 概率。
        # compute q(vt | v0) with gumbel noise  # 保留注释：说明操作。
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)  # 提取 log ᾱ。
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)  # 提取 log(1-ᾱ)。
        uniform = torch.rand_like(log_v0)  # 生成均匀随机数。
        gumbel_noise = -0.5 * torch.log(-torch.log(uniform + 1e-30) + 1e-30)  # 生成 Gumbel 噪声。
        # equation (3) with noise  # 保留注释：公式说明。
        log_probs = log_add_exp(  # 合并 v0 贡献与带噪声的均匀项。
            log_v0 + log_cumprod_alpha_t,  # v0 对应的缩放项。
            log_1_min_cumprod_alpha + gumbel_noise - np.log(self.num_classes)  # 带噪声的均匀项。
        )
        log_probs = F.log_softmax(log_probs, dim=-1)  # 归一化为 log 概率。
        return log_probs  # 返回 log 概率。

    def q_v_sample_with_noise(self, log_v0, t, batch):  # 从带噪声的 q(v_t | v_0) 中采样。
        # log_v0 + noise -> log_qvt_v0  # 保留注释：说明操作。
        # sample_index == new categories  # 保留注释：采样结果。
        # log_sample == noised one hot categories  # 保留注释：log one-hot 表示。
        log_qvt_v0 = self.q_v_pred_with_noise(log_v0, t, batch)  # 计算带噪声的条件分布。
        # log(vt|v0) = log(a_t*v_0 + (1-a_t)*noise)  # 保留注释：公式说明。
        sample_index = log_qvt_v0.argmax(dim=-1)  # 取最大值索引作为采样结果。
        return sample_index, log_qvt_v0  # 返回索引与 log 概率。

    # atom type generative process  # 保留注释：以下处理离散生成过程。
    def q_v_posterior(self, log_v0, log_vt, t, batch):  # 计算离散后验 q(v_{t-1} | v_t, v_0)。
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)  # 保留注释：公式说明。
        t_minus_1 = t - 1  # 计算 t-1。
        # Remove negative values, will not be used anyway for final decoder  # 保留注释：说明下行原因。
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # 将负值裁剪为 0。
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)  # 计算 q(v_{t-1} | v_0)。
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)  # 累加联合项。
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)  # 归一化得到后验。
        return log_vt1_given_vt_v0  # 返回 log 概率。

    def q_v_posterior_with_noise(self, log_v0, log_vt, t, batch):  # 计算带噪声的离散后验 q(v_{t-1} | v_t, v_0)。
        # log_v0 = log_ligand_v_recon; log_vt = log_ligand_v  # 保留注释：参数说明。
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)  # 保留注释：公式说明。
        t_minus_1 = t - 1  # 计算 t-1。
        # Remove negative values, will not be used anyway for final decoder  # 保留注释：说明下行原因。
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # 将负值裁剪为 0。
        # q(v_t-1|v_0)  # 保留注释：计算前一步分布。
        log_qvt1_v0 = self.q_v_pred_with_noise(log_v0, t_minus_1, batch)  # 使用带噪声的预测。
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)  # 累加联合项。
        # q(vt-1 | v0)* q(vt | vt-1, v0) / q(vt | v0)  # 保留注释：公式说明。
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)  # 归一化得到后验。
        return log_vt1_given_vt_v0  # 返回 log 概率。

    def kl_v_prior(self, log_x_start, batch):  # 计算离散变量对先验的 KL。
        num_graphs = batch.max().item() + 1  # 获取批次数。
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)  # 计算最终时刻分布。
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))  # 构造均匀先验 log 概率。
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)  # 计算 KL 散度。
        kl_prior = scatter_mean(kl_prior, batch, dim=0)  # 按批次平均。
        return kl_prior  # 返回 KL 值。

    def _predict_x0_from_eps(self, xt, eps, t, batch):  # 根据噪声估计还原 x0。
        # 验证输入有效性
        if xt.numel() == 0 or eps.numel() == 0 or batch.numel() == 0:
            raise ValueError(
                f"_predict_x0_from_eps: Empty input detected. xt.numel()={xt.numel()}, "
                f"eps.numel()={eps.numel()}, batch.numel()={batch.numel()}. "
                f"This should not happen in normal operation."
            )
        
        # 验证维度一致性
        if xt.shape[0] != eps.shape[0] or xt.shape[0] != batch.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Dimension mismatch. xt.shape={xt.shape}, "
                f"eps.shape={eps.shape}, batch.shape={batch.shape}"
            )
        
        # 提取系数并计算 x0
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, batch)  # 提取系数
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, batch)  # 提取系数
        
        # 验证提取的系数形状
        if sqrt_recip_alphas_cumprod_t.shape[0] != xt.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Coefficient shape mismatch. "
                f"sqrt_recip_alphas_cumprod_t.shape={sqrt_recip_alphas_cumprod_t.shape}, "
                f"xt.shape={xt.shape}, t={t}, batch.shape={batch.shape}"
            )
        
        pos0_from_e = sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * eps  # 按扩散公式反推 x0。
        
        # 验证输出不为空
        if pos0_from_e.numel() == 0:
            raise ValueError(
                f"_predict_x0_from_eps: Returned empty tensor. xt.shape={xt.shape}, "
                f"eps.shape={eps.shape}, t={t}, batch.shape={batch.shape}, "
                f"sqrt_recip_alphas_cumprod_t.shape={sqrt_recip_alphas_cumprod_t.shape}"
            )
        
        # 验证输出形状
        if pos0_from_e.shape[0] != xt.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Output shape mismatch. pos0_from_e.shape={pos0_from_e.shape}, "
                f"xt.shape={xt.shape}"
            )
        
        return pos0_from_e  # 返回估计的原始坐标。

    def q_pos_posterior(self, x0, xt, t, batch):  # 计算连续变量后验均值。
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)  # 保留注释：说明用途。
        # 检查输入是否为空
        if batch.numel() == 0 or x0.numel() == 0 or xt.numel() == 0:  # 如果批次或输入为空。
            # 如果输入为空，应该抛出错误而不是返回空张量（这会掩盖真正的问题）
            raise ValueError(
                f"q_pos_posterior: Empty input detected. batch.numel()={batch.numel()}, "
                f"x0.numel()={x0.numel()}, xt.numel()={xt.numel()}. "
                f"This should not happen in normal operation. Please check batch_ligand and ligand_pos initialization."
            )
        
        # 确保 x0 和 xt 大小一致
        if x0.shape[0] != xt.shape[0]:  # 如果大小不一致。
            raise ValueError(f"x0 and xt must have the same size in dimension 0, got {x0.shape[0]} and {xt.shape[0]}")
        
        # 确保 batch 大小与 x0/xt 一致
        if batch.shape[0] != x0.shape[0]:  # 如果批次大小不一致。
            raise ValueError(
                f"batch must have the same size as x0/xt in dimension 0, got batch.shape[0]={batch.shape[0]} "
                f"and x0.shape[0]={x0.shape[0]}. batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        
        # 验证 t 的格式和范围
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                raise ValueError(f"q_pos_posterior: t is empty tensor")
            num_graphs = t.shape[0]
            if batch.max().item() >= num_graphs or batch.min().item() < 0:
                raise IndexError(
                    f"q_pos_posterior: batch indices out of range. batch range=[{batch.min().item()}, {batch.max().item()}], "
                    f"num_graphs={num_graphs}, t shape={t.shape}"
                )
        
        c0_coef = extract(self.posterior_mean_c0_coef, t, batch)  # 提取 x0 系数。
        ct_coef = extract(self.posterior_mean_ct_coef, t, batch)  # 提取 xt 系数。
        
        # 确保提取的系数大小正确
        if c0_coef.shape[0] != x0.shape[0]:  # 如果系数大小不匹配。
            raise ValueError(
                f"c0_coef size {c0_coef.shape[0]} does not match x0 size {x0.shape[0]}. "
                f"t shape={t.shape if isinstance(t, torch.Tensor) else type(t)}, "
                f"batch shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        if ct_coef.shape[0] != xt.shape[0]:  # 如果系数大小不匹配。
            raise ValueError(
                f"ct_coef size {ct_coef.shape[0]} does not match xt size {xt.shape[0]}. "
                f"t shape={t.shape if isinstance(t, torch.Tensor) else type(t)}, "
                f"batch shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        
        pos_model_mean = c0_coef * x0 + ct_coef * xt  # 根据闭式公式计算均值。
        return pos_model_mean  # 返回后验均值。

    def kl_pos_prior(self, pos0, batch):  # 计算连续变量对先验的 KL。
        num_graphs = batch.max().item() + 1  # 获取批次数。
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)  # 提取最终 ᾱ。
        pos_model_mean = a_pos.sqrt() * pos0  # 计算均值项。
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())  # 计算方差的 log。
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)  # 与标准正态比较。
        kl_prior = scatter_mean(kl_prior, batch, dim=0)  # 按批次平均。
        return kl_prior  # 返回 KL。

    def sample_time(self, num_graphs, device, method):  # 采样扩散时间步。
        if method == 'importance':  # 重要性采样模式。
            if not (self.Lt_count > 10).all():  # 若统计不足。
                return self.sample_time(num_graphs, device, method='symmetric')  # 回退到对称采样。

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001  # 对损失历史开方平滑。
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.  # 保留注释：避免第 0 项不稳定。
            pt_all = Lt_sqrt / Lt_sqrt.sum()  # 归一化得到时间分布。

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)  # 按分布抽样时间步。
            pt = pt_all.gather(dim=0, index=time_step)  # 记录对应概率。
            return time_step, pt  # 返回抽样结果与概率。

        elif method == 'symmetric':  # 对称采样模式。
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)  # 随机采样前半部分。
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]  # 构造成对时间步并截断。
            pt = torch.ones_like(time_step).float() / self.num_timesteps  # 使用均匀概率。
            return time_step, pt  # 返回采样结果与概率。

        else:  # 未支持的采样模式。
            raise ValueError  # 抛出异常提示。

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):  # 计算位置通道的单步变分损失。
        # fixed pos variance  # 保留注释：使用固定的后验方差。
        pos_log_variance = extract(self.posterior_logvar, t, batch)  # 提取对应时间步的对数方差。
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)  # 计算真实后验均值。
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)  # 估计 KL 散度。
        kl_pos = kl_pos / np.log(2.)  # 换算为以 2 为底的对数。

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)  # 计算重建负对数似然。
        assert kl_pos.shape == decoder_nll_pos.shape  # 确认形状一致。
        mask = (t == 0).float()[batch]  # 构造掩码，t=0 时使用重建损失。
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)  # 按批次聚合损失。
        return loss_pos  # 返回位置损失。

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):  # 计算离散类型通道的单步损失。
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]  # 计算 KL 散度。
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0  # 计算重建负对数似然。
        assert kl_v.shape == decoder_nll_v.shape  # 确认形状一致。
        mask = (t == 0).float()[batch]  # 构造掩码选择使用哪种损失。
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)  # 聚合得到批次平均。
        return loss_v  # 返回类型损失。

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):  # 计算扩散训练损失。
        num_graphs = batch_protein.max().item() + 1  # 统计图数量。
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)  # 根据配置中心化坐标。

        # 1. sample noise levels  # 保留注释：步骤 1 采样噪声。
        if time_step is None:  # 若未指定时间步。
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)  # 按策略采样时间。
        else:  # 若外部指定时间步。
            pt = torch.ones_like(time_step).float() / self.num_timesteps  # 使用均匀概率。
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )  # 提取 ᾱ_t。

        # 2. perturb pos and v  # 保留注释：步骤 2 扰动位置与类型。
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)  # 将 ᾱ_t 展开到配体节点。
        pos_noise = torch.zeros_like(ligand_pos)  # 初始化位置噪声。
        pos_noise.normal_()  # 采样标准正态噪声。
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps  # 保留注释：扩散公式。
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std  # 生成扰动位置。
        # Vt = a * V0 + (1-a) / K  # 保留注释：类别扰动公式。
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)  # 将原始类别转为 log one-hot。
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)  # 采样离散扰动。

        # 3. forward-pass NN, feed perturbed pos and v, output noise  # 保留注释：步骤 3 前向网络。
        preds = self(
            protein_pos=protein_pos,  # 输入蛋白坐标。
            protein_v=protein_v,  # 输入蛋白原子特征。
            batch_protein=batch_protein,  # 输入蛋白批次索引。

            init_ligand_pos=ligand_pos_perturbed,  # 输入扰动后的配体坐标。
            init_ligand_v=ligand_v_perturbed,  # 输入扰动后的配体类别。
            batch_ligand=batch_ligand,  # 输入配体批次索引。
            time_step=time_step  # 输入时间步。
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']  # 提取网络预测。
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed  # 估计预测噪声。
        # atom position  # 保留注释：以下处理位置分支。
        if self.model_mean_type == 'noise':  # 模型预测噪声时。
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)  # 反推 x0。
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算均值。
        elif self.model_mean_type == 'C0':  # 模型预测 x0 时。
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 直接使用预测。
        else:  # 未知类型。
            raise ValueError  # 抛出异常。

        # atom pos loss  # 保留注释：位置损失计算。
        if self.model_mean_type == 'C0':  # 若预测的是 x0。
            target, pred = ligand_pos, pred_ligand_pos  # 使用位置与预测进行比较。
        elif self.model_mean_type == 'noise':  # 若预测的是噪声。
            target, pred = pos_noise, pred_pos_noise  # 使用噪声与预测比较。
        else:  # 未知类型。
            raise ValueError  # 抛出异常。
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)  # 计算 MSE 并按批次聚合。
        loss_pos = torch.mean(loss_pos)  # 对所有图取平均。

        # atom type loss  # 保留注释：类别损失计算。
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)  # 将 logits 转为 log 概率。
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)  # 计算模型后验。
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)  # 计算真实后验。
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)  # 计算 KL 损失。
        loss_v = torch.mean(kl_v)  # 对 KL 损失取平均。
        loss_v2 = None  # 预留额外损失项。
        if self.loss_v2_weight and self.loss_v2_weight > 0:  # 若启用额外交叉熵损失。
            loss_v2 = F.nll_loss(log_ligand_v_recon, ligand_v)  # 计算 NLL。
        loss = loss_pos + loss_v * self.loss_v_weight  # 组合位置与类型损失。
        if loss_v2 is not None:  # 若额外损失存在。
            loss = loss + self.loss_v2_weight * loss_v2  # 加权累加。

        return {  # 返回损失字典。
            'loss_pos': loss_pos,  # 位置损失。
            'loss_v': loss_v,  # 类型损失。
            'loss_v2': loss_v2 if loss_v2 is not None else torch.tensor(0., device=loss_pos.device),  # 额外损失或零张量。
            'loss': loss,  # 总损失。
            'x0': ligand_pos,  # 原始配体位置。
            'pred_ligand_pos': pred_ligand_pos,  # 预测位置。
            'pred_ligand_v': pred_ligand_v,  # 预测类别 logits。
            'pred_pos_noise': pred_pos_noise,  # 预测噪声。
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)  # 预测类别分布。
        }

    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):  # 估计对数似然所需的 KL 项。
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')  # 以蛋白中心进行坐标平移。
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()  # 确保时间步合法。
        if (time_step == self.num_timesteps).all():  # 当 t=T 时直接返回先验 KL。
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)  # 计算位置先验 KL。
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)  # 构造 log one-hot。
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)  # 计算离散先验 KL。
            return kl_pos_prior, kl_v_prior  # 返回先验项。

        # perturb pos and v  # 保留注释：以下对位置与类型进行扰动。
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )  # 提取 ᾱ_t。
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)  # 展开到配体节点。
        pos_noise = torch.zeros_like(ligand_pos)  # 初始化噪声。
        pos_noise.normal_()  # 采样标准正态噪声。
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps  # 保留注释：扩散公式。
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std  # 生成扰动位置。
        # Vt = a * V0 + (1-a) / K  # 保留注释：类别扰动公式。
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)  # 构造 log one-hot 类别。
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)  # 采样扰动类别。

        preds = self(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。

            init_ligand_pos=ligand_pos_perturbed,  # 扰动后的配体坐标。
            init_ligand_v=ligand_v_perturbed,  # 扰动后的配体类别。
            batch_ligand=batch_ligand,  # 配体批次。
            time_step=time_step  # 时间步。
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']  # 提取预测输出。
        if self.model_mean_type == 'C0':  # 仅支持模型直接预测 x0 的情况。
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算后验均值。
        else:  # 其他模式未实现。
            raise ValueError  # 抛出异常。

        # atom type  # 保留注释：以下处理离散变量。
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)  # 转换为 log 概率。
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)  # 计算模型后验。
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)  # 计算真实后验。

        # t = [T-1, ... , 0]  # 保留注释：遍历时间序列。
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算位置 KL。
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)  # 计算离散 KL。
        return kl_pos, kl_v  # 返回 KL 项。

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):  # 获取最终隐藏表示。
        preds = self(
            protein_pos=protein_pos,  # 输入蛋白坐标。
            protein_v=protein_v,  # 输入蛋白特征。
            batch_protein=batch_protein,  # 输入蛋白批次索引。

            init_ligand_pos=ligand_pos,  # 输入配体坐标。
            init_ligand_v=ligand_v,  # 输入配体类型。
            batch_ligand=batch_ligand,  # 输入配体批次索引。
            fix_x=True  # 固定坐标，仅提取特征。
        )
        return preds  # 返回包含隐藏表示的字典。

    def _dynamic_diffusion(self, protein_pos, protein_v, batch_protein,
                           ligand_pos, log_ligand_v, batch_ligand,
                           time_indices, step_size,
                           add_noise=0.0, pos_clip=None, v_clip=None,
                           record_traj=True, pos_only=False, 
                           enable_normalization=True, enable_monitoring=False,
                           clip_strategy='norm', use_adaptive_step=False,
                           use_with_noise=False, use_time_scale=True):  # 执行动态扩散迭代（用于采样或精炼），增强数值稳定性和监控。use_with_noise: 是否在类别更新中使用带噪声的方法。use_time_scale: 是否启用时间步对步长的乘算。
        num_graphs = batch_protein.max().item() + 1  # 获取批次数。
        pos_traj, log_v_traj = [], []  # 准备轨迹存储列表。
        
        # 初始化性能监控指标
        metrics = {'grad_norms': [], 'lambda_values': [], 'local_pos_grad_norms': [], 
                   'global_pos_grad_norms': [], 'combined_pos_grad_norms': [],
                   'local_v_grad_norms': [], 'global_v_grad_norms': [], 
                   'combined_v_grad_norms': []} if enable_monitoring else {}  # 监控指标字典。

        for i, t_scalar in enumerate(time_indices):  # 遍历时间调度列表（使用 enumerate 以便计算时间跨度）。
            t = torch.full(size=(num_graphs,), fill_value=int(t_scalar), dtype=torch.long, device=protein_pos.device)  # 构造当前批次时间张量。
            
            # 动态计算时间跨度 n（用于跳步采样时的步长缩放）
            if i < len(time_indices) - 1:
                n = abs(t_scalar - time_indices[i + 1])
            else:
                n = abs(time_indices[i] - time_indices[i - 1]) if i > 0 else 1
            n = max(1, int(n))  # 确保 n >= 1
            
            # 计算时间尺度缩放因子（基于布朗运动位移性质：位移与 sqrt(时间) 成正比）
            # 根据配置决定是否启用时间缩放
            if use_time_scale:
                time_scale_factor = np.sqrt(n)
            else:
                time_scale_factor = 1.0  # 不启用时，缩放因子为 1（相当于不缩放）
            
            # 验证输入维度（在模型前向传播前）
            if ligand_pos.numel() == 0 or log_ligand_v.numel() == 0 or batch_ligand.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: Empty input detected at time step {t_scalar}. "
                    f"ligand_pos.numel()={ligand_pos.numel()}, log_ligand_v.numel()={log_ligand_v.numel()}, "
                    f"batch_ligand.numel()={batch_ligand.numel()}, num_graphs={num_graphs}"
                )
            
            if ligand_pos.shape[0] != batch_ligand.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Dimension mismatch at time step {t_scalar}. "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"batch_ligand.range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"num_graphs={num_graphs}"
                )
            
            # 验证批次索引范围
            if batch_ligand.max().item() >= num_graphs or batch_ligand.min().item() < 0:
                raise ValueError(
                    f"_dynamic_diffusion: Invalid batch_ligand indices at time step {t_scalar}. "
                    f"batch_ligand.range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"num_graphs={num_graphs}"
                )
            
            ligand_input = self._format_forward_ligand_input(log_ligand_v)  # 根据模式准备配体输入。
            
            # 验证配体输入格式
            if isinstance(ligand_input, torch.Tensor):
                if ligand_input.numel() == 0:
                    raise ValueError(
                        f"_dynamic_diffusion: Empty ligand_input after formatting at time step {t_scalar}. "
                        f"log_ligand_v.shape={log_ligand_v.shape}, ligand_input.shape={ligand_input.shape}"
                    )
                if ligand_input.shape[0] != batch_ligand.shape[0]:
                    raise ValueError(
                        f"_dynamic_diffusion: ligand_input shape mismatch at time step {t_scalar}. "
                        f"ligand_input.shape={ligand_input.shape}, batch_ligand.shape={batch_ligand.shape}"
                    )
            
            # 模型前向传播
            preds = self(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次索引。
                init_ligand_pos=ligand_pos,  # 当前配体坐标。
                init_ligand_v=ligand_input,  # 当前配体类别表示。
                batch_ligand=batch_ligand,  # 配体批次索引。
                time_step=t  # 当前时间步。
            )

            # 验证模型输出
            if 'pred_ligand_pos' not in preds:
                raise ValueError(
                    f"_dynamic_diffusion: Model output missing 'pred_ligand_pos' key at time step {t_scalar}. "
                    f"Available keys: {list(preds.keys())}"
                )
            
            pred_ligand_pos = preds['pred_ligand_pos']
            if pred_ligand_pos.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: Model output 'pred_ligand_pos' is empty at time step {t_scalar}. "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"protein_pos.shape={protein_pos.shape}, protein_v.shape={protein_v.shape}, "
                    f"num_graphs={num_graphs}. This indicates a problem with model forward pass or model weights."
                )
            
            if pred_ligand_pos.shape[0] != ligand_pos.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Model output shape mismatch at time step {t_scalar}. "
                    f"pred_ligand_pos.shape={pred_ligand_pos.shape}, ligand_pos.shape={ligand_pos.shape}, "
                    f"batch_ligand.shape={batch_ligand.shape}"
                )

            if self.model_mean_type == 'noise':  # 当模型预测噪声时。
                pred_pos_noise = pred_ligand_pos - ligand_pos  # 计算预测噪声。
                
                # 验证预测噪声的形状
                if pred_pos_noise.shape[0] != ligand_pos.shape[0]:
                    raise ValueError(
                        f"_dynamic_diffusion: Predicted noise shape mismatch at time step {t_scalar}. "
                        f"pred_pos_noise.shape={pred_pos_noise.shape}, ligand_pos.shape={ligand_pos.shape}"
                    )
                
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)  # 反推 x0。
            elif self.model_mean_type == 'C0':  # 当模型直接预测 x0 时。
                pos0_from_e = pred_ligand_pos  # 直接使用预测。
            else:  # 未知模式。
                raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")  # 抛出异常。
            
            # 验证 pos0_from_e 不为空
            if pos0_from_e.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: pos0_from_e is empty at time step {t_scalar}. "
                    f"model_mean_type={self.model_mean_type}, pred_ligand_pos.shape={pred_ligand_pos.shape}, "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}"
                )
            
            if pos0_from_e.shape[0] != ligand_pos.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: pos0_from_e shape mismatch at time step {t_scalar}. "
                    f"pos0_from_e.shape={pos0_from_e.shape}, ligand_pos.shape={ligand_pos.shape}"
                )

            # 在调用 q_pos_posterior 前验证批次索引的有效性（TargetDiff 原始代码的保护机制）
            if batch_ligand.max().item() >= num_graphs:
                raise ValueError(
                    f"batch_ligand contains invalid indices: max={batch_ligand.max().item()}, "
                    f"num_graphs={num_graphs}. This should not happen in normal operation. "
                    f"batch_ligand should contain values in [0, {num_graphs-1}]."
                )
            if batch_ligand.min().item() < 0:
                raise ValueError(
                    f"batch_ligand contains negative indices: min={batch_ligand.min().item()}. "
                    f"This should not happen in normal operation."
                )
            
            # 在调用 q_pos_posterior 前再次验证输入
            if pos0_from_e.shape[0] != ligand_pos.shape[0] or ligand_pos.shape[0] != batch_ligand.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Input dimension mismatch before q_pos_posterior at time step {t_scalar}. "
                    f"pos0_from_e.shape={pos0_from_e.shape}, ligand_pos.shape={ligand_pos.shape}, "
                    f"batch_ligand.shape={batch_ligand.shape}, num_graphs={num_graphs}"
                )
            
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)  # 计算局部均值。
            
            # 检查维度一致性（TargetDiff 原始代码中，这应该总是匹配的）
            if pos_model_mean.shape[0] != ligand_pos.shape[0]:
                if enable_monitoring:  # 如果启用监控，记录详细信息。
                    metrics['dimension_mismatches'] = metrics.get('dimension_mismatches', [])
                    metrics['dimension_mismatches'].append({
                        'pos_model_mean_shape': pos_model_mean.shape,
                        'ligand_pos_shape': ligand_pos.shape,
                        'batch_ligand_shape': batch_ligand.shape,
                        'time_step': t_scalar,
                        'num_graphs': num_graphs,
                        'batch_ligand_range': [batch_ligand.min().item(), batch_ligand.max().item()]
                    })
                # 直接抛出错误，不要尝试修复（这会破坏批次索引的一致性）
                raise ValueError(
                    f"Dimension mismatch after q_pos_posterior: pos_model_mean.shape={pos_model_mean.shape}, "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"num_graphs={num_graphs}, batch_ligand_range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}]. "
                    f"This should not happen in normal operation. Please check the extract function and batch_ligand values."
                )
            
            local_pos_grad = pos_model_mean - ligand_pos  # 局部梯度（后验均值与当前点差）。
            global_pos_grad = pos0_from_e - ligand_pos  # 全局梯度（预测 x0 与当前点差）。
            
            # 计算原始梯度范数（用于监控和自适应计算，在归一化之前）
            local_pos_grad_norm_val = torch.norm(local_pos_grad, dim=-1).mean().item()  # 局部梯度平均范数。
            global_pos_grad_norm_val = torch.norm(global_pos_grad, dim=-1).mean().item()  # 全局梯度平均范数。
            grad_norm_ratio = global_pos_grad_norm_val / (local_pos_grad_norm_val + 1e-8) if local_pos_grad_norm_val > 0 else 1.0  # 计算梯度范数比率。
            
            # 增强数值稳定性：在梯度融合前添加归一化（可选）
            if enable_normalization:  # 若启用归一化。
                local_pos_grad_norm = torch.norm(local_pos_grad, dim=-1, keepdim=True)  # 计算局部梯度范数。
                local_pos_grad = local_pos_grad / (local_pos_grad_norm + 1e-8)  # 归一化局部梯度。
                global_pos_grad_norm = torch.norm(global_pos_grad, dim=-1, keepdim=True)  # 计算全局梯度范数。
                global_pos_grad = global_pos_grad / (global_pos_grad_norm + 1e-8)  # 归一化全局梯度。
            
            # 根据梯度信息计算 lambda（支持自适应模式）
            lambda_val = self._compute_grad_lambda(int(t_scalar), grad_norm_ratio=grad_norm_ratio)  # 根据时间计算融合系数。
            
            if self.use_grad_fusion:  # 若启用梯度融合。
                combined_pos_grad = lambda_val * global_pos_grad + (1.0 - lambda_val) * local_pos_grad  # 融合全局与局部梯度。
            else:  # 未启用融合。
                combined_pos_grad = local_pos_grad  # 使用局部梯度。
            
            # 使用多策略梯度裁剪
            combined_pos_grad = clamp_by_norm(combined_pos_grad, pos_clip, clip_strategy=clip_strategy)  # 对梯度进行裁剪。
            
            # 自适应步长调整（可选）
            current_step_size = step_size  # 默认使用原始步长。
            if use_adaptive_step:  # 若启用自适应步长。
                combined_pos_grad_norm = torch.norm(combined_pos_grad, dim=-1).mean().item()  # 计算融合后梯度范数。
                current_step_size = adaptive_step_size(combined_pos_grad_norm, step_size)  # 自适应调整步长。
            
            # 应用时间尺度缩放因子到实际步长
            actual_step_size = current_step_size * time_scale_factor
            
            ligand_pos = ligand_pos + actual_step_size * combined_pos_grad  # 按缩放后的步长更新配体坐标。
            if add_noise and add_noise > 0:  # 如果需要添加额外噪声。
                ligand_pos = ligand_pos + add_noise * torch.randn_like(ligand_pos)  # 加入高斯扰动。

            if not pos_only:  # 若同时更新类别。
                log_ligand_v_recon = F.log_softmax(preds['pred_ligand_v'], dim=-1)  # 计算当前预测分布。
                # 根据配置选择使用标准方法或带噪声方法计算后验
                if use_with_noise:  # 使用带噪声的后验计算，提供更平滑的分布。
                    log_model_prob = self.q_v_posterior_with_noise(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算带噪声的局部后验。
                else:  # 使用标准方法。
                    log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算局部后验。
                local_v_grad = cal_kl_gradient(log_model_prob, log_ligand_v)  # 计算局部梯度。
                global_v_grad = cal_kl_gradient(log_ligand_v_recon, log_ligand_v)  # 计算全局梯度。
                
                # 计算原始类别梯度范数（用于监控，在归一化之前）
                local_v_grad_norm_val = torch.norm(local_v_grad, dim=-1).mean().item()  # 局部类别梯度平均范数。
                global_v_grad_norm_val = torch.norm(global_v_grad, dim=-1).mean().item()  # 全局类别梯度平均范数。
                
                # 增强数值稳定性：类别梯度归一化（可选）
                if enable_normalization:  # 若启用归一化。
                    local_v_grad_norm = torch.norm(local_v_grad, dim=-1, keepdim=True)  # 计算局部类别梯度范数。
                    local_v_grad = local_v_grad / (local_v_grad_norm + 1e-8)  # 归一化局部类别梯度。
                    global_v_grad_norm = torch.norm(global_v_grad, dim=-1, keepdim=True)  # 计算全局类别梯度范数。
                    global_v_grad = global_v_grad / (global_v_grad_norm + 1e-8)  # 归一化全局类别梯度。
                
                if self.use_grad_fusion:  # 梯度融合。
                    combined_v_grad = lambda_val * global_v_grad + (1.0 - lambda_val) * local_v_grad  # 融合类别梯度。
                else:  # 不融合。
                    combined_v_grad = local_v_grad  # 使用局部梯度。
                
                # 使用多策略梯度裁剪
                combined_v_grad = clamp_by_norm(combined_v_grad, v_clip, clip_strategy=clip_strategy)  # 裁剪梯度。
                
                # 自适应步长调整（类别梯度）
                if use_adaptive_step:  # 若启用自适应步长。
                    combined_v_grad_norm = torch.norm(combined_v_grad, dim=-1).mean().item()  # 计算融合后类别梯度范数。
                    current_step_size_v = adaptive_step_size(combined_v_grad_norm, step_size)  # 自适应调整步长。
                    actual_step_size_v = current_step_size_v * time_scale_factor  # 应用时间尺度缩放因子
                else:
                    actual_step_size_v = step_size * time_scale_factor  # 应用时间尺度缩放因子
                
                log_ligand_v = F.log_softmax(log_ligand_v + actual_step_size_v * combined_v_grad, dim=-1)  # 使用缩放后的步长更新 log 概率。
            
            # 性能监控：记录关键指标
            if enable_monitoring:  # 若启用监控。
                metrics['lambda_values'].append(lambda_val)  # 记录 lambda 值。
                metrics['local_pos_grad_norms'].append(local_pos_grad_norm_val)  # 记录局部位置梯度范数。
                metrics['global_pos_grad_norms'].append(global_pos_grad_norm_val)  # 记录全局位置梯度范数。
                combined_pos_grad_norm_val = torch.norm(combined_pos_grad, dim=-1).mean().item()  # 计算融合后梯度范数。
                metrics['combined_pos_grad_norms'].append(combined_pos_grad_norm_val)  # 记录融合后位置梯度范数。
                metrics['grad_norms'].append(combined_pos_grad_norm_val)  # 记录主要梯度范数。
                if not pos_only:  # 若更新了类别。
                    combined_v_grad_norm_val = torch.norm(combined_v_grad, dim=-1).mean().item()  # 融合后类别梯度范数。
                    metrics['local_v_grad_norms'].append(local_v_grad_norm_val)  # 记录局部类别梯度范数。
                    metrics['global_v_grad_norms'].append(global_v_grad_norm_val)  # 记录全局类别梯度范数。
                    metrics['combined_v_grad_norms'].append(combined_v_grad_norm_val)  # 记录融合后类别梯度范数。

            if record_traj:  # 如需记录轨迹。
                pos_traj.append(ligand_pos.detach().cpu())  # 保存当前坐标。
                log_v_traj.append(log_ligand_v.detach().cpu())  # 保存当前类别分布。

        # 返回最新状态、轨迹和监控指标
        if enable_monitoring:  # 若启用了监控。
            return ligand_pos, log_ligand_v, pos_traj, log_v_traj, metrics  # 返回最新状态与轨迹及监控指标。
        return ligand_pos, log_ligand_v, pos_traj, log_v_traj  # 返回最新状态与轨迹。

    def _build_lambda_schedule(self, start_t, end_t, coeff_a, coeff_b):  # 构建自适应时间调度列表。
        """构建基于 lambda 函数的时间调度序列。
        
        Args:
            start_t: 起始时间步
            end_t: 结束时间步
            coeff_a: lambda 调度参数 a
            coeff_b: lambda 调度参数 b
        
        Returns:
            list: 时间步索引列表
        """
        start_t = int(np.clip(start_t, 0, self.num_timesteps - 1))  # 将起点裁剪到合法范围。
        end_t = int(np.clip(end_t, 0, self.num_timesteps - 1))  # 将终点裁剪到合法范围。
        if start_t == end_t:  # 若起终点一致。
            return [start_t]  # 直接返回单个时间步。

        decreasing = start_t > end_t  # 判断是否递减。
        step_sign = -1 if decreasing else 1  # 根据方向决定步符号。
        t = start_t  # 初始化当前时间。
        indices = []  # 初始化时间列表。

        while True:  # 迭代构建调度。
            indices.append(t)  # 记录当前时间。
            if t == end_t:  # 达到终点时结束。
                break
            lambda_t = float(t) / float(self.num_timesteps)  # 归一化时间。
            lambda_t = max(min(lambda_t, 1.0), 0.0)  # 限制在 [0,1]。
            n = coeff_a * lambda_t + coeff_b  # 根据参数计算步长。
            step = max(1, int(round(n)))  # 至少前进 1 步。
            t_next = t + step_sign * step  # 计算下一时间。
            if decreasing and t_next < end_t:  # 避免越界。
                t_next = end_t
            if not decreasing and t_next > end_t:  # 避免越界。
                t_next = end_t
            if t_next == t:  # 若步长为零，强制前进一格。
                t_next = t + step_sign
            t = int(np.clip(t_next, 0, self.num_timesteps - 1))  # 更新当前时间并裁剪。

        return indices  # 返回构建好的时间序列。

    def _build_linear_schedule(self, start_t, end_t, coeff_a, coeff_b):  # 构建基于线性函数的时间调度列表。
        """构建基于线性函数 f = ax + b 的时间调度序列。
        
        使用线性函数计算跳步步长：step_size = a * progress + b
        其中 progress 是进度（从1到0），b 是步长下限，a+b 是步长上限。
        第一步的步长 = a + b（上限），最后一步的步长 = b（下限），中间线性过渡。
        
        Args:
            start_t: 起始时间步
            end_t: 结束时间步
            coeff_a: 线性函数系数 a（步长上限 - 步长下限）
            coeff_b: 线性函数系数 b（步长下限）
        
        Returns:
            list: 时间步索引列表
        """
        start_t = int(np.clip(start_t, 0, self.num_timesteps - 1))  # 将起点裁剪到合法范围。
        end_t = int(np.clip(end_t, 0, self.num_timesteps - 1))  # 将终点裁剪到合法范围。
        if start_t == end_t:  # 若起终点一致。
            return [start_t]  # 直接返回单个时间步。

        decreasing = start_t > end_t  # 判断是否递减。
        step_sign = -1 if decreasing else 1  # 根据方向决定步符号。
        t = start_t  # 初始化当前时间。
        indices = []  # 初始化时间列表。
        initial_range = abs(start_t - end_t)  # 初始时间范围

        while True:  # 迭代构建调度。
            indices.append(t)  # 记录当前时间。
            if t == end_t:  # 达到终点时结束。
                break
            
            # 计算进度：剩余时间范围 / 初始时间范围
            # 第一步：progress = 1.0（步长最大）
            # 最后一步：progress = 0.0（步长最小）
            remaining_range = abs(t - end_t)
            if initial_range > 0:
                progress = float(remaining_range) / float(initial_range)
            else:
                progress = 0.0
            progress = max(0.0, min(1.0, progress))  # 限制在 [0,1]
            
            # 使用线性函数 f = a * x + b 计算步长
            # 第一步（progress=1.0）：step_size = a * 1 + b = a + b（上限）
            # 最后一步（progress=0.0）：step_size = a * 0 + b = b（下限）
            n = coeff_a * progress + coeff_b  # 根据线性函数计算步长。
            step = max(1, int(round(n)))  # 至少前进 1 步。
            t_next = t + step_sign * step  # 计算下一时间。
            if decreasing and t_next < end_t:  # 避免越界。
                t_next = end_t
            if not decreasing and t_next > end_t:  # 避免越界。
                t_next = end_t
            if t_next == t:  # 若步长为零，强制前进一格。
                t_next = t + step_sign
            t = int(np.clip(t_next, 0, self.num_timesteps - 1))  # 更新当前时间并裁剪。

        return indices  # 返回构建好的时间序列。

    @torch.no_grad()
    def sample_diffusion_large_step(self, protein_pos, protein_v, batch_protein,
                                    init_ligand_pos, init_ligand_v, batch_ligand,
                                    num_steps=None, center_pos_mode=None, pos_only=False,
                                    step_stride=None, step_size=None, add_noise=None,
                                    pos_clip=None, v_clip=None, log_ligand_input_mode='auto'):

        if num_steps is None:  # 如果未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        if center_pos_mode is None:  # 若未指定中心化模式。
            center_pos_mode = self.center_pos_mode  # 使用默认配置。

        defaults = self.dynamic_large_step_defaults or {}  # 读取大步采样的默认参数。
        schedule_mode = defaults.get('schedule', 'lambda')  # 获取调度模式。
        use_lambda_schedule = schedule_mode == 'lambda'  # 判断是否使用 lambda 调度。
        use_linear_schedule = schedule_mode == 'linear'  # 判断是否使用线性调度。
        if not use_lambda_schedule and not use_linear_schedule:  # 若使用固定步长调度。
            step_stride = int(step_stride if step_stride is not None else defaults.get('stride', 50))  # 设置步长。
            step_stride = max(step_stride, 1)  # 保证步长至少为 1。
        step_size = float(step_size if step_size is not None else defaults.get('step_size', 1.0))  # 设置更新步幅。
        add_noise = float(add_noise if add_noise is not None else defaults.get('noise_scale', 0.0))  # 设置附加噪声。
        pos_clip = pos_clip if pos_clip is not None else defaults.get('pos_clip', self.dynamic_pos_step_clip)  # 位置梯度裁剪阈值。
        v_clip = v_clip if v_clip is not None else defaults.get('v_clip', self.dynamic_v_step_clip)  # 类别梯度裁剪阈值。

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 按需中心化坐标。

        log_mode = log_ligand_input_mode  # 决定 log 输入模式。
        if log_mode == 'auto':  # 自动模式根据配置选择。
            log_mode = self.ligand_v_input if self.ligand_v_input in ('log_prob', 'logits', 'prob') else 'auto'  # 若模型原本使用概率形式则保持一致。
        log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=log_mode)  # 将输入转换为 log 概率。

        min_t = max(self.num_timesteps - num_steps, 0)  # 计算最小时间步。
        if use_lambda_schedule:  # 使用 lambda 调度。
            lambda_coeff_a = defaults.get('lambda_coeff_a', 80.0)  # 调度参数 a。
            lambda_coeff_b = defaults.get('lambda_coeff_b', 20.0)  # 调度参数 b。
            lambda_floor = defaults.get('time_lower', self.num_timesteps // 2)  # 调度下限。
            lambda_floor = int(np.clip(lambda_floor, min_t, self.num_timesteps - 1))  # 限制在合法范围。
            time_indices = self._build_lambda_schedule(
                start_t=self.num_timesteps - 1,  # 起始于最后一个时间步。
                end_t=lambda_floor,  # 结束于调度下限。
                coeff_a=lambda_coeff_a,  # 传入参数 a。
                coeff_b=lambda_coeff_b  # 传入参数 b。
            )
        elif use_linear_schedule:  # 使用线性调度。
            lambda_floor = defaults.get('time_lower', self.num_timesteps // 2)  # 调度下限。
            lambda_floor = int(np.clip(lambda_floor, min_t, self.num_timesteps - 1))  # 限制在合法范围。
            # 支持两种配置方式：直接指定上下限（推荐），或指定系数（向后兼容）
            if 'linear_step_upper' in defaults or 'linear_step_lower' in defaults:
                # 新方式：直接指定上下限，自动计算系数
                linear_step_upper = defaults.get('linear_step_upper', 30.0)  # 步长上限。
                linear_step_lower = defaults.get('linear_step_lower', 20.0)  # 步长下限。
                linear_coeff_a = linear_step_upper - linear_step_lower  # 自动计算：a = 上限 - 下限。
                linear_coeff_b = linear_step_lower  # b = 下限。
            else:
                # 旧方式：直接指定系数（向后兼容）
                linear_coeff_a = defaults.get('linear_coeff_a', 10.0)  # 线性函数系数 a。
                linear_coeff_b = defaults.get('linear_coeff_b', 20.0)  # 线性函数系数 b（步长下限）。
            time_indices = self._build_linear_schedule(
                start_t=self.num_timesteps - 1,  # 起始于最后一个时间步。
                end_t=lambda_floor,  # 结束于调度下限。
                coeff_a=linear_coeff_a,  # 线性函数系数 a。
                coeff_b=linear_coeff_b  # 线性函数系数 b（步长下限）。
            )
        else:  # 使用固定步长。
            time_indices = list(range(self.num_timesteps - 1, min_t - 1, -step_stride))  # 按步长向前遍历时间。

        # 大步阶段通常不使用带噪声方法，保持标准方法以获得更快的探索
        use_with_noise = defaults.get('use_with_noise', False)  # 大步阶段默认不使用带噪声方法。
        use_adaptive_step = defaults.get('use_adaptive_step', False)  # 从配置读取自适应步长开关，默认关闭。
        use_time_scale = defaults.get('use_time_scale', True)  # 从配置读取时间缩放开关，默认启用。
        ligand_pos, log_ligand_v, pos_traj, log_v_traj = self._dynamic_diffusion(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            ligand_pos=ligand_pos,  # 当前配体坐标。
            log_ligand_v=log_ligand_v,  # 当前配体 log 分布。
            batch_ligand=batch_ligand,  # 配体批次。
            time_indices=time_indices,  # 调度时间序列。
            step_size=step_size,  # 每次迭代步幅。
            add_noise=add_noise,  # 附加噪声强度。
            pos_clip=pos_clip,  # 位置梯度裁剪阈值。
            v_clip=v_clip,  # 类别梯度裁剪阈值。
            record_traj=True,  # 记录轨迹以便分析。
            pos_only=pos_only,  # 是否仅更新位置。
            use_with_noise=use_with_noise,  # 是否使用带噪声方法（大步阶段通常为 False）。
            use_adaptive_step=use_adaptive_step,  # 是否启用自适应步长机制。
            use_time_scale=use_time_scale  # 是否启用时间步对步长的乘算。
        )

        ligand_pos = ligand_pos + offset[batch_ligand]  # 将坐标移回原参考系。
        if pos_traj:  # 若记录了轨迹。
            offset_cpu = offset[batch_ligand].detach().cpu()  # 将偏移转到 CPU。
            pos_traj = [p + offset_cpu for p in pos_traj]  # 为每个轨迹补回偏移。

        return {  # 返回采样结果。
            'pos': ligand_pos,  # 最终配体位置。
            'log_v': log_ligand_v,  # 最终 log 概率。
            'v': log_ligand_v.argmax(dim=-1),  # 最终类别索引。
            'pos_traj': pos_traj,  # 位置轨迹。
            'log_v_traj': log_v_traj,  # 类别轨迹。
            'time_indices': time_indices,  # 使用的时间调度。
            'batch_ligand': batch_ligand,  # 配体批次信息。
            'offset': offset  # 记录用于还原的偏移。
        }

    @torch.no_grad()
    def sample_diffusion_refinement(self, protein_pos, protein_v, batch_protein,
                                    init_ligand_pos, init_ligand_v, batch_ligand,
                                    center_pos_mode=None, pos_only=False,
                                    step_stride=None, step_size=None, add_noise=None,
                                    pos_clip=None, v_clip=None,
                                    time_upper=None, time_lower=0, num_cycles=1,
                                    log_ligand_input_mode='auto'):

        if center_pos_mode is None:  # 未指定中心化模式时使用默认值。
            center_pos_mode = self.center_pos_mode

        defaults = self.dynamic_refine_defaults or {}  # 读取精炼阶段默认参数。
        schedule_mode = defaults.get('schedule', 'lambda')  # 获取调度模式。
        use_lambda_schedule = schedule_mode == 'lambda'  # 是否使用 lambda 调度。
        use_linear_schedule = schedule_mode == 'linear'  # 是否使用线性调度。
        if not use_lambda_schedule and not use_linear_schedule:  # 固定步长调度。
            step_stride = int(step_stride if step_stride is not None else defaults.get('stride', 10))  # 设置步长。
            step_stride = max(step_stride, 1)  # 确保步长至少为 1。
        step_size = float(step_size if step_size is not None else defaults.get('step_size', 0.2))  # 设置步幅。
        add_noise = float(add_noise if add_noise is not None else defaults.get('noise_scale', 0.0))  # 设置噪声强度。
        pos_clip = pos_clip if pos_clip is not None else defaults.get('pos_clip', self.dynamic_pos_step_clip)  # 位置梯度裁剪。
        v_clip = v_clip if v_clip is not None else defaults.get('v_clip', self.dynamic_v_step_clip)  # 类别梯度裁剪。

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 按需中心化。

        log_mode = log_ligand_input_mode  # 决定 log 输入模式。
        if log_mode == 'auto':  # 自动选择模式。
            log_mode = self.ligand_v_input if self.ligand_v_input in ('log_prob', 'logits', 'prob') else 'auto'
        log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=log_mode)  # 转换为 log 概率。

        if time_upper is None:  # 若未指定上界。
            time_upper = int(defaults.get('time_upper', min(self.num_timesteps - 1, 500)))  # 采用默认值或 500。
        time_upper = int(np.clip(time_upper, 0, self.num_timesteps - 1))  # 限制在合法范围。
        time_lower = int(np.clip(time_lower, 0, time_upper))  # 下界不超过上界。
        
        # 根据官方 GlintDM/DiffDynamic 实现，在精炼采样初始化时使用带噪声的预测
        use_with_noise = defaults.get('use_with_noise', True)  # 默认在精炼阶段使用带噪声方法。
        if use_with_noise and not pos_only:  # 如果启用且需要更新类别。
            # 使用带噪声的方法初始化 log_ligand_v，提供更平滑的起点
            init_t = torch.full(size=(batch_ligand.max().item() + 1,), fill_value=time_upper, 
                               dtype=torch.long, device=protein_pos.device)  # 构造初始时间步。
            log_ligand_v = self.q_v_pred_with_noise(log_ligand_v, init_t, batch_ligand)  # 使用带噪声的预测初始化。

        if use_lambda_schedule:  # 使用 lambda 调度。
            lambda_coeff_a = defaults.get('lambda_coeff_a', 40.0)  # 调度参数 a。
            lambda_coeff_b = defaults.get('lambda_coeff_b', 5.0)  # 调度参数 b。
            time_indices = self._build_lambda_schedule(
                start_t=time_upper,  # 从上界开始。
                end_t=time_lower,  # 到下界结束。
                coeff_a=lambda_coeff_a,  # 传入参数 a。
                coeff_b=lambda_coeff_b  # 传入参数 b。
            )
        elif use_linear_schedule:  # 使用线性调度。
            # 支持两种配置方式：直接指定上下限（推荐），或指定系数（向后兼容）
            if 'linear_step_upper' in defaults or 'linear_step_lower' in defaults:
                # 新方式：直接指定上下限，自动计算系数
                linear_step_upper = defaults.get('linear_step_upper', 15.0)  # 步长上限。
                linear_step_lower = defaults.get('linear_step_lower', 5.0)  # 步长下限。
                linear_coeff_a = linear_step_upper - linear_step_lower  # 自动计算：a = 上限 - 下限。
                linear_coeff_b = linear_step_lower  # b = 下限。
            else:
                # 旧方式：直接指定系数（向后兼容）
                linear_coeff_a = defaults.get('linear_coeff_a', 10.0)  # 线性函数系数 a。
                linear_coeff_b = defaults.get('linear_coeff_b', 5.0)  # 线性函数系数 b（步长下限）。
            time_indices = self._build_linear_schedule(
                start_t=time_upper,  # 从上界开始。
                end_t=time_lower,  # 到下界结束。
                coeff_a=linear_coeff_a,  # 线性函数系数 a。
                coeff_b=linear_coeff_b  # 线性函数系数 b（步长下限）。
            )
        else:  # 固定步长调度。
            time_indices = list(range(time_upper, time_lower - 1, -step_stride))  # 生成等间隔时间序列。

        pos_traj_total, log_v_traj_total = [], []  # 初始化轨迹列表。
        ligand_pos_current, log_ligand_v_current = ligand_pos, log_ligand_v  # 初始化当前状态。
        use_adaptive_step = defaults.get('use_adaptive_step', False)  # 从配置读取自适应步长开关，默认关闭。
        use_time_scale = defaults.get('use_time_scale', True)  # 从配置读取时间缩放开关，默认启用。
        for _ in range(max(int(num_cycles), 1)):  # 重复执行若干周期。
            ligand_pos_current, log_ligand_v_current, pos_traj, log_v_traj = self._dynamic_diffusion(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次。
                ligand_pos=ligand_pos_current,  # 当前配体坐标。
                log_ligand_v=log_ligand_v_current,  # 当前配体 log 概率。
                batch_ligand=batch_ligand,  # 配体批次。
                time_indices=time_indices,  # 调度时间序列。
                step_size=step_size,  # 步幅。
                add_noise=add_noise,  # 噪声强度。
                pos_clip=pos_clip,  # 位置梯度裁剪阈值。
                v_clip=v_clip,  # 类别梯度裁剪阈值。
                record_traj=True,  # 记录轨迹。
                pos_only=pos_only,  # 是否仅更新位置。
                use_with_noise=use_with_noise,  # 是否使用带噪声方法。
                use_adaptive_step=use_adaptive_step,  # 是否启用自适应步长机制（从配置读取）。
                use_time_scale=use_time_scale  # 是否启用时间步对步长的乘算（从配置读取）。
            )
            pos_traj_total.extend(pos_traj)  # 累积位置轨迹。
            log_v_traj_total.extend(log_v_traj)  # 累积类别轨迹。

        ligand_pos_final = ligand_pos_current + offset[batch_ligand]  # 将最终坐标还原到原参考系。
        if pos_traj_total:  # 若存在轨迹。
            offset_cpu = offset[batch_ligand].detach().cpu()  # 偏移转到 CPU。
            pos_traj_total = [p + offset_cpu for p in pos_traj_total]  # 为轨迹还原偏移。

        return {  # 返回精炼结果。
            'pos': ligand_pos_final,  # 最终配体位置。
            'log_v': log_ligand_v_current,  # 最终 log 概率。
            'v': log_ligand_v_current.argmax(dim=-1),  # 最终类别索引。
            'pos_traj': pos_traj_total,  # 位置轨迹。
            'log_v_traj': log_v_traj_total,  # 类别轨迹。
            'time_indices': time_indices,  # 使用的时间调度。
            'batch_ligand': batch_ligand,  # 配体批次。
            'offset': offset  # 记录中心化偏移。
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:  # 若未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        num_graphs = batch_protein.max().item() + 1  # 统计图数量。

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 中心化坐标。

        pos_traj, v_traj = [], []  # 初始化位置与类别轨迹。
        v0_pred_traj, vt_pred_traj = [], []  # 初始化预测概率轨迹。
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v  # 初始化配体状态。
        # time sequence  # 保留注释：构造时间序列。
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))  # 生成逆序时间列表。
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):  # 遍历时间步，显示进度条。
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)  # 构造当前时间张量。
            preds = self(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次索引。

                init_ligand_pos=ligand_pos,  # 当前配体坐标。
                init_ligand_v=ligand_v,  # 当前配体类别表示。
                batch_ligand=batch_ligand,  # 配体批次索引。
                time_step=t  # 当前时间步。
            )
            # Compute posterior mean and variance  # 保留注释：计算后验均值与方差。
            if self.model_mean_type == 'noise':  # 若预测噪声。
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos  # 获取噪声预测。
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)  # 反推 x0。
                v0_from_e = preds['pred_ligand_v']  # 保存类型预测 logits。
            elif self.model_mean_type == 'C0':  # 若预测 x0。
                pos0_from_e = preds['pred_ligand_pos']  # 直接使用位置预测。
                v0_from_e = preds['pred_ligand_v']  # 保存类型预测 logits。
            else:  # 未实现类型。
                raise ValueError  # 抛出异常。

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)  # 后验均值。
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)  # 对数方差。
            # no noise when t == 0  # 保留注释：t=0 不加噪声。
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)  # 构造掩码。
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)  # 按后验均值与方差采样下一个位置。
            ligand_pos = ligand_pos_next  # 更新位置。

            if not pos_only:  # 若需要更新类别。
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)  # 计算 log 概率。
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)  # 当前类别转换为 log one-hot。
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算后验。
                ligand_v_next = log_sample_categorical(log_model_prob)  # 采样下一步类别。

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())  # 记录预测 logits。
                vt_pred_traj.append(log_model_prob.clone().cpu())  # 记录后验概率。
                ligand_v = ligand_v_next  # 更新类别。

            ori_ligand_pos = ligand_pos + offset[batch_ligand]  # 将位置还原到原参考系。
            pos_traj.append(ori_ligand_pos.clone().cpu())  # 保存位置轨迹。
            v_traj.append(ligand_v.clone().cpu())  # 保存类别轨迹。

        ligand_pos = ligand_pos + offset[batch_ligand]  # 最终位置还原偏移。
        return {  # 返回采样结果。
            'pos': ligand_pos,  # 最终配体位置。
            'v': ligand_v,  # 最终类别标签。
            'pos_traj': pos_traj,  # 位置轨迹。
            'v_traj': v_traj,  # 类别轨迹。
            'v0_traj': v0_pred_traj,  # 每步预测 logits 轨迹。
            'vt_traj': vt_pred_traj  # 每步后验轨迹。
        }


class DiffDynamic(ScorePosNet3D):  # 定义 DiffDynamic 模型，继承 ScorePosNet3D。

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):  # 初始化 DiffDynamic 模型。
        if not hasattr(config, 'ligand_v_input'):  # 若配置缺少 ligand_v_input。
            config.ligand_v_input = 'log_prob'  # 默认采用 log 概率输入。
        if not hasattr(config, 'use_grad_fusion'):  # 若未指定是否使用梯度融合。
            config.use_grad_fusion = True  # 默认开启梯度融合。
        if not hasattr(config, 'grad_fusion_lambda'):  # 若未提供融合调度。
            config.grad_fusion_lambda = {'mode': 'linear', 'start': 0.8, 'end': 0.2}  # 使用线性调度默认值。
        if not hasattr(config, 'loss_v2_weight'):  # 若未指定第二类别损失权重。
            config.loss_v2_weight = 1.0  # 默认开启额外类别损失。
        super().__init__(config, protein_atom_feature_dim, ligand_atom_feature_dim)  # 调用父类构造函数。
        # 注意：_build_lambda_schedule 方法已从父类 ScorePosNet3D 继承，无需重复定义。

    @torch.no_grad()
    def dynamic_sample_diffusion(self, protein_pos, protein_v, batch_protein,
                                 init_ligand_pos, init_log_ligand_v, batch_ligand,
                                 num_steps=None, center_pos_mode=None, pos_only=False):
        if num_steps is None:  # 若未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        if center_pos_mode is None:  # 若未指定中心化模式。
            center_pos_mode = self.center_pos_mode  # 使用默认配置。

        large_cfg = self.dynamic_large_step_defaults or {}  # 读取大步采样配置。
        refine_cfg = self.dynamic_refine_defaults or {}  # 读取精炼配置。

        large_schedule = large_cfg.get('schedule', 'lambda')  # 获取大步调度类型。
        large_stride = None if large_schedule in ('lambda', 'linear') else large_cfg.get('stride')  # 确定大步步长（lambda/linear 调度时不需要 stride）。
        large_res = self.sample_diffusion_large_step(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            init_ligand_pos=init_ligand_pos,  # 初始配体坐标。
            init_ligand_v=init_log_ligand_v,  # 初始配体 log 概率。
            batch_ligand=batch_ligand,  # 配体批次。
            num_steps=num_steps,  # 总步数。
            center_pos_mode=center_pos_mode,  # 中心化模式。
            pos_only=pos_only,  # 是否仅更新位置。
            step_stride=large_stride,  # 步长。
            step_size=large_cfg.get('step_size'),  # 步幅。
            add_noise=large_cfg.get('noise_scale'),  # 噪声强度。
            pos_clip=large_cfg.get('pos_clip'),  # 位置裁剪。
            v_clip=large_cfg.get('v_clip'),  # 类别裁剪。
            log_ligand_input_mode='log_prob'  # 大步阶段固定使用 log 概率模式。
        )

        refine_schedule = refine_cfg.get('schedule', 'lambda')  # 获取精炼调度类型。
        refine_stride = None if refine_schedule in ('lambda', 'linear') else refine_cfg.get('stride')  # 精炼步长（lambda/linear 调度时不需要 stride）。
        refine_res = self.sample_diffusion_refinement(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            init_ligand_pos=large_res['pos'],  # 以上阶段输出的位置作为起点。
            init_ligand_v=large_res['log_v'],  # 使用大步阶段输出的 log 概率。
            batch_ligand=batch_ligand,  # 配体批次。
            center_pos_mode=center_pos_mode,  # 中心化模式。
            pos_only=pos_only,  # 是否仅更新位置。
            step_stride=refine_stride,  # 精炼步长。
            step_size=refine_cfg.get('step_size'),  # 精炼步幅。
            add_noise=refine_cfg.get('noise_scale'),  # 精炼噪声。
            pos_clip=refine_cfg.get('pos_clip'),  # 位置裁剪。
            v_clip=refine_cfg.get('v_clip'),  # 类别裁剪。
            time_upper=refine_cfg.get('time_upper'),  # 精炼上界。
            time_lower=refine_cfg.get('time_lower', 0),  # 精炼下界。
            num_cycles=refine_cfg.get('cycles', 1),  # 精炼循环次数。
            log_ligand_input_mode='log_prob'  # 精炼阶段同样使用 log 概率。
        )

        pos_traj = []  # 汇总位置轨迹。
        log_v_traj = []  # 汇总类别轨迹。
        if large_res.get('pos_traj'):  # 若大步阶段有轨迹。
            pos_traj.extend(large_res['pos_traj'])  # 合并轨迹。
        if refine_res.get('pos_traj'):  # 若精炼阶段有轨迹。
            pos_traj.extend(refine_res['pos_traj'])  # 合并轨迹。

        if large_res.get('log_v_traj'):  # 汇总类别轨迹。
            log_v_traj.extend(large_res['log_v_traj'])
        if refine_res.get('log_v_traj'):
            log_v_traj.extend(refine_res['log_v_traj'])

        return {  # 返回采样结果及元数据。
            'pred_ligand_pos': refine_res['pos'],  # 最终位置预测。
            'pred_ligand_v': refine_res['log_v'],  # 最终 log 概率。
            'pos_traj': pos_traj,  # 综合位置轨迹。
            'log_v_traj': log_v_traj,  # 综合类别轨迹。
            'meta': {  # 元信息。
                'large_step_time_indices': large_res.get('time_indices'),  # 大步阶段时间调度。
                'refine_time_indices': refine_res.get('time_indices')  # 精炼阶段时间调度。
            }
        }


def extract(coef, t, batch):  # 根据时间步从常量表中取出并匹配批次。
    """
    DiffDynamic/TargetDiff 兼容实现：从常量表中提取对应时间步和批次的系数。
    
    Args:
        coef: 形状为 (num_timesteps,) 的一维张量，包含所有时间步的系数
        t: 可以是：
           - 形状为 (num_graphs,) 的张量，每个元素是对应图的时间步索引
           - 长度为 num_graphs 的列表，每个元素是对应图的时间步索引
        batch: 形状为 (num_atoms,) 的张量，每个元素是原子所属的图索引（在 [0, num_graphs-1] 范围内）
    
    Returns:
        形状为 (num_atoms, 1) 的张量，每个原子对应其所属图的时间步系数
    """
    # 确保 t 是张量格式
    if not isinstance(t, torch.Tensor):
        if isinstance(t, (list, tuple)):
            t = torch.tensor(t, dtype=torch.long, device=coef.device)
        else:
            # 标量情况：转换为单元素张量
            t = torch.tensor([t], dtype=torch.long, device=coef.device)
    
    # 确保 t 在正确的设备上
    if t.device != coef.device:
        t = t.to(coef.device)
    
    # 确保 batch 在正确的设备上
    if batch.device != coef.device:
        batch = batch.to(coef.device)
    
    # 验证输入有效性
    if batch.numel() == 0:
        raise ValueError(f"extract: batch is empty, cannot extract coefficients")
    
    if t.numel() == 0:
        raise ValueError(f"extract: t is empty, cannot extract coefficients")
    
    # 验证时间步索引范围
    if t.max().item() >= coef.shape[0] or t.min().item() < 0:
        raise IndexError(
            f"extract: time step indices out of range. t range=[{t.min().item()}, {t.max().item()}], "
            f"coef shape={coef.shape}, num_timesteps={coef.shape[0]}. "
            f"This may indicate that the sampling time step exceeds the model's training range."
        )
    
    # 计算批次数
    num_graphs = t.shape[0]
    
    # 验证 batch 索引范围
    if batch.max().item() >= num_graphs or batch.min().item() < 0:
        raise IndexError(
            f"extract: batch indices out of range. batch range=[{batch.min().item()}, {batch.max().item()}], "
            f"num_graphs={num_graphs}, batch.shape={batch.shape}, t.shape={t.shape}, "
            f"t range=[{t.min().item()}, {t.max().item()}]. "
            f"This indicates a mismatch between batch_ligand and the number of graphs."
        )
    
    # 执行索引：coef[t] 返回形状为 (num_graphs,) 的张量
    try:
        coef_t = coef[t]  # 形状: (num_graphs,)
    except IndexError as e:
        raise IndexError(
            f"extract: Failed to index coef with t. coef.shape={coef.shape}, "
            f"t={t}, t.shape={t.shape}, t range=[{t.min().item()}, {t.max().item()}]. "
            f"Original error: {e}"
        ) from e
    
    # 验证 coef_t 的形状
    if coef_t.shape[0] != num_graphs:
        raise RuntimeError(
            f"extract: coef[t] shape mismatch. Expected first dimension {num_graphs}, "
            f"got {coef_t.shape[0]}. coef.shape={coef.shape}, t={t}, t.shape={t.shape}"
        )
    
    # 使用 batch 索引：coef_t[batch] 返回形状为 (num_atoms,) 的张量
    try:
        result = coef_t[batch]  # 形状: (num_atoms,)
    except IndexError as e:
        raise IndexError(
            f"extract: Failed to index coef_t with batch. coef_t.shape={coef_t.shape}, "
            f"batch.shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}], "
            f"num_graphs={num_graphs}. Original error: {e}"
        ) from e
    
    # 验证结果形状
    if result.shape[0] != batch.shape[0]:
        raise RuntimeError(
            f"extract: Result shape mismatch. Expected first dimension {batch.shape[0]}, "
            f"got {result.shape[0]}. coef_t.shape={coef_t.shape}, batch.shape={batch.shape}"
        )
    
    # 扩展最后一维以便广播：形状变为 (num_atoms, 1)
    return result.unsqueeze(-1)
