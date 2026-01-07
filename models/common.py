# 总结：
# - 提供几何建模常用的基础模块：径向与角度展开、激活函数、MLP 等。
# - 实现多种特征构造与上下文拼接工具，支持蛋白-配体场景的图结构处理。
# - 提供混合边连接函数，组合全连接与 kNN 构造分子图边集。

import torch  # 导入 PyTorch 主库。
import torch.nn as nn  # 导入神经网络模块，包含常用层和容器。
import torch.nn.functional as F  # 导入函数式接口，提供激活等函数。
from torch_geometric.nn import knn_graph  # 从 PyG 导入 kNN 图构建函数。


class GaussianSmearing(nn.Module):  # 定义高斯平滑模块，用于将距离映射到高斯基函数空间。
    """将标量距离嵌入到高斯基展开中。"""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):  # 初始化函数，设置范围和高斯数量。
        super(GaussianSmearing, self).__init__()  # 调用父类构造函数。
        self.start = start  # 保存距离下界。
        self.stop = stop  # 保存距离上界。
        self.num_gaussians = num_gaussians  # 保存高斯核数量。
        if fixed_offset:  # 判断是否使用自定义的一组离散偏移。
            # customized offset  # 保留原注释：自定义偏移值。
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])  # 定义离散偏移张量。
        else:  # 如果不使用固定偏移。
            offset = torch.linspace(start, stop, num_gaussians)  # 按均匀间隔生成偏移。
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2  # 计算高斯指数因子，控制宽度。
        self.register_buffer('offset', offset)  # 注册偏移为缓冲区，便于与模型同步到设备。

    def __repr__(self):  # 定义对象字符串表示，便于调试。
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'  # 返回包含参数信息的字符串。

    def forward(self, dist):  # 定义前向传播，将距离映射到高斯展开。
        dist = dist.view(-1, 1) - self.offset.view(1, -1)  # 扩展输入距离并减去偏移，得到距偏移差。
        return torch.exp(self.coeff * torch.pow(dist, 2))  # 对差值平方并乘系数，然后取指数得到高斯响应。


class AngleExpansion(nn.Module):  # 定义角度展开模块，使用余弦基函数表达角度。
    """使用余弦基展开角度特征。"""
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):  # 初始化角度展开参数。
        super(AngleExpansion, self).__init__()  # 调用父类构造函数。
        l_mul = 1. / torch.linspace(stop, start, half_expansion)  # 计算左侧倒数系数序列。
        r_mul = torch.linspace(start, stop, half_expansion)  # 计算右侧线性系数序列。
        coeff = torch.cat([l_mul, r_mul], dim=-1)  # 拼接左右序列构成完整系数。
        self.register_buffer('coeff', coeff)  # 将系数注册为缓冲区。

    def forward(self, angle):  # 定义前向传播，展开角度特征。
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))  # 将角度乘系数并取余弦得到展开特征。


class Swish(nn.Module):  # 定义可学习参数化的 Swish 激活函数。
    """可学习 β 的 Swish 激活。"""
    def __init__(self):  # 初始化 Swish 模块。
        super(Swish, self).__init__()  # 调用父类构造函数。
        self.beta = nn.Parameter(torch.tensor(1.0))  # 定义可学习的缩放参数 beta。

    def forward(self, x):  # 定义前向传播。
        return x * torch.sigmoid(self.beta * x)  # 使用参数化公式计算 Swish 激活。


NONLINEARITIES = {  # 定义常见激活函数映射表。
    "tanh": nn.Tanh(),  # 映射字符串 "tanh" 到双曲正切激活。
    "relu": nn.ReLU(),  # 映射字符串 "relu" 到 ReLU 激活。
    "softplus": nn.Softplus(),  # 映射字符串 "softplus" 到 Softplus 激活。
    "elu": nn.ELU(),  # 映射字符串 "elu" 到 ELU 激活。
    "swish": Swish(),  # 映射字符串 "swish" 到自定义 Swish 激活。
    'silu': nn.SiLU()  # 映射字符串 "silu" 到 SiLU 激活。
}


class MLP(nn.Module):  # 定义多层感知机，保持隐藏层维度一致。
    """MLP with the same hidden dim across all layers."""  # 保留原文档字符串。

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):  # 初始化 MLP 参数。
        super().__init__()  # 调用父类构造函数。
        layers = []  # 初始化层列表。
        for layer_idx in range(num_layer):  # 遍历每一层配置。
            if layer_idx == 0:  # 第一层连接输入到隐藏层。
                layers.append(nn.Linear(in_dim, hidden_dim))  # 添加线性层。
            elif layer_idx == num_layer - 1:  # 最后一层连接隐藏层到输出层。
                layers.append(nn.Linear(hidden_dim, out_dim))  # 添加线性层。
            else:  # 中间层保持隐藏维度。
                layers.append(nn.Linear(hidden_dim, hidden_dim))  # 添加线性层。
            if layer_idx < num_layer - 1 or act_last:  # 判断是否需要添加归一化和激活。
                if norm:  # 如果开启归一化。
                    layers.append(nn.LayerNorm(hidden_dim))  # 添加 LayerNorm。
                layers.append(NONLINEARITIES[act_fn])  # 添加指定的激活函数。
        self.net = nn.Sequential(*layers)  # 将层列表封装为顺序容器。

    def forward(self, x):  # 定义前向传播。
        return self.net(x)  # 将输入依次通过子模块。


def outer_product(*vectors):  # 定义外积函数，用于组合多个向量特征。
    """计算多个向量的外积并展平。

    Args:
        *vectors: 若干形状兼容的张量，要求第一维为批次。

    Returns:
        torch.Tensor: 展平后的外积结果。
    """
    # 检查是否有空张量
    if len(vectors) == 0:  # 如果没有输入向量。
        raise ValueError("outer_product requires at least one vector")
    
    # 如果第一个向量为空，直接返回空张量
    if vectors[0].numel() == 0:  # 如果第一个向量元素数为0。
        # 计算输出维度：累乘所有向量除批次外的维度
        total_dim = 1
        for v in vectors:  # 遍历所有向量。
            if len(v.shape) > 1:  # 如果向量有多维。
                total_dim *= v.shape[1]  # 累乘除批次外的维度。
            # 如果是一维向量，维度为1，不需要累乘
        return torch.empty((0, total_dim), device=vectors[0].device, dtype=vectors[0].dtype)  # 返回空张量。
    
    for index, vector in enumerate(vectors):  # 遍历传入的所有向量。
        if index == 0:  # 第一个向量初始化外积结果。
            out = vector.unsqueeze(-1)  # 在末尾添加维度以准备广播。
        else:  # 之后的向量逐步与当前结果相乘。
            # 检查当前向量是否为空
            if vector.numel() == 0:  # 如果当前向量为空。
                # 计算期望的输出维度
                total_dim = 1
                for v in vectors[:index+1]:  # 遍历到当前向量的所有向量。
                    if len(v.shape) > 1:  # 如果向量有多维。
                        total_dim *= v.shape[1]  # 累乘除批次外的维度。
                return torch.empty((0, total_dim), device=vector.device, dtype=vector.dtype)  # 返回空张量。
            out = out * vector.unsqueeze(1)  # 扩展向量维度后执行乘法。
            if out.numel() > 0:  # 如果结果不为空。
                out = out.view(out.shape[0], -1).unsqueeze(-1)  # 展平除批次外的维度并恢复形状。
            else:  # 如果结果为空。
                # 计算期望的输出维度
                total_dim = 1
                for v in vectors[:index+1]:  # 遍历到当前向量的所有向量。
                    if len(v.shape) > 1:  # 如果向量有多维。
                        total_dim *= v.shape[1]  # 累乘除批次外的维度。
                return torch.empty((0, total_dim), device=out.device, dtype=out.dtype)  # 返回空张量。
    return out.squeeze()  # 最终移除多余维度返回结果。


def get_h_dist(dist_metric, hi, hj):  # 定义函数计算隐藏特征距离。
    """根据指定度量计算隐藏表示之间的距离/相似度。"""
    if dist_metric == 'euclidean':  # 如果使用欧式距离。
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)  # 计算平方欧氏距离并保持最后一维。
        return h_dist  # 返回距离张量。
    elif dist_metric == 'cos_sim':  # 如果使用余弦相似度。
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)  # 计算 hi 的 L2 范数。
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)  # 计算 hj 的 L2 范数。
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)  # 计算归一化点积得到余弦相似度。
        return h_dist, hj_norm  # 返回相似度和 hj 的范数以备后续使用。


def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):  # 定义函数生成距离特征。
    """根据模式组合距离展开与节点类型信息。"""
    if mode == 'origin':  # 原始模式直接返回距离。
        r_feat = r  # 不做处理。
    elif mode == 'basic':  # 基础模式使用展开函数。
        r_feat = r_exp_func(r)  # 对距离执行展开。
    elif mode == 'sparse':  # 稀疏模式结合节点类型信息。
        src, dst = edge_index  # 解包边的起点与终点索引。
        nt_src = node_type[src]  # [n_edges, 8]  # 根据边索引获取源节点类型特征。
        nt_dst = node_type[dst]  # 获取目标节点类型特征。
        r_exp = r_exp_func(r)  # 对距离执行展开。
        r_feat = outer_product(nt_src, nt_dst, r_exp)  # 利用外积组合节点类型与距离特征。
    else:  # 对于未知模式。
        raise ValueError(mode)  # 抛出异常提示模式无效。
    return r_feat  # 返回生成的距离特征。


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):  # 定义蛋白-配体上下文拼接函数。
    """拼接蛋白/配体特征与坐标并返回掩码。"""
    # previous version has problems when ligand atom types are fixed  # 保留注释：说明历史问题。
    # (due to sorting randomly in case of same element)  # 保留注释：说明问题原因。

    # 验证输入不为空
    if batch_protein.numel() == 0 and batch_ligand.numel() == 0:
        raise ValueError(
            f"compose_context: Both batch_protein and batch_ligand are empty. "
            f"batch_protein.shape={batch_protein.shape}, batch_ligand.shape={batch_ligand.shape}"
        )
    
    if batch_ligand.numel() == 0:
        raise ValueError(
            f"compose_context: batch_ligand is empty. batch_protein.shape={batch_protein.shape}, "
            f"batch_ligand.shape={batch_ligand.shape}, h_ligand.shape={h_ligand.shape if hasattr(h_ligand, 'shape') else 'N/A'}"
        )

    # 确保 batch_protein 和 batch_ligand 在同一设备上
    if batch_protein.device != batch_ligand.device:
        batch_ligand = batch_ligand.to(batch_protein.device)
    
    # 创建配体掩码（在排序前创建，确保正确映射）
    n_protein = batch_protein.size(0)
    n_ligand = batch_ligand.size(0)
    mask_ligand_before_sort = torch.cat(
        [
            torch.zeros(n_protein, dtype=torch.bool),
            torch.ones(n_ligand, dtype=torch.bool),
        ],
        dim=0,
    ).to(batch_protein.device)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)  # 拼接批次索引，得到上下文批次向量。
    batch_ctx_cpu = batch_ctx.detach().cpu()
    
    # 验证拼接后的批次索引长度与掩码长度一致
    if batch_ctx.numel() != mask_ligand_before_sort.numel():
        raise ValueError(
            f"compose_context: batch_ctx and mask_ligand_before_sort size mismatch. "
            f"batch_ctx.numel()={batch_ctx.numel()}, mask_ligand_before_sort.numel()={mask_ligand_before_sort.numel()}"
        )
    
    if batch_ctx.numel() == 0:
        raise ValueError("compose_context: batch_ctx is empty after concatenation, cannot sort.")

    ctx_min = batch_ctx_cpu.min().item()
    ctx_max = batch_ctx_cpu.max().item()
    if ctx_min == ctx_max:
        # 所有批次索引完全相同，直接保持原顺序以避免在某些 CUDA 环境下的排序问题
        sort_idx_cpu = torch.arange(batch_ctx_cpu.size(0), dtype=torch.long)
    else:
        sort_idx_cpu = torch.sort(batch_ctx_cpu, stable=True).indices
    sort_idx = sort_idx_cpu.to(batch_ctx.device)

    # 验证排序索引的有效性
    if sort_idx.numel() != mask_ligand_before_sort.numel():
        raise ValueError(
            f"compose_context: sort_idx size mismatch. "
            f"sort_idx.numel()={sort_idx.numel()}, mask_ligand_before_sort.numel()={mask_ligand_before_sort.numel()}"
        )
    if sort_idx.min().item() < 0 or sort_idx.max().item() >= mask_ligand_before_sort.numel():
        raise ValueError(
            f"compose_context: sort_idx out of range. "
            f"sort_idx.range=[{sort_idx.min().item()}, {sort_idx.max().item()}], "
            f"expected [0, {mask_ligand_before_sort.numel() - 1}]"
        )

    # 使用排序索引重排掩码
    mask_ligand = mask_ligand_before_sort[sort_idx]

    # 如果掩码在 GPU 排序后变为空，尝试回退到 CPU 排序以规避驱动问题
    if mask_ligand.sum().item() == 0 and batch_ligand.numel() > 0:
        mask_cpu = mask_ligand_before_sort.detach().cpu()[sort_idx_cpu]
        mask_ligand = mask_cpu.to(mask_ligand_before_sort.device)
        sort_idx = sort_idx_cpu.to(batch_ctx.device)

    batch_ctx = batch_ctx[sort_idx]  # 根据排序结果重排批次索引。
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)  # 重排隐藏特征。
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)  # 重排坐标。

    # 验证 mask_ligand 的有效性
    if mask_ligand.sum().item() == 0:
        # 使用CPU计算unique()以避免CUDA相关问题，并添加更详细的调试信息
        batch_protein_unique = batch_protein.cpu().unique().tolist() if batch_protein.numel() > 0 else []
        batch_ligand_unique = batch_ligand.cpu().unique().tolist() if batch_ligand.numel() > 0 else []
        batch_ctx_unique = batch_ctx.cpu().unique().tolist() if batch_ctx.numel() > 0 else []
        
        # 检查排序索引是否正确
        sort_idx_min = sort_idx.min().item()
        sort_idx_max = sort_idx.max().item()
        expected_max = mask_ligand_before_sort.numel() - 1
        
        # 检查掩码在排序前是否正确
        mask_before_sum = mask_ligand_before_sort.sum().item()
        
        raise ValueError(
            f"compose_context: mask_ligand contains no True values after sorting. "
            f"batch_protein.size(0)={batch_protein.size(0)}, batch_ligand.size(0)={batch_ligand.size(0)}, "
            f"sort_idx.shape={sort_idx.shape}, sort_idx.range=[{sort_idx_min}, {sort_idx_max}], expected_max={expected_max}, "
            f"mask_ligand.sum()={mask_ligand.sum().item()}, mask_before_sort.sum()={mask_before_sum}, "
            f"batch_protein.unique()={batch_protein_unique}, batch_ligand.unique()={batch_ligand_unique}, "
            f"batch_ctx.unique()={batch_ctx_unique}, "
            f"batch_protein.min()={batch_protein.min().item()}, batch_protein.max()={batch_protein.max().item()}, "
            f"batch_ligand.min()={batch_ligand.min().item()}, batch_ligand.max()={batch_ligand.max().item()}"
        )
    
    if mask_ligand.sum().item() != batch_ligand.numel():
        raise ValueError(
            f"compose_context: mask_ligand count mismatch. mask_ligand.sum()={mask_ligand.sum().item()}, "
            f"batch_ligand.numel()={batch_ligand.numel()}"
        )

    return h_ctx, pos_ctx, batch_ctx, mask_ligand  # 返回拼接后的特征、坐标、批次和配体掩码。


def compose_context_prop(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):  # 定义属性预测场景的上下文拼接函数。
    """与 `compose_context` 类似，但返回蛋白掩码（用于性质预测）。"""
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)  # 拼接批次索引。
    sort_idx = batch_ctx.argsort()  # 使用默认排序获取索引。

    mask_protein = torch.cat([  # 构建蛋白掩码，标记蛋白节点。
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),  # 蛋白节点标记为 True。
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),  # 配体节点标记为 False。
    ], dim=0)[sort_idx]  # 按排序索引重排掩码。

    batch_ctx = batch_ctx[sort_idx]  # 重排批次索引。
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)  # 重排隐藏特征。
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)  # 重排坐标。

    return h_ctx, pos_ctx, batch_ctx  # 返回拼接后的特征、坐标和批次索引（无需掩码）。


class ShiftedSoftplus(nn.Module):  # 定义移位 Softplus 激活，用于平衡输出。
    """Softplus 激活并减去在 0 处的偏移量，使输出居零。"""
    def __init__(self):  # 初始化函数。
        super().__init__()  # 调用父类构造函数。
        self.shift = torch.log(torch.tensor(2.0)).item()  # 计算 Softplus(0) 的值作为偏移。

    def forward(self, x):  # 定义前向传播。
        return F.softplus(x) - self.shift  # 计算 softplus 并减去偏移，使得 x=0 时输出为 0。


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):  # 定义混合边连接，组合配体内部全连接与蛋白配体 kNN。
    """组合配体全连接与蛋白-配体 kNN，返回边索引。"""
    # fully-connected for ligand atoms  # 保留注释：配体内部全连接。
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))  # 构造重复的目标索引形成全连接。
    src = ligand_index.repeat(len(ligand_index))  # 构造重复的源索引形成全连接。
    mask = dst != src  # 创建掩码排除自环。
    dst, src = dst[mask], src[mask]  # 应用掩码保留非自环边。
    ll_edge_index = torch.stack([src, dst])  # 将源和目标堆叠为边索引矩阵。

    # knn for ligand-protein edges  # 保留注释：蛋白配体间使用 kNN。
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)  # 计算配体与蛋白坐标差。
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)  # 求取欧氏距离矩阵。
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices  # 找到每个配体最近的 k 个蛋白索引。
    knn_p_idx = protein_index[knn_p_idx]  # 将距离矩阵索引映射回全局蛋白索引。
    knn_l_idx = torch.unsqueeze(ligand_index, 1)  # 扩展配体索引维度。
    knn_l_idx = knn_l_idx.repeat(1, k)  # 复制以匹配 kNN 结果尺寸。
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)  # 堆叠蛋白和配体索引构成边。
    pl_edge_index = pl_edge_index.view(2, -1)  # 重塑形状为 (2, num_edges)。
    return ll_edge_index, pl_edge_index  # 返回配体内部边和配体-蛋白边。


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):  # 定义批量混合边连接函数。
    """在批量数据上应用 `hybrid_edge_connection` 并聚合结果。"""
    batch_size = batch.max().item() + 1  # 计算批次大小。
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []  # 初始化存储列表。
    with torch.no_grad():  # 在无梯度环境下构建边索引提升效率。
        for i in range(batch_size):  # 遍历每个批次。
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]  # 找到当前批次的配体索引。
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]  # 找到当前批次的蛋白索引。
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]  # 根据索引提取配体和蛋白坐标。
            ll_edge_index, pl_edge_index = hybrid_edge_connection(  # 调用函数构建混合边。
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)  # 保存配体内部边。
            batch_pl_edge_index.append(pl_edge_index)  # 保存配体-蛋白边。
            if add_p_index:  # 如果需要额外蛋白内部边。
                all_pos = torch.cat([protein_pos, ligand_pos], 0)  # 将蛋白和配体坐标拼接。
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')  # 在拼接坐标上构建 kNN 图。
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]  # 仅保留终点索引属于蛋白的边。
                p_src, p_dst = p_edge_index  # 解包源和目标索引。
                all_index = torch.cat([protein_index, ligand_index], 0)  # 拼接全局索引映射。
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)  # 将局部索引映射回全局索引。
                batch_p_edge_index.append(p_edge_index)  # 保存蛋白内部边。

    if add_p_index:  # 若包含蛋白内部边。
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(  # 将三类边拼接。
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:  # 仅包含配体内部和配体-蛋白边。
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]  # 拼接两类边。
    edge_index = torch.cat(edge_index, -1)  # 将所有批次的边拼接为整体索引。
    return edge_index  # 返回最终的边索引张量。
