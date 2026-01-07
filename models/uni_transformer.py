# 总结：
# - 实现多阶 UniTransformer 模块，用于蛋白-配体联合注意力与几何更新。
# - 提供从坐标到特征 (x2h) 与特征到坐标 (h2x) 的注意力层、信息融合与结构化更新。
# - 支持可配置的邻接构造、权重网络、归一化和激活策略，适配多种精炼需求。

import numpy as np  # 导入 NumPy，用于数值计算。
import torch  # 导入 PyTorch 主库。
import torch.nn as nn  # 导入神经网络模块。
import torch.nn.functional as F  # 导入函数式 API。
from torch_geometric.nn import radius_graph, knn_graph  # 导入图构建工具。
from torch_scatter import scatter_softmax, scatter_sum  # 导入散射 Softmax 和求和函数。

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product  # 导入通用组件。


class BaseX2HAttLayer(nn.Module):  # 定义从坐标到隐藏特征的基础注意力层。
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):  # 初始化注意力层参数。
        super().__init__()  # 调用父类构造函数。
        self.input_dim = input_dim  # 记录输入特征维度。
        self.hidden_dim = hidden_dim  # 记录隐藏维度。
        self.output_dim = output_dim  # 记录输出维度。
        self.n_heads = n_heads  # 记录多头数量。
        self.act_fn = act_fn  # 保存激活函数名称。
        self.edge_feat_dim = edge_feat_dim  # 记录边特征维度。
        self.r_feat_dim = r_feat_dim  # 记录径向特征维度。
        self.ew_net_type = ew_net_type  # 指定边权类型。
        self.out_fc = out_fc  # 指示是否使用输出全连接层。

        # attention key func  # 保留注释：以下构建注意力键网络。
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim  # 计算键/值输入维度。
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 构建键网络。

        # attention value func  # 保留注释：构建注意力值网络。
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 构建值网络。

        # attention query func  # 保留注释：构建查询网络。
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 构建查询网络。
        if ew_net_type == 'r':  # 当边权基于径向特征时。
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())  # 使用径向特征映射边权。
        elif ew_net_type == 'm':  # 当边权基于消息特征时。
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())  # 使用输出特征映射边权。

        if self.out_fc:  # 若启用输出全连接层。
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 定义融合层。

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):  # 前向传播计算隐藏特征更新。
        N = h.size(0)  # 节点数量。
        src, dst = edge_index  # 边的源、目标索引。
        hi, hj = h[dst], h[src]  # 取出目标与源节点特征。

        # multi-head attention  # 保留注释：多头注意力流程。
        # decide inputs of k_func and v_func  # 保留注释：构造键值输入。
        kv_input = torch.cat([r_feat, hi, hj], -1)  # 拼接径向特征与节点特征。
        if edge_feat is not None:  # 若存在额外边特征。
            kv_input = torch.cat([edge_feat, kv_input], -1)  # 追加边特征。

        # compute k  # 保留注释：计算键。
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  # 生成键并拆成多头。
        # compute v  # 保留注释：计算值。
        v = self.hv_func(kv_input)  # 生成值向量。

        if self.ew_net_type == 'r':  # 根据径向特征确定边权。
            e_w = self.ew_net(r_feat)  # 计算边权。
        elif self.ew_net_type == 'm':  # 根据消息确定边权。
            e_w = self.ew_net(v[..., :self.hidden_dim])  # 利用前半部分特征生成边权。
        elif e_w is not None:  # 若外部传入边权。
            e_w = e_w.view(-1, 1)  # 调整形状。
        else:  # 未提供边权时。
            e_w = 1.  # 使用单位权重。
        v = v * e_w  # 将边权应用到值向量。
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)  # 将值拆分为多头。

        # compute q  # 保留注释：计算查询。
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)  # 构造查询张量。

        # compute attention weights  # 保留注释：计算注意力权重。
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]  # Softmax 聚合权重。

        # perform attention-weighted message-passing  # 保留注释：应用注意力消息传递。
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)  # 权重与值相乘。
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)  # 将消息聚合到节点。
        output = output.view(-1, self.output_dim)  # 展平多头维度。
        if self.out_fc:  # 若需要后处理。
            output = self.node_output(torch.cat([output, h], -1))  # 拼接残差并通过 MLP。

        output = output + h  # 添加残差连接。
        return output  # 返回更新后的隐藏特征。


class BaseH2XAttLayer(nn.Module):  # 定义从隐藏特征到坐标更新的基础注意力层。
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):  # 初始化参数。
        super().__init__()  # 调用父类构造函数。
        self.input_dim = input_dim  # 记录输入特征维度。
        self.hidden_dim = hidden_dim  # 记录隐藏维度。
        self.output_dim = output_dim  # 记录输出维度。
        self.n_heads = n_heads  # 多头数量。
        self.edge_feat_dim = edge_feat_dim  # 边特征维度。
        self.r_feat_dim = r_feat_dim  # 径向特征维度。
        self.act_fn = act_fn  # 激活函数名称。
        self.ew_net_type = ew_net_type  # 边权类型。

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim  # 计算键/值输入维度。

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 键网络。
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)  # 值网络输出每头权重。
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)  # 查询网络。
        if ew_net_type == 'r':  # 若边权基于径向特征。
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())  # 生成边权映射。

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):  # 前向传播计算坐标更新。
        N = h.size(0)  # 节点数量。
        src, dst = edge_index  # 解析边索引。
        hi, hj = h[dst], h[src]  # 获取目标与源节点特征。

        # multi-head attention  # 保留注释：多头注意力流程。
        # decide inputs of k_func and v_func  # 保留注释：构造键值输入。
        kv_input = torch.cat([r_feat, hi, hj], -1)  # 拼接径向和节点特征。
        if edge_feat is not None:  # 若存在额外边特征。
            kv_input = torch.cat([edge_feat, kv_input], -1)  # 拼接边特征。

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  # 计算键并拆分多头。
        v = self.xv_func(kv_input)  # 计算值向量（每头标量权重）。
        if self.ew_net_type == 'r':  # 根据径向特征生成边权。
            e_w = self.ew_net(r_feat)  # 计算边权。
        elif self.ew_net_type == 'm':  # 当边权固定为 1。
            e_w = 1.  # 使用单位权重。
        elif e_w is not None:  # 若外部提供边权。
            e_w = e_w.view(-1, 1)  # 调整形状。
        else:  # 默认情况。
            e_w = 1.  # 使用单位权重。
        v = v * e_w  # 应用边权。

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]  # 将标量缩放方向向量。
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)  # 计算查询。

        # Compute attention weights  # 保留注释：计算注意力权重。
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)  # Softmax 权重。

        # Perform attention-weighted message-passing  # 保留注释：注意力加权消息传递。
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)  # 权重乘以矢量消息。
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)  # 聚合到目标节点。
        return output.mean(1)  # [num_nodes, 3]  # 对多头结果求平均得到位移。


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):  # 定义既更新隐藏特征又更新坐标的注意力块。
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):  # 初始化块级配置。
        super().__init__()  # 调用基类构造函数。
        self.hidden_dim = hidden_dim  # 记录隐藏特征维度。
        self.n_heads = n_heads  # 记录注意力头数量。
        self.edge_feat_dim = edge_feat_dim  # 保存边特征维度。
        self.num_r_gaussian = num_r_gaussian  # 保存径向高斯展开数量。
        self.norm = norm  # 指示内部 MLP 是否使用归一化。
        self.act_fn = act_fn  # 保存激活函数名称。
        self.num_x2h = num_x2h  # 记录 x2h 子层数量。
        self.num_h2x = num_h2x  # 记录 h2x 子层数量。
        self.r_min, self.r_max = r_min, r_max  # 保存最小与最大距离阈值。
        self.num_node_types = num_node_types  # 记录节点类型数以决定外积尺寸。
        self.ew_net_type = ew_net_type  # 设置边权生成策略。
        self.x2h_out_fc = x2h_out_fc  # 标记 x2h 是否带额外输出全连接层。
        self.sync_twoup = sync_twoup  # 标记 h2x 是否使用原始隐藏特征。

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)  # 定义距离展开模块。

        self.x2h_layers = nn.ModuleList()  # 初始化 x2h 子层列表。
        for i in range(self.num_x2h):  # 构建每一个 x2h 子层。
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,  # 使用基础 x2h 注意力层。
                                r_feat_dim=num_r_gaussian * 4,  # 四种节点组合产生四倍径向特征。
                                act_fn=act_fn, norm=norm,  # 继承激活与归一化配置。
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)  # 使用统一的边权模式与输出设置。
            )
        self.h2x_layers = nn.ModuleList()  # 初始化 h2x 子层列表。
        for i in range(self.num_h2x):  # 构建每一个 h2x 子层。
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,  # 使用基础 h2x 注意力层。
                                r_feat_dim=num_r_gaussian * 4,  # 配置径向特征维度。
                                act_fn=act_fn, norm=norm,  # 继承激活与归一化策略。
                                ew_net_type=self.ew_net_type)  # 使用统一的边权模式。
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):  # 执行特征与坐标联合更新。
        src, dst = edge_index  # 获取边的源与目标索引。
        if self.edge_feat_dim > 0:  # 若边特征维度 > 0。
            edge_feat = edge_attr  # 边特征张量形状约为 [批量边数, 键类型数]。
        else:  # 否则。
            edge_feat = None  # 不使用额外边特征。

        rel_x = x[dst] - x[src]  # 计算边的相对坐标。
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  # 求取欧式距离。

        h_in = h  # 保存当前隐藏特征。
        # 4 separate distance embedding for p-p, p-l, l-p, l-l  # 保留注释：外积生成四类距离编码。
        for i in range(self.num_x2h):  # 遍历每个 x2h 子层。
            dist_feat = self.distance_expansion(dist)  # 对距离做高斯展开。
            dist_feat = outer_product(edge_attr, dist_feat)  # 与边特征做外积以区分节点类型组合。
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)  # 执行 x2h 注意力。
            h_in = h_out  # 更新隐藏特征供下一层使用。
        x2h_out = h_in  # 保存 x2h 流水线输出。

        new_h = h if self.sync_twoup else x2h_out  # 根据配置选择 h2x 输入隐藏特征。
        for i in range(self.num_h2x):  # 遍历每个 h2x 子层。
            dist_feat = self.distance_expansion(dist)  # 重新计算最新距离的高斯展开。
            dist_feat = outer_product(edge_attr, dist_feat)  # 与边特征做外积。
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)  # 计算坐标增量。
            if not fix_x:  # 若允许更新坐标。
                x = x + delta_x * mask_ligand[:, None]  # 仅更新掩码标记的配体节点坐标。
            rel_x = x[dst] - x[src]  # 重新计算相对坐标。
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  # 重新计算距离。

        return x2h_out, x  # 返回新的隐藏特征与坐标。


class UniTransformerO2TwoUpdateGeneral(nn.Module):  # 定义支持两次更新机制的 UniTransformer 结构。
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):  # 初始化整体参数。
        super().__init__()  # 调用基类构造。
        # 构建网络主体。
        self.num_blocks = num_blocks  # 堆叠的块数量。
        self.num_layers = num_layers  # 每个块内部层数。
        self.hidden_dim = hidden_dim  # 隐藏特征维度。
        self.n_heads = n_heads  # 注意力头数量。
        self.num_r_gaussian = num_r_gaussian  # 距离展开高斯数量。
        self.edge_feat_dim = edge_feat_dim  # 边特征维度。
        self.act_fn = act_fn  # 激活函数名称。
        self.norm = norm  # 是否启用归一化。
        self.num_node_types = num_node_types  # 节点类型数量。
        # radius graph / knn graph  # 保留注释：支持的图构建模式。
        self.cutoff_mode = cutoff_mode  # 截断方式（radius / knn / hybrid）。
        self.k = k  # kNN 中的邻居数量。
        self.ew_net_type = ew_net_type  # [r, m, none]  # 边权模式。

        self.num_x2h = num_x2h  # 每层内 x2h 子层数量。
        self.num_h2x = num_h2x  # 每层内 h2x 子层数量。
        self.num_init_x2h = num_init_x2h  # 初始块中 x2h 层数量。
        self.num_init_h2x = num_init_h2x  # 初始块中 h2x 层数量。
        self.r_max = r_max  # 距离上限。
        self.x2h_out_fc = x2h_out_fc  # 是否在 x2h 后附加输出 MLP。
        self.sync_twoup = sync_twoup  # 是否同步使用原始隐藏特征进行坐标更新。
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)  # 定义全局距离展开器。
        if self.ew_net_type == 'global':  # 若使用全局边权网络。
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)  # 根据距离特征预测全局边权。

        self.init_h_emb_layer = self._build_init_h_layer()  # 构建初始嵌入块。
        self.base_block = self._build_share_blocks()  # 构建共享的后续块。

    def __repr__(self):  # 定义对象字符串表示，便于打印。
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '  # 拼接关键配置与结构信息。

    def _build_init_h_layer(self):  # 构建初始嵌入模块。
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer  # 返回初始化层实例。

    def _build_share_blocks(self):  # 构建共享的主体块列表。
        # Equivariant layers  # 保留注释：以下构建等变层。
        base_block = []  # 初始化块列表。
        for l_idx in range(self.num_layers):  # 按层数循环构建。
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)  # 将块加入列表。
        return nn.ModuleList(base_block)  # 返回 ModuleList 以注册子模块。

    def _connect_edge(self, x, mask_ligand, batch):  # 根据配置构建边索引。
        if self.cutoff_mode == 'radius':  # 使用半径图。
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')  # 构建半径图。
        elif self.cutoff_mode == 'knn':  # 使用 kNN 图。
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')  # 构建 kNN 图。
        elif self.cutoff_mode == 'hybrid':  # 使用混合构图。
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)  # 组合全连接配体和 kNN 蛋白。
        else:  # 不支持的模式。
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')  # 抛出异常。
        return edge_index  # 返回边索引。

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):  # 根据节点身份生成边类型独热编码。
        src, dst = edge_index  # 解包源和目标。
        edge_type = torch.zeros(len(src)).to(edge_index)  # 初始化类型张量。
        n_src = mask_ligand[src] == 1  # 源节点是否为配体。
        n_dst = mask_ligand[dst] == 1  # 目标节点是否为配体。
        edge_type[n_src & n_dst] = 0  # 配体-配体边。
        edge_type[n_src & ~n_dst] = 1  # 配体-蛋白边。
        edge_type[~n_src & n_dst] = 2  # 蛋白-配体边。
        edge_type[~n_src & ~n_dst] = 3  # 蛋白-蛋白边。
        edge_type = F.one_hot(edge_type, num_classes=4)  # 转换为独热编码。
        return edge_type  # 返回边类型张量。

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):  # 执行完整前向传播。

        all_x = [x]  # 初始化坐标轨迹列表。
        all_h = [h]  # 初始化特征轨迹列表。

        for b_idx in range(self.num_blocks):  # 遍历每个块。
            edge_index = self._connect_edge(x, mask_ligand, batch)  # 根据当前坐标构建图。
            src, dst = edge_index  # 解包源和目标索引。

            # edge type (dim: 4)  # 保留注释：边类型为四维独热。
            edge_type = self._build_edge_type(edge_index, mask_ligand)  # 计算边类型独热编码。
            if self.ew_net_type == 'global':  # 若使用全局边权。
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)  # 计算边距离。
                dist_feat = self.distance_expansion(dist)  # 距离高斯展开。
                logits = self.edge_pred_layer(dist_feat)  # 预测边权 logits。
                e_w = torch.sigmoid(logits)  # 通过 sigmoid 转为权重。
            else:  # 否则不使用全局边权。
                e_w = None  # 传递 None 让下游自己决定。

            for l_idx, layer in enumerate(self.base_block):  # 遍历共享块中的每一层。
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)  # 更新隐藏特征与坐标。
            all_x.append(x)  # 记录当前坐标。
            all_h.append(h)  # 记录当前隐藏特征。

        outputs = {'x': x, 'h': h}  # 构造输出。
        if return_all:  # 若需要返回所有层轨迹。
            outputs.update({'all_x': all_x, 'all_h': all_h})  # 附加轨迹。
        return outputs  # 返回结果。
