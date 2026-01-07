# 总结：
# - 实现适用于性质预测的 EGNN 编码器，聚焦于隐藏特征更新，不修改坐标。
# - 提供基础等变层 `EnBaseLayer` 与整体堆叠 `EnEquiEncoder`，支持可调邻接策略。
# - 采用径向高斯展开与边权调制，对节点特征执行多层聚合。

import torch  # 导入 PyTorch 主库。
import torch.nn as nn  # 导入神经网络模块。
from torch_scatter import scatter_sum  # 导入 scatter_sum 聚合函数。
from torch_geometric.nn import radius_graph, knn_graph  # 导入半径图与 kNN 图构建工具。
from models.common import GaussianSmearing, MLP  # 导入高斯展开与 MLP 工具。


class EnBaseLayer(nn.Module):  # 定义性质预测 EGNN 的基础交互层。
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='relu', norm=False):  # 初始化层参数。
        super().__init__()  # 调用父类构造。
        self.r_min = 0.  # 设置距离下界。
        self.r_max = 10. ** 2  # 设置距离上界平方，覆盖较大尺度。
        self.hidden_dim = hidden_dim  # 记录隐藏维度。
        self.num_r_gaussian = num_r_gaussian  # 记录径向高斯数量。
        self.edge_feat_dim = edge_feat_dim  # 保存边特征维度。
        self.update_x = update_x  # 标记是否更新坐标（在性质预测中通常为 False）。
        self.act_fn = act_fn  # 保存激活函数名称。
        self.norm = norm  # 是否启用归一化。
        if num_r_gaussian > 1:  # 当使用多基径向展开时。
            self.r_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian, fixed_offset=False)  # 定义距离展开器。
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,  # 定义边特征 MLP。
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())  # 边重要性权重网络。
        if self.update_x:  # 若需要更新坐标。
            self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)  # 定义坐标更新网络。
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)  # 节点特征更新网络。

    def forward(self, h, edge_index, edge_attr):  # 前向传播，仅返回隐藏特征增量。
        dst, src = edge_index  # 获取边的目标与源索引。
        hi, hj = h[dst], h[src]  # 提取目标与源节点特征。
        # \phi_e in Eq(3)  # 保留注释：对齐原论文公式。
        mij = self.edge_mlp(torch.cat([edge_attr, hi, hj], -1))  # 生成边消息。
        eij = self.edge_inf(mij)  # 计算边重要性权重。
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])  # 聚合加权消息至目标节点。

        # h update in Eq(6)  # 保留注释：隐藏特征更新公式。
        # h = h + self.node_mlp(torch.cat([mi, h], -1))
        output = self.node_mlp(torch.cat([mi, h], -1))  # 计算节点特征增量。
        # if self.update_x:  # 保留注释：原始 EGNN 的坐标更新逻辑。
        #     # x update in Eq(4)
        #     xi, xj = x[dst], x[src]
        #     delta_x = scatter_sum((xi - xj) * self.x_mlp(mij), dst, dim=0)
        #     x = x + delta_x

        return output  # 返回特征更新量。


class EnEquiEncoder(nn.Module):  # 定义用于性质预测的等变编码器。
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0,
                 update_x=True, act_fn='relu', norm=False):  # 初始化编码器参数。
        super().__init__()  # 调用父类构造函数。
        # Build the network  # 保留注释：构建网络结构。
        self.num_layers = num_layers  # 记录层数。
        self.hidden_dim = hidden_dim  # 隐藏特征维度。
        self.edge_feat_dim = edge_feat_dim  # 边特征维度。
        self.num_r_gaussian = num_r_gaussian  # 距离展开基数。
        self.update_x = update_x  # 是否更新坐标（此模块通常不使用）。
        self.act_fn = act_fn  # 激活函数名称。
        self.norm = norm  # 是否启用归一化。
        self.k = k  # kNN 邻居数量。
        self.cutoff = cutoff  # 距离截断阈值。
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian, fixed_offset=False)  # 定义距离展开模块。
        self.net = self._build_network()  # 构建层堆叠。

    def _build_network(self):  # 构建编码器内部的层列表。
        # Equivariant layers  # 保留注释：构建等变层。
        layers = []  # 初始化层列表。
        for l_idx in range(self.num_layers):  # 循环创建每一层。
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)  # 将层加入列表。
        return nn.ModuleList(layers)  # 转换为 ModuleList 以注册子模块。

    def forward(self, node_attr, pos, batch):  # 前向传播，输出更新后的节点特征。
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)  # 保留注释：可切换半径图构建。
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')  # 使用 kNN 构建图。
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)  # 计算边长度。
        edge_attr = self.distance_expansion(edge_length)  # 生成边的距离特征。

        h = node_attr  # 初始化节点特征。
        for interaction in self.net:  # 依次应用每一层。
            h = h + interaction(h, edge_index, edge_attr)  # 残差更新特征。
        return h  # 返回编码后的节点特征。
