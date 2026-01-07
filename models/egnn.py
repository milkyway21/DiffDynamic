# 总结：
# - 定义 EGNN 模型的基础层与整体网络，处理蛋白-配体图的特征与几何更新。
# - 通过径向高斯展开与边特征融合，实现隐藏表示与坐标的联合更新。
# - 支持多种邻接构造模式（kNN/混合），并输出中间层结果以便分析。

import torch  # 导入 PyTorch 主库，用于张量计算。
import torch.nn as nn  # 导入神经网络模块，用于构建可学习组件。
import torch.nn.functional as F  # 导入函数式接口，提供激活等通用函数。
from torch_scatter import scatter_sum  # 导入 scatter_sum，用于聚合边信息。
from torch_geometric.nn import radius_graph, knn_graph  # 导入图构建工具（半径图与 kNN 图）。
from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, NONLINEARITIES  # 导入公共模块中的工具。


class EnBaseLayer(nn.Module):  # 定义单层等变网络，处理节点特征与坐标更新。
    """单层 EGNN：结合边消息更新节点特征并可选更新坐标。"""
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):  # 初始化基础层。
        super().__init__()  # 调用父类构造函数。
        self.r_min = 0.  # 记录距离最小值。
        self.r_max = 10.  # 记录距离最大值。
        self.hidden_dim = hidden_dim  # 保存隐藏特征维度。
        self.num_r_gaussian = num_r_gaussian  # 保存径向高斯展开维度。
        self.edge_feat_dim = edge_feat_dim  # 保存附加边特征维度。
        self.update_x = update_x  # 标记是否需要更新坐标。
        self.act_fn = act_fn  # 保存激活函数名称。
        self.norm = norm  # 保存是否启用归一化的开关。
        if num_r_gaussian > 1:  # 当需要高斯展开时。
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)  # 构建高斯展开模块。
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,  # 构建边特征 MLP。
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)  # 配置层数和激活。
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())  # 构建边影响力网络。
        if self.update_x:  # 当需要更新坐标时。
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)  # 保留原实现注释。
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]  # 构建坐标更新网络的前两层。
            layer = nn.Linear(hidden_dim, 1, bias=False)  # 创建最终的线性层输出步长。
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # 初始化权重以保持稳定。
            x_mlp.append(layer)  # 添加线性层到序列。
            x_mlp.append(nn.Tanh())  # 添加 Tanh 限幅，避免更新过大。
            self.x_mlp = nn.Sequential(*x_mlp)  # 将层列表封装为顺序网络。

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)  # 构建节点更新 MLP。

    def forward(self, h, x, edge_index, mask_ligand, edge_attr=None):  # 定义前向传播，更新节点特征与坐标。
        src, dst = edge_index  # 提取边的源索引与目标索引。
        hi, hj = h[dst], h[src]  # 获取目标与源节点的隐藏特征。
        # \phi_e in Eq(3)  # 保留原注释：对应论文公式。
        rel_x = x[dst] - x[src]  # 计算相对位置向量。
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)  # 计算距离平方。
        if self.num_r_gaussian > 1:  # 根据配置选择距离特征。
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))  # 使用高斯展开处理距离。
        else:  # 当不使用高斯展开时。
            d_feat = d_sq  # 直接使用距离平方作为特征。
        if edge_attr is not None:  # 若提供额外边特征。
            edge_feat = torch.cat([d_feat, edge_attr], -1)  # 将距离特征与边属性拼接。
        else:  # 没有额外特征时。
            edge_feat = d_sq  # 仅使用距离平方。

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))  # 通过边 MLP 计算消息向量。
        eij = self.edge_inf(mij)  # 通过边影响力网络得到权重。
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])  # 将消息按目标节点聚合。

        # h update in Eq(6)  # 保留原注释：对应节点特征更新公式。
        h = h + self.node_mlp(torch.cat([mi, h], -1))  # 更新节点隐藏特征。
        if self.update_x:  # 当需要更新坐标时。
            # x update in Eq(4)  # 保留原注释：对应坐标更新公式。
            xi, xj = x[dst], x[src]  # 获取边两端坐标。
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable  # 保留原注释：解释归一化。
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)  # 聚合坐标增量。
            x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated  # 仅更新配体坐标。

        return h, x  # 返回更新后的隐藏特征与坐标。


class EGNN(nn.Module):  # 定义完整的 EGNN 模型，堆叠多个基础层。
    """堆叠 `EnBaseLayer` 的 EGNN 网络，支持 knn/混合边策略。"""
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0, cutoff_mode='knn',
                 update_x=True, act_fn='silu', norm=False):  # 初始化 EGNN 参数。
        super().__init__()  # 调用父类构造函数。
        # Build the network  # 保留原注释：构建网络结构。
        self.num_layers = num_layers  # 保存层数。
        self.hidden_dim = hidden_dim  # 保存隐藏维度。
        self.edge_feat_dim = edge_feat_dim  # 保存边特征维度。
        self.num_r_gaussian = num_r_gaussian  # 保存径向展开维度。
        self.update_x = update_x  # 标记是否更新坐标。
        self.act_fn = act_fn  # 保存激活函数名称。
        self.norm = norm  # 保存是否启用归一化。
        self.k = k  # 保存 kNN 中的 k 值。
        self.cutoff = cutoff  # 保存截断距离。
        self.cutoff_mode = cutoff_mode  # 保存边构建模式。
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian)  # 构建距离展开模块用于输入特征。
        self.net = self._build_network()  # 构建基础层堆叠。

    def _build_network(self):  # 构建等变层堆叠。
        # Equivariant layers  # 保留原注释：说明用途。
        layers = []  # 初始化层列表。
        for l_idx in range(self.num_layers):  # 遍历层索引。
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,  # 构建基础层。
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)  # 传入共享配置。
            layers.append(layer)  # 将层添加到列表。
        return nn.ModuleList(layers)  # 将层列表封装为 ModuleList 便于注册。

    # todo: refactor  # 保留原注释：说明未来需要重构。
    def _connect_edge(self, x, mask_ligand, batch):  # 根据模式构建边索引。
        # if self.cutoff_mode == 'radius':  # 保留原注释：半径模式待定。
        #     edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')  # 保留注释说明潜在实现。
        if self.cutoff_mode == 'knn':  # kNN 模式。
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')  # 构建基于 kNN 的边集。
        elif self.cutoff_mode == 'hybrid':  # 混合模式。
            edge_index = batch_hybrid_edge_connection(  # 使用自定义混合边构造。
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:  # 不支持的模式。
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')  # 抛出异常提示。
        return edge_index  # 返回边索引。

    # todo: refactor  # 保留原注释：说明需要重构。
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):  # 根据节点身份构建边类型编码。
        src, dst = edge_index  # 解包源和目标索引。
        edge_type = torch.zeros(len(src)).to(edge_index)  # 创建零向量用于记录类型。
        n_src = mask_ligand[src] == 1  # 判断源节点是否为配体。
        n_dst = mask_ligand[dst] == 1  # 判断目标节点是否为配体。
        edge_type[n_src & n_dst] = 0  # 配体-配体边编码为 0。
        edge_type[n_src & ~n_dst] = 1  # 配体-蛋白边编码为 1。
        edge_type[~n_src & n_dst] = 2  # 蛋白-配体边编码为 2。
        edge_type[~n_src & ~n_dst] = 3  # 蛋白-蛋白边编码为 3。
        edge_type = F.one_hot(edge_type, num_classes=4)  # 将类别索引转换为独热编码。
        return edge_type  # 返回边类型张量。

    def forward(self, h, x, mask_ligand, batch, return_all=False):  # 定义前向传播。
        all_x = [x]  # 初始化列表存储每层坐标。
        all_h = [h]  # 初始化列表存储每层隐藏特征。
        for l_idx, layer in enumerate(self.net):  # 遍历每一层。
            edge_index = self._connect_edge(x, mask_ligand, batch)  # 根据当前坐标构建边。
            edge_type = self._build_edge_type(edge_index, mask_ligand)  # 构建边类型特征。
            h, x = layer(h, x, edge_index, mask_ligand, edge_attr=edge_type)  # 通过基础层更新特征与坐标。
            all_x.append(x)  # 记录当前层坐标。
            all_h.append(h)  # 记录当前层特征。
        outputs = {'x': x, 'h': h}  # 构建输出字典。
        if return_all:  # 若需要返回所有层信息。
            outputs.update({'all_x': all_x, 'all_h': all_h})  # 将列表加入输出。
        return outputs  # 返回最终结果。
