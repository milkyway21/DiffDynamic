# 总结：
# - 提供分子性质预测网络，基于蛋白-配体联合编码器生成图级表示。
# - 支持裸特征与携带先验编码的两种模型变体，聚合上下文后输出目标性质。
# - 内置数据增强（坐标噪声）、任务筛选与可选外部特征融合逻辑。

import torch  # 导入 PyTorch 主库。
import torch.nn as nn  # 导入神经网络模块。
import torch.nn.functional as F  # 导入函数式 API。
from torch_scatter import scatter  # 导入 scatter 用于批量聚合。

from models.property_pred.prop_egnn import EnEquiEncoder  # 导入 EGNN 编码器实现。
from models.common import compose_context_prop, ShiftedSoftplus  # 导入上下文拼接与平移 Softplus 激活。


def get_encoder(config):  # 根据配置构建编码器实例。
    if config.name == 'egnn' or config.name == 'egnn_enc':  # 支持 egnn 与 egnn_enc 两种别名。
        net = EnEquiEncoder(  # 实例化等变编码器。
            num_layers=config.num_layers,  # 设置层数。
            edge_feat_dim=config.edge_dim,  # 指定边特征维度。
            hidden_dim=config.hidden_dim,  # 指定隐藏维度。
            num_r_gaussian=config.num_r_gaussian,  # 设置距离高斯展开数量。
            act_fn=config.act_fn,  # 指定激活函数。
            norm=config.norm,  # 控制是否使用归一化。
            update_x=False,  # 性质预测阶段不更新坐标。
            k=config.knn,  # 指定 kNN 邻居数量。
            cutoff=config.cutoff,  # 指定截断距离。
        )
    else:  # 未知编码器名称。
        raise ValueError(config.name)  # 抛出异常提示。
    return net  # 返回编码器实例。


class PropPredNet(nn.Module):  # 定义基础性质预测网络。
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, output_dim=3):  # 初始化网络结构。
        super(PropPredNet, self).__init__()  # 调用父类构造函数。
        self.config = config  # 保存配置对象。
        self.hidden_dim = config.hidden_channels  # 读取隐藏维度配置。
        self.output_dim = output_dim  # 设置输出维度（例如多任务）。
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)  # 蛋白原子特征映射层。
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)  # 配体原子特征映射层。

        # self.mean = target_mean  # 保留注释：预留目标归一化逻辑。
        # self.std = target_std
        # self.register_buffer('target_mean', target_mean)
        # self.register_buffer('target_std', target_std)
        self.encoder = get_encoder(config.encoder)  # 构建图编码器。
        self.out_block = nn.Sequential(  # 定义输出头。
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 第一层线性映射。
            ShiftedSoftplus(),  # 使用平移 Softplus 激活。
            nn.Linear(self.hidden_dim, output_dim),  # 映射到目标维度。
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind):  # 前向传播，返回图级预测。
        h_protein = self.protein_atom_emb(protein_atom_feature)  # 嵌入蛋白原子特征。
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)  # 嵌入配体原子特征。

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(  # 拼接蛋白与配体上下文。
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(  # 调用编码器获取节点表示。
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        # Aggregate messages  # 保留注释：以下执行池化。
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)  # 对每个图求和聚合。
        output = self.out_block(pre_out)  # (N, C)  # 通过输出头得到预测。
        if output_kind is not None:  # 若需要根据类型筛选输出。
            output_mask = F.one_hot(output_kind - 1, self.output_dim)  # 构造 one-hot 掩码。
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)  # 选择指定任务的预测。
        return output  # 返回预测结果。

    def get_loss(self, batch, pos_noise_std, return_pred=False):  # 计算训练损失，可选返回预测值。
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std  # 为蛋白坐标添加高斯噪声。
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std  # 为配体坐标添加高斯噪声。
        pred = self(  # 前向计算预测。
            protein_pos=batch.protein_pos + protein_noise,  # 输入加噪后的蛋白坐标。
            protein_atom_feature=batch.protein_atom_feature.float(),  # 转为 float 防止类型问题。
            ligand_pos=batch.ligand_pos + ligand_noise,  # 输入加噪后的配体坐标。
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),  # 使用全特征向量。
            batch_protein=batch.protein_element_batch,  # 蛋白批次索引。
            batch_ligand=batch.ligand_element_batch,  # 配体批次索引。
            output_kind=batch.kind,  # 指定输出任务类型。
            # output_kind=None  # 保留注释：可选择禁用多任务筛选。
        )
        # pred = pred * y_std + y_mean  # 保留注释：可执行反归一化。
        loss_func = nn.MSELoss()  # 定义均方误差损失。
        loss = loss_func(pred.view(-1), batch.y)  # 计算损失，展平预测以匹配标签。
        if return_pred:  # 若需要返回预测值。
            return loss, pred  # 同时返回损失和预测。
        else:  # 否则仅返回损失。
            return loss


class PropPredNetEnc(nn.Module):  # 定义支持外部编码特征的性质预测网络。
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim,
                 enc_ligand_dim, enc_node_dim, enc_graph_dim, enc_feature_type=None, output_dim=1):  # 初始化结构。
        super(PropPredNetEnc, self).__init__()  # 调用父类构造函数。
        self.config = config  # 保存配置对象。
        self.hidden_dim = config.hidden_channels  # 隐藏特征维度。
        self.output_dim = output_dim  # 输出维度。
        self.enc_ligand_dim = enc_ligand_dim  # 配体侧额外特征维度。
        self.enc_node_dim = enc_node_dim  # 节点特征附加维度。
        self.enc_graph_dim = enc_graph_dim  # 图级特征维度。
        self.enc_feature_type = enc_feature_type  # 指定使用的外部特征类型。

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)  # 蛋白特征嵌入层。
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + enc_ligand_dim, self.hidden_dim)  # 配体特征嵌入层，可拼接额外输入。
        # self.mean = target_mean  # 保留注释：可注册目标均值。
        # self.std = target_std
        # self.register_buffer('target_mean', target_mean)
        # self.register_buffer('target_std', target_std)
        self.encoder = get_encoder(config.encoder)  # 获取编码器。
        if self.enc_node_dim > 0:  # 若需要节点级外部特征。
            self.enc_node_layer = nn.Sequential(  # 定义节点特征融合层。
                nn.Linear(self.hidden_dim + self.enc_node_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.out_block = nn.Sequential(  # 定义输出模块。
            nn.Linear(self.hidden_dim + self.enc_graph_dim, self.hidden_dim),  # 融合图级特征。
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind, enc_ligand_feature, enc_node_feature, enc_graph_feature):  # 带外部特征的前向计算。
        h_protein = self.protein_atom_emb(protein_atom_feature)  # 嵌入蛋白原子特征。
        if enc_ligand_feature is not None:  # 若提供配体额外特征。
            ligand_atom_feature = torch.cat([ligand_atom_feature, enc_ligand_feature], dim=-1)  # 拼接输入。
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)  # 嵌入配体特征。

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(  # 构建上下文图。
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(  # 编码上下文节点。
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        if enc_node_feature is not None:  # 若有节点级外部特征。
            h_ctx = torch.cat([h_ctx, enc_node_feature], dim=-1)  # 拼接特征。
            h_ctx = self.enc_node_layer(h_ctx)  # 通过融合层调整维度。

        # Aggregate messages  # 保留注释：执行聚合。
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)  # 聚合节点特征。
        if enc_graph_feature is not None:  # 若提供图级特征。
            pre_out = torch.cat([pre_out, enc_graph_feature], dim=-1)  # 拼接进入输出头。

        output = self.out_block(pre_out)  # (N, C)  # 通过输出头得到预测。
        if output_kind is not None:  # 根据任务类型筛选输出。
            output_mask = F.one_hot(output_kind - 1, self.output_dim)  # 构造 one-hot 掩码。
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)  # 选择对应任务的预测值。
        return output  # 返回预测结果。

    def get_loss(self, batch, pos_noise_std, return_pred=False):  # 计算损失，支持多种外部特征。
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std  # 为蛋白坐标注入噪声。
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std  # 为配体坐标注入噪声。

        # add features  # 保留注释：根据配置加载辅助特征。
        enc_ligand_feature, enc_node_feature, enc_graph_feature = None, None, None  # 初始化外部特征缓存。
        if self.enc_feature_type == 'nll_all':
            enc_graph_feature = batch.nll_all  # [num_graphs, 22]  # 使用全部 NLL 特征。
        elif self.enc_feature_type == 'nll':
            enc_graph_feature = batch.nll  # [num_graphs, 20]  # 使用部分 NLL 特征。
        elif self.enc_feature_type == 'final_h':
            enc_node_feature = batch.final_h  # [num_pl_atoms, 128]  # 使用最终隐藏节点特征。
        elif self.enc_feature_type == 'pred_ligand_v':
            enc_ligand_feature = batch.pred_ligand_v  # [num_l_atoms, 13]  # 使用预测的配体类别分布。
        elif self.enc_feature_type == 'pred_v_entropy_pre':
            enc_ligand_feature = batch.pred_v_entropy   # [num_l_atoms, 1]  # 使用节点熵特征。
        elif self.enc_feature_type == 'pred_v_entropy_post':
            enc_graph_feature = scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')   # [num_graphs, 1]  # 聚合熵特征。
        elif self.enc_feature_type == 'full':
            enc_graph_feature = torch.cat(
                [batch.nll_all, scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')], dim=-1)  # 拼接图级特征。
            enc_node_feature = batch.final_h  # 节点级使用最终隐藏表示。
            enc_ligand_feature = torch.cat([batch.pred_ligand_v, batch.pred_v_entropy], -1)  # 配体级拼接预测与熵。
        else:
            raise NotImplementedError  # 未实现的特征类型抛出异常。

        pred = self(  # 执行前向预测。
            protein_pos=batch.protein_pos + protein_noise,  # 传入加噪蛋白坐标。
            protein_atom_feature=batch.protein_atom_feature.float(),  # 传入蛋白特征。
            ligand_pos=batch.ligand_pos + ligand_noise,  # 传入加噪配体坐标。
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),  # 传入配体特征。
            batch_protein=batch.protein_element_batch,  # 蛋白批次索引。
            batch_ligand=batch.ligand_element_batch,  # 配体批次索引。
            output_kind=batch.kind,  # 指定输出任务类型。
            # output_kind=None,  # 保留注释：可禁用任务筛选。
            enc_ligand_feature=enc_ligand_feature,  # 传入配体额外特征。
            enc_node_feature=enc_node_feature,  # 传入节点额外特征。
            enc_graph_feature=enc_graph_feature  # 传入图级额外特征。
        )
        # pred = pred * y_std + y_mean  # 保留注释：可执行反归一化。
        loss_func = nn.MSELoss()  # 定义 MSE 损失。
        loss = loss_func(pred.view(-1), batch.y)  # 计算损失。
        if return_pred:  # 若需要返回预测值。
            return loss, pred  # 返回损失与预测。
        else:
            return loss  # 否则仅返回损失。
