
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.GAL import GraphAttentionLayer
from einops import rearrange, repeat

class DynamicGraphModalityInteraction(nn.Module):
    """
    动态图模态交互网络 (DGMIN)
    输入: N个模态特征，每个形状为 [B, 384, 768]
    输出: N个更新后的模态特征，每个形状为 [B, 384, 768]
    """
    
    def __init__(self, d_model=768, num_heads=12, num_layers=1, dropout=0.):
        super().__init__()
        self.d_model = d_model
        
        # 1. 动态门控机制
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),  # 生成4个门控权重
            nn.Softmax(dim=-1)
        )
        # self.gate_generator = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 3),  # 生成4个门控权重
        #     nn.Softmax(dim=-1)
        # )

        # 边预测网络：输入 [B, 2D]，输出 logits [B, 1]
        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
        # 4. 图注意力机制
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 5. 特征融合的MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 6. 残差连接的权重学习
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # self.svd_cache_threshold = 0.0  # 相似度阈值

    # def should_update_svd(self, current_feat, cached_feat, threshold=0.1):
    #     """判断是否需要更新SVD，逐个batch进行判断"""
    #     # 检查是否处于训练模式
    #     if self.training:
    #         # 在训练模式下，始终返回True（即所有batch都需要更新），以确保梯度连续性
    #         return torch.ones(current_feat.size(0), dtype=torch.bool, device=current_feat.device)
    #
    #     if cached_feat is None:
    #         # 如果没有缓存，所有batch都需要更新
    #         return torch.ones(current_feat.size(0), dtype=torch.bool, device=current_feat.device)
    #
    #     # 计算特征相似度，逐个batch进行
    #     # 在序列维度(N)上求平均，保留特征维度(C)，得到 [B, C]
    #     current_mean = torch.mean(current_feat, dim=1)  # [B, C]
    #     cached_mean = torch.mean(cached_feat, dim=1)  # [B, C]
    #
    #     # 计算每个batch的余弦相似度
    #     similarities = F.cosine_similarity(current_mean, cached_mean, dim=1)  # [B]
    #
    #     # 返回需要更新的batch的布尔掩码
    #     return similarities < (1 - threshold)  # [B] 如果相似度低于阈值，则需要更新
    
    def forward(self, modalities, tau, len_zx, number):
        """
        输入: modality_features - 列表
        输出: 字典 {modality_name: tensor [B, 384, 768]}
        """
        num_modalities = len(modalities)
        batch_size = modalities[0].shape[0]

        # if svd_cache is None:
        #     svd_cache = {
        #         'search': [None] * 5,  # 为最多5个模态分别缓存搜索区域的SVD结果
        #     }
        
        # ============ 第1步：提取全局特征表示 ============
        # 对每个模态提取全局特征 [B, 384, 768] -> [B, 768]
        global_features = []
        for i in range(num_modalities):
        #     # 分离模板和搜索区域特征
            feat = modalities[i]
            cls_feat = feat[:, :1, :]
            in_template_feat = feat[:, 1:1+len_zx[0]//number, :]  # 前64个token是初始模板
            dy_template_feat = feat[:, 1+len_zx[0]//number:1+len_zx[0], :]  # 64-128个token是动态模板
            search_feat = feat[:, -len_zx[1]:, :]    # 后256个token是搜索区域

            # def fast_svd(matrix):
            #     # 使用更快速的SVD实现，只计算必要的奇异值
            #     try:
            #         U, S, V = torch.linalg.svd(matrix, full_matrices=False)
            #         return U, S, V
            #     except:
            #         # 如果SVD失败，返回None表示失败
            #         return None, None, None

            # # 训练模式下，每次都重新计算，不使用缓存
            # if self.training:
            #     # 训练模式：直接并行计算所有batch的重构特征，不使用缓存
            #     # 对整个search_feat进行SVD分解
            #     search_U, search_S, search_V = fast_svd(search_feat)
            #
            #     # 检查SVD是否成功
            #     if search_U is None:  # SVD失败，直接使用原始特征
            #         search_reconstructed = search_feat
            #     else:  # SVD成功，进行正常的重构过程
            #         # 基于奇异值能量自适应确定保留的奇异值数量
            #         search_S_squared = search_S ** 2  # [B, 256]
            #         search_total_energy = torch.sum(search_S_squared, dim=-1, keepdim=True)  # [B, 1]
            #         search_cumulative_energy = torch.cumsum(search_S_squared, dim=-1)  # [B, 256]
            #         search_energy_ratio = search_cumulative_energy / search_total_energy  # [B, 256]
            #
            #         # 找到覆盖90%能量的最小k值
            #         search_k_mask = search_energy_ratio < 0.9  # [B, 256]
            #         search_k = torch.sum(search_k_mask, dim=-1, keepdim=True).long() + 1  # [B, 1]
            #         search_k = torch.min(search_k, torch.ones_like(search_k) * search_S.size(-1)).squeeze(-1)  # [B]
            #
            #         # 初始化重构特征张量
            #         search_reconstructed = torch.zeros_like(search_feat)

                #     # 并行重构所有batch的搜索区域特征
                #     for b in range(search_feat.size(0)):
                #         search_k_b = search_k[b].item()
                #
                #         # 重构当前batch的搜索区域特征
                #         search_U_reduced = search_U[b:b + 1, :, :search_k_b]  # [1, 256, k]
                #         search_S_reduced = search_S[b:b + 1, :search_k_b]  # [1, k]
                #         search_V_reduced = search_V[b:b + 1, :search_k_b, :]  # [1, k, 768]
                #
                #         # 使用正确的重构公式：U @ diag(S) @ Vh
                #         # 由于torch.linalg.svd返回Vh（V的转置），所以我们不需要再转置search_V_reduced
                #         search_reconstructed_b = torch.bmm(
                #             search_U_reduced * search_S_reduced.unsqueeze(1),
                #             # [1, 256, k] .* [1, 1, k] = [1, 256, k]
                #             search_V_reduced  # [1, k, 768] - Vh不需要再转置
                #         )  # [1, 256, 768]
                #
                #         # 存储重构结果到输出张量
                #         search_reconstructed[b:b + 1] = search_reconstructed_b
                #
                #     # 训练模式下不更新缓存
                #     # 仍然为当前帧创建临时缓存项以保持结构一致性
                # svd_cache['search'][i] = [search_reconstructed.clone(), search_feat.clone()]
            # else:
            #     # 推理模式：使用缓存机制
            #     cached_search_data = svd_cache['search'][i]  # 获取第i个模态的缓存
            #     update_mask = self.should_update_svd(search_feat,
            #                                          cached_search_data[-1] if cached_search_data is not None else None,
            #                                          self.svd_cache_threshold)
            #
            #     # 初始化重构特征张量
            #     search_reconstructed = torch.zeros_like(search_feat)
            #
            #     # 先计算所有batch的重构结果
            #     for b in range(search_feat.size(0)):
            #         if update_mask[b]:  # 如果当前batch需要更新
            #             # 提取当前batch的特征
            #             current_batch_feat = search_feat[b:b + 1, :, :]  # [1, 256, 768]
            #
            #             # 对当前batch进行SVD分解
            #             search_U, search_S, search_V = fast_svd(current_batch_feat)

                        # # 检查SVD是否成功
                        # if search_U is None:  # SVD失败，直接使用原始特征
                        #     search_reconstructed_b = current_batch_feat
                        # else:  # SVD成功，进行正常的重构过程
                        #     # 基于奇异值能量自适应确定保留的奇异值数量
                        #     search_S_squared = search_S ** 2
                        #     search_total_energy = torch.sum(search_S_squared, dim=-1, keepdim=True)  # [1, 1]
                        #     search_cumulative_energy = torch.cumsum(search_S_squared, dim=-1)  # [1, 256]
                        #     search_energy_ratio = search_cumulative_energy / search_total_energy  # [1, 256]
                        #
                        #     # 找到覆盖90%能量的最小k值
                        #     search_k_mask = search_energy_ratio < 0.9  # [1, 256]
                        #     search_k = torch.sum(search_k_mask, dim=-1, keepdim=True).long() + 1  # [1, 1]
                        #     search_k = torch.min(search_k, torch.ones_like(search_k) * search_S.size(-1)).squeeze(-1)  # [1]
                        #     search_k_b = search_k[0].item()
                        #
                        #     # 重构当前batch的搜索区域特征
                        #     search_U_reduced = search_U[:, :, :search_k_b]  # [1, 256, k]
                        #     search_S_reduced = search_S[:, :search_k_b]  # [1, k]
                        #     search_V_reduced = search_V[:, :search_k_b, :]  # [1, k, 768]

                #             # 使用正确的重构公式：U @ diag(S) @ Vh
                #             # 由于torch.linalg.svd返回Vh（V的转置），所以我们不需要再转置search_V_reduced
                #             search_reconstructed_b = torch.bmm(
                #                 search_U_reduced * search_S_reduced.unsqueeze(1),  # [1, 256, k] .* [1, 1, k] = [1, 256, k]
                #                 search_V_reduced  # [1, k, 768] - Vh不需要再转置
                #             )  # [1, 256, 768]
                #
                #         # 存储重构结果到临时输出张量
                #         search_reconstructed[b:b + 1] = search_reconstructed_b
                #     else:
                #         # 使用缓存中的重构特征
                #         cached_reconstructed = cached_search_data[0][b:b + 1]  # [1, 256, 768]
                #         search_reconstructed[b:b + 1] = cached_reconstructed
                #
                # # 所有batch处理完成后，一次性更新缓存
                # if cached_search_data is not None:
                #     # 更新缓存中的重构特征和原始特征
                #     cached_search_data[0][:] = search_reconstructed.clone()
                #     cached_search_data[1][:] = search_feat.clone()
                # else:
                #     # 初始化缓存：[重构特征, 原始特征]
                #     svd_cache['search'][i] = [search_reconstructed.clone(), search_feat.clone()]
            
            # 动态门控加权融合
            cls_global = cls_feat.mean(dim=1)
            in_template_global = in_template_feat.mean(dim=1)  # [B, 768]
            dy_template_global = dy_template_feat.mean(dim=1)  # [B, 768]
            search_global = search_feat.mean(dim=1)      # [B, 768]
            mixed_global = feat.mean(dim=1)
            
            # 生成门控权重
            gate_input = torch.cat([cls_global, in_template_global, dy_template_global, search_global], dim=-1)
            gates = self.gate_generator(gate_input)  # [B, 4]
            # gates = self.gate_generator(cls_global)  # [B, 4]
            #
            # 加权融合：模板权重 + 搜索权重
            global_feat = (gates[:, 0:1] * in_template_global +
                          gates[:, 1:2] * dy_template_global +
                          gates[:, 2:3] * search_global)

            global_features.append(cls_global + global_feat)
            # in_template_global = torch.mean(in_template_feat, dim=1, keepdim=True)
            # dy_template_global = torch.mean(dy_template_feat, dim=1, keepdim=True)
            # search_global = torch.mean(search_feat, dim=1, keepdim=True)
            # # 而是保留 4 个独立的 token
            # global_features.append(search_feat)  # [B, 4, 768]
        
        # ============ 第2步：构建模态图 ============
        # 创建节点特征 [B, N, 768]    
        nodes = torch.stack(global_features, dim=1)  # [B, N, 768]
        num_nodes = nodes.shape[1]
        # ========== 2. 构建所有节点对特征（用于边预测和后续GAL） ==========
        # 生成 [B, N, N, 2D] 的边特征
        nodes_i = nodes.unsqueeze(2)  # [B, N, 1, D]
        nodes_j = nodes.unsqueeze(1)  # [B, 1, N, D]
        pair_feat = torch.cat([
            nodes_i.expand(-1, -1, num_nodes, -1),
            nodes_j.expand(-1, num_nodes, -1, -1)
        ], dim=-1)  # [B, N, N, 2D]

        # ========== 计算余弦相似度 [B, N, N] ==========
        nodes_normalized = F.normalize(nodes, p=2, dim=-1)  # [B, N, D]
        cosine_sim = torch.bmm(nodes_normalized, nodes_normalized.transpose(1, 2))  # [B, N, N]

        # 从 pair_feat 预测因果 logits
        logits = self.edge_predictor(pair_feat)  # [B, N, N, 1]
        logits = logits.squeeze(-1)  # [B, N, N]

        # Gumbel-Softmax 采样
        def gumbel_noise(shape, eps=1e-20, device=None):
            U = torch.rand(shape).to(device)
            return -torch.log(-torch.log(U + eps) + eps)

        noise = gumbel_noise(shape=(batch_size, num_nodes, num_nodes), device=logits.device)
        adj = torch.sigmoid((logits + noise) / tau)  # [B, N, N]
        adj = adj + cosine_sim

        # def dag_regularization(adj):
        #     B, N, _ = adj.shape
        #     A_sq = adj * adj
        #     loss = 0
        #     for b in range(B):
        #         exp_A = torch.matrix_exp(A_sq[b])
        #         loss += torch.trace(exp_A) - N
        #     return loss / B
        #
        # dag_loss = dag_regularization(adj)


        # ========== 3. 图注意力传播（使用 pair_feat 作为边特征） ==========
        node_features = nodes
        for layer in self.graph_layers:
            node_features, attn_weights = layer(
                node_features,
                edge_features=pair_feat,
                causal_adj=adj
            )

        # # ============ 第3步：计算动态邻接矩阵 ============
        # # 准备边特征（用于图注意力）
        # edge_features_list = []
        # for i in range(num_modalities):
        #     for j in range(num_modalities):
        #         # 创建边特征：连接两个节点特征
        #         edge_feat = torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
        #         edge_features_list.append(edge_feat)
        #
        # edge_features = torch.stack(edge_features_list, dim=1)  # [B, N*N, 2C]
        # edge_features = edge_features.view(batch_size, num_modalities, num_modalities, -1)  # [B, N, N, 2C]
        
        # # ============ 第4步：图注意力传播（使用邻接矩阵） ============
        # # 多层图注意力传播
        # node_features = nodes
        # for layer in self.graph_layers:
        #     node_features, attn_weights = layer(
        #         node_features,
        #         edge_features=edge_features
        #     )
        
        # ============ 第5步：融合原始特征和更新后的全局特征 ============
        updated_features = []
        for i in range(num_modalities):
            # 获取原始特征 [B, 385, 768]
            original_feat = modalities[i]

            updated_global = node_features[:, i]  # [B, 768]

            # 将全局特征扩展到每个token
            global_expanded = updated_global.unsqueeze(1)  # [B, 1, 768]
            global_expanded = global_expanded.expand(-1, 385, -1)  # [B, 384, 768]

            # 融合原始特征和全局信息
            fusion_input = torch.cat([original_feat, global_expanded], dim=-1)
            fused_feat = self.fusion_mlp(fusion_input)  # [B, 384, 768]

            updated_feat = original_feat + fused_feat

            updated_features.append(updated_feat)

            # # 获取更新后的搜索区域特征 [B, L_search, D]
            # # node_features: [B, N*L_search, D]，按模态顺序排列
            # updated_search = node_features[:, i * len_zx[1]:(i + 1) * len_zx[1], :]  # [B, L_search, D]
            # updated_search = updated_search + global_features[i]
            # # 从原始特征中分离出模板区域（保持不变）
            # template_feat = original_feat[:, :-len_zx[1], :]  # [B, 1+len_zx[0], D]
            # # 将更新的搜索特征与模板特征拼接
            # updated_feat = torch.cat([template_feat, updated_search], dim=1)  # [B, L_total, D]

            # # 获取更新后的全局节点特征 [B, 768]
            # updated_global = node_features[:, i*4:i*4+global_features[i].shape[1],:]  # [B, 768]
            #
            # # 分离 4 种类型的 token
            # updated_cls = updated_global[:, 0:1, :]  # [B, 1, 768]
            # updated_in_template = updated_global[:, 1:2, :]  # [B, 1, 768]
            # updated_dy_template = updated_global[:, 2:3, :]  # [B, 1, 768]
            # updated_search = updated_global[:, 3:4, :]  # [B, 1, 768]
            #
            # # 根据原始特征中各区域的长度，将每种类型的 token 扩展到对应长度
            # # cls token 保持长度为 1
            # # 初始模板扩展回原长度
            # updated_in_template_expanded = updated_in_template.expand(-1, len_zx[0] // number,
            #                                                           -1)  # [B, len_zx[0]//number, 768]
            # # 动态模板扩展回原长度
            # updated_dy_template_expanded = updated_dy_template.expand(-1, len_zx[0] - len_zx[0] // number,
            #                                                           -1)  # [B, len_zx[0]-len_zx[0]//number, 768]
            # # 搜索区域扩展回原长度
            # updated_search_expanded = updated_search.expand(-1, len_zx[1], -1)  # [B, len_zx[1], 768]
            #
            # # 按照原始顺序拼接所有扩展后的特征
            # global_expanded = torch.cat([
            #     updated_cls,  # [B, 1, 768]
            #     updated_in_template_expanded,  # [B, len_zx[0]//number, 768]
            #     updated_dy_template_expanded,  # [B, len_zx[0]-len_zx[0]//number, 768]
            #     updated_search_expanded  # [B, len_zx[1], 768]
            # ], dim=1)  # [B, 385, 768]
            
            # # 融合原始特征和全局信息
            # fusion_input = torch.cat([original_feat, global_expanded], dim=-1)
            # fused_feat = self.fusion_mlp(fusion_input)  # [B, 385, 768]

            # updated_features.append(updated_feat)
        
        return updated_features, adj
