import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
import numpy as np


class InnerProductDecoder(Module):

    def __init__(self, input_dim, num_dis, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(input_dim, input_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.dropout = dropout
        self.num_dis = num_dis

    def forward(self, inputs, association=None):
        inputs = F.dropout(inputs, self.dropout, training=self.training)
        D = inputs[0 : self.num_dis, :]
        M = inputs[self.num_dis :, :]
        D = torch.mm(D, self.weight)
        D = torch.t(D)
        x = torch.mm(M, D)
        return x


class FoldCalibratedDecoder(Module):
    """Combine bilinear scores with fold-safe geometry and degree priors.

    The residual channels are initialized at zero, so optimization starts from
    the established bilinear decoder. ``association`` contains training links
    only and therefore cannot expose held-out edges.
    """

    def __init__(self, input_dim, num_dis, dropout=0.1):
        super().__init__()
        self.base = InnerProductDecoder(input_dim, num_dis, dropout=dropout)
        self.num_dis = num_dis
        self.dis_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.meta_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.cosine_scale = nn.Parameter(torch.zeros(()))
        self.degree_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _standardize(values):
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

    def forward(self, inputs, association=None):
        if association is None:
            raise ValueError("association is required by FoldCalibratedDecoder")

        base_score = self.base(inputs)
        diseases = F.normalize(
            self.dis_projection(inputs[: self.num_dis]), dim=-1
        )
        metabolites = F.normalize(
            self.meta_projection(inputs[self.num_dis :]), dim=-1
        )
        cosine_score = torch.matmul(metabolites, diseases.t())

        association = association.to(device=inputs.device, dtype=inputs.dtype)
        meta_degree = self._standardize(torch.log1p(association.sum(dim=1)))
        dis_degree = self._standardize(torch.log1p(association.sum(dim=0)))
        degree_prior = meta_degree.unsqueeze(1) + dis_degree.unsqueeze(0)

        return (
            base_score
            + self.cosine_scale * cosine_score
            + self.degree_scale * degree_prior
        )




class DynamicGraphLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        k_neighbors=10,
        dropout=0.1,
        use_static_graph=False,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.use_static_graph = use_static_graph

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
        if use_static_graph:
            # Keep the established dynamic-path initialization unchanged.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(32452843)
                self.static_conv = GCNConv(in_channels, out_channels)
            self.static_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x, edge_index):
        # --- 1. 维度兼容处理 ---
        if x.dim() == 3:
            x = x.squeeze(0)  # 处理 [1, N, D] -> [N, D]
        N = x.size(0)

        # --- 2. 特征变换 ---
        x_l = self.lin_l(x)  # [N, D]
        x_r = self.lin_r(x)  # [N, D]

        # --- 3. 计算相似度 ---
        x_r_norm = F.normalize(x_r, dim=1)
        similarity_matrix = torch.matmul(x_r_norm, x_r_norm.t())  # [N, N]

        # --- 4. Top-K 选择 ---
        k = min(self.k_neighbors, N)
        # top_k_idx 存储的是“邻居节点的 ID”
        top_k_val, top_k_idx = torch.topk(similarity_matrix, k=k, dim=-1)  # [N, k]

        # --- 5. 【核心修复】特征聚合 ---
        # 你的正确写法：直接通过 ID 索引提取特征向量
        # x_l[top_k_idx] -> [N, k, D]
        neighbor_features = x_l[top_k_idx]

        # 权重处理 [N, k, 1]
        weights = F.softmax(top_k_val, dim=-1).unsqueeze(-1)

        # 加权求和 [N, D]
        out = torch.sum(neighbor_features * weights, dim=1)

        # --- 6. 门控与输出 ---
        out = out + x_l
        gate_val = self.gate(torch.cat([x_l, out], dim=-1))
        out = gate_val * out + (1 - gate_val) * x_l

        if self.use_static_graph:
            static_out = self.static_conv(x, edge_index)
            out = out + torch.tanh(self.static_scale) * static_out

        return self.out_proj(out)










class ThermalEchoMambaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, K=11, use_sparse=False, diffusion_rate=0.2):
        """
        基于流形热扩散回波的 Mamba 编码器
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param K: 模拟的时间步数 (相当于序列长度)，K在我的超参数中变成了T
        :param use_sparse: 是否使用稀疏矩阵运算
        :param diffusion_rate: 热扩散系数 (物理方程中的 dt)，通常 0.1~0.5
        """
        super().__init__()
        self.K = K
        self.use_sparse = use_sparse
        self.diffusion_rate = diffusion_rate

        # 1. 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. 时间位置编码 (可学习参数)
        # 告诉 Mamba 当前是第几秒的扩散状态
        self.time_encoding = nn.Parameter(torch.randn(1, K + 1, hidden_dim) * 0.02)

        # 3. Mamba 模块 (处理时序序列)
        # 注意：这里我用了 SimpleMamba 以防你环境没配好，如有需要可换回官方 Mamba
        self.mamba = Mamba(d_model=hidden_dim)

        # 4. 层归一化 (用于稳定扩散过程)
        self.ln = nn.LayerNorm(hidden_dim, dtype=torch.float32)

        # 5. 输出聚合 (注意力机制)
        self.agg_attention = nn.Linear(hidden_dim, 1)

    def forward(self, adj, features):
        """
        :param adj: 邻接矩阵，形状 (N, N)，可以是稠密也可以是稀疏
        :param features: 节点特征，形状 (N, input_dim)
        :return: 节点嵌入，形状 (N, hidden_dim)
        """
        N, d_in = features.shape
        device = features.device

        # --- 第一步：预处理度矩阵 (用于归一化拉普拉斯算子) ---
        # 计算 D^(-1/2)
        if self.use_sparse:
            deg = torch.sparse.sum(adj, dim=1).to_dense()
        else:
            deg = adj.sum(dim=-1)

        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.  # 处理孤立点

        # --- 第二步：热扩散模拟 (生成序列) ---
        # 初始状态 u(0) = 投影后的特征
        u_t = self.input_proj(features)  # [N, hidden_dim]
        u_t = self.ln(u_t)


        # 列表用于存储每个时间步的热分布快照
        sequence_list = [u_t]

        for k in range(1, self.K + 1):
            # 模拟热传导方程: du/dt = -L * u
            # 离散化近似: u(t+1) = u(t) - rate * (I - D^-1/2 A D^-1/2) * u(t)

            # 1. 计算对称归一化传播: D^-1/2 * A * D^-1/2 * u
            u_norm = u_t * deg_inv_sqrt.view(-1, 1)  # 左乘 D^-1/2

            if self.use_sparse:
                u_prop = torch.sparse.mm(adj, u_norm)
            else:
                u_prop = torch.matmul(adj, u_norm)

            u_prop = u_prop * deg_inv_sqrt.view(-1, 1)  # 右乘 D^-1/2

            # 2. 计算拉普拉斯算子作用: L * u = u - u_prop
            laplacian_u = u_t - u_prop
            # 3. 更新状态 (显式欧拉法)
            u_t = (u_t - self.diffusion_rate * laplacian_u).float()
            # 4. 归一化并记录
            u_t = self.ln(u_t)
            sequence_list.append(u_t)

        # 堆叠成序列: [N, K+1, hidden_dim]
        thermal_sequence = torch.stack(sequence_list, dim=1)

        # --- 第三步：Mamba 处理 ---
        # 加入时间位置编码
        thermal_sequence = thermal_sequence + self.time_encoding

        # 输入 Mamba 捕捉时序依赖 (回波特征)
        out_seq = self.mamba(thermal_sequence)  # [N, K+1, hidden_dim]

        # --- 第四步：注意力聚合 ---
        logits = self.agg_attention(out_seq)  # [N, K+1, 1]
        weights = torch.softmax(logits, dim=1)
        out = torch.sum(weights * out_seq, dim=1)  # [N, hidden_dim]

        return out


class DiffusionWaveletTransformerEncoder(nn.Module):
    """Encode a node through a sequence of graph-diffusion scales.

    Each token combines a low-pass diffusion state with the band-pass change
    introduced at that scale.  Attention is applied across scales for each
    node, so memory grows with ``num_nodes * hops`` rather than
    ``num_nodes ** 2``.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        K=9,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        diffusion_rate=0.2,
        use_sparse=False,
        liquid_adapter=False,
        liquid_dropout=0.2,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.K = K
        self.diffusion_rate = diffusion_rate
        self.use_sparse = use_sparse
        self.liquid_adapter = liquid_adapter

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.wavelet_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale_encoding = nn.Parameter(
            torch.zeros(1, K + 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.scale_encoding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if liquid_adapter:
            # Restore the RNG afterwards so the fusion/decoder initialization
            # remains identical to the non-adapter baseline.
            with torch.random.fork_rng(devices=[]):
                self.liquid_cell = LiquidTimeConstantCell(
                    hidden_dim, hidden_dim, dropout=liquid_dropout
                )
                self.liquid_log_intervals = nn.Parameter(torch.zeros(K + 1))
                self.liquid_projection = nn.Linear(
                    hidden_dim, hidden_dim, bias=False
                )
                self.liquid_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                self.liquid_gate_bias = nn.Parameter(
                    torch.full((hidden_dim,), -2.0)
                )
                nn.init.zeros_(self.liquid_projection.weight)
                nn.init.zeros_(self.liquid_gate.weight)
                nn.init.zeros_(self.liquid_gate.bias)

    def _propagate(self, adj, x, degree_scale):
        x_scaled = x * degree_scale
        if self.use_sparse:
            propagated = torch.sparse.mm(adj, x_scaled)
        else:
            propagated = torch.matmul(adj, x_scaled)
        return propagated * degree_scale

    def forward(self, adj, features):
        adj = adj.to(dtype=features.dtype)
        degree = (
            torch.sparse.sum(adj, dim=1).to_dense()
            if self.use_sparse
            else adj.sum(dim=-1)
        )
        degree_scale = degree.clamp_min(1.0).pow(-0.5).unsqueeze(-1)

        state = self.input_proj(features)
        state = F.layer_norm(state, (state.size(-1),))
        zero_band = torch.zeros_like(state)
        tokens = [self.wavelet_proj(torch.cat([state, zero_band], dim=-1))]

        for _ in range(self.K):
            low_pass = self._propagate(adj, state, degree_scale)
            next_state = state + self.diffusion_rate * (low_pass - state)
            band_pass = next_state - state
            tokens.append(
                self.wavelet_proj(torch.cat([next_state, band_pass], dim=-1))
            )
            state = next_state

        sequence = torch.stack(tokens, dim=1)
        sequence = self.dropout(sequence + self.scale_encoding)
        if self.liquid_adapter:
            intervals = (
                F.softplus(self.liquid_log_intervals).view(-1, 1, 1) + 1e-3
            )
            hidden = torch.zeros_like(sequence[:, 0])
            liquid_states = []
            for step in range(self.K + 1):
                hidden = self.liquid_cell(
                    sequence[:, step], hidden, intervals[step]
                )
                liquid_states.append(hidden)
            liquid_sequence = torch.stack(liquid_states, dim=1)
            gate = torch.sigmoid(
                self.liquid_gate(
                    torch.cat([sequence, liquid_sequence], dim=-1)
                )
                + self.liquid_gate_bias
            )
            sequence = sequence + gate * self.liquid_projection(liquid_sequence)
        encoded = self.encoder(sequence)

        # Node-conditioned pooling makes the useful diffusion radius adaptive.
        query = self.query_proj(encoded[:, 0]).unsqueeze(1)
        logits = self.pool_score(torch.tanh(encoded + query))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights * encoded, dim=1)
        return self.output_norm(pooled + encoded[:, 0])


class LiquidTimeConstantCell(nn.Module):
    """Closed-form update for an input-conditioned liquid time constant.

    The update is the exact constant-input solution of
    ``tau * dh/dt = candidate - h`` over one learned path interval.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.time_constant = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, delta_t):
        context = torch.cat([x, hidden], dim=-1)
        candidate = torch.tanh(self.candidate(context))
        tau = self.time_constant(context).clamp_min(1e-3)
        decay = torch.exp(-delta_t / tau)
        next_hidden = decay * hidden + (1.0 - decay) * candidate
        return self.norm(hidden + self.dropout(next_hidden - hidden))


class MetaPathLiquidEncoder(nn.Module):
    """Encode alternating disease-metabolite meta-path trajectories.

    For every fold, ``association`` contains training links only. Starting
    from node features, states alternate across the bipartite relation:
    D<-M / M<-D, D<-M<-D / M<-D<-M, and so on. A liquid cell then models
    how each node representation changes with meta-path length.
    """

    def __init__(self, input_dim, hidden_dim, num_steps=4, dropout=0.1):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        self.num_steps = num_steps
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.path_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.path_encoding = nn.Parameter(
            torch.zeros(1, num_steps + 1, hidden_dim)
        )
        self.log_intervals = nn.Parameter(torch.zeros(num_steps + 1))
        nn.init.trunc_normal_(self.path_encoding, std=0.02)

        self.cell = LiquidTimeConstantCell(
            hidden_dim, hidden_dim, dropout=dropout
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize_association(association):
        association = association.float()
        meta_scale = association.sum(dim=1).clamp_min(1.0).pow(-0.5)
        dis_scale = association.sum(dim=0).clamp_min(1.0).pow(-0.5)
        return (
            meta_scale.unsqueeze(1)
            * association
            * dis_scale.unsqueeze(0)
        )

    def forward(self, association, features):
        association = association.to(device=features.device, dtype=features.dtype)
        transition = self._normalize_association(association)
        num_dis = association.size(1)

        projected = self.input_proj(features)
        dis_state = F.layer_norm(projected[:num_dis], (projected.size(-1),))
        meta_state = F.layer_norm(projected[num_dis:], (projected.size(-1),))
        zero_band = torch.zeros_like(projected)
        tokens = [self.path_proj(torch.cat([projected, zero_band], dim=-1))]

        for _ in range(self.num_steps):
            previous_dis = dis_state
            previous_meta = meta_state
            dis_state = torch.matmul(transition.t(), previous_meta)
            meta_state = torch.matmul(transition, previous_dis)
            dis_state = F.layer_norm(dis_state, (dis_state.size(-1),))
            meta_state = F.layer_norm(meta_state, (meta_state.size(-1),))
            state = torch.cat([dis_state, meta_state], dim=0)
            previous = torch.cat([previous_dis, previous_meta], dim=0)
            tokens.append(
                self.path_proj(torch.cat([state, state - previous], dim=-1))
            )

        sequence = torch.stack(tokens, dim=1)
        sequence = self.dropout(sequence + self.path_encoding)
        intervals = F.softplus(self.log_intervals).view(-1, 1, 1) + 1e-3

        hidden = torch.zeros_like(sequence[:, 0])
        hidden_states = []
        for step in range(self.num_steps + 1):
            hidden = self.cell(sequence[:, step], hidden, intervals[step])
            hidden_states.append(hidden)
        encoded = torch.stack(hidden_states, dim=1)

        query = self.query_proj(encoded[:, 0]).unsqueeze(1)
        weights = torch.softmax(
            self.pool_score(torch.tanh(encoded + query)), dim=1
        )
        pooled = torch.sum(weights * encoded, dim=1)
        return self.output_norm(pooled + encoded[:, 0])


class DiffusionWaveletLiquidEncoder(nn.Module):
    """Apply a liquid time-constant model to graph diffusion scales."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        K=9,
        dropout=0.2,
        diffusion_rate=0.2,
        use_sparse=False,
    ):
        super().__init__()
        self.K = K
        self.diffusion_rate = diffusion_rate
        self.use_sparse = use_sparse
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.wavelet_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale_encoding = nn.Parameter(torch.zeros(1, K + 1, hidden_dim))
        self.log_intervals = nn.Parameter(torch.zeros(K + 1))
        nn.init.trunc_normal_(self.scale_encoding, std=0.02)

        self.cell = LiquidTimeConstantCell(
            hidden_dim, hidden_dim, dropout=dropout
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _propagate(self, adj, x, degree_scale):
        x_scaled = x * degree_scale
        if self.use_sparse:
            propagated = torch.sparse.mm(adj, x_scaled)
        else:
            propagated = torch.matmul(adj, x_scaled)
        return propagated * degree_scale

    def forward(self, adj, features):
        adj = adj.to(dtype=features.dtype)
        degree = (
            torch.sparse.sum(adj, dim=1).to_dense()
            if self.use_sparse
            else adj.sum(dim=-1)
        )
        degree_scale = degree.clamp_min(1.0).pow(-0.5).unsqueeze(-1)

        state = self.input_proj(features)
        state = F.layer_norm(state, (state.size(-1),))
        zero_band = torch.zeros_like(state)
        tokens = [self.wavelet_proj(torch.cat([state, zero_band], dim=-1))]
        for _ in range(self.K):
            low_pass = self._propagate(adj, state, degree_scale)
            next_state = state + self.diffusion_rate * (low_pass - state)
            band_pass = next_state - state
            tokens.append(
                self.wavelet_proj(torch.cat([next_state, band_pass], dim=-1))
            )
            state = next_state

        sequence = torch.stack(tokens, dim=1)
        sequence = self.dropout(sequence + self.scale_encoding)
        intervals = F.softplus(self.log_intervals).view(-1, 1, 1) + 1e-3
        hidden = torch.zeros_like(sequence[:, 0])
        hidden_states = []
        for step in range(self.K + 1):
            hidden = self.cell(sequence[:, step], hidden, intervals[step])
            hidden_states.append(hidden)
        encoded = torch.stack(hidden_states, dim=1)

        query = self.query_proj(encoded[:, 0]).unsqueeze(1)
        weights = torch.softmax(
            self.pool_score(torch.tanh(encoded + query)), dim=1
        )
        pooled = torch.sum(weights * encoded, dim=1)
        return self.output_norm(pooled + encoded[:, 0])


class SparseTokenRouterFusion(nn.Module):
    """Fuse structural and attribute views with adaptive sparse token routing.

    Unlike the previous genetic search, routing scores are optimized directly
    by the link-prediction loss.  A straight-through top-k mask gives discrete
    token selection in the forward pass while retaining gradients.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4,
        dropout=0.1,
        num_blocks=8,
        keep_ratio=0.5,
    ):
        super().__init__()
        if input_dim % num_blocks != 0:
            raise ValueError("input_dim must be divisible by num_blocks")
        self.num_blocks = num_blocks
        self.block_dim = input_dim // num_blocks
        self.num_tokens = num_blocks * 2
        self.keep_tokens = max(2, int(round(self.num_tokens * keep_ratio)))

        self.token_proj = nn.Linear(self.block_dim, input_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, input_dim))
        self.block_embedding = nn.Parameter(
            torch.zeros(1, num_blocks, input_dim)
        )
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.block_embedding, std=0.02)

        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(input_dim)
        )
        self.pool = nn.Linear(input_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _tokenize(self, x, modality):
        blocks = x.reshape(x.size(0), self.num_blocks, self.block_dim)
        tokens = self.token_proj(blocks) + self.block_embedding
        return tokens + self.modality_embedding[modality]

    def forward(self, h_structure, h_attribute):
        structure_tokens = self._tokenize(h_structure, 0)
        attribute_tokens = self._tokenize(h_attribute, 1)
        tokens = torch.cat([structure_tokens, attribute_tokens], dim=1)

        scores = self.router(tokens).squeeze(-1)
        probabilities = torch.softmax(scores, dim=-1)
        top_indices = scores.topk(self.keep_tokens, dim=-1).indices
        hard_mask = torch.zeros_like(scores).scatter_(1, top_indices, 1.0)
        route_weight = hard_mask + probabilities - probabilities.detach()
        routed = tokens * route_weight.unsqueeze(-1)

        encoded = self.fusion_encoder(self.dropout(routed))
        pool_logits = self.pool(encoded).squeeze(-1)
        pool_logits = pool_logits.masked_fill(hard_mask == 0, -1e4)
        pool_weight = torch.softmax(pool_logits, dim=-1).unsqueeze(-1)
        fused = torch.sum(pool_weight * encoded, dim=1)
        return self.output(fused)





class GABlockFusion(nn.Module):
    """
    基于遗传算法的特征块选择融合
    在 UnifiedFusion 基础上，引入进化机制动态选择特征子空间进行交互
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            num_heads=4,
            dropout=0.1,
            num_blocks=8,  # 将特征切分为多少个块 (对应染色体长度)
            population_size=5,  # 种群大小 (进化搜索的候选方案数)
            selection_ratio=0.5  # 每次迭代保留前百分之多少的精英
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.pop_size = population_size
        self.block_size = input_dim // num_blocks

        # 1. 基础组件 (保留原框架)
        self.proj = nn.Linear(input_dim, output_dim)
        self.cross_attn_s2a = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_a2s = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(input_dim)

        # 2. 遗传算法组件
        # 初始化种群：每个个体是一个二进制向量，代表哪些块被选中
        # Shape: [population_size, num_blocks]
        self.population = torch.randint(0, 2, (population_size, num_blocks))

        # 适应度评分缓存
        self.fitness_scores = torch.zeros(population_size)

        # 可学习的块权重 (用于软选择，辅助硬选择的GA)
        self.block_weights = nn.Parameter(torch.ones(num_blocks))

    def _mask_features(self, x, mask):
        """
        根据掩码屏蔽特征块
        x: [Batch, Dim]
        mask: [num_blocks] (Binary)
        """
        # 将 mask 扩展为 [Dim]
        # 例如 mask=[1,0,1,0] -> [1,1,0,0,1,1,0,0] (假设块大小为2)
        mask_expanded = mask.repeat_interleave(self.block_size).to(x.device)
        return x * mask_expanded

    def _calculate_fitness(self, h_s, h_a, mask):
        """
        简化版适应度计算：基于重构误差或特征方差
        实际训练中，这通常由主任务的 Loss 反向传播决定，
        这里为了演示，我们使用特征的相关性作为启发式适应度
        """
        # 1. 应用掩码
        s_masked = self._mask_features(h_s, mask)
        a_masked = self._mask_features(h_a, mask)

        # 2. 计算交互后的能量 (简化模拟)
        # 这里仅仅计算被选中特征的方差，方差大说明信息量大
        # 实际应替换为下游任务的 Loss
        energy = torch.var(s_masked + a_masked, dim=0).sum()
        return energy

    def _evolve(self, h_s, h_a):
        """
        执行一代遗传进化
        """
        new_population = []

        # 1. 评估适应度
        for i in range(self.pop_size):
            mask = self.population[i]
            score = self._calculate_fitness(h_s, h_a, mask)
            self.fitness_scores[i] = score

        # 2. 选择 (Selection) - 保留前 K 个精英
        top_k = int(self.pop_size * 0.2)
        _, indices = torch.topk(self.fitness_scores, top_k)
        elites = self.population[indices]
        new_population.append(elites)

        # 3. 交叉与变异 (Crossover & Mutation) - 填充剩余种群
        while len(new_population) < self.pop_size:
            # 随机选两个父代
            p1 = elites[torch.randint(0, top_k, (1,))]
            p2 = elites[torch.randint(0, top_k, (1,))]

            # 交叉
            child = p1.clone()
            # 随机交换一半基因
            crossover_point = self.num_blocks // 2
            child[:, crossover_point:] = p2[:, crossover_point:]

            # 变异 (10%概率翻转)
            mutation_mask = torch.rand(1, self.num_blocks) < 0.1
            child ^= mutation_mask.long()

            new_population.append(child)

        # 更新种群
        self.population = torch.cat(new_population, dim=0)

    def forward(self, h_s, h_a):
        # h_s, h_a: [Batch, Dim]

        # 1. 进化搜索最佳特征块组合
        # 注意：在实际训练中，这一步可能不需要每一步都做，可以每隔几步做一次
        self._evolve(h_s, h_a)

        # 2. 获取当前最优个体 (Best Chromosome)
        best_mask = self.population[0]  # 假设第一个是适应度最高的

        # 3. 特征块选择与融合
        # 只使用被选中的特征块进行注意力计算
        h_s_selected = self._mask_features(h_s, best_mask)
        h_a_selected = self._mask_features(h_a, best_mask)

        # 4. 执行原框架的交叉注意力
        qs = h_s_selected.unsqueeze(1)
        ka = h_a_selected.unsqueeze(1)
        qa = h_a_selected.unsqueeze(1)
        ks = h_s_selected.unsqueeze(1)

        out_s, _ = self.cross_attn_s2a(qs, ka, ka)
        out_a, _ = self.cross_attn_a2s(qa, ks, ks)

        out_s = out_s.squeeze(1)
        out_a = out_a.squeeze(1)

        # 5. 融合
        out = (out_s + out_a) / 2

        return out


class GaijinGABlockFusion(nn.Module):
        """
        遗传算法引导的多模态特征块选择 + 跨注意力融合
        创新：GA 自动搜索最优特征子空间，用于跨模态注意力融合
        """

        def __init__(
                self,
                input_dim,
                output_dim,
                num_heads=4,
                dropout=0.1,
                num_blocks=8,  # 染色体长度
                population_size=10,  # 种群数量
                evolve_steps=10,  # 每10步进化一次
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_blocks = num_blocks
            self.pop_size = population_size
            self.block_size = input_dim // num_blocks
            self.evolve_steps = evolve_steps
            self.register_buffer('step_counter', torch.tensor(0))

            # 跨注意力
            self.cross_attn_s2a = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)
            self.cross_attn_a2s = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)

            # 归一化 & 残差
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
            self.dropout = nn.Dropout(dropout)

            # 遗传算法种群
            self.register_buffer('population', torch.randint(0, 2, (population_size, num_blocks)))
            self.register_buffer('best_mask', torch.randint(0, 2, (num_blocks,)))

            # 软选择权重（让GA不破坏梯度流）
            self.gate = nn.Parameter(torch.ones(num_blocks))

            # 投影
            self.proj = nn.Linear(input_dim, output_dim)

        def _get_mask_weight(self, mask):
            """ 二进制mask → 软权重 """
            return mask.repeat_interleave(self.block_size).unsqueeze(0)

        def _evaluate_population(self, h_s, h_a):
            """ 评估种群适应度：基于多模态特征相似度 """
            fitness = []
            for mask in self.population:
                w = self._get_mask_weight(mask)
                s = h_s * w
                a = h_a * w
                sim = torch.cosine_similarity(s, a, dim=-1).mean()
                fitness.append(sim.item())
            return torch.tensor(fitness)

        def _evolve(self, h_s, h_a):
            """ GA 核心：选择 + 交叉 + 变异 """
            fitness = self._evaluate_population(h_s, h_a)
            best_idx = fitness.argmax()
            self.best_mask = self.population[best_idx].clone()

            # 精英保留
            elite_num = max(1, self.pop_size // 5)
            _, idx = torch.topk(fitness, elite_num)
            elites = self.population[idx]

            new_pop = [elites]
            while len(new_pop) < self.pop_size:
                p1 = elites[torch.randint(len(elites), ())]
                p2 = elites[torch.randint(len(elites), ())]

                # 单点交叉
                point = self.num_blocks // 2
                child = p1.clone()
                child[point:] = p2[point:]

                # 变异

                mutate = torch.rand(self.num_blocks) < 0.1

                #child = torch.where(mutate, 1 - child, child)
                child = torch.where(mutate.to(child.device), 1 - child, child)
                new_pop.append(child.unsqueeze(0))

            self.population = torch.cat(new_pop, dim=0)

        def forward(self, h_s, h_a):
            B, C = h_s.shape

            # 每 N 步进化一次
            if self.training and self.step_counter % self.evolve_steps == 0:
                self._evolve(h_s, h_a)
            self.step_counter += 1

            # 使用最优mask进行软选择
            w = self._get_mask_weight(self.best_mask)
            h_s_masked = h_s * w
            h_a_masked = h_a * w

            # 跨注意力
            qs = h_s_masked.unsqueeze(1)
            ka = h_a_masked.unsqueeze(1)
            qa = h_a_masked.unsqueeze(1)
            ks = h_s_masked.unsqueeze(1)

            s_att, _ = self.cross_attn_s2a(qs, ka, ka)
            a_att, _ = self.cross_attn_a2s(qa, ks, ks)

            s_att = s_att.squeeze(1)
            a_att = a_att.squeeze(1)

            # 残差
            s_out = self.norm1(h_s + self.dropout(s_att))
            a_out = self.norm1(h_a + self.dropout(a_att))

            # 融合
            out = (s_out + a_out) / 2
            out = self.norm2(out)
            return self.proj(out)




class GABlockFusionSingle(nn.Module):
    """
    遗传算法引导的特征块选择 + 单向跨注意力融合
    改进点：
    1. 去掉双向交互，只保留单向 cross-attention
    2. 注意力不再作用于长度为1的伪序列，而是作用于 block tokens
    3. GA 的 best mask 与可学习 soft gate 结合
    4. 修复 population 更新逻辑
    5. 避免全0 mask
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4,
        dropout=0.1,
        num_blocks=8,
        population_size=10,
        evolve_steps=10,
        mutation_rate=0.1,
    ):
        super().__init__()

        assert input_dim % num_blocks == 0, \
            f"input_dim={input_dim} 必须能被 num_blocks={num_blocks} 整除"
        block_size = input_dim // num_blocks
        assert block_size % num_heads == 0, \
            f"每个 block 的维度 block_size={block_size} 必须能被 num_heads={num_heads} 整除"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pop_size = population_size
        self.evolve_steps = evolve_steps
        self.mutation_rate = mutation_rate

        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

        # 单向跨注意力：以 h_s 为 query，以 h_a 为 key/value
        # 注意这里 embed_dim 是 block_size，因为我们把每个 block 当成一个 token
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=block_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输入归一化
        self.norm_s = nn.LayerNorm(input_dim)
        self.norm_a = nn.LayerNorm(input_dim)

        # 残差后归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # FFN，增强融合能力
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # 遗传算法种群
        population = torch.randint(0, 2, (population_size, num_blocks))
        population = self._fix_zero_masks(population)
        best_mask = population[0].clone()

        self.register_buffer("population", population)
        self.register_buffer("best_mask", best_mask)

        # 可学习软门控：让 GA 的硬选择之外，还有连续可学习权重
        self.gate = nn.Parameter(torch.ones(num_blocks))

        # 输出投影
        self.proj = nn.Linear(input_dim, output_dim)

    def _fix_zero_masks(self, masks):
        """
        避免出现全0染色体
        masks: [P, num_blocks] or [num_blocks]
        """
        if masks.dim() == 1:
            if masks.sum() == 0:
                idx = torch.randint(0, self.num_blocks, (1,), device=masks.device)
                masks[idx] = 1
            return masks

        zero_rows = masks.sum(dim=1) == 0
        if zero_rows.any():
            row_ids = zero_rows.nonzero(as_tuple=False).flatten()
            col_ids = torch.randint(0, self.num_blocks, (len(row_ids),), device=masks.device)
            masks[row_ids, col_ids] = 1
        return masks

    def _hard_block_weight(self, mask):
        """
        仅用于 GA 适应度评估的硬 mask
        mask: [num_blocks]
        return: [1, num_blocks, 1]
        """
        return mask.float().view(1, self.num_blocks, 1)

    def _soft_block_weight(self, mask):
        """
        最终前向传播使用：硬 mask * 可学习 soft gate
        mask: [num_blocks]
        return: [1, num_blocks, 1]
        """
        soft_gate = torch.sigmoid(self.gate)              # [num_blocks]
        weight = mask.float() * soft_gate                 # [num_blocks]
        return weight.view(1, self.num_blocks, 1)

    def _split_blocks(self, x):
        """
        x: [B, C] -> [B, num_blocks, block_size]
        """
        B, C = x.shape
        return x.view(B, self.num_blocks, self.block_size)

    @torch.no_grad()
    def _evaluate_population(self, h_s, h_a):
        """
        评估每个染色体的适应度
        思路：选择后的两个模态在 block 空间里越相似，fitness 越高
        h_s, h_a: [B, C]
        return: [pop_size]
        """
        s_blk = self._split_blocks(h_s)   # [B, M, D]
        a_blk = self._split_blocks(h_a)   # [B, M, D]

        fitness = []
        for mask in self.population:
            w = self._hard_block_weight(mask)             # [1, M, 1]
            s = (s_blk * w).reshape(h_s.size(0), -1)      # [B, C]
            a = (a_blk * w).reshape(h_a.size(0), -1)      # [B, C]

            sim = F.cosine_similarity(s, a, dim=-1, eps=1e-8).mean()
            fitness.append(sim)

        return torch.stack(fitness, dim=0)                # [pop_size]

    @torch.no_grad()
    def _evolve(self, h_s, h_a):
        """
        GA: 选择 + 交叉 + 变异
        """
        fitness = self._evaluate_population(h_s, h_a)     # [pop_size]

        # 更新 best mask
        best_idx = fitness.argmax()
        self.best_mask.copy_(self.population[best_idx])

        # 精英保留
        elite_num = max(2, self.pop_size // 4)
        elite_num = min(elite_num, self.pop_size)

        elite_idx = torch.topk(fitness, elite_num).indices
        elites = self.population[elite_idx]               # [elite_num, num_blocks]

        new_population = [e.clone() for e in elites]

        while len(new_population) < self.pop_size:
            p1 = elites[torch.randint(0, elite_num, (1,), device=elites.device).item()]
            p2 = elites[torch.randint(0, elite_num, (1,), device=elites.device).item()]

            # 单点交叉
            if self.num_blocks > 1:
                point = torch.randint(1, self.num_blocks, (1,), device=elites.device).item()
                child = torch.cat([p1[:point], p2[point:]], dim=0)
            else:
                child = p1.clone()

            # 变异
            mutate = (torch.rand(self.num_blocks, device=elites.device) < self.mutation_rate)
            child = torch.where(mutate, 1 - child, child)

            # 防止全0
            child = self._fix_zero_masks(child)

            new_population.append(child)

        new_population = torch.stack(new_population[:self.pop_size], dim=0)
        self.population.copy_(new_population)

    def forward(self, h_s, h_a):
        """
        h_s: [B, C]  主模态（query）
        h_a: [B, C]  辅模态（key/value）
        return: [B, output_dim]
        """
        assert h_s.shape == h_a.shape, "两个模态输入维度必须一致"
        B, C = h_s.shape

        # 周期性进化
        if self.training and (int(self.step_counter.item()) % self.evolve_steps == 0):
            self._evolve(h_s.detach(), h_a.detach())
        self.step_counter += 1

        # 归一化
        h_s = self.norm_s(h_s)
        h_a = self.norm_a(h_a)

        # 切成 block tokens
        s_blk = self._split_blocks(h_s)   # [B, num_blocks, block_size]
        a_blk = self._split_blocks(h_a)   # [B, num_blocks, block_size]

        # 采用 best_mask + soft gate
        w = self._soft_block_weight(self.best_mask)   # [1, num_blocks, 1]
        s_blk = s_blk * w
        a_blk = a_blk * w

        # 单向跨注意力：s <- a
        attn_out, _ = self.cross_attn(
            query=s_blk,
            key=a_blk,
            value=a_blk
        )  # [B, num_blocks, block_size]

        # 展平回 [B, C]
        attn_out = attn_out.reshape(B, C)
        h_s_flat = s_blk.reshape(B, C)

        # 残差 + FFN
        out = self.norm1(h_s_flat + self.dropout(attn_out))
        out = self.norm2(out + self.dropout(self.ffn(out)))

        return self.proj(out)
