import torch
import torch.nn as nn


class LiquidSequenceCell(nn.Module):
    """Input-conditioned liquid time-constant cell."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        context_dim = input_dim + hidden_dim
        self.candidate = nn.Linear(context_dim, hidden_dim)
        self.time_constant = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Softplus(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, delta_t):
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e4, neginf=-1e4)
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        delta_t = torch.nan_to_num(delta_t, nan=1.0, posinf=1e4, neginf=1e-4)
        context = torch.cat([inputs, hidden], dim=-1)
        candidate = torch.tanh(self.candidate(context))
        tau = self.time_constant(context).clamp_min(1e-3)
        decay = torch.exp(-delta_t / tau)
        updated = decay * hidden + (1.0 - decay) * candidate
        return self.norm(hidden + self.dropout(updated - hidden))


class LiquidNeighborSequencePredictor(nn.Module):
    """Read a ranked metabolite-neighbor sequence with configurable liquid dynamics."""

    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        dropout=0.1,
        semantic_dim=0,
        timescale_mode="dual",
        fast_interval_scale=2.0,
        slow_interval_scale=0.5,
        multi_scale_neighbors=(16, 32, 64),
        use_multi_scale=True,
        liquid_only_fusion=False,
        semantic_fusion_mode="feature",
    ):
        super().__init__()
        if timescale_mode not in {"single", "dual"}:
            raise ValueError(f"Unknown timescale mode: {timescale_mode}")
        if fast_interval_scale <= 0 or slow_interval_scale <= 0:
            raise ValueError("Liquid interval scales must be positive.")
        self.timescale_mode = timescale_mode
        self.use_multi_scale = use_multi_scale
        self.liquid_only_fusion = liquid_only_fusion
        if semantic_fusion_mode not in {"feature", "logit"}:
            raise ValueError(f"Unknown semantic fusion mode: {semantic_fusion_mode}")
        self.semantic_fusion_mode = semantic_fusion_mode
        self.multi_scale_neighbors = tuple(sorted(set(multi_scale_neighbors)))
        if not self.multi_scale_neighbors or self.multi_scale_neighbors[0] <= 0:
            raise ValueError("Multi-scale neighbor lengths must be positive.")
        self.meta_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dis_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.semantic_dim = semantic_dim
        if semantic_dim > 0:
            self.semantic_meta_encoder = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.semantic_dis_encoder = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.semantic_bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            self.semantic_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            # Independent semantic tower. It is fused only at the prediction
            # layer, so BERT cannot destabilize liquid state transitions.
            self.semantic_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.branch_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fast_cell = LiquidSequenceCell(hidden_dim, hidden_dim, dropout)
        self.slow_cell = LiquidSequenceCell(hidden_dim, hidden_dim, dropout)
        self.reverse_fast_cell = LiquidSequenceCell(hidden_dim, hidden_dim, dropout)
        self.reverse_slow_cell = LiquidSequenceCell(hidden_dim, hidden_dim, dropout)
        self.fast_log_scale = nn.Parameter(
            torch.tensor(float(fast_interval_scale)).log()
        )
        self.slow_log_scale = nn.Parameter(
            torch.tensor(float(slow_interval_scale)).log()
        )
        self.interval_logits = nn.Parameter(torch.zeros(1))
        self.similarity_interval_log_scale = nn.Parameter(torch.zeros(1))
        self.forward_mix_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.backward_mix_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.direction_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        classifier_input_dim = hidden_dim if liquid_only_fusion else hidden_dim * 2 + 2
        if semantic_dim > 0 and semantic_fusion_mode == "feature":
            self.topology_fusion_encoder = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.semantic_fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        meta_features,
        dis_features,
        edge_index,
        neighbor_indices,
        neighbor_similarities,
        train_association,
        use_liquid=True,
        interval_mode="fixed",
        sequence_mode="bidirectional",
        use_static_shortcuts=True,
        semantic_meta_features=None,
        semantic_dis_features=None,
    ):
        meta_index, dis_index = edge_index
        meta_all = self.meta_encoder(meta_features)
        dis_all = self.dis_encoder(dis_features)
        target_meta = meta_all[meta_index]
        target_dis = dis_all[dis_index]
        anchor = self.pair_encoder(torch.cat([target_meta, target_dis], dim=-1))
        use_semantic = semantic_meta_features is not None or semantic_dis_features is not None
        if use_semantic:
            if semantic_meta_features is None or semantic_dis_features is None:
                raise ValueError("Both semantic feature tensors are required.")
            if self.semantic_dim <= 0:
                raise ValueError("Model was created without a text semantic branch.")
            semantic_meta = self.semantic_meta_encoder(semantic_meta_features[meta_index])
            semantic_dis = self.semantic_dis_encoder(semantic_dis_features[dis_index])
            semantic_state = self.semantic_encoder(
                torch.cat(
                    [
                        semantic_meta * semantic_dis,
                        torch.abs(semantic_meta - semantic_dis),
                        self.semantic_bilinear(semantic_meta, semantic_dis),
                    ],
                    dim=-1,
                )
            )

        neighbors = neighbor_indices[meta_index]
        similarities = neighbor_similarities[meta_index]
        observations = train_association[neighbors, dis_index.unsqueeze(1)]
        if use_static_shortcuts:
            weighted_mean = (similarities * observations).sum(
                dim=1
            ) / similarities.sum(dim=1).clamp_min(1e-8)
            observed_fraction = observations.mean(dim=1)
        else:
            weighted_mean = torch.zeros_like(observations[:, 0])
            observed_fraction = torch.zeros_like(observations[:, 0])

        if use_liquid:
            disease_context = target_dis
            sequence_length = neighbors.size(1)
            base_interval = torch.nn.functional.softplus(self.interval_logits) + 1e-3
            if interval_mode == "similarity":
                minimum = similarities[:, :1]
                span = (similarities[:, -1:] - minimum).clamp_min(1e-6)
                coordinates = (similarities - minimum) / span
                intervals = torch.cat(
                    [coordinates[:, :1], coordinates[:, 1:] - coordinates[:, :-1]],
                    dim=1,
                ).clamp_min(1e-4)
                intervals = (
                    intervals
                    * sequence_length
                    * self.similarity_interval_log_scale.exp()
                )
            elif interval_mode == "fixed":
                intervals = None
            else:
                raise ValueError(f"Unknown interval mode: {interval_mode}")
            positions = torch.linspace(
                0.0,
                1.0,
                sequence_length,
                dtype=meta_features.dtype,
                device=meta_features.device,
            )
            def build_event(step):
                neighbor_state = meta_all[neighbors[:, step]]
                similarity = similarities[:, step : step + 1]
                observation = observations[:, step : step + 1]
                position = positions[step].expand_as(similarity)
                return self.event_encoder(
                    torch.cat(
                        [
                            neighbor_state,
                            disease_context,
                            similarity,
                            observation,
                            similarity * observation,
                            position,
                        ],
                        dim=-1,
                    )
                )

            def liquid_step(event, delta_t, fast_hidden, slow_hidden, fast_cell, slow_cell):
                fast_hidden = fast_cell(
                    event, fast_hidden, delta_t * self.fast_log_scale.exp()
                )
                if self.timescale_mode == "dual":
                    slow_hidden = slow_cell(
                        event, slow_hidden, delta_t * self.slow_log_scale.exp()
                    )
                return fast_hidden, slow_hidden

            def mix_timescales(fast_hidden, slow_hidden, gate_layer):
                if self.timescale_mode == "single":
                    return fast_hidden
                gate = torch.sigmoid(
                    gate_layer(torch.cat([fast_hidden, slow_hidden, anchor], dim=-1))
                )
                return gate * fast_hidden + (1.0 - gate) * slow_hidden

            def encode_scale(length):
                fast_hidden = anchor
                slow_hidden = anchor
                for step in range(length):
                    event = build_event(step)
                    delta_t = (
                        base_interval
                        if intervals is None
                        else intervals[:, step : step + 1]
                    )
                    fast_hidden, slow_hidden = liquid_step(
                        event, delta_t, fast_hidden, slow_hidden, self.fast_cell, self.slow_cell
                    )
                forward_state = mix_timescales(
                    fast_hidden, slow_hidden, self.forward_mix_gate
                )
                if sequence_mode == "unidirectional":
                    return forward_state
                if sequence_mode != "bidirectional":
                    raise ValueError(f"Unknown sequence mode: {sequence_mode}")
                reverse_fast_hidden = anchor
                reverse_slow_hidden = anchor
                for step in range(length - 1, -1, -1):
                    event = build_event(step)
                    delta_t = (
                        base_interval
                        if intervals is None
                        else intervals[:, step : step + 1]
                    )
                    reverse_fast_hidden, reverse_slow_hidden = liquid_step(
                        event,
                        delta_t,
                        reverse_fast_hidden,
                        reverse_slow_hidden,
                        self.reverse_fast_cell,
                        self.reverse_slow_cell,
                    )
                backward_state = mix_timescales(
                    reverse_fast_hidden, reverse_slow_hidden, self.backward_mix_gate
                )
                direction_gate = torch.sigmoid(
                    self.direction_gate(
                        torch.cat([forward_state, backward_state, anchor], dim=-1)
                    )
                )
                return direction_gate * forward_state + (1.0 - direction_gate) * backward_state

            if self.use_multi_scale:
                scales = [scale for scale in self.multi_scale_neighbors if scale <= sequence_length]
            else:
                scales = []
            if sequence_length not in scales:
                scales.append(sequence_length)
            scale_states = torch.stack([encode_scale(scale) for scale in scales], dim=1)
            if len(scales) == 1:
                liquid_state = scale_states[:, 0]
            else:
                anchor_per_scale = anchor.unsqueeze(1).expand_as(scale_states)
                scale_logits = self.scale_gate(
                    torch.cat([scale_states, anchor_per_scale], dim=-1)
                ).squeeze(-1)
                scale_weights = torch.softmax(scale_logits, dim=1)
                liquid_state = (scale_weights.unsqueeze(-1) * scale_states).sum(dim=1)

        else:
            liquid_state = torch.zeros_like(anchor)

        if self.liquid_only_fusion:
            classifier_input = liquid_state
        else:
            classifier_input = torch.cat(
                [
                    anchor,
                    liquid_state,
                    weighted_mean.unsqueeze(1),
                    observed_fraction.unsqueeze(1),
                ],
                dim=-1,
            )
        if use_semantic:
            if self.semantic_fusion_mode == "feature":
                topology_state = self.topology_fusion_encoder(classifier_input)
                interaction = torch.cat(
                    [
                        topology_state,
                        semantic_state,
                        topology_state * semantic_state,
                        torch.abs(topology_state - semantic_state),
                    ],
                    dim=-1,
                )
                gate = self.semantic_fusion_gate(interaction)
                fused_state = gate * topology_state + (1.0 - gate) * semantic_state
                decoder_input = torch.cat(
                    [
                        fused_state,
                        topology_state,
                        semantic_state,
                        topology_state * semantic_state,
                    ],
                    dim=-1,
                )
                fused_logit = self.fusion_decoder(decoder_input).squeeze(-1)
                return torch.nan_to_num(fused_logit, nan=0.0, posinf=30.0, neginf=-30.0)
            liquid_logit = self.classifier(classifier_input).squeeze(-1)
            semantic_logit = self.semantic_classifier(semantic_state).squeeze(-1)
            # Learned convex fusion of independent liquid and BERT predictions.
            alpha = torch.sigmoid(self.branch_gate(anchor)).squeeze(-1)
            return torch.nan_to_num(alpha * liquid_logit + (1.0 - alpha) * semantic_logit, nan=0.0, posinf=30.0, neginf=-30.0)
        liquid_logit = self.classifier(classifier_input).squeeze(-1)
        return torch.nan_to_num(liquid_logit, nan=0.0, posinf=30.0, neginf=-30.0)
