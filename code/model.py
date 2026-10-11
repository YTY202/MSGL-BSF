from layers import *
from param import parameter_parser

args = parameter_parser()


class DFFRGM(nn.Module):

    def __init__(
        self,
        attribute_dim,
        structural_dim,
        hidden_dim,
        out_dim,
        gcn_num_layers,  # Number of layers of residual connection in GCN
        num_dis,
    ):
        super(DFFRGM, self).__init__()
        self.attribute_dim = attribute_dim
        self.structural_dim = structural_dim
        self.hidden_dim = (hidden_dim,)
        self.out_dim = (out_dim,)
        self.num_dis = (num_dis,)
        # Homogeneous Graph Convolutional Layer: (Residual Connection)
        #self.resgcnlayer = ResGCNEncoder(feat_dim, hidden_dim, gcn_num_layers, args.att_dropout)
        self.textencoder = DynamicGraphLargeKernelConv(
            attribute_dim,
            hidden_dim,
            use_static_graph=args.attribute_encoder == "dual_neighborhood",
        )
        self.use_propagation_prior = args.propagation_prior
        if self.use_propagation_prior:
            self.meta_propagation_scale = nn.Parameter(torch.zeros(()))
            self.dis_propagation_scale = nn.Parameter(torch.zeros(()))

        self.structural_encoder = args.structural_encoder
        if self.structural_encoder == "metapath_liquid":
            self.strencoder = MetaPathLiquidEncoder(
                structural_dim,
                hidden_dim,
                num_steps=args.metapath_steps,
                dropout=args.liquid_dropout,
            )
        elif self.structural_encoder == "hybrid_metapath_liquid":
            self.diffusion_encoder = DiffusionWaveletTransformerEncoder(
                structural_dim,
                hidden_dim,
                K=args.hops,
                num_heads=args.transformer_heads,
                num_layers=args.transformer_layers,
                dropout=args.transformer_dropout,
                diffusion_rate=args.diffusion_rate,
                use_sparse=False,
            )
            self.metapath_encoder = MetaPathLiquidEncoder(
                structural_dim,
                hidden_dim,
                num_steps=args.metapath_steps,
                dropout=args.liquid_dropout,
            )
            self.metapath_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.hybrid_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.hybrid_gate_bias = nn.Parameter(torch.full((hidden_dim,), -2.0))
            self.hybrid_norm = nn.LayerNorm(hidden_dim)
            nn.init.zeros_(self.hybrid_gate.weight)
            nn.init.zeros_(self.hybrid_gate.bias)
        elif self.structural_encoder == "diffusion_liquid":
            self.strencoder = DiffusionWaveletLiquidEncoder(
                structural_dim,
                hidden_dim,
                K=args.hops,
                dropout=args.liquid_dropout,
                diffusion_rate=args.diffusion_rate,
                use_sparse=False,
            )
        elif self.structural_encoder == "hybrid_diffusion_liquid":
            self.diffusion_encoder = DiffusionWaveletTransformerEncoder(
                structural_dim,
                hidden_dim,
                K=args.hops,
                num_heads=args.transformer_heads,
                num_layers=args.transformer_layers,
                dropout=args.transformer_dropout,
                diffusion_rate=args.diffusion_rate,
                use_sparse=False,
            )
            # The auxiliary branch uses an isolated RNG stream so the main
            # baseline path and downstream fusion retain their initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.model_seed + 104729)
                self.liquid_encoder = DiffusionWaveletLiquidEncoder(
                    structural_dim,
                    hidden_dim,
                    K=args.hops,
                    dropout=args.liquid_dropout,
                    diffusion_rate=args.diffusion_rate,
                    use_sparse=False,
                )
                self.liquid_residual = nn.Linear(
                    hidden_dim, hidden_dim, bias=False
                )
                self.liquid_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                self.liquid_gate_bias = nn.Parameter(
                    torch.full((hidden_dim,), args.liquid_gate_bias)
                )
                self.liquid_fusion_norm = (
                    nn.LayerNorm(hidden_dim)
                    if args.liquid_fusion_norm
                    else nn.Identity()
                )
                if args.liquid_residual_init == 0:
                    nn.init.zeros_(self.liquid_residual.weight)
                else:
                    with torch.no_grad():
                        self.liquid_residual.weight.copy_(
                            args.liquid_residual_init
                            * torch.eye(hidden_dim, device=self.liquid_residual.weight.device)
                        )
                nn.init.zeros_(self.liquid_gate.weight)
                nn.init.zeros_(self.liquid_gate.bias)
        elif self.structural_encoder == "liquid_adapter_transformer":
            self.strencoder = DiffusionWaveletTransformerEncoder(
                structural_dim,
                hidden_dim,
                K=args.hops,
                num_heads=args.transformer_heads,
                num_layers=args.transformer_layers,
                dropout=args.transformer_dropout,
                diffusion_rate=args.diffusion_rate,
                use_sparse=False,
                liquid_adapter=True,
                liquid_dropout=args.liquid_dropout,
            )
        else:
            self.strencoder = DiffusionWaveletTransformerEncoder(
                structural_dim,
                hidden_dim,
                K=args.hops,
                num_heads=args.transformer_heads,
                num_layers=args.transformer_layers,
                dropout=args.transformer_dropout,
                diffusion_rate=args.diffusion_rate,
                use_sparse=False,
            )
        # Decoder: Link Prediction
        self.fusion2 = SparseTokenRouterFusion(
            hidden_dim,
            out_dim,
            num_heads=args.fusion_head,
            dropout=args.fusion_dropout,
            num_blocks=args.fusion_blocks,
            keep_ratio=args.router_keep_ratio,
        )
        if args.decoder == "fold_calibrated":
            self.decoder = FoldCalibratedDecoder(out_dim, num_dis)
        else:
            self.decoder = InnerProductDecoder(out_dim, num_dis)

    def forward(
        self,
        dis_data,
        meta_data,
        feature,
        adj,
        association=None,
        propagation_priors=None,
    ):

        # Feature Update from the homogeneous view ===
        # x_dis = self.resgcnlayer(dis_data.x, dis_data.edge_index)
        # x_meta = self.resgcnlayer(meta_data.x, meta_data.edge_index)
        x_dis = self.textencoder(dis_data.x, dis_data.edge_index)
        x_meta = self.textencoder(meta_data.x, meta_data.edge_index)
        x_att = torch.cat((x_dis, x_meta), dim=0)
        # Features Update from the heterogeneous view
        if self.structural_encoder == "metapath_liquid":
            if association is None:
                raise ValueError(
                    "association is required by the meta-path liquid encoder"
                )
            x_str = self.strencoder(association, feature)
        elif self.structural_encoder == "hybrid_metapath_liquid":
            if association is None:
                raise ValueError(
                    "association is required by the hybrid liquid encoder"
                )
            diffusion = self.diffusion_encoder(adj, feature)
            metapath = self.metapath_encoder(association, feature)
            gate_input = torch.cat([diffusion, metapath], dim=-1)
            gate = torch.sigmoid(
                self.hybrid_gate(gate_input) + self.hybrid_gate_bias
            )
            x_str = self.hybrid_norm(
                diffusion + gate * self.metapath_residual(metapath)
            )
        elif self.structural_encoder == "hybrid_diffusion_liquid":
            diffusion = self.diffusion_encoder(adj, feature)
            liquid = self.liquid_encoder(adj, feature)
            gate_input = torch.cat([diffusion, liquid], dim=-1)
            gate = torch.sigmoid(
                self.liquid_gate(gate_input) + self.liquid_gate_bias
            )
            x_str = self.liquid_fusion_norm(
                diffusion + gate * self.liquid_residual(liquid)
            )
        else:
            x_str = self.strencoder(adj, feature)

        # Integration of Two View
        x_fusion = self.fusion2(x_str, x_att)

        # Decode into Link Scores
        output = self.decoder(x_fusion, association=association)
        if self.use_propagation_prior:
            if propagation_priors is None:
                raise ValueError("propagation_priors are required when enabled")
            meta_prior, dis_prior = propagation_priors
            output = (
                output
                + self.meta_propagation_scale * meta_prior
                + self.dis_propagation_scale * dis_prior
            )
        return output
