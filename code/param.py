import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feat_dim", type=int, default=64, help="Input node feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--out_dim", type=int, default=128, help="Output dimension")
    parser.add_argument(
        "--fusion_head", type=int, default=8, help="The number of heads during cross-attention fusion"
    )
    parser.add_argument(
        "--fusion_dropout", type=float, default=0.1, help="Dropout during cross-attention fusion"
    )
    parser.add_argument(
        "--attribute_encoder",
        choices=("dynamic", "dual_neighborhood"),
        default="dynamic",
        help="Attribute graph encoder used for the homogeneous views",
    )
    parser.add_argument(
        "--structural_feature_mode",
        choices=("pca_spectral", "text_spectral", "multiview_spectral"),
        default="pca_spectral",
        help="Base node channels propagated by the structural liquid/diffusion encoder",
    )
    parser.add_argument(
        "--normalize_structural_base",
        action="store_true",
        default=False,
        help="Row-normalize the PCA structural channels before spectral coordinates",
    )
    parser.add_argument(
        "--association_degree_features",
        action="store_true",
        default=False,
        help="Append fold-safe standardized log-degree channels to spectral coordinates",
    )
    parser.add_argument(
        "--gcn_layers", type=int, default=4, help="Number of residual connection layers"
    )
    parser.add_argument(
        "--att_dropout", type=float, default=0.1, help="Dropout during the update of homogeneous graphs"
    )
    parser.add_argument("--hops", type=int, default=9, help="Structural layer neighbor hop count")
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--diffusion_rate", type=float, default=0.2)
    parser.add_argument(
        "--structural_encoder",
        choices=(
            "diffusion_transformer",
            "metapath_liquid",
            "hybrid_metapath_liquid",
            "diffusion_liquid",
            "hybrid_diffusion_liquid",
            "liquid_adapter_transformer",
        ),
        default="hybrid_diffusion_liquid",
        help="Structural sequence encoder used for the ablation experiment",
    )
    parser.add_argument(
        "--metapath_steps",
        type=int,
        default=2,
        help="Maximum alternating disease-metabolite meta-path length",
    )
    parser.add_argument(
        "--liquid_dropout",
        type=float,
        default=0.2,
        help="Dropout in the liquid meta-path encoder",
    )
    parser.add_argument(
        "--liquid_gate_bias",
        type=float,
        default=-2.0,
        help="Initial bias of the liquid residual gate",
    )
    parser.add_argument(
        "--liquid_residual_init",
        type=float,
        default=0.0,
        help="Identity scale for the liquid residual bypass (0 keeps the zero-start baseline)",
    )
    parser.add_argument(
        "--liquid_fusion_norm",
        action="store_true",
        default=False,
        help="Normalize the diffusion/liquid residual mixture before fusion",
    )
    parser.add_argument(
        "--liquid_lr_scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for auxiliary liquid parameters",
    )
    parser.add_argument(
        "--liquid_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay applied only to auxiliary liquid parameters",
    )
    parser.add_argument(
        "--association_rank",
        type=int,
        default=16,
        help="Rank of fold-safe bipartite spectral coordinates",
    )
    parser.add_argument(
        "--association_degree_alpha",
        type=float,
        default=0.0,
        help="Two-sided degree correction used before association SVD",
    )
    parser.add_argument("--fusion_blocks", type=int, default=8)
    parser.add_argument("--router_keep_ratio", type=float, default=0.75)
    parser.add_argument(
        "--decoder",
        choices=("bilinear", "fold_calibrated"),
        default="bilinear",
        help="Link decoder; fold_calibrated uses training-fold degree statistics only",
    )
    parser.add_argument(
        "--propagation_prior",
        action="store_true",
        help="Add learnable disease/meta similarity propagation residuals",
    )
    parser.add_argument("--kneighbor", type=int, default=5, help="k-nearest neighbor support vector")
    parser.add_argument(
        "--gaussian_kernel",
        choices=("none", "gip", "blend", "feature"),
        default="blend",
        help="Fold-safe Gaussian interaction-profile graph or feature mode",
    )
    parser.add_argument(
        "--gaussian_alpha",
        type=float,
        default=0.1,
        help="Weight of the Gaussian similarity in blend mode",
    )
    parser.add_argument(
        "--gaussian_gamma_scale",
        type=float,
        default=1.0,
        help="Multiplier for the inverse mean profile distance used by the Gaussian kernel",
    )
    parser.add_argument("--epochs", type=int, default=800, help="Number of training rounds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
        help="First cross-validation fold to run (1-based)",
    )
    parser.add_argument(
        "--max_run_folds",
        type=int,
        default=0,
        help="Run only the first N folds; 0 runs every fold.",
    )
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed for positive folds, validation negatives and train negatives",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
        help="Seed for fold-specific model initialization",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="Parameter EMA decay used for validation; 0 disables EMA",
    )
    parser.add_argument(
        "--lr_schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument(
        "--ranking_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the pairwise softplus ranking objective added to BCE",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Binary target smoothing strength used only for the training BCE",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Number of sampled training negatives per positive edge",
    )
    parser.add_argument(
        "--negative_strategy",
        choices=("random", "hard", "mixed"),
        default="random",
        help="Training negative sampler; hard/mixed mine negatives using current scores",
    )
    parser.add_argument(
        "--hard_negative_fraction",
        type=float,
        default=0.5,
        help="Fraction of each negative batch selected as hard negatives in mixed mode",
    )
    parser.add_argument(
        "--hard_negative_warmup",
        type=int,
        default=50,
        help="Keep random negatives for this many epochs before hard mining",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="",
        help="Optional directory for best validation prediction archives",
    )
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA"
    )
    parser.add_argument("--peak_lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="Optimizer used for the fold model",
    )
    return parser.parse_args()
