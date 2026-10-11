import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from liquid_neighbor_sequence import LiquidNeighborSequencePredictor
from utils import (
    evaluate_scores,
    load_similarity_features,
    load_similarity_matrices,
    make_positive_folds,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data1",
        help="Dataset directory relative to the project root, e.g. data1 or data2.",
    )
    parser.add_argument("--neighbors", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ranking_weight", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_run_folds", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--disable_liquid", action="store_true")
    parser.add_argument(
        "--interval_mode", choices=("fixed", "similarity"), default="fixed"
    )
    parser.add_argument(
        "--sequence_mode",
        choices=("unidirectional", "bidirectional"),
        default="bidirectional",
        help="Read the ranked similarity sequence in one or both directions.",
    )
    parser.add_argument(
        "--timescale_mode",
        choices=("single", "dual"),
        default="dual",
        help="Use one liquid state or gated fast/slow liquid states.",
    )
    parser.add_argument("--fast_interval_scale", type=float, default=2.0)
    parser.add_argument("--slow_interval_scale", type=float, default=0.5)
    parser.add_argument(
        "--multi_scale_neighbors",
        type=int,
        nargs="+",
        default=(16, 32, 64),
        help="Neighbor-prefix lengths used by the shared multi-scale liquid encoder.",
    )
    parser.add_argument("--disable_multi_scale", action="store_true")
    parser.add_argument("--disable_static_shortcuts", action="store_true")
    parser.add_argument(
        "--liquid_only_fusion",
        action="store_true",
        help="Use only the bidirectional liquid dynamic-neighborhood representation for classification.",
    )
    parser.add_argument("--use_text_semantic_branch", action="store_true")
    parser.add_argument("--semantic_encoder", choices=("tfidf", "bert"), default="tfidf")
    parser.add_argument(
        "--semantic_fusion_mode",
        choices=("feature", "logit"),
        default="feature",
        help="Fuse topology and semantic representations before decoding, or fuse independent logits.",
    )
    parser.add_argument("--semantic_feature_dim", type=int, default=64)
    parser.add_argument("--bert_model_path", type=str, default="")
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--bert_max_length", type=int, default=256)
    parser.add_argument("--prediction_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="liquid_neighbor_sequence")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def build_ranked_neighbors(similarity, count):
    similarity = np.maximum(similarity, 0.0).astype(np.float32, copy=True)
    np.fill_diagonal(similarity, 0.0)
    count = min(max(1, count), similarity.shape[1] - 1)
    indices = np.argpartition(similarity, -count, axis=1)[:, -count:]
    values = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(values, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    return indices.astype(np.int64), values.astype(np.float32)


def load_text_semantic_features(
    project_root,
    data_dir_name,
    feature_dim,
    encoder_name="tfidf",
    bert_model_path="",
    bert_batch_size=32,
    bert_max_length=256,
    device=None,
):
    """Build aligned text features from dataset metadata files.

    data1 keeps its historical Excel schema; rebuilt saliva datasets use CSV
    metadata with HMDB/MeSH identifiers and a ``Description`` column.
    """
    data_dir = project_root / data_dir_name
    cache_dir = project_root / "result" / "text_semantic_cache" / data_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    has_curated_corpus = (
        (data_dir / "metabolite_bert_corpus.csv").exists()
        and (data_dir / "disease_bert_corpus.csv").exists()
    )
    corpus_suffix = "_curated_v1" if has_curated_corpus else ""
    cache_stem = (
        f"bert_{Path(bert_model_path).name}_len{bert_max_length}{corpus_suffix}"
        if encoder_name == "bert"
        else f"tfidf_svd_dim{feature_dim}{corpus_suffix}"
    )
    cache_path = cache_dir / f"{cache_stem}.npz"
    if cache_path.exists():
        with np.load(cache_path) as archive:
            cached_meta = np.nan_to_num(archive["meta"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            cached_dis = np.nan_to_num(archive["dis"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return normalize(cached_meta, norm="l2"), normalize(cached_dis, norm="l2")

    if has_curated_corpus:
        meta_table = pd.read_csv(data_dir / "metabolite_bert_corpus.csv")
        dis_table = pd.read_csv(data_dir / "disease_bert_corpus.csv")
        meta_text = meta_table["BERT Text"].fillna("").astype(str).tolist()
        dis_text = dis_table["BERT Text"].fillna("").astype(str).tolist()
    elif (data_dir / "metabolite_metadata.csv").exists():
        meta_table = pd.read_csv(data_dir / "metabolite_metadata.csv")
        dis_table = pd.read_csv(data_dir / "disease_metadata.csv")
        meta_name_column, dis_name_column, desc_column = (
            "Metabolite Name", "Disease Name", "Description"
        )
    else:
        meta_table = pd.read_excel(data_dir / "Meta_describe.xlsx")
        dis_table = pd.read_excel(data_dir / "disease_describe.xlsx")
        meta_name_column, dis_name_column, desc_column = (
            "Metabolite Name", "Disease Name", "Describe"
        )

    if not has_curated_corpus:
        def combine(table, name_column):
            names = table[name_column].fillna("").astype(str)
            descriptions = table[desc_column].fillna("").astype(str)
            descriptions = descriptions.mask(
                descriptions.str.lower().eq("no describe"), ""
            )
            return (names + " " + descriptions).tolist()

        meta_text = combine(meta_table, meta_name_column)
        dis_text = combine(dis_table, dis_name_column)
    documents = meta_text + dis_text
    if encoder_name == "bert":
        if not bert_model_path:
            raise ValueError("--bert_model_path is required for BERT semantic encoding.")
        from transformers import AutoModel, AutoTokenizer

        bert_device = device or torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model_path, local_files_only=True
        )
        encoder = AutoModel.from_pretrained(
            bert_model_path, local_files_only=True
        ).to(bert_device)
        encoder.eval()
        vectors = []
        with torch.no_grad():
            for start in range(0, len(documents), bert_batch_size):
                batch = documents[start : start + bert_batch_size]
                tokens = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=bert_max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(bert_device) for key, value in tokens.items()}
                hidden = encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                vectors.append(normalize(pooled.cpu().numpy(), norm="l2"))
        semantic = np.concatenate(vectors, axis=0).astype(np.float32)
        # Defensive sanitization for degenerate/empty text rows in local
        # checkpoints; a single non-finite embedding would poison all logits.
        semantic = np.nan_to_num(semantic, nan=0.0, posinf=0.0, neginf=0.0)
        semantic = normalize(semantic, norm="l2")
        meta_semantic = semantic[: len(meta_text)]
        dis_semantic = semantic[len(meta_text) :]
        np.savez_compressed(cache_path, meta=meta_semantic, dis=dis_semantic)
        del encoder
        if torch.cuda.is_available() and bert_device.type == "cuda":
            torch.cuda.empty_cache()
        return meta_semantic, dis_semantic

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    text_matrix = vectorizer.fit_transform(documents)
    rank = min(feature_dim, text_matrix.shape[0] - 1, text_matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=rank, n_iter=7, random_state=0)
    semantic = normalize(reducer.fit_transform(text_matrix), norm="l2").astype(
        np.float32
    )
    if rank < feature_dim:
        semantic = np.pad(semantic, ((0, 0), (0, feature_dim - rank)))
    meta_semantic = semantic[: len(meta_text)]
    dis_semantic = semantic[len(meta_text) :]
    np.savez_compressed(cache_path, meta=meta_semantic, dis=dis_semantic)
    return meta_semantic, dis_semantic


def edge_tensor(edges, device):
    return torch.tensor(edges.T, dtype=torch.long, device=device)


def predict_batches(
    model,
    edges,
    meta_features,
    dis_features,
    neighbor_indices,
    neighbor_similarities,
    train_association,
    batch_size,
    use_liquid,
    interval_mode,
    sequence_mode,
    use_static_shortcuts,
    semantic_meta_features,
    semantic_dis_features,
):
    outputs = []
    for start in range(0, len(edges), batch_size):
        batch = edge_tensor(edges[start : start + batch_size], meta_features.device)
        outputs.append(
            model(
                meta_features,
                dis_features,
                batch,
                neighbor_indices,
                neighbor_similarities,
                train_association,
                use_liquid=use_liquid,
                interval_mode=interval_mode,
                sequence_mode=sequence_mode,
                use_static_shortcuts=use_static_shortcuts,
                semantic_meta_features=semantic_meta_features,
                semantic_dis_features=semantic_dis_features,
            )
        )
    return torch.cat(outputs)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    data_dir = root / args.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    association = pd.read_csv(
        data_dir / "association_matrix.csv", header=0
    ).to_numpy(np.float32)
    meta_similarity, dis_similarity = load_similarity_matrices(root, args.data_dir)
    meta_array, dis_array = load_similarity_features(
        root,
        args.feature_dim,
        0,
        similarities=(meta_similarity, dis_similarity),
        data_dir_name=args.data_dir,
    )
    neighbor_index_array, neighbor_value_array = build_ranked_neighbors(
        meta_similarity, args.neighbors
    )
    meta_features = torch.tensor(meta_array, device=device)
    dis_features = torch.tensor(dis_array, device=device)
    if args.use_text_semantic_branch:
        meta_semantic_array, dis_semantic_array = load_text_semantic_features(
            root,
            args.data_dir,
            args.semantic_feature_dim,
            encoder_name=args.semantic_encoder,
            bert_model_path=args.bert_model_path,
            bert_batch_size=args.bert_batch_size,
            bert_max_length=args.bert_max_length,
            device=device,
        )
        if (
            len(meta_semantic_array) != len(meta_array)
            or len(dis_semantic_array) != len(dis_array)
        ):
            raise ValueError("Text descriptions are not aligned with dataset node order.")
        semantic_meta_features = torch.tensor(meta_semantic_array, device=device)
        semantic_dis_features = torch.tensor(dis_semantic_array, device=device)
    else:
        semantic_meta_features = None
        semantic_dis_features = None
    neighbor_indices = torch.tensor(neighbor_index_array, device=device)
    neighbor_similarities = torch.tensor(neighbor_value_array, device=device)
    positive_folds = make_positive_folds(association, args.folds, args.split_seed)
    all_negatives = np.argwhere(association < 1)
    use_liquid = not args.disable_liquid
    use_static_shortcuts = not args.disable_static_shortcuts
    print(
        f"dataset={args.data_dir}, association={association.shape}, device={device}, "
        f"neighbors={args.neighbors}, use_liquid={use_liquid}, "
        f"feature_dim={args.feature_dim}, hidden_dim={args.hidden_dim}, "
        f"interval_mode={args.interval_mode}, "
        f"sequence_mode={args.sequence_mode}, "
        f"timescale_mode={args.timescale_mode}, "
        f"interval_scales=({args.fast_interval_scale}, {args.slow_interval_scale}), "
        f"multi_scale={not args.disable_multi_scale}, scales={args.multi_scale_neighbors}, "
        f"static_shortcuts={use_static_shortcuts}, "
        f"liquid_only_fusion={args.liquid_only_fusion}, "
        f"text_semantic_branch={args.use_text_semantic_branch}, "
        f"semantic_encoder={args.semantic_encoder}, "
        f"semantic_fusion_mode={args.semantic_fusion_mode}"
    )
    fold_results = []
    run_folds = args.folds if args.max_run_folds <= 0 else args.max_run_folds

    for fold in range(min(run_folds, args.folds)):
        fold_seed = args.model_seed + fold * 1009
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        val_pos = positive_folds[fold]
        train_matrix = association.copy()
        train_matrix[val_pos[:, 0], val_pos[:, 1]] = 0.0
        train_pos = np.argwhere(train_matrix > 0)
        val_rng = np.random.default_rng(args.split_seed + 30011 + fold)
        val_neg = all_negatives[
            val_rng.choice(len(all_negatives), len(val_pos), replace=False)
        ]
        train_negative_mask = association < 1
        train_negative_mask = train_negative_mask.copy()
        train_negative_mask[val_neg[:, 0], val_neg[:, 1]] = False
        train_negative_pool = np.argwhere(train_negative_mask)
        train_rng = np.random.default_rng(args.split_seed + 50021 + fold)
        train_neg = train_negative_pool[
            train_rng.choice(len(train_negative_pool), len(train_pos), replace=False)
        ]
        train_edges = np.concatenate([train_pos, train_neg])
        train_labels = np.concatenate(
            [np.ones(len(train_pos), np.float32), np.zeros(len(train_neg), np.float32)]
        )
        val_edges = np.concatenate([val_pos, val_neg])
        val_labels = np.concatenate(
            [np.ones(len(val_pos), np.float32), np.zeros(len(val_neg), np.float32)]
        )
        train_association = torch.tensor(train_matrix, device=device)
        model = LiquidNeighborSequencePredictor(
            args.feature_dim,
            args.hidden_dim,
            args.dropout,
            semantic_dim=(
                int(meta_semantic_array.shape[1])
                if args.use_text_semantic_branch
                else 0
            ),
            timescale_mode=args.timescale_mode,
            fast_interval_scale=args.fast_interval_scale,
            slow_interval_scale=args.slow_interval_scale,
            multi_scale_neighbors=args.multi_scale_neighbors,
            use_multi_scale=not args.disable_multi_scale,
            liquid_only_fusion=args.liquid_only_fusion,
            semantic_fusion_mode=args.semantic_fusion_mode,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
        )
        best = {"epoch": 0, "auc": 0.0, "prc": 0.0, "scores": None, "metrics": None}
        stale = 0
        print(f"------ Fold {fold + 1}/{args.folds} ------")
        for epoch in range(args.epochs):
            start_time = time.time()
            order = np.random.permutation(len(train_edges))
            model.train()
            losses = []
            for start in range(0, len(order), args.batch_size):
                selection = order[start : start + args.batch_size]
                batch_edges = train_edges[selection]
                batch_labels = torch.tensor(train_labels[selection], device=device)
                optimizer.zero_grad()
                logits = predict_batches(
                    model,
                    batch_edges,
                    meta_features,
                    dis_features,
                    neighbor_indices,
                    neighbor_similarities,
                    train_association,
                    len(batch_edges),
                    use_liquid,
                    args.interval_mode,
                    args.sequence_mode,
                    use_static_shortcuts,
                    semantic_meta_features,
                    semantic_dis_features,
                )
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                if args.ranking_weight > 0:
                    positive = logits[batch_labels > 0.5]
                    negative = logits[batch_labels < 0.5]
                    pairs = min(len(positive), len(negative))
                    if pairs:
                        loss = loss + args.ranking_weight * F.softplus(
                            -(positive[:pairs] - negative[:pairs])
                        ).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_logits = predict_batches(
                    model,
                    val_edges,
                    meta_features,
                    dis_features,
                    neighbor_indices,
                    neighbor_similarities,
                    train_association,
                    args.batch_size,
                    use_liquid,
                    args.interval_mode,
                    args.sequence_mode,
                    use_static_shortcuts,
                    semantic_meta_features,
                    semantic_dis_features,
                )
                val_scores = torch.sigmoid(val_logits).cpu().numpy()
            result = evaluate_scores(val_labels, val_scores)
            print(
                f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f} | "
                f"ACC: {result['acc']:.4f} | Precision: {result['precision']:.4f} | "
                f"Recall: {result['recall']:.4f} | F1: {result['f1']:.4f} | "
                f"AUC: {result['auc']:.4f} | AUPR: {result['prc']:.4f} | "
                f"Time: {time.time() - start_time:.2f}"
            )
            if result["auc"] > best["auc"] + 1e-6:
                best = {
                    "epoch": epoch + 1,
                    "auc": result["auc"],
                    "prc": result["prc"],
                    "scores": val_scores.copy(),
                    "metrics": result,
                }
                if args.checkpoint_dir:
                    checkpoint_dir = Path(args.checkpoint_dir)
                    if not checkpoint_dir.is_absolute():
                        checkpoint_dir = root / checkpoint_dir
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "auc": result["auc"],
                            "prc": result["prc"],
                        },
                        checkpoint_dir / f"{args.run_name}_fold{fold + 1}.pt",
                    )
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        fold_results.append(best)
        print(
            f"Fold {fold + 1} Best @Epoch {best['epoch']} | "
            f"ACC: {best['metrics']['acc']:.4f} | "
            f"Precision: {best['metrics']['precision']:.4f} | "
            f"Recall: {best['metrics']['recall']:.4f} | "
            f"F1: {best['metrics']['f1']:.4f} | "
            f"AUC: {best['auc']:.4f} | AUPR: {best['prc']:.4f}"
        )
        if args.prediction_dir:
            output = Path(args.prediction_dir)
            if not output.is_absolute():
                output = root / output
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / f"{args.run_name}_fold{fold + 1}.npz",
                scores=best["scores"],
                labels=val_labels,
                positive_edges=val_pos.T,
                negative_edges=val_neg.T,
            )
    metric_keys = ["acc", "precision", "recall", "f1", "auc", "prc"]
    metric_names = {"acc": "ACC", "precision": "Precision", "recall": "Recall",
                    "f1": "F1", "auc": "AUC", "prc": "AUPR"}
    metric_values = {
        key: np.asarray([item["metrics"][key] for item in fold_results])
        for key in metric_keys
    }
    aucs = metric_values["auc"]
    prcs = metric_values["prc"]
    print("\n## Liquid Neighbor Sequence Finished")
    for key in metric_keys:
        values = metric_values[key]
        print(f"{metric_names[key]} folds: {np.round(values, 4).tolist()}")
        print(f"{metric_names[key]} mean: {values.mean():.4f}, std: {values.std():.4f}")


if __name__ == "__main__":
    main()
