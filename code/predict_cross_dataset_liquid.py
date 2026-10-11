"""Zero-shot external evaluation of Dataset-1 MSI-FSDLNet checkpoints.

Target association labels are used only to form the evaluation set. The
association observation tensor passed to the liquid encoder is all zeros, so
no target association is exposed to the model during inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from liquid_neighbor_sequence import LiquidNeighborSequencePredictor
from utils import evaluate_scores, load_similarity_features, load_similarity_matrices
from train_liquid_neighbor_sequence import build_ranked_neighbors, load_text_semantic_features, predict_batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data2_saliva_mesh")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--bert_model_path", default="/root/BiomedBERT")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--negative_seed", type=int, default=0)
    parser.add_argument("--output_dir", default="result/cross_dataset_saliva")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def make_model(saved_args, semantic_dim, device):
    return LiquidNeighborSequencePredictor(
        saved_args["feature_dim"],
        saved_args["hidden_dim"],
        saved_args["dropout"],
        semantic_dim=semantic_dim,
        timescale_mode=saved_args["timescale_mode"],
        fast_interval_scale=saved_args["fast_interval_scale"],
        slow_interval_scale=saved_args["slow_interval_scale"],
        multi_scale_neighbors=saved_args["multi_scale_neighbors"],
        use_multi_scale=not saved_args["disable_multi_scale"],
        liquid_only_fusion=saved_args["liquid_only_fusion"],
        semantic_fusion_mode=saved_args["semantic_fusion_mode"],
    ).to(device)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    checkpoints = sorted((root / args.checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {root / args.checkpoint_dir}")
    first = torch.load(checkpoints[0], map_location="cpu")
    saved_args = first["args"]
    feature_dim = int(saved_args["feature_dim"])

    association = pd.read_csv(data_dir / "association_matrix.csv", header=0).to_numpy(np.float32)
    meta_similarity, dis_similarity = load_similarity_matrices(root, args.data_dir)
    meta_array, dis_array = load_similarity_features(
        root, feature_dim, 0, similarities=(meta_similarity, dis_similarity), data_dir_name=args.data_dir
    )
    neighbor_index_array, neighbor_value_array = build_ranked_neighbors(meta_similarity, args.neighbors)
    meta_semantic_array, dis_semantic_array = load_text_semantic_features(
        root,
        args.data_dir,
        int(saved_args.get("semantic_feature_dim", 64)),
        encoder_name="bert",
        bert_model_path=args.bert_model_path,
        bert_batch_size=int(saved_args.get("bert_batch_size", 32)),
        bert_max_length=int(saved_args.get("bert_max_length", 256)),
        device=device,
    )
    if len(meta_semantic_array) != association.shape[0] or len(dis_semantic_array) != association.shape[1]:
        raise ValueError("Target semantic features are not aligned with the association matrix")

    positive_edges = np.argwhere(association > 0)
    negative_pool = np.argwhere(association < 1)
    rng = np.random.default_rng(args.negative_seed)
    negative_edges = negative_pool[rng.choice(len(negative_pool), len(positive_edges), replace=False)]
    edges = np.concatenate([positive_edges, negative_edges])
    labels = np.concatenate([np.ones(len(positive_edges), np.float32), np.zeros(len(negative_edges), np.float32)])

    meta_features = torch.tensor(meta_array, device=device)
    dis_features = torch.tensor(dis_array, device=device)
    neighbor_indices = torch.tensor(neighbor_index_array, device=device)
    neighbor_similarities = torch.tensor(neighbor_value_array, device=device)
    semantic_meta = torch.tensor(meta_semantic_array, device=device)
    semantic_dis = torch.tensor(dis_semantic_array, device=device)
    # Strict external inference: target labels never enter liquid observations.
    zero_association = torch.zeros_like(torch.tensor(association, device=device))

    checkpoint_scores = []
    with torch.no_grad():
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = make_model(checkpoint["args"], semantic_meta.shape[1], device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            logits = predict_batches(
                model,
                edges,
                meta_features,
                dis_features,
                neighbor_indices,
                neighbor_similarities,
                zero_association,
                args.batch_size,
                use_liquid=not checkpoint["args"]["disable_liquid"],
                interval_mode=checkpoint["args"]["interval_mode"],
                sequence_mode=checkpoint["args"]["sequence_mode"],
                use_static_shortcuts=not checkpoint["args"]["disable_static_shortcuts"],
                semantic_meta_features=semantic_meta,
                semantic_dis_features=semantic_dis,
            )
            checkpoint_scores.append(torch.sigmoid(logits).cpu().numpy())
            del model

    fold_scores = np.stack(checkpoint_scores)
    scores = fold_scores.mean(axis=0)
    metrics = evaluate_scores(labels, scores)
    np.savez_compressed(
        output_dir / "saliva_zero_shot_predictions.npz",
        labels=labels,
        scores=scores,
        checkpoint_scores=fold_scores,
        positive_edges=positive_edges.T,
        negative_edges=negative_edges.T,
    )
    result = {
        "protocol": "zero-shot external evaluation with zero target-association observations",
        "source_dataset": "data1",
        "target_dataset": args.data_dir,
        "checkpoints": len(checkpoints),
        "positive_samples": int(len(positive_edges)),
        "negative_samples": int(len(negative_edges)),
        **{key: float(value) for key, value in metrics.items()},
    }
    (output_dir / "saliva_zero_shot_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
