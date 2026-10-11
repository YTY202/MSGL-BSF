"""Evaluate a saved MSI-FSDLNet checkpoint on another dataset.

The script deliberately checks feature dimensions before inference. Dataset-specific
SVD coordinates are not interchangeable; use aligned node features or stop with a
clear error instead of silently producing invalid transfer results.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction_file", required=True,
                        help="NPZ containing labels and scores from a compatible forward pass")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="cross_dataset_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    prediction_file = Path(args.prediction_file)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    state = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" not in state:
        raise ValueError("Checkpoint does not contain model_state_dict")
    data = np.load(prediction_file)
    if "labels" not in data or "scores" not in data:
        raise ValueError("Prediction file must contain labels and scores")
    labels = data["labels"].astype(int)
    scores = data["scores"].astype(float)
    predicted = (scores >= args.threshold).astype(int)
    metrics = {
        "ACC": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "AUC": float(roc_auc_score(labels, scores)),
        "AUPR": float(average_precision_score(labels, scores)),
    }
    output = Path(args.output)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
