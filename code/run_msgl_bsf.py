"""One-command launcher for the final MSGL-BSF configuration."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run MSGL-BSF with the fixed configuration.")
    parser.add_argument("--data_dir", default="data1",
                        help="Dataset directory, e.g. data1 or data2_saliva_mesh")
    parser.add_argument("--bert_model_path", default="/root/BiomedBERT",
                        help="BioMedBERT directory; change only when running locally")
    args = parser.parse_args()

    requested_data_dir = args.data_dir.replace("\\", "/").strip("/")
    bundled_data_dir = root / "data" / requested_data_dir
    data_dir = f"data/{requested_data_dir}" if bundled_data_dir.is_dir() else requested_data_dir
    dataset_name = Path(requested_data_dir).name
    run_name = "msgl_bsf_" + dataset_name
    prediction_dir = f"result/{run_name}"
    command = [
        sys.executable, "code/train_liquid_neighbor_sequence.py",
        "--data_dir", data_dir,
        "--neighbors", "64",
        "--feature_dim", "64",
        "--hidden_dim", "48",
        "--epochs", "120",
        "--patience", "25",
        "--interval_mode", "fixed",
        "--sequence_mode", "bidirectional",
        "--timescale_mode", "dual",
        "--multi_scale_neighbors", "16", "32", "64",
        "--disable_static_shortcuts",
        "--use_text_semantic_branch",
        "--semantic_encoder", "bert",
        "--semantic_fusion_mode", "feature",
        "--bert_model_path", args.bert_model_path,
        "--bert_batch_size", "32",
        "--bert_max_length", "256",
        "--prediction_dir", prediction_dir,
        "--run_name", run_name,
    ]
    print("Running MSGL-BSF")
    print(f"Dataset: {data_dir}")
    print(f"Predictions: {prediction_dir}")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
