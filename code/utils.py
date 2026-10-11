import random

from sklearn import metrics
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_similarity_matrices(project_root, data_dir_name="data1"):
    data_dir = project_root / data_dir_name
    meta_similarity = pd.read_csv(
        data_dir / "metabolite_simi_network.csv", header=0
    ).to_numpy(dtype=np.float32)
    disease_similarity = pd.read_csv(
        data_dir / "diease_simi_network.csv", header=0
    ).to_numpy(dtype=np.float32)
    return meta_similarity, disease_similarity


def load_similarity_features(
    project_root, feature_dim, seed, similarities=None, data_dir_name="data1"
):
    cache_dir = project_root / "result" / "rclmc_feature_cache" / data_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"similarity_svd_dim{feature_dim}_seed{seed}.npz"
    if cache_path.exists():
        with np.load(cache_path) as archive:
            return archive["meta"], archive["dis"]

    if similarities is None:
        meta_similarity, disease_similarity = load_similarity_matrices(
            project_root, data_dir_name
        )
    else:
        meta_similarity, disease_similarity = similarities

    def reduce(matrix):
        rank = min(feature_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
        reducer = TruncatedSVD(n_components=rank, n_iter=7, random_state=seed)
        reduced = reducer.fit_transform(matrix)
        if rank < feature_dim:
            reduced = np.pad(reduced, ((0, 0), (0, feature_dim - rank)))
        return normalize(reduced, norm="l2").astype(np.float32)

    meta_features = reduce(meta_similarity)
    disease_features = reduce(disease_similarity)
    np.savez_compressed(cache_path, meta=meta_features, dis=disease_features)
    return meta_features, disease_features


def make_positive_folds(association, folds, seed):
    positive_edges = np.argwhere(association > 0).tolist()
    random.Random(seed).shuffle(positive_edges)
    fold_size = len(positive_edges) // folds
    usable = fold_size * folds
    result = [
        np.asarray(positive_edges[index * fold_size:(index + 1) * fold_size])
        for index in range(folds)
    ]
    if usable < len(positive_edges):
        result[-1] = np.concatenate(
            [result[-1], np.asarray(positive_edges[usable:])], axis=0
        )
    return result


def evaluate_scores(labels, scores):
    auc = metrics.roc_auc_score(labels, scores)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, scores)
    prc = metrics.auc(recall_curve, precision_curve)
    predictions = (scores >= 0.5).astype(np.float32)
    return {
        "auc": auc,
        "prc": prc,
        "acc": metrics.accuracy_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions, zero_division=0),
        "recall": metrics.recall_score(labels, predictions, zero_division=0),
        "f1": metrics.f1_score(labels, predictions, zero_division=0),
    }


def constructNet(association_matrix):  # construct association matrix
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def constructNet2(association_matrix, drug_m, meta_n):  # construct association matrix
    # Keep the global node order consistent with every feature tensor and the
    # decoder: diseases first, metabolites second.
    mat1 = torch.cat((drug_m, association_matrix.t()), dim=1)
    mat2 = torch.cat((association_matrix, meta_n), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def constructADJNet(meta_adj, dis_adj, adj):
    if isinstance(adj, list):
        adj = np.array(adj)
    if isinstance(meta_adj, list):
        meta_adj = np.array(meta_adj)
    if isinstance(dis_adj, list):
        dis_adj = np.array(dis_adj)
    mat1 = np.hstack((meta_adj, adj))
    mat2 = np.hstack((adj.T, dis_adj))
    return np.vstack((mat1, mat2))

    return features


def constructFEANet(dis_fea, meta_fea):

    if dis_fea.shape[1] != meta_fea.shape[1]:
        raise ValueError(
            "Feature dimensions do not match between disease and metabolite."
        )

    features = torch.cat([meta_fea, dis_fea], dim=0)  # 按行拼接 [M+N, D]
    return features  # 自动保持 device 和 requires_grad


def pca_reduce(embeddings, n_components=64):

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced


def constructDecoder(association_tensor):
    met_tensor = torch.zeros(association_tensor.shape[0], association_tensor.shape[0])
    dis_tensor = torch.zeros(association_tensor.shape[1], association_tensor.shape[1])
    tensor1 = torch.cat((met_tensor, association_tensor), 1)
    tensor2 = torch.cat((association_tensor.T, dis_tensor), 1)
    adj = torch.cat((tensor1, tensor2), 0)
    return adj


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1
    return link_labels


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)
    ]
    thresholds = np.asmatrix(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1_score, 4),
    ]


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(
            fprs[i],
            tprs[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUC: %.4f" % (i + 1, auc[i]),
        )

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="BlueViolet",
        alpha=0.9,
        label=r"Mean AUC: %.4f $\pm$ %.4f" % (mean_auc, auc_std),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(
            recalls[i],
            precisions[i],
            alpha=0.4,
            linestyle="--",
            label="Fold %d AUPR: %.4f" % (i + 1, prc[i]),
        )

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(
        mean_recall,
        mean_precision,
        color="BlueViolet",
        alpha=0.9,
        label=r"Mean AUPR: %.4f $\pm$ %.4f" % (mean_prc, prc_std),
    )  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle="--", color="black", alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.legend(loc="lower left")
    plt.savefig(directory / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def to_tensor(data, dtype=torch.float32, device=None):

    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        tensor = torch.tensor(data.values, dtype=dtype)
    elif isinstance(data, list):
        tensor = torch.tensor(data, dtype=dtype)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_neg_edge_index).to(device)
    return (
        val_scores.cpu().numpy(),
        val_labels.cpu().numpy(),
        get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy()),
    )


def knn_binarize(sim_matrix, k=2, symmetric=True):
    if isinstance(sim_matrix, pd.DataFrame):
        sim_matrix = sim_matrix.values
    n = sim_matrix.shape[0]
    adj = np.zeros_like(sim_matrix, dtype=int)
    for i in range(n):
        neighbors = np.argsort(sim_matrix[i])[::-1][:k]
        adj[i, neighbors] = 1
    if symmetric:
        adj = np.maximum(adj, adj.T)
    return adj
def read_excel_to_text_list(excel_path, column_name="descripe"):
    """
    从Excel文件中读取指定列的文本，转为字符串列表
    :param excel_path: Excel文件路径（如 "./data.xlsx"）
    :param column_name: 目标列名（默认是descripe）
    :return: 文本列表（空值会被过滤，所有元素转为字符串）
    """
    try:
        # 1. 读取Excel文件（支持.xlsx/.xls格式）
        # engine='openpyxl' 支持.xlsx，engine='xlrd' 支持.xls（需提前安装：pip install openpyxl xlrd）
        df = pd.read_excel(excel_path, engine="openpyxl")

        # 2. 检查列名是否存在（处理大小写/拼写错误提示）
        if column_name not in df.columns:
            raise ValueError(
                f"列名 '{column_name}' 不存在！Excel中的列名列表：{list(df.columns)}"
            )

        # 3. 提取目标列，转为列表，并处理空值/非字符串类型
        text_list = df[column_name].fillna("").astype(str).tolist()

        # 4. 过滤空字符串（可选，根据你的需求调整）
        text_list = [text.strip() for text in text_list if text.strip()]

        print(f"成功读取 {len(text_list)} 条文本数据！")
        return text_list

    except FileNotFoundError:
        print(f"错误：找不到Excel文件，请检查路径是否正确 → {excel_path}")
        return []
    except Exception as e:
        print(f"读取失败：{str(e)}")
        return []
