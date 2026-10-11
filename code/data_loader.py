from param import *
from pathlib import Path
from utils import *
from torch_geometric.data import Data
from LLMEncoder import *
import torch.nn.functional as F
import hashlib

project_root = Path(__file__).parent.parent
data_dir = project_root / "data1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _feature_cache_path(args):
    source_files = (
        data_dir / "disease_describe.xlsx",
        data_dir / "Meta_describe.xlsx",
        data_dir / "diease_simi_network.csv",
        data_dir / "metabolite_simi_network.csv",
    )
    fingerprint = "|".join(
        f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in source_files
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    cache_dir = project_root / "result" / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"dual_view_seed{args.seed}_{digest}.npz"


def load_data():
    args = parameter_parser()
    dis_node_text = read_excel_to_text_list("../data1/disease_describe.xlsx", "Describe")
    meta_node_text = read_excel_to_text_list("../data1/Meta_describe.xlsx", "Describe")

    cache_path = _feature_cache_path(args)
    cached = np.load(cache_path) if cache_path.exists() else None
    if cached is None:
        encoder = DualViewTextEncoder(out_dim=64, random_state=args.seed)
        all_node_embeddings = encoder.fit_transform(
            dis_node_text + meta_node_text,
            device="cpu",
        ).numpy()
        dis_node_embedding_array = all_node_embeddings[:len(dis_node_text)]
        meta_node_embedding_array = all_node_embeddings[len(dis_node_text):]
    else:
        dis_node_embedding_array = cached["dis_text"]
        meta_node_embedding_array = cached["meta_text"]
    dis_node_embeddings = torch.from_numpy(dis_node_embedding_array).to(device)
    meta_node_embeddings = torch.from_numpy(meta_node_embedding_array).to(device)
    print(dis_node_embeddings.shape)
    print(meta_node_embeddings.shape)

    # Read the adjacency matrix (disease-metabolite association)
    Adj = pd.read_csv(data_dir / "association_matrix.csv", header=0)
    Ori_adj = np.asmatrix(np.where(Adj == 1))  # 交叉验证用
    print(f"Disease metabolite adjacency matrix:\n{Adj}")

    # Read the disease similarity matrix
    Dis_simi = pd.read_csv(data_dir / "diease_simi_network.csv", header=0)
    print(f"Disease Similarity Matrix:\n{Dis_simi}")
    Dis_adj=knn_binarize(Dis_simi, k=args.kneighbor)
    print(Dis_adj.shape)


    Ori_dis_index_matrix = np.asmatrix(np.where(Dis_adj == 1))
    Dis_index_matrix = torch.tensor(Ori_dis_index_matrix, dtype=torch.long).to(device)

    Dis_features = pd.read_csv(data_dir / "diease_simi_network.csv", header=0)
    if cached is None:
        dis_feature_array = pca_reduce(Dis_features, n_components=64)
    else:
        dis_feature_array = cached["dis_pca"]
    Dis_features = torch.tensor(dis_feature_array, dtype=torch.float).to(device)

    Dis_features_text = dis_node_embeddings.float()
    # Read the metabolite similarity matrix
    Meta_simi = pd.read_csv(data_dir / "metabolite_simi_network.csv", header=0)
    print(f"Metabolite similarity matrix:\n{Meta_simi}")

    Meta_adj=knn_binarize(Meta_simi,k=args.kneighbor)
    Ori_meta_index_matrix = np.asmatrix(np.where(Meta_adj == 1))
    Meta_index_matrix = torch.tensor(Ori_meta_index_matrix, dtype=torch.long).to(device)

    Meta_features = pd.read_csv(data_dir / "metabolite_simi_network.csv", header=0)
    if cached is None:
        Meta_features = pca_reduce(Meta_features, n_components=64)
        np.savez_compressed(
            cache_path,
            dis_text=dis_node_embedding_array,
            meta_text=meta_node_embedding_array,
            dis_pca=np.asarray(dis_feature_array, dtype=np.float32),
            meta_pca=np.asarray(Meta_features, dtype=np.float32),
        )
        print(f"Saved feature cache: {cache_path}")
    else:
        Meta_features = cached["meta_pca"]
        print(f"Loaded feature cache: {cache_path}")
    Meta_features = torch.tensor(Meta_features, dtype=torch.float).to(device)

    Meta_features_text = meta_node_embeddings.float()

    # Early multimodal fusion: lexical semantics and similarity-network
    # coordinates remain distinguishable channels instead of being averaged.
    Dis_attribute_features = torch.cat(
        [Dis_features_text, F.normalize(Dis_features, dim=1)], dim=1
    )
    Meta_attribute_features = torch.cat(
        [Meta_features_text, F.normalize(Meta_features, dim=1)], dim=1
    )

    Meta_data = Data(x=Meta_attribute_features,edge_index=Meta_index_matrix,Meta_adj=Meta_adj,Meta_simi=Meta_simi)
    Dis_data = Data(x=Dis_attribute_features, edge_index=Dis_index_matrix, Dis_adj=Dis_adj, Dis_simi=Dis_simi)

    feature = torch.cat((Dis_features, Meta_features))
    Dis_adj=to_tensor(Dis_adj)
    Meta_adj=to_tensor(Meta_adj)



    return Adj, Dis_data, Meta_data, Ori_adj, Dis_adj, Meta_adj, feature
