from __future__ import annotations

import pickle
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from .utils import (
    edge_homophily,
    feature_variation_signal,
    normalize_adj,
    row_normalize,
    sparse_mx_to_torch_sparse,
    stratified_split,
)


PLANETOID_BASE_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")
GEOM_GCN_BASE_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data"
DGL_DATASET_BASE_URL = "https://data.dgl.ai/dataset"
HETEROPHILOUS_GRAPHS_BASE_URL = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"
)
GEOM_GCN_DATASETS = {
    "texas": "texas",
    "wisconsin": "wisconsin",
    "cornell": "cornell",
    "chameleon": "chameleon",
    "squirrel": "squirrel",
    "film": "actor",
    "actor": "actor",
}
GEOM_GCN_CORE_FILES = ("out1_node_feature_label.txt", "out1_graph_edges.txt")
HETEROPHILOUS_GRAPH_DATASETS = {
    "roman-empire": "roman_empire",
    "roman_empire": "roman_empire",
    "amazon-ratings": "amazon_ratings",
    "amazon_ratings": "amazon_ratings",
    "minesweeper": "minesweeper",
    "tolokers": "tolokers",
    "questions": "questions",
}


@dataclass
class GraphDataset:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    adj_norm: torch.Tensor
    adj_no_self_norm: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    gate_signal: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)


def _to_dense_float(mx: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(mx):
        return mx.toarray().astype(np.float32)
    return np.asarray(mx, dtype=np.float32)


def _build_dataset(
    name: str,
    features: sp.spmatrix | np.ndarray,
    labels: np.ndarray,
    adj: sp.spmatrix,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    metadata: Dict[str, Any] | None = None,
    normalize_features: bool = True,
) -> GraphDataset:
    if normalize_features and sp.issparse(features):
        features = row_normalize(features)
    features_dense = _to_dense_float(features)
    adj = adj.tolil().astype(np.float32)
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()

    gate_signal = feature_variation_signal(adj, features_dense)
    adj_norm = normalize_adj(adj, add_self_loops=True)
    adj_no_self_norm = normalize_adj(adj, add_self_loops=False)

    meta = dict(metadata or {})
    meta["num_nodes"] = int(adj.shape[0])
    meta["num_edges"] = int(sp.triu(adj, k=1).nnz)
    meta["edge_homophily"] = edge_homophily(adj, labels)

    return GraphDataset(
        name=name,
        x=torch.tensor(features_dense, dtype=torch.float32, device=device),
        y=torch.tensor(labels, dtype=torch.long, device=device),
        adj_norm=sparse_mx_to_torch_sparse(adj_norm, device=device),
        adj_no_self_norm=sparse_mx_to_torch_sparse(adj_no_self_norm, device=device),
        train_idx=torch.tensor(train_idx, dtype=torch.long, device=device),
        val_idx=torch.tensor(val_idx, dtype=torch.long, device=device),
        test_idx=torch.tensor(test_idx, dtype=torch.long, device=device),
        gate_signal=torch.tensor(gate_signal, dtype=torch.float32, device=device),
        metadata=meta,
    )


def _download_file(url: str, out_path: Path, timeout: int = 60) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            tmp_path.write_bytes(response.read())
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return out_path


def load_heterophilous_graph(
    spec: Dict[str, Any],
    data_root: Path,
    seed: int,
    device: torch.device,
) -> GraphDataset:
    requested = str(spec.get("dataset", spec.get("name", "roman-empire"))).lower()
    if requested not in HETEROPHILOUS_GRAPH_DATASETS:
        raise ValueError(f"Unsupported heterophilous-graphs dataset: {requested}")
    dataset_file = HETEROPHILOUS_GRAPH_DATASETS[requested]
    out_path = data_root / "raw" / "heterophilous_graphs" / f"{dataset_file}.npz"
    _download_file(
        f"{HETEROPHILOUS_GRAPHS_BASE_URL}/{dataset_file}.npz",
        out_path,
        timeout=180,
    )

    data = np.load(out_path)
    features = np.asarray(data["node_features"], dtype=np.float32)
    labels = np.asarray(data["node_labels"], dtype=np.int64)
    edges = np.asarray(data["edges"], dtype=np.int64)
    train_masks = np.asarray(data["train_masks"], dtype=bool)
    val_masks = np.asarray(data["val_masks"], dtype=bool)
    test_masks = np.asarray(data["test_masks"], dtype=bool)

    if train_masks.ndim != 2:
        raise ValueError(f"Expected 2D train masks in {out_path}, got shape={train_masks.shape}")
    if train_masks.shape[1] != len(labels) and train_masks.shape[0] == len(labels):
        train_masks = train_masks.T
        val_masks = val_masks.T
        test_masks = test_masks.T
    if train_masks.shape[1] != len(labels):
        raise ValueError(
            f"Mask width mismatch in {out_path}: masks={train_masks.shape}, nodes={len(labels)}"
        )
    split_id = int(seed) % int(train_masks.shape[0])

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((values, (rows, cols)), shape=(len(labels), len(labels))).tocsr()

    return _build_dataset(
        name=str(spec.get("name", requested.replace("_", "-"))),
        features=features,
        labels=labels,
        adj=adj,
        train_idx=np.flatnonzero(train_masks[split_id]).astype(np.int64),
        val_idx=np.flatnonzero(val_masks[split_id]).astype(np.int64),
        test_idx=np.flatnonzero(test_masks[split_id]).astype(np.int64),
        device=device,
        metadata={
            "dataset_type": "heterophilous_graphs",
            "source": dataset_file,
            "split_id": split_id,
            "make_undirected": True,
        },
        normalize_features=False,
    )


def make_contextual_sbm(spec: Dict[str, Any], seed: int, device: torch.device) -> GraphDataset:
    params = spec.get("params", {})
    num_classes = int(params.get("num_classes", 3))
    nodes_per_class = int(params.get("nodes_per_class", 120))
    feature_dim = int(params.get("feature_dim", 64))
    avg_degree = float(params.get("avg_degree", 10.0))
    homophily = float(params.get("homophily", 0.5))
    feature_noise = float(params.get("feature_noise", 1.0))
    train_per_class = int(params.get("train_per_class", 20))
    val_per_class = int(params.get("val_per_class", 30))

    rng = np.random.default_rng(seed)
    n = num_classes * nodes_per_class
    labels = np.repeat(np.arange(num_classes), nodes_per_class).astype(np.int64)

    p_in = avg_degree * homophily / max(nodes_per_class - 1, 1)
    p_out = avg_degree * (1.0 - homophily) / max(n - nodes_per_class, 1)
    p_in = min(max(p_in, 0.0), 1.0)
    p_out = min(max(p_out, 0.0), 1.0)

    rows, cols = [], []
    for i in range(n):
        same = labels[i] == labels[i + 1 :]
        probs = np.where(same, p_in, p_out)
        draw = rng.random(n - i - 1) < probs
        js = np.where(draw)[0] + i + 1
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
    row = np.array(rows + cols, dtype=np.int64)
    col = np.array(cols + rows, dtype=np.int64)
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    centers = rng.normal(size=(num_classes, feature_dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    features = centers[labels] + feature_noise * rng.normal(size=(n, feature_dim)).astype(np.float32)

    train_idx, val_idx, test_idx = stratified_split(
        labels,
        seed=seed,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
    )
    name = spec.get("name", f"synthetic_h{homophily:.2f}")
    return _build_dataset(
        name=name,
        features=features,
        labels=labels,
        adj=adj,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        device=device,
        metadata={
            "dataset_type": "synthetic",
            "target_homophily": homophily,
            "p_in": p_in,
            "p_out": p_out,
            "seed": seed,
        },
    )


def _download_planetoid_file(dataset: str, suffix: str, out_dir: Path) -> Path:
    filename = f"ind.{dataset}.{suffix}"
    out_path = out_dir / filename
    url = f"{PLANETOID_BASE_URL}/{filename}"
    return _download_file(url, out_path)


def _parse_index_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_planetoid(spec: Dict[str, Any], data_root: Path, device: torch.device) -> GraphDataset:
    dataset = str(spec.get("dataset", spec.get("name", "cora"))).lower()
    if dataset not in {"cora", "citeseer", "pubmed"}:
        raise ValueError(f"Unsupported Planetoid dataset: {dataset}")

    out_dir = data_root / "raw" / "planetoid"
    paths = {suffix: _download_planetoid_file(dataset, suffix, out_dir) for suffix in PLANETOID_FILES}

    objects = []
    for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph"):
        with open(paths[suffix], "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = np.array(_parse_index_file(paths["test.index"]), dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        full_range = np.arange(test_idx_reorder.min(), test_idx_reorder.max() + 1)
        tx_ext = sp.lil_matrix((len(full_range), x.shape[1]), dtype=np.float32)
        tx_ext[test_idx_range - test_idx_reorder.min(), :] = tx
        ty_ext = np.zeros((len(full_range), y.shape[1]), dtype=np.float32)
        ty_ext[test_idx_range - test_idx_reorder.min(), :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1).astype(np.int64)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32).tocsr()
    train_idx = np.arange(y.shape[0], dtype=np.int64)
    val_idx = np.arange(y.shape[0], y.shape[0] + 500, dtype=np.int64)
    test_idx = test_idx_range.astype(np.int64)

    name = spec.get("name", dataset)
    return _build_dataset(
        name=name,
        features=features,
        labels=labels,
        adj=adj,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        device=device,
        metadata={"dataset_type": "planetoid", "source": dataset},
        normalize_features=bool(spec.get("normalize_features", True)),
    )


def _download_geom_gcn_file(dataset_dir: str, filename: str, out_dir: Path) -> Path:
    local_path = out_dir / dataset_dir / filename
    if local_path.exists():
        return local_path
    _ensure_dgl_geom_gcn_zip(dataset_dir, out_dir)
    if local_path.exists():
        return local_path
    url = f"{GEOM_GCN_BASE_URL}/{dataset_dir}/{filename}"
    return _download_file(url, local_path)


def _ensure_dgl_geom_gcn_zip(dataset_dir: str, out_dir: Path) -> None:
    dataset_path = out_dir / dataset_dir
    dataset_path.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{dataset_dir}.zip"
    url = f"{DGL_DATASET_BASE_URL}/{dataset_dir}.zip"
    _download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_path)


def _parse_geom_gcn_nodes(path: Path) -> tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    all_lines = path.read_text(encoding="utf-8").splitlines()
    header = all_lines[0]
    lines = all_lines[1:]
    feature_dim_match = re.search(r"feature_amount:(\d+)", header)
    feature_dim = int(feature_dim_match.group(1)) if feature_dim_match else None
    node_ids, raw_features, labels = [], [], []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        node_ids.append(int(parts[0]))
        raw_features.append([float(value) for value in parts[1].split(",")])
        labels.append(int(parts[2]))

    order = np.argsort(np.array(node_ids, dtype=np.int64))
    sorted_node_ids = np.array(node_ids, dtype=np.int64)[order]
    sorted_raw_features = [raw_features[int(i)] for i in order]
    sorted_labels = np.array(labels, dtype=np.int64)[order]

    is_sparse_index_format = feature_dim is not None and any(
        len(values) != feature_dim for values in sorted_raw_features
    )
    if is_sparse_index_format:
        features = np.zeros((len(sorted_raw_features), feature_dim), dtype=np.float32)
        for row, values in enumerate(sorted_raw_features):
            for value in values:
                col = int(value)
                if 0 <= col < feature_dim:
                    features[row, col] = 1.0
    else:
        features = np.array(sorted_raw_features, dtype=np.float32)

    node_id_to_pos = {int(node_id): int(pos) for pos, node_id in enumerate(sorted_node_ids)}
    return features, sorted_labels, node_id_to_pos


def _parse_geom_gcn_edges(path: Path, node_id_to_pos: Dict[int, int], make_undirected: bool) -> sp.csr_matrix:
    rows, cols = [], []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        src, dst = [int(value) for value in line.split("\t")]
        rows.append(node_id_to_pos[src])
        cols.append(node_id_to_pos[dst])
    if make_undirected:
        rows, cols = rows + cols, cols + rows
    data = np.ones(len(rows), dtype=np.float32)
    num_nodes = len(node_id_to_pos)
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def _mask_to_idx(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.astype(bool))[0].astype(np.int64)


def load_geom_gcn(spec: Dict[str, Any], data_root: Path, seed: int, device: torch.device) -> GraphDataset:
    dataset = str(spec.get("dataset", spec.get("name", "texas"))).lower()
    if dataset not in GEOM_GCN_DATASETS:
        supported = ", ".join(sorted(GEOM_GCN_DATASETS))
        raise ValueError(f"Unsupported Geom-GCN dataset: {dataset}. Supported: {supported}")

    dataset_dir = GEOM_GCN_DATASETS[dataset]
    split_id = int(spec.get("split", seed % 10))
    if not 0 <= split_id <= 9:
        raise ValueError("Geom-GCN split must be in [0, 9]")

    out_dir = data_root / "raw" / "geom_gcn"
    paths = {
        filename: _download_geom_gcn_file(dataset_dir, filename, out_dir)
        for filename in GEOM_GCN_CORE_FILES
    }
    split_filename = f"{dataset_dir}_split_0.6_0.2_{split_id}.npz"
    split_path = _download_geom_gcn_file(dataset_dir, split_filename, out_dir)

    features, labels, node_id_to_pos = _parse_geom_gcn_nodes(paths["out1_node_feature_label.txt"])
    adj = _parse_geom_gcn_edges(
        paths["out1_graph_edges.txt"],
        node_id_to_pos=node_id_to_pos,
        make_undirected=bool(spec.get("make_undirected", True)),
    )
    split = np.load(split_path)
    train_idx = _mask_to_idx(split["train_mask"])
    val_idx = _mask_to_idx(split["val_mask"])
    test_idx = _mask_to_idx(split["test_mask"])

    return _build_dataset(
        name=spec.get("name", dataset),
        features=features,
        labels=labels,
        adj=adj,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        device=device,
        metadata={
            "dataset_type": "geom_gcn",
            "source": dataset_dir,
            "split_id": split_id,
            "make_undirected": bool(spec.get("make_undirected", True)),
        },
    )


def load_dataset(spec: Dict[str, Any], data_root: Path, seed: int, device: torch.device) -> GraphDataset:
    dataset_type = str(spec.get("type", "synthetic")).lower()
    if dataset_type == "synthetic":
        return make_contextual_sbm(spec, seed=seed, device=device)
    if dataset_type == "planetoid":
        return load_planetoid(spec, data_root=data_root, device=device)
    if dataset_type in {"geom_gcn", "geometric", "heterophily"}:
        return load_geom_gcn(spec, data_root=data_root, seed=seed, device=device)
    if dataset_type in {"heterophilous_graphs", "critical_heterophily", "platonov"}:
        return load_heterophilous_graph(spec, data_root=data_root, seed=seed, device=device)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
