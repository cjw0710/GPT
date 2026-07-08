from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def row_normalize(mx: sp.spmatrix) -> sp.csr_matrix:
    mx = mx.tocsr().astype(np.float32)
    rowsum = np.asarray(mx.sum(1)).flatten()
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(inv).dot(mx).tocsr()


def normalize_adj(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(adj.sum(1)).flatten()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    d_mat = sp.diags(inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).tocsr()


def sparse_mx_to_torch_sparse(mx: sp.spmatrix, device: torch.device) -> torch.Tensor:
    mx = mx.tocsr().astype(np.float32)
    crow_indices = torch.from_numpy(mx.indptr.astype(np.int64)).to(device=device)
    col_indices = torch.from_numpy(mx.indices.astype(np.int64)).to(device=device)
    values = torch.from_numpy(mx.data).to(device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
        csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=mx.shape, device=device)
    return csr.to_sparse_coo().coalesce()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def binary_roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary ROC-AUC from class-1 scores without extra dependencies."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"binary_roc_auc expects two-logit outputs, got shape={tuple(logits.shape)}")
    labels_cpu = labels.detach().cpu().to(torch.long)
    scores = (logits[:, 1] - logits[:, 0]).detach().cpu().to(torch.float64)
    positives = labels_cpu == 1
    negatives = labels_cpu == 0
    n_pos = int(positives.sum().item())
    n_neg = int(negatives.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(len(scores), dtype=torch.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end].item() == sorted_scores[start].item():
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[positives].sum().item()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def edge_homophily(adj: sp.spmatrix, labels: np.ndarray) -> float:
    coo = sp.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return 0.0
    same = labels[coo.row] == labels[coo.col]
    return float(np.mean(same))


def feature_variation_signal(adj: sp.spmatrix, features: np.ndarray) -> np.ndarray:
    adj = adj.tocsr().astype(np.float32)
    adj = adj - sp.diags(adj.diagonal())
    neigh = row_normalize(adj).dot(features)
    deg = np.asarray(adj.sum(1)).flatten()
    isolated = deg == 0
    if isolated.any():
        neigh[isolated] = features[isolated]
    diff = np.linalg.norm(features - neigh, axis=1)
    mean = diff.mean()
    std = diff.std() + 1e-8
    z = (diff - mean) / std
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)[:, None]


def stratified_split(
    labels: np.ndarray,
    seed: int,
    train_per_class: int = 20,
    val_per_class: int = 30,
    test_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_train = min(train_per_class, max(1, len(idx) // 5))
        n_val = min(val_per_class, max(1, (len(idx) - n_train) // 4))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train : n_train + n_val])
        remaining = idx[n_train + n_val :]
        if test_per_class is not None:
            remaining = remaining[:test_per_class]
        test_idx.extend(remaining)
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)
