from __future__ import annotations

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harp_gnn.utils import normalize_adj


def main() -> None:
    dataset = "pubmed"
    root = Path("data/raw/planetoid")
    with open(root / f"ind.{dataset}.graph", "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    test_idx = np.loadtxt(root / f"ind.{dataset}.test.index", dtype=np.int64)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32).tocsr()
    adj = adj.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()

    degree = np.asarray(adj.sum(1)).ravel()
    print("raw shape", adj.shape, "nnz", adj.nnz)
    for name, idx in {
        "train": np.arange(60),
        "val": np.arange(60, 560),
        "test": test_idx,
        "all": np.arange(adj.shape[0]),
    }.items():
        split_degree = degree[idx]
        print(
            name,
            "degree min/mean/max",
            float(split_degree.min()),
            float(split_degree.mean()),
            float(split_degree.max()),
            "isolated",
            int((split_degree == 0).sum()),
        )

    for label, matrix in {
        "self": normalize_adj(adj, add_self_loops=True),
        "noself": normalize_adj(adj, add_self_loops=False),
    }.items():
        rowsum = np.asarray(matrix.sum(1)).ravel()
        print(label, "nnz", matrix.nnz, "value min/max", float(matrix.data.min()), float(matrix.data.max()))
        for name, idx in {
            "train": np.arange(60),
            "val": np.arange(60, 560),
            "test": test_idx,
            "all": np.arange(adj.shape[0]),
        }.items():
            split_rowsum = rowsum[idx]
            print(
                label,
                name,
                "rowsum min/mean/max",
                float(split_rowsum.min()),
                float(split_rowsum.mean()),
                float(split_rowsum.max()),
            )
        print(label, "row0 indices", matrix.getrow(0).indices[:12].tolist())
        print(label, "row0 values", matrix.getrow(0).data[:12].tolist())


if __name__ == "__main__":
    main()
