from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harp_gnn.data import (  # noqa: E402
    GEOM_GCN_CORE_FILES,
    GEOM_GCN_DATASETS,
    _download_geom_gcn_file,
    _parse_geom_gcn_edges,
    _parse_geom_gcn_nodes,
)
from src.harp_gnn.utils import normalize_adj  # noqa: E402


def _coo_row_sums(matrix: sp.spmatrix) -> tuple[int, float, float]:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack((coo.row, coo.col)).astype(np.int64), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, tuple(coo.shape)).coalesce()
    coalesced_indices = sparse.indices().cpu().numpy()
    coalesced_values = sparse.values().cpu().numpy()
    row_sums = np.zeros(coo.shape[0], dtype=np.float64)
    np.add.at(row_sums, coalesced_indices[0], coalesced_values)
    expected = np.asarray(matrix.sum(1)).ravel()
    return sparse._nnz(), float(np.max(np.abs(row_sums - expected))), float(coalesced_values.max())


def _csr_row_sums(matrix: sp.spmatrix) -> tuple[int, float, float]:
    csr = matrix.tocsr().astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
        sparse = torch.sparse_csr_tensor(
            torch.tensor(csr.indptr, dtype=torch.int64),
            torch.tensor(csr.indices, dtype=torch.int64),
            torch.tensor(csr.data, dtype=torch.float32),
            size=csr.shape,
        )
    row_sums = (sparse @ torch.ones((csr.shape[1], 1))).squeeze().cpu().numpy()
    expected = np.asarray(matrix.sum(1)).ravel()
    return sparse._nnz(), float(np.max(np.abs(row_sums - expected))), float(sparse.values().max().item())


def _planetoid_adj(dataset: str, data_root: Path) -> sp.csr_matrix:
    graph_path = data_root / "raw" / "planetoid" / f"ind.{dataset}.graph"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32).tocsr()
    adj = adj.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    return adj


def _geom_adj(dataset: str, data_root: Path) -> sp.csr_matrix:
    dataset_dir = GEOM_GCN_DATASETS[dataset]
    out_dir = data_root / "raw" / "geom_gcn"
    paths = {
        filename: _download_geom_gcn_file(dataset_dir, filename, out_dir)
        for filename in GEOM_GCN_CORE_FILES
    }
    _, _, node_id_to_pos = _parse_geom_gcn_nodes(paths["out1_node_feature_label.txt"])
    return _parse_geom_gcn_edges(
        paths["out1_graph_edges.txt"],
        node_id_to_pos=node_id_to_pos,
        make_undirected=True,
    )


def _iter_datasets(kind: str) -> Iterable[tuple[str, sp.csr_matrix]]:
    data_root = Path("data")
    if kind in {"all", "planetoid"}:
        for dataset in ("cora", "citeseer", "pubmed"):
            yield f"planetoid/{dataset}", _planetoid_adj(dataset, data_root)
    if kind in {"all", "geom_gcn"}:
        for dataset in ("texas", "wisconsin", "cornell", "chameleon", "squirrel", "actor"):
            yield f"geom_gcn/{dataset}", _geom_adj(dataset, data_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch sparse COO and CSR conversion fidelity.")
    parser.add_argument("--kind", choices=["all", "planetoid", "geom_gcn"], default="all")
    args = parser.parse_args()

    print("dataset,scipy_nnz,coo_nnz,coo_max_rowsum_diff,coo_max_value,csr_nnz,csr_max_rowsum_diff,csr_max_value")
    for name, adj in _iter_datasets(args.kind):
        norm = normalize_adj(adj, add_self_loops=True)
        coo_nnz, coo_diff, coo_max = _coo_row_sums(norm)
        csr_nnz, csr_diff, csr_max = _csr_row_sums(norm)
        print(
            f"{name},{norm.nnz},{coo_nnz},{coo_diff:.6g},{coo_max:.6g},"
            f"{csr_nnz},{csr_diff:.6g},{csr_max:.6g}"
        )


if __name__ == "__main__":
    main()
