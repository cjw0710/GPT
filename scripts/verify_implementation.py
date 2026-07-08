from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harp_gnn.models import (  # noqa: E402
    HARPAdaptive,
    HARPEgoSep,
    HARPGNN,
    HARPLogitBlend,
    HARPStructGate,
    HARPX,
    MixHop,
    SGC,
)
from src.harp_gnn.run_experiment import _project_root  # noqa: E402
from src.harp_gnn.utils import binary_roc_auc, normalize_adj, sparse_mx_to_torch_sparse  # noqa: E402


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    diff = _max_abs(actual, expected)
    if diff > atol:
        raise AssertionError(f"{name} max abs diff {diff:.6g} exceeds tolerance {atol:.6g}")


def _toy_adjacency() -> sp.csr_matrix:
    rows = np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5], dtype=np.int64)
    cols = np.array([1, 1, 2, 2, 0, 3, 4, 3, 5, 0, 2, 5], dtype=np.int64)
    vals = np.array([1.0, 0.5, 1.0, 1.0, 0.7, 1.3, 1.0, 0.4, 1.0, 0.8, 0.2, 1.0], dtype=np.float32)
    directed = sp.coo_matrix((vals, (rows, cols)), shape=(6, 6), dtype=np.float32).tocsr()
    undirected = (directed + directed.T).tolil()
    undirected.setdiag(0.0)
    csr = undirected.tocsr()
    csr.eliminate_zeros()
    return csr


def verify_sparse_conversion() -> torch.Tensor:
    torch.manual_seed(11)
    norm = normalize_adj(_toy_adjacency(), add_self_loops=True)
    adj = sparse_mx_to_torch_sparse(norm, device=torch.device("cpu"))

    if not adj.is_sparse:
        raise AssertionError("sparse_mx_to_torch_sparse did not return a sparse COO tensor")
    if not adj.is_coalesced():
        raise AssertionError("sparse_mx_to_torch_sparse returned an uncoalesced tensor")
    if adj.dtype != torch.float32:
        raise AssertionError(f"expected float32 sparse values, got {adj.dtype}")

    expected_dense = torch.from_numpy(norm.toarray()).to(dtype=torch.float32)
    _assert_close("sparse conversion dense values", adj.to_dense(), expected_dense, atol=1e-7)

    x = torch.randn((norm.shape[0], 4), dtype=torch.float32)
    expected_spmm = torch.from_numpy(norm.dot(x.numpy())).to(dtype=torch.float32)
    actual_spmm = torch.sparse.mm(adj, x)
    _assert_close("sparse conversion matmul", actual_spmm, expected_spmm, atol=1e-6)
    print("[ok] CSR-backed sparse conversion matches SciPy dense values and matmul.")
    return adj


def verify_harp_projected_equivalence(adj: torch.Tensor) -> None:
    torch.manual_seed(17)
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    gate_signal = torch.linspace(0.1, 0.9, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)

    variants = [
        ("input", "feature"),
        ("input", "scalar"),
        ("branch", "feature"),
    ]
    for gate_type, gate_granularity in variants:
        torch.manual_seed(23)
        reference = HARPGNN(
            in_dim=x.shape[1],
            hidden_dim=7,
            out_dim=3,
            dropout=0.0,
            hops=3,
            use_layer_norm=True,
            branch_mode="both",
            use_gate_signal=True,
            gate_type=gate_type,
            gate_granularity=gate_granularity,
            propagate_projected=False,
        )
        projected = HARPGNN(
            in_dim=x.shape[1],
            hidden_dim=7,
            out_dim=3,
            dropout=0.0,
            hops=3,
            use_layer_norm=True,
            branch_mode="both",
            use_gate_signal=True,
            gate_type=gate_type,
            gate_granularity=gate_granularity,
            propagate_projected=True,
        )
        projected.load_state_dict(reference.state_dict())
        reference.eval()
        projected.eval()

        with torch.no_grad():
            expected = reference(x, adj, gate_signal)
            actual = projected(x, adj, gate_signal)
        _assert_close(
            f"HARP projected equivalence ({gate_type}, {gate_granularity})",
            actual,
            expected,
            atol=5e-6,
        )
    print("[ok] HARP projected propagation is algebraically equivalent on toy graphs.")


def _manual_powers(adj: torch.Tensor, x: torch.Tensor, max_power: int) -> list[torch.Tensor]:
    powers = [x]
    cur = x
    for _ in range(max_power):
        cur = torch.sparse.mm(adj, cur)
        powers.append(cur)
    return powers


def verify_fixed_feature_caches(adj: torch.Tensor) -> None:
    torch.manual_seed(31)
    x = torch.randn((adj.shape[0], 4), dtype=torch.float32)
    manual = _manual_powers(adj, x, 3)

    sgc = SGC(in_dim=x.shape[1], out_dim=3, hops=2)
    sgc.eval()
    with torch.no_grad():
        first = sgc(x, adj)
        cached_x = sgc._cached_x
        second = sgc(x, adj)
    if cached_x is None or sgc._cached_x is not cached_x:
        raise AssertionError("SGC did not reuse the cached propagated features for identical inputs")
    _assert_close("SGC repeated forward", first, second, atol=1e-7)
    _assert_close("SGC cached A^K X", cached_x, manual[2], atol=1e-6)

    mixhop = MixHop(in_dim=x.shape[1], hidden_dim=6, out_dim=3, dropout=0.0, powers=[0, 1, 3])
    mixhop.eval()
    with torch.no_grad():
        first = mixhop(x, adj)
        cached_bases = mixhop._cached_bases
        second = mixhop(x, adj)
    if cached_bases is None or mixhop._cached_bases is not cached_bases:
        raise AssertionError("MixHop did not reuse the cached propagation bases for identical inputs")
    _assert_close("MixHop repeated forward", first, second, atol=1e-7)
    for power in mixhop.powers:
        _assert_close(f"MixHop cached A^{power} X", cached_bases[power], manual[power], atol=1e-6)
    print("[ok] SGC and MixHop fixed-feature propagation caches are stable and numerically correct.")


def verify_harpx_projected_equivalence(adj: torch.Tensor) -> None:
    torch.manual_seed(43)
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    gate_signal = torch.linspace(0.2, 0.8, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)

    torch.manual_seed(47)
    reference = HARPX(
        num_nodes=adj.shape[0],
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        hops=3,
        use_layer_norm=True,
        branch_mode="both",
        use_gate_signal=True,
        gate_type="input",
        gate_granularity="feature",
        propagate_projected=False,
    )
    projected = HARPX(
        num_nodes=adj.shape[0],
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        hops=3,
        use_layer_norm=True,
        branch_mode="both",
        use_gate_signal=True,
        gate_type="input",
        gate_granularity="feature",
        propagate_projected=True,
    )
    projected.load_state_dict(reference.state_dict())
    reference.eval()
    projected.eval()

    with torch.no_grad():
        expected = reference(x, adj, gate_signal)
        actual = projected(x, adj, gate_signal)
    _assert_close("HARP-X projected equivalence", actual, expected, atol=5e-6)

    diagnostics = projected.diagnostics(x, adj, gate_signal)
    if diagnostics.get("gate_features") != 7.0:
        raise AssertionError("HARP-X diagnostics did not expose the feature-wise gate width")
    print("[ok] HARP-X projected propagation and diagnostics are stable on toy graphs.")


def verify_harp_struct_gate_projected_equivalence(adj: torch.Tensor) -> None:
    torch.manual_seed(53)
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    gate_signal = torch.linspace(0.3, 0.7, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)

    torch.manual_seed(59)
    reference = HARPStructGate(
        num_nodes=adj.shape[0],
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        hops=3,
        use_layer_norm=True,
        branch_mode="both",
        use_gate_signal=True,
        gate_granularity="feature",
        propagate_projected=False,
    )
    projected = HARPStructGate(
        num_nodes=adj.shape[0],
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        hops=3,
        use_layer_norm=True,
        branch_mode="both",
        use_gate_signal=True,
        gate_granularity="feature",
        propagate_projected=True,
    )
    projected.load_state_dict(reference.state_dict())
    reference.eval()
    projected.eval()

    with torch.no_grad():
        expected = reference(x, adj, gate_signal)
        actual = projected(x, adj, gate_signal)
    _assert_close("HARP-SGate projected equivalence", actual, expected, atol=5e-6)

    diagnostics = projected.diagnostics(x, adj, gate_signal)
    if diagnostics.get("gate_features") != 7.0:
        raise AssertionError("HARP-SGate diagnostics did not expose the feature-wise gate width")
    if diagnostics.get("structure_norm", 0.0) <= 0.0:
        raise AssertionError("HARP-SGate diagnostics did not expose a positive structure norm")
    print("[ok] HARP-SGate projected propagation and diagnostics are stable on toy graphs.")


def verify_harp_ego_sep(adj: torch.Tensor) -> None:
    torch.manual_seed(61)
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    gate_signal = torch.linspace(0.15, 0.85, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)
    model = HARPEgoSep(
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        hops=2,
        use_layer_norm=True,
        use_gate_signal=True,
        gate_granularity="feature",
    )
    if not getattr(model, "use_no_self_adj", False):
        raise AssertionError("HARP-ESep should request the no-self adjacency")
    model.eval()
    with torch.no_grad():
        logits = model(x, adj, gate_signal)
    if logits.shape != (adj.shape[0], 3):
        raise AssertionError(f"HARP-ESep logits shape mismatch: {tuple(logits.shape)}")
    weights = model.filter_weights()
    _assert_close(
        "HARP-ESep low-weight simplex",
        torch.tensor(sum(weights["low"])),
        torch.tensor(1.0),
        atol=1e-6,
    )
    _assert_close(
        "HARP-ESep high-weight simplex",
        torch.tensor(sum(weights["high"])),
        torch.tensor(1.0),
        atol=1e-6,
    )
    diagnostics = model.diagnostics(x, adj, gate_signal)
    if diagnostics.get("gate_features") != 7.0:
        raise AssertionError("HARP-ESep diagnostics did not expose the feature-wise gate width")
    if diagnostics.get("ego_norm", 0.0) <= 0.0 or diagnostics.get("residual_norm", 0.0) <= 0.0:
        raise AssertionError("HARP-ESep diagnostics did not expose positive ego/residual norms")
    print("[ok] HARP-ESep no-self propagation, filters, and diagnostics are stable on toy graphs.")


def verify_harp_adaptive(adj: torch.Tensor) -> None:
    torch.manual_seed(67)
    no_self_adj = sparse_mx_to_torch_sparse(
        normalize_adj(_toy_adjacency(), add_self_loops=False),
        device=torch.device("cpu"),
    )
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    gate_signal = torch.linspace(0.05, 0.95, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)
    model = HARPAdaptive(
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        harp_hops=3,
        esep_hops=2,
        use_layer_norm=True,
        use_gate_signal=True,
        selector_granularity="scalar",
        propagate_projected=True,
    )
    if not getattr(model, "use_dual_adj", False):
        raise AssertionError("HARP-Adaptive should request both self-loop and no-self adjacency tensors")
    model.eval()
    with torch.no_grad():
        logits = model(x, (adj, no_self_adj), gate_signal)
    if logits.shape != (adj.shape[0], 3):
        raise AssertionError(f"HARP-Adaptive logits shape mismatch: {tuple(logits.shape)}")
    weights = model.filter_weights()
    for key in ("self_low", "self_high", "esep_low", "esep_high"):
        _assert_close(
            f"HARP-Adaptive {key} simplex",
            torch.tensor(sum(weights[key])),
            torch.tensor(1.0),
            atol=1e-6,
        )
    diagnostics = model.diagnostics(x, (adj, no_self_adj), gate_signal)
    if diagnostics.get("selector_features") != 1.0:
        raise AssertionError("HARP-Adaptive scalar selector did not expose a single selector feature")
    if diagnostics.get("adaptive_hidden_norm", 0.0) <= 0.0:
        raise AssertionError("HARP-Adaptive diagnostics did not expose a positive hidden norm")
    if not 0.0 <= diagnostics.get("selector_mean", -1.0) <= 1.0:
        raise AssertionError("HARP-Adaptive selector mean is outside [0, 1]")
    print("[ok] HARP-Adaptive dual-adjacency selector, filters, and diagnostics are stable on toy graphs.")


def verify_harp_logit_blend(adj: torch.Tensor) -> None:
    torch.manual_seed(71)
    no_self_adj = sparse_mx_to_torch_sparse(
        normalize_adj(_toy_adjacency(), add_self_loops=False),
        device=torch.device("cpu"),
    )
    x = torch.randn((adj.shape[0], 5), dtype=torch.float32)
    y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    idx = torch.arange(adj.shape[0], dtype=torch.long)
    gate_signal = torch.linspace(0.05, 0.95, steps=adj.shape[0], dtype=torch.float32).unsqueeze(-1)
    model = HARPLogitBlend(
        in_dim=x.shape[1],
        hidden_dim=7,
        out_dim=3,
        dropout=0.0,
        harp_hops=3,
        esep_hops=2,
        use_layer_norm=True,
        auxiliary_weight=0.2,
        propagate_projected=True,
    )
    if not getattr(model, "use_dual_adj", False):
        raise AssertionError("HARP-Blend should request both self-loop and no-self adjacency tensors")
    model.eval()
    with torch.no_grad():
        logits = model(x, (adj, no_self_adj), gate_signal)
    if logits.shape != (adj.shape[0], 3):
        raise AssertionError(f"HARP-Blend logits shape mismatch: {tuple(logits.shape)}")
    model.train()
    aux = model.auxiliary_loss(x, (adj, no_self_adj), gate_signal, y, idx)
    if float(aux.item()) <= 0.0:
        raise AssertionError("HARP-Blend auxiliary loss should be positive on toy labels")
    diagnostics = model.diagnostics(x, (adj, no_self_adj), gate_signal)
    if not 0.0 <= diagnostics.get("blend_weight", -1.0) <= 1.0:
        raise AssertionError("HARP-Blend weight is outside [0, 1]")
    print("[ok] HARP-Blend dual-adjacency logit mixture and auxiliary branch loss are stable on toy graphs.")


def verify_binary_roc_auc() -> None:
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    logits = torch.tensor(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )
    auc = binary_roc_auc(logits, labels)
    if abs(auc - 1.0) > 1e-12:
        raise AssertionError(f"Expected perfect ROC-AUC, got {auc:.6f}")

    tied_logits = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    tied_auc = binary_roc_auc(tied_logits, labels)
    if abs(tied_auc - 0.875) > 1e-12:
        raise AssertionError(f"Expected tie-aware ROC-AUC 0.875, got {tied_auc:.6f}")
    try:
        binary_roc_auc(torch.randn(4, 3), labels)
    except ValueError:
        pass
    else:
        raise AssertionError("binary_roc_auc should reject non-binary logits")
    print("[ok] Binary ROC-AUC metric is tie-aware and validates binary logits.")


def verify_config_project_root() -> None:
    root = Path(__file__).resolve().parents[1]
    config_paths = [
        root / "configs" / "critical_heterophily_binary_smoke.yaml",
        root / "configs" / "in_progress" / "critical_heterophily_binary_tolokers.yaml",
    ]
    for config_path in config_paths:
        resolved = _project_root(config_path)
        if resolved != root:
            raise AssertionError(f"Config root mismatch for {config_path}: got {resolved}, expected {root}")
    print("[ok] Experiment configs resolve to the project root from root and nested config directories.")


def main() -> None:
    torch.set_num_threads(1)
    adj = verify_sparse_conversion()
    verify_harp_projected_equivalence(adj)
    verify_fixed_feature_caches(adj)
    verify_harpx_projected_equivalence(adj)
    verify_harp_struct_gate_projected_equivalence(adj)
    verify_harp_ego_sep(adj)
    verify_harp_adaptive(adj)
    verify_harp_logit_blend(adj)
    verify_binary_roc_auc()
    verify_config_project_root()
    print("[done] Implementation invariants verified.")


if __name__ == "__main__":
    main()
