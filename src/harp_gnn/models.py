from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def spmm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(adj, x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return spmm(adj, self.linear(x))


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        bias: bool = False,
    ):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim, bias=bias)
        self.gc2 = GraphConvolution(hidden_dim, out_dim, bias=bias)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gc2(x, adj)


class SGC(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hops: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.hops = hops
        self._cached_key: tuple[int, int, torch.device, torch.dtype] | None = None
        self._cached_x: torch.Tensor | None = None

    def _propagated_features(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        key = (id(x), id(adj), x.device, x.dtype)
        if self._cached_key == key and self._cached_x is not None:
            return self._cached_x
        propagated = x
        for _ in range(self.hops):
            propagated = spmm(adj, propagated)
        self._cached_key = key
        self._cached_x = propagated.detach()
        return self._cached_x

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        return self.linear(self._propagated_features(x, adj))


class APPNP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 10,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, dropout=dropout)
        self.hops = hops
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        h0 = self.mlp(x, adj)
        z = h0
        for _ in range(self.hops):
            z = (1.0 - self.alpha) * spmm(adj, z) + self.alpha * h0
        return z


class MixHop(nn.Module):
    """Multi-hop feature mixing with separate projections per adjacency power."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        powers: list[int] | None = None,
    ):
        super().__init__()
        self.powers = powers or [0, 1, 2]
        if min(self.powers) < 0:
            raise ValueError("MixHop powers must be non-negative")
        self.dropout = dropout
        self.projections = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in self.powers])
        self.classifier = nn.Linear(hidden_dim * len(self.powers), out_dim)
        self._cached_key: tuple[int, int, torch.device, torch.dtype] | None = None
        self._cached_bases: list[torch.Tensor] | None = None

    def _propagation_bases(self, x: torch.Tensor, adj: torch.Tensor) -> list[torch.Tensor]:
        key = (id(x), id(adj), x.device, x.dtype)
        if self._cached_key == key and self._cached_bases is not None:
            return self._cached_bases
        max_power = max(self.powers)
        bases = [x.detach()]
        cur = x
        for _ in range(max_power):
            cur = spmm(adj, cur)
            bases.append(cur.detach())
        self._cached_key = key
        self._cached_bases = bases
        return bases

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        bases = self._propagation_bases(x, adj)
        hidden = []
        for power, projection in zip(self.powers, self.projections):
            hidden.append(F.relu(projection(bases[power])))
        h = torch.cat(hidden, dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class GPRGNN(nn.Module):
    """Generalized PageRank GNN with learnable propagation coefficients."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 10,
        alpha: float = 0.1,
        init: str = "ppr",
    ):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim, out_dim, dropout=dropout)
        self.hops = hops
        self.gamma = nn.Parameter(self._init_gamma(hops, alpha, init))

    @staticmethod
    def _init_gamma(hops: int, alpha: float, init: str) -> torch.Tensor:
        init = init.lower()
        if init == "ppr":
            coeffs = [alpha * (1.0 - alpha) ** k for k in range(hops)]
            coeffs.append((1.0 - alpha) ** hops)
            return torch.tensor(coeffs, dtype=torch.float32)
        if init == "sgc":
            coeffs = torch.zeros(hops + 1, dtype=torch.float32)
            coeffs[-1] = 1.0
            return coeffs
        if init == "uniform":
            return torch.full((hops + 1,), 1.0 / (hops + 1), dtype=torch.float32)
        raise ValueError(f"Unknown GPR-GNN init: {init}")

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, adj)
        out = self.gamma[0] * z
        for k in range(1, self.hops + 1):
            z = spmm(adj, z)
            out = out + self.gamma[k] * z
        return out


class FAGCNStyle(nn.Module):
    """Sparse frequency-adaptive GCN-style baseline.

    This implementation follows the FAGCN idea of signed, edge-conditioned
    propagation weights, but is intentionally kept as an in-repository baseline
    scaffold rather than an official-code reproduction.
    """

    use_no_self_adj = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        layers: int = 2,
        epsilon: float = 0.3,
        negative_slope: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("FAGCN-style baseline requires at least one propagation layer")
        self.dropout = dropout
        self.layers = layers
        self.epsilon = epsilon
        self.negative_slope = negative_slope
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.attn_src = nn.ModuleList(nn.Linear(hidden_dim, 1, bias=False) for _ in range(layers))
        self.attn_dst = nn.ModuleList(nn.Linear(hidden_dim, 1, bias=False) for _ in range(layers))
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity() for _ in range(layers)
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _signed_message(self, h: torch.Tensor, adj: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        adj = adj.coalesce()
        row, col = adj.indices()
        values = adj.values()
        src_score = self.attn_src[layer](h).squeeze(-1)
        dst_score = self.attn_dst[layer](h).squeeze(-1)
        edge_gate = torch.tanh(
            F.leaky_relu(src_score[row] + dst_score[col], negative_slope=self.negative_slope)
        )
        signed_adj = torch.sparse_coo_tensor(
            adj.indices(),
            values * edge_gate,
            size=adj.shape,
            dtype=h.dtype,
            device=h.device,
        ).coalesce()
        out = spmm(signed_adj, h)
        return out, edge_gate

    def _hidden(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        collect_gates: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = F.relu(self.encoder(F.dropout(x, p=self.dropout, training=self.training)))
        gates: list[torch.Tensor] = []
        for layer in range(self.layers):
            propagated, edge_gate = self._signed_message(
                F.dropout(h, p=self.dropout, training=self.training),
                adj,
                layer,
            )
            if collect_gates:
                gates.append(edge_gate.detach())
            h = self.epsilon * h + propagated
            h = self.norms[layer](F.relu(h))
        return h, gates

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self._hidden(x, adj, collect_gates=False)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, gates = self._hidden(x, adj, collect_gates=True)
                values = torch.cat(gates) if gates else torch.empty(0, dtype=x.dtype, device=x.device)
                if values.numel() == 0:
                    return {}
                return {
                    "fagcn_edge_gate_mean": float(values.mean().item()),
                    "fagcn_edge_gate_abs_mean": float(values.abs().mean().item()),
                    "fagcn_edge_gate_positive_frac": float((values > 0).float().mean().item()),
                    "fagcn_edge_gate_min": float(values.min().item()),
                    "fagcn_edge_gate_max": float(values.max().item()),
                    "fagcn_layers": float(self.layers),
                    "fagcn_epsilon": float(self.epsilon),
                }
        finally:
            self.train(was_training)


class LINKX(nn.Module):
    """Sparse-friendly LINKX-style feature and adjacency encoder."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.adj_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.adj_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.x_linear = nn.Linear(in_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.adj_weight)
        nn.init.zeros_(self.adj_bias)
        nn.init.xavier_uniform_(self.x_linear.weight)
        nn.init.zeros_(self.x_linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        x_dropped = F.dropout(x, p=self.dropout, training=self.training)
        x_hidden = F.relu(self.x_linear(x_dropped))
        adj_hidden = F.relu(spmm(adj, self.adj_weight) + self.adj_bias)
        h = torch.cat([x_hidden, adj_hidden], dim=-1)
        return self.fuse(h)


class H2GCN(nn.Module):
    """H2GCN-style ego, one-hop, and two-hop representation separation."""

    use_no_self_adj = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        layers: int = 2,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("H2GCN requires at least one propagation layer")
        self.dropout = dropout
        self.layers = layers
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.propagation_layers = nn.ModuleList(
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(layers)
        )
        self.classifier = nn.Linear(hidden_dim * (layers + 1), out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.encoder(h))
        reps = [h]
        for layer in self.propagation_layers:
            one_hop = spmm(adj, h)
            two_hop = spmm(adj, one_hop)
            h = torch.cat([one_hop, two_hop], dim=-1)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(layer(h))
            reps.append(h)
        out = torch.cat(reps, dim=-1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.classifier(out)


class HARPGNN(nn.Module):
    """Adaptive low/high-pass polynomial graph model.

    Low-pass bases are A^k X. High-pass bases are residual differences
    A^{k-1}X - A^kX. A learned node gate mixes both families.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 3,
        use_layer_norm: bool = True,
        gate_bias_init: float = -1.5,
        branch_mode: str = "both",
        use_gate_signal: bool = True,
        gate_type: str = "input",
        gate_granularity: str = "feature",
        propagate_projected: bool = False,
    ):
        super().__init__()
        if hops < 1:
            raise ValueError("HARP-GNN requires hops >= 1")
        self.hops = hops
        self.dropout = dropout
        self.branch_mode = branch_mode
        self.use_gate_signal = use_gate_signal
        self.gate_type = gate_type
        self.gate_granularity = gate_granularity
        self.propagate_projected = propagate_projected
        if gate_granularity not in {"feature", "scalar"}:
            raise ValueError(f"Unknown HARP gate_granularity: {gate_granularity}")
        gate_out_dim = 1 if gate_granularity == "scalar" else hidden_dim
        self.low_proj = nn.Linear(in_dim, hidden_dim)
        self.high_proj = nn.Linear(in_dim, hidden_dim)
        self.low_weights = nn.Parameter(torch.zeros(hops + 1))
        self.high_weights = nn.Parameter(torch.zeros(hops))
        self.input_gate: nn.Module | None = None
        self.branch_gate: nn.Module | None = None
        if self.branch_mode == "both" and self.gate_type == "input":
            input_gate_out = nn.Linear(hidden_dim, gate_out_dim)
            nn.init.constant_(input_gate_out.bias, gate_bias_init)
            self.input_gate = nn.Sequential(
                nn.Linear(in_dim + 1, hidden_dim),
                nn.ReLU(),
                input_gate_out,
                nn.Sigmoid(),
            )
        elif self.branch_mode == "both" and self.gate_type == "branch":
            branch_gate_out = nn.Linear(hidden_dim, gate_out_dim)
            nn.init.constant_(branch_gate_out.bias, gate_bias_init)
            self.branch_gate = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                branch_gate_out,
                nn.Sigmoid(),
            )
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _compute_branches(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.propagate_projected:
            return self._compute_projected_branches(x, adj)

        bases = [x]
        cur = x
        for _ in range(self.hops):
            cur = spmm(adj, cur)
            bases.append(cur)

        low_alpha = torch.softmax(self.low_weights, dim=0)
        high_alpha = torch.softmax(self.high_weights, dim=0)

        low_hidden = 0.0
        for k, base in enumerate(bases):
            low_hidden = low_hidden + low_alpha[k] * self.low_proj(base)

        high_hidden = 0.0
        for k in range(1, self.hops + 1):
            residual = bases[k - 1] - bases[k]
            high_hidden = high_hidden + high_alpha[k - 1] * self.high_proj(residual)
        return low_hidden, high_hidden

    def _compute_projected_branches(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_bases = [F.linear(x, self.low_proj.weight, bias=None)]
        high_bases = [F.linear(x, self.high_proj.weight, bias=None)]
        low_cur = low_bases[0]
        high_cur = high_bases[0]
        for _ in range(self.hops):
            low_cur = spmm(adj, low_cur)
            high_cur = spmm(adj, high_cur)
            low_bases.append(low_cur)
            high_bases.append(high_cur)

        low_alpha = torch.softmax(self.low_weights, dim=0)
        high_alpha = torch.softmax(self.high_weights, dim=0)

        low_hidden = 0.0
        for k, base in enumerate(low_bases):
            low_hidden = low_hidden + low_alpha[k] * base
        if self.low_proj.bias is not None:
            low_hidden = low_hidden + self.low_proj.bias

        high_hidden = 0.0
        for k in range(1, self.hops + 1):
            residual = high_bases[k - 1] - high_bases[k]
            high_hidden = high_hidden + high_alpha[k - 1] * residual
        if self.high_proj.bias is not None:
            high_hidden = high_hidden + self.high_proj.bias
        return low_hidden, high_hidden

    def _compute_gate(
        self,
        x: torch.Tensor,
        gate_signal: torch.Tensor | None,
        low_hidden: torch.Tensor,
        high_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if gate_signal is None or not self.use_gate_signal:
            gate_signal = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

        if self.gate_type == "input":
            gate_input = torch.cat([x, gate_signal], dim=-1)
            if self.input_gate is None:
                raise RuntimeError("Input gate was not initialized")
            return self.input_gate(F.dropout(gate_input, p=self.dropout, training=self.training))
        if self.gate_type == "branch":
            gate_input = torch.cat([low_hidden, high_hidden, torch.abs(low_hidden - high_hidden)], dim=-1)
            if self.branch_gate is None:
                raise RuntimeError("Branch gate was not initialized")
            return self.branch_gate(F.dropout(gate_input, p=self.dropout, training=self.training))
        raise ValueError(f"Unknown HARP gate_type: {self.gate_type}")

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        low_hidden, high_hidden = self._compute_branches(x, adj)

        if self.branch_mode == "low":
            h = low_hidden
        elif self.branch_mode == "high":
            h = high_hidden
        elif self.branch_mode == "both":
            gate = self._compute_gate(x, gate_signal, low_hidden, high_hidden)
            h = (1.0 - gate) * low_hidden + gate * high_hidden
        else:
            raise ValueError(f"Unknown HARP branch_mode: {self.branch_mode}")
        h = self.norm(F.relu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)

    def filter_weights(self) -> Dict[str, list[float]]:
        return {
            "low": torch.softmax(self.low_weights.detach().cpu(), dim=0).tolist(),
            "high": torch.softmax(self.high_weights.detach().cpu(), dim=0).tolist(),
        }

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if self.branch_mode != "both":
            return {}
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                low_hidden, high_hidden = self._compute_branches(x, adj)
                gate = self._compute_gate(x, gate_signal, low_hidden, high_hidden)
                values = gate.detach().flatten()
                quantiles = torch.quantile(
                    values,
                    torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype, device=values.device),
                )
                return {
                    "gate_mean": float(values.mean().item()),
                    "gate_std": float(values.std(unbiased=False).item()),
                    "gate_q25": float(quantiles[0].item()),
                    "gate_median": float(quantiles[1].item()),
                    "gate_q75": float(quantiles[2].item()),
                    "gate_gt_0_5": float((values > 0.5).float().mean().item()),
                    "gate_min": float(values.min().item()),
                    "gate_max": float(values.max().item()),
                    "gate_features": float(gate.shape[1]),
                }
        finally:
            self.train(was_training)


class HARPX(HARPGNN):
    """HARP residual fusion with a sparse LINKX-style adjacency branch.

    This diagnostic variant tests whether residual low/high-pass feature
    evidence complements a direct adjacency-pattern encoder on larger
    heterophily graphs.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 3,
        use_layer_norm: bool = True,
        gate_bias_init: float = -1.5,
        branch_mode: str = "both",
        use_gate_signal: bool = True,
        gate_type: str = "input",
        gate_granularity: str = "feature",
        propagate_projected: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=gate_bias_init,
            branch_mode=branch_mode,
            use_gate_signal=use_gate_signal,
            gate_type=gate_type,
            gate_granularity=gate_granularity,
            propagate_projected=propagate_projected,
        )
        self.x_linear = nn.Linear(in_dim, hidden_dim)
        self.adj_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.adj_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reset_harpx_parameters()

    def reset_harpx_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_linear.weight)
        nn.init.zeros_(self.x_linear.bias)
        nn.init.xavier_uniform_(self.adj_weight)
        nn.init.zeros_(self.adj_bias)

    def _harp_hidden(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None) -> torch.Tensor:
        low_hidden, high_hidden = self._compute_branches(x, adj)
        if self.branch_mode == "low":
            h = low_hidden
        elif self.branch_mode == "high":
            h = high_hidden
        elif self.branch_mode == "both":
            gate = self._compute_gate(x, gate_signal, low_hidden, high_hidden)
            h = (1.0 - gate) * low_hidden + gate * high_hidden
        else:
            raise ValueError(f"Unknown HARP branch_mode: {self.branch_mode}")
        return self.norm(F.relu(h))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        harp_hidden = self._harp_hidden(x, adj, gate_signal)
        x_hidden = F.relu(self.x_linear(F.dropout(x, p=self.dropout, training=self.training)))
        adj_hidden = F.relu(spmm(adj, self.adj_weight) + self.adj_bias)
        h = torch.cat([harp_hidden, x_hidden, adj_hidden], dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fuse(h)
        return self.classifier(h)


class HARPStructGate(HARPGNN):
    """Structure-conditioned HARP gate.

    Unlike HARP-X, this diagnostic variant does not concatenate an adjacency
    branch into the final classifier. It uses sparse adjacency-pattern evidence
    to condition the node-wise low/high-pass gate.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 3,
        use_layer_norm: bool = True,
        gate_bias_init: float = -1.5,
        branch_mode: str = "both",
        use_gate_signal: bool = True,
        gate_granularity: str = "feature",
        propagate_projected: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=gate_bias_init,
            branch_mode=branch_mode,
            use_gate_signal=use_gate_signal,
            gate_type="input",
            gate_granularity=gate_granularity,
            propagate_projected=propagate_projected,
        )
        self.input_gate = None
        self.branch_gate = None
        self.adj_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.adj_bias = nn.Parameter(torch.zeros(hidden_dim))
        gate_out_dim = 1 if gate_granularity == "scalar" else hidden_dim
        gate_out = nn.Linear(hidden_dim, gate_out_dim)
        nn.init.constant_(gate_out.bias, gate_bias_init)
        self.structure_gate = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            gate_out,
            nn.Sigmoid(),
        )
        self.reset_structure_parameters()

    def reset_structure_parameters(self) -> None:
        nn.init.xavier_uniform_(self.adj_weight)
        nn.init.zeros_(self.adj_bias)

    def _structure_hidden(self, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(spmm(adj, self.adj_weight) + self.adj_bias)

    def _compute_structure_gate(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None,
        low_hidden: torch.Tensor,
        high_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if gate_signal is None or not self.use_gate_signal:
            gate_signal = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        structure_hidden = self._structure_hidden(adj)
        gate_input = torch.cat(
            [low_hidden, high_hidden, torch.abs(low_hidden - high_hidden), structure_hidden, gate_signal],
            dim=-1,
        )
        return self.structure_gate(F.dropout(gate_input, p=self.dropout, training=self.training))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        low_hidden, high_hidden = self._compute_branches(x, adj)
        if self.branch_mode == "low":
            h = low_hidden
        elif self.branch_mode == "high":
            h = high_hidden
        elif self.branch_mode == "both":
            gate = self._compute_structure_gate(x, adj, gate_signal, low_hidden, high_hidden)
            h = (1.0 - gate) * low_hidden + gate * high_hidden
        else:
            raise ValueError(f"Unknown HARP branch_mode: {self.branch_mode}")
        h = self.norm(F.relu(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if self.branch_mode != "both":
            return {}
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                low_hidden, high_hidden = self._compute_branches(x, adj)
                gate = self._compute_structure_gate(x, adj, gate_signal, low_hidden, high_hidden)
                structure_hidden = self._structure_hidden(adj)
                values = gate.detach().flatten()
                quantiles = torch.quantile(
                    values,
                    torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype, device=values.device),
                )
                return {
                    "gate_mean": float(values.mean().item()),
                    "gate_std": float(values.std(unbiased=False).item()),
                    "gate_q25": float(quantiles[0].item()),
                    "gate_median": float(quantiles[1].item()),
                    "gate_q75": float(quantiles[2].item()),
                    "gate_gt_0_5": float((values > 0.5).float().mean().item()),
                    "gate_min": float(values.min().item()),
                    "gate_max": float(values.max().item()),
                    "gate_features": float(gate.shape[1]),
                    "structure_norm": float(structure_hidden.norm(dim=-1).mean().item()),
                }
        finally:
            self.train(was_training)


class HARPEgoSep(nn.Module):
    """Ego-separated residual polynomial HARP diagnostic.

    This variant follows the H2GCN lesson that ego and neighbor evidence should
    remain separated on heterophilous graphs. It encodes node features first,
    propagates hidden states with a no-self adjacency, applies HARP-style
    low/high residual filtering in hidden space, and classifies from the
    concatenated ego and residual-filtered views.
    """

    use_no_self_adj = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        hops: int = 2,
        use_layer_norm: bool = True,
        gate_bias_init: float = -1.0,
        use_gate_signal: bool = True,
        gate_granularity: str = "feature",
    ):
        super().__init__()
        if hops < 1:
            raise ValueError("HARP-ESep requires hops >= 1")
        if gate_granularity not in {"feature", "scalar"}:
            raise ValueError(f"Unknown HARP-ESep gate_granularity: {gate_granularity}")
        self.hops = hops
        self.dropout = dropout
        self.use_gate_signal = use_gate_signal
        self.gate_granularity = gate_granularity
        gate_out_dim = 1 if gate_granularity == "scalar" else hidden_dim

        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.low_weights = nn.Parameter(torch.zeros(hops + 1))
        self.high_weights = nn.Parameter(torch.zeros(hops))
        gate_out = nn.Linear(hidden_dim, gate_out_dim)
        nn.init.constant_(gate_out.bias, gate_bias_init)
        self.gate = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            gate_out,
            nn.Sigmoid(),
        )
        self.norm_ego = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.norm_residual = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.classifier = nn.Linear(2 * hidden_dim, out_dim)

    def _hidden_bases(self, x: torch.Tensor, adj: torch.Tensor) -> list[torch.Tensor]:
        h0 = F.relu(self.encoder(F.dropout(x, p=self.dropout, training=self.training)))
        bases = [h0]
        cur = h0
        for _ in range(self.hops):
            cur = spmm(adj, cur)
            bases.append(cur)
        return bases

    def _compute_views(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate_signal is None or not self.use_gate_signal:
            gate_signal = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        bases = self._hidden_bases(x, adj)
        low_alpha = torch.softmax(self.low_weights, dim=0)
        high_alpha = torch.softmax(self.high_weights, dim=0)

        low_hidden = 0.0
        for k, base in enumerate(bases):
            low_hidden = low_hidden + low_alpha[k] * base

        high_hidden = 0.0
        for k in range(1, self.hops + 1):
            high_hidden = high_hidden + high_alpha[k - 1] * (bases[k - 1] - bases[k])

        gate_input = torch.cat(
            [bases[0], low_hidden, high_hidden, torch.abs(low_hidden - high_hidden), gate_signal],
            dim=-1,
        )
        gate = self.gate(F.dropout(gate_input, p=self.dropout, training=self.training))
        residual_hidden = (1.0 - gate) * low_hidden + gate * high_hidden
        return bases[0], residual_hidden, gate

    def forward(self, x: torch.Tensor, adj: torch.Tensor, gate_signal: torch.Tensor | None = None) -> torch.Tensor:
        ego_hidden, residual_hidden, _ = self._compute_views(x, adj, gate_signal)
        ego_hidden = self.norm_ego(F.relu(ego_hidden))
        residual_hidden = self.norm_residual(F.relu(residual_hidden))
        h = torch.cat([ego_hidden, residual_hidden], dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)

    def filter_weights(self) -> Dict[str, list[float]]:
        return {
            "low": torch.softmax(self.low_weights.detach().cpu(), dim=0).tolist(),
            "high": torch.softmax(self.high_weights.detach().cpu(), dim=0).tolist(),
        }

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                ego_hidden, residual_hidden, gate = self._compute_views(x, adj, gate_signal)
                values = gate.detach().flatten()
                quantiles = torch.quantile(
                    values,
                    torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype, device=values.device),
                )
                return {
                    "gate_mean": float(values.mean().item()),
                    "gate_std": float(values.std(unbiased=False).item()),
                    "gate_q25": float(quantiles[0].item()),
                    "gate_median": float(quantiles[1].item()),
                    "gate_q75": float(quantiles[2].item()),
                    "gate_gt_0_5": float((values > 0.5).float().mean().item()),
                    "gate_min": float(values.min().item()),
                    "gate_max": float(values.max().item()),
                    "gate_features": float(gate.shape[1]),
                    "ego_norm": float(ego_hidden.norm(dim=-1).mean().item()),
                    "residual_norm": float(residual_hidden.norm(dim=-1).mean().item()),
                }
        finally:
            self.train(was_training)


class HARPAdaptive(nn.Module):
    """Adaptive selector between self-loop HARP and no-self HARP-ESep.

    This candidate tests whether the model can preserve WebKB behavior from the
    original self-loop residual HARP branch while using ego-separated no-self
    propagation on larger heterophily graphs.
    """

    use_dual_adj = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        harp_hops: int = 3,
        esep_hops: int = 2,
        use_layer_norm: bool = True,
        harp_gate_bias_init: float = -1.5,
        esep_gate_bias_init: float = -1.0,
        selector_bias_init: float = 0.0,
        use_gate_signal: bool = True,
        selector_granularity: str = "scalar",
        propagate_projected: bool = False,
    ):
        super().__init__()
        if selector_granularity not in {"feature", "scalar"}:
            raise ValueError(f"Unknown HARP-Adaptive selector_granularity: {selector_granularity}")
        self.dropout = dropout
        self.use_gate_signal = use_gate_signal
        self.selector_granularity = selector_granularity
        selector_out_dim = 1 if selector_granularity == "scalar" else hidden_dim

        self.self_harp = HARPGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            hops=harp_hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=harp_gate_bias_init,
            branch_mode="both",
            use_gate_signal=use_gate_signal,
            gate_type="input",
            gate_granularity="feature",
            propagate_projected=propagate_projected,
        )
        self.esep = HARPEgoSep(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
            hops=esep_hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=esep_gate_bias_init,
            use_gate_signal=use_gate_signal,
            gate_granularity="feature",
        )
        selector_out = nn.Linear(hidden_dim, selector_out_dim)
        nn.init.constant_(selector_out.bias, selector_bias_init)
        self.selector = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            selector_out,
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _split_adj(adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(adj, tuple):
            if len(adj) != 2:
                raise ValueError("HARP-Adaptive expects a pair of adjacency tensors")
            return adj
        return adj, adj

    def _compute_hidden(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_adj, no_self_adj = self._split_adj(adj)
        if gate_signal is None or not self.use_gate_signal:
            gate_signal = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        self_hidden = self.self_harp(x, self_adj, gate_signal)
        esep_hidden = self.esep(x, no_self_adj, gate_signal)
        selector_input = torch.cat(
            [self_hidden, esep_hidden, torch.abs(self_hidden - esep_hidden), gate_signal],
            dim=-1,
        )
        selector = self.selector(F.dropout(selector_input, p=self.dropout, training=self.training))
        hidden = (1.0 - selector) * self_hidden + selector * esep_hidden
        return hidden, selector, esep_hidden - self_hidden

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, _, _ = self._compute_hidden(x, adj, gate_signal)
        hidden = self.norm(F.relu(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.classifier(hidden)

    def filter_weights(self) -> Dict[str, list[float]]:
        self_weights = self.self_harp.filter_weights()
        esep_weights = self.esep.filter_weights()
        return {
            "self_low": self_weights["low"],
            "self_high": self_weights["high"],
            "esep_low": esep_weights["low"],
            "esep_high": esep_weights["high"],
        }

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                hidden, selector, delta = self._compute_hidden(x, adj, gate_signal)
                values = selector.detach().flatten()
                quantiles = torch.quantile(
                    values,
                    torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype, device=values.device),
                )
                return {
                    "selector_mean": float(values.mean().item()),
                    "selector_std": float(values.std(unbiased=False).item()),
                    "selector_q25": float(quantiles[0].item()),
                    "selector_median": float(quantiles[1].item()),
                    "selector_q75": float(quantiles[2].item()),
                    "selector_gt_0_5": float((values > 0.5).float().mean().item()),
                    "selector_features": float(selector.shape[1]),
                    "adaptive_hidden_norm": float(hidden.norm(dim=-1).mean().item()),
                    "esep_minus_self_norm": float(delta.norm(dim=-1).mean().item()),
                }
        finally:
            self.train(was_training)


class HARPLogitBlend(nn.Module):
    """Graph-level logit blending between self-loop HARP and no-self HARP-ESep.

    This deliberately conservative candidate tests whether a single learned
    branch weight is more stable than the node-wise HARP-Adaptive selector.
    """

    use_dual_adj = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        harp_hops: int = 3,
        esep_hops: int = 2,
        use_layer_norm: bool = True,
        harp_gate_bias_init: float = -1.5,
        esep_gate_bias_init: float = -1.0,
        blend_logit_init: float = 0.0,
        learn_blend: bool = True,
        auxiliary_weight: float = 0.2,
        use_gate_signal: bool = True,
        propagate_projected: bool = False,
    ):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
        self.self_harp = HARPGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=harp_hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=harp_gate_bias_init,
            branch_mode="both",
            use_gate_signal=use_gate_signal,
            gate_type="input",
            gate_granularity="feature",
            propagate_projected=propagate_projected,
        )
        self.esep = HARPEgoSep(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=esep_hops,
            use_layer_norm=use_layer_norm,
            gate_bias_init=esep_gate_bias_init,
            use_gate_signal=use_gate_signal,
            gate_granularity="feature",
        )
        self.blend_logit = nn.Parameter(torch.tensor(float(blend_logit_init)), requires_grad=learn_blend)

    @staticmethod
    def _split_adj(adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(adj, tuple):
            if len(adj) != 2:
                raise ValueError("HARP-Blend expects a pair of adjacency tensors")
            return adj
        return adj, adj

    def _branch_logits(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self_adj, no_self_adj = self._split_adj(adj)
        return self.self_harp(x, self_adj, gate_signal), self.esep(x, no_self_adj, gate_signal)

    def blend_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.blend_logit)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_logits, esep_logits = self._branch_logits(x, adj, gate_signal)
        weight = self.blend_weight()
        return (1.0 - weight) * self_logits + weight * esep_logits

    def auxiliary_loss(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None,
        y: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.auxiliary_weight <= 0.0:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        self_logits, esep_logits = self._branch_logits(x, adj, gate_signal)
        return self.auxiliary_weight * 0.5 * (
            F.cross_entropy(self_logits[idx], y[idx]) + F.cross_entropy(esep_logits[idx], y[idx])
        )

    def filter_weights(self) -> Dict[str, list[float]]:
        self_weights = self.self_harp.filter_weights()
        esep_weights = self.esep.filter_weights()
        return {
            "self_low": self_weights["low"],
            "self_high": self_weights["high"],
            "esep_low": esep_weights["low"],
            "esep_high": esep_weights["high"],
        }

    def diagnostics(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        gate_signal: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self_logits, esep_logits = self._branch_logits(x, adj, gate_signal)
                weight = float(self.blend_weight().item())
                return {
                    "blend_weight": weight,
                    "self_logit_norm": float(self_logits.norm(dim=-1).mean().item()),
                    "esep_logit_norm": float(esep_logits.norm(dim=-1).mean().item()),
                    "logit_gap_norm": float((esep_logits - self_logits).norm(dim=-1).mean().item()),
                }
        finally:
            self.train(was_training)


def build_model(name: str, in_dim: int, out_dim: int, params: Dict[str, Any], num_nodes: int | None = None) -> nn.Module:
    hidden_dim = int(params.get("hidden_dim", 64))
    dropout = float(params.get("dropout", 0.5))
    name = name.lower()
    if name == "mlp":
        return MLP(in_dim, hidden_dim, out_dim, dropout=dropout)
    if name == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, dropout=dropout, bias=bool(params.get("bias", False)))
    if name == "sgc":
        return SGC(in_dim, out_dim, hops=int(params.get("hops", 2)))
    if name == "appnp":
        return APPNP(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 10)),
            alpha=float(params.get("alpha", 0.1)),
        )
    if name == "mixhop":
        powers = params.get("powers", [0, 1, 2])
        return MixHop(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            powers=[int(power) for power in powers],
        )
    if name in {"gprgnn", "gpr_gnn", "gpr-gnn"}:
        return GPRGNN(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 10)),
            alpha=float(params.get("alpha", 0.1)),
            init=str(params.get("init", "ppr")),
        )
    if name in {"fagcn", "fagcn_style", "fa_gcn", "fa-gcn"}:
        return FAGCNStyle(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            layers=int(params.get("layers", 2)),
            epsilon=float(params.get("epsilon", 0.3)),
            negative_slope=float(params.get("negative_slope", 0.2)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
        )
    if name == "linkx":
        if num_nodes is None:
            raise ValueError("LINKX requires num_nodes")
        return LINKX(num_nodes, in_dim, hidden_dim, out_dim, dropout=dropout)
    if name == "h2gcn":
        return H2GCN(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            layers=int(params.get("layers", 2)),
        )
    if name in {"harp_x", "harpx", "harp-linkx", "harp_linkx"}:
        if num_nodes is None:
            raise ValueError("HARP-X requires num_nodes")
        return HARPX(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 3)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            gate_bias_init=float(params.get("gate_bias_init", -1.5)),
            branch_mode=str(params.get("branch_mode", "both")),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            gate_type=str(params.get("gate_type", "input")),
            gate_granularity=str(params.get("gate_granularity", "feature")),
            propagate_projected=bool(params.get("propagate_projected", False)),
        )
    if name in {"harp_sgate", "harp_struct_gate", "harp-struct-gate", "harp_structure_gate"}:
        if num_nodes is None:
            raise ValueError("HARP-SGate requires num_nodes")
        return HARPStructGate(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 3)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            gate_bias_init=float(params.get("gate_bias_init", -1.5)),
            branch_mode=str(params.get("branch_mode", "both")),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            gate_granularity=str(params.get("gate_granularity", "feature")),
            propagate_projected=bool(params.get("propagate_projected", False)),
        )
    if name in {"harp_esep", "harp_ego_sep", "harp-ego-sep", "harp_egosep"}:
        return HARPEgoSep(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 2)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            gate_bias_init=float(params.get("gate_bias_init", -1.0)),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            gate_granularity=str(params.get("gate_granularity", "feature")),
        )
    if name in {"harp_adaptive", "harp_ada", "harp-adaptive", "harp_adaptive_esep"}:
        return HARPAdaptive(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            harp_hops=int(params.get("harp_hops", params.get("hops", 3))),
            esep_hops=int(params.get("esep_hops", 2)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            harp_gate_bias_init=float(params.get("harp_gate_bias_init", params.get("gate_bias_init", -1.5))),
            esep_gate_bias_init=float(params.get("esep_gate_bias_init", -1.0)),
            selector_bias_init=float(params.get("selector_bias_init", 0.0)),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            selector_granularity=str(params.get("selector_granularity", "scalar")),
            propagate_projected=bool(params.get("propagate_projected", False)),
        )
    if name in {"harp_blend", "harp_logit_blend", "harp-blend"}:
        return HARPLogitBlend(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            harp_hops=int(params.get("harp_hops", params.get("hops", 3))),
            esep_hops=int(params.get("esep_hops", 2)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            harp_gate_bias_init=float(params.get("harp_gate_bias_init", params.get("gate_bias_init", -1.5))),
            esep_gate_bias_init=float(params.get("esep_gate_bias_init", -1.0)),
            blend_logit_init=float(params.get("blend_logit_init", 0.0)),
            learn_blend=bool(params.get("learn_blend", True)),
            auxiliary_weight=float(params.get("auxiliary_weight", 0.2)),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            propagate_projected=bool(params.get("propagate_projected", False)),
        )
    if name in {"harp", "harp_gnn", "harp-gnn"} or name.startswith("harp_"):
        return HARPGNN(
            in_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            hops=int(params.get("hops", 3)),
            use_layer_norm=bool(params.get("use_layer_norm", True)),
            gate_bias_init=float(params.get("gate_bias_init", -1.5)),
            branch_mode=str(params.get("branch_mode", "both")),
            use_gate_signal=bool(params.get("use_gate_signal", True)),
            gate_type=str(params.get("gate_type", "input")),
            gate_granularity=str(params.get("gate_granularity", "feature")),
            propagate_projected=bool(params.get("propagate_projected", False)),
        )
    raise ValueError(f"Unknown model: {name}")
