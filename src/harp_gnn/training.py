from __future__ import annotations

import copy
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .data import GraphDataset
from .models import build_model
from .utils import accuracy, binary_roc_auc, set_seed


def _model_adj(model: torch.nn.Module, dataset: GraphDataset) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if getattr(model, "use_dual_adj", False):
        return dataset.adj_norm, dataset.adj_no_self_norm
    if getattr(model, "use_no_self_adj", False):
        return dataset.adj_no_self_norm
    return dataset.adj_norm


def _metric_value(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> float:
    if metric == "accuracy":
        return accuracy(logits, labels)
    if metric in {"roc_auc", "auc"}:
        return binary_roc_auc(logits, labels)
    raise ValueError(f"Unknown evaluation metric: {metric}")


def _eval_split(
    model: torch.nn.Module,
    dataset: GraphDataset,
    idx: torch.Tensor,
    metric: str,
) -> tuple[float, float]:
    model.eval()
    adj = _model_adj(model, dataset)
    with torch.no_grad():
        logits = model(dataset.x, adj, dataset.gate_signal)
        loss = F.cross_entropy(logits[idx], dataset.y[idx]).item()
        value = _metric_value(logits[idx], dataset.y[idx], metric)
    return loss, value


def train_one_model(
    dataset: GraphDataset,
    model_name: str,
    seed: int,
    training_params: Dict[str, Any],
    model_params: Dict[str, Any],
) -> Dict[str, Any]:
    set_seed(seed)
    params = dict(training_params)
    params.update(model_params)
    build_name = str(params.get("type", model_name))
    model = build_model(
        build_name,
        dataset.num_features,
        dataset.num_classes,
        params,
        num_nodes=int(dataset.metadata["num_nodes"]),
    ).to(dataset.x.device)

    lr = float(params.get("lr", 0.01))
    weight_decay = float(params.get("weight_decay", 5e-4))
    max_epochs = int(params.get("epochs", 200))
    patience = int(params.get("patience", 50))
    metric = str(params.get("metric", "accuracy")).lower()
    metric_column = "acc" if metric == "accuracy" else metric

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        adj = _model_adj(model, dataset)
        optimizer.zero_grad()
        logits = model(dataset.x, adj, dataset.gate_signal)
        train_loss = F.cross_entropy(logits[dataset.train_idx], dataset.y[dataset.train_idx])
        if hasattr(model, "auxiliary_loss"):
            train_loss = train_loss + model.auxiliary_loss(
                dataset.x,
                adj,
                dataset.gate_signal,
                dataset.y,
                dataset.train_idx,
            )
        train_loss.backward()
        optimizer.step()

        val_loss, val_acc = _eval_split(model, dataset, dataset.val_idx, metric)
        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    elapsed = time.time() - started
    model.load_state_dict(best_state)
    train_loss, train_acc = _eval_split(model, dataset, dataset.train_idx, metric)
    val_loss, val_acc = _eval_split(model, dataset, dataset.val_idx, metric)
    test_loss, test_acc = _eval_split(model, dataset, dataset.test_idx, metric)
    adj = _model_adj(model, dataset)

    result: Dict[str, Any] = {
        "dataset": dataset.name,
        "model": model_name,
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "metric_name": metric,
        "train_metric": train_acc,
        "val_metric": val_acc,
        "test_metric": test_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
    }
    result[f"train_{metric_column}"] = train_acc
    result[f"val_{metric_column}"] = val_acc
    result[f"test_{metric_column}"] = test_acc
    result.update({f"meta_{k}": v for k, v in dataset.metadata.items()})
    if hasattr(model, "filter_weights"):
        result["filter_weights"] = str(model.filter_weights())
    if hasattr(model, "diagnostics"):
        diagnostics = model.diagnostics(dataset.x, adj, dataset.gate_signal)
        result.update({f"diag_{k}": v for k, v in diagnostics.items()})
    return result
