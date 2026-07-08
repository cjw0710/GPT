# Structural HARP Diagnostic Note

This note records a bounded diagnostic experiment, not a main manuscript claim.
The goal was to test whether sparse adjacency-pattern evidence can reduce the
current HARP-GNN deficits on larger heterophily graphs.

## Variants

HARP-X keeps the original low/high-pass residual HARP fusion and concatenates it
with:

- a feature MLP branch, and
- a LINKX-style sparse adjacency branch with one trainable embedding row per node.

The fused hidden representation is then passed through a small nonlinear fusion
layer and the classifier.

HARP-SGate uses the same sparse adjacency branch differently: it feeds the
adjacency-pattern representation into the low/high-pass gate, rather than
concatenating it into the final classifier.

HARP-ESep removes self-loops from propagation, encodes ego features first, then
applies HARP-style residual filtering in hidden space. The classifier receives
the ego representation and the gated residual representation separately.

HARP-Adaptive trains both the original self-loop HARP branch and the no-self
HARP-ESep branch in one model, then learns a node-wise selector over their
hidden representations.

HARP-Blend is a more conservative follow-up: it mixes the self-loop HARP and
no-self HARP-ESep class logits with one learned graph-level scalar, with
auxiliary supervision on both branches.

## Diagnostic Protocol

- Config: `configs/harp_x_diagnostic.yaml`
- Results: `results/harp_x_diagnostic.csv`
- Table: `paper/tables/harp_x_diagnostic_results.tex`
- Datasets: Texas, Wisconsin, Cornell, Actor, Chameleon, Squirrel
- Models: LINKX, H2GCN, HARP-GNN, HARP-X, HARP-SGate, HARP-ESep,
  HARP-Adaptive, HARP-Blend
- Splits/seeds: 2 fixed splits/seeds

Because this is only a two-seed diagnostic, it should not be cited as a primary
benchmark result.

## Mean Test Accuracy

| Dataset | Best diagnostic model | HARP-GNN | HARP-X | HARP-SGate | HARP-ESep | HARP-Adaptive | HARP-Blend |
|---|---:|---:|---:|---:|---:|---:|---:|
| Texas | HARP-GNN/HARP-SGate 86.49 | 86.49 | 85.14 | 86.49 | 75.68 | 77.03 | 83.78 |
| Wisconsin | HARP-GNN 84.31 | 84.31 | 79.41 | 82.35 | 77.45 | 78.43 | 81.37 |
| Cornell | H2GCN/LINKX 77.03 | 74.32 | 75.68 | 70.27 | 68.92 | 71.62 | 75.68 |
| Actor | HARP-Adaptive 36.58 | 35.39 | 35.53 | 35.30 | 35.76 | 36.58 | 35.95 |
| Chameleon | HARP-ESep 66.12 | 55.37 | 56.91 | 56.03 | 66.12 | 57.35 | 55.92 |
| Squirrel | LINKX/HARP-ESep 47.31 | 37.13 | 39.10 | 35.54 | 47.31 | 40.20 | 36.60 |

## Selector Diagnostics

| Dataset | HARP-Adaptive selector mean | HARP-Blend weight | Failure mode |
|---|---:|---:|---|
| Texas | 0.87 | 0.55 | Adaptive over-selects ESep; Blend partly recovers self-loop HARP |
| Wisconsin | 0.91 | 0.53 | Adaptive over-selects ESep; Blend remains below HARP |
| Cornell | 0.86 | 0.55 | Adaptive over-selects ESep; Blend recovers HARP-X-level accuracy |
| Actor | 0.87 | 0.49 | Adaptive is the best two-seed diagnostic row |
| Chameleon | 0.29 | 0.42 | Both selectors under-select ESep, losing the ESep gain |
| Squirrel | 0.47 | 0.51 | Neither selector reaches the ESep/LINKX row |

## Interpretation

The late-fusion and structure-gated variants remain negative diagnostics:
simply appending a LINKX-style adjacency branch, or feeding that branch into the
gate, does not solve the large-heterophily failure mode.

HARP-ESep is the first promising structural direction. It improves over
HARP-GNN by more than 10 percentage points on Chameleon and Squirrel in this
two-seed diagnostic, matching or exceeding the strongest diagnostic baseline on
those rows. However, it is much worse on Texas, Wisconsin, and Cornell.

The two adaptive follow-ups did not solve that tradeoff. HARP-Adaptive learns
the wrong direction: it over-selects ESep on the small WebKB graphs where ESep
hurts, and under-selects ESep on Chameleon/Squirrel where ESep helps. HARP-Blend
is stabler on WebKB, but its graph-level weight stays near the middle and does
not capture the large Chameleon/Squirrel ESep gain.

This suggests the next serious method should neither globally replace HARP with
HARP-ESep nor rely on an unconstrained selector. Instead, the paper needs either:

1. a principled adaptive choice between self-loop residual HARP and no-self
   ego-separated HARP-ESep, probably with an explicit graph-level structural
   prior or validation-calibrated branch selection, or
2. a revised architecture that preserves WebKB behavior while using no-self
   ego-neighbor separation on larger heterophily graphs.

The full 10-split HARP-ESep run on Actor/Chameleon/Squirrel remains the strongest
candidate evidence collected so far. Adaptive branch selection is still an open
method problem, not a solved contribution.
