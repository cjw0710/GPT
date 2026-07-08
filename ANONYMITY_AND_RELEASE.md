# Anonymous Release Notes

This repository is prepared for anonymous review. It intentionally avoids
author names, affiliations, emails, personal paths, and persistent external
links in the manuscript source.

## Review Policy Note

For AAAI-style double-blind review, submit the frozen Code and Data artifact
through the official submission system. Do not cite an external GitHub URL in
the paper or supplementary document unless the conference instructions
explicitly allow it, because web-hosted material can change after submission
and can compromise anonymity.

The GitHub mirror is therefore a convenience copy of the same artifact, not a
replacement for the official submitted ZIP.

## Recommended Anonymous GitHub Setup

1. Create a fresh GitHub account that does not reveal author identity.
2. Create a public repository such as `harp-select-aaai27-artifact`.
3. Push this artifact from a clean local clone:

```bash
git init
git checkout -b main
git add .
git commit -m "Initial anonymous reproducibility artifact"
git remote add origin https://github.com/<anonymous-account>/harp-select-aaai27-artifact.git
git push -u origin main
```

4. Do not connect the anonymous account to personal SSH keys, personal email,
institutional email, personal websites, or identifiable GitHub organizations.

## What Is Included

- Source code for the implemented GNN models and experiment runner.
- Experiment configs for all reported and diagnostic runs.
- CSV outputs used to generate the paper tables.
- Paper source, generated tables, figures, and reproducibility checklist.
- Verification scripts for implementation invariants, reported numbers,
  manuscript integrity, claim boundaries, and submission readiness.

Raw public datasets are not bundled. The loaders fetch or read public benchmark
files under `data/` when full experiments are rerun.
