# Deep Lyapunov Critics for Safe Financial Reinforcement Learning

This repository contains a single end-to-end experiment script that trains **Pontryagin-guided direct policy optimization (PG-DPO)** agents with **Lyapunov-style safety constraints** in a continuous-time Merton-style market. The current code focuses on a **multi-asset** setting with projection to the simplex for long-only, fully-invested portfolios.

> The earlier README referenced a multi-module codebase. The project has since been simplified to one script that handles configuration, model definitions, training, and plotting.

---

## What's implemented

- **Baseline PG-DPO (unconstrained)**
  - Learns consumption and portfolio weights using PMP-guided updates.
  - Uses a critic to regress to Monte Carlo returns at \(t=0\).

- **Constrained Anchored PG-DPO (CA-P-PGDPO)**
  - Adds a residual policy on top of the baseline policy to reduce constraint violation.
  - Enforces soft constraints via Lagrange multipliers:
    - ruin probability \(\leq \epsilon_{\text{ruin}}\)
    - performance degradation relative to the baseline
    - consumption deviation from the baseline
    - portfolio risk budget

- **Projection layer**
  - `FastSimplexProjection` maps raw portfolio weights to the probability simplex for long-only exposure.

- **Diagnostics & plots**
  - Contour plots for consumption and each asset's portfolio weight are written to `./plots_ca_ppgdpo/` after training.

---

## Environment & installation

The script targets Python 3.9+ and depends on:

- `torch`
- `numpy`
- `matplotlib`

Install the dependencies (example with `pip`):

```bash
pip install torch numpy matplotlib
```

---

## Running the experiment

Execute the main script directly:

```bash
python "Deep Lyapunov Critics for Safe Financial Reinforcement Learning.py"
```

The run proceeds in two stages:

1. **Baseline PG-DPO** (`train_baseline_ppgdpo_multi`): trains an unconstrained policy for `cfg.baseline_iters` iterations.
2. **Anchored constrained PG-DPO** (`train_anchored_constrained_ppgdpo`): trains a residual policy for `cfg.anchored_iters` iterations using constraint-aware losses.

Key configuration options are set at the top of the script via the `cfg` object (e.g., number of assets `d`, time step `dt`, risk aversion `gamma`, ruin tolerance `eps_ruin`, and learning rates). Edit them in the file to change experiment settings.

After training, contour plots for consumption and portfolio weights are saved in the `plots_ca_ppgdpo` directory.

---

## Repository layout

```
Deep Lyapunov Critics for Safe Financial Reinforcement Learning.py  # Full experiment pipeline
README.md                                                         # You are here
```

---

## Citation & contact

If you build on this code or its ideas, please cite or mention the repository. For questions or collaboration, contact **chln0124@skku.edu**.
