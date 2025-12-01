# Deep Lyapunov Critics for Safe Financial Reinforcement Learning

![status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

This repository contains a single end-to-end experiment script that trains **Pontryagin-guided direct policy optimization (PG-DPO)** agents with **Lyapunov-style safety constraints** in a continuous-time Merton-style market. The current code focuses on a **multi-asset** setting with projection to the simplex for long-only, fully-invested portfolios.

---

## Project Overview
> The earlier README referenced a multi-module codebase. The project has since been simplified to one script that handles configuration, model definitions, training, and plotting.

This repository contains research code for **Pontryagin-Guided Direct Policy Optimization (PG-DPO)**  
applied to **continuous-time Merton–style consumption–investment problems** with:
---

- **single-asset** and **multi-asset** markets,
- **projection operators** (simplex / leverage constraints),
- and **Lyapunov critics** that enforce **safety constraints** such as
  wealth barriers and ruin avoidance.
## What's implemented

The main idea is to combine
- **Baseline PG-DPO (unconstrained)**
  - Learns consumption and portfolio weights using PMP-guided updates.
  - Uses a critic to regress to Monte Carlo returns at \(t=0\).

1. **Pontryagin Maximum Principle (PMP)** to shape the policy gradient via the Hamiltonian, and  
2. **Lyapunov-style value functions** to penalize unsafe wealth trajectories,
- **Constrained Anchored PG-DPO (CA-P-PGDPO)**
  - Adds a residual policy on top of the baseline policy to reduce constraint violation.
  - Enforces soft constraints via Lagrange multipliers:
    - ruin probability \(\leq \epsilon_{\text{ruin}}\)
    - performance degradation relative to the baseline
    - consumption deviation from the baseline
    - portfolio risk budget

so that we obtain policies that are **near-optimal and safe** in high-dimensional asset spaces.
- **Projection layer**
  - `FastSimplexProjection` maps raw portfolio weights to the probability simplex for long-only exposure.

> **WIP:** The codebase is under active development. APIs and experiment scripts may change.
- **Diagnostics & plots**
  - Contour plots for consumption and each asset's portfolio weight are written to `./plots_ca_ppgdpo/` after training.

---

## Key Contributions
## Environment & installation

- **P-PGDPO for Merton-type models**  
  - Single-asset PG-DPO: PMP-guided updates for consumption \(c_t\) and portfolio weight \(\pi_t\).  
  - Multi-asset PG-DPO: projection layer (simplex) to handle long-only, fully-invested portfolios in \(d\)-dimensional markets.
The script targets Python 3.9+ and depends on:

- **Deep Lyapunov Critics for safety**  
  - A separate **LyapunovNet** learns a Lyapunov function \(V_L(t,W)\) that approximates
    the long-run accumulation of **safety costs** (wealth barrier + ruin).  
  - Actor loss includes:
    - **Lyapunov TD loss** (Bellman-style fit for \(V_L\)),  
    - **Lyapunov drift penalty** \( [V_L(t_{k+1},W_{k+1}) - V_L(t_k,W_k)]_+ \),  
    - **ruin penalty** when wealth approaches a ruin barrier.
- `torch`
- `numpy`
- `matplotlib`

- **Closed-form Merton benchmark**  
  - Single-asset: compare learned \(\pi(t,W)\) and \(c(t,W)\) to the analytical Merton solution  
    (constant optimal weight \(\pi^\star\) and Riccati-type optimal consumption ratio \(\kappa^\star(t)\)).  
  - Multi-asset: compare learned portfolio weights to the unconstrained Merton portfolio
    \(\pi^\star = \frac{1}{\gamma}\Sigma^{-1}(\mu-r\mathbf{1})\).
Install the dependencies (example with `pip`):

- **High-dimensional projected setting**  
  - Multi-asset experiments with up to \(d=64\) assets,  
  - covariance matrices generated on-the-fly and enforced to be positive semidefinite,  
  - **FastSimplexProjection** layer for efficient GPU-friendly projection to the simplex.
```bash
pip install torch numpy matplotlib
```

---

## Methodology

### 1. Mathematical framework

- **Model:** Continuous-time Merton consumption–investment problem with CRRA utility:
  \[
  dW_t = W_t\Big(r + \pi_t^\top(\mu - r\mathbf{1})\Big)dt - c_t\,dt
          + W_t \pi_t^\top \Sigma^{1/2} dB_t.
  \]

- **Objective:** maximize
  \[
  \mathbb{E}\Big[ \int_0^T e^{-\delta t}\,u(c_t)\,dt + \beta\,u(W_T)\Big],
  \quad
  u(x) = \frac{x^{1-\gamma}}{1-\gamma}.
  \]

- **Single-asset PG-DPO:**
  - PolicyNet outputs \((\pi(t,W), \kappa(t))\) with \(c_t = \kappa(t)W_t\).  
  - Critic (ValueNet) approximates \(V(t,W)\); costate \(Y = \partial V / \partial W\) is obtained via autograd.  
  - Hamiltonian
    \[
    H(t,W,\pi,c,Y) = u(c) + Y\cdot\big(W(r + \pi(\mu-r)) - c\big).
    \]
  - Actor targets are defined by **gradient ascent on \(H\)**:
    \[
    \pi_{\text{target}} = \pi + \eta_\pi \frac{\partial H}{\partial \pi}, \quad
    c_{\text{target}}   = c   + \eta_c \frac{\partial H}{\partial c},
    \]
    and the policy is trained to regress to these PMP-guided targets.

- **Multi-asset PG-DPO:**
  - PolicyNet outputs raw weights, then applies **simplex projection** to enforce
    \(\pi_t \in \Delta^{d-1}\) (long-only, fully invested).
  - Hamiltonian gradient w.r.t. \(\pi\) plays the same role; targets are projected back onto the simplex.

### 2. Lyapunov critic and safety costs

- Define **instantaneous safety cost** \(s(W)\) using:
  - **wealth barrier** \(W_{\text{safe}}\),  
  - **ruin threshold** \(W_{\text{ruin}}\) (strongly penalized).
  \[
  s(W) = \alpha (W_{\text{safe}} - W)_+^2
       + \beta  (W_{\text{ruin}} - W)_+^2, \quad \beta \gg \alpha.
  \]

- **LyapunovNet** learns
  \[
  V_L(t,W) \approx \mathbb{E}\Big[\sum_{k\ge 0} \gamma_L^k\,s(W_{t+k}) \,\Big|\,W_t=W\Big]
  \]
  via a TD-like loss:
  \[
  V_L(t_k,W_k) \approx s(W_k) + \gamma_L\,V_L(t_{k+1},W_{k+1}).
  \]

- **Actor loss** includes:
  - Lyapunov TD loss (for training LyapunovNet),
  - Lyapunov drift penalty \(\mathbb{E}[(V_L(t_{k+1},W_{k+1}) - V_L(t_k,W_k))_+]\),
  - ruin penalty \(\mathbb{E}[s(W_{k+1})]\).

This makes the final policy **PMP-guided for optimality** while being **Lyapunov-guided for safety**.
## Running the experiment

---
Execute the main script directly:

## Status & Roadmap
```bash
python "Deep Lyapunov Critics for Safe Financial Reinforcement Learning.py"
```

- ✅ Single-asset PG-DPO with Lyapunov safety  
  - PMP-guided updates for \(\pi, c\)  
  - LyapunovNet for wealth barrier and ruin penalty  
  - Contour plots of \(\pi(t,W)\) and \(c(t,W)\) vs analytic Merton solution.
The run proceeds in two stages:

- ✅ Multi-asset PG-DPO with projection  
  - Fast simplex projection layer  
  - High-dimensional covariance matrix generation  
  - Lyapunov critic integrated analogously to single-asset case.
1. **Baseline PG-DPO** (`train_baseline_ppgdpo_multi`): trains an unconstrained policy for `cfg.baseline_iters` iterations.
2. **Anchored constrained PG-DPO** (`train_anchored_constrained_ppgdpo`): trains a residual policy for `cfg.anchored_iters` iterations using constraint-aware losses.

- ☐ Experiments & diagnostics  
  - Systematic sweeps over \(\gamma, T, d\)  
  - Sensitivity to Lyapunov weights (\(\alpha,\beta,\lambda_{\text{lyap}}\))  
  - Comparison with unconstrained Merton portfolios.
Key configuration options are set at the top of the script via the `cfg` object (e.g., number of assets `d`, time step `dt`, risk aversion `gamma`, ruin tolerance `eps_ruin`, and learning rates). Edit them in the file to change experiment settings.

- ☐ Planned extensions  
  - CVaR-based safety costs  
  - Regime-switching / stochastic interest rate models  
  - Transaction costs and no-trade regions.
After training, contour plots for consumption and portfolio weights are saved in the `plots_ca_ppgdpo` directory.

---

## Installation
## Repository layout

```bash
git clone https://github.com/your-username/deep-lyapunov-critics-finrl.git
cd deep-lyapunov-critics-finrl
pip install -r requirements.txt
```

## Usage

Run the main script for either the single-asset or multi-asset setting:

```bash
python deep_lyapunov_merton.py --mode single --iters 2000
python deep_lyapunov_merton.py --mode multi --d 64 --iters 2000
Deep Lyapunov Critics for Safe Financial Reinforcement Learning.py  # Full experiment pipeline
README.md                                                         # You are here
```

Key configuration knobs (edit in the script or via CLI flags): horizon \(T\), risk aversion \(\gamma\), discount \(\delta\),
Lyapunov weights \(\alpha, \beta, \lambda_{\text{lyap}}\), and projection choices.

## Repository Structure

```
deep_lyapunov_merton.py     # Main entry point (single & multi-asset PG-DPO + Lyapunov)
models/
  policy.py                 # PolicyNet definitions (single / multi)
  value.py                  # ValueNet (critic for PG-DPO)
  lyapunov.py               # LyapunovNet architectures
algos/
  pgdpo_single.py           # Single-asset PG-DPO training loop
  pgdpo_multi.py            # Multi-asset PG-DPO training loop (projection)
  projection.py             # FastSimplexProjection and other operators
utils/
  grids.py                  # Time/wealth grid generation
  merton_analytic.py        # Closed-form Merton solution (pi*, kappa*)
  plotting.py               # Contour & diagnostic plotting
outputs_pgdpo/              # Saved figures / logs
```
---

## Citation & contact

If you build on this code or its ideas, please cite or mention the repository. For questions or collaboration, contact chln0124@skku.edu.
