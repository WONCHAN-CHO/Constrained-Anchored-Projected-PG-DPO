# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 23:35:50 2025

@author: WONCHAN
"""

import os, math, numpy as np, torch, torch.nn as nn, torch.optim as optim
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CFG:
    pass


cfg = CFG()

#  Model / training configuration
cfg.d = 3
cfg.T = 5.0
cfg.dt = 0.05
cfg.steps = int(cfg.T / cfg.dt)

cfg.gamma = 5.0
cfg.delta = 0.03
cfg.beta = 1.0

cfg.r = 0.02

mu_vals = np.linspace(0.08, 0.16, cfg.d).astype(np.float32)
cfg.mu = torch.tensor(mu_vals, device=DEVICE, dtype=torch.float32)

vols = np.linspace(0.25, 0.45, cfg.d).astype(np.float32)
Sigma = np.diag((vols ** 2).astype(np.float32))
cfg.Sigma = torch.tensor(Sigma, device=DEVICE, dtype=torch.float32)
cfg.L_chol = torch.linalg.cholesky(cfg.Sigma)

cfg.W0 = 1.0
cfg.W_min = 0.4
cfg.W_max = 1e4

cfg.baseline_iters = 1500
cfg.anchored_iters = 1500
cfg.batch_size = 512

cfg.lr_actor = 3e-4
cfg.lr_critic = 3e-4
cfg.weight_decay = 0.0

cfg.eps_ruin = 0.10
cfg.delta_perf = 0.00
cfg.eps_cons = 0.05
cfg.lambda_dev = 1e-3
cfg.lr_lambda = 0.05

cfg.c_max_ratio = 0.8

# PMP / HJB penalty weights
cfg.alpha_pmp = 1e-2
cfg.alpha_hjb = 1e-3

cfg.plot_folder = "./plots_full_safe_ppgdpo"
os.makedirs(cfg.plot_folder, exist_ok=True)


#  Utilities
def utility_crra(c, gamma):
    eps = 1e-8
    if gamma == 1.0:
        return torch.log(c.clamp(min=eps))
    else:
        return (c.clamp(min=eps) ** (1.0 - gamma)) / (1.0 - gamma)


class FastSimplexProjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v):
        B, d = v.shape
        u, _ = torch.sort(v, dim=1, descending=True)
        cssv = torch.cumsum(u, dim=1) - 1
        ind = torch.arange(1, d + 1, device=v.device, dtype=v.dtype).view(1, -1)
        cond = u - cssv / ind > 0
        rho = cond.sum(dim=1, keepdim=True)
        rho = torch.clamp(rho, min=1)
        theta = cssv.gather(1, rho - 1) / rho
        w = torch.clamp(v - theta, min=0.0)
        return w


simplex_proj = FastSimplexProjection().to(DEVICE)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3, act=nn.SiLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(act())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TimeNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = MLP(1, 128)
        self.head = nn.Linear(128, out_dim)

    def forward(self, t):
        h = self.mlp(t)
        return self.head(h)


#  Policies
class BaselinePolicy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        # kappa(t) network (time-only)
        self.time_net = TimeNet(1)
        # portfolio network
        self.pi_net = MLP(2, 128)
        self.head_pi = nn.Linear(128, d)

    def forward(self, t, w):
        # consumption ratio kappa(t) ≥ 0
        kappa_raw = self.time_net(t)
        kappa = torch.nn.functional.softplus(kappa_raw)
        c = kappa * w
        c_cap = cfg.c_max_ratio * w
        c = torch.minimum(c, c_cap)

        # portfolio
        x_pi = torch.cat([t, torch.log(w.clamp(min=1e-8))], dim=-1)
        h = self.pi_net(x_pi)
        raw_pi = self.head_pi(h)
        pi = simplex_proj(raw_pi)
        return c, pi, kappa


class ResidualPolicy(nn.Module):
    def __init__(self, d, scale_pi=0.1, scale_kappa=0.1):
        super().__init__()
        self.d = d
        self.time_net = TimeNet(1)
        self.pi_net = MLP(2, 128)
        self.head_pi = nn.Linear(128, d)
        self.scale_pi = scale_pi
        self.scale_kappa = scale_kappa

    def forward(self, t, w):
        dk_raw = self.time_net(t)
        dk = torch.tanh(dk_raw) * self.scale_kappa

        x_pi = torch.cat([t, torch.log(w.clamp(min=1e-8))], dim=-1)
        h = self.pi_net(x_pi)
        dpi = torch.tanh(self.head_pi(h)) * self.scale_pi
        return dk, dpi


class AnchoredPolicy(nn.Module):
    def __init__(self, base_pol: BaselinePolicy, res_pol: ResidualPolicy):
        super().__init__()
        self.base_pol = base_pol
        self.res_pol = res_pol

    def forward(self, t, w):
        with torch.no_grad():
            c0, pi0, kappa0 = self.base_pol(t, w)
        dk, dpi = self.res_pol(t, w)

        kappa = torch.nn.functional.softplus(kappa0 + dk)
        c = kappa * w
        c_cap = cfg.c_max_ratio * w
        c = torch.minimum(c, c_cap)

        pi = simplex_proj(pi0 + dpi)
        return c, pi, kappa, c0.detach(), pi0.detach(), kappa0.detach()


class CriticV(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(2, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, t, w):
        x = torch.cat([t, torch.log(w.clamp(min=1e-8))], dim=-1)
        h = self.mlp(x)
        return self.head(h)


#  Rollout with pathwise gradient
def rollout_pg(policy_fn, cfg, batch_size, collect_base=False):
    d = cfg.d
    dt = cfg.dt
    steps = cfg.steps
    r = cfg.r
    mu = cfg.mu
    Sigma = cfg.Sigma
    L_chol = cfg.L_chol

    W = torch.full((batch_size, 1), cfg.W0, device=DEVICE, dtype=torch.float32)
    ruin = torch.zeros(batch_size, 1, dtype=torch.bool, device=DEVICE)

    J_paths = []
    cons_paths = []
    risk_paths = []
    t_list = []
    w_list = []
    c_list = []
    pi_list = []
    base_cons_paths = [] if collect_base else None

    for k in range(steps):
        t = torch.full((batch_size, 1), k * dt, device=DEVICE, dtype=torch.float32)

        out = policy_fn(t, W)
        if collect_base:
            c, pi, kappa, c0, pi0, kappa0 = out
            base_cons_paths.append(c0)
        else:
            c, pi, kappa = out

        t_list.append(t)
        w_list.append(W)
        c_list.append(c)
        pi_list.append(pi)

        kappa_ratio = c / W.clamp(min=1e-8)

        dB = torch.randn(batch_size, d, device=DEVICE, dtype=torch.float32)
        dB = dB @ L_chol.T * math.sqrt(dt)

        excess = (mu - cfg.r).view(1, -1)
        port_ret = (pi * excess).sum(dim=1, keepdim=True)

        quad = torch.einsum("bi,ij,bj->b", pi, Sigma, pi).view(-1, 1)

        drift_step = (r + port_ret - kappa_ratio) * dt
        diff_val = (pi * dB).sum(dim=1, keepdim=True)

        expo = drift_step - 0.5 * quad * dt + diff_val
        expo = torch.clamp(expo, min=-50.0, max=50.0)
        W_next = W * torch.exp(expo)
        W_next = W_next.clamp(min=1e-10, max=cfg.W_max)

        reward_inst = torch.exp(-cfg.delta * t) * utility_crra(c, cfg.gamma) * dt

        ruin = ruin | (W_next < cfg.W_min)
        J_paths.append(reward_inst)
        cons_paths.append(c)
        risk_paths.append(quad)

        W = W_next

    t_T = torch.full((batch_size, 1), cfg.T, device=DEVICE, dtype=torch.float32)
    J_T = cfg.beta * torch.exp(-cfg.delta * t_T) * utility_crra(W, cfg.gamma)
    J_total = J_T + torch.stack(J_paths, dim=0).sum(dim=0)

    ruin_prob = ruin.float().mean()
    avg_risk = torch.stack(risk_paths, dim=0).mean()
    cons_tensor = torch.stack(cons_paths, dim=0)

    t_traj = torch.stack(t_list, dim=0)
    w_traj = torch.stack(w_list, dim=0)
    c_traj = torch.stack(c_list, dim=0)
    pi_traj = torch.stack(pi_list, dim=0)

    if collect_base:
        base_cons_tensor = torch.stack(base_cons_paths, dim=0)
        return (
            J_total,
            ruin_prob,
            avg_risk,
            cons_tensor,
            t_traj,
            w_traj,
            c_traj,
            pi_traj,
            base_cons_tensor,
        )
    else:
        return J_total, ruin_prob, avg_risk, cons_tensor, t_traj, w_traj, c_traj, pi_traj


#  PMP + HJB residuals (fixed graph bug)
def compute_pmp_hjb_residuals(critic, t_traj, w_traj, c_traj, pi_traj, cfg, num_samples=2048):
    steps, B, _ = t_traj.shape
    d = cfg.d

    t_flat = t_traj.reshape(-1, 1)
    w_flat = w_traj.reshape(-1, 1)
    c_flat = c_traj.reshape(-1, 1)
    pi_flat = pi_traj.reshape(-1, d)

    N = t_flat.shape[0]
    idx = torch.randint(0, N, (min(num_samples, N),), device=DEVICE)
    t_s = t_flat[idx]
    w_s = w_flat[idx]
    c_s = c_flat[idx]
    pi_s = pi_flat[idx]

    # make t_s, w_s leaf vars for critic derivatives
    t_s = t_s.detach().requires_grad_(True)
    w_s = w_s.detach().requires_grad_(True)

    v = critic(t_s, w_s)
    v_sum = v.sum()

    # joint gradient to avoid "backward through graph twice" error
    v_w, v_t = torch.autograd.grad(
        v_sum, (w_s, t_s), create_graph=True
    )

    v_ww = torch.autograd.grad(v_w.sum(), w_s, create_graph=True)[0]

    u_c = (c_s.clamp(min=1e-8) ** (-cfg.gamma))

    excess = (cfg.mu - cfg.r).view(1, -1)
    port_ret = (pi_s * excess).sum(dim=1, keepdim=True)
    quad = torch.einsum("bi,ij,bj->b", pi_s, cfg.Sigma, pi_s).view(-1, 1)

    f = cfg.r * w_s + w_s * port_ret - c_s
    g_sq = (w_s**2) * quad

    H = utility_crra(c_s, cfg.gamma) + v_w * f + 0.5 * v_ww * g_sq
    r_hjb = v_t + H

    dH_dc = u_c - v_w
    term1 = v_w * w_s * excess
    term2 = v_ww * (w_s**2) * (pi_s @ cfg.Sigma)
    dH_dpi = term1 + term2

    L_hjb = (r_hjb**2).mean()
    L_pmp_c = (dH_dc**2).mean()
    L_pmp_pi = (dH_dpi**2).sum(dim=1).mean()
    L_pmp = L_pmp_c + L_pmp_pi

    return L_pmp, L_hjb


#  Stage 0: Baseline Projected PG-DPO
def train_baseline_ppgdpo_multi():
    policy = BaselinePolicy(cfg.d).to(DEVICE)
    critic = CriticV().to(DEVICE)

    opt_actor = optim.Adam(policy.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

    for it in range(cfg.baseline_iters):
        def pol(t, w):
            c, pi, kappa = policy(t, w)
            return c, pi, kappa

        (
            J_total,
            ruin_prob,
            avg_risk,
            cons_tensor,
            t_traj,
            w_traj,
            c_traj,
            pi_traj,
        ) = rollout_pg(pol, cfg, cfg.batch_size, collect_base=False)
        J_mean = J_total.mean()

        L_pmp, L_hjb = compute_pmp_hjb_residuals(critic, t_traj, w_traj, c_traj, pi_traj, cfg)

        actor_loss = -J_mean + cfg.alpha_pmp * L_pmp + cfg.alpha_hjb * L_hjb

        opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        opt_actor.step()

        with torch.no_grad():
            t0 = torch.zeros(cfg.batch_size, 1, device=DEVICE, dtype=torch.float32)
            w0 = torch.full((cfg.batch_size, 1), cfg.W0, device=DEVICE, dtype=torch.float32)
            target_v = J_total

        opt_critic.zero_grad()
        v_pred = critic(t0, w0).view(-1, 1)
        critic_loss = ((v_pred - target_v.detach()) ** 2).mean()
        critic_loss.backward()
        opt_critic.step()

        if (it + 1) % 300 == 0:
            print(
                f"[Baseline] it={it+1:4d}  J={J_mean.item():+.4e}  "
                f"ruin={ruin_prob.item():.4f}  risk={avg_risk.item():.4f}  "
                f"L_pmp={L_pmp.item():.3e}  L_hjb={L_hjb.item():.3e}"
            )

    with torch.no_grad():
        def pol_eval(t, w):
            c, pi, kappa = policy(t, w)
            return c, pi, kappa

        (
            J_total,
            ruin_prob,
            avg_risk,
            cons_tensor,
            t_traj,
            w_traj,
            c_traj,
            pi_traj,
        ) = rollout_pg(pol_eval, cfg, 4096, collect_base=False)
        J0 = J_total.mean().item()
        ruin0 = ruin_prob.item()
        risk0 = avg_risk.item()
    print(f"[Baseline] Final: J0={J0:+.4e}, ruin={ruin0:.4f}, risk={risk0:.4f}")

    return policy, critic, J0


#  Stage 1: Anchored constrained PG-DPO
def train_anchored_constrained_ppgdpo(base_policy, critic, baseline_J0):
    res_policy = ResidualPolicy(cfg.d).to(DEVICE)
    anchored_policy = AnchoredPolicy(base_policy, res_policy).to(DEVICE)

    opt_res = optim.Adam(res_policy.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)

    lambda_ruin = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    lambda_perf = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    lambda_cons = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)
    lambda_risk = torch.tensor(0.5, device=DEVICE, dtype=torch.float32)

    for it in range(cfg.anchored_iters):
        def pol(t, w):
            c, pi, kappa, c0, pi0, kappa0 = anchored_policy(t, w)
            return c, pi, kappa, c0, pi0, kappa0

        (
            J_total,
            ruin_prob,
            avg_risk,
            cons_tensor,
            t_traj,
            w_traj,
            c_traj,
            pi_traj,
            base_cons_tensor,
        ) = rollout_pg(pol, cfg, cfg.batch_size, collect_base=True)
        J_mean = J_total.mean()

        L_pmp, L_hjb = compute_pmp_hjb_residuals(critic, t_traj, w_traj, c_traj, pi_traj, cfg)

        g_ruin = ruin_prob - cfg.eps_ruin
        g_perf = baseline_J0 - J_mean - cfg.delta_perf
        cons_diff = (cons_tensor - base_cons_tensor) ** 2
        cons_dev = cons_diff.mean()
        g_cons = cons_dev - cfg.eps_cons
        g_risk = avg_risk - 0.5 * cfg.d

        dev_penalty = 0.0
        for p in res_policy.parameters():
            dev_penalty = dev_penalty + (p**2).mean()
        dev_penalty = cfg.lambda_dev * dev_penalty

        actor_loss = (
            -J_mean
            + cfg.alpha_pmp * L_pmp
            + cfg.alpha_hjb * L_hjb
            + lambda_ruin * g_ruin
            + lambda_perf * g_perf
            + lambda_cons * g_cons
            + lambda_risk * g_risk
            + dev_penalty
        )

        opt_res.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(res_policy.parameters(), max_norm=10.0)
        opt_res.step()

        with torch.no_grad():
            lambda_ruin = torch.clamp(lambda_ruin + cfg.lr_lambda * g_ruin, min=0.0)
            lambda_perf = torch.clamp(lambda_perf + cfg.lr_lambda * g_perf, min=0.0)
            lambda_cons = torch.clamp(lambda_cons + cfg.lr_lambda * g_cons, min=0.0)
            lambda_risk = torch.clamp(lambda_risk + cfg.lr_lambda * g_risk, min=0.0)

        with torch.no_grad():
            t0 = torch.zeros(cfg.batch_size, 1, device=DEVICE, dtype=torch.float32)
            w0 = torch.full((cfg.batch_size, 1), cfg.W0, device=DEVICE, dtype=torch.float32)
            target_v = J_total

        opt_critic.zero_grad()
        v_pred = critic(t0, w0).view(-1, 1)
        critic_loss = ((v_pred - target_v.detach()) ** 2).mean()
        critic_loss.backward()
        opt_critic.step()

        if (it + 1) % 300 == 0:
            print(
                f"[Anchored] it={it+1:4d}  J={J_mean.item():+.4e}  ruin={ruin_prob.item():.4f}  "
                f"risk={avg_risk.item():.4f}  g_r={g_ruin.item():+.3e}  "
                f"g_p={g_perf.item():+.3e}  g_c={g_cons.item():+.3e}  "
                f"L_pmp={L_pmp.item():.3e}  L_hjb={L_hjb.item():.3e}"
            )

    return anchored_policy


#  Visualization
def contour_policy_c_pi(policy_fn, title_prefix):
    d = cfg.d
    t_vals = torch.linspace(0.0, cfg.T, 128, device=DEVICE, dtype=torch.float32).view(-1, 1)
    w_vals = torch.linspace(0.2, 5.0, 128, device=DEVICE, dtype=torch.float32).view(-1, 1)

    TT, WW = torch.meshgrid(t_vals.view(-1), w_vals.view(-1), indexing="ij")
    t_grid = TT.reshape(-1, 1)
    w_grid = WW.reshape(-1, 1)

    with torch.no_grad():
        out = policy_fn(t_grid, w_grid)
        c = out[0]
        pi = out[1]

    C = c.view(128, 128).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(TT.cpu().numpy(), WW.cpu().numpy(), C, levels=64)
    fig.colorbar(cs)
    ax.set_xlabel("t")
    ax.set_ylabel("w")
    ax.set_title(f"{title_prefix}  c(t,w)")
    fig.savefig(os.path.join(cfg.plot_folder, f"{title_prefix}_c.png"), dpi=150)
    plt.close(fig)

    for j in range(d):
        Pj = pi[:, j].view(128, 128).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        cs = ax.contourf(TT.cpu().numpy(), WW.cpu().numpy(), Pj, levels=64)
        fig.colorbar(cs)
        ax.set_xlabel("t")
        ax.set_ylabel("w")
        ax.set_title(f"{title_prefix}  pi_{j}(t,w)")
        fig.savefig(os.path.join(cfg.plot_folder, f"{title_prefix}_pi{j}.png"), dpi=150)
        plt.close(fig)


#  Main
def main():
    print("Configuration Loaded. Mode: Full Safe Projected PG-DPO (Anchored)")
    print("Stage 0: Baseline Projected PG-DPO (unconstrained)")
    base_policy, critic, J0 = train_baseline_ppgdpo_multi()

    print("\nStage 1: Anchored constrained PG-DPO (CA-P-PGDPO)")
    anchored_policy = train_anchored_constrained_ppgdpo(base_policy, critic, J0)

    print("\nPlotting baseline policy surfaces...")
    contour_policy_c_pi(lambda t, w: base_policy(t, w), "Baseline_PPGDPO")

    print("Plotting anchored constrained policy surfaces...")
    contour_policy_c_pi(lambda t, w: anchored_policy(t, w), "CA_PPGDPO_safe")

    print("finished")


if __name__ == "__main__":
    main()
