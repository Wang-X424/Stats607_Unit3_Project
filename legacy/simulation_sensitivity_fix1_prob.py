
# Sensitivity simulation evaluated DIRECTLY on the edge-probability matrix.
# Same data-generating process / model / training as simulation_sensitivity_fix1_AUC.py,
# but instead of AUC against the sampled A, we compare the estimated P_hat with the
# true P = sigmoid(alpha + Z Z^T), which is known in simulation.
#
# Metrics (per block, per estimator):
#   relfro_all : ||P_hat - P||_F / ||P||_F                         (whole block)
#   relfro_avg : mean_i ||P_hat_i - P_i||_2 / ||P_i||_2             (row-wise, averaged over nodes)
#   relfro_cent: ||P_hat - P||_F / ||P - pbar||_F, baseline = constant predictor (stored only)
#   rmse       : ||P_hat - P||_F / sqrt(#pairs)                     (stored, not plotted by default)
#   kl         : mean_ij KL(Bern(P_ij) || Bern(P_hat_ij))
#   relkl      : kl / kl_null, kl_null = mean_ij KL(Bern(P_ij) || Bern(pbar)),
#                pbar = mean of P over the block (best constant predictor).  1 = no better
#                than knowing the density, 0 = perfect.
#   relfro_theta (plug-in only): ||Theta_hat - Theta||_F / ||Theta||_F on the logit scale
#
# Estimators:
#   pi : plug-in      P_hat_ij = sigmoid(mu_i^T mu_j + b)
#   mc : MC posterior P_hat_ij = E_q[sigmoid(z_i^T z_j + b)], averaged over MC_SAMPLES draws
#
# Blocks (rows/cols in training order): onlyX x A (main, cold start), onlyX x onlyA,
# onlyX x both, A x A (off-diagonal).
#
# Usage:  python simulation_sensitivity_fix1_prob.py 30 40 30 0
# Quick smoke test: SIM_QUICK=1 python simulation_sensitivity_fix1_prob.py 30 40 30 0

import os
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import matplotlib.ticker as mticker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# torch.set_num_threads(4) # when run this .py in Greatlake, decomment this sentence

# ============================================================
# SHARED CONFIG (fixed across all scenarios)
# ============================================================
SEED        = 42
Z_DIM_TRUE  = 8

Z_VAR   = 1.0                 # Var(Z_ik) in the data-generating process (was 4 before)
Z_SCALE = math.sqrt(Z_VAR)
TAU2    = Z_VAR               # prior variance in the VAE's KL term; matched to the truth

percents = {'onlyA': int(sys.argv[1]), 'both': int(sys.argv[2]), 'onlyX': int(sys.argv[3]), 'test': int(sys.argv[4])}
PROP_ONLYA  = percents['onlyA']
PROP_BOTH   = percents['both']
PROP_ONLYX  = percents['onlyX']

# VAE hyperparameters
Z_DIM_FIT   = 8
H_DIM_FIT   = 128
LR          = 1e-3
L2_PAR      = 1e-2
SELF_WEIGHT = 1.0

# Training
N_ITER      = 1200

# Evaluation
MC_SAMPLES   = 50       # posterior draws for the MC estimator
MC_EVERY_LOG = False    # True: MC metrics at every logged iter; False: only at the final iter (cheaper)
MIN_EXP_DEG  = 1.0      # relfro_avg skips rows whose expected degree sum_j P_ij < this (row norm ~ 0)

# ============================================================
# SCENARIO GRID
# With Z_VAR = 1, sd(Z_i^T Z_j) ~ 2.8, so alpha = -20 / -15 give (almost) empty graphs.
# alpha in {-8,...,0} spans density ~1% to ~50%; alpha = -6 (density ~3.7%, expected
# degree ~26 at n=700) roughly matches the density of the old Z_VAR=4, alpha=-20 setting.
# ============================================================
N_LEVELS       = [100, 400, 700, 1100, 1500]
XDIM_LEVELS    = [100, 400, 700, 1100, 1500]
ALPHA_LEVELS   = [-8, -6, -4, -2, 0]

FIXED_ALPHA = -6
FIXED_N     = 700
FIXED_XDIM  = 700

N_REPS_SENS = 20

if os.environ.get('SIM_QUICK') == '1':          # tiny grid for a smoke test
    N_LEVELS, XDIM_LEVELS, ALPHA_LEVELS = [100, 200], [50, 100], [-6, -2]
    FIXED_ALPHA, FIXED_N, FIXED_XDIM = -6, 200, 100
    N_REPS_SENS, N_ITER, MC_SAMPLES = 2, 200, 20

NITER_LOG = np.insert(np.arange(49, N_ITER, 50), 0, 0)
print('NITER_LOG:', NITER_LOG)

SENS_DIR = './sensitivity_results_prob{}_zvar{:g}{}'.format(
    PROP_ONLYA, Z_VAR, '_quick' if os.environ.get('SIM_QUICK') == '1' else '')
os.makedirs(SENS_DIR, exist_ok=True)


# ============================================================
# Data generation / split / preprocessing  (unchanged except Z scale)
# ============================================================
def generate_data(n, z_dim_true, X_dim, alpha, seed):
    """
      Z_i ~ N(0, Z_VAR * I_d)
      A_ij | Z ~ Bernoulli(sigmoid(alpha + Z_i^T Z_j))
      X_ik | Z ~ Bernoulli(sigmoid([W2*ReLU(W1*Z+b1)+b2]_k))
    """
    rng = np.random.RandomState(seed)
    Z_true = rng.randn(n, z_dim_true).astype(np.float32) * Z_SCALE

    logits_A = alpha + Z_true @ Z_true.T
    prob_A   = 1.0 / (1.0 + np.exp(-logits_A))
    U        = rng.rand(n, n)
    A_upper  = (U < prob_A).astype(np.int32)
    A_true   = np.triu(A_upper, k=1)
    A_true   = A_true + A_true.T

    rng2    = np.random.RandomState(seed + 1000)
    h_dim_g = 32
    W1g = rng2.randn(z_dim_true, h_dim_g).astype(np.float32)
    b1g = rng2.randn(h_dim_g).astype(np.float32) * 0.1
    W2g = rng2.randn(h_dim_g, X_dim).astype(np.float32)
    b2g = rng2.randn(X_dim).astype(np.float32) * 0.1
    H        = np.maximum(0, Z_true @ W1g + b1g)
    logits_X = H @ W2g + b2g
    prob_X   = 1.0 / (1.0 + np.exp(-logits_X))
    X_true   = (rng2.rand(n, X_dim) < prob_X).astype(np.int32)

    density = A_true.sum() / (n * (n - 1))
    print(f'  Generated: n={n}, X_dim={X_dim}, edge density={density:.4f}')
    return Z_true, A_true, X_true


def split_nodes(n, prop_onlyA, prop_both, prop_onlyX, seed):
    rng     = np.random.RandomState(seed)
    perm    = rng.permutation(n)
    n_oA    = int(np.round(n * prop_onlyA / 100))
    n_b     = int(np.round(n * prop_both  / 100))
    n_oX    = min(int(np.round(n * prop_onlyX / 100)), n - n_oA - n_b)
    idx_oA  = perm[:n_oA]
    idx_b   = perm[n_oA : n_oA + n_b]
    idx_oX  = perm[n_oA + n_b : n_oA + n_b + n_oX]
    print(f'  Split: onlyA={len(idx_oA)}, both={len(idx_b)}, onlyX={len(idx_oX)}')
    return idx_oA, idx_b, idx_oX


def preprocess(data_X_obs, data_A_obs, id_onlyA, id_both, id_onlyX,
               id_A, id_X, X_dim, self_weight):

    edge_prob       = (data_A_obs.sum() / 2.0) / (
                       data_A_obs.shape[0] * (data_A_obs.shape[0]-1) / 2.0 + 1e-10)
    logit_edge_prob = float(np.log(edge_prob / (1.0 - edge_prob + 1e-10) + 1e-10))

    _X_obs_mean = np.mean(data_X_obs, axis=0)
    _X_more     = np.concatenate((_X_obs_mean.reshape(1, X_dim),
                                   data_X_obs[:len(id_both), :]))
    _A_avg   = data_A_obs[np.ix_(id_onlyA, id_both)]
    _D       = np.sum(_A_avg, axis=1)
    _no_link = 1 * (_D == 0)
    _A_avg   = np.concatenate((_no_link.reshape(len(id_onlyA), 1), _A_avg), axis=1)
    _D       = 1.0 / np.sum(_A_avg, axis=1)
    data_A_avg  = np.matmul(np.diag(_D), _A_avg)
    _X_onlyA    = np.matmul(data_A_avg, _X_more)
    data_X_fill = np.concatenate((_X_onlyA, data_X_obs))

    node_degree = np.sum(data_A_obs, axis=1)
    node_degree = np.where(node_degree == 0, 1, node_degree)
    self_weight_vec = self_weight * node_degree
    _ATilde = data_A_obs + np.diag(self_weight_vec)

    _DTilde0 = np.sum(_ATilde, axis=1)
    _DTilde0 = 1.0 / np.power(_DTilde0, 0.5)
    data_AHat = np.matmul(np.matmul(np.diag(_DTilde0), _ATilde), np.diag(_DTilde0))

    mask_A_col = np.tile(np.arange(len(id_A)), (len(id_A), 1))
    mask_A_obs = np.transpose(mask_A_col) < mask_A_col
    return logit_edge_prob, data_X_fill, data_AHat, mask_A_obs


class VAE(torch.nn.Module):
    def __init__(self, X_dim, h_dim, z_dim, logit_edge_prob):
        super().__init__()
        xi = lambda i, o: torch.randn(i, o) * (1.0 / (i/2.0)**0.5)
        self.PX_W1      = torch.nn.Parameter(xi(z_dim,  h_dim))
        self.PX_b1      = torch.nn.Parameter(torch.zeros(h_dim))
        self.PX_W2      = torch.nn.Parameter(xi(h_dim,  X_dim))
        self.PX_b2      = torch.nn.Parameter(torch.zeros(X_dim))
        self.PA_dens    = torch.nn.Parameter(torch.tensor(logit_edge_prob, dtype=torch.float32))
        self.Q_W1       = torch.nn.Parameter(xi(X_dim,  h_dim))
        self.Q_b1       = torch.nn.Parameter(torch.zeros(h_dim))
        self.Q_W2_mu    = torch.nn.Parameter(xi(h_dim,  z_dim))
        self.Q_b2_mu    = torch.nn.Parameter(torch.zeros(z_dim))
        self.Q_W2_sigma = torch.nn.Parameter(xi(h_dim,  z_dim))
        self.Q_b2_sigma = torch.nn.Parameter(torch.zeros(z_dim))

    def PX(self, z):
        logits = F.relu(z @ self.PX_W1 + self.PX_b1) @ self.PX_W2 + self.PX_b2
        return torch.sigmoid(logits), logits

    def PA(self, z):
        logits = z @ z.t() + self.PA_dens
        return torch.sigmoid(logits), logits

    def Q(self, X_fill, AHat, n_A, n_onlyX):
        X_A, X_oX = X_fill[:n_A], X_fill[n_A:n_A+n_onlyX]
        h_oX = F.relu(X_oX @ self.Q_W1 + self.Q_b1)
        z_oX_mu  = h_oX @ self.Q_W2_mu    + self.Q_b2_mu
        z_oX_lv  = h_oX @ self.Q_W2_sigma + self.Q_b2_sigma
        h_A  = F.relu((AHat @ X_A) @ self.Q_W1 + self.Q_b1)
        z_A_mu   = (AHat @ h_A) @ self.Q_W2_mu    + self.Q_b2_mu
        z_A_lv   = (AHat @ h_A) @ self.Q_W2_sigma + self.Q_b2_sigma
        return (torch.cat([z_A_mu,  z_oX_mu],  0),
                torch.cat([z_A_lv,  z_oX_lv],  0))

    def sample_z(self, mu, lv):
        return mu + torch.exp(lv / 2.0) * torch.randn_like(mu)

    def l2_reg(self):
        return (self.PX_W1.pow(2).sum() + self.PX_W2.pow(2).sum() +
                self.Q_W1.pow(2).sum()  + self.Q_W2_mu.pow(2).sum() +
                self.Q_W2_sigma.pow(2).sum()) / 2


# ============================================================
# Probability-matrix metrics (float64, computed in log space)
# ============================================================
def bern_kl(logp, log1mp, logq, log1mq):
    """Elementwise KL(Bern(p) || Bern(q)) from log-probabilities (stable when p, q ~ 0 or 1)."""
    p = torch.exp(logp)
    return p * (logp - logq) + (1.0 - p) * (log1mp - log1mq)


def prob_metrics(theta, logq, log1mq, mask=None):
    """
    theta        : true logits of the block, (r, c) float64
    logq, log1mq : log P_hat and log(1 - P_hat) of the block, (r, c) float64
    mask         : optional bool (r, c); False entries are excluded (e.g. diagonal of A x A)
    """
    if mask is None:
        mask = torch.ones_like(theta, dtype=torch.bool)
    w = mask.to(theta.dtype)
    m = w.sum()

    logp, log1mp = F.logsigmoid(theta), F.logsigmoid(-theta)
    p, q = torch.exp(logp), torch.exp(logq)
    sq   = (q - p).pow(2) * w

    # whole-block Frobenius
    relfro_all = torch.sqrt(sq.sum()) / torch.sqrt((p.pow(2) * w).sum())
    rmse       = torch.sqrt(sq.sum() / m)
    pbar_p     = (p * w).sum() / m
    # centred version: baseline is the constant predictor pbar (1 - relfro_cent^2 is an R^2)
    relfro_cent = torch.sqrt(sq.sum()) / torch.sqrt(((p - pbar_p).pow(2) * w).sum())

    # row-wise relative Frobenius, averaged over rows with non-negligible expected degree
    row_num  = torch.sqrt(sq.sum(1))
    row_den  = torch.sqrt((p.pow(2) * w).sum(1))
    valid    = (p * w).sum(1) >= MIN_EXP_DEG
    relfro_avg = (row_num[valid] / row_den[valid]).mean() if valid.any() else torch.tensor(float('nan'))

    # KL, and KL relative to the best constant predictor pbar
    kl   = (bern_kl(logp, log1mp, logq, log1mq) * w).sum() / m
    pbar = ((p * w).sum() / m).clamp(1e-300, 1 - 1e-16)
    kl0  = (bern_kl(logp, log1mp, torch.log(pbar), torch.log1p(-pbar)) * w).sum() / m

    return {'relfro_all': relfro_all.item(), 'relfro_avg': relfro_avg.item(),
            'relfro_cent': relfro_cent.item(),
            'rmse': rmse.item(), 'kl': kl.item(), 'relkl': (kl / kl0).item(),
            'frac_rows': valid.double().mean().item()}


def mc_log_probs(model, z_mu, z_lv, S):
    """log E_q[sigmoid(theta)] and log E_q[1 - sigmoid(theta)] for all pairs, via running logsumexp."""
    mu, lv = z_mu.double(), z_lv.double()
    b = model.PA_dens.double()
    acc1 = acc0 = None
    for _ in range(S):
        z  = mu + torch.exp(lv / 2.0) * torch.randn_like(mu)
        th = z @ z.t() + b
        l1, l0 = F.logsigmoid(th), F.logsigmoid(-th)
        acc1 = l1 if acc1 is None else torch.logaddexp(acc1, l1)
        acc0 = l0 if acc0 is None else torch.logaddexp(acc0, l0)
    return acc1 - math.log(S), acc0 - math.log(S)


METRIC_KEYS = ['relfro_all', 'relfro_avg', 'relfro_cent', 'rmse', 'kl', 'relkl', 'frac_rows']


def evaluate(model, z_mu, z_lv, theta_true, blocks, do_mc):
    """theta_true: (n, n) float64 true logits in training order. Returns flat dict of metrics."""
    out = {}
    theta_hat = z_mu.double() @ z_mu.double().t() + model.PA_dens.double()
    est = {'pi': (F.logsigmoid(theta_hat), F.logsigmoid(-theta_hat))}
    if do_mc:
        est['mc'] = mc_log_probs(model, z_mu, z_lv, MC_SAMPLES)

    for bname, (rows, cols, offdiag) in blocks.items():
        th   = theta_true[rows][:, cols]
        mask = None
        if offdiag:
            mask = ~torch.eye(len(rows), dtype=torch.bool, device=th.device)
        for ename in ['pi', 'mc']:
            if ename in est:
                lq, l1q = est[ename]
                res = prob_metrics(th, lq[rows][:, cols], l1q[rows][:, cols], mask)
            else:
                res = {k: float('nan') for k in METRIC_KEYS}
            for k, v in res.items():
                out[f'{ename}_{k}_{bname}'] = v
        # logit-scale relative error (plug-in is the posterior mean of theta, so no MC needed)
        th_hat = theta_hat[rows][:, cols]
        w = 1.0 if mask is None else mask.double()
        out[f'pi_relfro_theta_{bname}'] = (torch.sqrt(((th_hat - th).pow(2) * w).sum())
                                           / torch.sqrt((th.pow(2) * w).sum())).item()
    return out


# ============================================================
# Scenario list (dedup by (N, X_DIM, Alpha))
# ============================================================
def make_label(N, X, A):
    return f'N{N}_X{X}_A{str(A).replace(".","").replace("-","neg")}'

seen = {}
for N in N_LEVELS:
    for X in XDIM_LEVELS:
        lbl = make_label(N, X, FIXED_ALPHA)
        if lbl not in seen:
            seen[lbl] = {'label': lbl, 'N_NODES': N, 'X_DIM': X, 'Alpha': FIXED_ALPHA}
for X in XDIM_LEVELS:
    for A in ALPHA_LEVELS:
        lbl = make_label(FIXED_N, X, A)
        if lbl not in seen:
            seen[lbl] = {'label': lbl, 'N_NODES': FIXED_N, 'X_DIM': X, 'Alpha': A}
for N in N_LEVELS:
    for A in ALPHA_LEVELS:
        lbl = make_label(N, FIXED_XDIM, A)
        if lbl not in seen:
            seen[lbl] = {'label': lbl, 'N_NODES': N, 'X_DIM': FIXED_XDIM, 'Alpha': A}

SENSITIVITY_SCENARIOS = list(seen.values())
print(f'Total unique scenarios: {len(SENSITIVITY_SCENARIOS)}')


# ============================================================
# run_one_scenario
# ============================================================
def run_one_scenario(cfg, n_reps, n_iter, seed_base=SEED):
    N       = cfg['N_NODES']
    X_DIM_s = cfg['X_DIM']
    alpha   = cfg['Alpha']
    niter_log = np.insert(np.arange(49, n_iter, 50), 0, 0)

    all_rep = []
    for rep in range(n_reps):
        Z_true, A_true, X_true = generate_data(
            N, Z_DIM_TRUE, X_DIM_s, alpha, seed=seed_base + rep)

        idx_oA, idx_b, idx_oX = split_nodes(
            N, PROP_ONLYA, PROP_BOTH, PROP_ONLYX, seed=seed_base + rep + 100)

        num_train = np.concatenate([idx_oA, idx_b, idx_oX])
        n         = len(num_train)
        id_onlyA  = np.arange(len(idx_oA))
        id_both   = np.arange(len(idx_oA), len(idx_oA) + len(idx_b))
        id_onlyX  = np.arange(len(idx_oA) + len(idx_b), n)
        id_A      = np.concatenate([id_onlyA, id_both])
        id_X      = np.concatenate([id_both,  id_onlyX])

        data_A_all = A_true[np.ix_(num_train, num_train)]
        data_X_obs = X_true[num_train][id_X]
        data_A_obs = data_A_all[np.ix_(id_A, id_A)]

        # true logits in training order (float64)
        Zp = torch.tensor(Z_true[num_train], dtype=torch.float64, device=device)
        theta_true = alpha + Zp @ Zp.t()

        t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
        blocks = {
            'onlyX_A':     (t(id_onlyX), t(id_A),     False),
            'onlyX_onlyA': (t(id_onlyX), t(id_onlyA), False),
            'onlyX_both':  (t(id_onlyX), t(id_both),  False),
            'A_A':         (t(id_A),     t(id_A),     True),
        }

        logit_ep, data_X_fill, data_AHat, mask_A_obs = preprocess(
            data_X_obs, data_A_obs, id_onlyA, id_both, id_onlyX,
            id_A, id_X, X_DIM_s, SELF_WEIGHT)

        t_Xo   = torch.tensor(data_X_obs,  dtype=torch.float32, device=device)
        t_Ao   = torch.tensor(data_A_obs,  dtype=torch.float32, device=device)
        t_Xf   = torch.tensor(data_X_fill, dtype=torch.float32, device=device)
        t_AH   = torch.tensor(data_AHat,   dtype=torch.float32, device=device)
        t_mask = torch.tensor(mask_A_obs,  dtype=torch.bool,    device=device)

        model = VAE(X_DIM_s, H_DIM_FIT, Z_DIM_FIT, logit_ep).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=1e-5)

        results = []
        for it in range(n_iter):
            model.train(); opt.zero_grad()
            z_mu, z_lv = model.Q(t_Xf, t_AH, len(id_A), len(id_onlyX))
            z_s        = model.sample_z(z_mu, z_lv)
            _, lgX     = model.PX(z_s)
            _, lgA     = model.PA(z_s)
            lgX_obs    = lgX[len(id_onlyA):len(id_onlyA) + len(id_X)]
            lgA_obs    = lgA[:len(id_A), :len(id_A)]
            loss = (F.binary_cross_entropy_with_logits(lgA_obs[t_mask], t_Ao[t_mask], reduction='sum')
                  + F.binary_cross_entropy_with_logits(lgX_obs, t_Xo, reduction='sum')
                  + 0.5 * torch.sum((torch.exp(z_lv) + z_mu.pow(2)) / TAU2 - 1.0 - z_lv + np.log(TAU2))
                  + model.l2_reg() * L2_PAR)
            loss.backward()
            opt.step()
            scheduler.step()

            if it in niter_log:
                model.eval()
                is_last = (it == niter_log[-1])
                with torch.no_grad():
                    zm, zlv = model.Q(t_Xf, t_AH, len(id_A), len(id_onlyX))
                    rc = {'iter': it, 'loss': loss.item(),
                          'logvar_mean_onlyX': zlv[len(id_A):].mean().item(),
                          'logvar_mean_A':     zlv[:len(id_A)].mean().item()}
                    rc.update(evaluate(model, zm, zlv, theta_true, blocks,
                                       do_mc=(MC_EVERY_LOG or is_last)))
                results.append(rc)

        all_rep.append(results)
        f = results[-1]
        print(f'  rep {rep+1}/{n_reps} | onlyX_A  relfro_all mc={f["mc_relfro_all_onlyX_A"]:.4f} '
              f'pi={f["pi_relfro_all_onlyX_A"]:.4f} | relfro_avg mc={f["mc_relfro_avg_onlyX_A"]:.4f} '
              f'| relKL mc={f["mc_relkl_onlyX_A"]:.4f} pi={f["pi_relkl_onlyX_A"]:.4f}')
    return all_rep


# ============================================================
# Readable tab-separated log per scenario
# ============================================================
def write_scenario_txt(res, txt_path):
    cols = [k for k in res[0][-1].keys() if k not in ('iter', 'loss')]
    header = ['rep', 'iter', 'loss'] + cols
    with open(txt_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for rep_idx, rep in enumerate(res):
            for rc in rep:
                row = [rep_idx, rc['iter'], rc['loss']] + [rc.get(c, float('nan')) for c in cols]
                f.write('\t'.join(str(v) for v in row) + '\n')


# ============================================================
# Main loop (with resume support)
# ============================================================
sensitivity_results = {}

for cfg in SENSITIVITY_SCENARIOS:
    label     = cfg['label']
    save_path = os.path.join(SENS_DIR, f'sens_{label}.pkl')

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            sensitivity_results[label] = pickle.load(f)
        txt_path = os.path.join(SENS_DIR, f'sens_{label}.txt')
        if not os.path.exists(txt_path):
            write_scenario_txt(sensitivity_results[label], txt_path)
        print(f'[skip] {label} — loaded from cache')
        continue

    print(f'\n{"="*55}')
    print(f'Scenario: {label}  (N={cfg["N_NODES"]}, p={cfg["X_DIM"]}, alpha={cfg["Alpha"]})')
    print(f'{"="*55}')

    res = run_one_scenario(cfg, n_reps=N_REPS_SENS, n_iter=N_ITER)
    sensitivity_results[label] = res

    with open(save_path, 'wb') as f:
        pickle.dump(res, f)
    write_scenario_txt(res, os.path.join(SENS_DIR, f'sens_{label}.txt'))
    print(f'[saved] {label}')

print('\nAll scenarios complete.')


# ============================================================
# Plotting: 3 columns (fixed alpha / fixed n / fixed p) x one row per metric
# ============================================================
def get_final_with_err(sensitivity_results, label, metric):
    """(mean, sd) across reps of the last non-nan logged value of `metric` (sd: ddof=0)."""
    vals_per_rep = []
    for rep in sensitivity_results[label]:
        vals = [r[metric] for r in rep if metric in r and not np.isnan(r[metric])]
        if vals:
            vals_per_rep.append(vals[-1])
    if not vals_per_rep:
        print(f'Warning: all nan for {label} / {metric}')
        return float('nan'), float('nan')
    arr = np.asarray(vals_per_rep, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def plot_sensitivity_final(sensitivity_results, metrics, row_labels, save_path=None, title=None):

    COLORS  = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#0072B2']
    MARKERS = ['o', '^', 's', 'D', 'P', 'v']
    MS, LW, BAND_ALPHA = 3, 1.8, 0.18

    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.0 * n_rows), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.18)

    col_headers = [
        f'$\\alpha$ = {FIXED_ALPHA} (fixed)',
        f'$n$ = {FIXED_N} (fixed)',
        f'$p$ = {FIXED_XDIM} (fixed)',
    ]
    for col, hdr in enumerate(col_headers):
        axes[0, col].set_title(hdr, fontsize=10, pad=6, color='#444444', fontstyle='italic')

    panels = [
        (N_LEVELS,    '$n$', XDIM_LEVELS,  lambda x: f'$p$ = {x}',
         lambda n, x: make_label(n, x, FIXED_ALPHA)),
        (XDIM_LEVELS, '$p$', ALPHA_LEVELS, lambda a: f'$\\alpha$ = {a}',
         lambda x, a: make_label(FIXED_N, x, a)),
        (N_LEVELS,    '$n$', ALPHA_LEVELS, lambda a: f'$\\alpha$ = {a}',
         lambda n, a: make_label(n, FIXED_XDIM, a)),
    ]

    # shared y-range within each row, covering mean +/- sd (errors are >= 0, not bounded by 1)
    row_ylims = {}
    for row, metric in enumerate(metrics):
        vals = []
        for x_levels, _, line_levels, _, lbl_fn in panels:
            for lv in line_levels:
                for xv in x_levels:
                    m, s = get_final_with_err(sensitivity_results, lbl_fn(xv, lv), metric)
                    if not np.isnan(m):
                        s = 0.0 if np.isnan(s) else s
                        vals += [m + s, max(0.0, m - s)]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.08 * (hi - lo) if hi > lo else 0.05 * max(hi, 1e-3)
            row_ylims[row] = (max(0.0, lo - pad), hi + pad)
        else:
            row_ylims[row] = (0.0, 1.0)

    for row, metric in enumerate(metrics):
        for col, (x_levels, x_label, line_levels, line_fmt, lbl_fn) in enumerate(panels):
            ax = axes[row, col]
            for li, lv in enumerate(line_levels):
                ms_, sds = [], []
                for xv in x_levels:
                    m, s = get_final_with_err(sensitivity_results, lbl_fn(xv, lv), metric)
                    ms_.append(m); sds.append(s)
                means = np.asarray(ms_, dtype=float)
                sds   = np.asarray(sds, dtype=float)
                xpos  = np.asarray(x_levels, dtype=float)

                valid = ~np.isnan(means) & ~np.isnan(sds)
                ax.fill_between(xpos, np.maximum(means - sds, 0.0), means + sds, where=valid,
                                color=COLORS[li], alpha=BAND_ALPHA, linewidth=0, zorder=2,
                                interpolate=False)
                ax.plot(xpos, means, color=COLORS[li], marker=MARKERS[li], markersize=MS,
                        linewidth=LW, linestyle='-', label=line_fmt(lv), zorder=3)

            ax.set_ylim(*row_ylims[row])
            ax.set_xscale('linear')
            ax.set_xlim(min(x_levels) * 0.85, max(x_levels) * 1.15)
            ax.set_xticks(list(x_levels))
            ax.set_xticklabels([str(v) for v in x_levels], fontsize=9)
            ax.minorticks_off()
            ax.set_xlabel(x_label, fontsize=10)
            ax.tick_params(axis='y', labelsize=9)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
            ax.spines[['top', 'right']].set_visible(False)
            if col > 0:
                ax.tick_params(axis='y', labelleft=False)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)
            if row == 0:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.4, edgecolor='#cccccc',
                          handlelength=1.6, borderpad=0.6)

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.close(fig)


BLOCK = 'onlyX_A'   # main evaluation block (cold-start nodes vs. network-observed nodes)
PLOT_METRICS = ['relfro_all', 'relfro_avg', 'relkl']
ROW_LABELS   = ['all rel. Frobenius', 'avg rel. Frobenius', 'rel. KL']

for est, est_name in [('mc', 'MC posterior mean'), ('pi', 'plug-in')]:
    plot_sensitivity_final(
        sensitivity_results,
        metrics=[f'{est}_{m}_{BLOCK}' for m in PLOT_METRICS],
        row_labels=ROW_LABELS,
        title=f'{BLOCK}, {est_name} estimator (lower is better)',
        save_path=os.path.join(SENS_DIR, f'sensitivity_prob_{est}_{BLOCK}.pdf'))
