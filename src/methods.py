"""Method under study: a VAE with a GCN encoder and an inner-product edge decoder,
trained on partially observed (A, X), plus the two estimators of P that are compared.

Estimators of P_ij = sigmoid(alpha + z_i^T z_j):
    plug-in (pi):  P_hat_ij = sigmoid(mu_i^T mu_j + b)
    MC     (mc):   P_hat_ij = E_q[sigmoid(z_i^T z_j + b)]   (S posterior draws)

Unit 3: this is the OPTIMISED implementation. The Unit 2 version is frozen in
src/baseline/methods.py; tests/test_regression.py checks that the two agree.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import config


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess(X_obs, A_obs, id_onlyA, id_both, id_onlyX, id_A, X_dim, self_weight):
    """Fill missing features of onlyA nodes, build the normalised adjacency for the GCN.

    * onlyA nodes get X filled with the average of their 'both' neighbours' features
      (or the global feature mean if they have no such neighbour);
    * A_hat = D^{-1/2} (A + self_weight * diag(deg)) D^{-1/2};
    * mask_A_obs selects the strict upper triangle of the observed adjacency.

    OPTIMISED (Unit 3): diagonal scalings are applied by broadcasting (O(n^2)) instead of
    multiplying by dense np.diag matrices (O(n^3)); the result is mathematically identical.
    The edge-density logit is guarded against an empty observed graph.
    """
    n_pairs = A_obs.shape[0] * (A_obs.shape[0] - 1) / 2.0
    edge_prob = (A_obs.sum() / 2.0) / (n_pairs + 1e-10)
    logit_edge_prob = float(np.log(edge_prob / (1.0 - edge_prob + 1e-10) + 1e-10))

    X_mean = np.mean(X_obs, axis=0)
    X_more = np.concatenate((X_mean.reshape(1, X_dim), X_obs[:len(id_both), :]))
    A_avg = A_obs[np.ix_(id_onlyA, id_both)]
    no_link = 1 * (A_avg.sum(axis=1) == 0)
    A_avg = np.concatenate((no_link.reshape(len(id_onlyA), 1), A_avg), axis=1)
    A_avg = (1.0 / A_avg.sum(axis=1))[:, None] * A_avg            # was np.diag(.) @ A_avg
    X_fill = np.concatenate((A_avg @ X_more, X_obs))

    deg = A_obs.sum(axis=1)
    deg = np.where(deg == 0, 1, deg)
    A_tilde = A_obs + np.diag(self_weight * deg)
    d_inv_sqrt = 1.0 / np.sqrt(A_tilde.sum(axis=1))
    A_hat = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]   # was diag @ A @ diag

    cols = np.tile(np.arange(len(id_A)), (len(id_A), 1))
    mask_A_obs = cols.T < cols
    return logit_edge_prob, X_fill, A_hat, mask_A_obs


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class VAE(torch.nn.Module):
    def __init__(self, X_dim, h_dim, z_dim, logit_edge_prob):
        super().__init__()
        xi = lambda i, o: torch.randn(i, o) * (1.0 / (i / 2.0) ** 0.5)  # noqa: E731
        self.PX_W1 = torch.nn.Parameter(xi(z_dim, h_dim))
        self.PX_b1 = torch.nn.Parameter(torch.zeros(h_dim))
        self.PX_W2 = torch.nn.Parameter(xi(h_dim, X_dim))
        self.PX_b2 = torch.nn.Parameter(torch.zeros(X_dim))
        self.PA_dens = torch.nn.Parameter(torch.tensor(logit_edge_prob, dtype=torch.float32))
        self.Q_W1 = torch.nn.Parameter(xi(X_dim, h_dim))
        self.Q_b1 = torch.nn.Parameter(torch.zeros(h_dim))
        self.Q_W2_mu = torch.nn.Parameter(xi(h_dim, z_dim))
        self.Q_b2_mu = torch.nn.Parameter(torch.zeros(z_dim))
        self.Q_W2_sigma = torch.nn.Parameter(xi(h_dim, z_dim))
        self.Q_b2_sigma = torch.nn.Parameter(torch.zeros(z_dim))

    def PX(self, z):
        logits = F.relu(z @ self.PX_W1 + self.PX_b1) @ self.PX_W2 + self.PX_b2
        return torch.sigmoid(logits), logits

    def PA(self, z):
        logits = z @ z.t() + self.PA_dens
        return torch.sigmoid(logits), logits

    def Q(self, X_fill, A_hat, n_A, n_onlyX):
        """Two-layer GCN for network-observed nodes, plain MLP (same weights) for onlyX nodes."""
        X_A, X_oX = X_fill[:n_A], X_fill[n_A:n_A + n_onlyX]
        h_oX = F.relu(X_oX @ self.Q_W1 + self.Q_b1)
        mu_oX = h_oX @ self.Q_W2_mu + self.Q_b2_mu
        lv_oX = h_oX @ self.Q_W2_sigma + self.Q_b2_sigma
        h_A = F.relu((A_hat @ X_A) @ self.Q_W1 + self.Q_b1)
        mu_A = (A_hat @ h_A) @ self.Q_W2_mu + self.Q_b2_mu
        lv_A = (A_hat @ h_A) @ self.Q_W2_sigma + self.Q_b2_sigma
        return torch.cat([mu_A, mu_oX], 0), torch.cat([lv_A, lv_oX], 0)

    def Q_pre(self, AX_A, A_hat, X_oX, lv_bound=None):
        """Same encoder as Q, but takes the constant product A_hat @ X_A precomputed once.

        lv_bound: optional symmetric clamp on the log-variance (overflow guard for exp(lv));
        it is the identity on every value observed (max |lv| about 12, see make stability-check).
        """
        h_oX = F.relu(X_oX @ self.Q_W1 + self.Q_b1)
        mu_oX = h_oX @ self.Q_W2_mu + self.Q_b2_mu
        lv_oX = h_oX @ self.Q_W2_sigma + self.Q_b2_sigma
        h_A = F.relu(AX_A @ self.Q_W1 + self.Q_b1)
        Ah = A_hat @ h_A                                    # computed once, used twice
        mu_A = Ah @ self.Q_W2_mu + self.Q_b2_mu
        lv_A = Ah @ self.Q_W2_sigma + self.Q_b2_sigma
        mu, lv = torch.cat([mu_A, mu_oX], 0), torch.cat([lv_A, lv_oX], 0)
        if lv_bound is not None:
            lv = lv.clamp(-lv_bound, lv_bound)
        return mu, lv

    def sample_z(self, mu, lv):
        return mu + torch.exp(lv / 2.0) * torch.randn_like(mu)

    def l2_reg(self):
        return (self.PX_W1.pow(2).sum() + self.PX_W2.pow(2).sum() + self.Q_W1.pow(2).sum()
                + self.Q_W2_mu.pow(2).sum() + self.Q_W2_sigma.pow(2).sum()) / 2


def neg_elbo(model, lgA_obs, A_obs, mask, lgX_obs, X_obs, z_mu, z_lv, tau2, l2):
    """Negative ELBO (summed BCE for observed edges and features + KL to N(0, tau2 I)) + L2."""
    return (F.binary_cross_entropy_with_logits(lgA_obs[mask], A_obs[mask], reduction="sum")
            + F.binary_cross_entropy_with_logits(lgX_obs, X_obs, reduction="sum")
            + 0.5 * torch.sum((torch.exp(z_lv) + z_mu.pow(2)) / tau2 - 1.0 - z_lv + np.log(tau2))
            + model.l2_reg() * l2)


def fit_vae(obs, X_dim, model_cfg=config.MODEL, device="cpu", log_iters=(), on_log=None):
    """Train the VAE on one replicate's observed data (optimised training loop).

    obs      : dict from dgps.observe
    on_log   : callback(it, loss, model, z_mu, z_lv) evaluated (no grad) at iterations in log_iters
    Returns the trained model and the final (z_mu, z_lv).

    Changes relative to the Unit 2 loop (src/baseline/methods.py), none of which changes the
    objective or the random-number stream:
      * A_hat @ X_A does not depend on the parameters -> computed once, not every iteration;
      * the X decoder is evaluated only on rows with observed features (both + onlyX),
        the edge decoder only on the n_A x n_A observed block (not on all n x n pairs);
      * the observed upper-triangle targets and their indices are gathered once.
    """
    id_onlyA, id_both, id_onlyX, id_A = obs["id_onlyA"], obs["id_both"], obs["id_onlyX"], obs["id_A"]
    n_A, n_oX, n_oA = len(id_A), len(id_onlyX), len(id_onlyA)
    lv_bound = model_cfg.get("lv_bound")

    logit_ep, X_fill, A_hat, _ = preprocess(obs["X_obs"], obs["A_obs"], id_onlyA, id_both,
                                            id_onlyX, id_A, X_dim, model_cfg["self_weight"])
    t_Xo = torch.tensor(obs["X_obs"], dtype=torch.float32, device=device)
    t_Xf = torch.tensor(X_fill, dtype=torch.float32, device=device)
    t_AH = torch.tensor(A_hat, dtype=torch.float32, device=device)
    t_AX = t_AH @ t_Xf[:n_A]                         # constant: hoisted out of the loop
    t_XoX = t_Xf[n_A:n_A + n_oX]
    iu = torch.triu_indices(n_A, n_A, offset=1, device=device)
    t_Aup = torch.tensor(obs["A_obs"], dtype=torch.float32, device=device)[iu[0], iu[1]]

    model = VAE(X_dim, model_cfg["h_dim_fit"], model_cfg["z_dim_fit"], logit_ep).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=model_cfg["n_iter"],
                                                       eta_min=model_cfg["lr_min"])
    log_iters = set(int(i) for i in log_iters)
    tau2, l2 = model_cfg["tau2"], model_cfg["l2"]
    log_tau2 = float(np.log(tau2))

    for it in range(model_cfg["n_iter"]):
        model.train()
        opt.zero_grad()
        z_mu, z_lv = model.Q_pre(t_AX, t_AH, t_XoX, lv_bound)
        z_s = model.sample_z(z_mu, z_lv)
        lgX_obs = (F.relu(z_s[n_oA:] @ model.PX_W1 + model.PX_b1) @ model.PX_W2 + model.PX_b2)
        z_A = z_s[:n_A]
        lgA_up = (z_A @ z_A.t() + model.PA_dens)[iu[0], iu[1]]
        loss = (F.binary_cross_entropy_with_logits(lgA_up, t_Aup, reduction="sum")
                + F.binary_cross_entropy_with_logits(lgX_obs, t_Xo, reduction="sum")
                + 0.5 * torch.sum((torch.exp(z_lv) + z_mu.pow(2)) / tau2 - 1.0 - z_lv + log_tau2)
                + model.l2_reg() * l2)
        loss.backward()
        opt.step()
        sched.step()

        if it in log_iters and on_log is not None:
            model.eval()
            with torch.no_grad():
                zm, zlv = model.Q_pre(t_AX, t_AH, t_XoX, lv_bound)
                on_log(it, loss.item(), model, zm, zlv)

    model.eval()
    with torch.no_grad():
        z_mu, z_lv = model.Q_pre(t_AX, t_AH, t_XoX, lv_bound)
    return model, z_mu, z_lv


# ------------------------------------------------------------------
# Estimators of P (returned as log P_hat and log(1 - P_hat), float64)
# ------------------------------------------------------------------
def plugin_log_probs(z_mu, b, cols=None):
    """log sigmoid(theta_hat) and log(1 - sigmoid(theta_hat)); cols restricts the columns."""
    z = z_mu.detach().double()
    zc = z if cols is None else z[cols]
    theta_hat = z @ zc.t() + b.detach().double()
    return F.logsigmoid(theta_hat), F.logsigmoid(-theta_hat)


def mc_log_probs(z_mu, z_lv, b, S, cols=None):
    """log E_q[sigmoid(theta)] and log E_q[1 - sigmoid(theta)] via a running logsumexp.

    Draws are identical to the baseline (one full n x d draw per sample, same RNG order);
    cols restricts the n x n logits to the n x |cols| columns that are actually evaluated.
    """
    # detach: if called with grad enabled, the Parameter b would make autograd keep all S
    # n x n float64 intermediates alive (~5 GB at n = 1500); the estimator never needs gradients
    mu, lv, b = z_mu.detach().double(), z_lv.detach().double(), b.detach().double()
    sd = torch.exp(lv / 2.0)
    acc1 = acc0 = None
    for _ in range(S):
        z = mu + sd * torch.randn_like(mu)
        th = z @ (z if cols is None else z[cols]).t() + b
        l1, l0 = F.logsigmoid(th), F.logsigmoid(-th)
        acc1 = l1 if acc1 is None else torch.logaddexp(acc1, l1)
        acc0 = l0 if acc0 is None else torch.logaddexp(acc0, l0)
    return acc1 - math.log(S), acc0 - math.log(S)
