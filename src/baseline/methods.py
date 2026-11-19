"""Method under study: a VAE with a GCN encoder and an inner-product edge decoder,
trained on partially observed (A, X), plus the two estimators of P that are compared.

Estimators of P_ij = sigmoid(alpha + z_i^T z_j):
    plug-in (pi):  P_hat_ij = sigmoid(mu_i^T mu_j + b)
    MC     (mc):   P_hat_ij = E_q[sigmoid(z_i^T z_j + b)]   (S posterior draws)

Model, preprocessing and training are identical to
legacy/simulation_sensitivity_fix1_prob.py, except that the torch RNG is now seeded.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import config


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess(X_obs, A_obs, id_onlyA, id_both, id_onlyX, id_A, X_dim, self_weight):
    """Fill missing features of onlyA nodes, build the normalised adjacency for the GCN.

    * onlyA nodes get X filled with the average of their 'both' neighbours' features
      (or the global feature mean if they have no such neighbour);
    * A_hat = D^{-1/2} (A + self_weight * diag(deg)) D^{-1/2};
    * mask_A_obs selects the strict upper triangle of the observed adjacency.
    """
    edge_prob = (A_obs.sum() / 2.0) / (A_obs.shape[0] * (A_obs.shape[0] - 1) / 2.0 + 1e-10)
    logit_edge_prob = float(np.log(edge_prob / (1.0 - edge_prob + 1e-10) + 1e-10))

    X_mean = np.mean(X_obs, axis=0)
    X_more = np.concatenate((X_mean.reshape(1, X_dim), X_obs[:len(id_both), :]))
    A_avg = A_obs[np.ix_(id_onlyA, id_both)]
    no_link = 1 * (A_avg.sum(axis=1) == 0)
    A_avg = np.concatenate((no_link.reshape(len(id_onlyA), 1), A_avg), axis=1)
    A_avg = np.diag(1.0 / A_avg.sum(axis=1)) @ A_avg
    X_fill = np.concatenate((A_avg @ X_more, X_obs))

    deg = A_obs.sum(axis=1)
    deg = np.where(deg == 0, 1, deg)
    A_tilde = A_obs + np.diag(self_weight * deg)
    d_inv_sqrt = 1.0 / np.sqrt(A_tilde.sum(axis=1))
    A_hat = np.diag(d_inv_sqrt) @ A_tilde @ np.diag(d_inv_sqrt)

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
    """Train the VAE on one replicate's observed data.

    obs      : dict from dgps.observe
    on_log   : callback(it, loss, model, z_mu, z_lv) evaluated (no grad) at iterations in log_iters
    Returns the trained model and the final (z_mu, z_lv).
    """
    id_onlyA, id_both, id_onlyX, id_A = obs["id_onlyA"], obs["id_both"], obs["id_onlyX"], obs["id_A"]
    n_A, n_oX, n_oA = len(id_A), len(id_onlyX), len(id_onlyA)

    logit_ep, X_fill, A_hat, mask = preprocess(obs["X_obs"], obs["A_obs"], id_onlyA, id_both,
                                               id_onlyX, id_A, X_dim, model_cfg["self_weight"])
    t_Xo = torch.tensor(obs["X_obs"], dtype=torch.float32, device=device)
    t_Ao = torch.tensor(obs["A_obs"], dtype=torch.float32, device=device)
    t_Xf = torch.tensor(X_fill, dtype=torch.float32, device=device)
    t_AH = torch.tensor(A_hat, dtype=torch.float32, device=device)
    t_mask = torch.tensor(mask, dtype=torch.bool, device=device)

    model = VAE(X_dim, model_cfg["h_dim_fit"], model_cfg["z_dim_fit"], logit_ep).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=model_cfg["n_iter"],
                                                       eta_min=model_cfg["lr_min"])
    log_iters = set(int(i) for i in log_iters)

    for it in range(model_cfg["n_iter"]):
        model.train()
        opt.zero_grad()
        z_mu, z_lv = model.Q(t_Xf, t_AH, n_A, n_oX)
        z_s = model.sample_z(z_mu, z_lv)
        _, lgX = model.PX(z_s)
        _, lgA = model.PA(z_s)
        lgX_obs = lgX[n_oA:n_oA + len(obs["id_X"])]
        lgA_obs = lgA[:n_A, :n_A]
        loss = neg_elbo(model, lgA_obs, t_Ao, t_mask, lgX_obs, t_Xo, z_mu, z_lv,
                        model_cfg["tau2"], model_cfg["l2"])
        loss.backward()
        opt.step()
        sched.step()

        if it in log_iters and on_log is not None:
            model.eval()
            with torch.no_grad():
                zm, zlv = model.Q(t_Xf, t_AH, n_A, n_oX)
                on_log(it, loss.item(), model, zm, zlv)

    model.eval()
    with torch.no_grad():
        z_mu, z_lv = model.Q(t_Xf, t_AH, n_A, n_oX)
    return model, z_mu, z_lv


# ------------------------------------------------------------------
# Estimators of P (returned as log P_hat and log(1 - P_hat), float64)
# ------------------------------------------------------------------
def plugin_log_probs(z_mu, b):
    theta_hat = z_mu.double() @ z_mu.double().t() + b.double()
    return F.logsigmoid(theta_hat), F.logsigmoid(-theta_hat)


def mc_log_probs(z_mu, z_lv, b, S):
    """log E_q[sigmoid(theta)] and log E_q[1 - sigmoid(theta)] via a running logsumexp."""
    mu, lv, b = z_mu.double(), z_lv.double(), b.double()
    acc1 = acc0 = None
    for _ in range(S):
        z = mu + torch.exp(lv / 2.0) * torch.randn_like(mu)
        th = z @ z.t() + b
        l1, l0 = F.logsigmoid(th), F.logsigmoid(-th)
        acc1 = l1 if acc1 is None else torch.logaddexp(acc1, l1)
        acc0 = l0 if acc0 is None else torch.logaddexp(acc0, l0)
    return acc1 - math.log(S), acc0 - math.log(S)
