"""Performance measures comparing an estimate P_hat with the true P = sigmoid(Theta).

All computations are in float64 and, where probabilities can underflow, in log space.

    relfro_all  = ||P_hat - P||_F / ||P||_F           (1 = no better than predicting P_hat = 0)
    relfro_avg  = mean_i ||P_hat_i - P_i|| / ||P_i||   (rows with expected degree >= min_exp_deg)
    relfro_cent = ||P_hat - P||_F / ||P - pbar||_F     (1 = no better than the constant pbar)
    rmse        = ||P_hat - P||_F / sqrt(#pairs)
    kl          = mean_ij KL(Bern(P_ij) || Bern(P_hat_ij))
    relkl       = kl / KL(P || pbar)                   (1 = no better than knowing the density)
    relfro_theta (plug-in only) = ||Theta_hat - Theta||_F / ||Theta||_F
"""
import torch
import torch.nn.functional as F

from . import config
from .methods import mc_log_probs, plugin_log_probs


def bern_kl(logp, log1mp, logq, log1mq):
    """Elementwise KL(Bern(p) || Bern(q)) from log-probabilities."""
    p = torch.exp(logp)
    return p * (logp - logq) + (1.0 - p) * (log1mp - log1mq)


def prob_metrics(theta, logq, log1mq, mask=None, min_exp_deg=config.EVAL["min_exp_deg"]):
    """Metrics for one block. theta: true logits (r, c) float64; mask excludes entries (e.g. diagonal)."""
    if mask is None:
        mask = torch.ones_like(theta, dtype=torch.bool)
    w = mask.to(theta.dtype)
    m = w.sum()

    logp, log1mp = F.logsigmoid(theta), F.logsigmoid(-theta)
    p, q = torch.exp(logp), torch.exp(logq)
    sq = (q - p).pow(2) * w

    relfro_all = torch.sqrt(sq.sum()) / torch.sqrt((p.pow(2) * w).sum())
    rmse = torch.sqrt(sq.sum() / m)
    pbar_p = (p * w).sum() / m
    relfro_cent = torch.sqrt(sq.sum()) / torch.sqrt(((p - pbar_p).pow(2) * w).sum())

    row_num = torch.sqrt(sq.sum(1))
    row_den = torch.sqrt((p.pow(2) * w).sum(1))
    valid = (p * w).sum(1) >= min_exp_deg
    relfro_avg = (row_num[valid] / row_den[valid]).mean() if valid.any() else torch.tensor(float("nan"))

    kl = (bern_kl(logp, log1mp, logq, log1mq) * w).sum() / m
    pbar = pbar_p.clamp(1e-300, 1 - 1e-16)
    kl0 = (bern_kl(logp, log1mp, torch.log(pbar), torch.log1p(-pbar)) * w).sum() / m

    return {"relfro_all": relfro_all.item(), "relfro_avg": relfro_avg.item(),
            "relfro_cent": relfro_cent.item(), "rmse": rmse.item(), "kl": kl.item(),
            "relkl": (kl / kl0).item(), "frac_rows": valid.double().mean().item()}


def prepare_truth(theta_true, blocks, min_exp_deg=config.EVAL["min_exp_deg"]):
    """Precompute every quantity that depends only on the truth, once per replicate.

    In the baseline these (log p, ||P||, pbar, KL to the constant predictor, row norms ...)
    were recomputed at each of the 25 logged iterations for both estimators.
    theta_true may be n x n_A (only the columns that any block uses).
    """
    out = {}
    for bname, (rows, cols, offdiag) in blocks.items():
        th = theta_true[rows][:, cols]
        mask = (~torch.eye(len(rows), dtype=torch.bool, device=th.device) if offdiag
                else torch.ones_like(th, dtype=torch.bool))
        w = mask.to(th.dtype)
        m = w.sum()
        logp, log1mp = F.logsigmoid(th), F.logsigmoid(-th)
        p = torch.exp(logp)
        pbar_p = (p * w).sum() / m
        pbar = pbar_p.clamp(1e-300, 1 - 1e-16)
        kl0 = (bern_kl(logp, log1mp, torch.log(pbar), torch.log1p(-pbar)) * w).sum() / m
        valid = (p * w).sum(1) >= min_exp_deg
        out[bname] = dict(rows=rows, cols=cols, offdiag=offdiag, th=th, w=w, m=m, logp=logp,
                          log1mp=log1mp, p=p, kl0=kl0, valid=valid,
                          den_all=torch.sqrt((p.pow(2) * w).sum()),
                          den_cent=torch.sqrt(((p - pbar_p).pow(2) * w).sum()),
                          row_den=torch.sqrt((p.pow(2) * w).sum(1)),
                          den_theta=torch.sqrt((th.pow(2) * w).sum()))
    return out


def _metrics_pre(T, logq, log1mq):
    """prob_metrics() with the truth-only terms taken from prepare_truth (same arithmetic)."""
    w, p = T["w"], T["p"]
    q = torch.exp(logq)
    sq = (q - p).pow(2) * w
    sqs = sq.sum()
    valid = T["valid"]
    row_num = torch.sqrt(sq.sum(1))
    relfro_avg = ((row_num[valid] / T["row_den"][valid]).mean() if valid.any()
                  else torch.tensor(float("nan")))
    kl = (bern_kl(T["logp"], T["log1mp"], logq, log1mq) * w).sum() / T["m"]
    return {"relfro_all": (torch.sqrt(sqs) / T["den_all"]).item(),
            "relfro_avg": relfro_avg.item(),
            "relfro_cent": (torch.sqrt(sqs) / T["den_cent"]).item(),
            "rmse": torch.sqrt(sqs / T["m"]).item(), "kl": kl.item(),
            "relkl": (kl / T["kl0"]).item(), "frac_rows": valid.double().mean().item()}


def evaluate_fast(model, z_mu, z_lv, truth, cols_needed, do_mc,
                  mc_samples=config.EVAL["mc_samples"]):
    """Optimised evaluate(): estimators computed only on the columns any block uses,
    truth-side terms reused from prepare_truth. Returns the same keys and values."""
    b = model.PA_dens
    est = {"pi": plugin_log_probs(z_mu, b, cols_needed)}
    if do_mc:
        est["mc"] = mc_log_probs(z_mu, z_lv, b, mc_samples, cols_needed)
    zc = z_mu.double()
    theta_hat = zc @ zc[cols_needed].t() + b.double()
    out = {}
    for bname, T in truth.items():
        rows, cols = T["rows"], T["cols"]
        for ename in ["pi", "mc"]:
            if ename in est:
                lq, l1q = est[ename]
                res = _metrics_pre(T, lq[rows][:, cols], l1q[rows][:, cols])
            else:
                res = {k: float("nan") for k in config.METRIC_KEYS}
            for k, v in res.items():
                out[f"{ename}_{k}_{bname}"] = v
        th_hat = theta_hat[rows][:, cols]
        out[f"pi_relfro_theta_{bname}"] = (torch.sqrt(((th_hat - T["th"]).pow(2) * T["w"]).sum())
                                           / T["den_theta"]).item()
    return out


def make_blocks(obs, device="cpu"):
    """Evaluation blocks (rows, cols, exclude_diagonal) in training order."""
    t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)  # noqa: E731
    return {
        "onlyX_A": (t(obs["id_onlyX"]), t(obs["id_A"]), False),       # main: cold start
        "onlyX_onlyA": (t(obs["id_onlyX"]), t(obs["id_onlyA"]), False),
        "onlyX_both": (t(obs["id_onlyX"]), t(obs["id_both"]), False),
        "A_A": (t(obs["id_A"]), t(obs["id_A"]), True),
    }


def evaluate(model, z_mu, z_lv, theta_true, blocks, do_mc, mc_samples=config.EVAL["mc_samples"],
             min_exp_deg=config.EVAL["min_exp_deg"]):
    """Flat dict {'{est}_{metric}_{block}': value} for both estimators and all blocks."""
    out = {}
    b = model.PA_dens
    est = {"pi": plugin_log_probs(z_mu, b)}
    if do_mc:
        est["mc"] = mc_log_probs(z_mu, z_lv, b, mc_samples)
    theta_hat = z_mu.double() @ z_mu.double().t() + b.double()

    for bname, (rows, cols, offdiag) in blocks.items():
        th = theta_true[rows][:, cols]
        mask = ~torch.eye(len(rows), dtype=torch.bool, device=th.device) if offdiag else None
        for ename in ["pi", "mc"]:
            if ename in est:
                lq, l1q = est[ename]
                res = prob_metrics(th, lq[rows][:, cols], l1q[rows][:, cols], mask, min_exp_deg)
            else:
                res = {k: float("nan") for k in config.METRIC_KEYS}
            for k, v in res.items():
                out[f"{ename}_{k}_{bname}"] = v
        th_hat = theta_hat[rows][:, cols]
        w = 1.0 if mask is None else mask.double()
        out[f"pi_relfro_theta_{bname}"] = (torch.sqrt(((th_hat - th).pow(2) * w).sum())
                                           / torch.sqrt((th.pow(2) * w).sum())).item()
    return out
