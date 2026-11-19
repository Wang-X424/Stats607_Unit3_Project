"""Data-generating process: latent-position network + nonlinear binary features.

    Z_i       ~ N(0, z_var * I_d)
    A_ij | Z  ~ Bernoulli(sigmoid(alpha + Z_i^T Z_j)),  i < j, symmetric, zero diagonal
    X_ik | Z  ~ Bernoulli(sigmoid([W2 ReLU(W1 Z_i + b1) + b2]_k))

The functions are numerically identical to legacy/simulation_sensitivity_fix1_prob.py.
"""
import numpy as np

from .. import config


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def true_logits(Z, alpha):
    """Theta = alpha + Z Z^T (n x n). The estimand is P = sigmoid(Theta)."""
    return alpha + Z @ Z.T


def generate_data(n, X_dim, alpha, seed, dgp=config.DGP, verbose=False):
    """Return (Z_true, A_true, X_true) for one replicate."""
    d, z_scale, h = dgp["z_dim_true"], dgp["z_scale"], dgp["x_hidden"]

    rng = np.random.RandomState(seed)
    Z = rng.randn(n, d).astype(np.float32) * z_scale

    prob_A = sigmoid(true_logits(Z, alpha))
    U = rng.rand(n, n)
    A = np.triu((U < prob_A).astype(np.int32), k=1)
    A = A + A.T

    rng2 = np.random.RandomState(seed + dgp["seed_offset_x"])
    W1 = rng2.randn(d, h).astype(np.float32)
    b1 = rng2.randn(h).astype(np.float32) * 0.1
    W2 = rng2.randn(h, X_dim).astype(np.float32)
    b2 = rng2.randn(X_dim).astype(np.float32) * 0.1
    H = np.maximum(0, Z @ W1 + b1)
    prob_X = sigmoid(H @ W2 + b2)
    X = (rng2.rand(n, X_dim) < prob_X).astype(np.int32)

    if verbose:
        print(f"  Generated: n={n}, X_dim={X_dim}, edge density={A.sum() / (n * (n - 1)):.4f}")
    return Z, A, X


def split_nodes(n, seed, split=config.SPLIT):
    """Randomly assign nodes to onlyA / both / onlyX (percentages of n)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_oA = int(np.round(n * split["onlyA"] / 100))
    n_b = int(np.round(n * split["both"] / 100))
    n_oX = min(int(np.round(n * split["onlyX"] / 100)), n - n_oA - n_b)
    return perm[:n_oA], perm[n_oA:n_oA + n_b], perm[n_oA + n_b:n_oA + n_b + n_oX]


def observe(Z, A, X, idx_oA, idx_b, idx_oX):
    """Reorder nodes as [onlyA, both, onlyX] and mask what is unobserved.

    Returns a dict with the observed adjacency (onlyA+both rows/cols), the observed
    features (both+onlyX rows), the index sets in training order, and the latent
    positions in training order (used only for evaluation).
    """
    order = np.concatenate([idx_oA, idx_b, idx_oX])
    n = len(order)
    n_oA, n_b = len(idx_oA), len(idx_b)
    id_onlyA = np.arange(n_oA)
    id_both = np.arange(n_oA, n_oA + n_b)
    id_onlyX = np.arange(n_oA + n_b, n)
    id_A = np.concatenate([id_onlyA, id_both])
    id_X = np.concatenate([id_both, id_onlyX])

    A_all = A[np.ix_(order, order)]
    return {
        "A_obs": A_all[np.ix_(id_A, id_A)],
        "X_obs": X[order][id_X],
        "Z_ord": Z[order],
        "id_onlyA": id_onlyA, "id_both": id_both, "id_onlyX": id_onlyX,
        "id_A": id_A, "id_X": id_X,
    }
