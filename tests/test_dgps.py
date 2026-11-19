"""DGP verification: the generated data have the properties the design specifies."""
import numpy as np
import pytest

from src import config
from src.dgps import generate_data, observe, sigmoid, split_nodes, true_logits


def test_adjacency_is_simple_undirected_graph():
    Z, A, X = generate_data(200, 50, -2, seed=1)
    assert A.shape == (200, 200) and X.shape == (200, 50) and Z.shape == (200, 8)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    assert set(np.unique(A)) <= {0, 1} and set(np.unique(X)) <= {0, 1}


@pytest.mark.parametrize("alpha", [-6, -2, 0])
def test_edge_density_matches_true_probabilities(alpha):
    """Observed density is within 4 binomial SEs of the mean of P over i<j."""
    n = 400
    Z, A, _ = generate_data(n, 20, alpha, seed=7)
    iu = np.triu_indices(n, 1)
    P = sigmoid(true_logits(Z.astype(np.float64), alpha))[iu]
    se = np.sqrt((P * (1 - P)).sum()) / len(P)
    assert abs(A[iu].mean() - P.mean()) < 4 * se


def test_latent_variance_matches_config():
    Z, _, _ = generate_data(4000, 5, -6, seed=3)
    assert abs(Z.var() - config.DGP["z_var"]) < 0.05


def test_same_seed_same_data_and_different_seed_different_data():
    a = generate_data(100, 20, -4, seed=11)
    b = generate_data(100, 20, -4, seed=11)
    c = generate_data(100, 20, -4, seed=12)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    assert not np.array_equal(a[1], c[1])


@pytest.mark.parametrize("n", [100, 400, 700])
def test_split_is_partition_with_requested_proportions(n):
    oA, b, oX = split_nodes(n, seed=5)
    allidx = np.concatenate([oA, b, oX])
    assert len(np.unique(allidx)) == len(allidx) == n
    assert len(oA) == round(0.3 * n) and len(b) == round(0.4 * n)


def test_observe_masks_unobserved_parts():
    Z, A, X = generate_data(50, 10, -2, seed=0)
    oA, b, oX = split_nodes(50, seed=1)
    obs = observe(Z, A, X, oA, b, oX)
    assert obs["A_obs"].shape == (len(oA) + len(b),) * 2
    assert obs["X_obs"].shape == (len(b) + len(oX), 10)
    order = np.concatenate([oA, b, oX])
    np.testing.assert_array_equal(obs["Z_ord"], Z[order])
