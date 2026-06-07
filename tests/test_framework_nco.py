"""Unit tests for the Nested Clustered Optimization allocator (audit P3-2)
(`titan/research/framework/allocator_nco.py`).

Tests cover:
    1. Marčenko-Pastur denoise: PSD + unit diagonal, factor count, improved
       conditioning, q validation.
    2. compute_nco_weights: single-strategy, simplex (sum 1, long-only),
       the diversification property (a correlated pair splits one cluster's
       budget so a standalone uncorrelated stream out-weights each), cluster
       discovery, the max-weight cap, and both objective/denoise paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.research.framework.allocator_nco import (
    compute_nco_weights,
    marcenko_pastur_denoise,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _panel(corr: np.ndarray, names: list[str], *, n: int = 500, vol: float = 0.01, seed: int = 0):
    """Return a dict name -> return Series drawn from N(0, corr*vol²)."""
    rng = np.random.default_rng(seed)
    cov = np.asarray(corr, dtype=float) * (vol * vol)
    x = rng.multivariate_normal(np.zeros(len(names)), cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return {nm: pd.Series(x[:, i], index=idx) for i, nm in enumerate(names)}


def _block_corr(n_blocks: int, per_block: int, intra: float) -> np.ndarray:
    n = n_blocks * per_block
    c = np.zeros((n, n))
    for b in range(n_blocks):
        lo, hi = b * per_block, (b + 1) * per_block
        c[lo:hi, lo:hi] = intra
    np.fill_diagonal(c, 1.0)
    return c


# ── Marčenko-Pastur denoise ───────────────────────────────────────────────


def test_denoise_q_validation():
    with pytest.raises(ValueError, match="q"):
        marcenko_pastur_denoise(np.eye(4), q=0.0)


def test_denoise_psd_unit_diagonal_and_factor_count():
    names = [f"s{i}" for i in range(8)]
    panel = _panel(_block_corr(2, 4, intra=0.7), names, n=300, seed=1)
    df = pd.DataFrame(panel)
    corr = df.corr().to_numpy()
    q = len(df) / len(names)  # 37.5
    denoised, n_factors = marcenko_pastur_denoise(corr, q)
    # Unit diagonal + symmetric + PSD.
    assert np.allclose(np.diag(denoised), 1.0)
    assert np.allclose(denoised, denoised.T)
    assert np.linalg.eigvalsh(denoised).min() > -1e-8
    # At least one signal factor, fewer than N (some eigenvalues were noise).
    assert 1 <= n_factors < 8


def test_denoise_improves_conditioning():
    names = [f"s{i}" for i in range(8)]
    panel = _panel(_block_corr(2, 4, intra=0.7), names, n=300, seed=2)
    corr = pd.DataFrame(panel).corr().to_numpy()
    denoised, n_factors = marcenko_pastur_denoise(corr, q=300 / 8)
    if n_factors < 8:
        cond_before = np.linalg.cond(corr)
        cond_after = np.linalg.cond(denoised)
        # Averaging the noise eigenvalues lifts the smallest eigenvalue ->
        # the matrix is better conditioned.
        assert cond_after <= cond_before + 1e-6
        assert np.linalg.eigvalsh(denoised).min() >= np.linalg.eigvalsh(corr).min() - 1e-9


def test_denoise_idempotent_on_constant_residual():
    # A single-factor matrix (uniform off-diagonal) already has a flat noise
    # band (all n-1 small eigenvalues equal), so averaging them is a no-op:
    # denoising leaves the matrix unchanged and keeps the one signal factor.
    n = 5
    corr = np.full((n, n), 0.6)
    np.fill_diagonal(corr, 1.0)
    denoised, n_factors = marcenko_pastur_denoise(corr, q=1000.0)
    assert n_factors == 1  # one dominant eigenvalue (1 + (n-1)*0.6) above the bulk
    assert np.allclose(denoised, corr)


# ── compute_nco_weights ────────────────────────────────────────────────────


def test_nco_single_strategy():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    res = compute_nco_weights({"only": pd.Series(np.linspace(0.01, 0.02, 100), index=idx)})
    assert res.weights == {"only": 1.0}
    assert res.n_clusters == 1


def test_nco_weights_form_simplex():
    names = [f"s{i}" for i in range(6)]
    panel = _panel(_block_corr(2, 3, intra=0.8), names, n=400, seed=3)
    res = compute_nco_weights(panel)
    assert set(res.weights) == set(names)
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert all(w >= -1e-12 for w in res.weights.values())  # long-only


def test_nco_diversification_downweights_correlated_pair():
    # A,B nearly identical; C independent. NCO clusters {A,B} and {C}, so the
    # correlated pair splits one cluster's budget -> C out-weighs each of A,B.
    corr = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
    panel = _panel(corr, ["A", "B", "C"], n=600, seed=4)
    res = compute_nco_weights(panel)
    assert res.weights["C"] > res.weights["A"]
    assert res.weights["C"] > res.weights["B"]


def test_nco_discovers_two_clusters():
    names = ["a0", "a1", "a2", "b0", "b1", "b2"]
    panel = _panel(_block_corr(2, 3, intra=0.9), names, n=500, seed=5)
    res = compute_nco_weights(panel)
    assert res.n_clusters >= 2
    # The two members of block A land in the same cluster; b0 does not.
    cluster_of = {m: i for i, cl in enumerate(res.clusters) for m in cl}
    assert cluster_of["a0"] == cluster_of["a1"]
    assert cluster_of["a0"] != cluster_of["b0"]


def test_nco_respects_max_weight_cap():
    names = [f"s{i}" for i in range(6)]
    panel = _panel(_block_corr(2, 3, intra=0.8), names, n=400, seed=6)
    res = compute_nco_weights(panel, max_weight=0.20)
    assert all(w <= 0.20 + 1e-9 for w in res.weights.values())
    assert sum(res.weights.values()) == pytest.approx(1.0)


def test_nco_inverse_variance_objective_runs():
    names = [f"s{i}" for i in range(6)]
    panel = _panel(_block_corr(2, 3, intra=0.6), names, n=400, seed=7)
    res = compute_nco_weights(panel, objective="inverse_variance")
    assert res.objective == "inverse_variance"
    assert sum(res.weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="objective"):
        compute_nco_weights(panel, objective="sortino")


def test_nco_crisis_rho_and_no_denoise_paths():
    names = [f"s{i}" for i in range(5)]
    panel = _panel(_block_corr(1, 5, intra=0.3), names, n=400, seed=8)
    res_crisis = compute_nco_weights(panel, crisis_rho=0.8)
    res_raw = compute_nco_weights(panel, denoise=False)
    assert sum(res_crisis.weights.values()) == pytest.approx(1.0)
    assert sum(res_raw.weights.values()) == pytest.approx(1.0)
    # With denoise off, all eigenvalues are "kept" by construction.
    assert res_raw.n_factors_kept == 5
