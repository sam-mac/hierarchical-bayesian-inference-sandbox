"""Tests for hierarchical linear–Gaussian state-space helpers."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from bayes_tools.helpers.hierarchical_lgssm import (
    FFBSResult,
    ffbs_sample_states,
    kalman_filter_loglik,
    kalman_logp_pt,
)


def _build_simple_system(T: int = 4):
    rng = np.random.default_rng(42)
    A = np.array([[0.9]])
    B = np.array([[0.1]])
    H = np.array([[1.0]])
    D = np.array([[0.0]])
    Q = np.array([[0.05]])
    R = np.array([[0.1]])
    m0 = np.array([0.0])
    P0 = np.array([[1.0]])

    w = rng.normal(0.0, 1.0, size=(T, 1))
    x = np.zeros((T, 1))
    y = np.zeros((T, 1))
    x_prev = rng.normal(size=1)
    for t in range(T):
        x_t = A @ x_prev + B @ w[t] + rng.multivariate_normal(mean=np.zeros(1), cov=Q)
        y[t] = H @ x_t + D @ w[t] + rng.multivariate_normal(mean=np.zeros(1), cov=R)
        x[t] = x_t
        x_prev = x_t

    return y, w, A, B, H, D, Q, R, m0, P0


def test_kalman_loglik_matches_pytensor():
    y, w, A, B, H, D, Q, R, m0, P0 = _build_simple_system()
    loglik_np = kalman_filter_loglik(y, w, A, B, H, D, Q, R, m0, P0)

    mask = (~np.isnan(y)).astype(float)
    ll_symbolic = kalman_logp_pt(
        y=pt.constant(np.nan_to_num(y)),
        w=pt.constant(w),
        mask=pt.constant(mask),
        A=pt.constant(A),
        B=pt.constant(B),
        H=pt.constant(H),
        D=pt.constant(D),
        Q=pt.constant(Q),
        R=pt.constant(R),
        m0=pt.constant(m0),
        P0=pt.constant(P0),
    )
    ll_eval = pytensor.function([], ll_symbolic)()
    assert np.isclose(loglik_np, float(ll_eval), atol=1e-6)


def test_ffbs_returns_expected_shapes():
    y, w, A, B, H, D, Q, R, m0, P0 = _build_simple_system()
    result = ffbs_sample_states(y, w, A, B, H, D, Q, R, m0, P0)
    assert isinstance(result, FFBSResult)
    assert result.states.shape == y.shape
    assert np.isfinite(result.loglik)

