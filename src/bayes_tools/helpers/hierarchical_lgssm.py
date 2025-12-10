"""Hierarchical linear–Gaussian state-space model utilities.

This module provides a complete toolkit for building and fitting grouped
linear–Gaussian state-space models (LGSSMs) with hierarchical priors in PyMC.
The implementation follows a Kalman filter marginal likelihood approach where
the latent states are *not* sampled inside the PyMC model. Instead, a
PyTensor-based Kalman log-likelihood is used inside a ``pm.Potential`` to
evaluate ``p(y | \theta)`` exactly for a given set of parameters. Latent
states can then be drawn post hoc with a Forward-Filtering Backward-Sampling
(FFBS) pass.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.ifelse import ifelse
from pytensor.scan import scan


KalmanCache = Dict[str, NDArray[np.float64]]


@dataclass
class FFBSResult:
    """Result of a Forward-Filtering Backward-Sampling pass.

    Attributes
    ----------
    states:
        Drawn latent trajectory with shape ``(T, K)``.
    loglik:
        Log-likelihood of the observations under the provided parameters,
        computed during the filtering pass.
    """

    states: NDArray[np.float64]
    loglik: float


def kalman_filter_loglik(
    y: NDArray[np.floating],
    w: NDArray[np.floating],
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    H: NDArray[np.floating],
    D: NDArray[np.floating],
    Q: NDArray[np.floating],
    R: NDArray[np.floating],
    m0: NDArray[np.floating],
    P0: NDArray[np.floating],
    *,
    jitter: float = 1e-9,
    return_cache: bool = False,
) -> float | tuple[float, KalmanCache]:
    """Kalman filter log-likelihood for a single group.

    Parameters
    ----------
    y:
        Observations of shape ``(T, P)`` with ``np.nan`` entries marking
        missing data.
    w:
        Exogenous inputs of shape ``(T, M)``; set to zeros if not used.
    A, B, H, D, Q, R:
        State transition, input, observation, input-to-observation, process
        noise covariance and observation noise covariance matrices.
    m0, P0:
        Initial state mean and covariance.
    jitter:
        Diagonal jitter to keep covariance matrices positive definite.
    return_cache:
        Whether to also return a cache of filtering moments for FFBS.

    Returns
    -------
    float | tuple[float, KalmanCache]
        The log-likelihood, and optionally a cache for FFBS sampling.
    """

    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    T, P = y.shape
    K = A.shape[0]

    mask = ~np.isnan(y)
    y_filled = np.nan_to_num(y, nan=0.0)

    identity_p = np.eye(P)

    m_pred = np.zeros((T, K))
    P_pred = np.zeros((T, K, K))
    m_filt = np.zeros((T, K))
    P_filt = np.zeros((T, K, K))

    loglik = 0.0
    mean = m0.astype(float).copy()
    cov = P0.astype(float).copy()

    for t in range(T):
        if t == 0:
            mean_pred, cov_pred = mean, cov
        else:
            mean_pred = A @ mean + B @ w[t]
            cov_pred = A @ cov @ A.T + Q

        m_pred[t], P_pred[t] = mean_pred, cov_pred

        obs_mask = mask[t].astype(float)
        H_t = H * obs_mask[:, None]
        D_t = D * obs_mask[:, None]
        y_t = y_filled[t] * obs_mask
        R_t = (R * obs_mask[:, None] * obs_mask[None, :]) + np.diag(1.0 - obs_mask)

        innov = y_t - (H_t @ mean_pred + D_t @ w[t])
        innov_cov = H_t @ cov_pred @ H_t.T + R_t

        chol = np.linalg.cholesky(innov_cov + jitter * identity_p)
        solve = np.linalg.solve
        innov_cov_inv_innov = solve(chol.T, solve(chol, innov))
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        n_obs = int(obs_mask.sum())

        loglik += -0.5 * (n_obs * np.log(2.0 * np.pi) + logdet + innov.T @ innov_cov_inv_innov)

        ph_t = cov_pred @ H_t.T
        kalman_gain = solve(chol.T, solve(chol, ph_t.T)).T

        mean = mean_pred + kalman_gain @ innov
        cov = cov_pred - kalman_gain @ (H_t @ cov_pred)
        cov = 0.5 * (cov + cov.T)

        m_filt[t], P_filt[t] = mean, cov

    if not return_cache:
        return loglik

    cache: KalmanCache = {
        "m_pred": m_pred,
        "P_pred": P_pred,
        "m_filt": m_filt,
        "P_filt": P_filt,
        "mask": mask.astype(float),
    }
    return loglik, cache


def ffbs_sample_states(
    y: NDArray[np.floating],
    w: NDArray[np.floating],
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    H: NDArray[np.floating],
    D: NDArray[np.floating],
    Q: NDArray[np.floating],
    R: NDArray[np.floating],
    m0: NDArray[np.floating],
    P0: NDArray[np.floating],
    *,
    jitter: float = 1e-9,
) -> FFBSResult:
    """Run Forward-Filtering Backward-Sampling for a single group."""

    loglik, cache = kalman_filter_loglik(
        y,
        w,
        A,
        B,
        H,
        D,
        Q,
        R,
        m0,
        P0,
        jitter=jitter,
        return_cache=True,
    )

    m_pred = cache["m_pred"]
    P_pred = cache["P_pred"]
    m_filt = cache["m_filt"]
    P_filt = cache["P_filt"]

    T, K = y.shape[0], A.shape[0]
    states = np.zeros((T, K))

    chol_final = np.linalg.cholesky(P_filt[T - 1] + jitter * np.eye(K))
    states[T - 1] = m_filt[T - 1] + chol_final @ np.random.randn(K)

    for t in range(T - 2, -1, -1):
        next_cov = P_pred[t + 1] + jitter * np.eye(K)
        gain = np.linalg.solve(next_cov, (A @ P_filt[t]).T).T

        mean = m_filt[t] + gain @ (states[t + 1] - m_pred[t + 1])
        cov = P_filt[t] - gain @ P_pred[t + 1] @ gain.T
        cov = 0.5 * (cov + cov.T)

        chol = np.linalg.cholesky(cov + jitter * np.eye(K))
        states[t] = mean + chol @ np.random.randn(K)

    return FFBSResult(states=states, loglik=float(loglik))


def kalman_logp_pt(
    y: pt.TensorLike,
    w: pt.TensorLike,
    mask: pt.TensorLike,
    A: pt.TensorLike,
    B: pt.TensorLike,
    H: pt.TensorLike,
    D: pt.TensorLike,
    Q: pt.TensorLike,
    R: pt.TensorLike,
    m0: pt.TensorLike,
    P0: pt.TensorLike,
    *,
    jitter: float = 1e-6,
) -> pt.TensorVariable:
    """PyTensor Kalman log-likelihood for use inside a PyMC model."""

    dtype = pytensor.config.floatX

    y = pt.as_tensor_variable(y, dtype=dtype)
    w = pt.as_tensor_variable(w, dtype=dtype)
    mask = pt.as_tensor_variable(mask, dtype=dtype)
    A = pt.as_tensor_variable(A, dtype=dtype)
    B = pt.as_tensor_variable(B, dtype=dtype)
    H = pt.as_tensor_variable(H, dtype=dtype)
    D = pt.as_tensor_variable(D, dtype=dtype)
    Q = pt.as_tensor_variable(Q, dtype=dtype)
    R = pt.as_tensor_variable(R, dtype=dtype)
    m0 = pt.as_tensor_variable(m0, dtype=dtype)
    P0 = pt.as_tensor_variable(P0, dtype=dtype)

    T = y.shape[0]
    P = y.shape[1]
    identity_p = pt.eye(P, dtype=dtype)

    def step(
        t: pt.TensorVariable,
        y_t: pt.TensorVariable,
        w_t: pt.TensorVariable,
        mask_t: pt.TensorVariable,
        mean_prev: pt.TensorVariable,
        cov_prev: pt.TensorVariable,
        ll_prev: pt.TensorVariable,
        A_t: pt.TensorVariable,
        B_t: pt.TensorVariable,
        H_t: pt.TensorVariable,
        D_t: pt.TensorVariable,
        Q_t: pt.TensorVariable,
        R_t: pt.TensorVariable,
    ) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        mean_pred = ifelse(pt.eq(t, 0), mean_prev, A_t @ mean_prev + B_t @ w_t)
        cov_pred = ifelse(pt.eq(t, 0), cov_prev, A_t @ cov_prev @ A_t.T + Q_t)

        obs_mask = mask_t
        H_masked = H_t * obs_mask[:, None]
        D_masked = D_t * obs_mask[:, None]
        y_obs = y_t * obs_mask
        R_masked = (R_t * obs_mask[:, None] * obs_mask[None, :]) + pt.diag(1.0 - obs_mask)

        innov = y_obs - (H_masked @ mean_pred + D_masked @ w_t)
        innov_cov = H_masked @ cov_pred @ H_masked.T + R_masked + jitter * identity_p

        chol = pt.linalg.cholesky(innov_cov)
        innov_cov_inv_innov = pt.linalg.solve(chol.T, pt.linalg.solve(chol, innov))
        logdet = 2.0 * pt.sum(pt.log(pt.diag(chol)))
        n_obs = pt.sum(obs_mask)

        ll_t = -0.5 * (n_obs * pt.log(2.0 * pt.pi) + logdet + innov.dot(innov_cov_inv_innov))

        ph_t = cov_pred @ H_masked.T
        kalman_gain = pt.linalg.solve(chol.T, pt.linalg.solve(chol, ph_t.T)).T

        mean_new = mean_pred + kalman_gain @ innov
        cov_new = cov_pred - kalman_gain @ (H_masked @ cov_pred)
        cov_new = 0.5 * (cov_new + cov_new.T)

        return mean_new, cov_new, ll_prev + ll_t

    (mean_seq, cov_seq, ll_seq), _ = scan(
        fn=step,
        sequences=[pt.arange(T), y, w, mask],
        outputs_info=[m0, P0, pt.as_tensor_variable(0.0, dtype=dtype)],
        non_sequences=[A, B, H, D, Q, R],
    )

    return ll_seq[-1]


def hierarchical_matrix_mvnormal(
    name: str,
    n_groups: int,
    shape: Tuple[int, int],
    *,
    eta: float = 2.0,
    sd_scale: float = 0.3,
) -> pt.TensorVariable:
    """Create a matrix-valued hierarchical MVN prior using LKJCholeskyCov."""

    dimension = int(np.prod(shape))
    mu = pm.Normal(f"mu_{name}", 0.0, 0.5, shape=dimension)
    chol, corr, _ = pm.LKJCholeskyCov(
        f"chol_{name}",
        n=dimension,
        eta=eta,
        sd_dist=pm.HalfNormal.dist(sd_scale),
        compute_corr=True,
    )
    z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=(n_groups, dimension))
    vec = mu + z @ chol.T
    mat = pm.Deterministic(name, vec.reshape((n_groups, *shape)))
    pm.Deterministic(f"corr_{name}", corr)
    return mat


def build_and_sample_model(
    y_data: NDArray[np.floating],
    w_data: NDArray[np.floating],
    K: int,
    M: int,
    *,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the hierarchical LGSSM in PyMC."""

    y_data = np.asarray(y_data, dtype=float)
    w_data = np.asarray(w_data, dtype=float)
    groups, time_steps, obs_dim = y_data.shape

    mask_data = (~np.isnan(y_data)).astype("float32")
    y_filled = np.nan_to_num(y_data, nan=0.0).astype("float32")

    with pm.Model() as model:
        y_pt = pm.Data("y", y_filled, mutable=False)
        mask_pt = pm.Data("mask", mask_data, mutable=False)
        w_pt = pm.Data("w", w_data.astype("float32"), mutable=False)

        H0 = np.zeros((obs_dim, K), dtype="float32")
        H0[:, : min(obs_dim, K)] = np.eye(obs_dim, min(obs_dim, K), dtype="float32")

        sigma_H = pm.HalfNormal("sigma_H", 0.05)
        H_dev = pm.Normal("H_dev", 0.0, sigma_H, shape=(groups, obs_dim, K))
        H_g = pm.Deterministic("H_g", H0[None, :, :] + H_dev)

        A_g = hierarchical_matrix_mvnormal("A_g", groups, (K, K), eta=2.0, sd_scale=0.2)
        B_g = pm.Normal("B_g", 0.0, 0.5, shape=(groups, K, M))
        D_g = pm.Normal("D_g", 0.0, 0.5, shape=(groups, obs_dim, M))

        corr_Q = pm.LKJCorr("corr_Q", n=K, eta=4.0)
        corr_R = pm.LKJCorr("corr_R", n=obs_dim, eta=4.0)

        mu_logsig_Q = pm.Normal("mu_logsig_Q", -2.0, 0.5, shape=K)
        sd_logsig_Q = pm.HalfNormal("sd_logsig_Q", 0.5, shape=K)
        logsig_Q_g = pm.Normal("logsig_Q_g", mu_logsig_Q, sd_logsig_Q, shape=(groups, K))
        sig_Q_g = pm.Deterministic("sig_Q_g", pm.math.exp(logsig_Q_g))
        Q_g = pm.Deterministic(
            "Q_g",
            sig_Q_g[:, :, None] * corr_Q[None, :, :] * sig_Q_g[:, None, :],
        )

        mu_logsig_R = pm.Normal("mu_logsig_R", -1.0, 0.5, shape=obs_dim)
        sd_logsig_R = pm.HalfNormal("sd_logsig_R", 0.5, shape=obs_dim)
        logsig_R_g = pm.Normal("logsig_R_g", mu_logsig_R, sd_logsig_R, shape=(groups, obs_dim))
        sig_R_g = pm.Deterministic("sig_R_g", pm.math.exp(logsig_R_g))
        R_g = pm.Deterministic(
            "R_g",
            sig_R_g[:, :, None] * corr_R[None, :, :] * sig_R_g[:, None, :],
        )

        m0 = pm.Normal("m0", 0.0, 1.0, shape=(groups, K))
        P0 = pt.eye(K) * 10.0

        ll_total = pt.as_tensor_variable(0.0)
        for g in range(groups):
            ll_g = kalman_logp_pt(
                y=y_pt[g],
                w=w_pt[g],
                mask=mask_pt[g],
                A=A_g[g],
                B=B_g[g],
                H=H_g[g],
                D=D_g[g],
                Q=Q_g[g],
                R=R_g[g],
                m0=m0[g],
                P0=P0,
            )
            ll_total = ll_total + ll_g

        pm.Potential("kalman_ll", ll_total)

        trace = pm.sample(draws=draws, tune=tune, chains=4, target_accept=target_accept)

    return model, trace


def sample_latent_states_from_trace(
    trace: az.InferenceData,
    y_data: NDArray[np.floating],
    w_data: NDArray[np.floating],
    K: int,
    *,
    n_draws: int = 200,
    P0_scale: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Sample latent trajectories from posterior draws using FFBS."""

    rng = np.random.default_rng() if rng is None else rng

    y_data = np.asarray(y_data, dtype=float)
    w_data = np.asarray(w_data, dtype=float)
    groups, time_steps, _ = y_data.shape

    posterior = trace.posterior.stack(sample=("chain", "draw"))
    n_samples = posterior.sizes["sample"]
    draw_indices = rng.choice(n_samples, size=min(n_draws, n_samples), replace=False)

    x_draws = np.zeros((len(draw_indices), groups, time_steps, K))

    for i, s in enumerate(draw_indices):
        A = posterior["A_g"].isel(sample=s).values
        B = posterior["B_g"].isel(sample=s).values
        H = posterior["H_g"].isel(sample=s).values
        D = posterior["D_g"].isel(sample=s).values
        Q = posterior["Q_g"].isel(sample=s).values
        R = posterior["R_g"].isel(sample=s).values
        m0 = posterior["m0"].isel(sample=s).values

        for g in range(groups):
            result = ffbs_sample_states(
                y=y_data[g],
                w=w_data[g],
                A=A[g],
                B=B[g],
                H=H[g],
                D=D[g],
                Q=Q[g],
                R=R[g],
                m0=m0[g],
                P0=np.eye(K) * P0_scale,
            )
            x_draws[i, g] = result.states

    return x_draws


def plot_param_recovery(
    trace: az.InferenceData,
    *,
    true_params: Optional[dict[str, NDArray[np.floating]]] = None,
    varnames: Iterable[str] = ("corr_Q", "corr_R"),
) -> None:
    """Plot posterior traces and optionally compare to known parameters."""

    az.plot_trace(trace, var_names=list(varnames))
    plt.tight_layout()
    plt.show()

    if true_params is None:
        return

    for var in varnames:
        posterior_mean = trace.posterior[var].mean(dim=("chain", "draw")).values
        print(f"{var} posterior mean:\n{posterior_mean}")
        print(f"{var} true:\n{true_params.get(var, None)}")


def plot_latent_trajectories(
    x_draws: NDArray[np.floating],
    *,
    x_true: Optional[NDArray[np.floating]] = None,
    group: int = 0,
    state_index: int = 0,
    cred_interval: tuple[float, float] = (0.05, 0.95),
) -> None:
    """Plot posterior mean and credible bands for a single latent trajectory."""

    trajectories = x_draws[:, group, :, state_index]
    mean = trajectories.mean(axis=0)
    lower = np.quantile(trajectories, cred_interval[0], axis=0)
    upper = np.quantile(trajectories, cred_interval[1], axis=0)

    plt.figure()
    plt.plot(mean, label="posterior mean")
    plt.fill_between(np.arange(trajectories.shape[1]), lower, upper, alpha=0.3, label="credible band")
    if x_true is not None:
        plt.plot(x_true[group, :, state_index], linestyle="--", label="true")
    plt.title(f"Group {group}, state {state_index}")
    plt.legend()
    plt.tight_layout()
    plt.show()

