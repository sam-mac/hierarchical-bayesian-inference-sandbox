"""Identified scalar LGSSM benchmark with known noise and a unit loading.

This is a deliberately restricted companion to the general latent-state model.
States are integrated out exactly, not fitted as separate NUTS parameters.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def scalar_kalman_log_likelihood(
    observations: Any, transition: Any, process_std: Any, emission_std: Any
) -> Any:
    """JAX log p(y | phi, q, r), summed over groups and time.

    Shapes are (groups, time) for observations and (groups,) for parameters.
    The initial state is N(0, 1); q and r are known standard deviations.
    This differentiable primitive assumes valid inputs; use the host-side
    validate_scalar_lgssm_inputs before starting inference.
    """
    import jax.numpy as jnp
    from jax import lax

    y = jnp.asarray(observations, dtype=float)
    phi = jnp.asarray(transition, dtype=y.dtype)
    q2 = jnp.square(jnp.asarray(process_std, dtype=y.dtype))
    r2 = jnp.square(jnp.asarray(emission_std, dtype=y.dtype))

    def step(carry: tuple[Any, Any], value: Any) -> tuple[tuple[Any, Any], Any]:
        mean, variance = carry
        innovation_variance = variance + r2
        residual = value - mean
        log_likelihood = -0.5 * (
            jnp.log(2.0 * jnp.pi * innovation_variance)
            + residual**2 / innovation_variance
        )
        gain = variance / innovation_variance
        filtered_mean = mean + gain * residual
        filtered_variance = (1.0 - gain) * variance
        return (
            phi * filtered_mean,
            phi**2 * filtered_variance + q2,
        ), log_likelihood

    _, terms = lax.scan(step, (jnp.zeros_like(phi), jnp.ones_like(phi)), y.T)
    return terms.sum()


def validate_scalar_lgssm_inputs(
    observations: NDArray[np.float64],
    process_std: NDArray[np.float64],
    emission_std: NDArray[np.float64],
) -> None:
    """Validate host arrays before tracing the model or starting MCMC.

    Value-dependent NumPy checks must not run inside JAX tracing.
    """
    y, q, r = (np.asarray(value) for value in (observations, process_std, emission_std))
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] < 2:
        raise ValueError("observations must have shape (groups >= 1, time >= 2)")
    if q.shape != (y.shape[0],) or r.shape != q.shape:
        raise ValueError("noise standard deviations must have shape (groups,)")
    if not all(np.isfinite(value).all() for value in (y, q, r)):
        raise ValueError("observations and noise standard deviations must be finite")
    if np.any(q <= 0) or np.any(r <= 0):
        raise ValueError("noise standard deviations must be strictly positive")


def hierarchical_scalar_lgssm_model(
    observations: NDArray[np.float64],
    process_std: NDArray[np.float64],
    emission_std: NDArray[np.float64],
) -> None:
    """Non-centered population hierarchy for stable group transitions.

    phi_g = tanh(location + scale * raw_g). Hyperparameters are on the
    unconstrained (atanh) scale, not the transition-coefficient scale.
    Call validate_scalar_lgssm_inputs on host arrays before inference.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError("observations must have shape (groups, time >= 2)")
    groups = observations.shape[0]
    if process_std.shape != (groups,) or emission_std.shape != (groups,):
        raise ValueError("noise standard deviations must have shape (groups,)")
    location = numpyro.sample("transition_location", dist.Normal(0.0, 0.8))
    scale = numpyro.sample("transition_scale", dist.HalfNormal(0.4))
    raw = numpyro.sample(
        "transition_raw", dist.Normal(0.0, 1.0).expand([groups]).to_event(1)
    )
    transition = numpyro.deterministic("transition", jnp.tanh(location + scale * raw))
    numpyro.factor(
        "observations",
        scalar_kalman_log_likelihood(
            observations, transition, process_std, emission_std
        ),
    )


def forecast_scalar_lgssm(
    observations: NDArray[np.float64],
    transition_draws: NDArray[np.float64],
    process_std: NDArray[np.float64],
    emission_std: NDArray[np.float64],
    *,
    horizon: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw held-out trajectories and return conditional means/variances.

    Transition draws have shape (draws, groups). The returned three arrays have
    shape (draws, groups, horizon). Means and variances define the per-draw
    Gaussian forecast marginal, useful for a stable mixture predictive score.
    Only training observations enter the filter, never held-out values.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    phi = np.asarray(transition_draws, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    q = np.asarray(process_std, dtype=np.float64)
    r = np.asarray(emission_std, dtype=np.float64)
    if y.ndim != 2 or phi.ndim != 2 or phi.shape[1] != y.shape[0] or y.shape[1] == 0:
        raise ValueError("incompatible observation and transition draw shapes")
    if q.shape != (y.shape[0],) or r.shape != q.shape:
        raise ValueError("noise standard deviations must have shape (groups,)")
    if (
        not all(np.isfinite(a).all() for a in (y, phi, q, r))
        or np.any(q <= 0)
        or np.any(r <= 0)
    ):
        raise ValueError(
            "observations and parameters must be finite; noise must be positive"
        )
    mean = np.zeros_like(phi)
    variance = np.ones_like(phi)
    for timestep, value in enumerate(y.T):
        if timestep:
            mean = phi * mean
            variance = phi**2 * variance + q**2
        gain = variance / (variance + r**2)
        mean = mean + gain * (value - mean)
        variance = (1.0 - gain) * variance

    rng = np.random.default_rng(seed)
    state = rng.normal(mean, np.sqrt(variance))
    shape = (*phi.shape, horizon)
    predictive = np.empty(shape)
    means = np.empty(shape)
    variances = np.empty(shape)
    for step in range(horizon):
        mean = phi * mean
        variance = phi**2 * variance + q**2
        means[:, :, step] = mean
        variances[:, :, step] = variance + r**2
        state = phi * state + rng.normal(size=phi.shape) * q
        predictive[:, :, step] = state + rng.normal(size=phi.shape) * r
    return predictive, means, variances
