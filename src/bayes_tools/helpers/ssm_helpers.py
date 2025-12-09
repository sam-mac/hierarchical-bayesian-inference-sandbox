"""State space model utilities for grouped structural time-series models."""


from typing import TypedDict

import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan


class StructuralSSMComponents(TypedDict):
    """Collection of latent state components returned by the SSM builder."""

    x: pm.model.Deterministic
    level: pm.model.Deterministic
    trend: pm.model.Deterministic
    ar: pm.model.Deterministic
    season: pm.model.Deterministic


def build_grouped_structural_ssm(
    name: str,
    *,
    n_groups: int,
    T: int,
    phi_trend: pt.TensorLike,
    rho_g: pt.TensorLike,
    sigma_level_g: pt.TensorLike,
    sigma_trend_g: pt.TensorLike,
    sigma_ar_g: pt.TensorLike,
    level0_g: pt.TensorLike,
    trend0_g: pt.TensorLike,
    ar0_g: pt.TensorLike,
    month_idx_t: pt.TensorLike,
    season_effect_12: pt.TensorLike,
    season_scale_g: pt.TensorLike,
) -> StructuralSSMComponents:
    """Build a grouped structural state-space model.

    The latent process per group ``g`` is defined as:

    - ``b_t = phi_trend * b_{t-1} + zeta_t``
    - ``l_t = l_{t-1} + b_{t-1} + eta_t``
    - ``r_t = rho_g * r_{t-1} + kappa_t``

    The observable latent location is
    ``x_{g,t} = l_{g,t} + r_{g,t} + season_scale_g[g] * season_effect_12[month(t)]``.

    Parameters
    ----------
    name:
        Base name for deterministic outputs.
    n_groups:
        Number of groups (``G``).
    T:
        Number of time steps (``T``).
    phi_trend:
        Damped trend coefficient shared across groups.
    rho_g:
        Group-specific autoregressive coefficients of shape ``(G,)`` in ``(-1, 1)``.
    sigma_level_g:
        Level innovation scales per group, shape ``(G,)``.
    sigma_trend_g:
        Trend innovation scales per group, shape ``(G,)``.
    sigma_ar_g:
        Autoregressive innovation scales per group, shape ``(G,)``.
    level0_g:
        Initial level states per group, shape ``(G,)``.
    trend0_g:
        Initial trend states per group, shape ``(G,)``.
    ar0_g:
        Initial autoregressive states per group, shape ``(G,)``.
    month_idx_t:
        Time index to month mapping of shape ``(T,)`` with values in ``0..11``.
    season_effect_12:
        Shared seasonal monthly effects of shape ``(12,)``.
    season_scale_g:
        Group-specific seasonal scaling factors, shape ``(G,)``.

    Returns
    -------
    StructuralSSMComponents
        Deterministic tensors for the latent state components and combined signal
        ``x``. Each component has shape ``(G, T)``.
    """

    # Innovations: time-major (T-1, G)
    eta_raw = pm.Normal(f"{name}_eta_raw", 0.0, 1.0, shape=(T - 1, n_groups))
    zeta_raw = pm.Normal(f"{name}_zeta_raw", 0.0, 1.0, shape=(T - 1, n_groups))
    kappa_raw = pm.Normal(f"{name}_kappa_raw", 0.0, 1.0, shape=(T - 1, n_groups))

    eta = eta_raw * sigma_level_g[None, :]
    zeta = zeta_raw * sigma_trend_g[None, :]
    kappa = kappa_raw * sigma_ar_g[None, :]

    def step(
        eta_t: pt.TensorLike,
        zeta_t: pt.TensorLike,
        kappa_t: pt.TensorLike,
        level_prev: pt.TensorLike,
        trend_prev: pt.TensorLike,
        ar_prev: pt.TensorLike,
        phi_trend_value: pt.TensorLike,
        rho_g_value: pt.TensorLike,
    ) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        trend_t = phi_trend_value * trend_prev + zeta_t
        level_t = level_prev + trend_prev + eta_t
        ar_t = rho_g_value * ar_prev + kappa_t
        return level_t, trend_t, ar_t

    (level_seq, trend_seq, ar_seq), _ = scan(
        fn=step,
        sequences=[eta, zeta, kappa],
        outputs_info=[level0_g, trend0_g, ar0_g],
        non_sequences=[phi_trend, rho_g],
    )

    # Add t=0 back; transpose to (G,T)
    level = pt.concatenate([level0_g[None, :], level_seq], axis=0).T
    trend = pt.concatenate([trend0_g[None, :], trend_seq], axis=0).T
    ar = pt.concatenate([ar0_g[None, :], ar_seq], axis=0).T

    # Seasonality: take month-specific effect for each t, then scale per group
    season_t = pt.take(season_effect_12, month_idx_t)  # (T,)
    season = season_scale_g[:, None] * season_t[None, :]  # (G,T)

    x = level + ar + season

    return {
        "x": pm.Deterministic(name, x),
        "level": pm.Deterministic(f"{name}_level", level),
        "trend": pm.Deterministic(f"{name}_trend", trend),
        "ar": pm.Deterministic(f"{name}_ar", ar),
        "season": pm.Deterministic(f"{name}_season", season),
    }
