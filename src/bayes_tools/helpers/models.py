"""Model-building helpers for hierarchical Bayesian time-series models."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt

from .ssm_helpers import build_grouped_structural_ssm


def build_grouped_structural_productivity_model(
    n_groups: int,
    n_time: int,
    group_idx: npt.NDArray[np.integer[Any]],
    time_idx: npt.NDArray[np.integer[Any]],
    month_idx_t: npt.NDArray[np.integer[Any]],
    y: npt.NDArray[np.floating[Any]],
) -> pm.Model:
    """Construct the grouped structural state-space model and likelihood.

    The model mirrors the reference specification provided by the data science team,
    wiring the :func:`build_grouped_structural_ssm` helper into a full PyMC model
    with hierarchical priors and an example productivity likelihood.

    Parameters
    ----------
    n_groups:
        Number of distinct groups represented in the panel.
    n_time:
        Number of time periods per group.
    group_idx:
        Observed group indices for each data point, integers in ``[0, n_groups)``.
    time_idx:
        Observed time indices for each data point, integers in ``[0, n_time)``.
    month_idx_t:
        Mapping from each time index to a month in ``[0, 11]``.
    y:
        Observed response vector to model (e.g., productivity).

    Returns
    -------
    pm.Model
        A compiled PyMC model with named deterministic components from the SSM and
        an observation likelihood for ``y``.
    """

    with pm.Model() as model:
        # ---------- Shared seasonal pattern (monthly dummies) ----------
        sigma_season = pm.HalfNormal("sigma_season", 0.5)
        season_raw_12 = pm.Normal("season_raw_12", 0.0, sigma_season, shape=12)

        # Sum-to-zero for identifiability (so seasonality doesn't steal the intercept)
        season_effect_12 = pm.Deterministic(
            "season_effect_12", season_raw_12 - pt.mean(season_raw_12)
        )

        # ---------- Damped trend coefficient (shared => stiffer, less leakage) ----------
        # logit(0.9) ≈ 2.197; this biases toward strong persistence but still mean-reverting
        phi_trend_z = pm.Normal("phi_trend_z", mu=2.2, sigma=0.6)
        phi_trend = pm.Deterministic("phi_trend", pm.math.sigmoid(phi_trend_z))  # (0,1)

        # ---------- Hierarchical correlated group effects (LKJ correlation) ----------
        # We'll correlate: [level0_g, trend0_g, atanh(rho_g), log_season_scale_g]
        k = 4

        mu_level0 = pm.Normal("mu_level0", 0.0, 1.0)
        mu_trend0 = pm.Normal("mu_trend0", 0.0, 0.3)
        mu_rho_z = pm.Normal("mu_rho_z", 0.0, 0.8)  # atanh scale
        mu_log_a = pm.Normal("mu_log_a", 0.0, 0.5)  # log scale for season amplitude

        mu_theta = pt.stack([mu_level0, mu_trend0, mu_rho_z, mu_log_a])  # (K,)

        chol, corr, sds = pm.LKJCholeskyCov(
            "group_chol",
            n=k,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        pm.Deterministic("group_corr", corr)
        pm.Deterministic("group_sds", sds)

        z = pm.Normal("group_z", 0.0, 1.0, shape=(n_groups, k))  # non-centered
        theta_g = pm.Deterministic("theta_g", mu_theta + z @ chol.T)  # (G,K)

        level0_g = theta_g[:, 0]
        trend0_g = theta_g[:, 1]
        rho_g = pm.Deterministic("rho_g", pt.tanh(theta_g[:, 2]))  # (-1,1)
        season_scale_g = pm.Deterministic("season_scale_g", pt.exp(theta_g[:, 3]))  # >0

        # ---------- Hierarchical shrinkage for process noise (keeps SSM stiff) ----------
        # Put informative-ish priors here to prevent productivity from bending the latent too much.
        # Tune the log(0.05), etc. to your scale.
        mu_log_sig_l = pm.Normal("mu_log_sigma_level", np.log(0.05), 0.7)
        sd_log_sig_l = pm.HalfNormal("sd_log_sigma_level", 0.3)
        sigma_level_g = pm.Deterministic(
            "sigma_level_g",
            pt.exp(mu_log_sig_l + sd_log_sig_l * pm.Normal("z_sig_l", 0, 1, shape=n_groups)),
        )

        mu_log_sig_b = pm.Normal("mu_log_sigma_trend", np.log(0.02), 0.7)
        sd_log_sig_b = pm.HalfNormal("sd_log_sigma_trend", 0.3)
        sigma_trend_g = pm.Deterministic(
            "sigma_trend_g",
            pt.exp(mu_log_sig_b + sd_log_sig_b * pm.Normal("z_sig_b", 0, 1, shape=n_groups)),
        )

        mu_log_sig_r = pm.Normal("mu_log_sigma_ar", np.log(0.05), 0.7)
        sd_log_sig_r = pm.HalfNormal("sd_log_sigma_ar", 0.3)
        sigma_ar_g = pm.Deterministic(
            "sigma_ar_g",
            pt.exp(mu_log_sig_r + sd_log_sig_r * pm.Normal("z_sig_r", 0, 1, shape=n_groups)),
        )

        # Initial AR state: keep it small / stationary-ish
        ar0_g = pm.Normal(
            "ar0_g", 0.0, sigma_ar_g / pt.sqrt(1 - rho_g**2 + 1e-6), shape=n_groups
        )

        # ---------- Build the grouped SSM ----------
        month_idx_data = pm.Data("month_idx_t", month_idx_t.astype("int32"))

        ssm = build_grouped_structural_ssm(
            "x",
            n_groups=n_groups,
            T=n_time,
            phi_trend=phi_trend,
            rho_g=rho_g,
            sigma_level_g=sigma_level_g,
            sigma_trend_g=sigma_trend_g,
            sigma_ar_g=sigma_ar_g,
            level0_g=level0_g,
            trend0_g=trend0_g,
            ar0_g=ar0_g,
            month_idx_t=month_idx_data,
            season_effect_12=season_effect_12,
            season_scale_g=season_scale_g,
        )
        x = ssm["x"]  # (G,T)

        # ---------- Example productivity likelihood ----------
        group_idx_data = pm.Data("group_idx", group_idx.astype("int32"))
        time_idx_data = pm.Data("time_idx", time_idx.astype("int32"))

        x_obs = x[group_idx_data, time_idx_data]  # (N,)

        alpha = pm.Normal("alpha", 0.0, 1.0)
        sigma_y = pm.HalfNormal("sigma_y", 1.0)

        mu_y = alpha * x_obs  # + other effects...

        pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=y)

    return model
