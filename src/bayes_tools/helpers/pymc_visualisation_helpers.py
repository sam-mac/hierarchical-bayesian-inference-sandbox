"""Plotting utilities for PyMC models and hierarchical datasets."""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import xarray as xr
from plotly import colors as plotly_colors

try:
    import arviz as az
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise ModuleNotFoundError(
        "arviz is required for pymc_visualisation_helpers"
    ) from exc

def _ensure_datetime(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    if dates.isna().any():
        raise ValueError("Date column contains non-convertible entries")
    return dates


@dataclass
class PriorPredictiveResult:
    draws: np.ndarray
    """Array of shape (n_obs, n_draws)."""
    draws_per_group: Mapping[str, np.ndarray]
    """Mapping from group id to draw matrix (n_points, n_draws)."""
    dates_per_group: Mapping[str, pd.Series]
    observations_per_group: Mapping[str, pd.Series]


def _sample_prior_predictive(
    model: pm.Model,
    prior_var: str,
    obs_dim: str,
    *,
    samples: int,
    random_seed: int | np.random.Generator | None,
) -> np.ndarray:
    with model:
        idata = pm.sample_prior_predictive(
            draws=samples, var_names=[prior_var], random_seed=random_seed
        )
    data_array = idata.prior_predictive[prior_var]
    if obs_dim not in data_array.dims:
        raise ValueError(
            f"Prior predictive variable '{prior_var}' does not include dimension '{obs_dim}'."
        )
    sample_dims = [dim for dim in data_array.dims if dim != obs_dim]
    if not sample_dims:
        stacked = data_array.expand_dims({"draw": 1})
        sample_dims = ["draw"]
        data_array = stacked
    stacked = data_array.stack(sample=sample_dims)
    ordered = stacked.transpose(obs_dim, "sample")
    values = np.asarray(ordered)
    return values


def _assign_group_data(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    obs_col: str,
    draws: np.ndarray,
) -> PriorPredictiveResult:
    df_reset = df.reset_index(drop=True)
    if len(df_reset) != draws.shape[0]:
        raise ValueError(
            "Number of rows in dataset does not match number of prior predictive observations."
        )
    group_map: dict[str, np.ndarray] = {}
    date_map: dict[str, pd.Series] = {}
    obs_map: dict[str, pd.Series] = {}
    for group, group_df in df_reset.groupby(group_col, sort=False):
        idx = group_df.index.to_numpy()
        dates = _ensure_datetime(group_df[date_col])
        order = np.argsort(dates.values)
        ordered_idx = idx[order]
        group_map[str(group)] = draws[ordered_idx, :]
        date_map[str(group)] = dates.iloc[order]
        obs_map[str(group)] = group_df.iloc[order][obs_col]
    return PriorPredictiveResult(draws=draws, draws_per_group=group_map, dates_per_group=date_map, observations_per_group=obs_map)


def plot_prior_predictive_vs_observed(
    df: pd.DataFrame,
    model: pm.Model,
    *,
    prior_var: str,
    obs_col: str,
    date_col: str = "date",
    group_col: str = "ou_code",
    obs_dim: str = "obs",
    mode: str = "summary",
    samples: int = 200,
    random_seed: int | np.random.Generator | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    max_sample_traces: int = 100,
    colours: Sequence[str] | None = None,
) -> go.Figure:
    """Plot prior predictive draws against observed data for each group."""

    if mode not in {"summary", "samples"}:
        raise ValueError("mode must be either 'summary' or 'samples'")
    if samples < 1:
        raise ValueError("samples must be a positive integer")
    if mode == "samples" and max_sample_traces < 1:
        raise ValueError("max_sample_traces must be at least 1 when mode='samples'")

    draws = _sample_prior_predictive(
        model, prior_var=prior_var, obs_dim=obs_dim, samples=samples, random_seed=random_seed
    )
    if transform is not None:
        draws = transform(draws)

    result = _assign_group_data(df, group_col, date_col, obs_col, draws)

    palette = list(colours) if colours is not None else list(plotly_colors.qualitative.Plotly)
    if not palette:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig = go.Figure()
    for idx, (group, group_draws) in enumerate(result.draws_per_group.items()):
        colour = palette[idx % len(palette)]
        dates = result.dates_per_group[group]
        obs = result.observations_per_group[group]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=obs,
                mode="lines+markers",
                name=f"{group} observed",
                line=dict(color=colour, width=2),
                marker=dict(color=colour),
                legendgroup=group,
            )
        )
        if mode == "summary":
            mean = group_draws.mean(axis=1)
            lo = np.quantile(group_draws, 0.05, axis=1)
            hi = np.quantile(group_draws, 0.90, axis=1)
            try:
                r, g, b = plotly_colors.hex_to_rgb(colour)
            except ValueError:
                converted = plotly_colors.convert_colors_to_same_type([colour])
                _, converted_colour = converted[0]
                r, g, b = plotly_colors.hex_to_rgb(converted_colour)
            opacity_colour = f"rgba({r},{g},{b},0.2)"
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=hi,
                    mode="lines",
                    line=dict(color=colour, width=0),
                    showlegend=False,
                    legendgroup=group,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=lo,
                    mode="lines",
                    line=dict(color=colour, width=0),
                    fill="tonexty",
                    fillcolor=opacity_colour,
                    name=f"{group} prior P05–P90",
                    legendgroup=group,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=mean,
                    mode="lines",
                    line=dict(color=colour, dash="dash"),
                    name=f"{group} prior mean",
                    legendgroup=group,
                )
            )
        else:
            n_traces = min(max_sample_traces, group_draws.shape[1])
            for draw_idx in range(n_traces):
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=group_draws[:, draw_idx],
                        mode="lines",
                        line=dict(color=colour, width=1),
                        opacity=0.2,
                        name=f"{group} prior sample" if draw_idx == 0 else None,
                        legendgroup=group,
                        showlegend=draw_idx == 0,
                        hoverinfo="skip",
                    )
                )
    fig.update_layout(
        title="Prior predictive distribution vs observed",
        xaxis_title="Date",
        yaxis_title=obs_col,
        legend_title=group_col,
    )
    return fig


def plot_latent_survey_process(
    df: pd.DataFrame,
    idata: az.InferenceData,
    ou_code: str,
    *,
    latent_var: str = "x_latent",
    obs_dim: str = "obs",
    date_col: str = "date",
    group_col: str = "ou_code",
    survey_col: str = "survey_score",
    smooth_factor: int = 6,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> go.Figure:
    """Plot posterior latent survey process for a given operating unit."""

    if smooth_factor < 1:
        raise ValueError("smooth_factor must be at least 1")

    if latent_var not in idata.posterior:
        raise KeyError(f"'{latent_var}' not present in posterior")

    df_reset = df.reset_index(drop=True)
    if ou_code not in df_reset[group_col].astype(str).unique():
        raise KeyError(f"Operating unit '{ou_code}' not found in dataset")

    data_array = idata.posterior[latent_var]
    if obs_dim not in data_array.dims:
        raise ValueError(
            f"Latent variable '{latent_var}' does not include dimension '{obs_dim}'."
        )
    sample_dims = [dim for dim in data_array.dims if dim != obs_dim]
    stacked = data_array.stack(sample=sample_dims)
    ordered = stacked.transpose(obs_dim, "sample")
    draws = np.asarray(ordered)

    ou_mask = df_reset[group_col].astype(str) == ou_code
    row_idx = np.flatnonzero(ou_mask)
    if row_idx.size == 0:
        raise KeyError(f"No rows found for operating unit '{ou_code}'")

    dates = _ensure_datetime(df_reset.loc[row_idx, date_col])
    order = np.argsort(dates.values)
    ordered_idx = row_idx[order]
    ordered_dates = dates.iloc[order]
    draws_ou = draws[ordered_idx, :]

    mean = draws_ou.mean(axis=1)
    lo = np.quantile(draws_ou, quantiles[0], axis=1)
    hi = np.quantile(draws_ou, quantiles[1], axis=1)

    survey_vals = df_reset.loc[ordered_idx, survey_col].to_numpy(dtype=float)
    if np.isfinite(survey_vals).any():
        survey_mean = np.nanmean(survey_vals)
        survey_std = np.nanstd(survey_vals)
        if not np.isfinite(survey_std) or survey_std <= 0:
            survey_std = 1.0
        survey_standardised = (survey_vals - survey_mean) / survey_std
    else:
        survey_standardised = np.full_like(survey_vals, np.nan)

    if smooth_factor > 1 and ordered_dates.size > 1:
        numeric = ordered_dates.to_numpy(dtype="datetime64[ns]").astype("int64")
        dense_count = (ordered_dates.size - 1) * smooth_factor + 1
        dense_numeric = np.linspace(numeric[0], numeric[-1], dense_count)
        dense_dates = pd.to_datetime(dense_numeric)
        mean_smooth = np.interp(dense_numeric, numeric, mean)
        lo_smooth = np.interp(dense_numeric, numeric, lo)
        hi_smooth = np.interp(dense_numeric, numeric, hi)
    else:
        dense_dates = ordered_dates
        mean_smooth = mean
        lo_smooth = lo
        hi_smooth = hi

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dense_dates,
            y=hi_smooth,
            mode="lines",
            line=dict(color="#1f77b4", width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dense_dates,
            y=lo_smooth,
            mode="lines",
            line=dict(color="#1f77b4", width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="90% credible interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dense_dates,
            y=mean_smooth,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="Posterior mean",
        )
    )
    if np.isfinite(survey_standardised).any():
        fig.add_trace(
            go.Scatter(
                x=ordered_dates,
                y=survey_standardised,
                mode="markers",
                marker=dict(symbol="x", size=9, color="#d62728"),
                name="Observed survey (z)",
            )
        )
    fig.update_layout(
        title=f"Latent survey process for OU {ou_code}",
        xaxis_title="Date",
        yaxis_title="Standardised survey score",
    )
    return fig
