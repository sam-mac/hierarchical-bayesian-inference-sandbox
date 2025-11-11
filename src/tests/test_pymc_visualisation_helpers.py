import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import arviz as az

from bayes_tools.helpers.pymc_visualisation_helpers import (
    plot_latent_survey_process,
    plot_prior_predictive_vs_observed,
)


def _make_dataset() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=6, freq="MS")
    groups = np.array(["A", "A", "B", "B", "A", "B"])
    values = np.linspace(0.5, 2.5, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "ou_code": groups,
            "value": values,
            "survey_score": values * 10.0 + 50.0,
        }
    )


def _build_simple_model(df: pd.DataFrame) -> pm.Model:
    coords = {"obs": np.arange(len(df))}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal(
            "value_obs",
            mu=mu,
            sigma=sigma,
            observed=df["value"],
            dims="obs",
        )
    return model


def test_plot_prior_predictive_summary_returns_figure():
    df = _make_dataset()
    model = _build_simple_model(df)
    fig = plot_prior_predictive_vs_observed(
        df,
        model,
        prior_var="value_obs",
        obs_col="value",
        samples=5,
        mode="summary",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_prior_predictive_samples_returns_figure():
    df = _make_dataset()
    model = _build_simple_model(df)
    fig = plot_prior_predictive_vs_observed(
        df,
        model,
        prior_var="value_obs",
        obs_col="value",
        samples=5,
        mode="samples",
        max_sample_traces=3,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_latent_survey_process_returns_figure():
    df = _make_dataset()
    n_obs = len(df)
    rng = np.random.default_rng(42)
    draws = rng.normal(size=(2, 3, n_obs))
    idata = az.from_dict(
        posterior={"x_latent": draws},
        coords={"obs": np.arange(n_obs)},
        dims={"x_latent": ["obs"]},
    )
    fig = plot_latent_survey_process(
        df,
        idata,
        ou_code="A",
        survey_col="survey_score",
        smooth_factor=4,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3
