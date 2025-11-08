import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")

from bayes_tools.helpers.visualisation_helpers import (
    _prepare_panel,
    plot_headcount_vs_productivity,
    plot_metric_timeseries,
    plot_productivity_vs_survey,
    plot_survey_heatmap,
)


def test_prepare_panel_with_level_and_groups(synthetic_panel: pd.DataFrame) -> None:
    panel = _prepare_panel(synthetic_panel, level="site")
    assert {"group_id", "level"}.issubset(panel.columns)
    assert (panel["level"] == "site").all()

    filtered = _prepare_panel(synthetic_panel, level="ou", groups=["OU1-1-1"])
    assert filtered["group_id"].unique().tolist() == ["OU1-1-1"]


def test_prepare_panel_invalid_level(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="level must be one of"):
        _prepare_panel(synthetic_panel, level="invalid")


def test_plot_metric_timeseries_returns_figure(synthetic_panel: pd.DataFrame) -> None:
    fig = plot_metric_timeseries(synthetic_panel, metric="productivity", level="ou")
    assert isinstance(fig, go.Figure)
    assert fig.data  # ensure at least one trace is plotted


def test_plot_metric_timeseries_invalid_metric(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Column 'missing_metric'"):
        plot_metric_timeseries(synthetic_panel, metric="missing_metric")


def test_plot_productivity_vs_survey_requires_data(synthetic_panel: pd.DataFrame) -> None:
    fig = plot_productivity_vs_survey(synthetic_panel, level="site")
    assert isinstance(fig, go.Figure)

    empty_panel = synthetic_panel.copy()
    empty_panel["survey_score"] = pd.NA
    with pytest.raises(ValueError, match="No survey responses"):
        plot_productivity_vs_survey(empty_panel, level="ou")


def test_plot_survey_heatmap_returns_figure(synthetic_panel: pd.DataFrame) -> None:
    fig = plot_survey_heatmap(synthetic_panel, level="ou", value="survey_score")
    assert isinstance(fig, go.Figure)
    assert fig.data and isinstance(fig.data[0], go.Heatmap)


def test_plot_headcount_vs_productivity_returns_figure(synthetic_panel: pd.DataFrame) -> None:
    fig = plot_headcount_vs_productivity(synthetic_panel, level="ou")
    assert isinstance(fig, go.Figure)
    assert fig.data


def test_plot_headcount_vs_productivity_requires_values(
    synthetic_panel: pd.DataFrame,
) -> None:
    empty_panel = synthetic_panel.copy()
    empty_panel["fte_operational"] = pd.NA
    empty_panel["productivity"] = pd.NA
    with pytest.raises(ValueError, match="No rows with both FTE and productivity"):
        plot_headcount_vs_productivity(empty_panel)
