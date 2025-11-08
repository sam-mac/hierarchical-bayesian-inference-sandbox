import pandas as pd
import pytest

from bayes_tools.helpers.synthetic_data_helpers import (
    aggregate_to_parent,
    make_hierarchical_ou_dataset,
)


def test_make_hierarchical_ou_dataset_structure(synthetic_panel: pd.DataFrame) -> None:
    """The generated data frame should contain expected columns and rows."""
    expected_columns = {
        "region_id",
        "site_id",
        "ou_code",
        "date",
        "productivity",
        "fte_operational",
        "survey_score",
        "n_respondents",
    }
    assert set(synthetic_panel.columns) == expected_columns
    # 1 region * 1 site * 2 OUs * 12 months
    assert len(synthetic_panel) == 24
    assert (synthetic_panel["productivity"] > 0).all()
    assert synthetic_panel["date"].dt.is_month_start.all()


def test_make_hierarchical_ou_dataset_reproducible() -> None:
    df_one = make_hierarchical_ou_dataset(seed=7)
    df_two = make_hierarchical_ou_dataset(seed=7)
    pd.testing.assert_frame_equal(df_one, df_two)


def test_aggregate_to_parent_site_level(synthetic_panel: pd.DataFrame) -> None:
    aggregated = aggregate_to_parent(synthetic_panel, level="site")
    assert {"site_id", "date"}.issubset(aggregated.columns)
    assert (aggregated["fte_operational"] >= 0).all()
    # Each month should have exactly one aggregated row for the single site
    assert aggregated.groupby("date").size().eq(1).all()


def test_aggregate_to_parent_region_level(synthetic_panel: pd.DataFrame) -> None:
    aggregated = aggregate_to_parent(synthetic_panel, level="region")
    assert {"region_id", "date"}.issubset(aggregated.columns)
    assert (aggregated["productivity"] >= 0).all()


def test_aggregate_to_parent_invalid_level(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="level must be 'site' or 'region'"):
        aggregate_to_parent(synthetic_panel, level="invalid")
