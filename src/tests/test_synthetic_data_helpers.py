import numpy as np
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


def test_make_hierarchical_ou_dataset_default_structure_balanced() -> None:
    n_regions = 2
    n_sites_per_region = 2
    n_ous_per_site = 3

    panel = make_hierarchical_ou_dataset(
        n_regions=n_regions,
        n_sites_per_region=n_sites_per_region,
        n_ous_per_site=n_ous_per_site,
        n_years=1,
        wave_missing_prob=0.0,
        seed=7,
    )

    months = panel["date"].dt.to_period("M").nunique()
    expected_ou = n_regions * n_sites_per_region * n_ous_per_site

    assert len(panel) == expected_ou * months
    assert panel["region_id"].nunique() == n_regions
    assert panel.groupby("region_id")["site_id"].nunique().eq(n_sites_per_region).all()
    assert panel.groupby("site_id")["ou_code"].nunique().eq(n_ous_per_site).all()


def test_make_hierarchical_ou_dataset_imbalanced_structure() -> None:
    panel_structure = {
        "North": {"North-A": 1, "North-B": 3},
        "South": {"South-A": 2},
    }

    panel = make_hierarchical_ou_dataset(
        panel_structure=panel_structure,
        n_years=1,
        wave_missing_prob=0.0,
        seed=5,
    )

    months = panel["date"].dt.to_period("M").nunique()
    expected_ou = sum(count for sites in panel_structure.values() for count in sites.values())

    assert len(panel) == expected_ou * months
    assert set(panel["region_id"]) == set(panel_structure)

    ou_counts = (
        panel.groupby(["region_id", "site_id"])["ou_code"]
        .nunique()
        .to_dict()
    )
    expected_counts = {
        (region_id, site_id): count
        for region_id, sites in panel_structure.items()
        for site_id, count in sites.items()
    }
    assert ou_counts == expected_counts


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


def test_make_hierarchical_ou_dataset_wave_month_control() -> None:
    panel = make_hierarchical_ou_dataset(
        n_regions=1,
        n_sites_per_region=1,
        n_ous_per_site=1,
        n_years=1,
        wave_months=(3,),
        wave_missing_prob=0.0,
        seed=123,
    )

    march_rows = panel[panel["date"].dt.month == 3]
    non_march_rows = panel[panel["date"].dt.month != 3]

    assert not march_rows.empty
    assert march_rows["survey_score"].notna().all()
    assert march_rows["n_respondents"].notna().all()
    assert non_march_rows["survey_score"].isna().all()
    assert non_march_rows["n_respondents"].isna().all()


def test_aggregate_to_parent_weighted_survey() -> None:
    panel = pd.DataFrame(
        {
            "region_id": ["R1", "R1"],
            "site_id": ["S1", "S1"],
            "ou_code": ["OU1", "OU2"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "productivity": [100.0, 150.0],
            "fte_operational": [10.0, 20.0],
            "survey_score": [60.0, 80.0],
            "n_respondents": [20.0, 10.0],
        }
    )

    aggregated = aggregate_to_parent(panel, level="site")
    assert len(aggregated) == 1

    row = aggregated.iloc[0]
    assert row["productivity"] == pytest.approx(250.0)
    assert row["fte_operational"] == pytest.approx(30.0)
    assert row["n_respondents"] == pytest.approx(30.0)
    expected_weighted_score = (60.0 * 20.0 + 80.0 * 10.0) / 30.0
    assert row["survey_score"] == pytest.approx(expected_weighted_score)


def test_aggregate_to_parent_handles_missing_survey_data() -> None:
    panel = pd.DataFrame(
        {
            "region_id": ["R1", "R1"],
            "site_id": ["S1", "S1"],
            "ou_code": ["OU1", "OU2"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "productivity": [10.0, 20.0],
            "fte_operational": [5.0, 7.0],
            "survey_score": [np.nan, np.nan],
            "n_respondents": [np.nan, np.nan],
        }
    )

    aggregated = aggregate_to_parent(panel, level="region")
    assert len(aggregated) == 1

    row = aggregated.iloc[0]
    assert np.isnan(row["survey_score"])
    assert np.isnan(row["n_respondents"])
    assert row["productivity"] == pytest.approx(30.0)
    assert row["fte_operational"] == pytest.approx(12.0)
