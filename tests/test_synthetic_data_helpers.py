import numpy as np
import pandas as pd
import pytest

from bayes_tools.helpers.synthetic_data_helpers import (
    aggregate_to_parent,
    make_hierarchical_ou_dataset,
)


def test_aggregate_to_parent_weighted_survey():
    df = make_hierarchical_ou_dataset(
        n_regions=1,
        n_sites_per_region=1,
        n_ous_per_site=3,
        n_years=1,
        wave_missing_prob=0.0,
        seed=123,
    )

    aggregated = aggregate_to_parent(df, level="site")

    required_columns = {
        "site_id",
        "date",
        "productivity",
        "fte_operational",
        "n_respondents",
        "survey_score",
    }
    assert required_columns.issubset(aggregated.columns)

    available = aggregated[aggregated["survey_score"].notna()]
    assert not available.empty, "Expected at least one survey wave after aggregation"
    with_survey = available.iloc[0]
    site_id = with_survey["site_id"]
    month = pd.Timestamp(with_survey["date"])

    mask = (
        (df["site_id"] == site_id)
        & (df["date"] == month)
        & df["survey_score"].notna()
        & df["n_respondents"].notna()
    )

    expected_survey = np.average(
        df.loc[mask, "survey_score"],
        weights=df.loc[mask, "n_respondents"],
    )
    expected_respondents = df.loc[mask, "n_respondents"].sum()

    assert with_survey["survey_score"] == pytest.approx(expected_survey)
    assert with_survey["n_respondents"] == pytest.approx(expected_respondents)
