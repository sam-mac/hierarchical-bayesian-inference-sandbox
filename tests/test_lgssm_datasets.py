import numpy as np
import pandas as pd

from bayes_tools.datasets.lgssm_datasets import (
    download_airline_passengers_dataset,
    make_synthetic_local_linear_trend_dataset,
    preprocess_lgssm_dataframe,
)


def test_preprocess_lgssm_dataframe_regularises_and_imputes() -> None:
    raw = pd.DataFrame(
        {
            "date_col": [
                "2022-01-01",
                "2022-01-01",
                "2022-03-01",
                "2022-05-01",
            ],
            "value_col": [10.0, 12.0, 20.0, 30.0],
        }
    )

    prepared = preprocess_lgssm_dataframe(
        raw,
        date_column="date_col",
        value_column="value_col",
        frequency="MS",
    )

    assert len(prepared) == 5
    assert list(prepared["is_observed"]) == [1, 0, 1, 0, 1]
    assert prepared["y"].notna().all()
    assert "lag1_residual" in prepared.columns
    assert np.isnan(prepared.loc[0, "lag1_residual"])


def test_make_synthetic_local_linear_trend_dataset_is_reproducible() -> None:
    first = make_synthetic_local_linear_trend_dataset(seed=101, n_periods=24)
    second = make_synthetic_local_linear_trend_dataset(seed=101, n_periods=24)

    pd.testing.assert_frame_equal(first, second)
    expected_columns = {
        "date",
        "y",
        "latent_level",
        "latent_trend",
        "ar_residual",
        "signal",
        "is_observed",
        "y_centered",
        "y_standardized",
        "deterministic_trend",
        "detrended_residual",
        "lag1_residual",
    }
    assert expected_columns.issubset(first.columns)
    assert first["is_observed"].eq(1).all()


def test_download_airline_passengers_dataset_uses_preprocessing(
    monkeypatch,
) -> None:
    fake_downloaded = pd.DataFrame(
        {
            "Month": ["1949-01-01", "1949-02-01", "1949-03-01"],
            "Passengers": [112, 118, 132],
        }
    )

    def fake_read_csv(url: str, **kwargs: object) -> pd.DataFrame:
        del url, kwargs
        return fake_downloaded

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    prepared = download_airline_passengers_dataset(extra_transforms=["log", "diff"])

    assert len(prepared) == 3
    assert prepared["y"].iloc[0] == 112
    assert "y_log" in prepared.columns
    assert "y_diff_1" in prepared.columns
