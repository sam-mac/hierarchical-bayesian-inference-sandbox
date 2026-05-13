from collections.abc import Sequence

import numpy as np
import pandas as pd


def preprocess_lgssm_dataframe(
    df: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    frequency: str = "MS",
    interpolate_limit_direction: str = "both",
) -> pd.DataFrame:
    """Clean a univariate time series into a regular panel suited for LGSSM testing.

    The preprocessing keeps an indicator for missing original observations, imputes
    gaps with time-based interpolation, and adds common transformed features used
    when validating local-level/linear-trend style models.
    """
    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column], utc=False)
    working = working[[date_column, value_column]].dropna(subset=[date_column])

    grouped = (
        working.groupby(date_column, as_index=False)[value_column]
        .mean()
        .sort_values(date_column)
    )

    full_index = pd.date_range(
        start=grouped[date_column].min(),
        end=grouped[date_column].max(),
        freq=frequency,
    )

    regular = grouped.set_index(date_column).reindex(full_index)
    regular.index.name = "date"
    regular = regular.rename(columns={value_column: "y_raw"})
    regular["is_observed"] = regular["y_raw"].notna().astype(int)

    regular["y"] = regular["y_raw"].interpolate(
        method="time", limit_direction=interpolate_limit_direction
    )

    if regular["y"].isna().any():
        regular["y"] = regular["y"].ffill().bfill()

    regular["time_idx"] = np.arange(len(regular), dtype=float)
    regular["y_centered"] = regular["y"] - regular["y"].mean()

    y_std = regular["y"].std(ddof=0)
    if y_std > 0:
        regular["y_standardized"] = regular["y_centered"] / y_std
    else:
        regular["y_standardized"] = 0.0

    linear_fit = np.polyfit(regular["time_idx"], regular["y"], deg=1)
    regular["deterministic_trend"] = np.polyval(linear_fit, regular["time_idx"])
    regular["detrended_residual"] = regular["y"] - regular["deterministic_trend"]
    regular["lag1_residual"] = regular["detrended_residual"].shift(1)

    return regular.reset_index()


def make_synthetic_local_linear_trend_dataset(
    *,
    n_periods: int = 180,
    start_date: str = "2010-01-01",
    frequency: str = "MS",
    initial_level: float = 100.0,
    initial_trend: float = 0.2,
    level_noise_sd: float = 0.4,
    trend_noise_sd: float = 0.05,
    ar_coefficient: float = 0.6,
    ar_noise_sd: float = 1.2,
    measurement_noise_sd: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate data for a local-level + linear-trend + AR(1) residual process."""
    if n_periods < 3:
        msg = "n_periods must be at least 3"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_periods, freq=frequency)

    level = np.zeros(n_periods, dtype=float)
    trend = np.zeros(n_periods, dtype=float)
    ar_residual = np.zeros(n_periods, dtype=float)

    level[0] = initial_level
    trend[0] = initial_trend

    for t in range(1, n_periods):
        trend[t] = trend[t - 1] + rng.normal(0.0, trend_noise_sd)
        level[t] = level[t - 1] + trend[t - 1] + rng.normal(0.0, level_noise_sd)
        ar_residual[t] = (
            ar_coefficient * ar_residual[t - 1] + rng.normal(0.0, ar_noise_sd)
        )

    signal = level + ar_residual
    observed = signal + rng.normal(0.0, measurement_noise_sd, size=n_periods)

    simulated = pd.DataFrame(
        {
            "date": dates,
            "y": observed,
            "latent_level": level,
            "latent_trend": trend,
            "ar_residual": ar_residual,
            "signal": signal,
        }
    )

    prepared = preprocess_lgssm_dataframe(
        simulated,
        date_column="date",
        value_column="y",
        frequency=frequency,
    )

    return prepared.merge(simulated, on="date", how="left", suffixes=("", "_sim"))


def download_univariate_time_series_csv(
    *,
    url: str,
    date_column: str,
    value_column: str,
    frequency: str = "MS",
    read_csv_kwargs: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Download and preprocess a univariate CSV time series for LGSSM experiments."""
    csv_kwargs = read_csv_kwargs.copy() if read_csv_kwargs is not None else {}
    downloaded = pd.read_csv(url, **csv_kwargs)
    return preprocess_lgssm_dataframe(
        downloaded,
        date_column=date_column,
        value_column=value_column,
        frequency=frequency,
    )


def download_airline_passengers_dataset(
    *,
    frequency: str = "MS",
    extra_transforms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Download and preprocess the classic airline-passengers monthly dataset.

    Parameters
    ----------
    frequency
        Expected output frequency.
    extra_transforms
        Optional transforms to add. Supported values: ``"log"``,
        ``"diff"``, ``"seasonal_diff_12"``.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    prepared = download_univariate_time_series_csv(
        url=url,
        date_column="Month",
        value_column="Passengers",
        frequency=frequency,
    )

    transforms = set(extra_transforms or [])
    if "log" in transforms:
        prepared["y_log"] = np.log(prepared["y"])
    if "diff" in transforms:
        prepared["y_diff_1"] = prepared["y"].diff()
    if "seasonal_diff_12" in transforms:
        prepared["y_diff_12"] = prepared["y"].diff(12)

    return prepared
