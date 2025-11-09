from collections.abc import Mapping

import numpy as np
import pandas as pd


def make_hierarchical_ou_dataset(
    n_regions: int = 3,
    n_sites_per_region: int = 3,
    n_ous_per_site: int = 4,
    n_years: int = 3,
    wave_months=(6, 12),
    wave_missing_prob: float = 0.1,  # chance a given OU skips a wave (simulate missingness)
    seed: int = 42,
    panel_structure: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    """
    Build a monthly hierarchical panel:
      Region -> Site -> OU
      - productivity: monthly, complete (positive)
      - FTE: monthly, correlated with productivity
      - survey_score: observed only at wave months, often missing otherwise (+ optional random missingness)
      - n_respondents: only where survey exists
    Columns:
      region_id, site_id, ou_code, date, productivity, fte_operational,
      survey_score, n_respondents
    Args:
        panel_structure: Optional explicit hierarchy specifying the number of
            operating units (OUs) per site. The mapping should follow the
            structure {region_id: {site_id: ou_count}}. When provided, the
            ``n_regions``, ``n_sites_per_region`` and ``n_ous_per_site``
            parameters are ignored.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=12 * n_years, freq="MS")
    wave_mask = np.isin(months.month, list(wave_months))

    if panel_structure is None:
        structure: dict[str, dict[str, int]] = {
            f"R{r}": {f"S{r}-{s}": n_ous_per_site for s in range(1, n_sites_per_region + 1)}
            for r in range(1, n_regions + 1)
        }
    else:
        structure = {
            region_id: {site_id: int(count) for site_id, count in sites.items()}
            for region_id, sites in panel_structure.items()
        }

    # Validate the requested structure.
    if not structure:
        msg = "panel_structure must define at least one region"
        raise ValueError(msg)

    for region_id, sites in structure.items():
        if not sites:
            msg = f"Region '{region_id}' must define at least one site"
            raise ValueError(msg)
        for site_id, ou_count in sites.items():
            if ou_count < 1:
                msg = f"Site '{site_id}' must specify at least one OU"
                raise ValueError(msg)

    # Group-level effects (log scale so effects multiply on original scale)
    region_eff = {
        region_id: rng.normal(0.0, 0.10)
        for region_id in structure
    }
    site_eff: dict[str, float] = {}
    data = []

    for r_index, (region_id, sites) in enumerate(structure.items(), start=1):
        for s_index, (site_id, ou_count) in enumerate(sites.items(), start=1):
            site_eff[site_id] = rng.normal(0.0, 0.08)

            for u_index in range(1, ou_count + 1):
                if panel_structure is None:
                    ou_code = f"OU{r_index}-{s_index}-{u_index}"
                else:
                    ou_code = f"{site_id}-OU{u_index}"

                # OU-specific baseline and drift
                base_level = rng.normal(100, 20)  # baseline productivity level
                ou_eff = rng.normal(0.0, 0.07)  # OU deviation (log scale)
                drift = rng.normal(0.003, 0.001)  # slow upward trend

                # Mild seasonality (shared pattern across all groups)
                season = np.sin(np.linspace(0, 2 * np.pi, 12))

                prod = np.empty(len(months))
                fte = np.empty(len(months))
                survey = np.full(len(months), np.nan)
                nresp = np.full(len(months), np.nan)

                # Generate monthly productivity (log-linear with hierarchical effects)
                for t, month in enumerate(months):
                    m_idx = month.month - 1
                    eps = rng.normal(0, 0.05)
                    log_prod = (
                        np.log(base_level)
                        + region_eff[region_id]
                        + site_eff[site_id]
                        + ou_eff
                        + drift * t
                        + 0.12 * season[m_idx]
                        + eps
                    )
                    prod[t] = max(np.exp(log_prod), 1e-6)  # ensure positive

                    # FTE correlates with productivity (plus OU offset noise)
                    fte[t] = np.clip(rng.normal(40 + 0.15 * prod[t], 6), 5, None)

                # Survey only at waves; allow some waves to be missing
                # Survey loosely tracks productivity + noise
                for t, month in enumerate(months):
                    if wave_mask[t]:
                        if rng.random() > wave_missing_prob:
                            survey[t] = (
                                55
                                + 0.25 * (prod[t] / prod.mean()) * 10
                                + rng.normal(0, 2.5)
                            )
                            # Respondents depend on FTE (cap and ensure integer)
                            nresp[t] = int(
                                np.clip(rng.normal(0.25 * fte[t], 6), 10, 120)
                            )

                # Append rows
                for t, month in enumerate(months):
                    data.append(
                        {
                            "region_id": region_id,
                            "site_id": site_id,
                            "ou_code": ou_code,
                            "date": month,
                            "productivity": float(prod[t]),
                            "fte_operational": float(fte[t]),
                            "survey_score": (
                                None if np.isnan(survey[t]) else float(survey[t])
                            ),
                            "n_respondents": (
                                None if np.isnan(nresp[t]) else int(nresp[t])
                            ),
                        }
                    )

    df = pd.DataFrame(data)
    # Ensure month start (useful if your downstream code expects this)
    df["date"] = (
        pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp(how="start")
    )
    return df


# --- Optional: convenience aggregator to parent levels (site or region) ---


def aggregate_to_parent(
    df: pd.DataFrame, level: str = "site"  # "site" or "region"
) -> pd.DataFrame:
    """
    Aggregate OU-level monthly data to site or region level.
    - productivity: sum (typical for throughput-like measures; change to 'mean' if that fits your domain)
    - fte_operational: sum
    - survey_score: respondent-weighted mean at wave months; NaN otherwise
    - n_respondents: sum at wave months; NaN otherwise
    Returns a new monthly panel at the requested level with analogous columns.
    """
    if level not in {"site", "region"}:
        raise ValueError("level must be 'site' or 'region'")

    key = "site_id" if level == "site" else "region_id"
    df = df.copy()

    def summarise(group: pd.DataFrame) -> pd.Series:
        respondents = group["n_respondents"].dropna()
        respondents_sum = respondents.sum() if not respondents.empty else np.nan

        mask = group["survey_score"].notna() & group["n_respondents"].notna()
        if mask.any():
            survey = float(
                np.average(
                    group.loc[mask, "survey_score"],
                    weights=group.loc[mask, "n_respondents"],
                )
            )
        else:
            survey = np.nan

        return pd.Series(
            {
                "productivity": group["productivity"].sum(),
                "fte_operational": group["fte_operational"].sum(),
                "n_respondents": respondents_sum,
                "survey_score": survey,
            }
        )

    agg = (
        df.groupby([key, "date"], sort=False)
        .apply(summarise, include_groups=False) # pyright: ignore[reportCallIssue]
        .reset_index()
    )
    return agg.rename(columns={key: f"{level}_id"})
