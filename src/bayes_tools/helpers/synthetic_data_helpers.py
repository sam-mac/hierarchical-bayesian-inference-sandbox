import numpy as np
import pandas as pd

def make_hierarchical_ou_dataset(
    n_regions: int = 3,
    n_sites_per_region: int = 3,
    n_ous_per_site: int = 4,
    n_years: int = 3,
    wave_months=(6, 12),
    wave_missing_prob: float = 0.1,# chance a given OU skips a wave (simulate missingness)
    seed: int = 42
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
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=12 * n_years, freq="MS")
    wave_mask = np.isin(months.month, list(wave_months))

    # Group-level effects (log scale so effects multiply on original scale)
    region_eff = {f"R{r+1}": rng.normal(0.0, 0.10) for r in range(n_regions)}
    site_eff = {}  # nested inside regions
    data = []

    for r in range(1, n_regions + 1):
        region_id = f"R{r}"
        for s in range(1, n_sites_per_region + 1):
            site_id = f"S{r}-{s}"
            site_eff[site_id] = rng.normal(0.0, 0.08)

            for u in range(1, n_ous_per_site + 1):
                ou_code = f"OU{r}-{s}-{u}"

                # OU-specific baseline and drift
                base_level = rng.normal(100, 20)  # baseline productivity level
                ou_eff = rng.normal(0.0, 0.07)    # OU deviation (log scale)
                drift = rng.normal(0.003, 0.001)  # slow upward trend

                # Mild seasonality (shared pattern across all groups)
                season = np.sin(np.linspace(0, 2*np.pi, 12))

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
                            nresp[t] = int(np.clip(rng.normal(0.25 * fte[t], 6), 10, 120))

                # Append rows
                for t, month in enumerate(months):
                    data.append({
                        "region_id": region_id,
                        "site_id": site_id,
                        "ou_code": ou_code,
                        "date": month,
                        "productivity": float(prod[t]),
                        "fte_operational": float(fte[t]),
                        "survey_score": (None if np.isnan(survey[t]) else float(survey[t])),
                        "n_respondents": (None if np.isnan(nresp[t]) else int(nresp[t]))
                    })

    df = pd.DataFrame(data)
    # Ensure month start (useful if your downstream code expects this)
    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.to_period("M")
        .to_timestamp(how="start")
    )
    return df


# --- Optional: convenience aggregator to parent levels (site or region) ---

def aggregate_to_parent(
    df: pd.DataFrame,
    level: str = "site"  # "site" or "region"
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

    # Mark wave months where survey exists at least for one OU
    df["is_wave"] = df["survey_score"].notna()

    # Aggregations (monthly)
    def weighted_mean_survey(g):
        # Weighted mean only over rows with survey; else NaN
        mask = g["survey_score"].notna() & g["n_respondents"].notna()
        if not mask.any():
            return np.nan
        return np.average(g.loc[mask, "survey_score"], weights=g.loc[mask, "n_respondents"])

    agg = (
        df.groupby([key, "date"], as_index=False)
          .agg(
              productivity=("productivity", "sum"),
              fte_operational=("fte_operational", "sum"),
              # respondents sum (only at wave rows)
              n_respondents=("n_respondents", lambda x: x.dropna().sum() if x.notna().any() else np.nan),
              # respondent-weighted mean for survey
              survey_score=("survey_score", weighted_mean_survey),
          )
    )

    agg = agg.rename(columns={key: f"{level}_id"})
    return agg
