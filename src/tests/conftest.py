import pandas as pd
import pytest  # type: ignore

from bayes_tools.helpers.synthetic_data_helpers import make_hierarchical_ou_dataset


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Return a small but deterministic hierarchical panel for tests."""
    return make_hierarchical_ou_dataset(
        n_regions=1,
        n_sites_per_region=1,
        n_ous_per_site=2,
        n_years=1,
        wave_months=(6, 12),
        wave_missing_prob=0.0,
        seed=123,
    )
