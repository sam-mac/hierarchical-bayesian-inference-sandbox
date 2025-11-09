import pandas as pd
from bayes_tools.helpers.synthetic_data_helpers import make_hierarchical_ou_dataset


def get_default_balanced_hierarchical_dataset() -> pd.DataFrame:
    return make_hierarchical_ou_dataset(
        n_regions=1,
        n_sites_per_region=4,
        n_ous_per_site=2,
        n_years=1,
        wave_months=(6, 12),
        wave_missing_prob=0.0,
    )
