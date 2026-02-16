"""Top-level package for hierarchical Bayesian LGSSM workflows.

The package is intentionally organized around a practical pipeline:

1. Generate or load grouped time-series data.
2. Define a hierarchical linear Gaussian state-space model (LGSSM).
3. Fit the model with NumPyro and NUTS.
4. Bridge inferred parameters to Dynamax utilities for filtering/smoothing.
"""

from bayes_tools.config import HierarchicalLGSSMConfig
from bayes_tools.data.synthetic import SyntheticDataset, simulate_hierarchical_lgssm
from bayes_tools.models.numpyro_hierarchical_lgssm import hierarchical_lgssm_model

__all__ = [
    "HierarchicalLGSSMConfig",
    "SyntheticDataset",
    "simulate_hierarchical_lgssm",
    "hierarchical_lgssm_model",
]
