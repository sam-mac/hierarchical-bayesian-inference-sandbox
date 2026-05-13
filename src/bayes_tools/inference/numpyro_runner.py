"""Inference entry points for NumPyro-powered hierarchical LGSSMs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayes_tools.config import HierarchicalLGSSMConfig
from bayes_tools.models.numpyro_hierarchical_lgssm import hierarchical_lgssm_model


@dataclass(frozen=True)
class MCMCRunConfig:
    """Execution settings for MCMC."""

    num_warmup: int = 200
    num_samples: int = 400
    num_chains: int = 1
    seed: int = 0


def run_numpyro_mcmc(
    observations: NDArray[np.float64],
    model_config: HierarchicalLGSSMConfig,
    mcmc_config: MCMCRunConfig | None = None,
) -> dict[str, Any]:
    """Run NUTS for the hierarchical LGSSM and return posterior samples."""
    if mcmc_config is None:
        mcmc_config = MCMCRunConfig()

    try:
        from jax import random
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover
        raise ImportError("run_numpyro_mcmc requires jax and numpyro") from exc

    kernel = NUTS(hierarchical_lgssm_model)
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_config.num_warmup,
        num_samples=mcmc_config.num_samples,
        num_chains=mcmc_config.num_chains,
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(mcmc_config.seed),
        observations=observations,
        config=model_config,
    )
    return mcmc.get_samples()
