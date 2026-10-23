import numpy as np
import pytest

from bayes_tools.config import HierarchicalLGSSMConfig
from bayes_tools.models.numpyro_hierarchical_lgssm import hierarchical_lgssm_model
from bayes_tools.utils.dependencies import module_available


def test_model_rejects_bad_observation_shape() -> None:
    config = HierarchicalLGSSMConfig(
        n_groups=2,
        n_timesteps=6,
        state_dim=2,
        emission_dim=1,
    )
    observations = np.zeros((2, 5, 1), dtype=float)

    with pytest.raises(ValueError):
        hierarchical_lgssm_model(observations=observations, config=config)


@pytest.mark.skipif(
    not module_available("numpyro") or not module_available("jax"),
    reason="jax/numpyro are optional dependencies",
)
def test_model_executes_with_valid_shapes() -> None:
    config = HierarchicalLGSSMConfig(
        n_groups=1,
        n_timesteps=4,
        state_dim=2,
        emission_dim=1,
    )
    observations = np.zeros((1, 4, 1), dtype=float)

    from numpyro import handlers

    trace = handlers.trace(
        handlers.seed(hierarchical_lgssm_model, rng_seed=0)
    ).get_trace(observations=observations, config=config)
    assert trace["state_0_0"]["value"].shape == (2,)
    assert trace["obs_0_3"]["is_observed"]


@pytest.mark.skipif(
    not module_available("numpyro") or not module_available("jax"),
    reason="jax/numpyro are optional dependencies",
)
def test_existing_runner_completes_a_tiny_chain() -> None:
    from bayes_tools.inference.numpyro_runner import MCMCRunConfig, run_numpyro_mcmc

    samples = run_numpyro_mcmc(
        np.zeros((1, 3, 1)),
        HierarchicalLGSSMConfig(1, 3, 1, 1),
        MCMCRunConfig(num_warmup=5, num_samples=5, num_chains=1, seed=7),
    )
    assert samples["transition_base"].shape == (5, 1, 1)
    assert all(np.isfinite(value).all() for value in samples.values())
    # Execution smoke only: five draws cannot establish convergence.
