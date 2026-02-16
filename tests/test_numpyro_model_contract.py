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

    hierarchical_lgssm_model(observations=observations, config=config)
