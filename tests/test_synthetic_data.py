import numpy as np

from bayes_tools.config import HierarchicalLGSSMConfig
from bayes_tools.data.synthetic import simulate_hierarchical_lgssm


def test_simulation_shapes_match_configuration() -> None:
    config = HierarchicalLGSSMConfig(
        n_groups=3,
        n_timesteps=10,
        state_dim=2,
        emission_dim=2,
    )

    dataset = simulate_hierarchical_lgssm(config, seed=123)

    assert dataset.states.shape == (3, 10, 2)
    assert dataset.observations.shape == (3, 10, 2)
    assert dataset.transition_matrices.shape == (3, 2, 2)
    assert dataset.emission_matrix.shape == (2, 2)
    assert dataset.process_std.shape == (3, 2)
    assert dataset.emission_std.shape == (3, 2)


def test_simulation_is_reproducible_for_fixed_seed() -> None:
    config = HierarchicalLGSSMConfig(
        n_groups=2,
        n_timesteps=8,
        state_dim=2,
        emission_dim=1,
    )

    dataset_a = simulate_hierarchical_lgssm(config, seed=99)
    dataset_b = simulate_hierarchical_lgssm(config, seed=99)

    np.testing.assert_allclose(dataset_a.states, dataset_b.states)
    np.testing.assert_allclose(dataset_a.observations, dataset_b.observations)
