import numpy as np

from bayes_tools.config import HierarchicalLGSSMConfig
from bayes_tools.datasets.synthetic import simulate_hierarchical_lgssm


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


def test_simulation_accepts_explicit_parameters_and_validates_shapes() -> None:
    import pytest

    config = HierarchicalLGSSMConfig(2, 6, 1, 1)
    transition = np.array([[[0.3]], [[0.7]]])
    dataset = simulate_hierarchical_lgssm(
        config, seed=2, transition_matrices=transition
    )
    np.testing.assert_array_equal(dataset.transition_matrices, transition)
    assert np.isfinite(dataset.observations).all()
    with pytest.raises(ValueError, match="transition_matrices"):
        simulate_hierarchical_lgssm(config, transition_matrices=np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="strictly positive"):
        simulate_hierarchical_lgssm(config, process_std=np.zeros((2, 1)))


def test_default_transitions_are_stable() -> None:
    dataset = simulate_hierarchical_lgssm(
        HierarchicalLGSSMConfig(10, 20, 3, 2), seed=32
    )
    assert (
        np.max(np.abs(np.linalg.eigvals(dataset.transition_matrices))) <= 0.95 + 1e-12
    )
