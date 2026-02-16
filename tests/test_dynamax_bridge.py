import numpy as np

from bayes_tools.inference.dynamax_bridge import posterior_means_to_dynamax_params


def test_posterior_conversion_produces_diagonal_covariances() -> None:
    transition = np.array([[[0.9, 0.1], [0.0, 0.8]]], dtype=float)
    emission = np.array([[1.2, -0.2]], dtype=float)
    process_std = np.array([[0.3, 0.4]], dtype=float)
    emission_std = np.array([[0.2]], dtype=float)

    params = posterior_means_to_dynamax_params(
        transition_matrices=transition,
        emission_matrix=emission,
        process_std=process_std,
        emission_std=emission_std,
        group_idx=0,
    )

    np.testing.assert_allclose(params.transition_covariance, np.diag([0.09, 0.16]))
    np.testing.assert_allclose(params.emission_covariance, np.diag([0.04]))
    np.testing.assert_allclose(params.initial_mean, np.zeros(2))
    np.testing.assert_allclose(params.initial_covariance, np.eye(2))
