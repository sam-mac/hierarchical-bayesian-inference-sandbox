"""Compatibility bridge to Dynamax LGSSM parameter structures."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DynamaxLGSSMParams:
    """Lightweight parameter bundle compatible with Dynamax conventions."""

    transition_matrix: NDArray[np.float64]
    transition_covariance: NDArray[np.float64]
    emission_matrix: NDArray[np.float64]
    emission_covariance: NDArray[np.float64]
    initial_mean: NDArray[np.float64]
    initial_covariance: NDArray[np.float64]


def posterior_means_to_dynamax_params(
    transition_matrices: NDArray[np.float64],
    emission_matrix: NDArray[np.float64],
    process_std: NDArray[np.float64],
    emission_std: NDArray[np.float64],
    group_idx: int,
) -> DynamaxLGSSMParams:
    """Convert posterior mean arrays into a single-group LGSSM parameter set."""
    transition_matrix = np.asarray(transition_matrices[group_idx], dtype=np.float64)
    process_diag = np.asarray(process_std[group_idx], dtype=np.float64) ** 2
    emission_diag = np.asarray(emission_std[group_idx], dtype=np.float64) ** 2

    state_dim = transition_matrix.shape[0]
    return DynamaxLGSSMParams(
        transition_matrix=transition_matrix,
        transition_covariance=np.diag(process_diag),
        emission_matrix=np.asarray(emission_matrix, dtype=np.float64),
        emission_covariance=np.diag(emission_diag),
        initial_mean=np.zeros(state_dim, dtype=np.float64),
        initial_covariance=np.eye(state_dim, dtype=np.float64),
    )
