"""Seeded, finite synthetic panels with explicit generating parameters."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayes_tools.config import HierarchicalLGSSMConfig


@dataclass(frozen=True)
class SyntheticDataset:
    """Observations, latent states, and the parameters that generated them."""

    states: NDArray[np.float64]
    observations: NDArray[np.float64]
    transition_matrices: NDArray[np.float64]
    emission_matrix: NDArray[np.float64]
    process_std: NDArray[np.float64]
    emission_std: NDArray[np.float64]


def simulate_hierarchical_lgssm(
    config: HierarchicalLGSSMConfig,
    *,
    seed: int = 0,
    transition_matrices: NDArray[np.float64] | None = None,
    emission_matrix: NDArray[np.float64] | None = None,
    process_std: NDArray[np.float64] | None = None,
    emission_std: NDArray[np.float64] | None = None,
) -> SyntheticDataset:
    """Simulate x[0] ~ N(0, I), x[t] ~ N(A x[t-1], Q), y ~ N(H x, R).

    Q and R are diagonal. Optional explicit parameters support recovery studies.
    Defaults share a transition baseline across groups and bound each generated
    transition's spectral radius by 0.95. They are convenient test parameters,
    not prior-predictive draws from the unconstrained NumPyro model.
    """
    config.validate()
    rng = np.random.default_rng(seed)
    groups, timesteps = config.n_groups, config.n_timesteps
    states_dim, emissions_dim = config.state_dim, config.emission_dim

    if transition_matrices is None:
        base = 0.65 * np.eye(states_dim)
        transition_matrices = base + rng.normal(
            0.0, config.transition_scale_prior, (groups, states_dim, states_dim)
        )
        radii = np.max(np.abs(np.linalg.eigvals(transition_matrices)), axis=1)
        transition_matrices /= np.maximum(1.0, radii / 0.95)[:, None, None]
    if emission_matrix is None:
        emission_matrix = np.eye(emissions_dim, states_dim)
    if process_std is None:
        process_std = np.exp(
            rng.normal(-1.0, config.process_noise_scale_prior, (groups, states_dim))
        )
    if emission_std is None:
        emission_std = np.exp(
            rng.normal(-0.5, config.emission_noise_scale_prior, (groups, emissions_dim))
        )

    arrays = []
    for name, value, shape, positive in [
        (
            "transition_matrices",
            transition_matrices,
            (groups, states_dim, states_dim),
            False,
        ),
        ("emission_matrix", emission_matrix, (emissions_dim, states_dim), False),
        ("process_std", process_std, (groups, states_dim), True),
        ("emission_std", emission_std, (groups, emissions_dim), True),
    ]:
        array = np.array(value, dtype=np.float64, copy=True)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite and have shape {shape}")
        if positive and np.any(array <= 0):
            raise ValueError(f"{name} must be strictly positive")
        arrays.append(array)
    transitions, emissions, q_std, r_std = arrays

    states = np.empty((groups, timesteps, states_dim), dtype=np.float64)
    states[:, 0] = rng.normal(size=(groups, states_dim))
    for t in range(1, timesteps):
        states[:, t] = np.einsum("gij,gj->gi", transitions, states[:, t - 1])
        states[:, t] += rng.normal(size=(groups, states_dim)) * q_std
    observations = np.einsum("es,gts->gte", emissions, states)
    observations += rng.normal(size=observations.shape) * r_std[:, None, :]
    return SyntheticDataset(states, observations, transitions, emissions, q_std, r_std)
