"""NumPyro model definitions for hierarchical LGSSMs."""

import numpy as np
from numpy.typing import NDArray

from bayes_tools.config import HierarchicalLGSSMConfig


def hierarchical_lgssm_model(
    observations: NDArray[np.float64],
    config: HierarchicalLGSSMConfig,
) -> None:
    """Hierarchical non-centered LGSSM in NumPyro."""
    config.validate()
    expected_shape = (config.n_groups, config.n_timesteps, config.emission_dim)
    if observations.shape != expected_shape:
        raise ValueError(f"observations shape must be {expected_shape}")

    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hierarchical_lgssm_model requires jax and numpyro to be installed"
        ) from exc

    obs = jnp.asarray(observations)
    state_dim = config.state_dim
    emission_dim = config.emission_dim

    transition_base = numpyro.sample(
        "transition_base",
        dist.Normal(0.0, 0.4).expand([state_dim, state_dim]),
    )
    transition_scale = numpyro.sample(
        "transition_scale",
        dist.HalfNormal(config.transition_scale_prior).expand([state_dim, state_dim]),
    )
    transition_raw = numpyro.sample(
        "transition_raw",
        dist.Normal(0.0, 1.0).expand([config.n_groups, state_dim, state_dim]),
    )
    transition_matrix = transition_base[None, :, :] + transition_scale[None, :, :] * transition_raw

    emission_matrix = numpyro.sample(
        "emission_matrix",
        dist.Normal(0.0, 1.0).expand([emission_dim, state_dim]),
    )

    process_log_mean = numpyro.sample(
        "process_log_mean", dist.Normal(-1.0, 0.8).expand([state_dim])
    )
    process_log_scale = numpyro.sample(
        "process_log_scale",
        dist.HalfNormal(config.process_noise_scale_prior).expand([state_dim]),
    )
    process_raw = numpyro.sample(
        "process_raw",
        dist.Normal(0.0, 1.0).expand([config.n_groups, state_dim]),
    )
    process_std = jnp.exp(process_log_mean[None, :] + process_log_scale[None, :] * process_raw)

    emission_log_mean = numpyro.sample(
        "emission_log_mean", dist.Normal(-0.5, 0.8).expand([emission_dim])
    )
    emission_log_scale = numpyro.sample(
        "emission_log_scale",
        dist.HalfNormal(config.emission_noise_scale_prior).expand([emission_dim]),
    )
    emission_raw = numpyro.sample(
        "emission_raw",
        dist.Normal(0.0, 1.0).expand([config.n_groups, emission_dim]),
    )
    emission_std = jnp.exp(
        emission_log_mean[None, :] + emission_log_scale[None, :] * emission_raw
    )

    for group_idx in range(config.n_groups):
        q_std = process_std[group_idx]
        r_std = emission_std[group_idx]
        transition = transition_matrix[group_idx]

        state = numpyro.sample(
            f"state_{group_idx}_0",
            dist.Normal(0.0, 1.0).expand([state_dim]).to_event(1),
        )
        y0_mean = emission_matrix @ state
        numpyro.sample(
            f"obs_{group_idx}_0", dist.Normal(y0_mean, r_std).to_event(1), obs=obs[group_idx, 0]
        )

        for timestep in range(1, config.n_timesteps):
            state_mean = transition @ state
            state = numpyro.sample(
                f"state_{group_idx}_{timestep}",
                dist.Normal(state_mean, q_std).to_event(1),
            )
            y_mean = emission_matrix @ state
            numpyro.sample(
                f"obs_{group_idx}_{timestep}",
                dist.Normal(y_mean, r_std).to_event(1),
                obs=obs[group_idx, timestep],
            )
