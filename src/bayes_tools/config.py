"""Configuration models for hierarchical LGSSM experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalLGSSMConfig:
    """Static dimensionality and prior scale settings.

    Attributes
    ----------
    n_groups:
        Number of independent panels/groups.
    n_timesteps:
        Number of timesteps in each group.
    state_dim:
        Latent state dimensionality.
    emission_dim:
        Observed emission dimensionality.
    transition_scale_prior:
        Scale for group-level transition perturbations in non-centered form.
    process_noise_scale_prior:
        Scale for log process noise hierarchy.
    emission_noise_scale_prior:
        Scale for log emission noise hierarchy.
    """

    n_groups: int
    n_timesteps: int
    state_dim: int
    emission_dim: int
    transition_scale_prior: float = 0.25
    process_noise_scale_prior: float = 0.5
    emission_noise_scale_prior: float = 0.5

    def validate(self) -> None:
        """Raise ``ValueError`` if any dimension or scale is invalid."""
        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")
        if self.n_timesteps <= 1:
            raise ValueError("n_timesteps must be greater than one")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.emission_dim <= 0:
            raise ValueError("emission_dim must be positive")
        if self.transition_scale_prior <= 0:
            raise ValueError("transition_scale_prior must be positive")
        if self.process_noise_scale_prior <= 0:
            raise ValueError("process_noise_scale_prior must be positive")
        if self.emission_noise_scale_prior <= 0:
            raise ValueError("emission_noise_scale_prior must be positive")
