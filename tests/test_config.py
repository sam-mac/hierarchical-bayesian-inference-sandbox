import pytest

from bayes_tools.config import HierarchicalLGSSMConfig


def test_validate_accepts_valid_configuration() -> None:
    config = HierarchicalLGSSMConfig(
        n_groups=2,
        n_timesteps=5,
        state_dim=2,
        emission_dim=1,
    )
    config.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_groups": 0, "n_timesteps": 5, "state_dim": 2, "emission_dim": 1},
        {"n_groups": 2, "n_timesteps": 1, "state_dim": 2, "emission_dim": 1},
        {"n_groups": 2, "n_timesteps": 5, "state_dim": 0, "emission_dim": 1},
        {"n_groups": 2, "n_timesteps": 5, "state_dim": 2, "emission_dim": 0},
    ],
)
def test_validate_rejects_invalid_dimensions(kwargs: dict[str, int]) -> None:
    config = HierarchicalLGSSMConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()
