import numpy as np
import pymc as pm

from bayes_tools.helpers import (
    StructuralSSMComponents,
    build_grouped_structural_productivity_model,
    build_grouped_structural_ssm,
)


def test_build_grouped_structural_ssm_shapes() -> None:
    n_groups = 2
    n_time = 4
    month_idx_t = np.array([0, 1, 2, 3], dtype="int32")

    with pm.Model():
        ssm: StructuralSSMComponents = build_grouped_structural_ssm(
            "test_x",
            n_groups=n_groups,
            T=n_time,
            phi_trend=0.8,
            rho_g=np.array([0.1, 0.2]),
            sigma_level_g=np.array([0.1, 0.2]),
            sigma_trend_g=np.array([0.05, 0.1]),
            sigma_ar_g=np.array([0.1, 0.15]),
            level0_g=np.array([0.0, 0.0]),
            trend0_g=np.array([0.0, 0.0]),
            ar0_g=np.array([0.0, 0.0]),
            month_idx_t=month_idx_t,
            season_effect_12=np.linspace(-0.2, 0.2, 12),
            season_scale_g=np.array([1.0, 1.5]),
        )

        assert ssm["x"].eval().shape == (n_groups, n_time)
        assert ssm["level"].eval().shape == (n_groups, n_time)
        assert ssm["ar"].eval().shape == (n_groups, n_time)
        assert ssm["season"].eval().shape == (n_groups, n_time)


def test_build_grouped_structural_productivity_model_variables() -> None:
    rng = np.random.default_rng(123)
    n_groups = 2
    n_time = 5
    n_obs = 6

    group_idx = rng.integers(0, n_groups, size=n_obs, dtype="int32")
    time_idx = rng.integers(0, n_time, size=n_obs, dtype="int32")
    month_idx_t = np.arange(n_time, dtype="int32") % 12
    y = rng.normal(size=n_obs).astype("float64")

    model = build_grouped_structural_productivity_model(
        n_groups=n_groups,
        n_time=n_time,
        group_idx=group_idx,
        time_idx=time_idx,
        month_idx_t=month_idx_t,
        y=y,
    )

    assert isinstance(model, pm.Model)
    assert "x" in model.named_vars
    assert "rho_g" in model.named_vars

    y_rv = next((rv for rv in model.observed_RVs if rv.name == "y_obs"), None)
    assert y_rv is not None
    observed_data = np.asarray(y_rv.tag.observations.eval())
    assert observed_data.shape == (n_obs,)
