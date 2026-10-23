import numpy as np
import pytest

from bayes_tools.models.scalar_lgssm import forecast_scalar_lgssm
from bayes_tools.utils.dependencies import module_available


def test_forecast_first_step_matches_hand_calculation() -> None:
    # One observed value: x|y has mean .8 and variance .2.
    predictive, means, variances = forecast_scalar_lgssm(
        np.array([[1.0]]),
        np.array([[0.5]]),
        np.array([0.3]),
        np.array([0.5]),
        horizon=2,
        seed=4,
    )
    np.testing.assert_allclose(means, [[[0.4, 0.2]]])
    np.testing.assert_allclose(variances, [[[0.39, 0.375]]])
    assert predictive.shape == (1, 1, 2)
    repeat, _, _ = forecast_scalar_lgssm(
        np.array([[1.0]]),
        np.array([[0.5]]),
        np.array([0.3]),
        np.array([0.5]),
        horizon=2,
        seed=4,
    )
    np.testing.assert_array_equal(predictive, repeat)


@pytest.mark.skipif(not module_available("jax"), reason="jax is optional")
def test_kalman_likelihood_matches_dense_gaussian() -> None:
    from bayes_tools.models.scalar_lgssm import scalar_kalman_log_likelihood

    y = np.array([[0.2, -0.1, 0.4, 0.3], [0.6, 0.2, -0.3, 0.1]])
    phi, q, r = np.array([0.7, -0.3]), np.array([0.4, 0.6]), np.array([0.2, 0.3])
    dense = 0.0
    for group in range(2):
        # x = L z with independent innovations Var(z) = [1, q², ...].
        l_matrix = np.zeros((4, 4))
        for t in range(4):
            for s in range(t + 1):
                l_matrix[t, s] = phi[group] ** (t - s)
        cov = l_matrix @ np.diag([1.0, *([q[group] ** 2] * 3)]) @ l_matrix.T
        cov += r[group] ** 2 * np.eye(4)
        dense += -0.5 * (
            4 * np.log(2 * np.pi)
            + np.linalg.slogdet(cov)[1]
            + y[group] @ np.linalg.solve(cov, y[group])
        )
    actual = scalar_kalman_log_likelihood(y, phi, q, r)
    np.testing.assert_allclose(actual, dense, rtol=1e-6)


@pytest.mark.skipif(not module_available("numpyro"), reason="numpyro is optional")
def test_scalar_model_has_finite_density_and_expected_sites() -> None:
    from numpyro import handlers
    from bayes_tools.models.scalar_lgssm import hierarchical_scalar_lgssm_model

    trace = handlers.trace(
        handlers.seed(hierarchical_scalar_lgssm_model, rng_seed=0)
    ).get_trace(np.zeros((3, 5)), np.full(3, 0.4), np.full(3, 0.2))
    assert trace["transition"]["value"].shape == (3,)
    assert np.all(np.abs(trace["transition"]["value"]) < 1)
    assert np.isfinite(trace["observations"]["fn"].log_factor).all()


@pytest.mark.skipif(not module_available("jax"), reason="jax is optional")
def test_zero_transition_boundary_and_likelihood_gradient() -> None:
    import jax
    import jax.numpy as jnp
    from bayes_tools.models.scalar_lgssm import scalar_kalman_log_likelihood

    y = np.array([[0.2, -0.1, 0.4]])
    q, r = np.array([0.4]), np.array([0.2])
    variance = np.array(
        [[1.0 + r[0] ** 2, q[0] ** 2 + r[0] ** 2, q[0] ** 2 + r[0] ** 2]]
    )
    expected = -0.5 * np.sum(np.log(2 * np.pi * variance) + y**2 / variance)
    np.testing.assert_allclose(
        scalar_kalman_log_likelihood(y, np.zeros(1), q, r), expected, rtol=1e-6
    )
    gradient = jax.grad(lambda phi: scalar_kalman_log_likelihood(y, phi, q, r))(
        jnp.array([0.3])
    )
    delta = 0.002
    finite_difference = (
        scalar_kalman_log_likelihood(y, np.array([0.3 + delta]), q, r)
        - scalar_kalman_log_likelihood(y, np.array([0.3 - delta]), q, r)
    ) / (2 * delta)
    np.testing.assert_allclose(gradient[0], finite_difference, atol=1e-3)


def test_forecast_rejects_invalid_shapes_and_noise() -> None:
    with pytest.raises(ValueError, match="positive"):
        forecast_scalar_lgssm(
            np.zeros((1, 3)),
            np.zeros((2, 1)),
            np.array([0.0]),
            np.array([0.2]),
            horizon=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="incompatible"):
        forecast_scalar_lgssm(
            np.zeros((1, 3)),
            np.zeros((2, 2)),
            np.array([0.4]),
            np.array([0.2]),
            horizon=2,
            seed=1,
        )


@pytest.mark.parametrize(
    "observations,process_std,emission_std",
    [
        (np.zeros((0, 3)), np.empty(0), np.empty(0)),
        (np.zeros((1, 1)), np.array([0.4]), np.array([0.2])),
        (np.zeros((1, 3)), np.array([-0.4]), np.array([0.2])),
        (np.zeros((1, 3)), np.array([0.4]), np.array([0.0])),
        (np.full((1, 3), np.nan), np.array([0.4]), np.array([0.2])),
        (np.zeros((1, 3)), np.array([0.4, 0.4]), np.array([0.2])),
    ],
)
def test_host_validator_rejects_invalid_inputs(
    observations, process_std, emission_std
) -> None:
    from bayes_tools.models.scalar_lgssm import validate_scalar_lgssm_inputs

    with pytest.raises(ValueError):
        validate_scalar_lgssm_inputs(observations, process_std, emission_std)


@pytest.mark.skipif(not module_available("jax"), reason="jax is optional")
def test_likelihood_accepts_integer_zero_transition() -> None:
    from bayes_tools.models.scalar_lgssm import scalar_kalman_log_likelihood

    value = scalar_kalman_log_likelihood(
        np.zeros((1, 3)), np.array([0]), np.array([1]), np.array([1])
    )
    assert np.isfinite(value)
