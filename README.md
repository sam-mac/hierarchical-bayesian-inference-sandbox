# Hierarchical Bayesian Inference Sandbox

A small, testable sandbox for linear Gaussian state-space models (LGSSMs):

- Seeded synthetic panels and pandas time-series preprocessing.
- An experimental non-centered, matrix-valued NumPyro model.
- A deliberately identified scalar model with an exact Kalman likelihood and a
  runnable fitted Bayesian example.
- A lightweight posterior-parameter conversion bundle following Dynamax
  conventions. This bundle is not itself a native Dynamax parameter object.

## Clean-checkout setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/sam-mac/hierarchical-bayesian-inference-sandbox.git
cd hierarchical-bayesian-inference-sandbox
uv sync --locked --python 3.13 --extra dev
uv run --no-sync pytest
```

The base install includes NumPy and pandas. It does not install JAX, NumPyro, or
Dynamax. Optional inference tests explicitly skip in this environment. `--locked`
checks that `pyproject.toml` and `uv.lock` agree; it never silently updates the lock.

For inference, including the fitted example:

```bash
uv sync --locked --python 3.13 --extra dev --extra inference
uv run --no-sync pytest
uv run --no-sync python examples/fit_scalar_lgssm.py --strict
```

Use `--no-sync` for these run commands so that the already-selected extras stay
installed. All tests are in `tests/`; dataset-download tests are mocked and need
no dataset network access. A GitHub Actions workflow defines both base and
inference-enabled checks, including the fitted example. The workflow still needs
to run on GitHub after these changes are published.

## One actual fitted example

The example fits four related scalar time series with NumPyro NUTS, using 100
training time points per group and withholding the next 20. It marginalizes the
latent states exactly via a Kalman filter. Four sequential chains use 1,000 warmup
steps and 1,000 retained draws each, double precision, and fixed independent seeds
for simulation, inference, and prediction. A single CPU device is sufficient.

```bash
uv run --no-sync python examples/fit_scalar_lgssm.py \
  --strict --output analysis/output/fitted_lgssm.json
```

The JSON records settings, installed versions, per-site split Rhat/ESS,
divergences, transition recovery intervals, and genuinely held-out forecast
metrics. `--strict` exits unsuccessfully when the basic MCMC screen fails
(nonfinite diagnostics, split Rhat >= 1.05, ESS < 100, or any divergence).
Recovery and forecast coverage are reported descriptively, not optimized or
asserted as unit-test targets.

An actual CPU reference run is saved in
[`examples/reference_result.json`](examples/reference_result.json):

- Zero post-warmup divergences; maximum classical split Rhat 1.0031;
  minimum NumPyro effective sample size 1,006.6.
- All four fixed generating transition coefficients lie inside their 90%
  equal-tailed posterior intervals.
- 74/80 held-out observations (92.5%) lie inside their pointwise 90% predictive
  intervals; forecast RMSE 0.505 and mean marginal log predictive density -0.763.
- Approximately one minute on the reference CPU environment; runtime varies.

See [`examples/README.md`](examples/README.md) for the statistical scope and how
to interpret the result. This is a fitted simulation example, not a real-data
validation or a calibration study. Older synthetic-data inspection charts, if
used separately, are not evidence of posterior fitting or convergence.

## Project structure

```text
src/bayes_tools/
  config.py
  datasets/
    synthetic.py                 # Seeded hierarchical LGSSM simulation
    lgssm_datasets.py             # pandas preprocessing and dataset helpers
  models/
    numpyro_hierarchical_lgssm.py # Experimental general latent-state model
    scalar_lgssm.py               # Identified scalar Kalman likelihood/forecast
  inference/
    numpyro_runner.py             # General-model NUTS entrypoint
    dynamax_bridge.py             # Posterior arrays -> parameter bundle
  utils/dependencies.py
examples/
  fit_scalar_lgssm.py
  reference_result.json
  README.md
tests/
```

The simulator is available as `bayes_tools.simulate_hierarchical_lgssm` or from
`bayes_tools.datasets.synthetic`. The previously documented
`bayes_tools.data.synthetic` module was absent from the repository; `datasets`
is the canonical source package. No `.gitignore` rule needs to be changed.

## Scope and reproducibility

The general model estimates the observation loading and noise scales as well as
latent states. Such unconstrained state-space parameterizations can have scale,
sign, and rotation ambiguities, and difficult posterior geometry. Its execution
smoke test is **not** evidence of recovery or convergence. The fitted scalar
example fixes the observation loading to one and treats the generating noise
standard deviations as known, so it does not validate that more general model.

The checked-in lock captures the optional inference stack as well as the base
stack. Updating dependencies should regenerate the lock and rerun both setups.
Fixed seeds and a lock improve repeatability, but do not promise bitwise-identical
MCMC trajectories across platforms. Python 3.13 on Linux CPU was tested; other
platforms and Python versions have not been verified here.
