# Fitted scalar LGSSM example

Run from the repository root after the inference-enabled setup in the main README:

```bash
uv run --no-sync python examples/fit_scalar_lgssm.py --strict
```

To write a fresh reference report explicitly:

```bash
uv run --no-sync python examples/fit_scalar_lgssm.py \
  --strict --output examples/reference_result.json
```

## Data and model

For group g, the generator uses:

- Initial state: x[g,0] ~ Normal(0, 1).
- Dynamics: x[g,t] ~ Normal(phi[g] * x[g,t-1], 0.4).
- Observation: y[g,t] ~ Normal(x[g,t], 0.2).

The second Normal argument above is a **standard deviation**, not a variance.
The four fixed generating transitions are 0.35, 0.50, 0.65, and 0.80. Each group
has 120 observations. Only the first 100 enter the likelihood.

The fitted hierarchy is non-centered:

- location ~ Normal(0, 0.8).
- scale ~ HalfNormal(0.4).
- raw[g] ~ Normal(0, 1).
- phi[g] = tanh(location + scale * raw[g]).

Location and scale live on the unconstrained atanh scale. They do not have a
specified generating truth because the four group coefficients were fixed,
not drawn from the hierarchy. Four groups also give limited information about
the population distribution.

The loading is fixed to one and both noise scales are known. This restriction
makes a modest, interpretable demonstration: it deliberately avoids claiming
joint recovery of state loading, process noise, and measurement noise. The
latent states are analytically integrated out with a scalar Kalman likelihood.
The likelihood has an independent dense multivariate-normal reference test,
plus initial-state and derivative checks.

## What the diagnostics establish

The report includes all sampled sites and the derived transition coefficients.
NumPyro computes classical split Rhat and effective sample size; these are not
rank-normalized Rhat or modern bulk/tail ESS. Four chains, low Rhat, reasonable
ESS, and zero divergences are useful basic screens, not proof of convergence.
The `--strict` flag rejects failures of the stated screen rather than presenting
them as a successful fit.

Reference transition recovery (90% equal-tailed intervals):

| Group | Truth | Posterior median | Interval |
|---|---:|---:|---|
| 0 | 0.35 | 0.269 | [0.075, 0.460] |
| 1 | 0.50 | 0.484 | [0.341, 0.621] |
| 2 | 0.65 | 0.664 | [0.537, 0.789] |
| 3 | 0.80 | 0.744 | [0.640, 0.850] |

Each posterior transition draw gets its own filtered terminal-state
mean and variance, using training data only. Forecasts then run freely for 20
steps, propagating terminal-state, process, observation, and parameter
uncertainty. The true hidden state and withheld observations never enter that
forecast calculation.

The report's log score is a mean **marginal** log predictive density: for each
held-out observation, first average its Gaussian density over posterior draws,
then take the logarithm. It is not the joint predictive density of a 20-step
trajectory. Predictive intervals use replicated trajectories, not latent-state
credible intervals.

This single run's 92.5% coverage is descriptive. The 80 held-out observations
are serially dependent, and one dataset cannot establish calibrated coverage.
Repeated independent simulations, broader parameter regimes, alternative
priors, and real-data model checks would be needed for stronger claims.

## Reproduce, do not cherry-pick

Simulation seed 20261023, fit seed 17, and predictive seed 29 are fixed. The
reference report records the exact environment and run settings. Do not change
the simulation seed to improve interval recovery or coverage. If a later run
has sampler warnings, inspect them and document any sampling/model changes.
Changing `--warmup`, `--samples`, or `--chains` changes the experiment, and the
resulting settings are recorded in the report. The tiny general-model chain
in the unit tests only checks execution; it is not a second validated fit.
