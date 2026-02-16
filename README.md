# Hierarchical Bayesian Inference Sandbox

A reorganized research-ready scaffold for hierarchical linear Gaussian state-space models (LGSSMs), focused on:

- **NumPyro** for non-centered hierarchical Bayesian inference.
- **Dynamax interoperability** for downstream filtering/smoothing workflows.
- **Deterministic synthetic data generation** for reproducible experiments.
- **Unit-test-first architecture** with isolated modules.

## Project structure

```text
src/bayes_tools/
  config.py                      # Typed model configuration
  data/
    synthetic.py                 # Hierarchical LGSSM simulation
  models/
    numpyro_hierarchical_lgssm.py
  inference/
    numpyro_runner.py            # NUTS entrypoint
    dynamax_bridge.py            # Posterior -> Dynamax conversion
  utils/
    dependencies.py
tests/
  test_config.py
  test_synthetic_data.py
  test_dynamax_bridge.py
  test_numpyro_model_contract.py
```

## Getting started

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
pytest
```

## Design notes

- Hierarchical parameters for transition and noise components use **non-centered parameterization**.
- The package remains usable even when JAX/NumPyro are not installed; inference calls fail with informative errors, and tests for model contracts are skipped.
- Dynamax integration is explicit and testable through a conversion layer rather than hidden side-effects.
