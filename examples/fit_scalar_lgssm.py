"""Fit one seeded, identified hierarchical LGSSM and report actual diagnostics."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from bayes_tools import HierarchicalLGSSMConfig, simulate_hierarchical_lgssm
from bayes_tools.models.scalar_lgssm import (
    forecast_scalar_lgssm,
    hierarchical_scalar_lgssm_model,
    validate_scalar_lgssm_inputs,
)


def encode_report(report: dict[str, object]) -> str:
    """Write failed diagnostics as JSON null rather than losing the report."""

    def clean(value: object) -> object:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps(clean(report), indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/output/fitted_lgssm.json")
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--strict", action="store_true", help="Fail if the basic MCMC screen fails."
    )
    args = parser.parse_args()
    if args.warmup < 100 or args.samples < 100 or args.chains < 2:
        parser.error(
            "use at least 100 warmup steps, 100 draws, and 2 chains for diagnostics"
        )
    try:
        import jax
        from jax import random
        from numpyro.diagnostics import summary
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise SystemExit(
            "Install inference dependencies: uv sync --locked --extra inference"
        ) from exc

    # Set before creating JAX arrays. Sequential chains also work on a single CPU.
    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    data_seed, fit_seed, predictive_seed = 20261023, 17, 29
    groups, train_length, horizon = 4, 100, 20
    truth = np.array([0.35, 0.5, 0.65, 0.8])
    process_std, emission_std = np.full(groups, 0.4), np.full(groups, 0.2)
    dataset = simulate_hierarchical_lgssm(
        HierarchicalLGSSMConfig(groups, train_length + horizon, 1, 1),
        seed=data_seed,
        transition_matrices=truth[:, None, None],
        emission_matrix=np.ones((1, 1)),
        process_std=process_std[:, None],
        emission_std=emission_std[:, None],
    )
    train = dataset.observations[:, :train_length, 0]
    heldout = dataset.observations[:, train_length:, 0]
    validate_scalar_lgssm_inputs(train, process_std, emission_std)
    sampler = MCMC(
        NUTS(hierarchical_scalar_lgssm_model, target_accept_prob=0.95),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    sampler.run(random.PRNGKey(fit_seed), train, process_std, emission_std)
    chains = sampler.get_samples(group_by_chain=True)
    diagnostics = summary(chains, prob=0.9, group_by_chain=True)
    site_diagnostics = {
        name: {
            "max_split_rhat": float(np.max(values["r_hat"])),
            "min_effective_sample_size": float(np.min(values["n_eff"])),
        }
        for name, values in diagnostics.items()
    }
    max_rhat = float(np.max([d["max_split_rhat"] for d in site_diagnostics.values()]))
    min_ess = float(
        np.min([d["min_effective_sample_size"] for d in site_diagnostics.values()])
    )
    divergences = int(np.asarray(sampler.get_extra_fields()["diverging"]).sum())
    passed = bool(
        np.isfinite([max_rhat, min_ess]).all()
        and max_rhat < 1.05
        and min_ess >= 100
        and divergences == 0
    )
    draws = np.asarray(chains["transition"]).reshape(-1, groups)
    lower, median, upper = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    recovery = [
        {
            "group": group,
            "true_transition": float(truth[group]),
            "posterior_median": float(median[group]),
            "equal_tailed_90_percent_interval": [
                float(lower[group]),
                float(upper[group]),
            ],
            "truth_in_interval": bool(lower[group] <= truth[group] <= upper[group]),
        }
        for group in range(groups)
    ]
    predictive, means, variances = forecast_scalar_lgssm(
        train,
        draws,
        process_std,
        emission_std,
        horizon=horizon,
        seed=predictive_seed,
    )
    predictive_lower, predictive_upper = np.quantile(predictive, [0.05, 0.95], axis=0)
    coverage = (heldout >= predictive_lower) & (heldout <= predictive_upper)
    predictive_mean = means.mean(axis=0)
    log_density = -0.5 * (
        np.log(2 * np.pi * variances) + (heldout[None, :, :] - means) ** 2 / variances
    )
    log_score = np.logaddexp.reduce(log_density, axis=0) - np.log(len(draws))
    report = {
        "example": "identified_scalar_hierarchical_lgssm",
        "settings": {
            "groups": groups,
            "train_timesteps": train_length,
            "heldout_timesteps": horizon,
            "chains": args.chains,
            "warmup_per_chain": args.warmup,
            "draws_per_chain": args.samples,
            "target_accept_probability": 0.95,
            "chain_method": "sequential",
            "jax_enable_x64": True,
            "data_seed": data_seed,
            "fit_seed": fit_seed,
            "predictive_seed": predictive_seed,
            "known_process_std": process_std.tolist(),
            "known_emission_std": emission_std.tolist(),
            "known_emission_loading": 1.0,
            "initial_state_distribution": "Normal(0, 1)",
        },
        "mcmc": {
            "divergences": divergences,
            "max_classical_split_rhat": max_rhat,
            "min_numpyro_effective_sample_size": min_ess,
            "basic_screen_passed": passed,
            "nonfinite_diagnostics": bool(not np.isfinite([max_rhat, min_ess]).all()),
            "screen": "finite split Rhat < 1.05, ESS >= 100, zero divergences at all reported sites",
            "by_site": site_diagnostics,
        },
        "transition_recovery": recovery,
        "heldout_forecast": {
            "n_observations": int(heldout.size),
            "covered_by_pointwise_90_percent_interval": int(coverage.sum()),
            "pointwise_90_percent_interval_coverage": float(coverage.mean()),
            "coverage_by_group": coverage.mean(axis=1).tolist(),
            "rmse_of_posterior_predictive_mean": float(
                np.sqrt(np.mean((heldout - predictive_mean) ** 2))
            ),
            "mean_marginal_log_predictive_density": float(log_score.mean()),
            "mean_pointwise_interval_width": float(
                np.mean(predictive_upper - predictive_lower)
            ),
        },
        "limitations": [
            "One seeded synthetic dataset, not a repeated-simulation calibration study or real-data validation.",
            "Known observation loading and noise scales: this does not validate joint recovery in the general unconstrained LGSSM.",
            "Only four groups: population location/scale are weakly informed and have no declared generating truth here.",
            "NumPyro classical split Rhat and ESS are basic screens, not rank-normalized bulk/tail diagnostics or proof of convergence.",
            "Held-out points are serially correlated; descriptive pointwise coverage is not an independent-binomial calibration estimate.",
            "The log score averages marginal forecast densities, not the joint density of each 20-step path.",
            "Identical seeds fix inputs and PRNG streams, but different hardware/library versions may change floating-point trajectories.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "jax", "jaxlib", "numpyro", "dynamax"]
            },
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded_report = encode_report(report)
    args.output.write_text(encoded_report + "\n")
    print(encoded_report)
    print(f"\nSaved {args.output}")
    if args.strict and not passed:
        raise SystemExit(
            "Basic MCMC screen failed; inspect diagnostics before interpreting this fit."
        )


if __name__ == "__main__":
    main()
