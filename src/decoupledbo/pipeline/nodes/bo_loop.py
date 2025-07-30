import logging
import time
from numbers import Real
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.utils import draw_sobol_samples
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize

from decoupledbo.modules.acquisition_optimisation_strategy import (
    AcquisitionOptimisationSpec,
    DiscreteKgOptimisationSpec,
    HvkgOptimisationSpec,
    JesOptimisationSpec,
)
from decoupledbo.modules.model import (
    MIN_NOISE_SE,
    MIN_NOISE_SE_FIXED,
    build_mll_and_model,
    set_hyperparameters,
)
from decoupledbo.modules.pareto.botorch_hypervolume import (
    estimate_hypervolume_from_posterior_mean,
)
from decoupledbo.modules.pareto.sample import (
    BoTorchModel,
    sample_points_on_pareto_front,
)
from decoupledbo.modules.performance_after_scalarisation import (
    estimate_expected_performance_after_scalarisation,
)
from decoupledbo.modules.scalarisations import scalarise_linear
from decoupledbo.modules.utils import is_power_of_2, log_duration, log_node
from decoupledbo.pipeline.constants import SMOKE_TEST, TKWARGS
from decoupledbo.pipeline.data_catalog import DataCatalog

logger = logging.getLogger(__name__)


N_PARETO_POINTS = 1000 if not SMOKE_TEST else 100


@log_node
def generate_initial_data(problem, catalog: DataCatalog, n=6):
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(-2)
    train_x = train_x.to(**TKWARGS)

    train_obj = problem(train_x)
    train_obj_true = problem(train_x, noise=False)

    catalog.save_initial_data(
        [train_x] * train_obj.shape[-1],
        [train_obj[..., i] for i in range(train_obj.shape[-1])],
        [train_obj_true[..., i] for i in range(train_obj.shape[-1])],
    )


@log_node
def fit_hyperparameters(model_config: dict, problem, catalog: DataCatalog, n=1000):
    """Use a sample of n points to fit hyper-parameters for the model"""
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_x = train_x.to(**TKWARGS)
    train_obj = problem(train_x)  # Includes observation noise

    mll, model = build_mll_and_model(
        model_config,
        [train_x] * train_obj.shape[-1],
        [train_obj[..., i] for i in range(train_obj.shape[-1])],
    )
    for i, gp in enumerate(model.models):
        if model_config["outputs"][i]["fix_zero_noise"]:
            gp.likelihood.noise = torch.as_tensor(MIN_NOISE_SE, **TKWARGS) ** 2
            gp.likelihood.noise_covar.raw_noise.requires_grad_(False)

    fit_gpytorch_mll(mll)

    catalog.save_model_hyperparameters(model.state_dict())


@log_node
def pregenerate_scalarisations(
    problem,
    catalog: DataCatalog,
    scalarisations_per_step: Optional[int],
    num_steps: int,
):
    """Generate scalarisations to use for scalarisation based methods

    Saves an `num_steps x scalarisations_per_step x num_objectives`-dim tensor of
    scalarisations.
    """
    if scalarisations_per_step is None:
        # Use a Sobol' sample to generate one scalarisation per step
        scalarisation_weights = sample_simplex(
            problem.num_objectives, num_steps, qmc=True, **TKWARGS
        ).unsqueeze(1)

    else:
        if not is_power_of_2(scalarisations_per_step):
            raise ValueError(
                f"For balance in QMC estimates, scalarisations_per_step should be a "
                f"power of 2. Got {scalarisations_per_step=}."
            )

        scalarisation_weights = torch.stack(
            [
                sample_simplex(
                    problem.num_objectives, scalarisations_per_step, qmc=True, **TKWARGS
                )
                for _ in range(num_steps)
            ]
        )

    catalog.save_scalarisations(scalarisation_weights)


_T = Dict[str, AcquisitionOptimisationSpec]
preset_optimisation_strategy_factories: _T = {
    "discrete_kg": DiscreteKgOptimisationSpec(
        n_discretisation_points_per_axis=11 if not SMOKE_TEST else 3,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=32 if not SMOKE_TEST else 4,
        # The DiscreteKnowledgeGradient only does batching with a 'for' loop, so there
        # is no point in setting batch_limit > 1
        batch_limit=1,
        max_iter=200,
    ),
    "hvkg": HvkgOptimisationSpec(
        num_pareto=10 if not SMOKE_TEST else 2,
        num_fantasies=32 if not SMOKE_TEST else 2,
        num_restarts=1,
        raw_samples=512 if not SMOKE_TEST else 4,
        curr_opt_num_restarts=20,
        curr_opt_raw_samples=1024,
        batch_limit=5,
        max_iter=200,
    ),
    "jes_lb": JesOptimisationSpec(
        estimation_type="LB",
        num_pareto_samples=10,
        num_pareto_points=10,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=512 if not SMOKE_TEST else 4,
        batch_limit=50,
        max_iter=200,
    ),
    "jes_lb2": JesOptimisationSpec(
        estimation_type="LB2",
        num_pareto_samples=10,
        num_pareto_points=10,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=512 if not SMOKE_TEST else 4,
        batch_limit=50,
        max_iter=200,
    ),
}
del _T


@log_node
def run_mobo(
    problem,
    *,
    catalog: DataCatalog,
    run_key: str,
    preset_optimisation_strategy: str,
    max_cumulative_cost: Real = np.inf,
    max_n_batch: Optional[int],
    costs=None,
    model_config: dict,
    separate_objective_evaluation=True,
    fixed_hyperparams=None,
):
    """Run the Multi-Objective optimization

    Initial data for the experiment is loaded from the initial_data.pt data set.

    Args:
        problem: The problem instance which can be queried to return new observations.
        catalog: A DataCatalog to use for saving checkpoints and results.
        run_key: A key to use when saving checkpoints and results (useful because we
            want to compare the results of run_mobo across runs with different
            parameters).
        preset_optimisation_strategy: The identifier for a preset optimisation strategy
            (e.g. 'discrete_kg', 'hvkg', 'jes_lb' or 'jes_lb2').
        max_cumulative_cost: The maximum cumulative cost to run for
        max_n_batch: The maximum number of batches to run for (if None, then will be set
            to the number of pregenerated scalarisations)
        costs: A list-like with evaluation costs for each objective (defaults to a list
            of ones).
        model_config: A dictionary containing config used to build the surrogate model.
        separate_objective_evaluation: If True, only one objective will be evaluated on
            each iteration. If False, then all objectives will be evaluated on each
            iteration.
        fixed_hyperparams: A dictionary of hyper-parameters used if fit_hyperparams is
            "never".
    """

    _zero_noise_objectives_str = ", ".join(
        str(i) for i, obj in enumerate(model_config["outputs"]) if obj["fix_zero_noise"]
    )
    logger.info(
        "Running BO loop with acquisition strategy '%s', "
        "hyper-parameter fitting '%s'%s and run key '%s'",
        preset_optimisation_strategy,
        model_config["fit_hyperparams"],
        (
            f" (noise fixed to zero for objective(s) {_zero_noise_objectives_str})"
            if _zero_noise_objectives_str
            else ""
        ),
        run_key,
    )
    acqf_optimisation_strategy = preset_optimisation_strategy_factories[
        preset_optimisation_strategy
    ]

    initial_data = catalog.load_initial_data(device=TKWARGS["device"])
    train_x = initial_data["train_x"]
    train_obj = initial_data["train_obj"]
    train_obj_true = initial_data["train_obj_true"]

    if not costs:
        costs = [1] * problem.num_objectives

    metrics_history = []
    query_history = {
        "iteration": [],
        "x": [],
        "obj_index": [],
        "obj": [],
        "obj_true": [],
        "cost": [],
        "acq_per_cost": [],
        "init": [],
        "scalarisation": [],
    }
    timings_history = []

    for i, (x, obj, obj_true) in enumerate(zip(train_x, train_obj, train_obj_true)):
        assert len(x) == len(obj) == len(obj_true)
        npts = len(x)
        query_history["iteration"].extend([0] * npts)
        query_history["x"].extend(x.detach().cpu().numpy())
        query_history["obj_index"].extend([i] * npts)
        query_history["obj"].extend(obj.detach().cpu().numpy())
        query_history["obj_true"].extend(obj_true.detach().cpu().numpy())
        query_history["cost"].extend([costs[i]] * npts)
        query_history["acq_per_cost"].extend([float("NaN")] * npts)
        query_history["init"].extend([True] * npts)
        query_history["scalarisation"].extend([None] * npts)

    # Initialize and fit hyper-parameters of the surrogate model. Observation noise
    # level is fixed instead of being inferred; the fitted mean is remembered, to be
    # used for all future iterations (if hyper-parameters are being refitted).
    t1 = time.monotonic()
    model = _initialize_model(
        model_config,
        fixed_hyperparams,
        problem,
        train_x,
        train_obj,
        catalog,
    )
    with torch.no_grad():
        initially_fitted_means = [
            (
                m.outcome_transform.untransform(m.mean_module.constant.data.detach())[
                    0
                ].squeeze()
                if hasattr(m, "outcome_transform") and m.outcome_transform is not None
                else m.mean_module.constant.data.detach()
            )
            for m in model.models
        ]

    catalog.save_checkpoint(
        run_key,
        iteration=0,
        model_state_dict=model.state_dict(),
        model_config=model_config,
        train_x=train_x,
        train_obj=train_obj,
        train_obj_true=train_obj_true,
        problem_bounds=problem.bounds,
    )
    t2 = time.monotonic()
    fit_time = t2 - t1

    # Record metrics
    with log_duration(
        logger, "Estimate posterior Pareto front with NSGA-II (initial)", logging.DEBUG
    ):
        post_pareto_set, post_pareto_front = sample_points_on_pareto_front(
            BoTorchModel(model, problem.bounds, TKWARGS),
            maximize=True,
            npoints=N_PARETO_POINTS,
        )
    catalog.save_posterior_pareto(
        run_key, iteration=0, pareto_set=post_pareto_set, pareto_front=post_pareto_front
    )
    perf = estimate_expected_performance_after_scalarisation(
        post_pareto_set,
        post_pareto_front,
        problem,
        scalarise_linear,
        tkwargs=TKWARGS,
    )
    hv = estimate_hypervolume_from_posterior_mean(
        post_pareto_set,
        post_pareto_front,
        problem,
        problem.ref_point,
        tkwargs=TKWARGS,
    )
    metrics_history.append({**hv, **perf, "cost": sum(query_history["cost"])})

    t3 = time.monotonic()
    metrics_time = t3 - t2

    timings_history.append(
        {
            "iteration": 0,
            "bo": 0,
            "fit": fit_time,
            "metrics": metrics_time,
        }
    )

    logger.info(
        f"Initial: "
        # f"HV >= {hv['pset_hv_lo']:4.2f}, "
        f"perf = {perf['actual_scalarperf']:4.2f}, "
        f"time (fit) = {fit_time:4.2f}s, "
        f"time (metrics) = {metrics_time:4.2f}s."
    )

    all_scalarisations = catalog.load_scalarisations(device=TKWARGS["device"])

    cumulative_cost = sum(
        costs[i] * len(train_obj[i]) for i in range(problem.num_objectives)
    )

    if max_n_batch is None:
        max_n_batch = len(all_scalarisations)

    try:
        iteration = 0
        while (cumulative_cost < max_cumulative_cost) and (iteration < max_n_batch):
            iteration += 1

            t0 = time.monotonic()

            scalarisation_weights = all_scalarisations[iteration - 1]

            if separate_objective_evaluation:
                # Optimize the acquisition function
                (
                    new_x,
                    new_i,
                    acq_per_cost,
                ) = acqf_optimisation_strategy.optimize_for_single_objective(
                    model,
                    costs,
                    input_dim=problem.bounds.shape[-1],
                    scalarisation_weights=scalarisation_weights,
                    # For now, we assume the reference point is known
                    hv_refpoint=problem.ref_point,
                    existing_sampled_x=[
                        normalize(_x, problem.bounds) for _x in train_x
                    ],
                    existing_sampled_obj=train_obj,
                )
                new_x = unnormalize(new_x, problem.bounds)

                # Observe the problem at this point
                cost_this_iteration = costs[new_i]
                cumulative_cost += cost_this_iteration
                new_obj, new_obj_true = observe_problem_single_objective(
                    new_x, new_i, problem
                )

                # Log the observation
                query_history["iteration"].append(iteration)
                query_history["x"].append(new_x.detach().cpu().numpy().squeeze(0))
                query_history["obj_index"].append(new_i)
                query_history["obj"].append(new_obj.item())
                query_history["obj_true"].append(new_obj_true.item())
                query_history["cost"].append(costs[new_i])
                query_history["acq_per_cost"].append(acq_per_cost.item())
                query_history["init"].append(False)
                query_history["scalarisation"].append(
                    scalarisation_weights[0].detach().cpu().numpy()
                    if scalarisation_weights.shape[0] == 1
                    else None
                )

                # Update training data
                train_x[new_i] = torch.cat([train_x[new_i], new_x])
                train_obj[new_i] = torch.cat([train_obj[new_i], new_obj])
                train_obj_true[new_i] = torch.cat([train_obj_true[new_i], new_obj_true])

            else:
                (
                    new_x,
                    acq_value,
                ) = acqf_optimisation_strategy.optimize_for_full_evaluation(
                    model,
                    input_dim=problem.bounds.shape[-1],
                    scalarisation_weights=scalarisation_weights,
                    # For now, we assume the reference point is known
                    hv_refpoint=problem.ref_point,
                    existing_sampled_x=[
                        normalize(_x, problem.bounds) for _x in train_x
                    ],
                    existing_sampled_obj=train_obj,
                )
                new_x = unnormalize(new_x, problem.bounds)
                acq_per_cost = acq_value / sum(costs)
                cost_this_iteration = sum(costs)
                cumulative_cost += cost_this_iteration
                new_obj, new_obj_true = observe_problem_full(new_x, problem)

                for i in range(new_obj_true.shape[-1]):
                    query_history["iteration"].append(iteration)
                    query_history["x"].append(new_x.detach().cpu().numpy().squeeze(0))
                    query_history["obj_index"].append(i)
                    query_history["obj"].append(new_obj.detach()[..., i].item())
                    query_history["obj_true"].append(
                        new_obj_true.detach()[..., i].item()
                    )
                    query_history["cost"].append(costs[i])
                    query_history["acq_per_cost"].append(acq_per_cost.item())
                    query_history["init"].append(False)
                    query_history["scalarisation"].append(
                        scalarisation_weights[0].detach().cpu().numpy()
                        if scalarisation_weights.shape[0] == 1
                        else None
                    )

                for i in range(new_obj_true.shape[-1]):
                    train_x[i] = torch.cat([train_x[i], new_x])
                    train_obj[i] = torch.cat([train_obj[i], new_obj[..., i]])
                    train_obj_true[i] = torch.cat(
                        [train_obj_true[i], new_obj_true[..., i]]
                    )

            t1 = time.monotonic()
            bo_time = t1 - t0

            # Initialize and fit hyper-parameters of the surrogate model for next
            # iteration
            model = _initialize_model(
                model_config,
                fixed_hyperparams,
                problem,
                train_x,
                train_obj,
                catalog,
                initially_fitted_means=initially_fitted_means,
            )

            catalog.save_checkpoint(
                run_key,
                iteration,
                model.state_dict(),
                model_config,
                train_x,
                train_obj,
                train_obj_true,
                problem.bounds,
            )
            t2 = time.monotonic()
            fit_time = t2 - t1

            # Record metrics
            with log_duration(
                logger,
                f"Estimate posterior Pareto front with NSGA-II (iteration {iteration})",
                logging.DEBUG,
            ):
                post_pareto_set, post_pareto_front = sample_points_on_pareto_front(
                    BoTorchModel(model, problem.bounds, TKWARGS),
                    maximize=True,
                    npoints=N_PARETO_POINTS,
                )
            catalog.save_posterior_pareto(
                run_key, iteration, post_pareto_set, post_pareto_front
            )
            perf = estimate_expected_performance_after_scalarisation(
                post_pareto_set,
                post_pareto_front,
                problem,
                scalarise_linear,
                tkwargs=TKWARGS,
            )
            hv = estimate_hypervolume_from_posterior_mean(
                post_pareto_set,
                post_pareto_front,
                problem,
                problem.ref_point,
                tkwargs=TKWARGS,
            )
            metrics_history.append({**hv, **perf, "cost": cost_this_iteration})

            t3 = time.monotonic()
            metrics_time = t3 - t2

            timings_history.append(
                {
                    "iteration": iteration,
                    "bo": bo_time,
                    "fit": fit_time,
                    "metrics": metrics_time,
                }
            )

            if scalarisation_weights.shape[0] == 1:
                scalarisation_str = ", ".join(
                    f"{w:.2f}" for w in scalarisation_weights[0]
                )
            else:
                scalarisation_str = f"expectation({scalarisation_weights.shape[0]})"

            _nchar = 1 + int(np.log10(max_cumulative_cost))
            logger.info(
                f"Batch {iteration:>2} (cost {cumulative_cost:>{_nchar}g} of {max_cumulative_cost:>{_nchar}g}): "
                f"scalarisation = [{scalarisation_str}], "
                f"obj index = {new_i if separate_objective_evaluation else 'all'}, "
                # f"HV >= {hv['pset_hv_lo']:5.2f}, "
                f"perf = {perf['actual_scalarperf']:5.2f}, "
                f"ACQ/cost = {acq_per_cost.item():>5.2f}, "
                f"time (BO) = {bo_time:>5.2f}s, "
                f"time (fit) = {fit_time:>4.2f}s, "
                f"time (metrics) = {metrics_time:>4.2f}s."
            )

    except KeyboardInterrupt:
        # Make sure we save all the results at the end and compress checkpoints, even
        # if we Ctrl+C
        catalog.save_metrics(run_key, pd.DataFrame(metrics_history))
        catalog.save_bo_run(run_key, pd.DataFrame(query_history))
        catalog.save_timings(run_key, pd.DataFrame(timings_history))
        catalog.compress_checkpoints(run_key)
        raise

    except RuntimeError:
        # For any other sort of error, still tarball the checkpoints, but do not save
        # the results
        catalog.compress_checkpoints(run_key)
        raise

    # Save the results and compress checkpoints if we didn't Ctrl+C as well
    catalog.save_metrics(run_key, pd.DataFrame(metrics_history))
    catalog.save_bo_run(run_key, pd.DataFrame(query_history))
    catalog.save_timings(run_key, pd.DataFrame(timings_history))
    catalog.compress_checkpoints(run_key)


def _initialize_model(
    config,
    fixed_hyperparams,
    problem,
    train_x,
    train_obj,
    catalog,
    initially_fitted_means=None,
):
    """Initialize the model, using the strategy determined by fit_hyperparams"""
    if config["fit_hyperparams"] == "never":
        if not isinstance(fixed_hyperparams, dict):
            raise ValueError(
                f"If using {config['fit_hyperparams']=!r} then fixed_hyperparams must "
                f"contain a dictionary of hyper-parameters. Got {fixed_hyperparams=!r}."
            )

        _, model = build_mll_and_model(config, train_x, train_obj, tkwargs=TKWARGS)

        min_noise_se = torch.as_tensor(MIN_NOISE_SE_FIXED, **TKWARGS)
        noise_vars = _extract_noise_vars_tensor(problem)
        noise_vars = torch.maximum(noise_vars, min_noise_se**2)
        for i in range(len(noise_vars)):
            if config["outputs"][i]["fix_zero_noise"]:
                noise_vars = min_noise_se**2
        set_hyperparameters(model, **fixed_hyperparams, noise_vars=noise_vars)
    else:
        mll, model = build_mll_and_model(config, train_x, train_obj, tkwargs=TKWARGS)
        for i, gp in enumerate(model.models):
            if config["outputs"][i]["fix_zero_noise"]:
                gp.likelihood.noise = torch.as_tensor(MIN_NOISE_SE, **TKWARGS) ** 2
                gp.likelihood.noise_covar.raw_noise.requires_grad_(False)

        if config["fit_hyperparams"] == "once":
            state_dict = catalog.load_model_hyperparameters(device=TKWARGS["device"])
            model.load_state_dict(state_dict)
        elif config["fit_hyperparams"] == "always":
            # If this isn't the first iteration of the Bayes-Opt then we don't want to
            # refit the mean because it will be biased towards higher samples.
            if initially_fitted_means is not None:
                for m, c in zip(model.models, initially_fitted_means):
                    if (
                        hasattr(m, "outcome_transform")
                        and m.outcome_transform is not None
                    ):
                        m.outcome_transform.eval()
                        c, _ = m.outcome_transform(c)
                        c = c.squeeze()
                    m.mean_module.requires_grad_(False)
                    m.mean_module.constant = c
            fit_gpytorch_mll(mll)
        else:
            raise ValueError(
                f"Unexpected value for fit_hyperparams. "
                f"Got {config['fit_hyperparams']!r}."
            )
    return model


def _extract_noise_vars_tensor(problem) -> torch.Tensor:
    """Extract tensor of noise variances from test problem"""
    if problem.noise_std is None:
        noise_vars = torch.tensor([0] * problem.num_objectives, **TKWARGS)
    elif isinstance(problem.noise_std, Real):
        noise_vars = torch.tensor(
            [problem.noise_std**2] * problem.num_objectives, **TKWARGS
        )
    elif isinstance(problem.noise_std, torch.Tensor):
        if problem.noise_std.ndim == 0:
            noise_vars = problem.noise_std.repeat(problem.num_objectives) ** 2
        elif problem.noise_std.ndim == 1:
            if len(problem.noise_std) == 1:
                noise_vars = problem.noise_std.repeat(problem.num_objectives) ** 2
            else:
                noise_vars = problem.noise_std**2
        else:
            raise ValueError(
                f"Unexpected dimensions for problem.noise_std. "
                f"Got {problem.noise_std.shape=}."
            )
    else:
        raise TypeError(
            f"Unexpected type for problem.noise_std: {type(problem.noise_std)=}"
        )
    return noise_vars


def observe_problem_single_objective(x, output_ix, problem):
    new_obj = problem(x)[..., output_ix]
    new_obj_true = problem(x, noise=False)[..., output_ix]
    return new_obj, new_obj_true


def observe_problem_full(x, problem):
    new_obj = problem(x)
    new_obj_true = problem(x, noise=False)
    return new_obj, new_obj_true


def build_model_from_checkpoint(cp):
    _, model = build_mll_and_model(cp["model_config"], cp["train_x"], cp["train_obj"])
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model
