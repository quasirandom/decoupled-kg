"""The main data pipeline

Module also contains a couple of miscellaneous nodes which don't live anywhere else.
"""

import logging
import os.path
from typing import List, Union

import torch

from decoupledbo.modules.gp_testproblem import (
    GPTestProblem,
    GPTestProblemModel,
    bounds_to_tensor,
    create_gp_problem_model,
    estimate_reference_point_and_hypervolume,
)
from decoupledbo.modules.pareto.sample import (
    BoTorchProblem,
    sample_points_on_pareto_front,
)
from decoupledbo.modules.scalarisations import scalarise_linear
from decoupledbo.modules.utils import as_tensor_or_none, log_node, set_random_seed
from decoupledbo.pipeline.cli import parse_commandline_arguments_and_read_config
from decoupledbo.pipeline.constants import LOG_FORMAT, SMOKE_TEST, TKWARGS
from decoupledbo.pipeline.data_catalog import DataCatalog
from decoupledbo.pipeline.nodes.bo_loop import (
    fit_hyperparameters,
    generate_initial_data,
    pregenerate_scalarisations,
    run_mobo,
)
from decoupledbo.pipeline.nodes.metrics import (
    estimate_best_possible_expected_performance_after_scalarisation,
)

logger = logging.getLogger(__name__)


# Run keys
EVAL_SEPARATE = "eval_separate"
EVAL_FULL = "eval_full"
ALL_RUN_KEYS = [EVAL_SEPARATE, EVAL_FULL]


@log_node
def gen_true_pareto_front(problem, catalog):
    """A node to generate a sample from the Pareto front"""
    pareto_set, pareto_front = sample_points_on_pareto_front(
        BoTorchProblem(problem, noise=False), maximize=True, npoints=1000
    )
    catalog.save_true_pareto(pareto_set, pareto_front)


@log_node
def make_test_problem(config, catalog):
    """Build a test problem from config"""
    noise_stdevs = as_tensor_or_none(config["noise_stdevs"], tkwargs=TKWARGS)

    if config["type"] == "gp-sample":
        if config["args"]["regenerate"]:
            regenerate_gp_test_problem_data(config["args"]["id"], catalog)

        problem, fixed_hyperparams = load_and_construct_gp_test_problem(
            config["args"]["id"], catalog, noise_stds=noise_stdevs
        )

    else:
        raise ValueError(f"Unrecognised test problem 'type'. Got {config['type']!r}.")

    return problem, fixed_hyperparams


@log_node
def regenerate_gp_test_problem_data(name, catalog):
    """Regenerate the GP test problem data

    The GP test problem is shared between namespaces to facilitate repeats on the same
    problem on the cluster. This node resamples the GP, calculates the reference point
    and hypervolume and saves the result.
    """
    bounds = [(0, 1), (0, 1)]
    fixed_hyperparams = dict(
        length_scales=[0.2, 1.8],
        output_scales=[1, 50],
        means=[0, 0],
    )

    problem_model, train_x, train_y = create_gp_problem_model(
        bounds,
        n_objectives=2,
        **fixed_hyperparams,
        input_seed=844761,
        output_seeds=[884805, 11120],
        **TKWARGS,
    )
    ref_point, hv = estimate_reference_point_and_hypervolume(
        problem_model, bounds_to_tensor(bounds, **TKWARGS), **TKWARGS
    )
    catalog.save_shared_gp_test_problem_data(
        name=name,
        bounds=bounds,
        fixed_hyperparams=fixed_hyperparams,
        model_state_dict=problem_model.state_dict(),
        train_x=train_x,
        train_y=train_y,
        ref_point=ref_point.tolist(),
        max_hv=float(hv),
        negate=False,
    )


@log_node
def load_and_construct_gp_test_problem(
    name, catalog, noise_stds: Union[List[float], torch.Tensor, None]
):
    """Load the shared GP test problem data and construct the GP test problem.

    The GP test problem is shared between namespaces to facilitate repeats on the same
    problem on the cluster. This node resamples the GP and saves the result.
    """
    # WARNING: This is the only node which is returning something rather than saving it.
    test_problem_data = catalog.load_shared_gp_test_problem_data(name)
    gp_model = GPTestProblemModel.reconstruct(
        state_dict=test_problem_data["model_state_dict"],
        input_samples=test_problem_data["train_x"],
        output_samples=test_problem_data["train_y"],
    )

    if noise_stds is not None:
        noise_stds = torch.as_tensor(noise_stds, **TKWARGS)

    problem = GPTestProblem(
        gp_model=gp_model.to(**TKWARGS),
        bounds=test_problem_data["bounds"],
        ref_point=test_problem_data["ref_point"],
        max_hv=test_problem_data["max_hv"],
        noise_stds=noise_stds,
        negate=test_problem_data["negate"],
    )
    return problem.to(**TKWARGS), test_problem_data["fixed_hyperparams"]


def setup_logging(file_path):
    if os.path.exists(file_path):
        raise ValueError(f"Log file already exists! {file_path}")

    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        filename=file_path,
        filemode="w",
    )

    # Also define a console handler for the root logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # Disable debug logging for certain loggers
    logging.getLogger("decoupledbo.modules.acquisition.discretekg").setLevel(
        logging.INFO
    )

    logging.info("Logging to file %s", file_path)


def run_pipeline(config, commandline_args, catalog):
    problem, fixed_hyperparams = make_test_problem(config["problem"], catalog)

    gen_true_pareto_front(problem, catalog)
    estimate_best_possible_expected_performance_after_scalarisation(
        scalarise_linear, catalog
    )

    generate_initial_data(problem, catalog, n=6)

    if config["model"]["fit_hyperparams"] == "once":
        fit_hyperparameters(config["model"], problem, catalog, n=1000)
    else:
        # Delete them to be sure we don't accidentally use old parameters where we may
        # have forgotten to add a switch on commandline_args.fit_hyperparams.
        catalog.delete_model_hyperparameters()

    max_n_batch = 2 if SMOKE_TEST else config["max_n_batch"]
    pregenerate_scalarisations(
        problem, catalog, commandline_args.scalarisations_per_step, max_n_batch
    )

    run_mobo(
        problem,
        catalog=catalog,
        separate_objective_evaluation=True,
        run_key=EVAL_SEPARATE,
        max_cumulative_cost=config["max_cumulative_cost"],
        max_n_batch=max_n_batch,
        preset_optimisation_strategy=commandline_args.acq_strategy,
        costs=[1, 10],  # Making the less useful objective more expensive
        model_config=config["model"],
        fixed_hyperparams=fixed_hyperparams,
    )
    run_mobo(
        problem,
        catalog=catalog,
        separate_objective_evaluation=False,
        run_key=EVAL_FULL,
        max_cumulative_cost=config["max_cumulative_cost"],
        max_n_batch=max_n_batch,
        preset_optimisation_strategy=commandline_args.acq_strategy,
        costs=[1, 10],
        model_config=config["model"],
        fixed_hyperparams=fixed_hyperparams,
    )


if __name__ == "__main__":
    # Note that this makes the dtype in TKWARGS somewhat redundant but catches more
    # cases where we cannot control the dtype, such as when tensors are created inside
    # botorch.
    torch.set_default_dtype(torch.double)

    config, commandline_args = parse_commandline_arguments_and_read_config()

    catalog = DataCatalog(config["namespace"])
    setup_logging(catalog.get_new_log_file_path())

    catalog.save_config(config)
    catalog.save_commandline_args(commandline_args)

    logger.info(f"Smoke test: {SMOKE_TEST}")
    logger.info(f"Config file: {commandline_args.config}")
    logger.info(f"Data namespace: {catalog.namespace!r}")
    logger.info(f"Seed: {commandline_args.seed}")
    if commandline_args.scalarisations_per_step is None:
        logger.info("Scalarisations per step: 1 (Sobol' between steps)")
    else:
        logger.info(
            f"Scalarisations per step: {commandline_args.scalarisations_per_step}"
        )
    logger.info(
        f"Problem type: {config['problem']['type']}; "
        f"ID: {config['problem']['args']['id']}"
    )
    logger.info(f"Observation noise: {config['problem']['noise_stdevs']}")

    # We set the random seed on a best-effort basis. Note that pytorch has some warnings
    # about other sources of non-determinism, especially when using the GPU, but not
    # all. In particular, they warn that random numbers may be different across
    # different computers, even with the same seed and pytorch version.
    # https://pytorch.org/docs/stable/notes/randomness.html
    if commandline_args.seed is not None:
        set_random_seed(commandline_args.seed)

    try:
        run_pipeline(config, commandline_args, catalog)
    except Exception as ex:
        logger.exception(ex)
        exit(1)
