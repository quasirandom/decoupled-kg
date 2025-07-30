"""Utility for estimating the expected scalarised performance metric"""

import logging

import numpy as np
import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.sampling import sample_simplex

from decoupledbo.modules.scalarisations import TScalarisation
from decoupledbo.pipeline.constants import SMOKE_TEST

N_SCALARISATIONS = 2**10
N_PARETO_POINTS = 1000 if not SMOKE_TEST else 100

logger = logging.getLogger(__name__)


def estimate_best_possible_expected_performance_after_scalarisation(
    pareto_front: np.ndarray,
    scalarise: TScalarisation,
    *,
    n_scalarisations=N_SCALARISATIONS,
    scalarisations_seed=None,
    tkwargs=None,
):
    """
    Estimate the expected value of the max possible scalarised objective.

    This is done using RQMC over a Sobol sample of the scalarisation weights. The finite
    set of points representing the Pareto front is used to estimate the maximum of each
    scalarisation at each point. Therefore, the pareto_front argument should be
    sufficiently fine for this purpose.

    Args:
        pareto_front: An npoints-by-m matrix of points representing the Pareto front.
        scalarise: A function to scalarise multiple objectives. Its signature should be
                scalarise(objective_values, scalarisation_weights)
        n_scalarisations: The number of different scalarisation weights to use when
            estimating the expectation via RQMC.
        scalarisations_seed: The seed to use when generating the scalarisation weights.
        tkwargs: Keyword arguments when creating tensors.
    """

    pareto_front = torch.from_numpy(pareto_front).to(**tkwargs)
    num_objectives = pareto_front.shape[-1]

    scalarisation_weights = sample_simplex(
        num_objectives,
        qmc=True,
        n=n_scalarisations,
        seed=scalarisations_seed,
        **tkwargs,
    )

    scalarised = scalarise(pareto_front, scalarisation_weights.unsqueeze(-2))
    max_scalarised, _ = torch.max(scalarised, dim=-1)
    return torch.mean(max_scalarised).item()


def estimate_expected_performance_after_scalarisation(
    posterior_pareto_set: np.ndarray,
    posterior_pareto_front: np.ndarray,
    problem: MultiObjectiveTestProblem,
    scalarise: TScalarisation,
    *,
    n_scalarisations=N_SCALARISATIONS,
    scalarisations_seed=None,
    tkwargs=None,
):
    """
    Estimate the expected value of the scalarised objective at the recommended inputs

    Args:
        posterior_pareto_set: An npoints-by-d matrix of points in the Pareto set
            associated with the posterior mean
        posterior_pareto_front: An npoints-by-m matrix of points in the Pareto front
            associated with the posterior mean (where the ith row corresponds to the ith
            row in `posterior_pareto_set`)
        problem: The problem, used to evaluate the true accuracy of the recommendations
            in the `posterior_pareto_set`
        scalarise: A function to scalarise multiple objectives. Its signature should be
                scalarise(objective_values, scalarisation_weights)
        n_scalarisations: The number of scalarisations to use in the RQMC estimate of
            the expectation over scalarisations
        scalarisations_seed: The seed to use for the RQMC estimate of the expectation
            over scalarisations
        tkwargs: A dictionary of keyword arguments to use when creating tensors

    Returns:
        Dict containing 'predicted_scalarperf' (the expected predicted scalarised
            objective) and 'actual_scalarperf' (the expected actual scalarised
            objective)
    """

    posterior_pareto_set = torch.from_numpy(posterior_pareto_set).to(**tkwargs)
    posterior_pareto_front = torch.from_numpy(posterior_pareto_front).to(**tkwargs)

    # Generate scalarisation weights using a Sobol sample (so we are doing QMC)
    scalarisation_weights = sample_simplex(
        problem.num_objectives,
        qmc=True,
        n=n_scalarisations,
        seed=scalarisations_seed,
        **tkwargs,
    )

    # Compute the recommendations for each scalarisation weight in the QMC
    scalarised = scalarise(posterior_pareto_front, scalarisation_weights.unsqueeze(-2))
    predicted_perfs, rec_indices = torch.max(scalarised, dim=-1)
    av_predicted_perf = predicted_perfs.mean().item()

    # Calculate the actual performance realised for each recommendation and the
    # corresponding average over scalarisations
    rec_designs = posterior_pareto_set[rec_indices]
    real_outputs = problem(rec_designs, noise=False)
    performances = scalarise(real_outputs, scalarisation_weights)
    av_performance = performances.mean().item()

    return {
        "predicted_scalarperf": av_predicted_perf,
        "actual_scalarperf": av_performance,
    }
