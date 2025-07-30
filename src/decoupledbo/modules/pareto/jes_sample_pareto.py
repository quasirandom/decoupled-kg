r"""
Functions for sampling the Pareto front for our implementation of JES (Tu et al., 2022).

Note that the algorithm isn't exactly the same as in the JES paper, because we prune the
Pareto front using crowding distance instead of hypervolume improvement. Also, we allow
the samples to contain different numbers of points rather than raising, since this is
necessary to support a mixture of samples with degenerate and non-degenerate Pareto
fronts.

This implementation is based on the one at `sample_optimal_points()` in
`botorch/acquisition/multi_objective/utils.py` but with the above-mentioned
modifications.

Tu, B., Gandy, A., Kantas, N. & Shafei, B. Joint Entropy Search for Multi-Objective
    Bayesian Optimization. (2022). In Advances in Neural Information Processing Systems.
    https://arxiv.org/abs/2210.02905
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from botorch.exceptions import UnsupportedError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.box_decompositions import (
    BoxDecompositionList,
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from pymoo.algorithms.moo.nsga2 import NSGA2, calc_crowding_distance
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.randomized_argsort import randomized_argsort
from torch import Tensor

logger = logging.getLogger(__name__)


def sample_discrete_pareto_optimal_points(
    model: Model,
    bounds: Tensor,
    num_samples: int,
    target_num_points: int,
    *,
    num_rffs: int = 512,
    nsga2_pop_size: int = 100,
    nsga2_generations: int = 500,
    maximize: bool = True,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Sample the GP using RFFs, then use NSGA-II (+ pruning) to find the Pareto front

    Args:
        model: The GP model
        bounds:  A `2 x d`-dim Tensor containing the input bounds
        num_samples:  The number of samples to generate
        target_num_points: The target number of points to return on the Pareto front for
            each sample (could be less in the case of a degenerate front)
        num_rffs: The number of random Fourier features to use for sampling the GP
        nsga2_pop_size: The population size to use in the NSGA-II algorithm
        nsga2_generations: The number of generations to use in the NSGA-II algorithm
        maximize: If True then maximize all objectives; if False then minimize all
            objectives

    Returns:
        A 2-element tuple containing a list of `num_samples` Pareto sets and a list of
            corresponding Pareto fronts. The ith Pareto set is an `num_points_i x d`-dim
            Tensor, while the ith Pareto front is an `num_points_i x m`-dim Tensor.
    """
    pareto_sets, pareto_fronts = [], []
    for i in range(num_samples):
        sample = get_gp_samples(
            model,
            model.num_outputs,
            n_samples=1,
            num_rff_features=num_rffs,
        )
        ps, pf = run_nsga2_with_pruning(
            sample,
            bounds,
            target_num_points,
            maximize=maximize,
            pop_size=nsga2_pop_size,
            generations=nsga2_generations,
        )
        pareto_sets.append(ps)
        pareto_fronts.append(pf)

    return pareto_sets, pareto_fronts


def run_nsga2_with_pruning(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_points: int,
    maximize: bool,
    *,
    pop_size: int,
    generations: int,
):
    """Run NSGA-II but prune the resulting set of points.

    NSGA-II isn't guaranteed to return `pop_size` Pareto optimal points. Indeed, some of
    the population may be dominated at the end of the optimisation. This function calls
    NSGA-II with a much larger population size. This population is then pruned to
    `num_points` points by repeatedly removing the point with the smallest crowding
    distance. This will encourage a more reliable number of non-dominated points.

    Note that it is still possible that we return fewer than `num_points` points. For
    example, if all objectives are maximised at the same location then the Pareto front
    will be degenerate (i.e. it will be a single point).
    """

    ps_unpruned_np, pf_unpruned_np = run_nsga2(
        model=model,
        bounds=bounds,
        num_objectives=model.num_outputs,
        generations=generations,
        pop_size=pop_size,
        maximize=maximize,
    )

    ps_np, pf_np = prune_pareto_front(ps_unpruned_np, pf_unpruned_np, num_points)

    # Convert from numpy back to torch
    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
    pareto_set = torch.tensor(ps_np, **tkwargs)
    pareto_front = torch.tensor(pf_np, **tkwargs)

    if model.num_outputs == 1:
        pareto_set = pareto_set.unsqueeze(0)
        pareto_front = pareto_front.unsqueeze(0)

    return pareto_set, pareto_front


def run_nsga2(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_objectives: int,
    generations: int = 100,
    pop_size: int = 100,
    maximize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Runs pymoo genetic algorithm NSGA2 to compute the Pareto set and front.
        https://pymoo.org/algorithms/moo/nsga2.html

    Args:
        model: The random Fourier feature GP sample.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_objectives: The number of objectives.
        generations: The number of generations of NSGA2.
        pop_size: The population size maintained at each step of NSGA2.
        maximize: If true we solve for the Pareto maximum.

    Returns:
        A two-element tuple containing

        - pareto_sets: A `num_pareto_points x d`-dim numpy array containing the Pareto
            optimal set of inputs
        - pareto_fronts: A `num_pareto_points x num_objectives`-dim numpy array
            containing the Pareto optimal set of objectives
    """
    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}

    d = bounds.shape[-1]
    weight = -1.0 if maximize else 1.0

    class PymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=d,
                n_obj=num_objectives,
                n_constr=0,
                xl=bounds[0].cpu().detach().numpy(),
                xu=bounds[1].cpu().detach().numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            xt = torch.tensor(x, **tkwargs)
            out["F"] = weight * model.posterior(xt).mean.cpu().detach().numpy()
            return out

    # use NSGA2 to generate a number of Pareto optimal points.
    results = minimize(
        problem=PymooProblem(),
        algorithm=NSGA2(
            pop_size=pop_size,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
            mutation=PolynomialMutation(eta=20),
            eliminate_duplicates=True,
        ),
        termination=MaximumGenerationTermination(generations),
    )

    return results.X, weight * results.F


def prune_pareto_front(
    pareto_set: np.ndarray, pareto_front: np.ndarray, num_points: int
):
    """Prune the Pareto front

    This is done using the crowding distance used within NSGA-II. However, unlike in
    NSGA-II, we remove points from the Pareto front one-by-one, recomputing the crowding
    distance at each iteration.

    Args:
        pareto_set: The Pareto set, returned by run_nsga2 (n_points x n_dim)
        pareto_front: The Pareto front, returned by run_nsga2 (n_points x n_dim)
        num_points: The desired number of points
    """
    idxs = np.arange(len(pareto_front), dtype=int)
    keep = np.ones(len(pareto_front), dtype=np.bool)

    while keep.sum() > num_points:
        crowd_dists = calc_crowding_distance(pareto_front[keep])
        min_ix = randomized_argsort(crowd_dists, order="ascending", method="numpy")[0]
        keep[idxs[keep][min_ix]] = False

    return pareto_set[keep], pareto_front[keep]


def compute_sample_box_decomposition(
    pareto_fronts: List[Tensor],
    partitioning: BoxDecomposition = DominatedPartitioning,
    maximize: bool = True,
    num_constraints: Optional[int] = 0,
):
    r"""Compute the box decomposition associated with some sampled Pareto fronts

    To take advantage of batch computations, we pad the hypercell bounds with a
    `2 x (M + K)`-dim Tensor of zeros `[0, 0]`.

    This also supports the single-objective and constrained optimization
    setting. An objective `y` is feasible if `y <= 0`.

    NOTE: This function is copied and modified from the function of the same name in
    botorch. It was modified to take the Pareto fronts as a list of Tensors instead of a
    single tensor, to allow for Pareto fronts with different numbers of points.

    Args:
        pareto_fronts: A `num_pareto_samples`-element list of
            `num_pareto_points x M`-dim Tensors containing the sampled optimal set of
             objectives.
        partitioning: A `BoxDecomposition` module that is used to obtain the
            hyper-rectangle bounds for integration. In the unconstrained case, this
            gives the partition of the dominated space. In the constrained case, this
            gives the partition of the feasible dominated space union the infeasible
            space.
        maximize: If true, the box-decomposition is computed assuming maximization.
        num_constraints: The number of constraints `K`.

    Returns:
        A `num_pareto_samples x 2 x J x (M + K)`-dim Tensor containing the bounds for
        the hyper-rectangles. The number `J` is the smallest number of boxes needed
        to partition all the Pareto samples.
    """
    if len(pareto_fronts) == 0:
        raise ValueError("Must supply at least one Pareto front!")

    tkwargs = {"dtype": pareto_fronts[0].dtype, "device": pareto_fronts[0].device}
    # We will later compute `norm.log_prob(NEG_INF)`, this is `-inf` if `NEG_INF` is
    # too small.
    NEG_INF = -1e10

    if not all(pf.ndim == 2 for pf in pareto_fronts):
        raise UnsupportedError(
            "Currently this only supports Pareto fronts of the shape "
            "`num_pareto_points x num_objectives`."
        )

    if not all(pf.shape[-1] == pareto_fronts[0].shape[-1] for pf in pareto_fronts):
        raise ValueError("All Pareto fronts should have the same number of objectives.")

    num_pareto_samples = len(pareto_fronts)
    M = pareto_fronts[0].shape[-1]
    K = num_constraints
    ref_point = torch.ones(M, **tkwargs) * NEG_INF
    weight = 1.0 if maximize else -1.0

    if M == 1:
        # Only consider a Pareto front with one element.
        extreme_values = torch.stack(
            [weight * torch.max(weight * pf, dim=-2).values for pf in pareto_fronts]
        )
        ref_point = weight * ref_point.expand(extreme_values.shape)

        if maximize:
            hypercell_bounds = torch.stack(
                [ref_point, extreme_values], dim=-2
            ).unsqueeze(-1)
        else:
            hypercell_bounds = torch.stack(
                [extreme_values, ref_point], dim=-2
            ).unsqueeze(-1)
    else:
        bd_list = []
        for i in range(num_pareto_samples):
            bd_list.append(
                partitioning(ref_point=ref_point, Y=weight * pareto_fronts[i])
            )

        # `num_pareto_samples x 2 x J x (M + K)`
        hypercell_bounds = (
            BoxDecompositionList(*bd_list).get_hypercell_bounds().movedim(0, 1)
        )

        # If minimizing, then the bounds should be negated and flipped
        if not maximize:
            hypercell_bounds = weight * torch.flip(hypercell_bounds, dims=[1])

    # Add an extra box for the inequality constraint.
    if K > 0:
        # `num_pareto_samples x 2 x (J - 1) x K`
        feasible_boxes = torch.zeros(
            hypercell_bounds.shape[:-1] + torch.Size([K]), **tkwargs
        )

        feasible_boxes[..., 0, :, :] = NEG_INF
        # `num_pareto_samples x 2 x (J - 1) x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, feasible_boxes], dim=-1)

        # `num_pareto_samples x 2 x 1 x (M + K)`
        infeasible_box = torch.zeros(
            hypercell_bounds.shape[:-2] + torch.Size([1, M + K]), **tkwargs
        )
        infeasible_box[..., 1, :, M:] = -NEG_INF
        infeasible_box[..., 0, :, 0:M] = NEG_INF
        infeasible_box[..., 1, :, 0:M] = -NEG_INF

        # `num_pareto_samples x 2 x J x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, infeasible_box], dim=-2)

    # `num_pareto_samples x 2 x J x (M + K)`
    return hypercell_bounds
