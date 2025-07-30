"""Module for sampling points on the pareto front using PyGMO"""

import logging
from typing import Optional, Tuple

import numpy as np
import pygmo
import torch
from botorch.models.model import Model as _BTModel
from botorch.test_functions.base import MultiObjectiveTestProblem as _BTMOTestProblem
from botorch.utils.transforms import normalize

logger = logging.getLogger(__name__)


def sample_points_on_pareto_front(
    problem, maximize=True, npoints=100
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a sample of points on the pareto front

    Points are calculated using NSGA-II.

    Args:
        problem: A problem instance (must be of a supported type)
        npoints: The number of points to sample
        maximize: Determines whether the problem is to be minimised or maximised

    Returns:
        Tuple[np.ndarray, np,ndarray]: Returns two arrays, one for the decision
            variables (npoints-by-d) and another for the objectives (npoints-by-m).
    """
    if not isinstance(problem, pygmo.problem):
        problem = pygmo.problem(problem)
    return _sample_points_on_pareto_front(problem, maximize, npoints)


def _sample_points_on_pareto_front(problem, maximize, npoints):
    if maximize:
        problem = _negate(problem)
    pop = pygmo.population(problem, size=npoints)
    uda = pygmo.nsga2(gen=100)
    uda.set_bfe(pygmo.bfe())
    algo = pygmo.algorithm(uda)
    pop = algo.evolve(pop)
    logger.debug("Num evaluations: %s", pop.problem.get_fevals())
    x = pop.get_x()
    f = -pop.get_f() if maximize else pop.get_f()
    return x, f


def _negate(problem: pygmo.problem):
    def negate(f):
        def new_f(self, *args, **kwargs):
            return -f(self, *args, **kwargs)

        return new_f

    def negate_list(f):
        def new_f(self, *args, **kwargs):
            return [-x for x in f(self, *args, **kwargs)]

        return new_f

    return pygmo.problem(
        pygmo.decorator_problem(
            problem,
            fitness_decorator=negate,
            batch_fitness_decorator=negate,
            gradient_decorator=negate,
            hessians_decorator=negate_list,
        )
    )


class BoTorchProblem:
    """A UDP wrapper around a BoTorch MultiObjectiveTestProblem"""

    def __init__(
        self,
        problem: _BTMOTestProblem,
        noise: bool = False,
        tkwargs: Optional[dict] = None,
    ):
        self._problem = problem
        self._noise = noise
        self._tkwargs = {**tkwargs} if tkwargs else {}

    @torch.no_grad()
    def fitness(self, x):
        x_pt = torch.from_numpy(x).to(**self._tkwargs)
        return self._problem(x_pt, noise=self._noise).cpu().numpy()

    @torch.no_grad()
    def batch_fitness(self, x):
        x_pt = torch.from_numpy(x).to(**self._tkwargs)
        x_mat = x_pt.reshape(-1, self._problem.num_objectives)
        out_mat = self._problem(x_mat, noise=self._noise)
        return out_mat.ravel().cpu().numpy()

    def get_nobj(self):
        return self._problem.num_objectives

    def get_bounds(self):
        return (
            self._problem.bounds[0].detach().cpu().numpy().copy(),
            self._problem.bounds[1].detach().cpu().numpy().copy(),
        )

    def get_name(self):
        return f"BoTorch problem: {type(self._problem).__name__}"


class BoTorchModel:
    """A UDP wrapper around a BoTorch Model"""

    def __init__(
        self, model: _BTModel, bounds: torch.Tensor, tkwargs: Optional[dict] = None
    ):
        self._botorch_model = model
        self._tkwargs = {**tkwargs} if tkwargs else {}
        self._bounds_tensor = bounds
        self._bounds_tuple = (
            bounds[0].detach().cpu().numpy(),
            bounds[1].detach().cpu().numpy(),
        )

    @torch.no_grad()
    def fitness(self, x):
        x_pt = torch.from_numpy(x).to(**self._tkwargs)
        x_pt = normalize(x_pt, self._bounds_tensor)
        # We need to unsqueeze two a q-batch dimension and a t-batch dimension
        # For some reason, for a ModelListGP the t-batch dimension is not optional
        x_pt = x_pt.unsqueeze(0).unsqueeze(0)
        mean_pt = self._botorch_model.posterior(x_pt).mean
        mean_pt = mean_pt.squeeze(0).squeeze(0)
        return mean_pt.cpu().numpy()

    @torch.no_grad()
    def batch_fitness(self, x):
        x_pt = torch.from_numpy(x).to(**self._tkwargs)
        x_mat = x_pt.reshape(-1, self._botorch_model.num_outputs)
        x_mat = normalize(x_mat, self._bounds_tensor)
        out_mat = self._botorch_model.posterior(x_mat.unsqueeze(-2)).mean.squeeze(-2)
        return out_mat.ravel().cpu().numpy()

    def get_nobj(self):
        return self._botorch_model.num_outputs

    def get_bounds(self):
        # Copy it just in case pygmo modifies it(!)
        return tuple(b.copy() for b in self._bounds_tuple)

    def get_name(self):
        return f"BoTorch model: {type(self._botorch_model).__name__}"
