"""Contains different acquisition function optimisation strategies

An acquisition optimisation strategy bundles an acquisition function with an
optimisation strategy and all configuration required for that strategy. The only
information not present at the time of creation is the model and the linear
scalarisation weights.

The strategies are:
  - C-MOKG: Proposed by this work (Buckingham et al. 2025) and implemented in
    `DiscreteKgOptimisationSpec`
  - HVKG: Proposed by Daulton et al. (2023) and implemented in `HvkgOptimisationSpec`
  - JES: Proposed by Tu et al. (2022) and implemented in `JesOptimisationSpec`

Note that the implementation of JES was modified in jes_sample_pareto.py to avoid
numerical instability in the original implementation.

References:
  - J.M. Buckingham, S. Rojas-Gonzalez, J. Branke. Knowledge Gradient for
    Multi-Objective Bayesian Optimization with Decoupled Evaluations. EMO 2025.
    https://www.doi.org/10.1007/978-981-96-3538-2_9, https://arxiv.org/abs/2302.01310
  - S. Daulton, M. Balandat and E. Bakshy. Hypervolume knowledge gradient: A lookahead
    approach for multi-objective Bayesian optimization with partial information. ICML,
    2023. https://proceedings.mlr.press/v202/daulton23a.html
  - B. Tu, A. Gandy, N. Kantas and B. Shafei. Joint Entropy Search for Multi-Objective
    Bayesian Optimization. Advances in Neural Information Processing Systems, 35. 2022.
    https://proceedings.neurips.cc/paper_files/paper/2022/hash/4086fe59dc3584708468fba0e459f6a7-Abstract-Conference.html
"""

import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Tuple, Union

import torch
from botorch.acquisition import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)
from botorch.models import ModelListGP
from botorch.models.cost import FixedCostModel
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from decoupledbo.modules.acquisition.discretekg import DiscreteKnowledgeGradient
from decoupledbo.modules.acquisition.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from decoupledbo.modules.pareto.jes_sample_pareto import (
    compute_sample_box_decomposition,
    sample_discrete_pareto_optimal_points,
)
from decoupledbo.modules.utils import make_torch_std_grid
from decoupledbo.pipeline.constants import TKWARGS

logger = logging.getLogger(__name__)


class AcquisitionOptimisationSpec(ABC):
    """Strategy for the evaluation and optimisation of the acquisition function"""

    @abstractmethod
    def optimize_for_single_objective(
        self,
        model: ModelListGP,
        costs: Union[Tensor, List],
        input_dim: int,
        *,
        scalarisation_weights: Tensor,
        hv_refpoint: Tensor,
        existing_sampled_x: List[Tensor],
        existing_sampled_obj: List[Tensor],
    ) -> Tuple[Tensor, int, Tensor]:
        """Optimize for evaluation of a single objective

        Not all the keyword-only arguments are used by all acquisition functions.

        Args:
            model: A surrogate model, fitted on normalised inputs (so that the input
                space is [0, 1] in each dimension)
            costs: A 1-dim tensor of costs for evaluating each objective
            input_dim: The number of dimensions in the input space

        Keyword Args:
            scalarisation_weights: A 1-dim tensor of scalarisation weights (only used by
                scalarisation based strategies)
            hv_refpoint: A 1-dim tensor specifying the reference point to use for
                hypervolume calculations
            existing_sampled_x: The x locations of existing samples in normalised space
                (one list element per objective since different objectives may be
                sampled different numbers of times)
            existing_sampled_obj: The objective values of existing samples (one list
                element per objective since different objectives may be sampled
                different numbers of times)

        Returns:
            A tuple `(new_x, new_i, acq_per_cost)` containing the best `x` location and
                objective `i` to evaluate, and the 'Acquisition value per cost' for
                this suggestion.
        """
        pass

    @abstractmethod
    def optimize_for_full_evaluation(
        self,
        model: Model,
        input_dim: int,
        *,
        scalarisation_weights: Tensor,
        hv_refpoint: Tensor,
        existing_sampled_x: List[Tensor],
        existing_sampled_obj: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Optimize for evaluation of all objectives

        Not all the keyword-only arguments are used by all acquisition functions.

        Args:
            model: A surrogate model, fitted on normalised inputs (so that the input
                space is [0, 1] in each dimension)
            input_dim: The number of dimensions in the input space

        Keyword Args:
            scalarisation_weights: A 1-dim tensor of scalarisation weights (only used by
                scalarisation based strategies)
            hv_refpoint: A 1-dim tensor specifying the reference point to use for
                hypervolume calculations
            existing_sampled_x: The x locations of existing samples (one list element
                per objective since different objectives may be sampled different
                numbers of times)
            existing_sampled_obj: The objective values of existing samples (one list
                element per objective since different objectives may be sampled
                different numbers of times)

        Returns:
            A tuple `(new_x, acq_value)` containing the best `x` location to evaluate,
                and the acquisition value for this suggestion.
        """
        pass

    @staticmethod
    def _choose_best_objective(candidates, costs):
        """Choose the best objective for an acquisition function which doesn't natively
        account for cost

        This is not required for acquisition functions which have a cost_aware_utility.

        Args:
            candidates: A tuple (objective_index, candidate_x, objective_value).
            costs: A list/tensor/... containing the costs for each objective
        """

        # Because in some implementations the acquisition function value can
        # (erroneously) come out as negative, we clip to zero here. To break ties when
        # all objectives have come back with negative values, we pick the one with the
        # lowest cost.
        best_i, best_x, best_acq_value = max(
            candidates, key=lambda x: (max(x[-1], 0) / costs[x[0]], -costs[x[0]])
        )
        best_acq_value_per_cost = best_acq_value / costs[best_i]
        return best_i, best_x, best_acq_value_per_cost


class DiscreteKgOptimisationSpec(AcquisitionOptimisationSpec):
    def __init__(
        self,
        n_discretisation_points_per_axis: int,
        num_restarts: int,
        raw_samples: int,
        batch_limit: int,
        max_iter: int,
    ):
        """Optimise the Knowledge Gradient based acquisition function using discrete KG

        The discretisation used is a grid of evenly spaced points.

        Args:
            n_discretisation_points_per_axis: An integer specifying the number of points
                to put along each axis on the grid used to discretise the input space
            num_restarts: The number of restarts to use when optimising
            raw_samples: The number of raw samples to use when choosing the initial
                search locations
            batch_limit: The maximum number of restarts to run in the same optimisation
                with L-BFGS-B
            max_iter: The maximum number of iterations of L-BFGS-B to allow
        """
        super().__init__()
        self.n_discretisation_points_per_axis = n_discretisation_points_per_axis
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_limit = batch_limit
        self.max_iter = max_iter

    def optimize_for_single_objective(
        self,
        model: ModelListGP,
        costs: Union[Tensor, List],
        input_dim: int,
        *,
        scalarisation_weights: Tensor,
        **_unused_kwargs,
    ) -> Tuple[Tensor, int, Tensor]:
        standard_bounds = _get_standard_bounds(input_dim)

        candidates = []
        for i in range(model.num_outputs):
            acq_func = DiscreteKnowledgeGradient(
                model,
                x_discretisation=make_torch_std_grid(
                    self.n_discretisation_points_per_axis, input_dim, TKWARGS
                ),
                scalarisation_weights=scalarisation_weights,
                target_output_ix=i,
            )
            candidate_x, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=1,  # Non-batch mode
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
            )
            if acq_value < 0:
                # Shouldn't happen with Discrete KG, but we include the warning just in
                # case
                logger.warning(
                    "Optimal acquisition function value is negative: "
                    "obj_index=%i, acq_value=%f",
                    i,
                    acq_value,
                )
            candidates.append((i, candidate_x.detach(), acq_value.detach()))

        best_i, best_x, best_kg_per_cost = self._choose_best_objective(
            candidates, costs
        )

        return best_x, best_i, best_kg_per_cost

    def optimize_for_full_evaluation(
        self,
        model: Model,
        input_dim: int,
        *,
        scalarisation_weights: Tensor,
        **_unused_kwargs,
    ) -> Tuple[Tensor, Tensor]:
        standard_bounds = _get_standard_bounds(input_dim)

        acq_func = DiscreteKnowledgeGradient(
            model,
            x_discretisation=make_torch_std_grid(
                self.n_discretisation_points_per_axis, input_dim, TKWARGS
            ),
            scalarisation_weights=scalarisation_weights,
        )
        candidate_x, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,  # Non-batch mode
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
        )
        if acq_value < 0:
            logger.warning(
                "Optimal acquisition function value is negative: acq_value=%f",
                acq_value,
            )

        return candidate_x.detach(), acq_value.detach()


class HvkgOptimisationSpec(AcquisitionOptimisationSpec):
    def __init__(
        self,
        num_pareto: int,
        num_fantasies: int,
        num_restarts: int,
        raw_samples: int,
        curr_opt_num_restarts: int,
        curr_opt_raw_samples: int,
        batch_limit: int,
        max_iter: int,
    ):
        """Optimise the Hypervolume Knowledge Gradient based acquisition function
        (Daulton et al. 2023)

        Args:
            num_pareto: The number of points to use when approximating the Pareto front
            num_fantasies: The number of fantasies to use when estimating the outer
                expectation
            num_restarts: The number of restarts to use when optimising
            raw_samples: The number of raw samples to use when choosing the initial
                search locations
            curr_opt_num_restarts: The number of restarts to use when calculating the
                optimum before the hypothesised observation (the "current optimum").
            curr_opt_raw_samples: The number of raw samples to use when choosing the
                initial search locations while searching for the "current optimum"
            batch_limit: The maximum number of restarts to run in the same optimisation
                with L-BFGS-B
            max_iter: The maximum number of iterations of L-BFGS-B to allow

        References:
            S. Daulton, M. Balandat and E. Bakshy. Hypervolume knowledge
                gradient: A lookahead approach for multi-objective Bayesian
                optimization with partial information. ICML, 2023.
                https://proceedings.mlr.press/v202/daulton23a.html
        """
        super().__init__()
        self.num_pareto = num_pareto
        self.num_fantasies = num_fantasies
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.curr_opt_num_restarts = curr_opt_num_restarts
        self.curr_opt_raw_samples = curr_opt_raw_samples
        self.batch_limit = batch_limit
        self.max_iter = max_iter

    def optimize_for_single_objective(
        self,
        model: ModelListGP,
        costs: Union[Tensor, List],
        input_dim: int,
        *,
        hv_refpoint: Tensor,
        **_unused_kwargs,
    ) -> Tuple[Tensor, int, Tensor]:
        cost_model = FixedCostModel(
            fixed_cost=torch.as_tensor(
                costs,
                dtype=hv_refpoint.dtype,
                device=hv_refpoint.device,
            )
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        standard_bounds = _get_standard_bounds(input_dim)

        current_opt = self._compute_current_optimum(model, hv_refpoint, standard_bounds)

        # If not passed explicitly, then the Sobol' samplers are created automatically.
        # By default, HV-KG creates a different sampler for each objective.
        acq_func = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=hv_refpoint,
            num_fantasies=self.num_fantasies,
            num_pareto=self.num_pareto,
            current_value=current_opt,
            cost_aware_utility=cost_aware_utility,
        )

        objective_candidates = []
        objective_vals = []
        for i in range(model.num_outputs):
            evaluation_mask = torch.zeros(
                1, model.num_outputs, dtype=torch.bool, device=standard_bounds.device
            )
            evaluation_mask[0, i] = 1
            acq_func.X_evaluation_mask = evaluation_mask
            candidate_x, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=1,  # Non-batch mode
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                sequential=False,
                options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
            )
            if acq_value <= 0:
                logger.warning(
                    "Optimal acquisition function value is not strictly positive "
                    "(after clipping to be at least zero): obj_index=%i, acq_value=%f",
                    i,
                    acq_value,
                )
            objective_candidates.append(candidate_x.detach())
            objective_vals.append(acq_value.detach())

        # We don't use self._choose_best_objective because the cost is already accounted
        # for in the acquisition function via the cost_aware_utility.
        best_i = torch.stack(objective_vals).argmax().item()
        best_x = objective_candidates[best_i]
        best_kg_per_cost = objective_vals[best_i]

        return best_x, best_i, best_kg_per_cost

    def optimize_for_full_evaluation(
        self,
        model: Model,
        input_dim: int,
        *,
        hv_refpoint: Tensor,
        **_unused_kwargs,
    ) -> Tuple[Tensor, Tensor]:
        standard_bounds = _get_standard_bounds(input_dim)

        current_opt = self._compute_current_optimum(model, hv_refpoint, standard_bounds)

        # If not passed explicitly, then the Sobol' samplers are created automatically.
        # By default, HV-KG creates a different sampler for each objective.
        acq_func = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=hv_refpoint,
            num_fantasies=self.num_fantasies,
            num_pareto=self.num_pareto,
            current_value=current_opt,
        )

        candidate_x, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,  # Non-batch mode
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
        )
        if acq_value < 0:
            logger.warning(
                "Optimal acquisition function value is negative: acq_value=%f",
                acq_value,
            )

        return candidate_x.detach(), acq_value.detach()

    def _compute_current_optimum(self, model, ref_point, bounds):
        """Compute the hypervolume of the current hypervolume maximising set"""
        pmean_acqf = _get_hv_value_function(
            model=model,
            ref_point=ref_point,
            use_posterior_mean=True,
        )
        _, current_optimum = optimize_acqf(
            acq_function=pmean_acqf,
            bounds=bounds,
            q=self.num_pareto,
            num_restarts=self.curr_opt_num_restarts,
            raw_samples=self.curr_opt_raw_samples,
            return_best_only=True,
            options={"batch_limit": self.batch_limit},
        )
        return current_optimum


class JesOptimisationSpec(AcquisitionOptimisationSpec):
    def __init__(
        self,
        estimation_type: str,
        num_pareto_samples: int,
        num_pareto_points: int,
        num_restarts: int,
        raw_samples: int,
        batch_limit: int,
        max_iter: int,
    ):
        super().__init__()
        self.estimation_type = estimation_type
        self.num_pareto_samples = num_pareto_samples
        self.num_pareto_points = num_pareto_points
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_limit = batch_limit
        self.max_iter = max_iter

    def optimize_for_single_objective(
        self,
        model: ModelListGP,
        costs: Union[Tensor, List],
        input_dim: int,
        *,
        existing_sampled_x: List[Tensor],
        existing_sampled_obj: List[Tensor],
        **_unused_kwargs,
    ) -> Tuple[Tensor, int, Tensor]:
        # Generate a set of points on the Pareto front of each GP sample. Note that
        # because the Pareto front could be degenerate in some samples and not others,
        # not all the elements of `pareto_fronts` need contain the same number of
        # points.
        standard_bounds = _get_standard_bounds(input_dim)
        pareto_sets, pareto_fronts = sample_discrete_pareto_optimal_points(
            model,
            standard_bounds,
            num_samples=self.num_pareto_samples,
            target_num_points=self.num_pareto_points,
        )

        hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)

        candidates = []
        for i in range(model.num_outputs):
            acq_func = qLowerBoundMultiObjectiveJointEntropySearch(
                model,
                pareto_sets,
                pareto_fronts,
                hypercell_bounds,
                estimation_type=self.estimation_type,
                target_output_ix=i,
            )
            candidate_x, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=1,  # Non-batch mode
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
            )
            candidates.append((i, candidate_x.detach(), acq_value.detach()))

        best_i, best_x, best_acqval_per_cost = self._choose_best_objective(
            candidates, costs
        )

        return best_x, best_i, best_acqval_per_cost

    def optimize_for_full_evaluation(
        self,
        model: Model,
        input_dim: int,
        *,
        existing_sampled_x: List[Tensor],
        existing_sampled_obj: List[Tensor],
        **_unused_kwargs,
    ) -> Tuple[Tensor, Tensor]:
        standard_bounds = _get_standard_bounds(input_dim)
        pareto_sets, pareto_fronts = sample_discrete_pareto_optimal_points(
            model,
            standard_bounds,
            num_samples=self.num_pareto_samples,
            target_num_points=self.num_pareto_points,
        )

        hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)

        acq_func = qLowerBoundMultiObjectiveJointEntropySearch(
            model,
            pareto_sets,
            pareto_fronts,
            hypercell_bounds,
            estimation_type=self.estimation_type,
        )
        candidate_x, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,  # Non-batch mode
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": self.batch_limit, "maxiter": self.max_iter},
        )

        return candidate_x.detach(), acq_value.detach()


def _get_standard_bounds(dim):
    standard_bounds = torch.zeros(2, dim, **TKWARGS)
    standard_bounds[1] = 1
    return standard_bounds
