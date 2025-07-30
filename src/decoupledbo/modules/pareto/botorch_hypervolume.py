"""
A module for estimating hypervolumes from the posterior mean using botorch's
DominatedPartitioning
"""

import torch
from botorch.exceptions import BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from torch import Tensor


def estimate_hypervolume_from_posterior_mean(
    pareto_set, pareto_front, true_problem, ref_point, *, tkwargs=None
):
    """Return bounds on the hyper-volume of the pareto front suggested by the model

    Using the model posterior mean, we obtain a Pareto set and Pareto front. When a
    decision maker picks a point on this predicted Pareto front, we will return them a
    corresponding point in the Pareto front which we predict will yield this outcome.
    However, if the surrogate has not mapped the true objective function very well, then
    the true value may be different. It is therefore instructive to look at the image of
    the suggested Pareto set and compare it to the suggested Pareto front.

    This function returns lower and (estimated) upper bounds for:
      - the hyper-volume of the Pareto front;
      - the hyper-volume of the true image of the Pareto set.
    """

    pareto_set = torch.from_numpy(pareto_set).to(**(tkwargs or {}))
    pareto_front = torch.from_numpy(pareto_front).to(**(tkwargs or {}))

    pareto_set_image = true_problem(pareto_set, noise=False)

    pfront_hv_lo, pfront_hv_hi = estimate_hypervolume(pareto_front, ref_point)
    pset_hv_lo, pset_hv_hi = estimate_hypervolume(pareto_set_image, ref_point)

    return {
        "pfront_hv_lo": pfront_hv_lo,
        "pfront_hv_hi": pfront_hv_hi,
        "pset_hv_lo": pset_hv_lo,
        "pset_hv_hi": pset_hv_hi,
    }


def calculate_reference_point(pareto_front: Tensor, buffer=0.01):
    """Return a suitable reference point for the given Pareto front

    This is computed as the minimum point in the pareto front, minus a buffer specified
    as a proportion of the range of the pareto front in each dimension (default 1%)

    Args:
        pareto_front: An n-by-d Tensor of points representing the pareto front
        buffer: The proportion of the range in each direction add to the reference point
    """
    if not pareto_front.ndim == 2:
        raise BotorchTensorDimensionError(
            f"Expected pareto_front to have 2-dimensions. Got {pareto_front.ndim=}."
        )

    min_vec = pareto_front.min(dim=-2).values
    max_vec = pareto_front.max(dim=-2).values
    ref_point = min_vec - buffer * (max_vec - min_vec)
    return ref_point


def estimate_hypervolume(pareto_front: Tensor, ref_point: Tensor, return_upper=True):
    """Calculate lower and upper bounds for the dominated hyper-volume.

    Use a sample of the continuous Pareto front to calculate lower and upper bounds for
    the dominated hyper-volume (assuming maximization).

    The upper bound is calculated using the volume of the complement of the set of
    points in the negative problem, within a bounding box determined by the reference
    point and maximum limits of the points sampled on the pareto_front.
    """

    # Calculate the lower bound
    bd_lo = DominatedPartitioning(ref_point, pareto_front)
    volume_lower = bd_lo.compute_hypervolume().item()

    if return_upper:
        # Calculate the upper bound (approx, since the ideal point is not known!)
        ideal, _ = pareto_front.max(dim=0)
        dominates_ref_pt = (pareto_front > ref_point).all(dim=-1)
        if not dominates_ref_pt.any():
            volume_upper = 0
        else:
            bd_hi = DominatedPartitioning(-ideal, -pareto_front[dominates_ref_pt])
            volume_upper_complement = bd_hi.compute_hypervolume().item()
            volume_box = torch.prod(ideal - ref_point).item()
            volume_upper = volume_box - volume_upper_complement

        return volume_lower, volume_upper
    else:
        return volume_lower
