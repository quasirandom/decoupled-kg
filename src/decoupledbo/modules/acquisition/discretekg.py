"""Discrete implementation of the multi-objective knowledge-gradient

`DiscreteKnowledgeGradient` is an implementation of the discrete knowledge gradient.
This doesn't have the problem of getting stuck in local minima like the one-shot version
because the inner maximisation is over a discrete set. However, it will not scale to
higher dimensional problems because the discretisation required will be too large.
"""

import logging
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import BotorchTensorDimensionError, UnsupportedError
from botorch.models import ModelListGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.utils import draw_sobol_samples, t_batch_mode_transform
from torch import Tensor

logger = logging.getLogger(__name__)


class DiscreteKnowledgeGradient(AcquisitionFunction):
    """
    Discrete knowledge gradient.

    Adapted from
    https://github.com/JuanUngredda/botorch/blob/main_experiment_manager/botorch/acquisition/analytic.py
    """

    @classmethod
    def create_with_sobol_sample(
        cls,
        model: Model,
        bounds: Tensor,
        num_discrete_points: int,
        scalarisation_weights: Optional[Tensor] = None,
        target_output_ix: Optional[int] = None,
    ):
        """Create a DiscreteKnowledgeGradient with a standard discretisation

        Args:
            model: A fitted model.
            bounds: A `2 x d` tensor of lower and upper bounds for each input dimension
            num_discrete_points: The number of points to use in the discretisation of
                the input (X) space.
            scalarisation_weights: An nscalar-by-d tensor with each row specifying the
                weights for a linear scalarisation of the multi-output model (only
                required for multi-output models)
            target_output_ix: The target output. If using a multi-output model,
                specifying a target output will calculate the KG assuming only this
                output is observed.
        """
        x_discretisation = draw_sobol_samples(bounds, num_discrete_points, q=1)
        x_discretisation = x_discretisation.squeeze(1)  # Remove q-batch dimension
        x_discretisation = x_discretisation.to(bounds)

        return cls(model, x_discretisation, scalarisation_weights, target_output_ix)

    def __init__(
        self,
        model: Model,
        x_discretisation: Tensor,
        scalarisation_weights: Optional[Tensor] = None,
        target_output_ix: Optional[int] = None,
    ):
        r"""
        Discrete Knowledge Gradient

        For multi-output models, this acquisition function uses linear scalarisation to
        convert to a scalar value. Taking expectation over scalarisations is supported
        by averaging over a finite number of distinct scalarisations supplied.

        MOKG(x,d) = E[ max_x' E[f(x') . w | w, f_d(x') + eps] ]
                                                        - E[ max_x' E[f(x') . w | w] ]

        Args:
            model: A fitted model.
            x_discretisation: A k-by-d tensor of `k` design points that will
                approximate the continuous space with a discretisation.
            scalarisation_weights: An nscalar-by-d tensor with each row specifying the
                weights for a linear scalarisation of the multi-output model (only
                required for multi-output models)
            target_output_ix: The target output. If using a multi-output model,
                specifying a target output will calculate the KG assuming only this
                output is observed.
        """
        super().__init__(model=model)

        if x_discretisation.dim() != 2:
            raise BotorchTensorDimensionError(
                f"Expected 'x_discretisation' to have two dimensions. "
                f"Got {x_discretisation.dim()=}."
            )

        if scalarisation_weights is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Models with more than one output must specify "
                    "'scalarisation_weights'."
                )
            else:
                scalarisation_weights = torch.tensor([[1.0]]).to(x_discretisation)

        if scalarisation_weights.dim() != 2:
            raise BotorchTensorDimensionError(
                f"Expected 'scalarisation_weights' to have two dimensions: The first "
                f"indexing different scalarisations to be averaged over and the second "
                f"indexing coordinates of the objective space. "
                f"Got {scalarisation_weights.dim()=}"
            )
        if scalarisation_weights.shape[-1] != model.num_outputs:
            raise BotorchTensorDimensionError(
                f"Expected the last dimension of 'scalarisation_weights' to have one "
                f"element per objective. Got {scalarisation_weights.shape[-1]=} != "
                f"{model.num_outputs}=model.num_outputs."
            )

        self.x_discretisation = x_discretisation
        self.scalarisation_weights = scalarisation_weights
        self.target_output_ix = target_output_ix

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        # This is copied from AnalyticAcquisitionFunction
        raise UnsupportedError(
            f"{type(self).__name__} does not account for X_pending yet."
        )

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        # Most of the logic here is simply to process the t-batch in a for loop.

        batch_shape, d = X.shape[:-2], X.shape[-1]

        if d != self.x_discretisation.shape[-1]:
            raise RuntimeError(
                f"Expected X to have last dimension matching 'self.x_discretisation'. "
                f"Got {X.shape[-1]=}, {self.x_discretisation.shape[-1]=}."
            )

        kgvals = torch.zeros(batch_shape.numel(), dtype=X.dtype, device=X.device)

        for i, xnew in enumerate(X.reshape(-1, d)):
            if self.target_output_ix is not None:
                kgvals[i] = calculate_discrete_kg_conditioning_on_single_output(
                    self.model,
                    xnew,
                    self.target_output_ix,
                    self.x_discretisation,
                    self.scalarisation_weights,
                )
            else:
                kgvals[i] = calculate_discrete_kg(
                    self.model, xnew, self.x_discretisation, self.scalarisation_weights
                )

        return kgvals.reshape(batch_shape)


def calculate_discrete_kg(model: Model, xnew, discretisation, scalarisation_weights):
    """Compute the discrete knowledge gradient

    Only linear scalarisations of multi-output models are supported.

    Args:
        model: A fitted model.
        xnew: A 1D tensor for the x-location at which to evaluate the KG.
        discretisation: A Nxd tensor of x-locations making up the discretisation of the
            input space.
        scalarisation_weights: An nscalar-by-d tensor with each row specifying the
            weights for a linear scalarisation of the multi-output model
    """
    if scalarisation_weights.dim() != 2:
        raise BotorchTensorDimensionError(
            "Expected 'scalarisation_weights' to have two dimensions: The first "
            "indexing different scalarisations to be averaged over and the second "
            "indexing coordinates of the objective space."
        )

    posterior = model.posterior(
        torch.cat([xnew.unsqueeze(0), discretisation]), observation_noise=False
    )
    posterior_xnew_noisy = model.posterior(xnew.unsqueeze(0), observation_noise=True)

    if not isinstance(posterior, GPyTorchPosterior):
        # Hopefully this will also ensure that the likelihood used is a form of
        # Gaussian noise (or behaves like it!).
        raise UnsupportedError(
            f"Only models returning a GPyTorchPosterior are supported. "
            f"Got {type(posterior)=}."
        )

    # We loop over the full function because we are passing in a different
    # ScalarizedPosteriorTransform for each set of weights. For linear scalarisations,
    # it would likely be faster to calculate the posterior once then vectorise.
    nscalar = scalarisation_weights.shape[0]
    kg = torch.zeros(nscalar).to(scalarisation_weights)
    for j in range(nscalar):
        posterior_transform = ScalarizedPosteriorTransform(scalarisation_weights[j])
        scalarised_posterior = posterior_transform(posterior)
        scalarised_posterior_xnew_noisy = posterior_transform(posterior_xnew_noisy)

        # Now extract:
        #   1) Mean at each point in the discretisation
        #   2) Covariances between values at xnew and each point in the discretisation
        #   3) Predictive variance at xnew
        # Additionally, we include the entries for xnew in these vectors to simulate
        # xnew being included in the discretisation.
        mean = scalarised_posterior.mean.squeeze(-1)
        covariances = scalarised_posterior.mvn.covariance_matrix[0]
        xnew_variance = scalarised_posterior_xnew_noisy.mvn.covariance_matrix[0, 0]

        # As the fantasy sample ynew varies, the posterior mean varies linearly
        #    m = mean + covariances @ xnew_variance**(-1) @ (ynew - E[ynew])
        # If we reparametrize as
        #    ynew = sqrt(xnew_variance) @ znew + E[ynew]
        # where 'znew' is a standard normal rv, then we get
        #    m = mean + covariances @ sqrt(xnew_variance)**(-1) @ znew
        # The intercept of this line is 'mean', while the slope is
        # 'covariances / sqrt(xnew_variance)'.
        znew_coefficients = covariances / xnew_variance.sqrt()

        epi_i, epi_x = calculate_epigraph_indices(
            intercepts=mean, slopes=znew_coefficients
        )

        epi_expectation = calculate_expected_value_of_piecewise_linear_function(
            intercepts=mean[epi_i], slopes=znew_coefficients[epi_i], boundaries=epi_x
        )

        kg[j] = epi_expectation - torch.max(mean)

    return kg.mean()


def calculate_discrete_kg_conditioning_on_single_output(
    model: ModelListGP, xnew, obj_idx_new, discretisation, scalarisation_weights
):
    """Compute the discrete knowledge gradient

    Only linear scalarisations are supported.

    Args:
        model: A fitted model.
        xnew: A 1D tensor for the x-location at which to evaluate the KG.
        obj_idx_new: The index of the objective at which to evaluate the KG.
        discretisation: A Nxd tensor of x-locations making up the discretisation of the
            input space.
        scalarisation_weights: A nscalar-by-d tensor of weights for linear
            scalarisations of the multi-output model.

    Note:
        This implementation is very similar to that of calculate_discrete_kg(), except
        for the handling of posteriors.
    """

    if scalarisation_weights.dim() != 2:
        raise BotorchTensorDimensionError(
            "Expected 'scalarisation_weights' to have two dimensions: The first "
            "indexing different scalarisations to be averaged over and the second "
            "indexing coordinates of the objective space."
        )

    # We query posteriors for each model individually, since gpytorch's multi-task MVN
    # does not expose a suitable interface for us (the representation of the covariance
    # matrix can vary and determining what it is requires using the private _interleaved
    # attribute).
    if not isinstance(model, ModelListGP):
        raise UnsupportedError(
            f"Input 'model' must be a 'ModelListGP'. Got {type(model)=}."
        )

    posteriors = [
        m.posterior(
            torch.cat([xnew.unsqueeze(0), discretisation]),
            observation_noise=False,
        )
        for m in model.models
    ]
    posteriors_xnew_noisy = [
        m.posterior(xnew.unsqueeze(0), observation_noise=True) for m in model.models
    ]
    if not all(isinstance(p, GPyTorchPosterior) for p in posteriors):
        # Hopefully this will also ensure that the likelihood used is a form of Gaussian
        # noise (or behaves like it!).
        raise UnsupportedError(
            f"Only models returning a GPyTorchPosterior are supported. "
            f"Got {[type(p) for p in posteriors]=}."
        )

    # Now extract:
    #   1) Mean at each point in the discretisation for each objective
    #   2) Covariances between values at xnew and each point in the discretisation, only
    #      for objective obj_idx_new
    #   3) Predictive variance at xnew for objective obj_idx_new
    # Additionally, we include the entries for xnew in these vectors to simulate xnew
    # being included in the discretisation.
    means = torch.cat([p.mean for p in posteriors], dim=-1)
    covariances_i = posteriors[obj_idx_new].mvn.covariance_matrix[0]
    xnew_variance_i = posteriors_xnew_noisy[obj_idx_new].mvn.covariance_matrix[0, 0]

    # As the fantasy sample ynew_i varies, the posterior mean for objective i varies
    # linearly (and the posterior mean for other objectives is unaffected)
    #    m_i = means_i + covariances_i @ xnew_variance_i**(-1) @ (ynew_i - E[ynew_i])
    # If we reparametrize as
    #    ynew_i = sqrt(xnew_variance_i) @ znew + E[ynew_i]
    # where 'znew' is a standard normal rv, then we get
    #    m_i = means_i + covariances_i @ sqrt(xnew_variance_i)**(-1) @ znew
    # The intercept of this line is 'means_i', while the slope is
    # 'covariances_i / sqrt(xnew_variance_i)'.
    znew_coefficients = covariances_i / xnew_variance_i.sqrt()

    # Applying the scalarisations, we get a straight line with intercept
    # 'means @ lambda' and slope 'lambda_i * znew_coefficients'
    # The resulting tensors have shape nscalar-by-d.
    d = scalarisation_weights.shape[-1]
    weights = scalarisation_weights.view(-1, 1, d)
    intercepts = torch.sum(weights * means, dim=-1)
    slopes = weights[..., obj_idx_new] * znew_coefficients

    # For each scalarisation, calculate the KG, then average.
    # We use a for loop because we cannot vectorise calculate_epigraph_indices. It is
    # possible to vectorise calculate_expected_value_of_piecewise_linear_function but I
    # haven't profiled the code to see if it's worth it.
    nscalar = scalarisation_weights.shape[0]
    kg = torch.zeros(nscalar).to(scalarisation_weights)
    for j in range(nscalar):
        epi_i, epi_x = calculate_epigraph_indices(intercepts[j], slopes[j])

        epi_expectation = calculate_expected_value_of_piecewise_linear_function(
            intercepts[j, epi_i], slopes[j, epi_i], epi_x
        )

        kg[j] = epi_expectation - torch.max(intercepts[j])

    return kg.mean()


def calculate_epigraph_indices(intercepts: Tensor, slopes: Tensor):
    """Calculate the epigraph of a set of lines

    The epigraph is the piecewise linear curve given by the pointwise maximum of the
    lines.

    Args:
        intercepts: The vector of n y-intercepts of the lines
        slopes: The vector of n slopes of the lines

    Returns:
        Tuple: A tuple containing
            - indices: The m indices of the lines in the epigraph, ordered from left to
                right
            - intersections: The m-1 points at which successive lines in the epigraph
                intersect (i.e. where the line which is maximum changes).
    """

    _verify_intercepts_and_slopes(intercepts, slopes)

    device = slopes.device
    # Short-circuit for case when all slopes are identical
    if torch.all(torch.abs(slopes) < 1e-9):
        logger.debug("Short-circuit epigraph since slopes are almost zero")
        _, indices = torch.max(intercepts, dim=0, keepdim=True)
        intersections = torch.tensor([], device=device, dtype=torch.double)
        return indices, intersections

    # Sort by ascending slope, then descending intercept
    original_indices = torch.arange(len(intercepts), device=device)
    intercepts, ix1 = torch.sort(intercepts, descending=True)
    slopes, ix2 = torch.sort(slopes[ix1], descending=False, stable=True)
    intercepts = intercepts[ix2]
    original_indices = original_indices[ix1[ix2]]

    # Calculate the epigraph (done in loop to avoid large v. matrix if there are many
    # lines)
    n_lines = len(intercepts)
    indices = [0]
    intersections = []
    i = 0
    while i < n_lines - 1:
        # Ignore intersections with lines with identical slope (we will already have the
        # one with the largest intercept). We filter these out before calculating
        # intersections_x since otherwise the nan values propagate to the gradient
        # calculation (even though those indices are not chosen for the next
        # intersection).
        [diff_slope_ix] = torch.nonzero(slopes[i] != slopes[i + 1 :], as_tuple=True)
        if len(diff_slope_ix) == 0:
            break

        # Note it's important to start from i+1 both here and when we calculate
        # diff_slope_ix
        j = torch.arange(i + 1, n_lines + 1)[diff_slope_ix]
        intersections_x = -(intercepts[i] - intercepts[j]) / (slopes[i] - slopes[j])
        first_intersection_ix = torch.argmin(intersections_x)

        indices.append(j[first_intersection_ix])
        intersections.append(intersections_x[first_intersection_ix])

        i = indices[-1]

    indices = torch.tensor(indices, device=device)
    if intersections:
        intersections = torch.stack(intersections)
    else:
        intersections = torch.tensor([], device=device, dtype=torch.double)

    # Convert back to original indices (before sorting and filtering)
    indices = original_indices[indices]

    return indices, intersections


def calculate_expected_value_of_piecewise_linear_function(
    intercepts: Tensor, slopes: Tensor, boundaries: Tensor
):
    """
    Calculate the expected value of a piecewise linear function of a standard normal rv

    That is, calculate E[f(Z)] where Z is a standard normal random variable and f is
    piecewise linear.
    """
    _verify_intercepts_and_slopes(intercepts, slopes)
    if boundaries.shape != (len(intercepts) - 1,):
        raise BotorchTensorDimensionError(
            f"Expected 'boundaries' to be a one-dimensional tensor with "
            f"{len(intercepts)} elements. Got {boundaries.shape=}."
        )

    device = boundaries.device
    boundaries = torch.cat(
        [
            torch.tensor([-torch.inf], device=device, dtype=boundaries.dtype),
            boundaries,
            torch.tensor([torch.inf], device=device, dtype=boundaries.dtype),
        ]
    )
    normal_dist = torch.distributions.Normal(0, 1)

    # Not possible to get pdf directly so must use log_prob()
    pdf = torch.exp(normal_dist.log_prob(boundaries))
    cdf = normal_dist.cdf(boundaries)

    # This formula can be easily checked since for indefinite integrals
    #    \int (a + b*x) pdf(x) dx = a * \int pdf(x) dx + b * \int x*pdf(x) dx
    #                             = a * cdf(x)         - b * pdf(x)
    # where the term proportional to b uses the formula for a standard normal pdf.
    expectation = torch.sum(
        intercepts * (cdf[1:] - cdf[:-1]) - slopes * (pdf[1:] - pdf[:-1])
    )
    return expectation


def _verify_intercepts_and_slopes(intercepts, slopes):
    if intercepts.dim() != 1 or slopes.dim() != 1:
        raise BotorchTensorDimensionError(
            f"Expected 'intercepts' and 'slopes' to both be one-dimensional tensors. "
            f"Got {intercepts.dim()=} and {slopes.dim()=}."
        )
    if intercepts.shape != slopes.shape:
        raise BotorchTensorDimensionError(
            f"Expected 'intercepts' and 'slopes' to have the same shape. "
            f"Got {intercepts.shape=} and {slopes.shape=}."
        )
    if intercepts.shape[-1] == 0:
        raise ValueError(
            f"Expected inputs to specify at least one line. "
            f"Got {intercepts.shape[-1]=}."
        )
