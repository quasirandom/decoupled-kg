import random
from numbers import Real
from typing import List, Optional, Tuple

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils import draw_sobol_samples
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from torch import Tensor

from decoupledbo.modules.pareto.botorch_hypervolume import (
    calculate_reference_point,
    estimate_hypervolume,
)
from decoupledbo.modules.pareto.sample import (
    BoTorchModel,
    sample_points_on_pareto_front,
)

N_PARETO_POINTS = 1000
NOISE_VARIANCE = 1e-8
""" The (almost zero) noise variance to use for the underlying GP model.

We choose something small but non-zero for numerical stability.

Note that this is NOT the observation noise on the test problem. Rather it is the noise
used when fitting the GP to the sampled data to get noiseless test problem.
"""


class GPTestProblem(MultiObjectiveTestProblem):
    """A multi-objective test problem which approximates a GP sample

    The approximation is formed by sampling the GP at a discrete number of points, then
    interpolating between these points using the posterior mean.
    """

    def __init__(
        self,
        gp_model: GPyTorchModel,
        *,
        bounds: List[Tuple[float, float]],
        ref_point: List[float],
        max_hv: float,
        noise_stds: Optional[torch.Tensor] = None,
        negate: bool = False,
    ):
        # These attributes are used by super().__init__ converted to buffers in
        # super().__init__()
        self.dim = len(bounds)
        self.num_objectives = gp_model.num_outputs
        self._bounds = bounds
        self._ref_point = ref_point
        self._max_hv = max_hv

        # WARNING: The MultiObjectiveTestProblem and superclasses type-hint this as a
        # float, but we are passing a tensor. However, in the current version of
        # botorch, this does what we want and scales the noise in each objective
        # separately.
        super().__init__(noise_std=noise_stds, negate=negate)

        if gp_model.training:
            raise RuntimeError("GP model must be in evaluation mode.")

        # Really we should be remembering the training data in the buffer! But botorch
        # doesn't do this and I don't know how to make it (and I don't need this
        # feature!)
        self.gp_model = gp_model

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.dim() <= 1:
            # Sometimes the gp_model.posterior introduces a batch dimension which messes
            # up the output. The behaviour is correct if the input has a t-batch
            # dimension. We *could* write something fancier, if necessary, but in order
            # to be robust to changes, we would need to understand better why some
            # gp_models introduce a t-batch dimension.
            raise RuntimeError("Need to supply a batch dimension!")

        # We squeeze and unsqueeze because posterior() requires a q-batch dimension
        X = X.unsqueeze(-2)

        # For some reason, botorch enables gradients by default in the posterior. We do
        # not want this (unless perhaps X.requires_grad, though whether you get the
        # right gradients in this case is untested!)
        if X.requires_grad:
            posterior = self.gp_model.posterior(X)
        else:
            with torch.no_grad():
                posterior = self.gp_model.posterior(X)

        return posterior.mean.squeeze(-2)


class GPTestProblemModel(ModelListGP):
    @classmethod
    def create_without_data(cls, input_dim, n_objectives, *, dtype=None, device=None):
        gp_model = cls(
            input_dim,
            n_objectives,
            train_x=torch.zeros(0, input_dim, dtype=dtype, device=device),
            train_y=torch.zeros(0, n_objectives, dtype=dtype, device=device),
        )
        gp_model.eval()
        gp_model.to(device=device, dtype=dtype)
        return gp_model

    @classmethod
    def reconstruct(cls, state_dict, input_samples, output_samples):
        """Reconstruct a GPTestProblemModel from training data and hyper-parameters

        Because the training data is not part of the state_dict, it is not enough to use
        model.load_state_dict().
        """
        input_dim = input_samples.shape[-1]
        n_objectives = output_samples.shape[-1]

        model = cls(input_dim, n_objectives, input_samples, output_samples)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def __init__(self, input_dim, n_objectives, train_x, train_y):
        models = []
        for i in range(n_objectives):
            gp = SingleTaskGP(
                train_X=torch.clone(train_x),
                train_Y=torch.clone(train_y[..., i].unsqueeze(-1)),
                likelihood=GaussianLikelihood(
                    noise_constraint=GreaterThan(NOISE_VARIANCE)
                ),
                mean_module=ConstantMean(),
                covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim)),
            )
            # Set the noise to (practically) zero so that when we sample the model, we
            # do so without noise. We can add observation noise to the GPTestProblem
            # later if desired.
            gp.likelihood.noise = NOISE_VARIANCE
            models.append(gp)

        super().__init__(*models)

    def set_hyperparameters(
        self, length_scales, output_scales, means, *, dtype=None, device=None
    ):
        for i, gp in enumerate(self.models):
            gp.mean_module.initialize(
                constant=torch.as_tensor(means[i], dtype=dtype, device=device)
            )
            gp.covar_module.outputscale = output_scales[i]
            gp.covar_module.base_kernel.lengthscale = length_scales[i]


def create_gp_problem_model(
    bounds: List[Tuple[float, float]],
    n_objectives: int,
    length_scales: List[Real],
    output_scales: List[Real],
    means: List[Real],
    nsamples: int = 100,
    input_seed: Optional[int] = None,
    output_seeds: Optional[List[int]] = None,
    dtype=None,
    device=None,
):
    """Create a GPTestProblem, sample it and condition on the result

    The finite sample used to create the "fantasy model" is also returned so that it can
    be serialized.
    """

    input_dim = len(bounds)
    prior_model = GPTestProblemModel.create_without_data(
        input_dim, n_objectives, dtype=dtype, device=device
    )
    prior_model.set_hyperparameters(
        length_scales, output_scales, means, dtype=dtype, device=device
    )
    prior_model.eval()

    bounds_tensor = bounds_to_tensor(bounds, device=device, dtype=dtype)

    # We have two sources of randomness - where to sample and the values taken by the
    # GP. For this we use two random number generators.
    if input_seed is None:
        input_seed = random.randint(0, 1_000_000)
    if output_seeds is None:
        output_seeds = [random.randint(0, 1_000_000) for _ in range(n_objectives)]

    input_samples = draw_sobol_samples(bounds_tensor, nsamples, q=1, seed=input_seed)
    input_samples = input_samples.squeeze(-2)

    # We use no_grad here else gradients are tracked inside. This in turn causes
    # problems when we try to use pygmo (because a deepcopy fails at some point).
    with torch.no_grad():
        # We do not use .fantasize() since we need to be able to save the sampled GP.
        # However, here we are doing exactly the job of .fantasize() - that is, sampling
        # the GP and then conditioning on the result.
        #
        # We also do this separately for each objective to give more control to the
        # caller when constructing the problem. This relies on the objectives being
        # independent, which is ensured by using a subclass of GPModelList.
        output_samples = []
        for i in range(n_objectives):
            sampler = SobolQMCNormalSampler(1, seed=output_seeds[i])
            posterior = prior_model.posterior(
                input_samples, observation_noise=False, output_indices=[i]
            )
            output_samples_i = sampler(posterior).squeeze(0)
            output_samples.append(output_samples_i)
        output_samples = torch.cat(output_samples, dim=-1)

    posterior_model = GPTestProblemModel(
        input_dim, n_objectives, input_samples, output_samples
    )
    posterior_model.set_hyperparameters(
        length_scales, output_scales, means, dtype=dtype, device=device
    )
    posterior_model.eval()

    return posterior_model, input_samples, output_samples


def bounds_to_tensor(bounds: List[Tuple[float, float]], *, device=None, dtype=None):
    bounds_tensor = torch.tensor(bounds, dtype=dtype, device=device)
    bounds_tensor = bounds_tensor.transpose(-1, -2)
    return bounds_tensor


def estimate_reference_point_and_hypervolume(
    model, bounds: Tensor, *, device=None, dtype=None
):
    _, pareto_front_np = sample_points_on_pareto_front(
        BoTorchModel(model, bounds), maximize=True, npoints=N_PARETO_POINTS
    )
    pareto_front = torch.from_numpy(pareto_front_np).to(device=device, dtype=dtype)

    ref_point = calculate_reference_point(pareto_front)
    hv_lo = estimate_hypervolume(pareto_front, ref_point, return_upper=False)

    return ref_point, hv_lo
