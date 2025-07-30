"""Contains a factory for building surrogate models from data and config"""

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

MIN_NOISE_SE = 1e-2
"""The minimum noise standard deviation for a _fitted_ noise GP

THIS MUST BE THE SAME AS MIN_INFERRED_NOISE_LEVEL INSIDE BOTORCH
"""
MIN_NOISE_SE_FIXED = 1e-4
"""The minimum noise standard deviation for a _fixed_ noise GP (almost zero)"""


def build_mll_and_model(
    config: dict, train_x: list[Tensor], train_obj: list[Tensor], tkwargs=None
):
    """Build a model and associated marginal log likelihood from model config

    WARNING: This function doesn't read all the fields passed in through the config
        dict. In particular, fix_zero_noise is currently ignored, which is misleading.
        This field is really to do with how you use the model once created (fitting
        hyper-parameters etc.). I haven't decided yet whether to incorporate it in the
        logic here, or whether to restructure the config file to make these separate
        sections.
    """

    bounds = torch.tensor(config["bounds"], **(tkwargs or {}))

    output_dim = len(config["outputs"])

    min_noise_se = (
        MIN_NOISE_SE_FIXED if config["fit_hyperparams"] == "never" else MIN_NOISE_SE
    )

    models = []
    for i in range(output_dim):
        try:
            gp = build_objective(
                config["outputs"][i], min_noise_se, train_x[i], train_obj[i], bounds
            )
        except Exception as ex:
            msg = f"Exception raised when parsing config for objective {i}: {str(ex)}"
            raise Exception(msg) from ex
        models.append(gp)

    model = ModelListGP(*models).to(**(tkwargs or {}))

    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return mll, model


def build_objective(config, min_noise_se, train_x, train_y, bounds):
    """Build a single objective"""
    train_x = normalize(train_x, bounds)
    train_y = train_y.unsqueeze(-1)

    input_dim = train_x.shape[-1]
    batch_shape = train_x.shape[:-2]

    likelihood = build_likelihood(config["likelihood"], min_noise_se, batch_shape)
    mean_module = ConstantMean(batch_shape=batch_shape)
    covar_module = build_covar_module(config["kernel"], input_dim, batch_shape)

    if config["standardize_output"]:
        outcome_transform = Standardize(m=1)
    else:
        outcome_transform = None

    gp = SingleTaskGP(
        train_x,
        train_y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module,
        outcome_transform=outcome_transform,
    )
    return gp


def build_likelihood(config, min_noise_se, batch_shape):
    """Build a Gaussian likelihood module"""

    # If missing from config then return None to trigger the default likelihood when
    # creating the SingleTaskGP.
    if config is None:
        return None

    prior = build_prior(config["noise_prior"])
    # Setting transform=None stops the constraint being enforced.
    # https://github.com/cornellius-gp/gpytorch/issues/2242#issuecomment-1399303074
    constraint = GreaterThan(
        min_noise_se**2, transform=None, initial_value=prior.mode
    )
    return GaussianLikelihood(
        noise_prior=prior, noise_constraint=constraint, batch_shape=batch_shape
    )


def build_covar_module(config, input_dim, batch_shape):
    # If missing from config then return None to trigger the default covar_module when
    # creating the SingleTaskGP.
    if config is None:
        return None

    kernel_catalog = {"matern": MaternKernel, "rbf": RBFKernel}

    if config["type"] not in kernel_catalog:
        raise ValueError(f"Unrecognised kernel 'type'. Got {config['type']!r}")
    else:
        kernel_type = kernel_catalog[config["type"]]

    outputscale_prior = build_prior(config["outputscale_prior"])
    lengthscale_prior = build_prior(config["lengthscale_prior"])

    inner_kernel = kernel_type(
        **config["args"],
        ard_num_dims=input_dim if config["ard"] else None,
        lengthscale_prior=lengthscale_prior,
        batch_shape=batch_shape,
    )

    return ScaleKernel(
        inner_kernel, outputscale_prior=outputscale_prior, batch_shape=batch_shape
    )


def build_prior(prior_config):
    # If missing from config then return None to trigger MLE (i.e. no prior) instead of
    # a MAP estimate
    if prior_config is None:
        return None

    prior_catalog = {"gamma": GammaPrior}

    if prior_config["type"] not in prior_catalog:
        raise ValueError(f"Unsupported prior 'type'. Got {prior_config['type']!r}.")
    else:
        prior_type = prior_catalog[prior_config["type"]]

    return prior_type(**prior_config["args"])
