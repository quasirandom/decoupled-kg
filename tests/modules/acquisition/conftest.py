import pytest
import torch
from botorch import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils import draw_sobol_samples
from gpytorch.mlls import SumMarginalLogLikelihood


@pytest.fixture(autouse=True)
def dtype():
    torch.set_default_dtype(torch.double)
    return torch.double


@pytest.fixture()
def bounds(dtype):
    return torch.tensor([[0, 0], [1, 1]], dtype=dtype)


@pytest.fixture(params=[True, False], ids=["noisy", "noiseless"])
def model(request, bounds):
    return _make_model(bounds, use_noise=request.param)


@pytest.fixture()
def noisy_model(bounds):
    return _make_model(bounds, use_noise=True)


def _make_model(bounds, use_noise):
    n_obj = 2
    n_samples = 10
    seed = 1234  # To make the test more deterministic
    train_x = draw_sobol_samples(bounds, n_samples, q=1, seed=seed)
    train_x = train_x.squeeze(-2)  # Remove the q-batch dimension
    train_y = torch.randn(n_samples, n_obj)
    model = ModelListGP(
        SingleTaskGP(torch.clone(train_x), train_y[..., 0].unsqueeze(-1)),
        SingleTaskGP(torch.clone(train_x), train_y[..., 1].unsqueeze(-1)),
    )
    if not use_noise:
        for gp in model.models:
            gp.likelihood.noise = torch.tensor(1e-4)
            gp.likelihood.noise_covar.raw_noise.requires_grad_(False)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


@pytest.fixture(params=["single", "trio"])
def scalarisation_weights(request):
    return _make_scalarisation_weights(request.param)


@pytest.fixture()
def scalarisation_weights_trio():
    return _make_scalarisation_weights("trio")


def _make_scalarisation_weights(key):
    if key == "single":
        return torch.tensor([[0.6, 0.4]])
    elif key == "trio":
        return torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    else:
        raise ValueError(f"Unrecognised parameter: {key!r}")
