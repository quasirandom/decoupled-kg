from numbers import Real
from typing import List, Union

import torch
from botorch.models import ModelListGP


def set_hyperparameters(
    model: ModelListGP,
    length_scales: List[torch.Tensor],
    output_scales: List[Real],
    means: List[Real],
    noise_vars: Union[List[Real], torch.Tensor],
    tkwargs=None,
):
    """Set hyper-parameters on the standard model

    The model provided should be created by `build_mll_and_model`.
    """
    if len(length_scales) != model.num_outputs:
        raise ValueError(
            f"Expected length_scales to have one element per model. "
            f"Got {len(length_scales)=}, {model.num_outputs=}."
        )
    if len(output_scales) != model.num_outputs:
        raise ValueError(
            f"Expected output_scales to have one element per model. "
            f"Got {len(output_scales)=}, {model.num_outputs=}."
        )
    if len(means) != model.num_outputs:
        raise ValueError(
            f"Expected means to have one element per model. "
            f"Got {len(means)=}, {model.num_outputs=}."
        )
    if len(noise_vars) != model.num_outputs:
        raise ValueError(
            f"Expected noise_vars to have one element per model. "
            f"Got {len(noise_vars)=}, {model.num_outputs=}."
        )

    for i, gp in enumerate(model.models):
        gp.mean_module.initialize(constant=torch.as_tensor(means[i], **(tkwargs or {})))
        gp.covar_module.outputscale = output_scales[i]
        gp.covar_module.base_kernel.lengthscale = length_scales[i]
        gp.likelihood.noise = noise_vars[i]
