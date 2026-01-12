from collections.abc import Callable
from typing import Any

import torch
from torchvision.models._api import WeightsEnum

from pixelverse.models.tessera import (
    TESSERA_WEIGHTS,
    tessera,
    tessera_s1_encoder,
    tessera_s2_encoder,
)

_models: dict[str, Callable[..., torch.nn.Module]] = {
    "tessera": tessera,
    "tessera_s2_encoder": tessera_s2_encoder,
    "tessera_s1_encoder": tessera_s1_encoder,
}

_model_weights: dict[str, WeightsEnum] = {
    "tessera": TESSERA_WEIGHTS.TESSERA,
    "tessera_s2_encoder": TESSERA_WEIGHTS.TESSERA_S2_ENCODER,
    "tessera_s1_encoder": TESSERA_WEIGHTS.TESSERA_S1_ENCODER,
}


def list_models() -> list[str]:
    return list(_models.keys())


def get_weights(name: str) -> WeightsEnum:
    if name not in _model_weights:
        raise KeyError(f"Model {name} not found")
    return _model_weights[name]


def create_model(
    name: str, pretrained: bool = True, *args: Any, **kwargs: Any
) -> tuple[torch.nn.Module, Callable]:
    """
    Load neural network model and transforms.

    Parameters
    ----------
    name : str
        Name of model to instantiate.
    pretrained : bool
        Load pretrained model weights. Default: True
    *args
        Additional arguments passed to model creation function.
    **kwargs
        Extra keyword arguments to model function: refer to each model's documentation
        for a list of all possible arguments.

    Returns
    -------
    model : torch.nn.Module
        Loaded neural network model.
    transforms : Callable
        Transforms to apply on data before passing passing into model.

    """
    if pretrained:
        weights = get_weights(name=name)
        model = _models[name](weights, *args, **kwargs)
        transforms: Callable = weights.transforms
    else:
        model = _models[name](*args, **kwargs)
        transforms = torch.nn.Sequential(torch.nn.Identity())
    return model, transforms
