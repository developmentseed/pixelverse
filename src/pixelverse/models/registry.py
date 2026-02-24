from collections.abc import Callable
from typing import Any, cast

import torch
from torchgeo.models import Presto_Weights, Tessera_Weights, presto, tessera
from torchvision.models._api import WeightsEnum

_models: dict[str, Callable[..., torch.nn.Module]] = {
    "presto": presto,
    "tessera": tessera,
    "tessera_s2_encoder": tessera,
    "tessera_s1_encoder": tessera,
}

_model_weights: dict[str, WeightsEnum] = {
    "presto": Presto_Weights.PRESTO,
    "tessera": Tessera_Weights.TESSERA,
    "tessera_s2_encoder": Tessera_Weights.TESSERA_SENTINEL2_ENCODER,
    "tessera_s1_encoder": Tessera_Weights.TESSERA_SENTINEL1_ENCODER,
}


def list_models() -> list[str]:
    return list(_models.keys())


def get_weights(name: str) -> WeightsEnum:
    if name not in _model_weights:
        raise KeyError(f"Model {name} not found")
    return _model_weights[name]


def create_model(name: str, *args: Any, **kwargs: Any) -> tuple[torch.nn.Module, Callable]:
    """
    Load neural network model and transforms.

    Parameters
    ----------
    name : str
        Name of model to instantiate.
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
    weights = get_weights(name=name)
    model = _models[name](weights, *args, **kwargs)
    if name == "presto":
        model = cast(torch.nn.Module, model.encoder)
    transforms: Callable = weights.transforms
    return model, transforms
