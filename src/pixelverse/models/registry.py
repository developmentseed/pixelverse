from typing import Any

import torch

from pixelverse.models.tessera import TESSERA_WEIGHTS, tessera

_models = {"tessera": tessera}

_model_weights = {
    "tessera": TESSERA_WEIGHTS.TESSERA,
}


def list_models() -> list[str]:
    return list(_models.keys())


def create_model(
    name: str, pretrained: bool = True, *args: Any, **kwargs: Any
) -> tuple[torch.nn.Module, torch.nn.Sequential]:
    if pretrained:
        model = _models[name](_model_weights[name], *args, **kwargs)
        transforms = _model_weights[name].transforms
    else:
        model = _models[name](*args, **kwargs)
        transforms = torch.nn.Sequential(torch.nn.Identity())
    return model, transforms
