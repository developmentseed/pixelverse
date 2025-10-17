from typing import Any

import torch
from torchvision.models._api import Weights

from pixelverse.models.tessera import (
    TESSERA_WEIGHTS,
    tessera,
    tessera_s1_encoder,
    tessera_s2_encoder,
)

_models = {
    "tessera": tessera,
    "tessera_s2_encoder": tessera_s2_encoder,
    "tessera_s1_encoder": tessera_s1_encoder,
}

_model_weights = {
    "tessera": TESSERA_WEIGHTS.TESSERA,
    "tessera_s2_encoder": TESSERA_WEIGHTS.TESSERA_S2_ENCODER,
    "tessera_s1_encoder": TESSERA_WEIGHTS.TESSERA_S1_ENCODER,
}


def list_models() -> list[str]:
    return list(_models.keys())


def get_weights(name: str) -> Weights:
    assert name in _model_weights, f"Model {name} not found"
    return _model_weights[name]  # type: ignore


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
