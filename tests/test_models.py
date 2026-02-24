import torch
from torchgeo.models.presto import BANDS_GROUPS_IDX

import pixelverse as pv

SMOKE_INPUTS = {
    "tessera": (2, 2, 14),
    "tessera_s2_encoder": (2, 2, 11),
    "tessera_s1_encoder": (2, 2, 3),
}


def test_models():
    presto_channels = sum(len(group) for group in BANDS_GROUPS_IDX.values())

    for model_name in pv.list_models():
        model, transforms = pv.create_model(model_name)
        _ = pv.get_weights(model_name)

        assert isinstance(model, torch.nn.Module)
        assert isinstance(transforms, torch.nn.Module)

        if model_name == "presto":
            x = torch.randn(size=(2, 3, presto_channels), dtype=torch.float32)
            dynamic_world = torch.zeros((2, 3), dtype=torch.long)
            latlons = torch.tensor([[0.0, 0.0], [10.0, -20.0]], dtype=torch.float32)
            embeddings, kept_indices, removed_indices = model(transforms(x), dynamic_world, latlons)
            assert embeddings.shape[0] == x.shape[0]
            assert embeddings.dtype == torch.float32
            assert kept_indices.shape[0] == x.shape[0]
            assert removed_indices.shape[0] == x.shape[0]
            continue

        x = torch.randn(size=SMOKE_INPUTS[model_name], dtype=torch.float32)
        y = model(transforms(x))
        assert y.shape[0] == x.shape[0]
        assert y.dtype == torch.float32
