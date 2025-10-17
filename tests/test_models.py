import torch

import pixelverse as pv


def test_models():
    for model_name in pv.list_models():
        model, transforms = pv.create_model(model_name)
        weights = pv.get_weights(model_name)
        input_shape = [2 if shape is None else shape for shape in weights.meta["input_shape"][0]]
        output_shape = [2 if shape is None else shape for shape in weights.meta["output_shape"][0]]
        assert isinstance(model, torch.nn.Module)
        assert isinstance(transforms, torch.nn.Sequential)
        assert len(transforms)
        x = torch.randn(size=input_shape, dtype=torch.float32)
        y = model(transforms(x))
        assert y.shape == tuple(output_shape)
        assert y.dtype == torch.float32
