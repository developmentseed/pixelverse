import torch

import pixelverse as pv


def test_models():
    for model_name in pv.list_models():
        weights = pv.get_weights(model_name)
        if weights.meta.get("skip_pretrained_test", False):
            continue
        model, transforms = pv.create_model(model_name)
        input_shape = [2 if shape is None else shape for shape in weights.meta["input_shape"][0]]
        output_shape = [2 if shape is None else shape for shape in weights.meta["output_shape"][0]]
        assert isinstance(model, torch.nn.Module)
        assert isinstance(transforms, torch.nn.Sequential)
        assert len(transforms)
        x = torch.randn(size=input_shape, dtype=torch.float32)
        timestamps_shape = weights.meta.get("timestamps_shape")
        if timestamps_shape is not None:
            ts_shape = [2 if shape is None else shape for shape in timestamps_shape[0]]
            timestamps = torch.zeros(size=ts_shape, dtype=torch.long)
            y = model(transforms(x), timestamps=timestamps)
        else:
            y = model(transforms(x))
        assert y.shape == tuple(output_shape)
        assert y.dtype == torch.float32
