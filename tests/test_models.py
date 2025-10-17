import torch

import pixelverse as pv


def test_models():
    assert pv.list_models() == ["tessera"]
    for model_name in pv.list_models():
        model, transforms = pv.create_model(model_name)
        weights = pv.get_weights(model_name)
        assert isinstance(model, torch.nn.Module)
        assert isinstance(transforms, torch.nn.Sequential)
        assert len(transforms)
        b, t, c = 2, 10, weights.meta["num_channels"]
        x = torch.randn(b, t, c)
        y = transforms(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


@torch.inference_mode()
def test_tessera():
    device = torch.device("cpu")
    model = pv.models.Tessera().to(device).eval()
    b, t = 1, 10
    s2 = torch.randn(b, t, 10)
    s2_doy = torch.randint(1, 365, (b, t, 1))
    s1 = torch.randn(b, t, 2)
    s1_doy = torch.randint(1, 365, (b, t, 1))
    x = torch.cat([s2, s2_doy, s1, s1_doy], dim=-1).to(device)
    y = model(x)
    assert y.shape == (b, 128)
    assert y.dtype == torch.float32
    assert y.device == torch.device("cpu")
