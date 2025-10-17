import torch

from pixelverse.models.transforms import PixelTimeSeriesNormalize


def test_pixel_time_series_normalize():
    transform = PixelTimeSeriesNormalize(mean=[0.5, 0.3], std=[0.2, 0.4])

    # test batch
    b, t, c = 2, 10, 2
    x = torch.randn(b, t, c)
    y = transform(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

    # test sample
    x = torch.randn(t, c)
    y = transform(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
