import torch
import torchvision.transforms as T
from einops import rearrange


class PixelTimeSeriesNormalize(T.Normalize):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim in [2, 3], (
            "Input must be a 2D (time, channels) or 3D (batch, time, channels) tensor"
        )
        if tensor.ndim == 2:
            x = rearrange(tensor, "t c -> () c () t")
            x = super().forward(x)
            x = rearrange(x, "() c () t -> t c")
        elif tensor.ndim == 3:
            x = rearrange(tensor, "b t c -> b c () t")
            x = super().forward(x)
            x = rearrange(x, "b c () t -> b t c")
        return x
