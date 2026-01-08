from typing import Any, override

import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from torchvision.transforms.v2 import functional as F


class PixelTimeSeriesNormalize(T.Normalize):  # numpydoc ignore=PR02
    """
    Normalize a 2D (time, channels) or 3D (batch, time, channels) tensor image with mean
    and standard deviation per channel.

    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Parameters
    ----------
    mean : sequence
        Sequence of means for each channel.
    std : sequence
        Sequence of standard deviations for each channel.
    inplace : bool
        (Optional) Whether to make this operation in-place. Default: False

    Notes
    -----
    This transform acts out of place, i.e., it does not mutate the input tensor.

    """

    @override  # https://github.com/astral-sh/ruff/issues/15952#issuecomment-2635255995
    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        assert inpt.ndim in {2, 3}, (
            "Input must be a 2D (time, channels) or 3D (batch, time, channels) tensor, "
            f"got ndim: {inpt.ndim} with shape: {inpt.shape}"
        )
        if inpt.ndim == 2:
            x = rearrange(inpt, "t c -> () c () t")
            x = self._call_kernel(
                F.normalize, x, mean=self.mean, std=self.std, inplace=self.inplace
            )
            x = rearrange(x, "() c () t -> t c")
        elif inpt.ndim == 3:
            x = rearrange(inpt, "b t c -> b c () t")
            x = self._call_kernel(
                F.normalize, x, mean=self.mean, std=self.std, inplace=self.inplace
            )
            x = rearrange(x, "b c () t -> b t c")
        return x
