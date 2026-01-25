from typing import Literal

import interpn
import numpy as np


# %%
def interpolate_2d(
    in_arr: np.ndarray,
    output_shape: tuple[int, int],
    method: Literal["nearest"] = "nearest",
) -> np.ndarray:
    """
    Interpolate a 2-D array into another shape.

    Uses interpn.

    Parameters
    ----------
    in_arr : np.ndarray
        Input array in shape (Height, Width).
    output_shape : tuple
        Desired output shape as (Height, Width).
    method : {"nearest"}
        Interpolation method. Default is 'nearest'.

    Returns
    -------
    np.ndarray
        Output array in shape (Height, Width).

    """
    in_height, in_width = in_arr.shape
    out_height, out_width = output_shape

    # input x/y grid coordinates
    xi = np.linspace(start=0, stop=out_width, num=in_width, endpoint=False, dtype="float32")
    yi = np.linspace(start=0, stop=out_height, num=in_height, endpoint=False, dtype="float32")

    # output x/y grid coordinates
    xo, yo = np.meshgrid(
        np.linspace(start=0, stop=out_width, num=out_width, endpoint=False, dtype="float32"),
        np.linspace(start=0, stop=out_height, num=out_height, endpoint=False, dtype="float32"),
        indexing="xy",
    )

    # perform interpolation
    out = interpn.interpn(
        obs=[yo, xo],  # output x/y coordinates
        grids=[yi, xi],  # input x/y coordinates
        vals=in_arr.astype("float32", casting="safe"),  # pixel values
        method=method,
    )

    # cast output to same dtype as input
    return out.astype(in_arr.dtype, casting="same_value")
