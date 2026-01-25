import numpy as np

from pixelverse.download import (
    interpolate_2d,
)


# %%
def test_interpolate_2d():
    """
    Test interpolating a 2-D array using nearest neighbour into a higher spatial resolution.
    """
    in_arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    out_arr = interpolate_2d(in_arr=in_arr, output_shape=(6, 8), method="nearest")
    assert out_arr.shape == (6, 8)
    np.testing.assert_equal(
        actual=out_arr,
        desired=np.array(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],
                [0, 0, 1, 1, 2, 2, 3, 3],
                [4, 4, 5, 5, 6, 6, 7, 7],
                [4, 4, 5, 5, 6, 6, 7, 7],
                [8, 8, 9, 9, 10, 10, 11, 11],
                [8, 8, 9, 9, 10, 10, 11, 11],
            ],
            dtype=np.uint16,
        ),
    )
