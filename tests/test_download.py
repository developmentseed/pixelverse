"""
Tests for downloading Sentinel-1 and Sentinel-2 imagery.
"""

import affine
import async_tiff
import numpy as np
import pandas as pd
import pystac
import pytest
import torch
import xarray as xr
from async_tiff import TIFF

from pixelverse.download import (
    decode_tile_to_tensor,
    fetch_tile,
    interpolate_2d,
    open_tiff,
    stac_to_xarray,
)


# %%
@pytest.fixture(scope="module", name="stac_item")
def fixture_stac_item() -> pystac.Item:
    """
    Sample Sentinel-2 L2A Collection 1 STAC item to use in integration tests.

    Returns
    -------
    pystac.Item

    """
    # import pystac_client
    # import json
    #
    # client = pystac_client.Client.open(url="https://earth-search.aws.element84.com/v1")
    # search = client.search(
    #     collections=["sentinel-2-c1-l2a"],
    #     bbox=(32.99, -0.99, 33.85, 0.0),
    #     datetime="2021-04-14",
    # )
    # items = search.items()
    # stac_item = next(items)
    # print(stac_item.to_dict())
    #
    # with open(file="tests/fixtures/s2_item.json", mode="w+") as fp:
    #     json.dump(obj=stac_item.to_dict(), fp=fp, indent=2)

    stac_item = pystac.Item.from_file(href="tests/fixtures/s2_item.json")
    return stac_item


async def test_open_tiff():
    """
    Test opening a single-band GeoTIFF.
    """
    tiff = await open_tiff(
        tiff_url="https://github.com/developmentseed/titiler/raw/1.1.0/src/titiler/core/tests/fixtures/B09.tif"
    )
    assert isinstance(tiff, TIFF)
    assert len(tiff.ifds) == 3


async def test_fetch_and_decode_tile():
    """
    Test fetching compressed bytes for a single GeoTIFF Tile, and decoding it into a
    torch.Tensor with shape (Height, Width, Channels).
    """
    store = async_tiff.store.S3Store(
        "e84-earth-search-sentinel-data", region="us-west-2", skip_signature=True
    )
    path = "sentinel-2-c1-l2a/36/N/XF/2021/12/S2B_T36NXF_20211212T080506_L2A/B04.tif"
    tiff = await async_tiff.TIFF.open(path=path, store=store)

    tile = await fetch_tile(tiff=tiff, x_index=0, y_index=0, z_index=0)
    tensor = await decode_tile_to_tensor(
        tile=tile, tile_height=1024, tile_width=1024, dtype=torch.uint16
    )
    assert tensor.shape == (1024, 1024, 1)  # HWC
    assert tensor.dtype == torch.uint16


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


async def test_stac_to_xarray(stac_item):
    """
    Test opening a STAC Item produces an xarray.Dataset tile.
    """
    ds: xr.Dataset = await stac_to_xarray(item=stac_item)
    assert tuple(ds.data_vars.keys()) == (
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "swir16",
        "swir22",
    )
    assert ds.sizes == {"x": 1024, "y": 1024, "time": 1}
    assert set(ds.dtypes.values()) == {np.dtypes.UInt16DType()}
    assert ds.xindexes["x"].crs == "EPSG:32736"  # ty:ignore[unresolved-attribute]
    assert ds.xindexes["y"].transform() == affine.Affine(  # ty:ignore[unresolved-attribute]
        a=10.0, b=0.0, c=399960.0, d=0.0, e=-10.0, f=9900040.0
    )
    assert ds.xindexes["time"].index[0] == pd.Timestamp("2021-04-14 08:20:24.721000")  # ty:ignore[unresolved-attribute]

    # Check four corner pixel values (ensure no accidental rotations in resampling)
    np.testing.assert_equal(
        ds.isel(x=0, y=0, time=0).to_array().data,  # top left corner
        [1876, 1820, 1800, 1879, 1782, 1678, 1818, 1600, 1761, 1634],
    )
    np.testing.assert_equal(
        ds.isel(x=-1, y=-1, time=0).to_array().data,  # bottom right corner
        [1273, 1254, 1125, 1144, 1124, 1114, 1111, 1111, 1091, 1070],
    )
