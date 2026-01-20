"""
Tests for downloading Sentinel-1 and Sentinel-2 imagery.
"""

import async_tiff
import torch
from async_tiff import TIFF

import pixelverse as pv
from pixelverse.download import decode_tile_to_tensor


# %%
async def test_open_tiff():
    """
    Test opening a single-band GeoTIFF.
    """
    tiff = await pv.download.open_tiff(
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

    tile = await pv.download.fetch_tile(tiff=tiff, x_index=0, y_index=0, z_index=0)
    tensor = await decode_tile_to_tensor(
        tile=tile, tile_height=1024, tile_width=1024, dtype=torch.uint16
    )
    assert tensor.shape == (1024, 1024, 1)  # HWC
    assert tensor.dtype == torch.uint16
