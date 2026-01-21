"""
Loading STAC Items to Xarray.

Using async-tiff for I/O, and rasterix for CRS/index handling.
"""

import asyncio
import itertools
import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

import async_tiff
import numpy as np
import pystac
import torch
import xarray as xr
from affine import Affine
from rasterix import RasterIndex


# %%
async def open_tiff(tiff_url: str, **kwargs) -> async_tiff.TIFF:
    """
    Open a Cloud-optimized GeoTIFF using async-tiff.

    Parameters
    ----------
    tiff_url : str
        A http or s3 URL to the GeoTIFF file.
    **kwargs
        Extra keyword arguments to pass to `async_tiff.store.from_url`.

    Returns
    -------
    async_tiff.TIFF
        Handle to the GeoTIFF file.

    """
    # Open GeoTIFF file
    url = urlparse(url=tiff_url)
    store = async_tiff.store.from_url(url=f"{url.scheme}://{url.hostname}", **kwargs)
    tiff = await async_tiff.TIFF.open(path=url.path, store=store)
    return tiff


async def fetch_tile(tiff: async_tiff.TIFF, x_index: int, y_index: int, z_index: int = 0):
    """
    Fetch a single compressed GeoTIFF tile.

    Parameters
    ----------
    tiff : async_tiff.TIFF
        Handle to the GeoTIFF file.
    x_index : int
        The column index within the IFD to read from.
    y_index : int
        The row index within the IFD to read from.
    z_index : int
        The IFD index to read from. Default is 0.

    Returns
    -------
    async_tiff.Tile
        Tile containing compressed bytes.

    """
    return await tiff.fetch_tile(x=x_index, y=y_index, z=z_index)


async def decode_tile_to_tensor(
    tile,  #: async_tiff.Tile,
    tile_height: int,
    tile_width: int,
    dtype: torch.dtype = torch.uint16,
) -> torch.Tensor:
    """
    Decode a TIFF tile for a single band into a 3-D tensor.

    Parameters
    ----------
    tile : async_tiff.Tile
        Tile containing compressed bytes.
    tile_height : int
        Height of the tile in pixels.
    tile_width : int
        Width of the tile in pixels.
    dtype : torch.dtype
        Data type of the tile. Default is torch.uint16.

    Returns
    -------
    torch.Tensor
        Tile of shape (Height, Width, Channel).

    """
    decoded_bytes = await tile.decode_async()
    tensor: torch.Tensor = torch.frombuffer(buffer=decoded_bytes, dtype=dtype)
    tensor = tensor.reshape(tile_height, tile_width, 1)
    return tensor


async def stac_to_xarray(
    item: pystac.Item,
    # bbox, chunks
    bands: Sequence[str] = (
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
    ),
    # resolution, dtype, nodata
) -> xr.Dataset:
    """
    Read a STAC Item into an xarray.Dataset.

    Partial re-implementation of odc.stac.load using async-tiff and rasterix.

    Parameters
    ----------
    item : pystac.Item
        STAC Item containing one/multiple STAC Assets to read from.
    bands : list[str]
        List of satellite band names to read from the STAC Item's assets. Default is
        the Sentinel-2 10m and 20m spatial resolution bands.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the pixel data per band as data_variables, and
        each band stacked in the shape of (time, height, width).

    """
    # Loop through STAC Assets (bands)
    assets: list[pystac.Asset] = [item.assets[band] for band in bands]
    # Get some metadata from first STAC Asset
    asset: pystac.Asset = assets[0]
    metadata: dict[str, Any] = asset.extra_fields

    async with asyncio.TaskGroup() as task_group:
        tasks_open: list[asyncio.Task] = [
            task_group.create_task(
                coro=open_tiff(tiff_url=asset.href, skip_signature=True),
                name=asset.extra_fields["eo:bands"][0]["name"],
            )
            for asset in assets
        ]

    # Get list of all TIFFs, i.e. different Sentinel-2 bands
    tiffs: list[async_tiff.TIFF] = await asyncio.gather(*tasks_open)

    # Retrieve some metadata about tiles in the first band TIFF
    ifd: async_tiff.ImageFileDirectory = tiffs[0].ifds[0]  # full-resolution IFD
    tile_width: int = ifd.tile_width
    tile_height: int = ifd.tile_height
    x_count: int = math.ceil(ifd.image_width / tile_width)
    y_count: int = math.ceil(ifd.image_height / tile_height)
    # Get cartesian product of x and y tile ids
    xy_ranges = itertools.product(range(x_count), range(y_count))

    # Decode tiles one by one, gathering all bands per tile at once
    for x_index, y_index in xy_ranges:  # Loop through GeoTIFF tiles
        # Fetch all bands belonging to the same tile index
        async with asyncio.TaskGroup() as task_group:
            tasks_fetch: list[asyncio.Task] = [
                task_group.create_task(
                    coro=fetch_tile(tiff=tiff, x_index=x_index, y_index=y_index, z_index=0)
                )
                for tiff in tiffs
            ]
        tiles: list[async_tiff.Tile] = await asyncio.gather(*tasks_fetch)

        # Decode all tiles across different bands
        dtype = getattr(torch, metadata["raster:bands"][0]["data_type"])
        async with asyncio.TaskGroup() as task_group:
            tasks_decode: list[asyncio.Task] = []
            for idx, tile in enumerate(tiles):
                ifd = tiffs[idx].ifds[0]
                task = task_group.create_task(
                    coro=decode_tile_to_tensor(
                        tile=tile,
                        tile_height=ifd.tile_height,
                        tile_width=ifd.tile_width,
                        dtype=dtype,
                    )
                )
                tasks_decode.append(task)

        tensors: list[torch.Tensor] = await asyncio.gather(*tasks_decode)

        # Prepare xarray data variables, with same 10m spatial resolution per band
        data_vars: dict[str, tuple[tuple, torch.Tensor]] = {}
        dims = ("x", "y", "time")
        for band, tensor in zip(bands, tensors, strict=True):
            tensor_ = tensor.permute(1, 0, 2)  # HWC -> WHC or XYT
            if tensor_.size() != (tile_width, tile_height, 1):
                continue  # TODO: resample 20m bands to 10m using nearest neighbour
            data_vars[band] = (dims, tensor_)

        # Raster coordinates from first band (assume all the same)
        width, height = metadata["proj:shape"]
        raster_index = RasterIndex.from_stac_proj_metadata(
            metadata=metadata, width=width, height=height
        )
        # Get local Affine transformation for the tile
        tile_affine_transform = raster_index.transform() * Affine.translation(
            xoff=x_index * ifd.tile_height, yoff=y_index * ifd.tile_width
        )
        tile_index = RasterIndex.from_transform(
            affine=tile_affine_transform,
            width=tile_width,
            height=tile_height,
            crs=item.properties["proj:code"],
        )

        # Build xarray.Dataset for multiple bands captured at one timestep
        ds_tile = xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates.from_xindex(index=tile_index).assign(
                time=[np.datetime64(item.properties["datetime"], "ns")]
            ),
        ).transpose("time", "y", "x")
        # ds_tile.blue.isel(time=0).plot.imshow()
        break

    return ds_tile
