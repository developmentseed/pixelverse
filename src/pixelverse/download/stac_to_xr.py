"""
Loading STAC Items to Xarray.

Using async-tiff for I/O, and rasterix for CRS/index handling.
"""

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


async def stac_to_xarray(
    items: list[pystac.Item],
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
    Read STAC Items into xarray.Dataset.

    Re-implementation of odc.stac.load using async-tiff and rasterix.

    Parameters
    ----------
    items : list[pystac.Item]
        Iterable of STAC Items to read from.
    bands : list[str]
        List of satellite band names to read from the STAC Item's assets. Default is
        the Sentinel-2 10m and 20m spatial resolution bands.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the pixel data per band as data_variables, and
        each band stacked in the shape of (time, height, width).

    """
    for item in items:  # Loop through STAC Items
        for band in bands:  # Loop through STAC Assets
            asset: pystac.Asset = item.assets[band]

            # Open GeoTIFF file
            url = urlparse(url=asset.href)
            store = async_tiff.store.from_url(
                url=f"{url.scheme}://{url.hostname}", skip_signature=True
            )
            tiff = await async_tiff.TIFF.open(path=url.path, store=store)

            # Get number of tiles in primary IFD
            ifd = tiff.ifds[0]  # full-resolution IFD
            tile_width: int = ifd.tile_width  # ty: ignore[invalid-assignment]
            tile_height: int = ifd.tile_height  # ty: ignore[invalid-assignment]
            x_count: int = math.ceil(ifd.image_width / tile_width)
            y_count: int = math.ceil(ifd.image_height / tile_height)
            # Get cartesian product of x and y tile ids
            xy_ranges = itertools.product(range(x_count), range(y_count))

            # Decode tiles one by one
            for x_index, y_index in xy_ranges:  # Loop through GeoTIFF tiles
                tile = await tiff.fetch_tile(x=x_index, y=y_index, z=0)

                decoded_bytes = await tile.decode_async()
                # dtype = metadata["raster:bands"][0]["data_type"]
                tensor: torch.Tensor = torch.frombuffer(buffer=decoded_bytes, dtype=torch.uint16)
                tensor = tensor.reshape(tile_height, tile_width, 1).permute(
                    1, 0, 2
                )  # HWC -> WHC or XYT

                # raster coordinates
                metadata: dict[str, Any] = asset.extra_fields
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

                # Build xarray.DataArray for single band
                da_tile = xr.DataArray(
                    data=tensor,
                    coords=xr.Coordinates.from_xindex(index=tile_index).assign(
                        time=[np.datetime64(item.properties["datetime"], "ns")]
                    ),
                    dims=("x", "y", "time"),
                    name=band,
                ).transpose("time", "y", "x")
                # da_tile.isel(time=0).plot.imshow()

                break

    return da_tile.to_dataset()
