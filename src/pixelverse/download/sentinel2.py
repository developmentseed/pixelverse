"""
Retrieve and process Sentinel-2 Collection 1 Level 2A multispectral data.
"""

import asyncio
from collections import defaultdict
from types import AsyncGeneratorType

import pandas as pd
import pystac
import xarray as xr
from odc.stac import stac_load
from pystac_client import Client

from pixelverse.download import stac_to_tiles


def get_s2_time_series(
    bbox: tuple[int | float, int | float, int | float, int | float],
    year: int,
    stac_host: str = "https://earth-search.aws.element84.com/v1",
    cloudcover_max: int = 20,
) -> xr.Dataset:
    """
    Fetch Sentinel-2 imagery for a bounding box for each month of a specified year.

    Parameters
    ----------
    bbox : tuple[float]
        Bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
    year : int
        Year for which to fetch the imagery.
    stac_host : str, optional
        STAC host URL. Defaults to Earth Search AWS.
    cloudcover_max : int
        Maximum cloud cover percentage for filtering images. Defaults to 50.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing a time series of Sentinel-2,
        with the lowest cloud cover image per month selected.
    """
    client = Client.open(stac_host)

    selected_items = []

    # Query each month separately
    for month in range(1, 13):
        # Calculate start and end dates for this month
        start_date = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            end_date = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(seconds=1)
        else:
            end_date = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(seconds=1)

        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            query={"eo:cloud_cover": {"lt": cloudcover_max}},
            sortby=["+properties.eo:cloud_cover"],
        )

        # select lowest cloud cover for each unique MGRS tile
        items = list(search.items())
        if items:
            tiles = defaultdict(list)
            for item in items:
                mgrs_tile = item.id.split("_")[1]
                tiles[mgrs_tile].append(item)

        selected_items.extend(
            min(tile_items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
            for tile_items in tiles.values()
        )
        if not items:
            print(
                f"No images found for {year}-{month} with cloud cover < {cloudcover_max}% "
                "using previous months' data"
            )

    if not selected_items:
        raise ValueError(f"No Sentinel-2 images found for bbox {bbox} in year {year}")

    # Load the selected items into an xarray dataset
    dset = stac_load(
        selected_items,
        bbox=bbox,
        chunks={"time": 1, "x": 2048, "y": 2048},
        bands=[
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
        ],
        resolution=10,  # 10m resolution,
        dtype="uint16",
        nodata=0,
    )

    # time dim is first day of each month that appeared in the dataset
    dset_monthly = dset.groupby("time.month").mean(dtype="uint16")
    dset_monthly["month"] = dset.time.groupby("time.month").min().values
    dset_monthly = dset_monthly.rename({"month": "time"})

    return dset_monthly


async def get_s2_time_series_tiles(
    items: list[pystac.Item],
) -> AsyncGeneratorType[xr.Dataset]:
    """
    Yield a stack of Sentinel-2 multi-band tiles for each month of a specified year.

    Calls [`stac_to_tiles`][pixelverse.download.stac_to_tiles] under the hood to do
    multiple tile fetches asynchronously.

    Parameters
    ----------
    items : list[pystac.Item]
        Multiple STAC Item(s) containing one/multiple STAC Assets to read from.

    Yields
    ------
    xr.Dataset
        An xarray.Dataset tile containing a time-series of pixel data per band as
        data_variables, with multiple items (at different timesteps) and bands stacked
        into a tensor of shape (time: N, height, width).

    """
    tile_generators: list[AsyncGeneratorType[xr.Dataset]] = [
        stac_to_tiles(item=item) for item in items
    ]

    while True:
        async with asyncio.TaskGroup() as task_group:
            tasks_tiles: list[asyncio.Task] = [
                task_group.create_task(
                    coro=anext(tile_gen),
                )
                for tile_gen in tile_generators
            ]

        # Get list of all TIFF tiles, i.e. multi-band Sentinel-2 images from different times
        tiles: list[xr.Dataset] = await asyncio.gather(*tasks_tiles)

        ds: xr.Dataset = xr.concat(objs=tiles, dim="time")
        # time dim is first day of each month that appeared in the dataset
        ds_monthly = ds.groupby("time.month").mean(dtype="uint16")
        ds_monthly["month"] = ds.time.groupby("time.month").min().values
        ds_monthly = ds_monthly.rename({"month": "time"})

        yield ds_monthly


def fill_missing_months_and_format(dset: xr.Dataset) -> xr.Dataset:
    """
    Fill missing months in the time series by forward filling previous data.

    Parameters
    ----------
    dset : xr.Dataset
        Input xarray Dataset with a time dimension representing months.
        Expected to be the output of `get_s2_time_series`.

    Returns
    -------
    xr.Dataset
        An xarray Dataset wit day of year data variable added and missing months filled.

    """
    # add doy variable to format for model inference
    dset["doy"] = dset.time.dt.dayofyear

    existing_times = pd.DatetimeIndex(dset.time.values)

    missing_months = sorted(set(range(1, 13)) - set(existing_times.month))

    new_dates = pd.DatetimeIndex(
        [
            pd.Timestamp(year=pd.Timestamp(dset.time.values[0]).year, month=m, day=15)
            for m in missing_months
        ]
    )

    combined_times = existing_times.append(new_dates).sort_values()
    dset_filled = dset.reindex(time=combined_times, method="ffill")

    return dset_filled
