from collections import defaultdict

import pandas as pd
import xarray as xr
from odc.stac import stac_load
from pystac_client import Client

EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"


def get_s2_times_series(
    bbox: tuple[float],
    year: int,
    stac_host: str = EARTHSEARCH_URL,
    cloudcover_max: int = 50,
) -> xr.Dataset:
    """Fetch Sentinel-2 imagery for a bounding box for each month of a specified year.

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

    if not selected_items:
        raise ValueError(f"No Sentinel-2 images found for bbox {bbox} in year {year}")

    # Load the selected items into an xarray dataset
    dset = stac_load(
        selected_items,
        bbox=bbox,  # type: ignore[invalid-argument-type]
        chunks={"time": 1, "x": 2048, "y": 2048},
        bands=[
            "blue",
            "green",
            "red",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "swir16",
            "swir22",
            "scl",
        ],
        resolution=10,  # 10m resolution,
    )

    # time dim is first day of each month that appeared in the dataset
    dset_monthly = dset.groupby("time.month").mean()
    dset_monthly["month"] = dset.time.groupby("time.month").min().values
    dset_monthly = dset_monthly.rename({"month": "time"})

    return dset_monthly
