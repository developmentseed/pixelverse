from pystac_client import Client
from sympy import li
import xarray as xr
from odc.stac import stac_load
import pandas as pd

EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"


def get_s2_times_series(
    bbox: list[int], year: int, stac_host: str = EARTHSEARCH_URL
) -> xr.Dataset:
    """Fetches Sentinel-2 imagery for a given bounding box for each month of a specified year.

    Args:
        bbox (list[int]): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        year (int): Year for which to fetch the imagery.
        stac_host (str): STAC host URL.

    Returns:
        xr.Dataset: An xarray Dataset containing a time series of Sentinel-2,
                   with the lowest cloud cover image per month selected.
    """
    client = Client.open(stac_host)

    selected_items = []

    # Query each month separately
    for month in range(1, 13):
        # Calculate start and end dates for this month
        start_date = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            end_date = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(
                seconds=1
            )
        else:
            end_date = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(
                seconds=1
            )

        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            sortby=["+properties.eo:cloud_cover"],
            limit=1,
        )

        selected_items.append(
            list(search.items())[0]
        )  # TO-DO fix to account for large areas

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
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "swir16",
            "swir22",
            "scl",
        ],
        resolution=10,  # 10m resolution
    )

    return dset
