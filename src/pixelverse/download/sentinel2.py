from collections import defaultdict

import numpy as np
import pandas as pd
from pandas._libs import properties
import xarray as xr
from odc.stac import stac_load
from pystac_client import Client
from pystac import Item
import rioxarray  # noqa: F401
from typing import List
EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"

# SCL band nodata/cloud classes to filter out
# 0: NO_DATA, 3: CLOUD_SHADOWS, 8: CLOUD_MEDIUM_PROBABILITY, 9: CLOUD_HIGH_PROBABILITY, 10: THIN_CIRRUS
CLOUDY_OR_NODATA = (0, 3, 8, 9, 10)


def process_month(items: list[Item], bbox: tuple[float]) -> xr.Dataset:
    result = None
    for item in items:
        print(f"Processing item: {item.id}")
        print(f"Item cloud cover: {item.properties["eo:cloud_cover"]}")
        dset = stac_load(
            [item],
            bbox=bbox,  # type: ignore[invalid-argument-type]
            chunks={"x": 2048, "y": 2048},
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
                "scl",
            ],
            resolution=10,
        )

        cloud_mask = np.isin(dset["scl"].astype("uint8").values, (3, 8, 9, 10))
        print(f"Cloud mask: {cloud_mask.sum()}")
        nodata_mask = np.isin(dset["scl"].astype("uint8").values, (0,))
        print(f"Nodata mask: {nodata_mask.sum()}")
        bad_mask = cloud_mask | nodata_mask
        print(f"Bad mask: {bad_mask.sum()}")
        if not bad_mask.any():
            return dset
        elif result is None:
            result = dset
        else:
            cloud_mask = np.isin(result["scl"].astype("uint8").values, (3, 8, 9, 10))
            nodata_mask = np.isin(result["scl"].astype("uint8").values, (0,))
            bad_mask = cloud_mask | nodata_mask
            result = result.where(~bad_mask.squeeze(), dset.squeeze())
            if not bad_mask.any():
                return result
    return result


def get_s2_times_series(
    bbox: tuple[float],
    year: int,
    stac_host: str = EARTHSEARCH_URL,
    cloudcover_max: int = 20,
    target_cloudcover: float = 0.05,
    months: List[int] = list(range(1, 13)),
) -> dict:
    """Fetch Sentinel-2 imagery for a bounding box for each month of a specified year.

    For each month and MGRS tile, finds images starting with the least cloudy,
    checks the SCL band for actual cloud/nodata coverage, and stops when finding
    an image with no nodata pixels and <target_cloudcover% cloud pixels.

    Data is lazily loaded per month - not downloaded until explicitly accessed.

    Parameters
    ----------
    bbox : tuple[float]
        Bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
    year : int
        Year for which to fetch the imagery.
    stac_host : str, optional
        STAC host URL. Defaults to Earth Search AWS.
    cloudcover_max : int
        Maximum cloud cover percentage for initial filtering. Defaults to 20.
    target_cloudcover : float, optional
        Target maximum cloud cover fraction (0-1). Defaults to 0.05 (5%).
        Stops when finding image with no nodata and <target_cloudcover% cloud pixels.

    Returns
    -------
    dict
        Dictionary mapping month (1-12) to selected STAC items for that month.
        Pass to composite_s2_monthly() which loads data month-by-month.
    """
    client = Client.open(stac_host)
    selected_items_by_month = {}

    # Query each month separately
    dsets = []
    for month in months:
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

        items = list(search.items())
        print(f"Found {len(items)} items for month {month}")
        print(f"scene level percent cloud cover: {[i.properties['eo:cloud_cover'] for i in items]}")
        dset = process_month(items, bbox)
        # if dset is not None:
        #     dset = fill_missing_months_and_format(dset)
        #     # Write RGB to GeoTIFF for this month
        #     rgb = dset[["red", "green", "blue"]].isel(time=0)
        #     rgb_scaled = (255 * rgb / 3000).clip(0, 255).astype("uint8")
        #     rgb_scaled.rio.to_raster(f"sample_s2_rgb_month_{month:02d}.png")
        
        dsets.append(dset)
        return dset

    return xr.concat(dsets, dim="time")


def fill_missing_months_and_format(dset: xr.Dataset) -> xr.Dataset:
    """Fill missing months in the time series by forward filling previous data.

    Parameters
    ----------
    dset : xr.Dataset
        Input xarray Dataset with a time dimension representing months.
        Expected to be the output of `get_s2_times_series`.

    Returns
    -------
    xr.Dataset
        An xarray Dataset wit day of year data variable added and missing months filled.
    """

    # add doy variable to format for model inference
    dset["doy"] = dset.time.dt.dayofyear

    existing_times = pd.DatetimeIndex(dset.time.values)

    missing_months = sorted(set(range(1, 13)) - set(existing_times.month))  # type: ignore[unresolved-attribute]

    new_dates = pd.DatetimeIndex(
        [
            pd.Timestamp(year=pd.Timestamp(dset.time.values[0]).year, month=m, day=15)
            for m in missing_months
        ]
    )

    combined_times = existing_times.append(new_dates).sort_values()
    dset_filled = dset.reindex(time=combined_times, method="ffill")

    return dset_filled


# SAMPLE_AOI_BBOX = (34.30401965358922, 0.483473285913126, 34.3178177763533, 0.497271408677202)
# # Get items (lazily loaded)
# items_by_month = get_s2_times_series(bbox=SAMPLE_AOI_BBOX, year=2021)

# print(items_by_month)
# Create monthly composites with cloud filling
# dset_s2_aoi = composite_s2_monthly(items_by_month, bbox=SAMPLE_AOI_BBOX)
