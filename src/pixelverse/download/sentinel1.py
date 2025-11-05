import numpy as np
import pandas as pd
import xarray as xr
from odc.stac import stac_load
from pystac_client import Client

MPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def get_s1_monthly_time_series(
    bbox: tuple[float], year: int, stac_host: str = MPC_URL
) -> xr.Dataset:
    """Fetch Sentinel-1 imagery for a bounding box and return average monthly values.

    Parameters
    ----------
    bbox : tuple[float]
        Bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
    year : int
        Year for which to fetch the imagery.
    stac_host : str, optional
        STAC host URL. Defaults to Microsoft Planetary Computer.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing a time series of Sentinel-1,
        with the lowest cloud cover image per month selected.
    """
    client = Client.open(stac_host)

    search = client.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
    )

    # Load the selected items into an xarray dataset, combining overlapping tiles
    dset = stac_load(
        search.items(),
        bbox=bbox,  # type: ignore[invalid-argument-type]
        chunks={"time": 1, "x": 2048, "y": 2048},
        bands=["vv", "vh"],
        groupby="solar_day",
        resampling="bilinear",
    )

    # get monthly means and represent with 1st day of each month as the datetime
    dset_monthly = dset.groupby("time.month").mean()
    dset_monthly["month"] = pd.date_range(f"{year}-01", periods=12, freq="MS")
    dset_monthly = dset_monthly.rename({"month": "time"})

    return dset_monthly


def linear_to_decibel(dataarray: xr.DataArray) -> xr.DataArray:
    """Transform VV or VH values from linear to decibel scale.

    Parameters
    ----------
    dataarray : xr.DataArray
        Input DataArray with VV or VH values in linear scale.

    Returns
    -------
    xr.DataArray
        DataArray with values converted to decibel scale using 10 * log_10(x).
    """
    # Mask out areas with 0 so that np.log10 is not undefined
    da_linear = dataarray.where(cond=dataarray != 0)
    da_decibel = 10 * np.log10(da_linear)
    return da_decibel


def process_s1_dataset(dset: xr.Dataset) -> xr.Dataset:
    """
    Process the Sentinel-1 xarray.Dataset by converting VV and VH bands
    from linear to decibel scale.

    Parameters
    ----------
    dset : xr.Dataset
        Input xarray Dataset containing 'vv' and 'vh' bands.

    Returns
    -------
    xr.Dataset
        Processed xarray Dataset with 'vv_processed' and 'vh_processed' bands.
    """
    dset["vv_processed"] = linear_to_decibel(dset["vv"])
    dset["vh_processed"] = linear_to_decibel(dset["vh"])
    return dset
