import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr

from pixelverse.models.registry import create_model

S2_BANDS = [
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
]


def generate_embeddings(s2_dset: xr.Dataset, model_name: str = "tessera_s2_encoder") -> xr.Dataset:
    """
    Generate embeddings for a given Sentinel-2 dataset using the specified model.

    Note: MPV function designed to work with small areas.

    Parameters
    ----------
    s2_dset : xr.Dataset
        Sentinel-2 dataset containing spectral bands and temporal information.
    model_name : str, optional
        Name of the model to use for generating embeddings. Default is "tessera_s2_encoder".

    Returns
    -------
    xr.Dataset
        Dataset containing generated embeddings with spatial coordinates and CRS information.
    """
    model = create_model(model_name, pretrained=True)
    model[0].eval()

    # add day of year variable if not present
    if "doy" not in s2_dset:
        s2_dset["doy"] = s2_dset.time.dt.dayofyear

    # Stack spatial dimensions into pixels
    s2_stacked = s2_dset[S2_BANDS].to_array(dim="band").stack(pixel=["y", "x"])

    # Prepare tensors
    # s2_stacked.values shape: (band, time, pixel) -> transpose to (pixel, time, band)
    s2_tensor = torch.from_numpy(s2_stacked.values.transpose(2, 1, 0))  # (pixel, time, band)

    # DOY tensor: expand to match number of pixels
    doy_tensor = (
        torch.from_numpy(s2_dset.doy.values).unsqueeze(0).expand(s2_tensor.shape[0], -1)
    )  # (pixel, time)

    # Concatenate bands and DOY along last dimension
    x = torch.cat([s2_tensor, doy_tensor.unsqueeze(-1)], dim=-1)  # (pixel, time, 11)

    # Run through model
    with torch.no_grad():
        embeddings = model[0](x.float())

    # Convert back to xarray with spatial structure
    dset_embeddings = (
        xr.DataArray(
            embeddings.numpy(),
            dims=["pixel", "feature"],
            coords={"pixel": s2_stacked.pixel},
        )
        .unstack("pixel")
        .to_dataset(name="embedding")
    )

    # Add back spatial info
    dset_embeddings.rio.write_crs(s2_dset.rio.crs, inplace=True)

    return dset_embeddings


def quantize_embeddings(embeddings: xr.DataArray) -> xr.Dataset:
    """
    Quantize embeddings from float32 to int8 to save space (4x compression).

    Normalizes each feature independently to the int8 range [-128, 127].
    The quantized embeddings can be used directly for similarity comparisons
    and other operations without dequantization.

    Parameters
    ----------
    embeddings : xr.DataArray
        Float32 embeddings with shape (feature, y, x)

    Returns
    -------
    xr.Dataset
        Dataset containing quantized int8 embeddings with 'embedding' variable
        and spatial coordinates preserved
    """

    min_vals = embeddings.min(dim=["y", "x"]).values
    max_vals = embeddings.max(dim=["y", "x"]).values

    scale = (max_vals - min_vals) / 255.0
    scale = np.where(scale == 0, 1, scale)

    quantized_values = np.round(
        (embeddings.values - min_vals[:, None, None]) / scale[:, None, None]
    )
    quantized_values = np.clip(quantized_values, 0, 255).astype(np.uint8)

    # Create Dataset directly - most efficient
    return xr.Dataset(
        data_vars={"embedding": (embeddings.dims, quantized_values)},
        coords=embeddings.coords,
        attrs={"quantized": True},
    )
