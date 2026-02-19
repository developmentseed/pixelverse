---
icon: lucide/rocket
---

# Pixelverse

Generate and store geospatial foundation model embeddings using cloud-native tooling!

## Example usage

### Import functions

```python
import rioxarray
import torch
import xarray as xr

from pixelverse.download.sentinel2 import get_s2_time_series
from pixelverse.generate_embeddings import generate_embeddings
```

### Define Area of Interest (AOI)

Use [`get_s2_time_series`][pixelverse.download.sentinel2.get_s2_time_series] to fetch
Sentinel-2 imagery and get xarray.Dataset datacube.

```python
SAMPLE_AOI_BBOX = (34.3040, 0.4835, 34.3178, 0.4973)
s2_dset: xr.Dataset = get_s2_time_series(bbox=SAMPLE_AOI_BBOX, year=2021)
s2_dset
```

### Generate embeddings

Use [`generate_embeddings`][pixelverse.generate_embeddings.generate_embeddings] to
create embeddings, e.g. with the Tessera Model:

```python
embeddings: xr.Dataset = generate_embeddings(
    s2_dset=s2_dset,
    model_name="tessera_s2_encoder",
)
embeddings
```

### (Optional) Quantize embeddings

Use [`quantize_embeddings`][pixelverse.generate_embeddings.quantize_embeddings] to
convert from float32 to int8 and save disk space.

```python
embeddings_quantized: xr.Dataset = quantize_embeddings(embeddings=embeddings)
```

### Write embeddings

Save output to a Cloud-Optimized GeoTIFF (COG) using
[`.rio.to_raster`][rioxarray.raster_dataset.RasterDataset.to_raster]

```python
embeddings["embedding"].rio.to_raster(
    "./sample_s2_embeddings_cog.tif",
    driver="COG",
    compress="ZSTD",
)
```

Or save to Zarr using [`.to_zarr`][xarray.Dataset.to_zarr].

```python
embeddings["embedding"].to_zarr("./sample_s2_embeddings.zarr", mode="w")
```
