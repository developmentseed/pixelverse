import numpy as np
import pytest
import xarray as xr

from pixelverse.generate_embeddings import generate_embeddings, quantize_embeddings


@pytest.fixture
def sample_s2_dataset():
    """Create a small sample Sentinel-2 dataset for testing."""
    np.random.seed(42)

    y_size, x_size = 10, 10
    n_times = 12  # 12 months

    # S2 bands
    bands = [
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

    y = np.arange(0, y_size * 10, 10)
    x = np.arange(0, x_size * 10, 10)
    time = np.arange("2023-01-15", "2024-01-15", dtype="datetime64[M]")[:n_times]

    data_vars = {}
    for band in bands:
        data_vars[band] = xr.DataArray(
            np.random.randint(0, 10000, size=(n_times, y_size, x_size), dtype=np.uint16),
            dims=["time", "y", "x"],
            coords={"time": time, "y": y, "x": x},
        )

    ds = xr.Dataset(data_vars)
    ds["doy"] = ds.time.dt.dayofyear
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


def test_generate_embeddings(sample_s2_dataset):
    """Test basic embeddings generation with valid input."""
    result = generate_embeddings(sample_s2_dataset)

    # Check output type
    assert isinstance(result, xr.Dataset)
    assert "embedding" in result.data_vars

    # Check dimensions
    assert "feature" in result.dims
    assert "y" in result.dims
    assert "x" in result.dims

    # Check shape
    y_size = sample_s2_dataset.sizes["y"]
    x_size = sample_s2_dataset.sizes["x"]
    assert result.embedding.shape[0] == 512  # 512 features expected
    assert result.embedding.shape[1] == y_size
    assert result.embedding.shape[2] == x_size

    # Check dtype
    assert result.embedding.dtype == np.float32

    # Check CRS
    assert result.rio.crs is not None
    assert str(result.rio.crs) == "EPSG:4326"


def test_quantize_embeddings_basic(sample_s2_dataset):
    """Test basic quantization functionality."""
    # Generate embeddings first
    embeddings_ds = generate_embeddings(sample_s2_dataset)
    embeddings = embeddings_ds.embedding

    # Quantize
    result = quantize_embeddings(embeddings)

    # Check output type
    assert isinstance(result, xr.Dataset)
    assert "embedding" in result.data_vars

    # Check dtype
    assert result.embedding.dtype == np.uint8


def test_quantize_embeddings_attributes(sample_s2_dataset):
    """Test that quantization sets correct attributes."""
    embeddings_ds = generate_embeddings(sample_s2_dataset)
    embeddings = embeddings_ds.embedding

    result = quantize_embeddings(embeddings)

    # Check attributes
    assert result.attrs.get("quantized")


def test_quantize_full_pipeline(sample_s2_dataset):
    """Test full pipeline: generate embeddings -> quantize."""
    # Generate
    embeddings_ds = generate_embeddings(sample_s2_dataset)

    # Quantize
    quantized_ds = quantize_embeddings(embeddings_ds.embedding)

    # Verify full pipeline
    assert isinstance(quantized_ds, xr.Dataset)
    assert quantized_ds.embedding.dtype == np.uint8
    assert quantized_ds.embedding.shape[0] == 512
    assert quantized_ds.embedding.shape[1:] == embeddings_ds.embedding.shape[1:]
