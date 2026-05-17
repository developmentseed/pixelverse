# Pixelverse

<p align="center">
    <br>
    <img src="./assets/logo.png" width="600"/>
    <br>
<p>

Cloud native tooling to generate and store pixelwise geospatial foundation model
embeddings.

We are working to **#FreeTheEmbeddings** and make geospatial embeddings available to
all.

> [!WARNING]
>
> `pixelverse` is in development -- expect frequent releases and possible bugs or
> missing features

## Getting started

We use all the new hip tools like [uv](https://docs.astral.sh/uv/), [ty](https://docs.astral.sh/ty/),
and [prek](https://prek.j178.dev/) to make this project easy to use.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
uv run prek install --install-hooks

# optionally run pre-commit hooks manually
uv run prek run --all-files
```

### Pixelwise Geospatial Foundation Models

`pixelverse` provides access to a growing collection of pixelwise geospatial foundation
models. You can list available models with:

```bash
uv run python -c "import pixelverse as pv; print(pv.list_models())"
```

#### Supported Models

- [**Tessera**](https://arxiv.org/abs/2506.20380): Tessera is a pixelwise time-series
  geospatial foundation model. It takes as input a 14-channel time series of pixel
  values, `(10 S2 bands, 1 S2 DOY, 2 S1 bands, and 1 S1 DOY)` and outputs a
  `128`-dimensional embedding.
- [**OLMoEarth**](https://github.com/allenai/olmoearth_pretrain_minimal): OLMoEarth
  Sentinel-2 variants are available in the registry as `olmoearth_nano`,
  `olmoearth_tiny`, `olmoearth_base`, and `olmoearth_large`. The current Pixelverse
  wrapper accepts `B,T,C,H,W` image tensors plus explicit timestamps (`B,T,3`) and
  returns patch-level embeddings. By default, `patch_size=1`, so output `H,W`
  matches input `H,W`.

Model-specific input requirements (for example, Sentinel-2 STAC band names) are
available in weights metadata:

```python
import pixelverse as pv

weights = pv.get_weights("olmoearth_nano")
print(weights.meta["s2_asset_names"])
```

### Citing

If you use this software in your research, please cite:

```bibtex
@misc{pixelverse,
  title={Pixelverse: Pixelwise Geospatial Embeddings for All},
  author={Morrissey, Martha and Corley, Isaac},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/developmentseed/pixelverse}}
}
```
