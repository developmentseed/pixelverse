from typing import Any, Literal, cast

import torch
from einops import rearrange
from olmoearth_pretrain_minimal import ModelID, Normalizer, OlmoEarthPretrain_v1, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms import v2

OLMOEARTH_S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]

OLMOEARTH_S2_STAC_BANDS = [
    "coastal",  # B01
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",  # B09
    "swir16",
    "swir22",
]

OLMOEARTH_S2_PATCH_SIZE = 1
OLMOEARTH_S2_INPUT_RES = 10

_MODEL_SIZE_TO_ID: dict[str, ModelID] = {
    "nano": ModelID.OLMOEARTH_V1_NANO,
    "tiny": ModelID.OLMOEARTH_V1_TINY,
    "base": ModelID.OLMOEARTH_V1_BASE,
    "large": ModelID.OLMOEARTH_V1_LARGE,
}

_MODEL_SIZE_TO_EMBED_DIM: dict[str, int] = {
    "nano": 128,
    "tiny": 192,
    "base": 768,
    "large": 1024,
}

OlmoEarthModelSize = Literal["nano", "tiny", "base", "large"]


def _s2_normalize_transform(std_multiplier: float = 2.0) -> torch.nn.Sequential:
    """Build torchvision normalization equivalent to OLMoEarth's min/max scaling.

    OLMoEarth Normalizer computes:
    (x - (mean - k*std)) / (2*k*std)
    which is exactly torchvision Normalize(mean=min_vals, std=range_vals).

    Parameters
    ----------
    std_multiplier : float
        Standard deviation multiplier used by upstream OLMoEarth normalization stats.

    Returns
    -------
    torch.nn.Sequential
        Channel-wise normalization transform for `B,T,C,H,W` tensors.
    """

    normalizer = Normalizer(std_multiplier=std_multiplier)
    modality = Modality.SENTINEL2_L2A
    modality_norm = normalizer.norm_config[modality.name]
    mean = []
    std = []
    for band in modality.band_order:
        mean_val = modality_norm[band]["mean"]
        std_val = modality_norm[band]["std"]
        mean.append(mean_val - std_multiplier * std_val)
        std.append(2 * std_multiplier * std_val)
    return torch.nn.Sequential(v2.Normalize(mean=mean, std=std))


def _validate_timestamps(timestamps: torch.Tensor, batch: int, time_steps: int) -> None:
    if timestamps.ndim != 3:
        raise ValueError(f"Expected timestamps shape [B, T, 3], got {tuple(timestamps.shape)}")
    if timestamps.shape[0] != batch or timestamps.shape[1] != time_steps:
        raise ValueError(
            "Timestamps must match input batch/time dimensions: "
            f"x has (B={batch}, T={time_steps}), timestamps has {tuple(timestamps.shape)}"
        )
    if timestamps.shape[2] != 3:
        raise ValueError(
            f"Expected timestamps last dim to be 3 [day, month, year], got {timestamps.shape[2]}"
        )


class OlmoEarthS2Encoder(torch.nn.Module):
    """Pixelverse wrapper around OLMoEarth S2 encoder.

    Input image tensor shape: [B, T, C, H, W]
    Timestamps tensor shape: [B, T, 3] where columns are [day, month, year].

    Parameters
    ----------
    model_size : {"nano", "tiny", "base", "large"}
        OLMoEarth model variant to load.
    load_weights : bool, default=True
        Whether to load pretrained weights from the upstream package.
    patch_size : int, default=1
        Patch size passed to the encoder.
    input_res : int, default=10
        Input spatial resolution (meters) passed to the encoder.
    fast_pass : bool, default=True
        Skip upstream mask-value checks when masks are known-valid.
    """

    def __init__(
        self,
        model_size: OlmoEarthModelSize,
        load_weights: bool = True,
        patch_size: int = OLMOEARTH_S2_PATCH_SIZE,
        input_res: int = OLMOEARTH_S2_INPUT_RES,
        fast_pass: bool = True,
    ):
        super().__init__()
        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.fast_pass = fast_pass

        model_id = _MODEL_SIZE_TO_ID[model_size]
        if load_weights:
            self._model = load_model_from_id(model_id, load_weights=True)
        else:
            self._model = OlmoEarthPretrain_v1(model_size=model_size)
        self.encoder = cast(Any, self._model.encoder)
        self.num_bandsets = self.encoder.patch_embeddings.tokenization_config.get_num_bandsets(
            "sentinel2_l2a"
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[2] != len(OLMOEARTH_S2_BANDS):
            raise ValueError(
                f"Expected {len(OLMOEARTH_S2_BANDS)} channels at dim=2, got {x.shape[2]}"
            )
        batch, time_steps, _, height, width = x.shape
        _validate_timestamps(timestamps, batch, time_steps)

        # OLMoEarth expects [B, H, W, T, C]
        x_olmo = rearrange(x, "b t c h w -> b h w t c").contiguous()
        timestamps = timestamps.to(device=x.device, dtype=torch.long)
        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=x_olmo,
            sentinel2_l2a_mask=torch.zeros(
                (batch, height, width, time_steps, self.num_bandsets),
                device=x.device,
                dtype=torch.long,
            ),
        )
        output = self.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=self.fast_pass,
        )
        return output["tokens_and_masks"].pool_spatially(PoolingType.MEAN)


def _olmoearth_meta(model_size: OlmoEarthModelSize, variant_name: str) -> dict[str, Any]:
    return {
        "model_id": _MODEL_SIZE_TO_ID[model_size].value,
        "bands": OLMOEARTH_S2_BANDS,
        "s2_stac_bands": OLMOEARTH_S2_STAC_BANDS,
        "num_channels": len(OLMOEARTH_S2_BANDS),
        "embed_dim": _MODEL_SIZE_TO_EMBED_DIM[model_size],
        "input_shape": [(None, None, len(OLMOEARTH_S2_BANDS), None, None)],
        "timestamps_shape": [(None, None, 3)],
        "output_shape": [(None, None, None, _MODEL_SIZE_TO_EMBED_DIM[model_size])],
        "patch_size": OLMOEARTH_S2_PATCH_SIZE,
        "input_res": OLMOEARTH_S2_INPUT_RES,
        "variant": variant_name,
    }


class OLMOEARTH_WEIGHTS(WeightsEnum):
    OLMOEARTH_NANO = Weights(
        url="",
        transforms=_s2_normalize_transform(),
        meta=_olmoearth_meta("nano", "olmoearth_nano"),
    )
    OLMOEARTH_TINY = Weights(
        url="",
        transforms=_s2_normalize_transform(),
        meta=_olmoearth_meta("tiny", "olmoearth_tiny"),
    )
    OLMOEARTH_BASE = Weights(
        url="",
        transforms=_s2_normalize_transform(),
        meta=_olmoearth_meta("base", "olmoearth_base"),
    )
    OLMOEARTH_LARGE = Weights(
        url="",
        transforms=_s2_normalize_transform(),
        meta=_olmoearth_meta("large", "olmoearth_large"),
    )


def _build_olmoearth_s2_encoder(
    model_size: OlmoEarthModelSize,
    weights: OLMOEARTH_WEIGHTS | None = None,
    *args: Any,
    **kwargs: Any,
) -> OlmoEarthS2Encoder:
    return OlmoEarthS2Encoder(model_size, weights is not None, *args, **kwargs)


def olmoearth_nano(
    weights: OLMOEARTH_WEIGHTS | None = None, *args: Any, **kwargs: Any
) -> OlmoEarthS2Encoder:
    return _build_olmoearth_s2_encoder("nano", weights, *args, **kwargs)


def olmoearth_tiny(
    weights: OLMOEARTH_WEIGHTS | None = None, *args: Any, **kwargs: Any
) -> OlmoEarthS2Encoder:
    return _build_olmoearth_s2_encoder("tiny", weights, *args, **kwargs)


def olmoearth_base(
    weights: OLMOEARTH_WEIGHTS | None = None, *args: Any, **kwargs: Any
) -> OlmoEarthS2Encoder:
    return _build_olmoearth_s2_encoder("base", weights, *args, **kwargs)


def olmoearth_large(
    weights: OLMOEARTH_WEIGHTS | None = None, *args: Any, **kwargs: Any
) -> OlmoEarthS2Encoder:
    return _build_olmoearth_s2_encoder("large", weights, *args, **kwargs)
