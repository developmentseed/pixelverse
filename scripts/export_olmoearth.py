# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "olmoearth-pretrain-minimal>=0.0.2",
#     "torch>=2.11.0",
# ]
# ///
"""
Script to export OlmoEarth_V1 model variants as .pt2 format.

Run using:
    uv run --no-project scripts/export_olmoearth.py --variant tiny

References:
- Herzog, H., Bastani, F., Zhang, Y., Tseng, G., Redmon, J., Sablon, H., Park, R.,
  Morrison, J., Buraczynski, A., Farley, K., Hansen, J., Howe, A., Johnson, P. A.,
  Otterlee, M., Schmitt, T., Pitelka, H., Daspit, S., Ratner, R., Wilhelm, C., …
  Beukema, P. (2025). OlmoEarth: Stable Latent Image Modeling for Multimodal Earth
  Observation (arXiv:2511.13655). arXiv. https://doi.org/10.48550/arXiv.2511.13655

"""

import argparse
from typing import Literal

import numpy as np
import torch
from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)
from torch.utils._pytree import _register_namedtuple  # noqa: PLC2701


def convert(variant: Literal["nano", "tiny", "base", "large"] = "nano") -> None:
    """
    Convert OlmoEarth_V1 model encoder from original .pth format on HuggingFace to
    .pt2 format.

    See https://huggingface.co/collections/allenai/olmoearth.

    Parameters
    ----------
    variant : str
        Model variant to load. Choose from 'nano', 'tiny', 'base', 'large'. Default is
        'nano'.

    """
    # 1. Load model weights from HuggingFace
    model_variant = f"OLMOEARTH_V1_{variant}".upper()
    print(f"Loading HuggingFace .pth weights for {model_variant}")  # noqa: T201
    model = load_model_from_id(model_id=ModelID[model_variant], load_weights=True)

    # 2. Define sample input
    # Create minimal sample (timestamps required, month must be long for embedding)
    timestamps = torch.zeros(1, 12, 3, dtype=torch.long)
    timestamps[:, :, 1] = torch.arange(12, dtype=torch.long)  # months 0-11
    # Initialize normalizer
    normalizer = Normalizer(std_multiplier=2.0)
    # Prepare Sentinel-2 L2A data: (batch, height, width, time, bands)
    # Bands must match Modality.SENTINEL2_L2A.band_order (12 bands)
    rng = np.random.default_rng(seed=42)
    sentinel2_data = rng.random(size=(1, 128, 128, 12, 12), dtype=np.float32)
    normalized_sentinel2 = normalizer.normalize(Modality.SENTINEL2_L2A, sentinel2_data)
    sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=torch.from_numpy(normalized_sentinel2).float(),
        sentinel2_l2a_mask=torch.zeros(1, 128, 128, 12, dtype=torch.long),
    )

    # 3. Save to .pt2 format
    exported = torch.export.export(
        model.encoder,  # ty: ignore[invalid-argument-type]
        args=(sample,),
        kwargs={"patch_size": 8, "fast_pass": True},
    )
    _register_namedtuple(
        MaskedOlmoEarthSample,
        serialized_type_name="olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes.MaskedOlmoEarthSample",
    )
    _register_namedtuple(
        TokensAndMasks,
        serialized_type_name="olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit.TokensAndMasks",
    )
    torch.export.save(exported, f := f"{model_variant.lower()}.pt2")
    print(f"{model_variant} exported to {f}")  # noqa: T201


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert OlmoEarth models from .pth to .pt2")
    parser.add_argument(
        "--variant",
        type=str,
        default="nano",
        choices=["nano", "tiny", "base", "large"],
    )
    args = parser.parse_args()

    convert(variant=args.variant)


if __name__ == "__main__":
    main()
