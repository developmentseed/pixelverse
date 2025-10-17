"""Code modified from https://github.com/ucam-eo/tessera."""

from typing import Any

import torch
from torchvision.models._api import Weights, WeightsEnum

from pixelverse.models.transforms import PixelTimeSeriesNormalize


class TemporalAwarePooling(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, 1)
        self.temporal_context = torch.nn.GRU(input_dim, input_dim, batch_first=True)

    def forward(self, x):
        # First capture temporal context through RNN
        x_context, _ = self.temporal_context(x)
        # Then calculate attention weights
        w = torch.softmax(self.query(x_context), dim=1)
        return (w * x).sum(dim=1)


class TemporalEncoding(torch.nn.Module):
    def __init__(self, d_model, num_freqs=64):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model

        # Learnable frequency parameters (more flexible than fixed frequencies)
        self.freqs = torch.nn.Parameter(
            torch.exp(torch.linspace(0, torch.log(torch.tensor(365.0)), num_freqs))
        )

        # Project Fourier features to the target dimension through a linear layer
        self.proj = torch.nn.Linear(2 * num_freqs, d_model)
        self.phase = torch.nn.Parameter(torch.zeros(1, 1, d_model))  # Learnable phase offset

    def forward(self, doy):
        # doy: (B, seq_len, 1)
        t = doy / 365.0 * 2 * torch.pi  # Normalize to the 0-2π range

        # Generate multi-frequency sine/cosine features
        t_scaled = t * self.freqs.view(1, 1, -1)  # (B, seq_len, num_freqs)
        sin = torch.sin(t_scaled + self.phase[..., : self.num_freqs])
        cos = torch.cos(t_scaled + self.phase[..., self.num_freqs : 2 * self.num_freqs])

        # Concatenate and project to the target dimension
        encoding = torch.cat([sin, cos], dim=-1)  # (B, seq_len, 2*num_freqs)
        return self.proj(encoding)  # (B, seq_len, d_model)


class TemporalPositionalEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, doy):
        # doy: [B, T] tensor containing DOY values (0-365)
        position = doy.unsqueeze(-1).float()  # Ensure float type
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        div_term = div_term.to(doy.device)

        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        band_num,
        latent_dim,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        # Total input dimension: bands
        input_dim = band_num

        # Embedding to increase dimension
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 4, latent_dim * 4),
        )

        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        # Transformer Encoder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=latent_dim * 4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Temporal Aware Pooling
        self.attn_pool = TemporalAwarePooling(latent_dim * 4)

    def forward(self, x):
        # x: (B, seq_len, 10 bands + 1 doy)
        # Split bands and doy
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]  # Last column is DOY
        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*4)
        temporal_encoding = self.temporal_encoder(doy)
        # Add temporal encoding to embedded bands (instead of random positional encoding)
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        x = self.attn_pool(x)
        return x


class Tessera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.s2_backbone = TransformerEncoder(
            band_num=10,
            latent_dim=self.embed_dim,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.s1_backbone = TransformerEncoder(
            band_num=2,
            latent_dim=self.embed_dim,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.dim_reducer = torch.nn.Sequential(torch.nn.Linear(self.embed_dim * 8, self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 14
        s2_x, s1_x = x[..., :11], x[..., 11:]
        s2_feat = self.s2_backbone(s2_x)  # (b, d)
        s1_feat = self.s1_backbone(s1_x)  # (b, d)
        fused = torch.cat([s2_feat, s1_feat], dim=-1)  # (b, 2d)
        fused = self.dim_reducer(fused)  # (b, embed_dim)
        return fused


S2_BAND_MEAN = [
    1711.0938,
    1308.8511,
    1546.4543,
    3010.1293,
    3106.5083,
    2068.3044,
    2685.0845,
    2931.5889,
    2514.6928,
    1899.4922,
]
S2_BAND_STD = [
    1926.1026,
    1862.9751,
    1803.1792,
    1741.7837,
    1677.4543,
    1888.7862,
    1736.3090,
    1715.8104,
    1514.5199,
    1398.4779,
]
S1_BAND_MEAN = [5484.0407, 3003.7812]
S1_BAND_STD = [1871.2334, 1726.0670]
TESSERA_MEAN = S2_BAND_MEAN + [0.0] + S1_BAND_MEAN + [0.0]
TESSERA_STD = S2_BAND_STD + [1.0] + S1_BAND_STD + [1.0]

tessera_transforms = torch.nn.Sequential(
    PixelTimeSeriesNormalize(mean=TESSERA_MEAN, std=TESSERA_STD, inplace=True),
)


class TESSERA_WEIGHTS(WeightsEnum):
    TESSERA = Weights(
        url="https://hf.co/isaaccorley/tessera/resolve/51afe75b724d387ef9fcb6f6e090a5be0b906919/model.pt",
        transforms=tessera_transforms,
        meta={
            "bands": [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B11",
                "B12",
                "S2",
                "S2_DOY",
                "VV",
                "VH",
                "S1_DOY",
            ],
            "num_channels": 14,
            "embed_dim": 128,
            "input_shape": [(None, None, 14)],
            "output_shape": [(None, 128)],
            "mean": TESSERA_MEAN,
            "std": TESSERA_STD,
        },
    )


def tessera(weights: TESSERA_WEIGHTS | None = None, *args: Any, **kwargs: Any) -> Tessera:
    model = Tessera(*args, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    return model
