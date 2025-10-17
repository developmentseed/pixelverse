"""Code modified from https://github.com/ucam-eo/tessera

Requires that the original checkpoint be manually downloaded from
https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing
"""

import torch
from torch.export.dynamic_shapes import Dim


class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, seq_len, dim)
        w = torch.softmax(self.query(x), dim=1)  # (B, seq_len, 1)
        return (w * x).sum(dim=1)


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
        self.s2_backbone = TransformerEncoder(
            band_num=10,
            latent_dim=128,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.s1_backbone = TransformerEncoder(
            band_num=2,
            latent_dim=128,
            nhead=8,
            num_encoder_layers=8,
            dim_feedforward=4096,
            dropout=0.1,
        )
        self.dim_reducer = torch.nn.Sequential(torch.nn.Linear(128 * 8, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape(b, t, c) where c=14, the first 11 channels are
                sentinel-2 (10 bands + 1 doy features) and the last 3 channels are
                sentinel-1 (2 bands + 1 doy features)
        """
        assert x.shape[-1] == 14
        s2_x, s1_x = x[..., :11], x[..., 11:]
        s2_feat = self.s2_backbone(s2_x)  # (b, d)
        s1_feat = self.s1_backbone(s1_x)  # (b, d)
        fused = torch.cat([s2_feat, s1_feat], dim=-1)  # (b, 2d)
        fused = self.dim_reducer(fused)  # (b, 128)
        return fused


if __name__ == "__main__":
    # Load the pretrained model for inference only without the projection using the pretrained config
    model = Tessera()
    model.eval()

    b, t = 2, 10
    s2 = torch.randn(b, t, 10)
    s2_doy = torch.randint(1, 365, (b, t, 1))
    s1 = torch.randn(b, t, 2)
    s1_doy = torch.randint(1, 365, (b, t, 1))

    x = torch.cat([s2, s2_doy, s1, s1_doy], dim=-1)
    print(model(x).shape)

    # Load and extract only the model state dict then save to model.pt
    path = "best_model_fsdp_20250427_084307.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    modules = ["s2_backbone", "s1_backbone", "dim_reducer"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    state_dict = {k: v for k, v in state_dict.items() if k.split(".")[0] in modules}
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), "model.pt")

    # Export the model and save to model_exported_program.pt2
    example_inputs = torch.randn(1, 10, 14)
    dims = (Dim.AUTO, Dim.AUTO, 14)
    model_program = torch.export.export(
        mod=model, args=(example_inputs,), dynamic_shapes={"x": dims}
    )
    torch.export.save(model_program, "model_exported_program.pt2")
