"""Code modified from https://github.com/ucam-eo/tessera.

Requires that the original checkpoint be manually downloaded from
https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing
"""

import torch
from torch.export.dynamic_shapes import Dim

import pixelverse as pv

if __name__ == "__main__":
    model = pv.models.Tessera()
    model.eval()

    b, t = 1, 10
    s2 = torch.randn(b, t, 10)
    s2_doy = torch.randint(1, 365, (b, t, 1))
    s1 = torch.randn(b, t, 2)
    s1_doy = torch.randint(1, 365, (b, t, 1))

    x = torch.cat([s2, s2_doy, s1, s1_doy], dim=-1)
    print(model(x).shape)

    # Load and extract only the model state dict then save to model.pt.
    path = "best_model_fsdp_20250427_084307.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    modules = ["s2_backbone", "s1_backbone", "dim_reducer"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    state_dict = {k: v for k, v in state_dict.items() if k.split(".")[0] in modules}
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), "model.pt")
    model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=True)

    # Export the encoders
    torch.save(model.s2_backbone.state_dict(), "s2_encoder.pt")
    torch.save(model.s1_backbone.state_dict(), "s1_encoder.pt")

    # Export the model and save to model_exported_program.pt2.
    example_inputs = torch.randn(1, 10, 14)
    dims = (Dim.AUTO, Dim.AUTO, 14)
    model_program = torch.export.export(
        mod=model, args=(example_inputs,), dynamic_shapes={"x": dims}
    )
    torch.export.save(model_program, "model_exported_program.pt2")

    # Export the s1 and s2 encoders
    example_inputs = torch.randn(1, 10, 11)
    dims = (Dim.AUTO, Dim.AUTO, 11)
    model_program = torch.export.export(
        mod=model.s2_backbone, args=(example_inputs,), dynamic_shapes={"x": dims}
    )
    torch.export.save(model_program, "s2_encoder_exported_program.pt2")

    example_inputs = torch.randn(1, 10, 3)
    dims = (Dim.AUTO, Dim.AUTO, 3)
    model_program = torch.export.export(
        mod=model.s1_backbone, args=(example_inputs,), dynamic_shapes={"x": dims}
    )
    torch.export.save(model_program, "s1_encoder_exported_program.pt2")
